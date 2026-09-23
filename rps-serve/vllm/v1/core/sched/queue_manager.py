# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from collections import deque

import numpy as np

from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request, RequestStatus


class ExponentialMovingAverage:
    def __init__(self, alpha, ema=None):
        self.alpha = alpha
        self.ema = self.curr = ema

    def update(self, value):
        self.curr = value
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

    def clear(self):
        self.ema = None


class QueueManager:
    cache = {
        "llava-onevision-qwen2-0.5b-ov-hf": {
            "text": 0.006347427730379662,
            "image": 0.08030594210512935,
            "video": 0.2981209339904599,
        },
        "Qwen2.5-VL-3B-Instruct": {
            "text": 0.016478755001333497,
            "image": 0.06494460209924728,
            "video": 1.9391672755708338,
        },
        "gemma-3-4b-it": {
            "text": 0.019657576298850694,
            "image": 0.1107543233949691,
            "video": 1.7952281059804371,
        },
        "llava-onevision-qwen2-7b-ov-chat-hf": {
            "text": 0.030164917991350366,
            "image": 0.30287961234990507,
            "video": 0.7456325789636467,
        },
        "Qwen2.5-VL-7B-Instruct": {
            "text": 0.030418283349467327,
            "image": 0.08162711962172761,
            "video": 2.334135525283831,
        },
        "gemma-3-12b-it": {
            "text": 0.05408658215463758,
            "image": 0.15471395227848553,
            "video": 2.4613284320330715,
        },
        "pixtral-12b": {
            "text": 0.06568552921417621,
            "image": 0.2168186946886126,
            "video": 4.097469515458215,
        },
    }

    def __init__(self, model, policy, track_stats: bool = False):
        self.combined: list[Request] = []
        self.running_req_ids: set[str] = set()
        self.waiting_req_ids: set[str] = set()
        self.preempted_req_ids: set[str] = set()
        self.ignored_req_ids: set[str] = set()
        self.skipped_req_ids: set[str] = set()
        self.last_running_idx = None
        self.first_waiting_idx = None

        self.track_stats = track_stats
        self.model_name = model.split("/")[-1]

        if self.track_stats:
            # alpha=0.004 and we keep history of 500 steps ~10-15s
            self.sand_queue_age = ExponentialMovingAverage(0.004)
            self.pebbles_queue_age = ExponentialMovingAverage(0.004)
            self.rocks_queue_age = ExponentialMovingAverage(0.004)

            self.sand_queue_size = ExponentialMovingAverage(0.004)
            self.pebbles_queue_size = ExponentialMovingAverage(0.004)
            self.rocks_queue_size = ExponentialMovingAverage(0.004)

            sand_tpref = pebbles_tpref = rocks_tpref = None

            if self.model_name in self.cache:
                sand_tpref = self.cache[self.model_name]["text"]
                pebbles_tpref = self.cache[self.model_name]["image"]
                rocks_tpref = self.cache[self.model_name]["video"]

            # TODO: Dynamically update them
            self.sand_prefill_time = ExponentialMovingAverage(0.004, sand_tpref)
            self.pebbles_prefill_time = ExponentialMovingAverage(0.004, pebbles_tpref)
            self.rocks_prefill_time = ExponentialMovingAverage(0.004, rocks_tpref)

            self.queue_stats = {"sand": [], "pebbles": [], "rocks": []}

        self.policy = policy

    def get_last_running_idx(self) -> int | None:
        return self.last_running_idx

    def get_first_waiting_idx(self) -> int | None:
        return self.first_waiting_idx

    def num_running_requests(self) -> int:
        return len(self.running_req_ids)

    def reorder_running_queue(self, running: list[Request]) -> list[Request]:
        running = sorted(running, key=lambda req: req.priority or 0)
        return running

    def reorder_waiting_queue(self, waiting: RequestQueue) -> deque[Request]:
        waiting = deque(sorted(waiting, key=lambda req: req.priority or 0))
        return waiting

    def combine_queues(self, running: list[Request], waiting: deque[Request]):
        # "running" = 0 < "preempted" = 1 < "waiting" = 2
        self.running_req_ids = set()
        self.waiting_req_ids = set()
        self.preempted_req_ids = set()
        combined = []

        for req in running:
            if req.status == RequestStatus.PREEMPTED:
                combined.append((1, req.priority, req))
                self.preempted_req_ids.add(req.request_id)
            else:
                combined.append((0, req.priority, req))
                self.running_req_ids.add(req.request_id)

        for req in waiting:
            combined.append((2, req.priority, req))
            self.waiting_req_ids.add(req.request_id)

        # Sort: by priority DESC, then by queue type
        combined.sort(key=lambda x: (x[1], x[0]))
        self.combined = [x[2] for x in combined]

        # Update last running idx
        self.last_running_idx = None
        for i in range(len(self.combined) - 1, -1, -1):
            if self.combined[i].request_id in self.running_req_ids:
                self.last_running_idx = i
                break

        # Update first waiting idx
        self.first_waiting_idx = None
        for i in range(len(self.combined)):
            if (
                self.combined[i].request_id in self.waiting_req_ids
                or self.combined[i].request_id in self.preempted_req_ids
            ):
                self.first_waiting_idx = i
                break

        return

    def update_queues(self) -> tuple[list[Request], RequestQueue]:
        # Costly iteration over combined queue, only when ignored request exists
        if self.ignored_req_ids or self.skipped_req_ids:
            running: list[Request] = []
            waiting: RequestQueue = create_request_queue(self.policy)

            for req in self.combined:
                request_id = req.request_id
                if request_id in self.running_req_ids:
                    running.append(req)
                elif request_id in self.ignored_req_ids:
                    waiting.add_request(req)
                else:
                    waiting.add_request(req)

            return running, waiting

        if self.last_running_idx is not None and self.first_waiting_idx is not None:
            assert self.last_running_idx == self.first_waiting_idx - 1

            request_queue = create_request_queue(self.policy)
            for request in self.combined[self.first_waiting_idx :]:
                request_queue.add_request(request)
            return self.combined[: self.first_waiting_idx], request_queue

        elif self.last_running_idx is not None and self.first_waiting_idx is None:
            return self.combined, create_request_queue(self.policy)

        elif self.last_running_idx is None and self.first_waiting_idx is not None:
            request_queue = create_request_queue(self.policy)
            for request in self.combined:
                request_queue.add_request(request)
            return [], request_queue

        else:
            return [], create_request_queue(self.policy)

    def update_idxs_after_preemption(self):
        # A) Move LR idx to the beginning of the combined queue
        # B1) Adjust FW idx if we have an empty running queue after preemption
        # B2) Adjust FW idx if "first_preempted_idx" < "first_waiting_idx"
        # B3) Adjust FW idx if we have an empty waiting queue before preemption
        if self.last_running_idx is None:
            return

        last_running_idx = self.last_running_idx

        # Update request id sets
        request_id = self.combined[last_running_idx].request_id
        self.running_req_ids.remove(request_id)
        self.preempted_req_ids.add(request_id)

        # Update last running idx
        self.last_running_idx = None
        for i in range(last_running_idx, -1, -1):
            if self.combined[i].request_id in self.running_req_ids:
                self.last_running_idx = i
                break

        # Update first waiting idx if necessary
        if (
            self.last_running_idx is None
            and len(self.waiting_req_ids) + len(self.preempted_req_ids) > 0
        ):
            assert self.first_waiting_idx == 0
            return

        if (
            self.first_waiting_idx is not None
            and last_running_idx < self.first_waiting_idx
        ):
            self.first_waiting_idx = last_running_idx
            return

        if self.first_waiting_idx is None:
            self.first_waiting_idx = len(self.combined) - 1

        return

    def update_idxs_after_new_scheduling(self):
        # A) Moves FW idx to the end of the combined queue
        # B1) Adjust LR idx if we have an empty waiting queue after scheduling
        # B2) Adjust LR idx if "old_first_waiting_idx" > "last_running_idx"
        # B3) Adjust LR idx if we have an empty running queue before scheduling
        if self.first_waiting_idx is None:
            return

        first_waiting_idx = self.first_waiting_idx

        # Update request id sets
        request_id = self.combined[first_waiting_idx].request_id

        if request_id in self.waiting_req_ids:
            self.waiting_req_ids.remove(request_id)

        if request_id in self.preempted_req_ids:
            self.preempted_req_ids.remove(request_id)

        self.running_req_ids.add(request_id)

        # Update first waiting idx
        self.first_waiting_idx = None
        for i in range(first_waiting_idx, len(self.combined)):
            if (
                self.combined[i].request_id in self.waiting_req_ids
                or self.combined[i].request_id in self.preempted_req_ids
            ):
                self.first_waiting_idx = i
                break

        # Update last running idx if necessary
        if self.first_waiting_idx is None and len(self.running_req_ids) > 0:
            self.last_running_idx = len(self.combined) - 1
            return

        if (
            self.last_running_idx is not None
            and first_waiting_idx > self.last_running_idx
        ):
            self.last_running_idx = first_waiting_idx
            return

        if self.last_running_idx is None:
            self.last_running_idx = 0

        return

    def update_idxs_after_ignored(self):
        if self.first_waiting_idx is None:
            return

        first_waiting_idx = self.first_waiting_idx

        # Update request id sets
        request_id = self.combined[first_waiting_idx].request_id
        self.waiting_req_ids.remove(request_id)
        self.ignored_req_ids.add(request_id)

        # Update first waiting idx
        self.first_waiting_idx = None
        for i in range(first_waiting_idx, len(self.combined)):
            if (
                self.combined[i].request_id in self.waiting_req_ids
                or self.combined[i].request_id in self.preempted_req_ids
            ):
                self.first_waiting_idx = i
                break

    def update_idxs_after_skipped(self):
        if self.first_waiting_idx is None:
            return

        first_waiting_idx = self.first_waiting_idx

        # Update request id sets
        request_id = self.combined[first_waiting_idx].request_id
        self.waiting_req_ids.remove(request_id)
        self.skipped_req_ids.add(request_id)

        # Update first waiting idx
        self.first_waiting_idx = None
        for i in range(first_waiting_idx, len(self.combined)):
            if (
                self.combined[i].request_id in self.waiting_req_ids
                or self.combined[i].request_id in self.preempted_req_ids
            ):
                self.first_waiting_idx = i
                break

    def update_queue_stats(self, requests: dict[str, Request]):
        if not self.track_stats:
            return

        sand_queue_size: int = 0
        pebbles_queue_size: int = 0
        rocks_queue_size: int = 0

        sand_queue_ages: list[float] = []
        pebbles_queue_ages: list[float] = []
        rocks_queue_ages: list[float] = []

        now = time.time()
        for request in requests.values():
            age = now - request.arrival_time
            request.age = age
            if request.request_md.category == "sand":
                sand_queue_size += 1
                sand_queue_ages.append(age)
            elif request.request_md.category == "pebbles":
                pebbles_queue_size += 1
                pebbles_queue_ages.append(age)
            elif request.request_md.category == "rocks":
                rocks_queue_size += 1
                rocks_queue_ages.append(age)
            else:
                continue

        self.sand_queue_size.update(sand_queue_size)
        self.pebbles_queue_size.update(pebbles_queue_size)
        self.rocks_queue_size.update(rocks_queue_size)

        self.sand_queue_age.update(np.mean(sand_queue_ages or [0.0]))
        self.pebbles_queue_age.update(np.mean(pebbles_queue_ages or [0.0]))
        self.rocks_queue_age.update(np.mean(rocks_queue_ages or [0.0]))

        # Save queue stats
        self.queue_stats["sand"].append(
            (
                now,
                self.sand_queue_size.ema,
                self.sand_queue_age.ema,
                self.sand_queue_size.curr,
                self.sand_queue_age.curr,
            )
        )
        self.queue_stats["pebbles"].append(
            (
                now,
                self.pebbles_queue_size.ema,
                self.pebbles_queue_age.ema,
                self.pebbles_queue_size.curr,
                self.pebbles_queue_age.curr,
            )
        )
        self.queue_stats["rocks"].append(
            (
                now,
                self.rocks_queue_size.ema,
                self.rocks_queue_age.ema,
                self.rocks_queue_size.curr,
                self.rocks_queue_age.curr,
            )
        )

    def save_stats(self):
        if not self.track_stats:
            return

        # Save queue stats
        for queue, stats in self.queue_stats.items():
            with open(f"{queue}-queue-stats.log", "w") as f:
                for record in stats:
                    log_entry = {
                        "timestamp": record[0],
                        "avg_queue_size": record[1],
                        "avg_queue_age": record[2],
                        "queue_size": record[3],
                        "queue_age": record[4],
                    }
                    f.write(json.dumps(log_entry) + "\n")
