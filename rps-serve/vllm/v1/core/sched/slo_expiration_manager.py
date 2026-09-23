# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import time

from vllm.v1.core.sched.queue_manager import QueueManager
from vllm.v1.request import Request


class SLOExpirationManager:
    # category -> (p, k_base, rps_offset)
    # rps_offset is only applied under the "rps" policy.
    _CATEGORY_PARAMS = {
        "sand": (3.5, 0.05, 0.1),
        "pebbles": (2.5, 0.003, 0.05),
        "rocks": (1.1, 0.00075, 0.0),
    }

    def __init__(self, queue_manager: QueueManager, policy: str = "fcfs") -> None:
        # Supported policies:
        # "fcfs": default fcfs
        # "catfcfs": categorical fcfs
        # "edf": earliest deadline first
        # "wait-only": waiting time term only
        # "rps": static priority + waiting time term

        self.queue_manager = queue_manager
        self.policy = policy

        self.eps = 0.0001  # 1ms
        self.min_score = 1e-12

    def compute_score(
        self,
        request: Request,
        avg_queue_age: float,
        queue_age: float,
        queue_size: int,
        avg_queue_size: float,
    ):
        if self.policy not in ("rps", "wait-only"):
            return 1

        params = self._CATEGORY_PARAMS.get(request.request_md.category)
        if params is None:
            return 1
        p, k_base, rps_offset = params

        len_ratio = queue_size / (avg_queue_size + self.eps)
        age_ratio = queue_age / (avg_queue_age + self.eps)

        w_age = 0.7
        w_len = 0.3
        excess = max(0.0, w_age * (age_ratio - 1) + w_len * (len_ratio - 1))

        B_max = 3.0
        theta = 2.0
        boost = round(1 + (B_max - 1) * (1 - math.exp(-theta * excess)))

        offset = rps_offset if self.policy == "rps" else 0.0
        k_eff = k_base * boost
        return 1 - math.exp(-k_eff * (request.age**p)) + offset

    def update_priorities(self, requests: dict[str, Request]) -> dict[str, Request]:
        if self.policy == "fcfs":
            for req in requests.values():
                # Set priority once
                if req.priority and req.priority > 0:
                    continue
                req.priority = int(req.arrival_time * 10**7)
            return requests

        if self.policy == "catfcfs":
            for req in requests.values():
                # Set priority once
                if req.priority and req.priority > 0:
                    continue

                if not (req.request_md and req.request_md.category):
                    req.priority = 10**19 + int(req.arrival_time * 10**7)
                    continue

                # Based on the category
                # Sand > Pebbles > Rocks
                # Tiebreaker: arrival time (FCFS)
                if req.request_md.category == "sand":
                    req.priority = int(req.arrival_time * 10**7)
                if req.request_md.category == "pebbles":
                    req.priority = 10**18 + int(req.arrival_time * 10**7)
                if req.request_md.category == "rocks":
                    req.priority = 10**19 + int(req.arrival_time * 10**7)
            return requests

        if self.policy == "edf":
            now = time.time()
            for req in requests.values():
                req.priority = req.request_md.slo - (now - req.arrival_time)

            scored = sorted(
                requests.values(), key=lambda r: (r.priority, r.arrival_time)
            )

            for i, req in enumerate(scored):
                # Priorities must be integers higher than 0
                req.priority = i + 1

            return {req.request_id: req for req in scored}

        avg_sand_queue_size = self.queue_manager.sand_queue_size.ema
        avg_pebbles_queue_size = self.queue_manager.pebbles_queue_size.ema
        avg_rocks_queue_size = self.queue_manager.rocks_queue_size.ema

        sand_queue_size = self.queue_manager.sand_queue_size.curr
        pebbles_queue_size = self.queue_manager.pebbles_queue_size.curr
        rocks_queue_size = self.queue_manager.rocks_queue_size.curr

        avg_sand_queue_age = self.queue_manager.sand_queue_age.ema
        avg_pebbles_queue_age = self.queue_manager.pebbles_queue_age.ema
        avg_rocks_queue_age = self.queue_manager.rocks_queue_age.ema

        sand_queue_age = self.queue_manager.sand_queue_age.curr
        pebbles_queue_age = self.queue_manager.pebbles_queue_age.curr
        rocks_queue_age = self.queue_manager.rocks_queue_age.curr

        # Compute urgency scores
        for request in requests.values():
            if request.request_md.category == "sand":
                score = self.ws * self.compute_score(
                    request,
                    avg_sand_queue_age,
                    sand_queue_age,
                    sand_queue_size,
                    avg_sand_queue_size,
                )
            elif request.request_md.category == "pebbles":
                score = self.wp * self.compute_score(
                    request,
                    avg_pebbles_queue_age,
                    pebbles_queue_age,
                    pebbles_queue_size,
                    avg_pebbles_queue_size,
                )
            elif request.request_md.category == "rocks":
                score = self.wr * self.compute_score(
                    request,
                    avg_rocks_queue_age,
                    rocks_queue_age,
                    rocks_queue_size,
                    avg_rocks_queue_size,
                )
            else:
                request.priority = -math.log(self.min_score)
                continue
            request.priority = -math.log(score)

        scored = sorted(requests.values(), key=lambda r: (r.priority, r.arrival_time))

        for i, req in enumerate(scored):
            # Priorities must be integers higher than 0
            req.priority = i + 1

        return {req.request_id: req for req in scored}
