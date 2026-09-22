import json
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import List, LiteralString, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import RequestOutput as vllmRequestOutput

from llmperf.constants import EXPERIMENTS_LOG, EXPERIMENTS_OUTPUTS_DIR, EXPERIMENTS_ENGINE_STATS_DIR
from llmperf.postprocessing.aggregator import Aggregator
from llmperf.postprocessing.filter import Filter

@dataclass
class RequestOutput:
    id: str
    prompt_tokens_cnt: Optional[int] = 0
    modality_tokens_cnt: Optional[int] = 0
    decode_tokens_cnt: Optional[int] = 0
    outputs: Optional[List[str]] = field(default_factory=list)

    # vLLM metrics
    vllm_id: Optional[int] = 0
    arrival_time: Optional[float] = 0.0
    input_processed_time: Optional[float] = 0.0
    last_token_time: Optional[float] = 0.0
    first_scheduled_time: Optional[float] = 0.0
    first_token_time: Optional[float] = 0.0
    time_in_queue: Optional[float] = 0.0
    finished_time: Optional[float] = 0.0
    scheduler_time: Optional[float] = 0.0
    model_forward_time: Optional[float] = 0.0
    model_execute_time: Optional[float] = 0.0
    model_encoder_time: Optional[float] = 0.0

    # Metadata
    estimated_time: Optional[float] = None
    category: Optional[str] = None
    stl: Optional[bool] = None
    aborted: Optional[bool] = None
    uid: Optional[str] = None
    slo: Optional[float] = None
    ttft_slo: Optional[float] = None
    tbt_slo: Optional[float] = None

    # Other
    prompt_preparation_time: Optional[float] = None

    @property
    def processor_time(self) -> float:
        return self.input_processed_time - self.arrival_time
    
    @property
    def encoder_time(self) -> float:
        return self.model_encoder_time
    
    @property
    def ttft(self) -> float:
        return self.processor_time + self.time_in_queue + (self.first_token_time - self.first_scheduled_time)
    
    @property
    def tbt(self) -> float:
        return 0.0 if self.decode_tokens_cnt <= 1 else (self.e2e - self.ttft) / (self.decode_tokens_cnt - 1)
    
    @property
    def e2e(self) -> float:
        return self.finished_time - self.arrival_time
    
    @property
    def queued_time(self) -> float:
        return self.time_in_queue
    
    @property
    def prefill_time(self) -> float:
        return self.first_token_time - self.first_scheduled_time
    
    @property
    def decode_time(self) -> float:
        return self.last_token_time - self.first_token_time
    
    @property
    def inference_time(self) -> float:
        return self.last_token_time - self.first_scheduled_time
    
    @classmethod
    def from_vllm_output(cls, req_id: str, req_output: "vllmRequestOutput", modality_token_index: int = -1) -> "RequestOutput":
        aborted = (
            len(req_output.outputs[0].token_ids) == 1 and \
            req_output.outputs[0].token_ids[0] == -1
        )

        estimated_time = category = stl = None
        if (hasattr(req_output, "request_md")) and (md := req_output.request_md):
            estimated_time = md.estimated_time
            category = md.category
            stl = md.stl

        uid = slo = ttft_slo = tbt_slo = None

        if (hasattr(req_output, "user_md")) and (umd := req_output.user_md):
            uid = umd.uid
            slo = umd.slo
            ttft_slo = umd.ttft_slo
            tbt_slo = umd.tbt_slo

        return cls(
            id=req_id,
            prompt_tokens_cnt=len(req_output.prompt_token_ids),
            modality_tokens_cnt=req_output.prompt_token_ids.count(modality_token_index),
            decode_tokens_cnt=len(req_output.outputs[0].token_ids),
            outputs=[co.text for co in req_output.outputs],
            vllm_id=req_output.request_id,
            arrival_time=req_output.metrics.arrival_time,
            input_processed_time=req_output.metrics.input_processed_time,
            last_token_time=req_output.metrics.last_token_ts,
            first_scheduled_time=req_output.metrics.scheduled_ts,
            first_token_time=req_output.metrics.first_token_ts,
            time_in_queue=req_output.metrics.scheduled_ts - req_output.metrics.queued_ts,
            finished_time=req_output.metrics.finished_time,
            scheduler_time=None,
            model_forward_time=None,
            model_execute_time=None,
            model_encoder_time=req_output.metrics.model_encoder_time,
            estimated_time=estimated_time,
            category=category,
            stl=stl,
            aborted=aborted,
            uid=uid,
            slo=slo,
            ttft_slo=ttft_slo,
            tbt_slo=tbt_slo
        )

@dataclass
class EngineStats:
    timestamps: List[float]
    kv_cache_usage: List[float]
    encoder_cache_usage: List[float]
    preempted_req_ids: List[List[str]]
    preempted_req_ts: List[List[float]]
    rescheduled_req_ids: List[List[str]]
    rescheduled_req_ts: List[List[float]]
    kv_cache_usage_per_category: List[dict[str,float]]
    encoder_cache_usage_per_category: List[dict[str,float]]

    num_preemptions: List[int] = field(init=False)

    def __post_init__(self):
        self.num_preemptions = [len(reqs) for reqs in self.preempted_req_ids]

@dataclass
class ExperimentOutput:
    id: str
    elapsed_time: float =  None
    request_outputs: List[RequestOutput] = field(default_factory=list)
    engine_stats: Optional[EngineStats] = None

    output_path: Optional[Union[str,LiteralString]] = None
    output_log_path: Optional[Union[str,LiteralString]] = None
    engine_stats_path: Optional[Union[str,LiteralString]] = None

    LATENCY_SLO_MAP = {
        "e2e": ("e2e", "slo"),
        "tbt": ("tbt", "tbt_slo"),
        "ttft": ("ttft", "ttft_slo"),
    }

    @property
    def num_requests(self) -> int:
        return len(self.request_outputs)
    
    @property
    def total_num_tokens(self) -> int:
        total_num_tokens = 0
        for request_output in self.request_outputs:
            prompt_tokens_cnt = request_output.prompt_tokens_cnt
            decode_tokens_cnt = request_output.decode_tokens_cnt

            total_num_tokens += prompt_tokens_cnt + decode_tokens_cnt
        return total_num_tokens
    
    @property
    def requests_per_second(self) -> float:
        return self.num_requests / self.elapsed_time
    
    @property
    def tokens_per_second(self) -> float:
        return self.total_num_tokens / self.elapsed_time

    def _latency(self, type: str, method: str = "mean", filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()

        request_latencies = []
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            if type == "normalized":
                request_latencies.append(ro.e2e / ro.decode_tokens_cnt)
            else:
                request_latencies.append(getattr(ro, type))
        
        return Aggregator.aggregate(request_latencies, method)

    def normalized_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("normalized", method, filter)

    def ttft_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("ttft", method, filter)

    def e2e_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("e2e", method, filter)

    def tbt_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("tbt", method, filter)
    
    def tiq_latency(self, method: str = "mean", filter: Filter = None) -> float:
        return self._latency("time_in_queue", method, filter)
    
    def preemption_latency(self, method: str = "mean", filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()

        filtered_req_ids = {ro.vllm_id for ro in self.request_outputs if filter.include(ro)}

        preemptions = []
        for preempted_ids, preempted_ts in zip(self.engine_stats.preempted_req_ids, self.engine_stats.preempted_req_ts):
            for req_id, ts in zip(preempted_ids, preempted_ts):
                preemptions.append((req_id, ts))

        reschedules = []
        for rescheduled_ids, rescheduled_ts in zip(self.engine_stats.rescheduled_req_ids, self.engine_stats.rescheduled_req_ts):
            for req_id, ts in zip(rescheduled_ids, rescheduled_ts):
                reschedules.append((req_id, ts))

        preempted_dict = defaultdict(list)
        for req_id, ts in preemptions:
            preempted_dict[req_id].append(ts)

        rescheduled_dict = defaultdict(list)
        for req_id, ts in reschedules:
            rescheduled_dict[req_id].append(ts)

        for req_id in preempted_dict:
            preempted_dict[req_id].sort()
        
        for req_id in rescheduled_dict:
            rescheduled_dict[req_id].sort()

        preemption_deltas = defaultdict(list)
        for req_id, pre_ts_list in preempted_dict.items():
            if req_id not in rescheduled_dict:
                continue

            res_ts_list = rescheduled_dict[req_id]
            res_idx = 0

            for pre_ts in pre_ts_list:
                while res_idx < len(res_ts_list) and res_ts_list[res_idx] <= pre_ts:
                    res_idx += 1
                if res_idx < len(res_ts_list):
                    delta = res_ts_list[res_idx] - pre_ts
                    preemption_deltas[req_id].append(delta)
                    res_idx += 1

        preemption_latencies = []
        for req_id, deltas in preemption_deltas.items():
            if req_id in filtered_req_ids:
                preemption_latencies.append(sum(deltas))
        
        return Aggregator.aggregate(preemption_latencies, method)
    
    def relative_delay(self, method: str = "mean", filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()

        filtered_req_ids = {ro.vllm_id for ro in self.request_outputs if filter.include(ro)}

        preemptions = []
        for preempted_ids, preempted_ts in zip(self.engine_stats.preempted_req_ids, self.engine_stats.preempted_req_ts):
            for req_id, ts in zip(preempted_ids, preempted_ts):
                preemptions.append((req_id, ts))

        reschedules = []
        for rescheduled_ids, rescheduled_ts in zip(self.engine_stats.rescheduled_req_ids, self.engine_stats.rescheduled_req_ts):
            for req_id, ts in zip(rescheduled_ids, rescheduled_ts):
                reschedules.append((req_id, ts))

        preempted_dict = defaultdict(list)
        for req_id, ts in preemptions:
            preempted_dict[req_id].append(ts)

        rescheduled_dict = defaultdict(list)
        for req_id, ts in reschedules:
            rescheduled_dict[req_id].append(ts)

        for req_id in preempted_dict:
            preempted_dict[req_id].sort()
        
        for req_id in rescheduled_dict:
            rescheduled_dict[req_id].sort()

        preemption_deltas = defaultdict(list)
        for req_id, pre_ts_list in preempted_dict.items():
            if req_id not in rescheduled_dict:
                continue

            res_ts_list = rescheduled_dict[req_id]
            res_idx = 0

            for pre_ts in pre_ts_list:
                while res_idx < len(res_ts_list) and res_ts_list[res_idx] <= pre_ts:
                    res_idx += 1
                if res_idx < len(res_ts_list):
                    delta = res_ts_list[res_idx] - pre_ts
                    preemption_deltas[req_id].append(delta)
                    res_idx += 1

        req_id_to_ro = {ro.vllm_id: ro for ro in self.request_outputs}
        relative_delays = []
        for req_id, deltas in preemption_deltas.items():
            if req_id in filtered_req_ids:
                preemption_latency = sum(deltas)
                tiq_latency = req_id_to_ro[req_id].time_in_queue
                e2e_latency = req_id_to_ro[req_id].e2e

                relative_delays.append((preemption_latency + tiq_latency) / e2e_latency)
        
        return Aggregator.aggregate(relative_delays, method)

    def latency_slo_delta(self, latency_type: str = "e2e", method: str = "mean", filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        if filter is None:
            filter = Filter()
        latency_attr, slo_attr = self.LATENCY_SLO_MAP[latency_type]

        request_deltas = []
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
             # latency - slo thres
            latency = getattr(ro, latency_attr)
            slo = getattr(ro, slo_attr)
            if slo is None or slo == 0.0:
                slo = slo_map.get(ro.id, float("inf"))
            request_deltas.append(latency - slo)

        return Aggregator.aggregate(request_deltas, method)

    def slo_headroom(self, latency_type: str = "e2e", method: str = "mean", filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        if filter is None:
            filter = Filter()
        latency_attr, slo_attr = self.LATENCY_SLO_MAP[latency_type]

        request_headrooms = []
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            latency = getattr(ro, latency_attr)
            slo = getattr(ro, slo_attr)
            if slo is None or slo == 0.0:
                slo = slo_map.get(ro.id, float("inf"))

            # slo thres - latency (for negative deltas = for positive headrooms)
            headroom = slo - latency
            if headroom > 0:
                request_headrooms.append(headroom)

        return Aggregator.aggregate(request_headrooms, method)

    def slo_violation(self, latency_type: str = "e2e", method: str = "mean", filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        if filter is None:
            filter = Filter()
        latency_attr, slo_attr = self.LATENCY_SLO_MAP[latency_type]

        request_violations = []
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            latency = getattr(ro, latency_attr)
            slo = getattr(ro, slo_attr)
            if slo is None or slo == 0.0:
                slo = slo_map.get(ro.id, float("inf"))

            # latency - slo thres (for positive deltas = for negative headrooms)
            violation = latency - slo
            if violation > 0:
                request_violations.append(violation)

        return Aggregator.aggregate(request_violations, method)

    def slo_violations(self, latency_type: str = "e2e", filter: Filter = None, slo_map: dict[str, float] = None) -> int:
        if filter is None:
            filter = Filter()
        latency_attr, slo_attr = self.LATENCY_SLO_MAP[latency_type]

        num_violations = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            latency = getattr(ro, latency_attr)
            slo = getattr(ro, slo_attr)
            if slo is None or slo == 0.0:
                slo = slo_map.get(ro.id, float("inf"))
            if latency > slo:
                num_violations += 1

        return num_violations

    def _slo_attainment(self, latency_attr: str, slo_attr: str, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        if filter is None:
            filter = Filter()
        if slo_map is None:
            slo_map = {}

        filtered_cnt = 0
        attained_cnt = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue

            filtered_cnt += 1

            slo = getattr(ro, slo_attr)
            if slo is None or slo == 0.0:
                slo = slo_map.get(ro.id, float("inf"))

            if getattr(ro, latency_attr) <= slo:
                attained_cnt += 1

        return attained_cnt / filtered_cnt * 100 if filtered_cnt else 0.0

    def e2e_slo_attainment(self, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        return self._slo_attainment("e2e", "slo", filter, slo_map)

    def tbt_slo_attainment(self, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        return self._slo_attainment("tbt", "tbt_slo", filter, slo_map)

    def ttft_slo_attainment(self, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        return self._slo_attainment("ttft", "ttft_slo", filter, slo_map)
    
    def throughput(self, filter: Filter = None) -> float:
        if filter is None:
            filter = Filter()

        request_cnt = 0
        start_time = float("inf")
        finished_time = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            start_time = min(start_time, ro.arrival_time)
            finished_time = max(finished_time, ro.finished_time)
            request_cnt += 1
        
        return request_cnt / (finished_time - start_time) if request_cnt else 0
    
    def goodput(self, filter: Filter = None, slo_map: dict[str, float] = None) -> float:
        if filter is None:
            filter = Filter()
        if slo_map is None:
            slo_map = {}

        attained_cnt = 0
        start_time = float("inf")
        finished_time = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue

            slo = ro.slo
            if slo is None:
                slo = slo_map.get(ro.id, float("inf"))

            # TODO: Rethink goodput SLO definition (TTFT?, TBT?)
            if ro.e2e <= slo:
                attained_cnt += 1

            start_time = min(start_time, ro.arrival_time)
            finished_time = max(finished_time, ro.finished_time)

        return attained_cnt / (finished_time - start_time) if attained_cnt else 0

    def preemptions(self, filter: Filter = None) -> int:
        if filter is None:
            return sum(self.engine_stats.num_preemptions)
        
        req_id_to_ro = { ro.vllm_id: ro for ro in self.request_outputs }

        num_preemptions = 0
        for preempted_req_ids in self.engine_stats.preempted_req_ids:
            for req_id in preempted_req_ids:
               if not filter.include(req_id_to_ro[req_id]):
                   continue
               num_preemptions += 1
               
        return num_preemptions
    
    def aborted(self, filter: Filter = None) -> int:
        if filter is None:
            filter = Filter(aborted=True)

        assert filter.include_aborted == True
        
        aborted = 0
        for ro in self.request_outputs:
            if not filter.include(ro):
                continue
            if ro.aborted:
                aborted += 1
               
        return aborted

    def kv_cache_usg(self, category: str = None, method: str = "mean") -> float:
        if category and self.engine_stats.kv_cache_usage_per_category and \
            category in self.engine_stats.kv_cache_usage_per_category[0]:
            usg = [usg[category] for usg in self.engine_stats.kv_cache_usage_per_category]
        else:
            usg = [usg for usg in self.engine_stats.kv_cache_usage]
    
        return Aggregator.aggregate(usg, method)

    def save_engine_stats(self, log_file: Union[str,LiteralString]):
        path = os.path.join(self.engine_stats_path or EXPERIMENTS_ENGINE_STATS_DIR, f"{self.id}.parquet")

        df = pd.read_json(log_file, lines=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression="brotli", compression_level=11)

    def load_engine_stats(self):
        path = os.path.join(self.engine_stats_path or EXPERIMENTS_ENGINE_STATS_DIR, f"{self.id}.parquet")
        df = pd.read_parquet(path)
        
        self.engine_stats = EngineStats(
            timestamps=df["timestamp"].tolist(),
            kv_cache_usage=df["kv_cache_usage"].tolist(),
            encoder_cache_usage=df["encoder_cache_usage"].tolist(),
            rescheduled_req_ids=[list(arr) for arr in df["rescheduled_req_ids"].tolist()],
            rescheduled_req_ts=[list(arr) for arr in df["rescheduled_req_ts"].tolist()],
            preempted_req_ids=[list(arr) for arr in df["preempted_req_ids"].tolist()],
            preempted_req_ts=[list(arr) for arr in df["preempted_req_ts"].tolist()],
            kv_cache_usage_per_category=df["kv_cache_usage_per_category"].tolist(),
            encoder_cache_usage_per_category=df["encoder_cache_usage_per_category"].tolist()
        )

    def save(self):
        path = os.path.join(self.output_path or EXPERIMENTS_OUTPUTS_DIR, f"{self.id}.jsonl")
        with open(path, "w", encoding="utf-8") as file:
            for request_output in self.request_outputs:
                file.write(json.dumps(asdict(request_output)) + "\n")

        system_output = {
            "id": self.id,
            "elapsed_time": self.elapsed_time,
            "num_requests": self.num_requests,
            "total_num_tokens": self.total_num_tokens,
            "requests_per_second": self.requests_per_second,
            "tokens_per_second": self.tokens_per_second
        }
        with open(self.output_log_path or EXPERIMENTS_LOG, "a", encoding="utf-8") as file:
                file.write(json.dumps(system_output) + "\n")

    def load(self):
        path = os.path.join(self.output_path or EXPERIMENTS_OUTPUTS_DIR, f"{self.id}.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                self.request_outputs.append(RequestOutput(**entry))

        if self.elapsed_time is None:
            start_time = min([ro.arrival_time for ro in self.request_outputs])
            finish_time = max([ro.finished_time for ro in self.request_outputs])
            self.elapsed_time = finish_time - start_time