import logging
import random
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import LiteralString, List, Optional, Union

from dataclasses import dataclass
from tqdm import tqdm

from llmperf.ingestion.dataset import Dataset
from llmperf.preprocessing.config import PreprocessingConfig
from llmperf.preprocessing.workload import Request, Workload
from llmperf.preprocessing.modality_info import (
    get_audio_path,
    get_audio_size,
    get_image_path,
    get_image_size,
    get_video_path,
    get_video_size,
)

@dataclass(slots=True)
class BasePreprocessing(ABC):
    """
    Base class for all workload preprocessing.
    Subclasses implement the workload-specific steps.
    """
    config: "PreprocessingConfig"

    def __post_init__(self):
        # Use a workload-scoped logger
        self.logger = logging.getLogger(
            f"preprocessing.{self.config.workload_alias}"
        )

    def run(self) -> "Workload":
        """
        Executes the ingestion lifecycle.
        """
        self.logger.info("Preprocessing started")

        start = time.time()

        timestamps = self.generate_timestamps()
        raw_requests = self.generate_requests()
        limited_requests = self.limit_requests(raw_requests)
        final_requests = self.align_workload(limited_requests, len(timestamps))
        
        workload = Workload(
            name=self.config.workload_name,
            path=self.config.storage_path,
            alias=self.config.workload_alias,
            requests=final_requests,
            timestamps=timestamps
        )
        workload.save()

        duration = round(time.time() - start, 3)

        self.logger.info(
            "Preprocessing completed",
            extra={
                "requests": len(final_requests),
                "storage_path": self.config.storage_path,
                "duration_seconds": duration,
            },
        )

        return workload
    
    @abstractmethod
    def generate_requests(self) -> List[Request]:
        raise NotImplementedError
    
    def limit_requests(self, requests: List[Request]) -> List[Request]:        
        num_requests = self.config.num_requests
        if num_requests is None:
            return requests
        return requests[:num_requests]
    
    def generate_timestamps(self) -> Optional[List[float]]:
        """Generates arrival times based on distribution config."""
        if self.config.timing_mode == "static":
            return []
        
        np.random.seed(self.config.seed)
        if self.config.timing_mode == "poisson":
            tq = self.config.time_quantum
            duration = self.config.duration
            lam = self.config.arrival_rate * (tq / 1000)
            quantums_per_sec = 1000 / tq
            arrival_times = np.random.poisson(
                lam=lam, size=int(duration * quantums_per_sec))
            timestamps = []
            for i, n in enumerate(arrival_times):
                timestamps += [i * (tq / 1000)] * n
        elif self.config.timing_mode == "gamma":
            duration = self.config.duration
            timestamps = []
            ts = 0.0
            while ts < duration:
                gamma_shape = self.config.gamma_shape
                gamma_scale = self.config.gamma_scale
                delta_time = np.random.gamma(gamma_shape, gamma_scale)
                ts = delta_time + ts
                timestamps.append(ts)
        else:
            raise ValueError(
                f"Unsupported timing mode: {self.config.timing_mode}"
            )

        return timestamps
    
    def align_workload(
        self,
        requests: List[Request],
        timestamps_cnt: int
    ) -> List[Request]:
        if self.config.timing_mode == "static":
            return requests
        
        # If there are more requests than needed, remove unnecessary requests
        if timestamps_cnt < len(requests):
            final_requests = requests[:timestamps_cnt]
        else:
            # If there are less requests than needed, cycle through requests
            final_requests = [
                requests[i % len(requests)] for i in range(timestamps_cnt)
            ]
        
        assert timestamps_cnt == len(final_requests)
        return final_requests

    def _get_modality_metadata(
        self, dir: Union[str, LiteralString], record: dict
    ) -> tuple[Optional[Union[str,LiteralString]], Optional[dict]]:
        if "image" in record:
            modality_path = get_image_path(dir, record)
            modality_size = get_image_size(dir, record)
        elif "video" in record:
            modality_path = get_video_path(dir, record)
            modality_size = get_video_size(dir, record)
        elif "audio" in record:
            modality_path = get_audio_path(dir, record)
            modality_size = get_audio_size(dir, record)
        else:
            return None, None
        
        return modality_path, modality_size

class SingleDatasetPreprocessing(BasePreprocessing):
    def generate_requests(self) -> List[Request]:
        data_input = self.config.data_input
        
        if isinstance(data_input, Workload):
            data_input.load()
            random.seed(self.config.seed)
            random.shuffle(data_input.requests)
            return data_input.requests
        
        if isinstance(data_input, Dataset):
            data = data_input.load()
            dir = data_input.path
        else:
            raise TypeError(
                f"Unsupported data input type: {type(data_input)}"
                "Provide a single data input of type Workload or Dataset"
        )

        requests: Request = []
        ids = []
        for record in tqdm(data):
            request_input = self.get_input(record)
            request_output = self.get_output(record)
            modality_path, modality_size = self._get_modality_metadata(
                dir, record
            )
            
            request_args = {
                "input": request_input,
                "output": request_output,
            }
            if modality_path:
                request_args["modality_path"] = modality_path
            if modality_size:
                request_args["modality_size"] = modality_size
            
            request = Request(**request_args)
            if request.id not in ids:
                requests.append(request)
                ids.append(request.id)

        random.seed(self.config.seed)
        random.shuffle(requests)
        
        return requests

    @abstractmethod
    def get_input(self, record: dict) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def get_output(self, record: dict) -> str:
        raise NotImplementedError
    
class MultiWorkloadPreprocessing(BasePreprocessing):
    def generate_requests(self) -> List[Request]:
        data_input = self.config.data_input
        if not isinstance(data_input, list):
            raise TypeError(
                f"Unsupported data input type: {type(data_input)}"
                "Provide a List[Workload]"
            )
        
        data = {}
        random.seed(self.config.seed)
        for workload in data_input:
            if not isinstance(workload, Workload):
                raise TypeError(
                    "All inputs must be Workload objects, "
                    f"found {type(workload)}"
                )
            workload.load()
            random.shuffle(workload.requests)
            data[workload.alias] = workload.requests
            

        return self.mix_workloads(data)

    @abstractmethod
    def mix_workloads(self, data: dict[str, List[Request]]) -> List[Request]:
        raise NotImplementedError
    

class PctMixPreprocessing(MultiWorkloadPreprocessing):
    def __post_init__(self):
        super().__post_init__()

        if self.config.timing_mode == "static" and not self.config.num_requests:
            raise ValueError(
                "num_requests is necessary when timing_mode='static'"
            )

    def mix_workloads(self, data: dict[str, List[Request]]) -> List[Request]:
        active_pairs = [
            (k, p)
            for k, p in zip(data.keys(), self.config.splits_pcts) if p > 0
        ]
        requests = []
        if self.config.timing_mode != "static":
            total_target= len(self.generate_timestamps())
        else:
            total_target = self.config.num_requests
        
        assigned_count = 0        
        for i, (workload, pct) in enumerate(active_pairs):
            if i == len(active_pairs) - 1:
                num_of_reqs = total_target - assigned_count
            else:
                num_of_reqs = int(pct/100 * total_target)
                assigned_count += num_of_reqs
            
            if num_of_reqs > 0:
                requests.extend(random.choices(data[workload], k=num_of_reqs))
        
        assert len(requests) == total_target
        return requests
    