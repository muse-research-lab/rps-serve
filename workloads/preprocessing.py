import argparse
import hashlib
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, LiteralString, Optional, Union

import ffmpeg
import numpy as np
from PIL import Image
from tqdm import tqdm

from ingestion import Dataset

def get_image_path(dir: Union[str,LiteralString], record: Dict) -> LiteralString:
    return os.path.join(dir, "images", record["image"])

def get_video_path(dir: Union[str,LiteralString], record: Dict) -> LiteralString:
    return os.path.join(dir, "videos", record["video"])

def get_audio_path(dir: Union[str,LiteralString], record: Dict) -> LiteralString:
    return os.path.join(dir, "audios", record["audio"])

def get_image_size(dir: Union[str,LiteralString], record: Dict) -> Dict:
    path = get_image_path(dir, record)

    with Image.open(path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "color_mode": img.mode,
            "file_size": os.path.getsize(path),
            "codec": img.format
        }

def count_video_frames(path: str) -> int:
    probe = ffmpeg.probe(
        path,
        select_streams='v:0',
        show_entries='stream=nb_read_frames',
        count_frames=None
    )
    stream = next((s for s in probe["streams"] if s.get("nb_read_frames")), None)

    if not stream or "nb_read_frames" not in stream:
        return 0

    return int(stream["nb_read_frames"])
    
def get_video_size(dir: Union[str,LiteralString], record: Dict) -> Dict:
    path = get_video_path(dir, record)
    
    probe = ffmpeg.probe(path)
    stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    fmt = probe["format"]

    frame_count = int(stream.get("nb_frames", 0))
    if frame_count == 0:
        frame_count = count_video_frames(path)

    return {
        "duration": int(float(fmt["duration"])),
        "frame_count": frame_count,
        "bit_rate": int(fmt["bit_rate"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "color_mode": stream["pix_fmt"],
        "file_size": float(fmt["size"]),
        "codec": stream["codec_name"]
    }

def get_audio_size(dir: Union[str,LiteralString], record: Dict) -> Dict:
    path = get_audio_path(dir, record)
    
    probe = ffmpeg.probe(path)
    stream, fmt = probe["streams"][0], probe["format"]

    return {
        "duration": int(float(fmt["duration"])),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "bit_rate": int(fmt["bit_rate"]),
        "file_size": float(fmt["size"]),
        "codec": stream["codec_name"]
    }

@dataclass
class Request:
    input: str
    output: str
    id: str = None
    modality_path: Optional[Union[str,LiteralString]] = None
    modality_size: Optional[Dict] = field(default_factory=dict)

    def __post_init__(self): 
        if not self.id:
            combined = f"{self.input}|{self.output}|{self.modality_path or ''}"
            # ! Not collision free, but sufficient for small number of requests
            # ! Workloads with < 1000 requests are collision free (~0.01%)
            hash_object = hashlib.sha256(combined.encode('utf-8'))
            self.id = hash_object.hexdigest()[:8]

@dataclass
class Workload:
    name: str
    path: Union[str, LiteralString]
    alias: str
    requests: Optional[List[Request]] = field(default_factory=list)
    timestamps: Optional[List[float]] = field(default_factory=list)

    def __post_init__(self):
        if self.timestamps and len(self.timestamps) != len(self.requests):
            raise ValueError(
                "The number of timestamps must match the number of requests."
            )

    def __hash__(self):
        return hash((self.name, self.alias))

    def __eq__(self, other):
        if isinstance(other, Workload):
            return self.name == other.name and self.alias == other.alias
        return False
    
    # TODO: Property to calculate modality pcts
    
    @property
    def file_path(self) -> str:
        return os.path.join(self.path, f"{self.alias}.jsonl")

    def save(self):
        """Saves requests and timestamps as zipped JSONL entries."""
        os.makedirs(self.path, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as file:
            for i, request in enumerate(self.requests):
                entry = {"request": asdict(request)}
                if self.timestamps:
                    entry["timestamp"] = self.timestamps[i]
                file.write(json.dumps(entry) + "\n")

    def load(self):
        """Loads data from disk. Clears existing in-memory data first."""
        if not os.path.exists(self.file_path):
            return

        self.requests = []
        self.timestamps = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                self.requests.append(Request(**entry["request"]))
                if "timestamp" in entry:
                    self.timestamps.append(entry["timestamp"])

DataSource = Union[Dataset, Workload]
DataInput = Union[DataSource, List[DataSource]]

TimingMode = Literal["static", "poisson", "gamma"]

@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    workload_name: str
    storage_path: Union[str, LiteralString]
    workload_alias: str
    
    data_input: DataInput
    timing_mode: TimingMode = "static"
    duration: int = 100
    seed: int = 0
    
    # Poisson distribution
    arrival_rate: float = 1.0
    time_quantum: int = 10

    # Gamma distribution
    gamma_shape: float = 2.0
    gamma_scale: float = 1.0

    # Max requests to use in the workload (static only)
    num_requests: Optional[int] = None
    # Percentages of each dataset/workload
    splits_pcts: Optional[List[float]] = field(default_factory=list)

    def __post_init__(self):
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.timing_mode != "static" and self.num_requests is not None:
            raise ValueError(
                "num_requests can only be set when timing_mode='static'"
            )
        if self.num_requests is not None and self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if isinstance(self.data_input, list):
            if len(self.data_input) != len(self.splits_pcts):
                raise ValueError(
                    f"Mismatched lengths: data_input has {len(self.data_input)}"
                    f" items, but splits_pcts has {len(self.splits_pcts)}."
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

class ShareGPTPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][0]["value"]
    
    def get_output(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][1]["value"]
    
class ShareGPTLongPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][-1]["from"] == "gpt"):
            return None
        if  len(record["conversations"]) < 6:
            return None
        return '\n'.join([c["value"] for c in record["conversations"][:-1]])
    
    def get_output(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][-1]["from"] == "gpt"):
            return None
        if  len(record["conversations"]) < 6:
            return None
        return record["conversations"][-1]["value"]

class LLaVAInstructComplexReasoningPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][0]["value"] \
            .replace("<image>\n", "").replace("\n<image>", "")

    
    def get_output(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][1]["value"]

class LLaVAVideoCaptionPreprocessing(SingleDatasetPreprocessing):
    def get_input(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][0]["value"] \
            .replace("<image>\n", "").replace("\n<image>", "")


    
    def get_output(self, record: dict) -> str:
        if not (record["conversations"][0]["from"] == "human" and
                record["conversations"][1]["from"] == "gpt"):
            return None
        return record["conversations"][1]["value"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        action="store_true",
        help="Run ShareGPT preprocessing",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Run LLaVA-Instruct-150k preprocessing",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Run LLaVA-Video preprocessing",
    )
    args = parser.parse_args()

    if not (args.text or args.image or args.video):
        parser.error("Specify at least one of --text, --image, or --video")

    if args.text:
        print("Preprocessing ShareGPT...")
        start_time = time.perf_counter()

        text_conv_ds = Dataset(
                name="Text Conversations",
                path=os.path.join(os.getcwd(), "raw", "ShareGPT"),
                file="sharegpt.jsonl",
                alias="text-conv",
            )
        
        config = PreprocessingConfig(
            workload_name="Text Conversations",
            storage_path="./static",
            workload_alias="text",
            data_input=text_conv_ds,
            timing_mode="static",
            num_requests=1000
        )

        preprocessing = ShareGPTPreprocessing(config)
        workload = preprocessing.run()

        config = PreprocessingConfig(
            workload_name="Long Text Conversations",
            storage_path="./static",
            workload_alias="text-long",
            data_input=text_conv_ds,
            timing_mode="static",
            num_requests=1000
        )

        preprocessing = ShareGPTLongPreprocessing(config)
        workload = preprocessing.run()

        print(
            f"Text workloads are ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )

    if args.image:
        print("Preprocessing LLaVA-Instruct-150k...")
        start_time = time.perf_counter()

        img_reason_ds = Dataset(
            name="Image Reasoning",
            path=os.path.join(os.getcwd(), "raw", "LLaVA-Instruct-150K"),
            file="complex_reasoning.jsonl",
            alias="img-reason",
        )

        config = PreprocessingConfig(
            workload_name="Image Reasoning",
            storage_path="./static",
            workload_alias="image",
            data_input=img_reason_ds,
            timing_mode="static",
            num_requests=1000
        )

        preprocessing = LLaVAInstructComplexReasoningPreprocessing(config)
        workload = preprocessing.run()

        print(
            f"Image workload is ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )

    if args.video:
        print("Preprocessing LLaVA-Video...")
        start_time = time.perf_counter()

        vid_desc_ds = Dataset(
            name="Video Description",
            path=os.path.join(os.getcwd(), "raw", "LLaVA-Video"),
            file="description.jsonl",
            alias="vid-desc",
        )

        config = PreprocessingConfig(
            workload_name="Video Description",
            storage_path="./static",
            workload_alias="video",
            data_input=vid_desc_ds,
            timing_mode="static",
            num_requests=1000
        )

        preprocessing = LLaVAVideoCaptionPreprocessing(config)
        workload = preprocessing.run()

        print(
            f"Video workload is ready! Elapsed time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )