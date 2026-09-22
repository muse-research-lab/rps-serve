from dataclasses import dataclass, field
from typing import List, Literal, LiteralString, Optional, Union

from llmperf.ingestion.dataset import Dataset
from llmperf.preprocessing.workload import Workload

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