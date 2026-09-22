from dataclasses import dataclass
from typing import Union, LiteralString


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    dataset_name: str
    storage_path: Union[str, LiteralString]
    records_file: str
    dataset_alias: str
    max_records: int | None = None
    seed: int = 0

    def __post_init__(self):
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive")
