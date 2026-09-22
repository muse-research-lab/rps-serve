import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Iterable, List, Any

from dataclasses import dataclass

from llmperf.ingestion.config import IngestionConfig
from llmperf.ingestion.dataset import Dataset

@dataclass(slots=True)
class BaseIngestion(ABC):
    """
    Base class for all dataset ingestions.
    Subclasses implement the dataset-specific steps.
    """
    config: "IngestionConfig"

    def __post_init__(self):
        # Use a dataset-scoped logger
        self.logger = logging.getLogger(
            f"ingestion.{self.config.dataset_alias}"
        )

    def run(self) -> "Dataset":
        """
        Executes the ingestion lifecycle.
        """
        self.logger.info("Ingestion started")

        start = time.time()

        raw = self.fetch_raw_data()
        cleaned = self.clean_data(raw)
        deduped = self.deduplicate(cleaned)
        limited = self.limit_records(deduped)
        
        os.makedirs(self.config.storage_path, exist_ok=True)
        self.store(limited)

        duration = round(time.time() - start, 3)

        self.logger.info(
            "Ingestion completed",
            extra={
                "records": len(limited),
                "storage_path": self.config.storage_path,
                "duration_seconds": duration,
            },
        )

        return Dataset(
            name=self.config.dataset_name,
            path=self.config.storage_path,
            file=self.config.records_file,
            alias=self.config.dataset_alias,
            data=limited
        )

    @abstractmethod
    def fetch_raw_data(self) -> Iterable[Any]:
        raise NotImplementedError

    def clean_data(self, records: Iterable[Any]) -> List[Any]:
        return list(records)

    def deduplicate(self, records: List[Any]) -> List[Any]:
        return records

    def limit_records(self, records: List[Any]) -> List[Any]:
        max_records = self.config.max_records
        if max_records is None:
            return records
        return records[:max_records]
    
    @abstractmethod
    def store(self, records: List[Any]):
        """
        Persist records in storage path.
        Images should be stored in images/ directory.
        Videos should be stored in videos/ directory.
        Audios should be stored in audios/ directory.

        Image records should have an "image" field with the image path.
        Video records should have an "video" field with the video path.
        Audio records should have an "audio" field with the audio path.
        
        Image, video, and audio paths should be relative to the modaility dirs:
        - images/
        - videos/
        - audios/
        
        All records should be stored in storage path inside the records file.
        """
        pass
