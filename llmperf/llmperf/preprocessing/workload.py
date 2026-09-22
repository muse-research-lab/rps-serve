import hashlib
import json
import os

from dataclasses import asdict, dataclass, field
from typing import Dict, List, LiteralString, Optional, Union

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
