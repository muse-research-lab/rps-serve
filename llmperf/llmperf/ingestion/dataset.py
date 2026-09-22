import json
import os
from dataclasses import dataclass, field
from typing import List, LiteralString, Union, Optional


@dataclass
class Dataset:
    name: str
    path: Union[str, LiteralString]
    file: str
    alias: str
    data: Optional[List[dict]] = field(default=None, repr=False)

    def __hash__(self):
        return hash((self.name, self.alias))
    
    def __eq__(self, other):
        if isinstance(other, Dataset):
            return self.name == other.name and self.alias == other.alias
        return False

    def load(self) -> List[dict]:
        """
        Loads data from the file if `data` is None, otherwise returns the
        existing data.
        """
        if self.data is not None:
            return self.data
        
        file_path = os.path.join(self.path, self.file)
        loaded_data = []
        with open(file_path, "r") as f:
            for line in f:
                loaded_data.append(json.loads(line))
        self.data = loaded_data
        return self.data
