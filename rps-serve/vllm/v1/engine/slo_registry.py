# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path


class SLORegistry:
    def __init__(self, filepaths: list[str]):
        """
        Initializes the component and calculates request-to-SLO mappings from
        JSONL log files.

        :param filepaths: A single path (str/Path) or a list of paths to JSONL files.
        """
        self.filepaths = [Path(fp) for fp in filepaths]

        self._req_to_slo: dict[str, float] = {}

        # Automatically load and parse data upon initialization
        self._load_all_slo_data()

    def _load_all_slo_data(self) -> None:
        """Streams JSONL files line-by-line, calculates SLO, and stores it."""
        for filepath in self.filepaths:
            try:
                # Open the file and iterate over lines dynamically (Memory Efficient)
                with open(filepath, encoding="utf-8") as file:
                    for line_num, line in enumerate(file, start=1):
                        cleaned_line = line.strip()
                        if not cleaned_line:
                            continue  # Skip empty lines

                        try:
                            entry = json.loads(cleaned_line)

                            req_id = entry.get("id")
                            arrival = entry.get("arrival_time")
                            finished = entry.get("finished_time")

                            # Validate required fields exist
                            if req_id and arrival is not None and finished is not None:
                                # SLO = finished_time - arrival_time
                                self._req_to_slo[req_id] = finished - arrival

                        except json.JSONDecodeError:
                            print(
                                "Warning: Skipping malformed JSON on line "
                                f"{line_num} in {filepath}"
                            )
                            continue

            except FileNotFoundError as err:
                raise FileNotFoundError(
                    f"Initialization failed: {filepath} does not exist."
                ) from err

    def get_slo(self, req_id: str) -> float:
        """
        Retrieves the calculated SLO (duration) for a given request ID.
        """
        if req_id not in self._req_to_slo:
            raise KeyError(f"Request ID '{req_id}' not found in SLO registry.")

        return self._req_to_slo[req_id]

    def get_slo_safe(self, req_id: str, default: float = 0.0) -> float:
        """
        A safer lookup method that returns a default value if the ID doesn't exist.
        """
        return self._req_to_slo.get(req_id, default)
