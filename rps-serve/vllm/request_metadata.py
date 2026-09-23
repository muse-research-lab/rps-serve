# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses


@dataclasses.dataclass
class RequestMetadata:
    # Estimated end-to-end latency in seconds
    estimated_time: float

    # Classified category
    category: str = "unknown"

    # Skip the line flag
    stl: bool = False

    slo: float = 0.0
