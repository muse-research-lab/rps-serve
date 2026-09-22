from typing import Literal
import numpy as np

class Aggregator:
    Method = Literal["mean", "p50", "p90", "p95", "p99"]

    _percentiles = {
        "p50": 50,
        "p90": 90,
        "p95": 95,
        "p99": 99,
    }

    @staticmethod
    def aggregate(data: list[float], method: Method = "mean") -> float:
        if not data:
            return 0.0

        if method == "mean":
            return float(np.mean(data))

        if method in Aggregator._percentiles:
            return float(np.percentile(data, Aggregator._percentiles[method]))

        raise ValueError(f"Unsupported aggregation method: {method}")
