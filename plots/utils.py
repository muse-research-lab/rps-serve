import re
import os
import json
import numpy as np

from collections import defaultdict
from statistics import mean

from output import ExperimentOutput

system_colors = {
    "vllm": "#888780",
    "rps-serve": "#378ADD",
    "rps-serve-nc": "#639922",
    "rps-serve-wo": "#7F77DD",
    "rps-serve-catfcfs": "#D85A30",
    "edf": "#639922",
    "mod-serve": "#D85A30",
    "mod-serve-rps": "#7F77DD"
}

system_names = {
    "vllm":         "vLLM",
    "edf":          "EDF",
    "mod-serve":    "ModServe",
    "mod-serve-rps":"ModServe & RPS",
    "rps-serve":    "RPS-Serve",
    "rps-serve-nc":    "RPS-Serve (Naive)",
    "rps-serve-catfcfs":    "RPS-Serve (Static Only)",
    "rps-serve-wo":    "RPS-Serve (Waiting Only)",
}

model_names = {
    "llava-ov": "LLaVA-7B",
    "llava-ov-large": "LLaVA-72B",
    "internvl-3.5-large": "InternVL-38B", # "InternVL3.5-38B",
    "gemma-4-large": "Gemma-31B", # "Gemma4-31B",
    "qwen-3.5-large": "Qwen-27B", # "Qwen3.5-27B",
}   

def get_cdf(data):
    N = len(data)

    x = np.sort(data)
    y = np.arange(N) / float(N)

    return x, y

def parse_benchmark_iso_file(filepath: str) -> dict:
    """
    Reads a file of JSON lines, extracts benchmark data, and organizes it
    into a nested dictionary: dict[model][workload]

    ID format example:
    text-static-small__llava-ov-large__vllm__iso__20260526-174501__poldef__
    maxlendef__batchdef__blocksdef__encbatchdef__encblocksdef__gpu0.95__swap0__
    stratuniform__maxframe32
    """
    patterns = {
        "workload": re.compile(r'^([^_]+)__'),
        "model":    re.compile(r'^[^_]+__([^_]+)__'),
    }

    results = defaultdict(dict)

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_id = record.get("id", "")

            extracted = {}
            for key, pattern in patterns.items():
                match = pattern.search(record_id)
                if not match:
                    break
                extracted[key] = match.group(1)
            else:
                results[extracted["model"]][extracted["workload"]] = record

    return results

def parse_benchmark_file(filepath: str) -> dict:
    """
    Reads a file of JSON lines, extracts benchmark data, and organizes it
    into a nested dictionary: dict[model][system][approach][rate]

    ID format example:
    mixed-heavy__llava-ov__vllm__tp1__20260525-114306__poldef__maxlendef__
    batchdef__blocksdef__encbatchdef__encblocksdef__gpu0.95__swap0__
    enginevllm__approachtp1__rate1.0__profpoisson__stratuniform__maxframe-1
    """
    # Regex patterns to extract fields from the ID
    patterns = {
        "model":    re.compile(r'^[^_]+__([^_]+)__'),
        "system":   re.compile(r'^[^_]+__[^_]+__([^_]+)__'),
        "approach": re.compile(r'__approach([^_]+)__'),
        "rate":     re.compile(r'__rate([^_]+)__'),
    }

    def nested_defaultdict():
        return defaultdict(nested_defaultdict)

    results = nested_defaultdict()

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip non-JSON lines silently
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_id = record.get("id", "")

            # Extract all four keys; skip the record if any is missing
            extracted = {}
            for key, pattern in patterns.items():
                match = pattern.search(record_id)
                if not match:
                    break
                extracted[key] = match.group(1)
            else:
                # All four keys found — extract them
                m = extracted["model"]
                s = extracted["system"]
                a = extracted["approach"]
                r = extracted["rate"]

                # If the entry already exists, it will be a list or a populated record
                current = results[m][s][a][r]
                
                if isinstance(current, list):
                    current.append(record)
                elif isinstance(current, dict) and current:  # Existing single record
                    results[m][s][a][r] = [current, record]
                else:  # Brand new uninitialized defaultdict leaf node
                    results[m][s][a][r] = [record]

    return results

def calculate_metric(eo: ExperimentOutput, metric: str, filter=None):
    _metric_dispatch: dict[str, tuple[str, list]] = {
        "normlat":     ("normalized_latency", []),
        "normlat_p90": ("normalized_latency", ["p90"]),
        "normlat_p95": ("normalized_latency", ["p95"]),
        "normlat_p99": ("normalized_latency", ["p99"]),
        "ttft":        ("ttft_latency",        []),
        "ttft_p90":    ("ttft_latency",        ["p90"]),
        "ttft_p95":    ("ttft_latency",        ["p95"]),
        "ttft_p99":    ("ttft_latency",        ["p99"]),
        "tp":          ("throughput",          []),
        "tbt":          ("tbt_latency",          []),
    }
    if metric not in _metric_dispatch:
        supported = ", ".join(_metric_dispatch)
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: {supported}")
    method_name, args = _metric_dispatch[metric]
    kwargs = {"filter": filter} if filter is not None else {}
    return getattr(eo, method_name)(*args, **kwargs)

def read_monitor_stats(eo_id, path):
    path = os.path.join(path, f"{eo_id}.jsonl")

    stats = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            if any(v != 0 for k, v in entry.items() if k != "ts"):
                stats.append(entry)
    return stats

def aggregate_monitor_stats(stats: list[dict]) -> dict:
    gpu_ids = []
    for i in range(3):
        if f"gpu{i}_compute_pct" in stats[0]:
            gpu_ids.append(i)
    result = {}

    def val(entry, key):
        return entry.get(key) or 0.0

    # Per-GPU aggregates
    for g in gpu_ids:
        compute_vals = [val(e, f"gpu{g}_compute_pct") for e in stats]
        mem_vals     = [val(e, f"gpu{g}_mem_bus_pct") for e in stats]
        vllm_kv      = [val(e, f"vllm{g}_kv_pct") for e in stats]
        vllm_running = [val(e, f"vllm{g}_running") for e in stats]
        vllm_waiting = [val(e, f"vllm{g}_waiting") for e in stats]

        result[f"gpu{g}"] = {
            "compute_pct_mean": mean(compute_vals),
            "compute_pct_max":  max(compute_vals),
            "mem_bus_pct_mean": mean(mem_vals),
            "mem_bus_pct_max":  max(mem_vals),
            "kv_pct_mean":      mean(vllm_kv),
            "kv_pct_max":       max(vllm_kv),
            "running_mean":     mean(vllm_running),
            "waiting_mean":     mean(vllm_waiting),
        }

    # Aggregate across all GPUs
    all_compute = [val(e, f"gpu{g}_compute_pct") for e in stats for g in gpu_ids]
    all_mem     = [val(e, f"gpu{g}_mem_bus_pct")  for e in stats for g in gpu_ids]
    all_kv      = [val(e, f"vllm{g}_kv_pct")      for e in stats for g in gpu_ids]
    all_running = [val(e, f"vllm{g}_running")      for e in stats for g in gpu_ids]
    all_waiting = [val(e, f"vllm{g}_waiting")      for e in stats for g in gpu_ids]

    result["all"] = {
        "compute_pct_mean": mean(all_compute),
        "compute_pct_max":  max(all_compute),
        "mem_bus_pct_mean": mean(all_mem),
        "mem_bus_pct_max":  max(all_mem),
        "kv_pct_mean":      mean(all_kv),
        "kv_pct_max":       max(all_kv),
        "running_mean":     mean(all_running),
        "waiting_mean":     mean(all_waiting),
    }

    return result