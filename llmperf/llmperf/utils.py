import importlib.metadata
import json
import subprocess

from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

def get_version(package_name: str) -> str:
    try:
        version = importlib.metadata.version(package_name)
        
        dist = importlib.metadata.distribution(package_name)
        dist_name = dist.metadata["Name"].replace("-", "_")
        dist_info_dir = Path(dist.locate_file("")).joinpath(f"{dist_name}-{version}.dist-info")
        direct_url_path = dist_info_dir / "direct_url.json"
        
        if direct_url_path.exists():
            with open(direct_url_path) as f:
                data = json.load(f)
                vcs_info = data.get("vcs_info", {})
                if "commit_id" in vcs_info:
                    return vcs_info.get("requested_revision") or vcs_info["commit_id"]

                url = data.get("url", "")
                parsed = urlparse(url)
                if parsed.scheme == "file":
                    path = Path(unquote(parsed.path))
                    if (path / ".git").exists():
                        try:
                            tag = subprocess.check_output(["git", "describe", "--tags"], cwd=path).decode().strip()
                            return tag
                        except subprocess.CalledProcessError:
                            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()[:9]
                            return commit

        return version
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def create_experiment_id(
    workload: str, model: str, approach: str,
    gpu_util: float, swap_space: int, num_gpu_blocks: Optional[int],
    max_model_len: Optional[int], max_num_batched_tokens: Optional[int],
    **kwargs
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version = get_version("vllm")

    max_model_len = max_model_len if max_model_len is not None else "def"
    max_num_batched_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else "def"
    num_gpu_blocks = num_gpu_blocks if num_gpu_blocks is not None else "def"

    parts = [
        workload,
        model,
        approach,
        version,
        timestamp,
        f"maxlen{max_model_len}",
        f"batch{max_num_batched_tokens}",
        f"blocks{num_gpu_blocks}",
        f"gpu{gpu_util}",
        f"swap{swap_space}",
    ]

    for key, value in kwargs.items():
        parts.append(f"{key}{value}")

    return "__".join(parts)
