#!/usr/bin/env python3
"""
Orchestrator: launch a baseline serving system, wait until it's healthy,
run the benchmark, then tear everything down.

Usage (CLI):
    python orchestrate.py --config config.yaml
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import yaml
import logging
from pathlib import Path

import tempfile

from utils import wait_until_healthy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("orchestrator")

# ── Baseline launchers ─────────────────────────────────────────────────────────

def _script(script_dir: str, name: str) -> str:
    return str(Path(script_dir) / name)


def _launch(cmd: list[str], label: str, env: dict | None = None) -> subprocess.Popen:
    logger.info(f"Launching {label}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)


def launch_baseline(b_cfg: dict) -> tuple[list[subprocess.Popen], list[str]]:
    """
    Launch the appropriate baseline processes.

    Returns:
        (procs, server_urls) — all spawned Popen objects and the base URLs
        that should be health-checked before starting the benchmark.
    """
    name       = b_cfg["name"]
    model      = b_cfg["model"]
    args       = b_cfg.get("args", {})

    repo_root = os.path.dirname(os.getcwd())
    script_dir = repo_root
    vllm_bin = os.path.join(repo_root, "vllm/.venv/bin/vllm")
    rps_bin = os.path.join(repo_root, "rps-serve/.venv/bin/vllm")

    env = os.environ.copy()

    # Helper to split "0,1" → ["0", "1"]
    def ports_to_urls(ports_str: str, host: str = "localhost") -> list[str]:
        return [f"http://{host}:{p.strip()}" for p in str(ports_str).split(",")]

    procs       = []
    server_urls = []

    # ── mod-serve ─────────────────────────────────────────────────────────────
    if name == "mod-serve":
        venv_bin = str(Path(vllm_bin).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        cmd = [
            vllm_bin, _script(script_dir, "mod-serve.py"),
            "--model",                model,
            "--encode-gpus",          str(args["encode_gpus"]),
            "--prefill-decode-gpus",  str(args["prefill_decode_gpus"]),
            "--encode-ports",         str(args["encode_ports"]),
            "--prefill-decode-ports", str(args["prefill_decode_ports"]),
        ]
        if "tensor_parallel_size" in args:
            cmd += ["--tensor-parallel-size", str(args["tensor_parallel_size"])]
        if "max_model_len" in args:
            cmd += ["--max-model-len", str(args["max_model_len"])]
        if "max_num_batched_tokens" in args:
            cmd += ["--max-num-batched-tokens", str(args["max_num_batched_tokens"])]
        if "num_gpu_blocks_override" in args:
            cmd += ["--num-gpu-blocks-override", str(args["num_gpu_blocks_override"])]
        procs.append(_launch(cmd, "mod-serve", env=env))

        server_urls += ports_to_urls("10001")
        server_urls += ports_to_urls(args["encode_ports"])
        server_urls += ports_to_urls(args["prefill_decode_ports"])
    # ── mod-serve-rps ─────────────────────────────────────────────────────────
    elif name == "mod-serve-rps":
        venv_bin = str(Path(vllm_bin).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        cmd = [
            vllm_bin, _script(script_dir, "mod-serve-rps.py"),
            "--model",                model,
            "--encode-gpus",          str(args["encode_gpus"]),
            "--prefill-decode-gpus",  str(args["prefill_decode_gpus"]),
            "--encode-ports",         str(args["encode_ports"]),
            "--prefill-decode-ports", str(args["prefill_decode_ports"]),
        ]
        if "tensor_parallel_size" in args:
            cmd += ["--tensor-parallel-size", str(args["tensor_parallel_size"])]
        if "max_model_len" in args:
            cmd += ["--max-model-len", str(args["max_model_len"])]
        if "max_num_batched_tokens" in args:
            cmd += ["--max-num-batched-tokens", str(args["max_num_batched_tokens"])]
        if "num_gpu_blocks_override" in args:
            cmd += ["--num-gpu-blocks-override", str(args["num_gpu_blocks_override"])]
        procs.append(_launch(cmd, "mod-serve", env=env))

        server_urls += ports_to_urls("10001")
        server_urls += ports_to_urls(args["encode_ports"])
        server_urls += ports_to_urls(args["prefill_decode_ports"])

    # ── vllm (single process, TP) ─────────────────────────────────────────────
    elif name == "vllm":
        venv_bin = str(Path(vllm_bin).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        max_model_len = str(getattr(args, "max_model_len", None) or "32768")
        cmd = [
            vllm_bin, "serve", model,
            "--enforce-eager", "--seed", "1024",
            "--dtype", "float16",
            "--max-model-len", max_model_len,
            "--trust-remote-code",
            "--enable-request-id-headers",
            "--port", "10001"
        ]
        if "tensor_parallel_size" in args:
            cmd += ["--tensor-parallel-size", str(args["tensor_parallel_size"])]
        if "max_num_batched_tokens" in args:
            cmd += ["--max-num-batched-tokens", str(args["max_num_batched_tokens"])]
        if "num_gpu_blocks_override" in args:
            cmd += ["--num-gpu-blocks-override", str(args["num_gpu_blocks_override"])]
        procs.append(_launch(cmd, "vllm", env=env))
        server_urls += (ports_to_urls("10001"))
    elif name in ["rps-serve", "edf", "catfcfs"]:
        venv_bin = str(Path(rps_bin).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        max_model_len = str(getattr(args, "max_model_len", None) or "32768")
        cmd = [
            rps_bin, "serve", model,
            "--enforce-eager", "--seed", "1024",
            "--dtype", "float16",
            "--max-model-len", max_model_len,
            "--trust-remote-code",
            "--enable-request-id-headers",
            "--port", "10001"
        ]
        if "tensor_parallel_size" in args:
            cmd += ["--tensor-parallel-size", str(args["tensor_parallel_size"])]
        if "max_num_batched_tokens" in args:
            cmd += ["--max-num-batched-tokens", str(args["max_num_batched_tokens"])]
        if "num_gpu_blocks_override" in args:
            cmd += ["--num-gpu-blocks-override", str(args["num_gpu_blocks_override"])]
        if "scheduling_policy" in args:
            cmd += ["--scheduling-policy", args["scheduling_policy"]]
        if "classifier" in args:
            cmd += ["--classifier", args["classifier"]]
        if "slo_registry_files" in args:
            cmd += ["--slo-registry-files", args["slo_registry_files"]]
        procs.append(_launch(cmd, name, env=env))
        server_urls += (ports_to_urls("10001"))
    else:
        raise ValueError(f"Unknown baseline: {name!r}")

    return procs, server_urls


# ── Teardown ──────────────────────────────────────────────────────────────────

def kill_all(procs: list[subprocess.Popen]) -> None:
    logger.info("Shutting down server processes...")
    for p in procs:
        try:
            p.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
    gone = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and procs:
        for p in list(procs):
            if p.poll() is not None:
                gone.append(procs.pop(procs.index(p)))
        time.sleep(0.5)
    for p in procs:                       # hard kill anything still alive
        try:
            p.kill()
        except ProcessLookupError:
            pass
    logger.info("All servers stopped.")


# ── Orchestration entry point ─────────────────────────────────────────────────

def main(config: dict | str) -> None:
    """
    Launch a baseline, wait for it to be healthy, run the benchmark, tear down.

    Args:
        config:   Benchmark + baseline config — dict, YAML file path, raw YAML
                  string, or None to parse from CLI.
    """
    # ── resolve config ────────────────────────────────────────────────────────
    if isinstance(config, str):
        try:
            with open(config) as f:
                config_data = yaml.safe_load(f)
        except (FileNotFoundError, OSError):
            config_data = yaml.safe_load(config)
    elif isinstance(config, dict):
        config_data = config
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    repo_root = os.path.dirname(os.getcwd())
    b_cfg = config_data.get("baseline_config", {})

    m_config = config_data.get("monitor_config", {})
    m_config["out_path"] = os.path.join(repo_root, "artifacts/monitor-stats")

    r_config = config_data.get("runner_config", {})
    r_config["output_path"] = os.path.join(repo_root, "artifacts/outputs")
    r_config["output_log_path"] = os.path.join(repo_root, "artifacts/benchmark-log.jsonl")

    # ── launch → wait → benchmark → teardown ─────────────────────────────────
    procs = []
    try:
        procs, server_urls = launch_baseline(b_cfg)

        wait_until_healthy(
            server_urls,
            logger,
            timeout=b_cfg.get("health_timeout", 120),
            interval=b_cfg.get("health_interval", 2),
        )
        
        benchmark_python = os.path.join(repo_root, ".venv/bin/python")
        benchmark_script = os.path.join(repo_root, "experiments/benchmark.py")
        config_json      = json.dumps(config_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(config_json)
            tmp_path = f.name

        try:
            cmd = [benchmark_python, "-u", benchmark_script, "--config", tmp_path]
            logger.info(f"Running benchmark: {' '.join(cmd)}")
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Benchmark exited with code {result.returncode}")
        finally:
            os.unlink(tmp_path)
    finally:
        kill_all(procs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch a baseline and run the benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    
    args     = parser.parse_args()
    config   = args.config
    
    main(config)