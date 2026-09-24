import argparse
import asyncio
import json
import pynvml
import re
import os
import requests
import threading
import time
import yaml
import logging

from datetime import datetime
from pathlib import Path

from llmperf.runner.config import RunnerConfig
from llmperf.runner.guidellm import GuideLLMRunner

from llmperf.config.models import Model
from llmperf.preprocessing.workload import Workload

from llmperf.promptpreparation.base import DefaultPromptPreparation, DefaultExtraPromptPreparation
from llmperf.promptpreparation.config import PromptPreparationConfig

from llmperf.postprocessing.output import ExperimentOutput


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("monitor")



models = {
    "llava-ov": Model(
        name="LLaVA-OneVision-7b",
        path="llava-onevision-qwen2-7b-ov-chat-hf",
        max_model_len=32768,
        alias="llava-ov",
        image_token_index=151646,
        video_token_index=151647
    ),
    "llava-ov-large": Model(
        name="LLaVA-OneVision-72b",
        path="llava-onevision-qwen2-72b-ov-chat-hf",
        max_model_len=32768,
        alias="llava-ov-large",
        image_token_index=151646,
        video_token_index=151647
    ),
    "gemma-4-large": Model(
        name="Gemma4-31B",
        path="gemma-4-31B-it",
        max_model_len=262144,
        alias="gemma-4-large",
        image_token_index=258880,
        video_token_index=258884
    ),
    "internvl-3.5-large": Model(
        name="InternVL3.5-38B",
        path="InternVL3_5-38B-HF",
        max_model_len=40960,
        alias="internvl-3.5-large",
        image_token_index=151671,
        video_token_index=151678
    ),
    "qwen-3.5-large": Model(
        name="Qwen3.5-27B",
        path="Qwen3.5-27B",
        max_model_len=262144,
        alias="qwen-3.5-large",
        image_token_index=248056,
        video_token_index=248057
    ),
}

# ── Monitor ────────────────────────────────────────────────────────────────────

class Monitor:
    """
    Collects GPU utilization and vLLM metrics at a fixed interval in a
    background thread. Call start() before the benchmark and stop() after;
    results are flushed to a JSONL file on stop().
    """

    def __init__(self, addrs: list[str], gpu_indices: list[int], interval: float, out_path: Path):
        self.addrs       = addrs
        self.gpu_indices = gpu_indices
        self.interval    = interval
        self.out_path    = out_path

        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._rows: list[dict] = []

    def _parse_metric(self, text: str, name: str) -> float | None:
        m = re.search(rf'^{re.escape(name)}\{{[^}}]*\}}\s+([\d.e+\-]+)', text, re.MULTILINE)
        return float(m.group(1)) if m else None

    def _fetch_vllm(self, addr: str) -> dict | None:
        try:
            url = f"http://{addr}/metrics"
            r = requests.get(url, timeout=2)
            r.raise_for_status()
            t = r.text
            return {
                "kv":      (self._parse_metric(t, "vllm:kv_cache_usage_perc") or 0.0) * 100,
                "running": self._parse_metric(t, "vllm:num_requests_running") or 0.0,
                "waiting": self._parse_metric(t, "vllm:num_requests_waiting") or 0.0,
            }
        except Exception as e:
            logger.warning(f"{url} — {e}")
            return None

    def start(self) -> None:
        pynvml.nvmlInit()
        total = pynvml.nvmlDeviceGetCount()
        if any(i >= total for i in self.gpu_indices):
            raise ValueError(
                f"GPU index out of range — system has {total} GPU(s): {list(range(total))}"
            )
        self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices}
        logger.info(f"Starting — GPUs {self.gpu_indices}, endpoints {self.addrs}")
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and block until it exits."""
        self._stop_event.set()
        self._thread.join()
        pynvml.nvmlShutdown()
        self._flush()
        logger.info(f"Saved {len(self._rows)} samples → {self.out_path}")

    def _run(self) -> None:
        start = time.monotonic()
        while not self._stop_event.is_set():
            t0  = time.monotonic()
            row: dict = {"ts": round(t0 - start, 3)}

            for i, handle in self._handles.items():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                row[f"gpu{i}_compute_pct"] = util.gpu
                row[f"gpu{i}_mem_bus_pct"] = util.memory

            for idx, addr in enumerate(self.addrs):
                v = self._fetch_vllm(addr)
                prefix = f"vllm{idx}"
                row[f"{prefix}_kv_pct"]  = v["kv"]      if v else None
                row[f"{prefix}_running"] = v["running"]  if v else None
                row[f"{prefix}_waiting"] = v["waiting"]  if v else None

            self._rows.append(row)

            elapsed = time.monotonic() - t0
            sleep_for = self.interval - elapsed
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)

    def _flush(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")


async def run_experiments(config_data: dict, monitor: Monitor) -> None:
    """
    Set up and execute a single benchmark run from a parsed config dictionary.

    Resolves the model and workload by alias, builds the prompt preparation
    and runner configs, then runs the benchmark via GuideLLMRunner.
    GPU and vLLM metrics are sampled in a background thread for the full
    duration of the run and saved to a JSONL file on completion.

    Args:
        config_data: Parsed benchmark configuration containing:
            - model_alias (str): Registered model identifier.
            - workload_alias (str): Registered workload identifier.
            - prompt_config (dict): Kwargs forwarded to PromptPreparationConfig.
            - runner_config (dict): Kwargs forwarded to RunnerConfig.
            - monitor_config (dict): Monitoring settings:
                - addrs (list[str]): vLLM /metrics endpoints to poll.
                - gpus (list[int]): GPU indices to sample.
                - interval (float): Seconds between samples (default 0.5).
                - out_path (str): Output path for the JSONL file.
        monitor: Already-started Monitor instance, stopped by the caller
                 after the event loop exits.
    """
    model    = models[config_data["model_alias"]]
    model.path = config_data.get("baseline_config", {}).get("model", None)

    workload = Workload(
        name=config_data["workload_alias"],
        path=os.path.join(os.path.dirname(os.getcwd()), "workloads/static"),
        alias=config_data["workload_alias"]
    )

    p_config = PromptPreparationConfig(model=model, **config_data["prompt_config"])
    
    if config_data["model_alias"] in ["gemma-4-large", "qwen-3.5-large"]:
        prompt_prep = DefaultExtraPromptPreparation(p_config)
    else:
        prompt_prep = DefaultPromptPreparation(p_config)

    r_config = RunnerConfig(
        model=model,
        workload=workload,
        prompt_preparation=prompt_prep,
        **config_data["runner_config"],
    )
    runner = GuideLLMRunner(r_config)

    try:
        await runner.run()
    finally:
        if runner._id:
            out_dir = monitor.out_path.parent
            monitor.out_path = out_dir / f"{runner._id}.jsonl"

            eo = ExperimentOutput(runner._id, output_path=config_data["runner_config"]["output_path"])
            eo.load()

            print("Avg. E2E Latency:", eo.e2e_latency())
            print("Avg. TTFT Latency:", eo.ttft_latency())
            print("Avg. TBT Latency:", eo.tbt_latency())


def main(config: dict | str) -> None:
    """
    Run the benchmark with the given config.

    The Monitor is started before the event loop and stopped after it fully
    closes, avoiding thread-join conflicts with asyncio shutdown.

    Args:
        config: One of:
          - dict: already-parsed config
          - str:  path to a YAML file, or a raw YAML string
    """
    if isinstance(config, dict):
        config_data = config
    elif isinstance(config, str):
        try:
            with open(config) as f:
                config_data = yaml.safe_load(f)
        except (FileNotFoundError, OSError):
            config_data = yaml.safe_load(config)
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    # Build monitor here so its lifetime is owned by main(), not the event loop
    monitor_cfg = config_data.get("monitor_config", {})
    stamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir     = Path(monitor_cfg.get("out_path", "."))
    out_path    = out_dir / f"monitor_{stamp}.jsonl"

    monitor = Monitor(
        addrs       = monitor_cfg.get("addrs", []),
        gpu_indices = monitor_cfg.get("gpus", []),
        interval    = monitor_cfg.get("interval", 0.5),
        out_path    = out_path,
    )

    monitor.start()
    try:
        # Event loop runs and exits cleanly — monitor thread is untouched
        asyncio.run(run_experiments(config_data, monitor))
    finally:
        # Stop AFTER the event loop is fully closed — no threading conflicts
        monitor.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)