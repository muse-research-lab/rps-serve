import importlib
import logging
import requests
import subprocess
import sys
import time

from pathlib import Path

def check_required_files(required: list[str], logger: logging.Logger) -> None:
    script_dir = Path(__file__).parent
    for fname in required:
        if not (script_dir / fname).exists():
            logger.error(f"Required file {fname} not found in {script_dir}")
            sys.exit(1)


def check_num_gpus(logger: logging.Logger) -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        num_gpus = len([l for l in result.stdout.strip().splitlines() if l])
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Could not query nvidia-smi. Make sure CUDA drivers are installed.")
        sys.exit(1)

    if num_gpus < 2:
        logger.error(f"You need at least 2 GPUs to run disaggregated prefill. Found: {num_gpus}")
        sys.exit(1)

    logger.info(f"Found {num_gpus} GPUs.")


def ensure_python_library_installed(lib: str, logger: logging.Logger) -> None:
    logger.info(f"Checking if {lib} is installed...")
    try:
        importlib.import_module(lib)
        logger.info(f"{lib} is installed.")
    except ImportError:
        logger.error(f"{lib} is NOT installed. Please run: pip install {lib}")
        sys.exit(1)


def make_cleanup(logger: logging.Logger, proxy_name: str, pids: list[subprocess.Popen]):
    def cleanup(*_) -> None:
        logger.info("Stopping everything...")
        for proc in pids:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

        subprocess.run(
            ["pkill", "-9", "-f", proxy_name],
            capture_output=True,
        )

        for proc in pids:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        subprocess.run(
            ["pkill", "-9", "-f", "VLLM::EngineCore"],
            capture_output=True,
        )

        subprocess.run(
            ["pkill", "-9", "-f", "from multiprocessing"],
            capture_output=True,
        )

        subprocess.run(
            ["pkill", "-9", "-f", "/home/konstantinos.papaioannou/rps-serve/scripts/benchmark.py"],
            capture_output=True
        )

        logger.info("Cleanup complete. Exiting.")
        sys.exit(0)

    return cleanup


def _health_url(base_url: str) -> str:
    """Derive a health/metrics URL from a base server URL."""
    return base_url.rstrip("/") + "/health"


def wait_until_healthy(
    urls: list[str],
    logger: logging.Logger,
    timeout: float = 120,
    interval: float = 2,
) -> None:
    """
    Poll each URL's /health endpoint until all respond with HTTP 200,
    or raise TimeoutError if they don't come up within `timeout` seconds.

    Args:
        urls:     Base server URLs to check (e.g. "http://localhost:20005").
        timeout:  Maximum seconds to wait before giving up.
        interval: Seconds between poll attempts.
    """
    health_urls = [_health_url(u) for u in urls]
    deadline    = time.monotonic() + timeout
    pending     = set(health_urls)

    logger.info(f"Waiting for {len(pending)} server(s) to be healthy...")

    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Servers did not become healthy within {timeout}s: {pending}"
            )
        for url in list(pending):
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    logger.info(f"✓ {url}")
                    pending.discard(url)
            except requests.RequestException:
                pass
        if pending:
            time.sleep(interval)

    logger.info("All servers healthy.")