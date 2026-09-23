#!/usr/bin/env python3

import argparse
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


import logging

from utils import (
    check_required_files,
    check_num_gpus,
    ensure_python_library_installed,
    make_cleanup,
    wait_until_healthy
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mod-serve")

proxy_port = 30001
libs = ("pandas", "datasets", "vllm", "quart", "msgpack")
local_media_path = "/"

proxy_name = "mod-serve-proxy.py"
pids: list[subprocess.Popen] = []
cleanup = make_cleanup(logger, proxy_name=proxy_name, pids=pids)
signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGUSR1, cleanup)

# =============================================================================
# Server helpers
# =============================================================================

def build_ec_config(role: str) -> str:
    config = {
        "ec_connector":"ECExampleConnector",
        "ec_role": role if role == "ec_producer" else "ec_consumer",        
        "ec_connector_extra_config": {
            "shared_storage_path": "/tmp/ec_cache"
        }
    }
    return json.dumps(config)


def launch_vllm_server(
    *,
    role: str,
    gpu_id: str,
    port: int,
    log_file: str,
    model: str,
    local_media_path: str,
    tp: str,
    mml: str,
    mnbt: str | None,
    nbo: str | None
) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", model,
        "--enforce-eager",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--seed", "1024",
        "--dtype", "float16",
        "--max-model-len", mml,
        "--trust-remote-code",
        "--enable-request-id-headers",
        "--allowed-local-media-path", local_media_path,
    ]

    if mnbt:
        cmd += ["--max-num-batched-tokens", mnbt]
    if nbo:
        cmd += ["--num-gpu-blocks-override", nbo]
    
    if role == "ec_producer":
        cmd += ["--no-enable-prefix-caching"]
        cmd += ["--gpu-memory-utilization", "0.01"]
        cmd += ["--tensor-parallel-size", tp]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}

    ec_config = build_ec_config(role)
    cmd += ["--ec-transfer-config", ec_config]

    if role == "ec_consumer":
        cmd += ["--tensor-parallel-size", tp]
        cmd += ["--scheduling-policy", "priority"]
        if model == "/srv/muse-lab/models/llava-onevision-qwen2-72b-ov-chat-hf":
            cmd += ["--classifier", "naive"]
        else:
            cmd += ["--classifier", "smart"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
        cmd[0] = "/home/konstantinos.papaioannou/rps-serve/rps-v1/.venv/bin/vllm"

    with open(log_file, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
    return proc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM Disaggregated Serving - P2P NCCL XeYpZd Architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID to serve",
    )
    parser.add_argument(
        "--prefill-decode-gpus",
        default="0",
        help="Comma-separated GPU IDs for prefill servers (e.g. '0' or '0,1')",
    )
    parser.add_argument(
        "--encode-gpus",
        default="1",
        help="Comma-separated GPU IDs for encode servers (e.g. '2' or '2,3')",
    )
    parser.add_argument(
        "--prefill-decode-ports",
        default="20003",
        help="Comma-separated ports for prefill servers (e.g. '20003' or '20003,20004')",
    )
    parser.add_argument(
        "--encode-ports",
        default="20007",
        help="Comma-separated ports for encode servers (e.g. '20007' or '20007,20008')",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        default="1",
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--max-model-len",
        default="32768",
        help="Max model length",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        default=None,
        help="Max num batch tokens",
    )
    parser.add_argument(
        "--num-gpu-blocks-override",
        default=None,
        help="Num GPU blocks override",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        dest="timeout_seconds",
        help="Seconds to wait for each server to become ready",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.tensor_parallel_size != "1":
        encode_gpus = [args.encode_gpus.strip()]
        prefill_decode_gpus = [args.prefill_decode_gpus.strip()]
    else:
        encode_gpus = [g.strip() for g in args.encode_gpus.split(",")]
        prefill_decode_gpus = [g.strip() for g in args.prefill_decode_gpus.split(",")]
    encode_ports = [int(p.strip()) for p in args.encode_ports.split(",")]
    prefill_decode_ports = [int(p.strip()) for p in args.prefill_decode_ports.split(",")]

    # --- Pre-flight checks ---
    check_required_files([proxy_name], logger)
    check_num_gpus(logger)
    for lib in libs:
        ensure_python_library_installed(lib, logger)

    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    logger.info("Launching disaggregated serving components...")
    logger.info("Log files: encode*.log, prefill*.log, decode*.log, proxy.log")

    shutil.rmtree("/tmp/ec_cache", ignore_errors=True)
    os.makedirs("/tmp/ec_cache", exist_ok=True)

    # --- Encoder servers ---
    encode_servers_urls = []
    logger.info(f"Starting {len(encode_gpus)} encoder server(s)...")
    for i, (gpu_id, port) in enumerate(zip(encode_gpus, encode_ports)):
        ec_port = 21001 + i
        log_file = f"encode{i + 1}.log"
        logger.info(f"Encode server {i + 1}: GPU {gpu_id}, Port {port}, EC Port {ec_port}")
        proc = launch_vllm_server(
            role="ec_producer",
            gpu_id=gpu_id,
            port=port,
            log_file=log_file,
            model=args.model,
            local_media_path=local_media_path,
            tp=args.tensor_parallel_size,
            mml=args.max_model_len,
            mnbt=args.max_num_batched_tokens,
            nbo=args.num_gpu_blocks_override
        )
        pids.append(proc)
        encode_servers_urls.append(f"http://10.10.4.97:{port}")

    # --- Prefill/Decode servers ---
    prefill_decode_servers_urls = []
    logger.info(f"Starting {len(prefill_decode_gpus)} prefill/decode server(s)...")
    for i, (gpu_id, port) in enumerate(zip(prefill_decode_gpus, prefill_decode_ports)):
        kv_port = 22001 + i
        log_file = f"prefill{i + 1}.log"
        logger.info(f"Prefill server {i + 1}: GPU {gpu_id}, Port {port}, EC Port {kv_port}")
        proc = launch_vllm_server(
            role="ec_consumer",
            gpu_id=gpu_id,
            port=port,
            log_file=log_file,
            model=args.model,
            local_media_path=local_media_path,
            tp=args.tensor_parallel_size,
            mml=args.max_model_len,
            mnbt=args.max_num_batched_tokens,
            nbo=args.num_gpu_blocks_override
        )
        pids.append(proc)
        prefill_decode_servers_urls.append(f"http://10.10.4.97:{port}")

    # --- Wait for readiness ---
    wait_until_healthy(
        [f"http://localhost:{port}" for port in prefill_decode_ports + encode_ports],
        logger, args.timeout_seconds
    )

    # --- Proxy server ---
    logger.info(f"Starting proxy server on port {proxy_port}...")
    proxy_proc = subprocess.Popen(
        [
            sys.executable, "mod-serve-proxy.py",
            "--host", "0.0.0.0",
            "--port", str(proxy_port),
            "--encode-servers-urls", ",".join(encode_servers_urls),
            "--prefill-decode-servers-urls", ",".join(prefill_decode_servers_urls)
        ],
        stdout=open("proxy.log", "w"),
        stderr=subprocess.STDOUT,
    )
    pids.append(proxy_proc)


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    os.environ.setdefault("UCX_TLS", "all")
    os.environ.setdefault("UCX_NET_DEVICES", "all")
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    main()