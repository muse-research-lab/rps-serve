#!/usr/bin/env python3
"""
Sweep over request rates, running the full orchestrator for each.

Usage:
    python sweep.py --config config.yaml --rates 0.5 1.25 2.0 2.75
"""

import argparse
import copy
import yaml
import logging
import sys

from orchestrator import main as run_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sweep")

def main(config: str, rates: list[float]) -> None:
    with open(config) as f:
        base_config = yaml.safe_load(f)

    logger.info(f"Rates {rates}")

    for rate in rates:
        max_requests = int(rate * 100)
        logger.info(f"Starting run — rate={rate} max_requests={max_requests}")

        cfg = copy.deepcopy(base_config)
        cfg["runner_config"]["rate"]         = rate
        cfg["runner_config"]["max_requests"] = max_requests

        try:
            run_orchestrator(cfg)
            logger.info(f"Completed run — rate={rate}")
        except Exception as e:
            logger.error(f"Run failed at rate={rate}: {e}")
            raise

    sys.stdin.flush()
    sys.stdout.flush()

    logger.info(f"Finished all {len(rates)} runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep over request rates.")
    parser.add_argument("--config", required=True, help="Path to base YAML config file")
    parser.add_argument("--rates", required=True, type=float, nargs="+",
                        help="Request rates to sweep over, e.g. --rates 0.5 1.25 2.0")
    args = parser.parse_args()
    main(args.config, args.rates)