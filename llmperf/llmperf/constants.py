import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))

MODELS_DIR = "/srv/muse-lab/models" # os.path.join(ROOT_DIR, "models")
DATASETS_DIR = "/srv/muse-lab/datasets-v1" # os.path.join(ROOT_DIR, "datasets")

ARTIFACTS_DIR = "/home/konstantinos.papaioannou/rps-serve/artifacts" # os.path.join(ROOT_DIR, "artifacts")
EXPERIMENTS_OUTPUTS_DIR = os.path.join(ARTIFACTS_DIR, "outputs")
EXPERIMENTS_ENGINE_STATS_DIR = os.path.join(ARTIFACTS_DIR, "stats")
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
WORKLOADS_DIR = os.path.join(ARTIFACTS_DIR, "workloads")
EXPERIMENTS_LOG = os.path.join(ARTIFACTS_DIR, "benchmark-log.jsonl")

ALL_STRATEGY_PARAMS = {
    "scene_change": {"content_threshold": 27.0},
    "motion_based": {"motion_threshold": 1.0},
    "sharpness_based": {"sharpness_threshold": 100.0},
}