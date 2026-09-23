#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "Using repo root: ${ROOT_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR}"
  uv venv --python 3.12.8 --seed --managed-python "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${ROOT_DIR}/workloads"

echo "Downloading workload data..."
curl -L "https://cloud.software.imdea.org/index.php/s/fw9DJZ8tkLY9RB6/download" | tar -xf - -C .

echo "Running preprocessing..."
python preprocessing.py --text --image --video

echo "Generating workloads..."
python generation.py

echo "Workloads generation complete."