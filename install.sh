#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.8}"

echo "Using repo root: ${ROOT_DIR}"

command -v uv >/dev/null 2>&1 || {
  echo "Error: uv is not installed or not on PATH." >&2
  echo "Install it from: https://docs.astral.sh/uv/" >&2
  exit 1
}

setup_venv() {
  local project_dir="$1"
  local venv_dir="${project_dir}/.venv"

  if [[ ! -d "${venv_dir}" ]]; then
    echo "Creating venv for ${project_dir}"
    uv venv --python "${PYTHON_VERSION}" --seed --managed-python "${venv_dir}"
  else
    echo "Using existing venv for ${project_dir}"
  fi

  deactivate 2>/dev/null || true
  # shellcheck disable=SC1091
  source "${venv_dir}/bin/activate"
}

install_rps_serve() {
  local dir="${ROOT_DIR}/rps-serve"
  setup_venv "${dir}"

  export VLLM_USE_PRECOMPILED=1
  export VLLM_PRECOMPILED_WHEEL_COMMIT="d735968f6d634ec849268f18e3b84ceb494fee79"
  export SETUPTOOLS_SCM_PRETEND_VERSION="0.21.1rc1.dev120+g70c5a0596"
  export VLLM_PRECOMPILED_WHEEL_VARIANT="cu130"

  uv pip install --editable "${dir}" --torch-backend=cu130
  uv pip install joblib
}

install_vllm_baseline() {
  local dir="${ROOT_DIR}/vllm"
  setup_venv "${dir}"

  export VLLM_USE_PRECOMPILED=1
  export VLLM_PRECOMPILED_WHEEL_COMMIT="33ef1941e217a2126d745caec6c6130d6aec3b31"
  export SETUPTOOLS_SCM_PRETEND_VERSION="0.21.1rc1.dev120+g70c5a0596"
  export VLLM_PRECOMPILED_WHEEL_VARIANT="cu130"

  uv pip install --editable "${dir}" --torch-backend=cu130
}

install_requirements() {
  local dir="$1"
  setup_venv "${dir}"

  if [[ -f "${dir}/requirements.txt" ]]; then
    uv pip install -r "${dir}/requirements.txt"
  fi
}

install_editable() {
  local dir="$1"
  setup_venv "${dir}"
  uv pip install -e "${dir}"
}

install_rps_serve
install_vllm_baseline
install_requirements "${ROOT_DIR}"
install_editable "${ROOT_DIR}/llmperf"
install_editable "${ROOT_DIR}/guidellm"

echo "Installation complete."