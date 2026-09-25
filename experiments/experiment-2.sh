#!/usr/bin/env bash
# Run serving sweeps at a fixed GPU budget (1, 2 or 4 GPUs) across the
# baselines that fit in that budget.
#
#   1 GPU : vllm-tp1, rps-tp1                                (llava-ov only)
#   2 GPUs: vllm-tp2, rps-tp2, mod-serve-tp1, mod-serve-rps-tp1
#           (llava-ov-large only runs vllm/rps at 2 GPUs)
#   4 GPUs: vllm-tp4, rps-tp4, mod-serve-tp2, mod-serve-rps-tp2
set -uo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") --gpus {1|2|4} [--dry-run]

  --gpus N    GPU budget for this run: 1, 2 or 4
  --dry-run   Print the commands without running them
  -h, --help  Show this help
EOF
}

GPUS=""
DRY_RUN=0
while (( $# )); do
    case "$1" in
        --gpus)
            [[ $# -ge 2 ]] || { echo "--gpus needs a value" >&2; usage >&2; exit 1; }
            GPUS=$2; shift ;;
        --gpus=*)  GPUS=${1#--gpus=} ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

case "$GPUS" in
    1|2|4) ;;
    "") echo "--gpus is required" >&2; usage >&2; exit 1 ;;
    *)  echo "Invalid --gpus value: $GPUS (expected 1, 2 or 4)" >&2; exit 1 ;;
esac

# ---- Baselines per GPU budget -------------------------------------------
# Monolithic baselines use all GPUs as TP; mod-serve baselines use TP = GPUs/2.
MONO_1GPU=(
    configs/config-rps-tp1.yaml
    configs/config-vllm-tp1.yaml
)
MONO_2GPU=(
    configs/config-rps-tp2.yaml
    configs/config-vllm-tp2.yaml
)
ALL_2GPU=(
    "${MONO_2GPU[@]}"
    configs/config-mod-serve-tp1.yaml
    configs/config-mod-serve-rps-tp1.yaml
)
ALL_4GPU=(
    configs/config-rps-tp4.yaml
    configs/config-vllm-tp4.yaml
    configs/config-mod-serve-tp2.yaml
    configs/config-mod-serve-rps-tp2.yaml
)

# ---- Rates ---------------------------------------------------------------
LLAVA_OV_RATES_1GPU=(1 3 5 6 7 8 9 10)
LLAVA_OV_RATES_2GPU=(1 3 5 6 7 8 9 10 12)
LLAVA_OV_RATES_4GPU=(1 3 5 6 7 8 9 10 12 15)
LLAVA_OV_LARGE_RATES=(0.5 0.75 1.0 1.25 1.5 2.0 2.25 2.5)

LLAVA_OV_LARGE_ARGS=(--model llava-ov-large --model-path ../llava-onevision-qwen2-72b-ov-chat-hf)

FAILED=()

# run_sweeps <label> <configs-array-name> <rates-array-name> [extra sweep.py args...]
run_sweeps() {
    local label=$1
    local -n _cfgs=$2 _rates=$3
    shift 3
    echo "=== ${label} @ ${GPUS} GPU(s) (rates ${_rates[*]}) ==="
    local cfg
    for cfg in "${_cfgs[@]}"; do
        local cmd=(python3 sweep.py --config "$cfg" --rates "${_rates[@]}" "$@")
        echo "+ ${cmd[*]}"
        (( DRY_RUN )) && continue
        if [[ ! -f $cfg ]]; then
            echo "Missing config: $cfg" >&2
            FAILED+=("${label}: ${cfg} (missing config)")
            continue
        fi
        "${cmd[@]}" || FAILED+=("${label}: ${cfg}")
    done
}

case "$GPUS" in
    1)
        run_sweeps llava-ov MONO_1GPU LLAVA_OV_RATES_1GPU
        echo "Skipping llava-ov-large (does not fit on 1 GPU)"
        ;;
    2)
        run_sweeps llava-ov       ALL_2GPU  LLAVA_OV_RATES_2GPU
        run_sweeps llava-ov-large MONO_2GPU LLAVA_OV_LARGE_RATES "${LLAVA_OV_LARGE_ARGS[@]}"
        ;;
    4)
        run_sweeps llava-ov       ALL_4GPU LLAVA_OV_RATES_4GPU
        run_sweeps llava-ov-large ALL_4GPU LLAVA_OV_LARGE_RATES "${LLAVA_OV_LARGE_ARGS[@]}"
        ;;
esac

if (( ${#FAILED[@]} )); then
    echo
    echo "Failed runs (${#FAILED[@]}):" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
echo "Experiment 2 (${GPUS} GPU) completed."