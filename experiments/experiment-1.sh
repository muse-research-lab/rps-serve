#!/usr/bin/env bash
# Run serving sweeps across all system configs for each model.
# The last three models (internvl-3.5-large, gemma-4-large, qwen-3.5-large)
# only run when --full is given.
set -uo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [--full] [--dry-run]

  --full      Also run internvl-3.5-large, gemma-4-large and qwen-3.5-large
  --dry-run   Print the commands without running them
  -h, --help  Show this help
EOF
}

FULL=0
DRY_RUN=0
while (( $# )); do
    case "$1" in
        --full)    FULL=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

CONFIGS=(
    configs/config-rps-tp4.yaml
    configs/config-vllm-tp4.yaml
    configs/config-mod-serve-tp2.yaml
    configs/config-mod-serve-rps-tp2.yaml
)

FAILED=()

# run_sweeps <label> <rate> [extra sweep.py args...]
run_sweeps() {
    local label=$1 rate=$2
    shift 2
    echo "=== ${label} (rate ${rate}) ==="
    local cfg
    for cfg in "${CONFIGS[@]}"; do
        local cmd=(python3 sweep.py --config "$cfg" --rates "$rate" "$@")
        echo "+ ${cmd[*]}"
        (( DRY_RUN )) && continue
        "${cmd[@]}" || FAILED+=("${label}: ${cfg}")
    done
}

# Always run
run_sweeps llava-ov       15.0
run_sweeps llava-ov-large 2.5 --model llava-ov-large     --model-path ../llava-onevision-qwen2-72b-ov-chat-hf

# Only with --full
if (( FULL )); then
    run_sweeps internvl-3.5-large 5.0 --model internvl-3.5-large --model-path ../InternVL3_5-38B-HF
    run_sweeps gemma-4-large      6.0 --model gemma-4-large      --model-path ../gemma-4-31B-it
    run_sweeps qwen-3.5-large     6.0 --model qwen-3.5-large     --model-path ../Qwen3.5-27B
else
    echo "Skipping internvl-3.5-large, gemma-4-large, qwen-3.5-large (pass --full to include)"
fi

if (( ${#FAILED[@]} )); then
    echo
    echo "Failed runs (${#FAILED[@]}):" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
echo "Experiment 1 completed."