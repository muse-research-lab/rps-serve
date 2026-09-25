#!/usr/bin/env bash
set -u

MODEL="${MODEL:-llava-ov}"
WORKLOADS=(text text-long image video)
failed=()

for w in "${WORKLOADS[@]}"; do
  cfg="config-${w}-${MODEL}.yaml"
  [[ -f "$cfg" ]] || { echo "missing $cfg"; failed+=("$w"); continue; }
  echo "==> $w ($MODEL)"
  python3 orchestrator.py --config "$cfg" || failed+=("$w")
done

(( ${#failed[@]} )) && { echo "Failed: ${failed[*]}"; exit 1; }
echo "Experment 4 (Characterization) done."