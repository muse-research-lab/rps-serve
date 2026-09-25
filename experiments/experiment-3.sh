#!/usr/bin/env bash
set -u

RATES="${RATES:-10.0}"
VARIANTS=(rps rps-nc rps-wo rps-so edf)
failed=()

for v in "${VARIANTS[@]}"; do
  cfg="configs/config-${v}-tp1.yaml"
  [[ -f "$cfg" ]] || { echo "missing $cfg"; failed+=("$v"); continue; }
  echo "==> $v (rates: $RATES)"
  python3 sweep.py --config "$cfg" --rates $RATES || failed+=("$v")
done

(( ${#failed[@]} )) && { echo "Failed: ${failed[*]}"; exit 1; }
echo "Experiment 3 completed."