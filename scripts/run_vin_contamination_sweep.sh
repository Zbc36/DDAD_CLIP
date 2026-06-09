#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${1:-cfgs/Vin_CLIP_ContaminationStudy.yaml}"

for RATE in 0.00 0.01 0.02 0.05 0.10 0.20 0.40; do
  bash scripts/run_vin_contamination_rate.sh "$RATE" "$BASE_CONFIG"
done

echo "==> Completed Vin contamination sweep."
