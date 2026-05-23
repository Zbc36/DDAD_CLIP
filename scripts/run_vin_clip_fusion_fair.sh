#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-cfgs/Vin_CLIP_Fusion_Fair.yaml}"

python main.py --config "$CONFIG" --mode weak_a
python main.py --config "$CONFIG" --mode weak_b
python main.py --config "$CONFIG" --mode clip
python main.py --config "$CONFIG" --mode eval_clip
python main.py --config "$CONFIG" --mode cache_fusion
python main.py --config "$CONFIG" --mode fusion
python main.py --config "$CONFIG" --mode eval_all

echo "==> Completed fair-baseline CLIP/Fusion pipeline."
