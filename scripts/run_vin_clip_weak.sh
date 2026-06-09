#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-cfgs/Vin_CLIP_Weak.yaml}"

#python main.py --config "$CONFIG" --mode weak_a
#python main.py --config "$CONFIG" --mode weak_b
#python main.py --config "$CONFIG" --mode weak_r --refine dual
python main.py --config "$CONFIG" --mode eval_weak_r
#python main.py --config "$CONFIG" --mode clip_teacher
#python main.py --config "$CONFIG" --mode score_unlabeled
#python main.py --config "$CONFIG" --mode select_pseudo
#python main.py --config "$CONFIG" --mode clip_student
#python main.py --config "$CONFIG" --mode eval_clip_official
#python main.py --config "$CONFIG" --mode clip_student_fusion
#python main.py --config "$CONFIG" --mode eval_clip_student_fusion

echo "==> Completed weakly-supervised CLIP student recovery evaluation."
