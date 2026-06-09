#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-cfgs/LAG_CLIP_Weak.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

steps=(
#  "weak_a"
#  "weak_b"
#  "weak_r"
  "eval_weak_r"
#  "clip_teacher"
#  "score_unlabeled"
#  "select_pseudo"
#  "clip_student"
  "eval_clip_official"
#  "clip_student_fusion"
  "eval_clip_student_fusion"
)

for mode in "${steps[@]}"; do
  echo "==> Running mode ${mode} with ${CONFIG}"
  python main.py --config "${CONFIG}" --mode "${mode}"
done

echo "==> Completed LAG DDAD -> CLIP teacher -> CLIP student -> CLIP student fusion pipeline."
