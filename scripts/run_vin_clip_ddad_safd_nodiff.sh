#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-cfgs/Vin_CLIP_DDAD_SAFD_NoDiff.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

steps=(
  "a"
  "b"
  "clip"
  "cache_fusion"
  "fusion"
  "eval_all"
)

for mode in "${steps[@]}"; do
  echo "==> Running mode ${mode} with ${CONFIG}"
  python main.py --config "${CONFIG}" --mode "${mode}"
done

echo "==> Completed CLIP + DDAD SAFD pipeline without diffusion."
