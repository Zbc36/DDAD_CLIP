#!/usr/bin/env bash
set -euo pipefail

RATE="${1:-0.00}"
BASE_CONFIG="${2:-cfgs/Vin_CLIP_ContaminationStudy.yaml}"

RATE_TAG="$(python - "$RATE" <<'PY'
import sys
rate = float(sys.argv[1])
print(f"{rate:.2f}".replace(".", "p"))
PY
)"

RUN_CONFIG="cfgs/.tmp_Vin_CLIP_ContaminationStudy_ar_${RATE_TAG}.yaml"

python - "$BASE_CONFIG" "$RUN_CONFIG" "$RATE" <<'PY'
import os
import sys
import yaml

base_config, run_config, rate_text = sys.argv[1:4]
rate = float(rate_text)
with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("Data", {})
cfg["Data"]["contamination_study_enabled"] = True
cfg["Data"]["unlabeled_train_abnormal_ratio"] = rate

cfg.setdefault("Exp", {})
rate_tag = f"{rate:.2f}".replace(".", "p")
cfg["Exp"]["out_dir"] = f"output/AEU/Vin_ContamStudy/ar_{rate_tag}/"

with open(run_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

cleanup() {
  rm -f "$RUN_CONFIG"
}
trap cleanup EXIT

echo "==> Running Vin contamination study with train abnormal ratio=${RATE}"
python main.py --config "$RUN_CONFIG" --mode weak_a
python main.py --config "$RUN_CONFIG" --mode weak_b
python main.py --config "$RUN_CONFIG" --mode weak_r --refine dual
python main.py --config "$RUN_CONFIG" --mode eval_weak_r
#python main.py --config "$RUN_CONFIG" --mode clip_teacher
python main.py --config "$RUN_CONFIG" --mode score_unlabeled
python main.py --config "$RUN_CONFIG" --mode select_pseudo

if python - "$RATE" <<'PY'
import sys
sys.exit(0 if abs(float(sys.argv[1])) <= 1.0e-12 else 1)
PY
then
  echo "==> normal-only baseline: skip clip_student and run teacher-initialized fusion"
  python main.py --config "$RUN_CONFIG" --mode clip_student_fusion
  python main.py --config "$RUN_CONFIG" --mode eval_clip_student_fusion
else
  python main.py --config "$RUN_CONFIG" --mode clip_student
  python main.py --config "$RUN_CONFIG" --mode eval_clip_official
  python main.py --config "$RUN_CONFIG" --mode clip_student_fusion
  python main.py --config "$RUN_CONFIG" --mode eval_clip_student_fusion
fi

echo "==> Completed Vin contamination study for ratio=${RATE}"
