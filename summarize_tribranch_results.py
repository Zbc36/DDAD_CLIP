import csv
import json
import os
import re


RESULT_LINE_PATTERN = re.compile(r"^(?P<name>.+?)\s+AUC:(?P<auc>[0-9.]+)\s+AP:(?P<ap>[0-9.]+)$")


def _parse_results_txt(path):
    parsed = {}
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = RESULT_LINE_PATTERN.match(line)
            if match is None:
                continue
            name = match.group("name").strip()
            parsed[name] = {
                "auc": float(match.group("auc")),
                "ap": float(match.group("ap")),
            }
    return parsed


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _best_ddad_variant(results_txt):
    candidates = {}
    if "DDAD-inter" in results_txt:
        candidates["DDAD-inter"] = results_txt["DDAD-inter"]
    if "DDAD-intra" in results_txt:
        candidates["DDAD-intra"] = results_txt["DDAD-intra"]
    if len(candidates) == 0:
        raise RuntimeError("Could not find DDAD-inter or DDAD-intra in results.txt")
    best_name, best_metrics = max(candidates.items(), key=lambda item: item[1]["auc"])
    return best_name, best_metrics


def _collect_dataset_row(dataset_name, out_dir):
    baseline_metrics = _parse_results_txt(os.path.join(out_dir, "results.txt"))
    eval_r_metrics = _load_json(os.path.join(out_dir, "eval_r_results.json"))
    eval_all_metrics = _load_json(os.path.join(out_dir, "eval_all_results.json"))

    best_ddad_variant, best_ddad_metrics = _best_ddad_variant(baseline_metrics)
    real_metrics = eval_all_metrics["real_image_level_metrics"]
    best_new_branch, best_new_metrics = max(real_metrics.items(), key=lambda item: item[1]["auc"])

    return {
        "dataset": dataset_name,
        "out_dir": out_dir,
        "baseline_ddad_variant": best_ddad_variant,
        "baseline_ddad_auc": best_ddad_metrics["auc"],
        "baseline_ddad_ap": best_ddad_metrics["ap"],
        "baseline_asr_auc": float(eval_r_metrics["auc"]),
        "baseline_asr_ap": float(eval_r_metrics["ap"]),
        "tribranch_ddad_auc": float(real_metrics["ddad"]["auc"]),
        "tribranch_ddad_ap": float(real_metrics["ddad"]["ap"]),
        "tribranch_diffusion_auc": float(real_metrics["diffusion"]["auc"]),
        "tribranch_diffusion_ap": float(real_metrics["diffusion"]["ap"]),
        "tribranch_clip_auc": float(real_metrics["clip"]["auc"]),
        "tribranch_clip_ap": float(real_metrics["clip"]["ap"]),
        "tribranch_fusion_auc": float(real_metrics["fusion"]["auc"]),
        "tribranch_fusion_ap": float(real_metrics["fusion"]["ap"]),
        "best_new_branch": best_new_branch,
        "best_new_branch_auc": float(best_new_metrics["auc"]),
        "best_new_branch_ap": float(best_new_metrics["ap"]),
        "delta_vs_asr_auc": float(best_new_metrics["auc"]) - float(eval_r_metrics["auc"]),
        "delta_vs_asr_ap": float(best_new_metrics["ap"]) - float(eval_r_metrics["ap"]),
    }


def main():
    experiments = [
        ("vin", "output/TRIBRANCH/vin_ae_ens3_g1"),
        ("bratumor", "output/TRIBRANCH/bratumor_ae_ens3_g2"),
        ("lag", "output/TRIBRANCH/lag_ae_ens3_g3"),
    ]
    rows = [_collect_dataset_row(dataset_name, out_dir) for dataset_name, out_dir in experiments]
    output_path = os.path.join("output", "TRIBRANCH", "tribranch_3dataset_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved summary to {}".format(output_path))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
