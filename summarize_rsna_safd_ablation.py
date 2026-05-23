import csv
import json
import os


EXPERIMENTS = [
    ("E0", "nosafd"),
    ("E1", "data_only"),
    ("E2", "recon_band_only"),
    ("E3", "clip_prior_only"),
    ("E4", "clip_post_only"),
    ("E5", "output_refine_only"),
    ("E6", "full_current"),
    ("E7", "full_extended"),
]


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _collect_row(root_dir, experiment_id, experiment_name):
    out_dir = os.path.join(
        root_dir,
        "output",
        "TRIBRANCH",
        "rsna_safd_ablation",
        "{}_{}".format(experiment_id, experiment_name),
    )
    metrics = _load_json(os.path.join(out_dir, "eval_all_results.json"))
    real_metrics = metrics["real_image_level_metrics"]
    synthetic_metrics = metrics.get("synthetic_image_level_metrics", {})
    synthetic_clip = synthetic_metrics.get("clip", {"auc": 0.0, "ap": 0.0})
    synthetic_fusion = synthetic_metrics.get("fusion", {"auc": 0.0, "ap": 0.0})
    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "out_dir": out_dir,
        "clip_real_auc": float(real_metrics["clip"]["auc"]),
        "clip_real_ap": float(real_metrics["clip"]["ap"]),
        "clip_syn_auc": float(synthetic_clip["auc"]),
        "clip_syn_ap": float(synthetic_clip["ap"]),
        "fusion_real_auc": float(real_metrics["fusion"]["auc"]),
        "fusion_real_ap": float(real_metrics["fusion"]["ap"]),
        "fusion_syn_auc": float(synthetic_fusion["auc"]),
        "fusion_syn_ap": float(synthetic_fusion["ap"]),
    }


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    rows = [_collect_row(root_dir, experiment_id, experiment_name) for experiment_id, experiment_name in EXPERIMENTS]
    output_path = os.path.join(root_dir, "output", "TRIBRANCH", "rsna_safd_ablation_summary.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved summary to {}".format(output_path))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
