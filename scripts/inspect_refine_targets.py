import argparse
import json
import math
import os
import sys

import torch
import yaml
from torchvision.utils import save_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import ensure_supported_band_config, get_configured_loader, get_model
from test import (
    build_refine_soft_target,
    get_refine_input_channels,
    get_refine_input_group_count,
    get_refine_method_name,
    get_refine_network_fusion_mode,
    get_refine_network_name,
    resolve_refine_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--refine", choices=["dual", "inter", "intra"], default="dual")
    parser.add_argument(
        "--target-mode",
        choices=["hard", "safd", "avg_blur", "gaussian_blur"],
        default=None,
        help="Optional override for RefineSolver.target_soft_mode.",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def get_refine_in(refine_name):
    if refine_name == "dual":
        return ["inter_dis", "intra_dis"]
    if refine_name == "inter":
        return ["inter_dis"]
    return ["intra_dis"]


def summarize_target_map(target_map, hard_mask):
    positive_mask = hard_mask > 0.5
    negative_mask = ~positive_mask

    positive_mean = target_map[positive_mask].mean().item() if positive_mask.any() else 0.0
    negative_mean = target_map[negative_mask].mean().item() if negative_mask.any() else 0.0
    peak_value = target_map.max().item()
    return {
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "peak_value": peak_value,
    }


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfgs = yaml.safe_load(f)

    if args.target_mode is not None:
        cfgs.setdefault("RefineSolver", {})["target_soft_mode"] = args.target_mode

    ensure_supported_band_config(cfgs)
    refine_in = get_refine_in(args.refine)
    gpu = cfgs["Exp"]["gpu"]
    torch.cuda.set_device(gpu)

    refine_model = get_model(
        network=get_refine_network_name(cfgs),
        in_channels=get_refine_input_channels(cfgs, refine_in),
        out_channels=2,
        refine_fusion_mode=get_refine_network_fusion_mode(cfgs, refine_in),
        refine_fusion_groups=get_refine_input_group_count(cfgs),
        refine_base_channels=int(cfgs.get("RefineSolver", {}).get("base_channels", 32)),
        cfgs=cfgs,
    )

    target_mode = cfgs.get("RefineSolver", {}).get("target_soft_mode", "hard")
    if target_mode == "safd":
        method_name = get_refine_method_name(refine_in, cfgs)
        refine_ckpt = resolve_refine_checkpoint(cfgs["Exp"]["out_dir"], method_name, cfgs)
        refine_model.load_state_dict(torch.load(refine_ckpt, map_location=torch.device("cuda:{}".format(gpu))))
        print("Loaded refine checkpoint:", refine_ckpt)
    refine_model.eval()

    loader = get_configured_loader(
        cfgs,
        dtype="train",
        bs=args.batch_size,
        workers=args.workers,
        self_sup=True,
        force_raw=True,
    )

    output_dir = args.output_dir or os.path.join(
        cfgs["Exp"]["out_dir"],
        "target_inspect_{}".format(target_mode),
    )
    os.makedirs(output_dir, exist_ok=True)

    raw_masks = []
    soft_masks = []
    stats = []
    with torch.no_grad():
        for _, (_, _, anomaly_mask) in enumerate(loader):
            anomaly_mask = anomaly_mask.cuda().float()
            soft_target = build_refine_soft_target(refine_model, anomaly_mask, cfgs)[:, 1:2]

            batch_size = anomaly_mask.size(0)
            for idx in range(batch_size):
                raw_mask = anomaly_mask[idx:idx + 1]
                soft_mask = soft_target[idx:idx + 1]
                raw_masks.append(raw_mask.cpu())
                soft_masks.append(soft_mask.cpu())
                stats.append(summarize_target_map(soft_mask, raw_mask))
                if len(raw_masks) >= args.num_samples:
                    break
            if len(raw_masks) >= args.num_samples:
                break

    if len(raw_masks) == 0:
        raise RuntimeError("No masks were generated for inspection.")

    raw_grid = torch.cat(raw_masks, dim=0)
    soft_grid = torch.cat(soft_masks, dim=0)
    nrow = max(1, int(math.sqrt(len(raw_masks))))
    save_image(raw_grid, os.path.join(output_dir, "raw_mask_grid.png"), nrow=nrow)
    save_image(soft_grid, os.path.join(output_dir, "soft_target_grid.png"), nrow=nrow)

    summary = {
        "target_mode": target_mode,
        "num_samples": len(stats),
        "positive_mean_avg": sum(item["positive_mean"] for item in stats) / len(stats),
        "negative_mean_avg": sum(item["negative_mean"] for item in stats) / len(stats),
        "peak_value_avg": sum(item["peak_value"] for item in stats) / len(stats),
        "per_sample": stats,
    }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved grids to:", output_dir)


if __name__ == "__main__":
    main()
