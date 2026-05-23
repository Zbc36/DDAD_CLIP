import argparse
import csv
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .semantic_band import SemanticBandModel
from .utils import ensure_supported_band_config, resolve_dataset_roots


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def summarize_values(values):
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}

    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / count
        std = variance ** 0.5
    else:
        std = 0.0

    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def tensor_to_uint8_image(tensor, clamp_range=(-1.0, 1.0)):
    tensor = tensor.detach().cpu().float()
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    low, high = clamp_range
    tensor = tensor.clamp(low, high)
    tensor = (tensor - low) / max(high - low, 1e-12)
    array = (tensor * 255.0).round().numpy().astype(np.uint8)
    return Image.fromarray(array, mode="L")


def diff_to_uint8_image(raw_tensor, reconstructed_tensor):
    diff = (reconstructed_tensor.detach().cpu().float() - raw_tensor.detach().cpu().float()).abs()
    if diff.dim() == 3:
        diff = diff.squeeze(0)

    max_diff = diff.max().item()
    if max_diff > 0:
        diff = diff / max_diff
    array = (diff * 255.0).round().numpy().astype(np.uint8)
    return Image.fromarray(array, mode="L")


def save_diagnostic_triptych(save_path, raw_tensor, reconstructed_tensor):
    raw_img = tensor_to_uint8_image(raw_tensor)
    fused_img = tensor_to_uint8_image(reconstructed_tensor)
    diff_img = diff_to_uint8_image(raw_tensor, reconstructed_tensor)

    width, height = raw_img.size
    canvas = Image.new("L", (width * 3, height))
    canvas.paste(raw_img, (0, 0))
    canvas.paste(fused_img, (width, 0))
    canvas.paste(diff_img, (width * 2, 0))
    canvas.save(save_path)


def compute_diagnostic_metrics(img_id, label, raw_tensor, reconstructed_tensor):
    raw_tensor = raw_tensor.detach().cpu().float()
    reconstructed_tensor = reconstructed_tensor.detach().cpu().float()
    diff = reconstructed_tensor - raw_tensor

    raw_energy = torch.mean(raw_tensor ** 2).item()
    fused_energy = torch.mean(reconstructed_tensor ** 2).item()

    return {
        "img_id": img_id,
        "label": int(label),
        "mse": torch.mean(diff ** 2).item(),
        "mae": torch.mean(diff.abs()).item(),
        "max_abs_diff": diff.abs().max().item(),
        "raw_min": raw_tensor.min().item(),
        "raw_max": raw_tensor.max().item(),
        "fused_min": reconstructed_tensor.min().item(),
        "fused_max": reconstructed_tensor.max().item(),
        "clip_fraction": ((reconstructed_tensor < -1.0) | (reconstructed_tensor > 1.0)).float().mean().item(),
        "energy_ratio": fused_energy / max(raw_energy, 1e-12),
    }


def write_diagnostic_report(split_dir, split, metrics, triptych_count, cfg_summary):
    metrics_path = os.path.join(split_dir, "{}_metrics.csv".format(split))
    summary_path = os.path.join(split_dir, "{}_summary.json".format(split))

    fieldnames = [
        "img_id",
        "label",
        "mse",
        "mae",
        "max_abs_diff",
        "raw_min",
        "raw_max",
        "fused_min",
        "fused_max",
        "clip_fraction",
        "energy_ratio",
    ]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    summary = {
        "split": split,
        "num_samples": len(metrics),
        "triptychs_saved": triptych_count,
        "config": cfg_summary,
        "mse": summarize_values([row["mse"] for row in metrics]),
        "mae": summarize_values([row["mae"] for row in metrics]),
        "max_abs_diff": summarize_values([row["max_abs_diff"] for row in metrics]),
        "clip_fraction": summarize_values([row["clip_fraction"] for row in metrics]),
        "energy_ratio": summarize_values([row["energy_ratio"] for row in metrics]),
        "worst_samples_by_mse": sorted(metrics, key=lambda row: row["mse"], reverse=True)[:10],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def load_split_from_json(data_json_path, split):
    with open(data_json_path, "r") as f:
        data_dict = json.load(f)

    items = []
    if split == "train":
        train_normal = data_dict["train"].get("0", [])
        items.extend({"img_name": name, "label": 0} for name in train_normal)

        unlabeled = data_dict["train"].get("unlabeled", {})
        items.extend({"img_name": name, "label": 0} for name in unlabeled.get("0", []))
        items.extend({"img_name": name, "label": 1} for name in unlabeled.get("1", []))
    elif split == "test":
        items.extend({"img_name": name, "label": 0} for name in data_dict["test"].get("0", []))
        items.extend({"img_name": name, "label": 1} for name in data_dict["test"].get("1", []))
    else:
        raise ValueError("split must be 'train' or 'test'")

    return items


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def save_one_pt(save_path, fused, final_map, coefficients_map, basis, label, img_id):
    torch.save({
        "fused": fused.cpu(),
        "final_map": final_map.cpu(),
        "coefficients_map": coefficients_map.cpu(),
        "basis": basis.cpu(),
        "label": int(label),
        "img_id": img_id
    }, save_path)


def precompute_split(
    raw_root,
    band_root,
    split,
    img_size,
    n_levels,
    patch_size,
    lambda_repulsion,
    fusion_mode,
    overwrite=False,
    diagnostic_root=None,
    diagnostic_limit=24,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_json_path = os.path.join(raw_root, "data.json")
    image_dir = os.path.join(raw_root, "images")
    split_save_dir = os.path.join(band_root, split)
    ensure_dir(split_save_dir)

    diagnostic_split_dir = None
    diagnostic_triptych_dir = None
    diagnostic_metrics = []
    diagnostic_triptych_count = 0
    if diagnostic_root is not None:
        diagnostic_split_dir = os.path.join(diagnostic_root, split)
        diagnostic_triptych_dir = os.path.join(diagnostic_split_dir, "triptychs")
        ensure_dir(diagnostic_triptych_dir)

    items = load_split_from_json(data_json_path, split)
    transform = build_transform(img_size)

    band_model = SemanticBandModel(
        n_levels=n_levels,
        patch_size=patch_size,
        lambda_repulsion=lambda_repulsion,
        fusion_mode=fusion_mode
    ).to(device)
    band_model.eval()

    print("Start precomputing {}, total {} samples".format(split, len(items)))
    with torch.no_grad():
        for item in tqdm(items):
            img_name = item["img_name"]
            label = item["label"]
            img_id = os.path.splitext(img_name)[0]

            img_path = os.path.join(image_dir, img_name)
            save_path = os.path.join(split_save_dir, img_id + ".pt")
            should_write_pt = overwrite or (not os.path.exists(save_path))
            if (not should_write_pt) and diagnostic_split_dir is None:
                continue

            if not os.path.exists(img_path):
                print("[Warning] image not found: {}".format(img_path))
                continue

            image = Image.open(img_path).convert("L")
            x = transform(image).unsqueeze(0).to(device)
            result = band_model(x)
            raw_tensor = x.squeeze(0).cpu()
            reconstructed_tensor = result["final_map"].sum(dim=1).squeeze(0).cpu()
            fused_tensor = result["fused"].squeeze(0).cpu().clamp(-1.0, 1.0)

            if should_write_pt:
                save_one_pt(
                    save_path=save_path,
                    fused=fused_tensor,
                    final_map=result["final_map"].squeeze(0),
                    coefficients_map=result["coefficients_map"].squeeze(0),
                    basis=result["basis"],
                    label=label,
                    img_id=img_id
                )

            if diagnostic_split_dir is not None:
                diagnostic_metrics.append(
                    compute_diagnostic_metrics(
                        img_id=img_id,
                        label=label,
                        raw_tensor=raw_tensor,
                        reconstructed_tensor=reconstructed_tensor,
                    )
                )
                if diagnostic_triptych_count < diagnostic_limit:
                    triptych_path = os.path.join(diagnostic_triptych_dir, img_id + ".png")
                    save_diagnostic_triptych(triptych_path, raw_tensor, reconstructed_tensor)
                    diagnostic_triptych_count += 1

    print("Finished {}, saved to: {}".format(split, split_save_dir))
    if diagnostic_split_dir is not None:
        write_diagnostic_report(
            split_dir=diagnostic_split_dir,
            split=split,
            metrics=diagnostic_metrics,
            triptych_count=diagnostic_triptych_count,
            cfg_summary={
                "img_size": img_size,
                "n_levels": n_levels,
                "patch_size": patch_size,
                "lambda_repulsion": lambda_repulsion,
                "fusion_mode": fusion_mode,
            },
        )
        print("Diagnostic report saved to: {}".format(diagnostic_split_dir))


def precompute_from_config(cfgs, split, overwrite=False, diagnostic_dir=None, diagnostic_limit=24):
    ensure_supported_band_config(cfgs)

    model_cfg = cfgs["Model"]
    if not model_cfg.get("use_semantic_band", False):
        raise ValueError("Model.use_semantic_band must be true when running band precompute.")

    data_cfg = cfgs["Data"]
    raw_root, band_root = resolve_dataset_roots(
        dataset=data_cfg["dataset"],
        data_root=data_cfg.get("data_root"),
        band_root=data_cfg.get("band_root"),
    )

    splits = ["train", "test"] if split == "all" else [split]
    for split_name in splits:
        precompute_split(
            raw_root=raw_root,
            band_root=band_root,
            split=split_name,
            img_size=data_cfg["img_size"],
            n_levels=model_cfg["band_n_levels"],
            patch_size=model_cfg["band_patch_size"],
            lambda_repulsion=model_cfg["band_lambda_repulsion"],
            fusion_mode=model_cfg["band_fusion_mode"],
            overwrite=overwrite,
            diagnostic_root=diagnostic_dir,
            diagnostic_limit=diagnostic_limit,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a band-enabled yaml config.")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--diagnostic-dir",
        default=None,
        help="Optional directory for saving raw/fused/diff triptychs and summary metrics.",
    )
    parser.add_argument(
        "--diagnostic-limit",
        type=int,
        default=24,
        help="How many triptych visualizations to save per split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfgs = yaml.safe_load(f)

    precompute_from_config(
        cfgs,
        split=args.split,
        overwrite=args.overwrite,
        diagnostic_dir=args.diagnostic_dir,
        diagnostic_limit=args.diagnostic_limit,
    )
