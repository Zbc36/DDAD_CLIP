import math

import torch
import torch.nn.functional as F


def get_refine_runtime_cfg(cfgs):
    solver_cfg = cfgs.get("RefineSolver", {})
    band_mode = str(solver_cfg.get("band_mode", "fixed_3band")).strip().lower()
    if band_mode not in {"none", "fixed_3band"}:
        raise ValueError("Unsupported RefineSolver.band_mode: {}".format(band_mode))

    score_topk_ratio = solver_cfg.get("score_topk_ratio")
    if score_topk_ratio is None:
        score_topk_ratio = cfgs.get("Ensemble", {}).get("real_score_topk_ratio", 0.01)
    score_topk_ratio = float(score_topk_ratio)
    if not 0.0 < score_topk_ratio <= 1.0:
        raise ValueError("RefineSolver.score_topk_ratio must be in (0, 1], got {}".format(score_topk_ratio))

    use_fixed_3band = band_mode == "fixed_3band"
    return {
        "band_mode": band_mode,
        "use_fixed_3band": use_fixed_3band,
        "bands_per_map": 3 if use_fixed_3band else 1,
        "score_topk_ratio": score_topk_ratio,
    }


def infer_refine_in_channels(refine_in, use_fixed_3band=True):
    base_channels = len(refine_in)
    if base_channels <= 0:
        raise ValueError("refine_in must contain at least one discrepancy map.")
    return base_channels * (3 if use_fixed_3band else 1)


def _gaussian_kernel2d(kernel_size, sigma, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def _gaussian_blur(x, kernel_size, sigma):
    if x.dim() != 4:
        raise ValueError("Expected a rank-4 tensor for blur, got {}".format(tuple(x.shape)))

    kernel = _gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(x.size(1), 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    x_padded = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    return F.conv2d(x_padded, kernel, groups=x.size(1))


def fixed_three_band_decomposition(x):
    small_scale = _gaussian_blur(x, kernel_size=5, sigma=1.0)
    large_scale = _gaussian_blur(x, kernel_size=9, sigma=2.0)
    low_band = large_scale
    mid_band = torch.abs(small_scale - large_scale)
    high_band = torch.abs(x - small_scale)
    return low_band, mid_band, high_band


def build_refine_input(inter_dis, intra_dis, refine_in, use_fixed_3band=True):
    selected_maps = []
    if "inter_dis" in refine_in:
        selected_maps.append(inter_dis)
    if "intra_dis" in refine_in:
        selected_maps.append(intra_dis)
    if len(selected_maps) == 0:
        raise ValueError("refine_in must request at least one discrepancy map.")

    if not use_fixed_3band:
        return torch.cat(selected_maps, dim=1).contiguous()

    band_maps = []
    for discrepancy_map in selected_maps:
        band_maps.extend(fixed_three_band_decomposition(discrepancy_map))
    return torch.cat(band_maps, dim=1).contiguous()


def topk_mean_score(score_map, ratio):
    if score_map.dim() != 4:
        raise ValueError("Expected score_map with rank 4, got {}".format(score_map.dim()))

    flat = score_map.float().view(score_map.size(0), -1)
    k = max(1, int(math.ceil(flat.size(1) * float(ratio))))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)
