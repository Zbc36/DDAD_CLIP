import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _dct_scale(freq, size):
    return math.sqrt(1.0 / size) if freq == 0 else math.sqrt(2.0 / size)


def build_dct_basis(patch_size):
    basis = []
    ordering = []
    for u in range(patch_size):
        for v in range(patch_size):
            patch_basis = torch.empty((patch_size, patch_size), dtype=torch.float32)
            alpha_u = _dct_scale(u, patch_size)
            alpha_v = _dct_scale(v, patch_size)
            for x in range(patch_size):
                for y in range(patch_size):
                    patch_basis[x, y] = (
                        alpha_u
                        * alpha_v
                        * math.cos(math.pi * (2 * x + 1) * u / (2 * patch_size))
                        * math.cos(math.pi * (2 * y + 1) * v / (2 * patch_size))
                    )
            basis.append(patch_basis.reshape(-1))
            ordering.append((u + v, u, v))

    sorted_indices = sorted(range(len(ordering)), key=lambda idx: ordering[idx])
    basis = torch.stack([basis[idx] for idx in sorted_indices], dim=0)
    return basis


class SemanticBandDecomposer(nn.Module):
    def __init__(self, n_levels, patch_size=16, lambda_repulsion=0.0):
        super().__init__()
        self.n_levels = n_levels
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size
        self.lambda_repulsion = lambda_repulsion

        if self.n_levels > self.patch_dim:
            raise ValueError(
                "band_n_levels must be <= patch_size^2, got {} and {}".format(self.n_levels, self.patch_dim)
            )

        full_basis = build_dct_basis(self.patch_size)
        self.register_buffer("basis", full_basis[:self.n_levels].contiguous())
        self.repulsion_loss = torch.tensor(0.0)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else x
        height_pad, width_pad = x_padded.shape[-2:]

        patches_flat = rearrange(
            x_padded,
            "b c (h p1) (w p2) -> (b c h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size
        )

        basis = self.basis.to(device=x.device, dtype=x.dtype)
        coefficients = patches_flat @ basis.T
        reconstructed_components = coefficients.unsqueeze(2) * basis.unsqueeze(0)

        band_map_padded = rearrange(
            reconstructed_components,
            "(b c h w) m (p1 p2) -> b (c m) (h p1) (w p2)",
            b=batch_size,
            c=channels,
            h=(height_pad // self.patch_size),
            w=(width_pad // self.patch_size),
            m=self.n_levels,
            p1=self.patch_size,
            p2=self.patch_size
        )

        coefficients_map = rearrange(
            coefficients,
            "(b c h w) m -> b (c m) h w",
            b=batch_size,
            c=channels,
            h=(height_pad // self.patch_size),
            w=(width_pad // self.patch_size),
            m=self.n_levels
        )

        if pad_h > 0 or pad_w > 0:
            band_map = band_map_padded[:, :, :height, :width]
        else:
            band_map = band_map_padded

        band_map = rearrange(band_map, "b (c m) h w -> b m c h w", c=channels, m=self.n_levels)
        coefficients_map = rearrange(coefficients_map, "b (c m) h w -> b m c h w", c=channels, m=self.n_levels)

        self.repulsion_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        return band_map, coefficients_map, basis


class BandFusion(nn.Module):
    def __init__(self, mode="concat"):
        super().__init__()
        assert mode in ["sum", "mean", "concat"]
        self.mode = mode

    def forward(self, band_map):
        if self.mode == "sum":
            return band_map.sum(dim=1)
        if self.mode == "mean":
            return band_map.mean(dim=1)

        batch_size, n_levels, channels, height, width = band_map.shape
        return band_map.reshape(batch_size, n_levels * channels, height, width)


class SemanticBandModel(nn.Module):
    def __init__(self, n_levels=16, patch_size=16, lambda_repulsion=0.0, fusion_mode="concat"):
        super().__init__()
        self.decomposer = SemanticBandDecomposer(
            n_levels=n_levels,
            patch_size=patch_size,
            lambda_repulsion=lambda_repulsion
        )
        self.fusion = BandFusion(mode=fusion_mode)

    def forward(self, x):
        final_map, coefficients_map, basis = self.decomposer(x)
        fused = self.fusion(final_map)
        return {
            "final_map": final_map,
            "coefficients_map": coefficients_map,
            "basis": basis,
            "fused": fused,
            "repulsion_loss": self.decomposer.repulsion_loss,
        }
