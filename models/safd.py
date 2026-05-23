import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def _patchify(x, patch_size):
    batch_size, channels, height, width = x.shape
    patches_h = height // patch_size
    patches_w = width // patch_size
    x = x.view(batch_size, channels, patches_h, patch_size, patches_w, patch_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    return x.view(batch_size * channels * patches_h * patches_w, patch_size * patch_size)


def _unpatchify_components(components, batch_size, channels, patches_h, patches_w, patch_size, n_levels):
    components = components.view(
        batch_size,
        channels,
        patches_h,
        patches_w,
        n_levels,
        patch_size,
        patch_size,
    )
    components = components.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
    return components.view(batch_size, n_levels, channels, patches_h * patch_size, patches_w * patch_size)


def _unpatchify_coefficients(coefficients, batch_size, channels, patches_h, patches_w, n_levels):
    coefficients = coefficients.view(batch_size, channels, patches_h, patches_w, n_levels)
    coefficients = coefficients.permute(0, 4, 1, 2, 3).contiguous()
    return coefficients


def _collapse_coefficients_to_map(coefficients):
    if coefficients.dim() == 5:
        return coefficients.abs().mean(dim=(1, 2)).unsqueeze(1)
    if coefficients.dim() == 4:
        return coefficients.abs().mean(dim=1, keepdim=True)
    raise ValueError("Expected coefficient tensor with rank 4 or 5, got {}".format(coefficients.dim()))


def _flatten_coefficient_bands(coefficients):
    if coefficients.dim() == 5:
        batch_size, n_levels = coefficients.shape[:2]
        return coefficients.abs().reshape(batch_size, n_levels, -1)
    if coefficients.dim() == 4:
        batch_size, n_levels = coefficients.shape[:2]
        return coefficients.abs().reshape(batch_size, n_levels, -1)
    raise ValueError("Expected coefficient tensor with rank 4 or 5, got {}".format(coefficients.dim()))


def pool_coefficient_descriptor(coefficients, topk_ratio=0.25):
    flat = _flatten_coefficient_bands(coefficients)
    mean_abs = flat.mean(dim=-1)
    std_abs = flat.std(dim=-1, unbiased=False)
    ratio = float(topk_ratio)
    k = max(1, int(math.ceil(flat.size(-1) * ratio)))
    topk_abs = torch.topk(flat, k=k, dim=-1).values.mean(dim=-1)
    return torch.cat([mean_abs, std_abs, topk_abs], dim=1)


def pool_global_coefficient_descriptor(coefficients, topk_ratio=0.25):
    if coefficients.dim() not in {4, 5}:
        raise ValueError("Expected coefficient tensor with rank 4 or 5, got {}".format(coefficients.dim()))
    flat = coefficients.abs().reshape(coefficients.size(0), -1)
    mean_abs = flat.mean(dim=-1, keepdim=True)
    std_abs = flat.std(dim=-1, unbiased=False, keepdim=True)
    ratio = float(topk_ratio)
    k = max(1, int(math.ceil(flat.size(-1) * ratio)))
    topk_abs = torch.topk(flat, k=k, dim=-1).values.mean(dim=-1, keepdim=True)
    return torch.cat([mean_abs, std_abs, topk_abs], dim=1)


def build_coefficient_heatmap(coefficients_map, output_size=None):
    heatmap = _collapse_coefficients_to_map(coefficients_map)
    if output_size is None:
        return heatmap
    return F.interpolate(heatmap, size=output_size, mode="bilinear", align_corners=False)


class SemanticBandDecomposer(nn.Module):
    """
    Learnable semantic band decomposition with QR-orthogonalized complex bases.

    This implementation follows the requested full-patch formulation: all patches are
    processed in one pass, and the module returns band maps, coefficient maps, and the
    orthogonal basis used for projection.
    """

    def __init__(self, n_levels, patch_size=16, lambda_repulsion=1.0e-8):
        super().__init__()
        if n_levels <= 0:
            raise ValueError("n_levels must be >= 1, got {}".format(n_levels))
        if patch_size <= 0:
            raise ValueError("patch_size must be >= 1, got {}".format(patch_size))

        self.n_levels = int(n_levels)
        self.patch_size = int(patch_size)
        self.patch_dim = self.patch_size * self.patch_size
        self.lambda_repulsion = float(lambda_repulsion)

        if self.n_levels > self.patch_dim:
            raise ValueError(
                "n_levels must be <= patch_size^2, got {} and {}".format(self.n_levels, self.patch_dim)
            )

        radius = torch.rand(self.n_levels) * 0.9
        theta = torch.rand(self.n_levels) * 2.0 * torch.pi
        initial_an = radius * torch.exp(1j * theta)
        self.a_n_real = nn.Parameter(initial_an.real.float())
        self.a_n_imag = nn.Parameter(initial_an.imag.float())

        t = torch.linspace(0, 2.0 * torch.pi, self.patch_dim, requires_grad=False)
        self.register_buffer("z", torch.exp(1j * t).to(torch.complex64))
        self.register_buffer("repulsion_loss", torch.tensor(0.0, dtype=torch.float32))

    def build_basis_matrix(self):
        a_n = torch.complex(self.a_n_real, self.a_n_imag)
        a_n_col = a_n.unsqueeze(1)
        numerator = torch.sqrt(1.0 - torch.abs(a_n) ** 2 + 1.0e-9).unsqueeze(1)
        denominator = 1.0 - torch.conj(a_n_col) * self.z
        return numerator / denominator

    def differentiable_hilbert(self, x_real):
        n = x_real.shape[-1]
        x_fft = torch.fft.fft(x_real, n=n, dim=-1)
        h = torch.zeros(n, device=x_real.device, dtype=x_fft.real.dtype)
        h[0] = 1.0
        if n > 1:
            h[1:n // 2] = 2.0
            h[n // 2] = 1.0 if n % 2 == 0 else 2.0
        x_fft = x_fft * h
        return torch.fft.ifft(x_fft, n=n, dim=-1)

    @torch.no_grad()
    def _constrain_an(self):
        current_an = torch.complex(self.a_n_real.data, self.a_n_imag.data)
        magnitudes = torch.abs(current_an)
        mask = magnitudes >= 0.999
        if torch.any(mask):
            scaled = current_an[mask] / (magnitudes[mask] * 1.001)
            self.a_n_real.data[mask] = scaled.real
            self.a_n_imag.data[mask] = scaled.imag

    def _analyze(self, x):
        if x.dim() != 4:
            raise ValueError("SemanticBandDecomposer expects a 4D tensor, got rank {}".format(x.dim()))

        batch_size, channels, height, width = x.shape
        output_dtype = x.dtype
        autocast_off = nullcontext()
        if x.is_cuda:
            autocast_off = torch.cuda.amp.autocast(enabled=False)

        with autocast_off:
            x = x.float()
            self._constrain_an()

            if self.n_levels > 1 and self.training:
                an_coords = torch.stack([self.a_n_real.float(), self.a_n_imag.float()], dim=1)
                pairwise_distances = torch.pdist(an_coords)
                self.repulsion_loss = self.lambda_repulsion * torch.sum(1.0 / (pairwise_distances + 1.0e-9))
            else:
                self.repulsion_loss = self.repulsion_loss.new_zeros(())

            pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
            x_padded = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else x
            height_pad, width_pad = x_padded.shape[-2:]
            patches_h = height_pad // self.patch_size
            patches_w = width_pad // self.patch_size

            patches_flat = _patchify(x_padded, self.patch_size).float()
            basis_non_orthogonal = self.build_basis_matrix().to(torch.complex64)
            q_matrix, _ = torch.linalg.qr(basis_non_orthogonal.transpose(0, 1))
            basis_orthogonal = q_matrix.transpose(0, 1).to(torch.complex64)

            patches_analytic = self.differentiable_hilbert(patches_flat).to(torch.complex64)
            coefficients_flat = patches_analytic @ basis_orthogonal.transpose(0, 1)

        return {
            "coefficients_flat": coefficients_flat,
            "basis_orthogonal": basis_orthogonal,
            "batch_size": batch_size,
            "channels": channels,
            "height": height,
            "width": width,
            "height_pad": height_pad,
            "width_pad": width_pad,
            "patches_h": patches_h,
            "patches_w": patches_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "output_dtype": output_dtype,
        }

    def coefficient_map(self, x, return_basis=False):
        analysis = self._analyze(x)
        coefficient_map = _unpatchify_coefficients(
            analysis["coefficients_flat"],
            batch_size=analysis["batch_size"],
            channels=analysis["channels"],
            patches_h=analysis["patches_h"],
            patches_w=analysis["patches_w"],
            n_levels=self.n_levels,
        )
        if return_basis:
            return coefficient_map, analysis["basis_orthogonal"]
        return coefficient_map

    def descriptor(self, x, topk_ratio=0.25):
        coefficients = self.coefficient_map(x)
        return pool_coefficient_descriptor(coefficients, topk_ratio=topk_ratio)

    def forward(self, x):
        analysis = self._analyze(x)
        reconstructed_components = analysis["coefficients_flat"].unsqueeze(2) * analysis["basis_orthogonal"].unsqueeze(0)
        band_map_padded = _unpatchify_components(
            reconstructed_components,
            batch_size=analysis["batch_size"],
            channels=analysis["channels"],
            patches_h=analysis["patches_h"],
            patches_w=analysis["patches_w"],
            patch_size=self.patch_size,
            n_levels=self.n_levels,
        )
        coefficient_map = _unpatchify_coefficients(
            analysis["coefficients_flat"],
            batch_size=analysis["batch_size"],
            channels=analysis["channels"],
            patches_h=analysis["patches_h"],
            patches_w=analysis["patches_w"],
            n_levels=self.n_levels,
        )

        if analysis["pad_h"] > 0 or analysis["pad_w"] > 0:
            band_map = band_map_padded[:, :, :, :analysis["height"], :analysis["width"]]
        else:
            band_map = band_map_padded

        return (
            band_map.real.to(analysis["output_dtype"]),
            coefficient_map.real.to(analysis["output_dtype"]),
            analysis["basis_orthogonal"],
        )


def build_safd_prior(coefficients_map, output_size):
    return build_coefficient_heatmap(coefficients_map, output_size=output_size)


def coefficient_difference_map(coeff_a, coeff_b, output_size):
    diff = build_coefficient_heatmap(coeff_a - coeff_b)
    return F.interpolate(diff, size=output_size, mode="bilinear", align_corners=False)


def coefficient_variance_map(coefficients_stack, output_size):
    if coefficients_stack.size(0) <= 1:
        variance = build_coefficient_heatmap(torch.zeros_like(coefficients_stack[0]))
    else:
        magnitude_std = torch.std(coefficients_stack.abs(), dim=0, unbiased=False)
        variance = build_coefficient_heatmap(magnitude_std)
    return F.interpolate(variance, size=output_size, mode="bilinear", align_corners=False)


def min_max_normalize_map(score_map):
    flat = score_map.view(score_map.size(0), -1)
    min_v = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    max_v = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
    return (score_map - min_v) / (max_v - min_v + 1.0e-6)
