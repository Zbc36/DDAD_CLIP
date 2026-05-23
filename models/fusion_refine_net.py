import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SemanticBandDecomposer(nn.Module):
    def __init__(self, n_levels, patch_size=16, lambda_repulsion=1.0e-8):
        super().__init__()
        self.n_levels = int(n_levels)
        self.patch_size = int(patch_size)
        self.patch_dim = self.patch_size * self.patch_size
        self.lambda_repulsion = float(lambda_repulsion)

        radius = torch.rand(self.n_levels) * 0.9
        theta = torch.rand(self.n_levels) * 2.0 * math.pi
        initial_an = radius * torch.exp(1j * theta)
        self.a_n_real = nn.Parameter(initial_an.real)
        self.a_n_imag = nn.Parameter(initial_an.imag)

        t = torch.linspace(0, 2.0 * math.pi, self.patch_dim, requires_grad=False)
        self.register_buffer("z", torch.exp(1j * t).to(torch.complex64))
        self.repulsion_loss = torch.tensor(0.0)

    def build_basis_matrix(self):
        a_n = torch.complex(self.a_n_real, self.a_n_imag)
        a_n_col = a_n.unsqueeze(1)
        numerator = torch.sqrt(1.0 - torch.abs(a_n) ** 2 + 1.0e-9).unsqueeze(1)
        denominator = 1.0 - torch.conj(a_n_col) * self.z.unsqueeze(0)
        return numerator / denominator

    def differentiable_hilbert(self, x_real):
        n = x_real.shape[-1]
        x_fft = torch.fft.fft(x_real, n=n, dim=-1)
        h = torch.zeros(n, device=x_real.device, dtype=x_fft.dtype)
        if n > 0 and n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        elif n > 0:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0
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

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Expected x with shape [B, C, H, W], got {}".format(tuple(x.shape)))

        batch_size, channels, height, width = x.shape
        self._constrain_an()

        if self.n_levels > 1 and self.training:
            an_coords = torch.stack([self.a_n_real, self.a_n_imag], dim=1)
            pairwise_distances = torch.pdist(an_coords)
            self.repulsion_loss = self.lambda_repulsion * torch.sum(1.0 / (pairwise_distances + 1.0e-9))
        else:
            self.repulsion_loss = x.new_tensor(0.0)

        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x
        padded_h, padded_w = x_padded.shape[-2:]

        patches_flat = rearrange(
            x_padded,
            "b c (h p1) (w p2) -> (b c h w) (p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        basis_non_orthogonal = self.build_basis_matrix()
        q_basis, _ = torch.linalg.qr(basis_non_orthogonal.transpose(0, 1))
        basis_orthogonal = q_basis.transpose(0, 1)

        patches_analytic = self.differentiable_hilbert(patches_flat.float())
        coefficients = patches_analytic @ basis_orthogonal.transpose(0, 1)
        reconstructed = coefficients.unsqueeze(2) * basis_orthogonal.unsqueeze(0)

        band_maps_padded = rearrange(
            reconstructed,
            "(b c h w) m (p1 p2) -> b m c (h p1) (w p2)",
            b=batch_size,
            c=channels,
            h=(padded_h // self.patch_size),
            w=(padded_w // self.patch_size),
            m=self.n_levels,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        coeff_maps = rearrange(
            coefficients,
            "(b c h w) m -> b m c h w",
            b=batch_size,
            c=channels,
            h=(padded_h // self.patch_size),
            w=(padded_w // self.patch_size),
            m=self.n_levels,
        )

        band_maps = band_maps_padded[:, :, :, :height, :width]
        return band_maps.real, coeff_maps.real, basis_orthogonal


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FusionRefineNet(nn.Module):
    def __init__(
        self,
        in_channels=128,
        out_channels=1,
        base_channels=128,
        aux_in_channels=0,
        dropout=0.1,
        band_mode="none",
        safd_levels=3,
        safd_patch_size=4,
        safd_lambda_repulsion=1.0e-8,
        safd_topk_ratio=0.01,
        safd_raw_in_channels=4,
        include_clip_score=True,
    ):
        super().__init__()
        del out_channels
        self.aux_in_channels = int(aux_in_channels)
        self.band_mode = str(band_mode).strip().lower()
        self.safd_enabled = self.band_mode == "safd"
        self.safd_levels = int(safd_levels)
        self.safd_topk_ratio = float(safd_topk_ratio)
        self.safd_raw_in_channels = int(safd_raw_in_channels)
        self.include_clip_score = bool(include_clip_score)

        if self.safd_enabled:
            self.band_decomposer = SemanticBandDecomposer(
                n_levels=self.safd_levels,
                patch_size=int(safd_patch_size),
                lambda_repulsion=float(safd_lambda_repulsion),
            )
            effective_in_channels = self.safd_raw_in_channels * self.safd_levels
            if self.include_clip_score:
                effective_in_channels += 1
        else:
            self.band_decomposer = None
            effective_in_channels = int(in_channels)

        self.expected_vector_dim = int(effective_in_channels)
        fusion_dim = self.expected_vector_dim + self.aux_in_channels
        hidden_dim = max(int(base_channels), fusion_dim)
        mid_dim = max(hidden_dim // 2, 32)
        self.fusion = nn.Sequential(
            MLPBlock(fusion_dim, hidden_dim, dropout=dropout),
            MLPBlock(hidden_dim, mid_dim, dropout=dropout),
            nn.Linear(mid_dim, 1),
        )
        final_linear = self.fusion[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _topk_mean_score(self, score_map, ratio):
        if score_map.dim() != 4:
            raise ValueError("Expected score_map with shape [B, C, H, W], got {}".format(tuple(score_map.shape)))
        flat = score_map.float().view(score_map.size(0), -1)
        k = max(1, int(math.ceil(flat.size(1) * float(ratio))))
        topk_values = torch.topk(flat, k=k, dim=1).values
        return topk_values.mean(dim=1)

    def _build_safd_vector(self, raw_maps, clip_global_logit=None):
        if raw_maps.dim() != 4:
            raise ValueError("Expected raw_maps with shape [B, C, H, W], got {}".format(tuple(raw_maps.shape)))
        if raw_maps.size(1) != self.safd_raw_in_channels:
            raise ValueError(
                "Expected raw_maps with {} channels, got {}".format(self.safd_raw_in_channels, raw_maps.size(1))
            )

        if self.include_clip_score:
            if clip_global_logit is None:
                raise ValueError("FusionRefineNet requires clip_global_logit when include_clip_score=True.")
            if clip_global_logit.dim() == 1:
                clip_global_logit = clip_global_logit.unsqueeze(1)
            clip_global_logit = clip_global_logit.float()

        band_maps, coeff_maps, basis = self.band_decomposer(raw_maps.float())
        score_maps = band_maps.permute(0, 2, 1, 3, 4).contiguous()
        batch_size, channels, levels, height, width = score_maps.shape
        flattened_maps = score_maps.view(batch_size * channels * levels, 1, height, width)
        band_scores = self._topk_mean_score(flattened_maps, self.safd_topk_ratio).view(batch_size, channels * levels)
        fusion_parts = [band_scores]
        if self.include_clip_score:
            fusion_parts.insert(0, clip_global_logit)
        fusion_vector = torch.cat(fusion_parts, dim=1)
        return fusion_vector, band_scores, band_maps, coeff_maps, basis

    def forward(self, fusion_vector=None, image_aux=None, clip_anchor=None, raw_maps=None, clip_global_logit=None):
        safd_repulsion_loss = None
        safd_band_scores = None
        safd_band_maps = None
        safd_coeff_maps = None
        safd_basis = None

        if self.safd_enabled:
            if raw_maps is None:
                raise ValueError("FusionRefineNet requires raw_maps when band_mode='safd'.")
            if self.include_clip_score and clip_global_logit is None:
                raise ValueError("FusionRefineNet requires clip_global_logit when band_mode='safd'.")
            fusion_vector, safd_band_scores, safd_band_maps, safd_coeff_maps, safd_basis = self._build_safd_vector(
                raw_maps,
                clip_global_logit,
            )
            safd_repulsion_loss = self.band_decomposer.repulsion_loss
        else:
            if fusion_vector is None:
                raise ValueError("FusionRefineNet requires fusion_vector when band_mode!='safd'.")
            if fusion_vector.dim() != 2:
                raise ValueError("Expected fusion_vector with shape [B, D], got {}".format(tuple(fusion_vector.shape)))

        if fusion_vector.size(1) != self.expected_vector_dim:
            raise ValueError(
                "Expected fusion_vector with {} channels, got {}".format(
                    self.expected_vector_dim,
                    fusion_vector.size(1),
                )
            )

        fused = fusion_vector
        if image_aux is None:
            image_aux = torch.zeros(
                fusion_vector.size(0),
                self.aux_in_channels,
                device=fusion_vector.device,
                dtype=fusion_vector.dtype,
            )
        if image_aux.dim() == 1:
            image_aux = image_aux.unsqueeze(0)
        if image_aux.size(1) != self.aux_in_channels:
            raise ValueError(
                "Expected image_aux with {} channels, got {}".format(self.aux_in_channels, image_aux.size(1))
            )
        if self.aux_in_channels > 0:
            fused = torch.cat([fused, image_aux.to(dtype=fused.dtype)], dim=1)

        residual_logit = self.fusion(fused)
        if clip_anchor is None:
            clip_anchor = torch.zeros_like(residual_logit)
        elif clip_anchor.dim() == 1:
            clip_anchor = clip_anchor.unsqueeze(1)
        clip_anchor = clip_anchor.to(dtype=residual_logit.dtype)

        if safd_repulsion_loss is None:
            safd_repulsion_loss = residual_logit.new_tensor(0.0)

        return {
            "image_logit": clip_anchor + residual_logit,
            "residual_logit": residual_logit,
            "fusion_vector": fusion_vector,
            "safd_repulsion_loss": safd_repulsion_loss,
            "safd_band_scores": safd_band_scores,
            "safd_band_maps": safd_band_maps,
            "safd_coeff_maps": safd_coeff_maps,
            "safd_basis": safd_basis,
        }
