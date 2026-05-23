import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps):
        return self.net(sinusoidal_embedding(timesteps, self.dim))


class DiffusionResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_emb):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.silu(out)
        out = out + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.silu(out)
        return out + residual


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.time_embedding = TimeEmbedding(time_dim)
        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DiffusionResBlock(base_channels, base_channels, time_dim)
        self.down2 = DiffusionResBlock(base_channels, base_channels * 2, time_dim)
        self.down3 = DiffusionResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid1 = DiffusionResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = DiffusionResBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up3 = DiffusionResBlock(base_channels * 8, base_channels * 2, time_dim)
        self.up2 = DiffusionResBlock(base_channels * 4, base_channels, time_dim)
        self.up1 = DiffusionResBlock(base_channels * 2, base_channels, time_dim)

        self.output_proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        time_emb = self.time_embedding(timesteps)

        x0 = self.input_proj(x)
        x1 = self.down1(x0, time_emb)
        x2 = self.down2(self.pool(x1), time_emb)
        x3 = self.down3(self.pool(x2), time_emb)

        mid = self.mid1(self.pool(x3), time_emb)
        mid = self.mid2(mid, time_emb)

        up = F.interpolate(mid, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        up = torch.cat([up, x3], dim=1)
        up = self.up3(up, time_emb)

        up = F.interpolate(up, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        up = torch.cat([up, x2], dim=1)
        up = self.up2(up, time_emb)

        up = F.interpolate(up, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up = torch.cat([up, x1], dim=1)
        up = self.up1(up, time_emb)
        return self.output_proj(up)


class GaussianDiffusionReconstructor(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=64,
        base_channels=32,
        time_dim=128,
        num_steps=1000,
        beta_start=1.0e-4,
        beta_end=2.0e-2,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.num_steps = int(num_steps)
        self.network = DiffusionUNet(in_channels=in_channels, base_channels=base_channels, time_dim=time_dim)

        betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, x, timesteps):
        return self.network(x, timesteps)

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.num_steps, (batch_size,), device=device)

    def q_sample(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def training_loss(self, x_start):
        timesteps = self.sample_timesteps(x_start.size(0), x_start.device)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise=noise)
        predicted_noise = self.forward(noisy, timesteps)
        return F.mse_loss(predicted_noise, noise)

    def predict_x0(self, x_t, timesteps, predicted_noise):
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        x0 = (x_t - sqrt_one_minus * predicted_noise) / (sqrt_alpha + 1.0e-6)
        return x0.clamp(-1.0, 1.0)

    def ddim_step(self, x_t, timestep, next_timestep):
        timestep_tensor = torch.full((x_t.size(0),), int(timestep), device=x_t.device, dtype=torch.long)
        predicted_noise = self.forward(x_t, timestep_tensor)
        x0 = self.predict_x0(x_t, timestep_tensor, predicted_noise)

        if next_timestep < 0:
            return x0

        alpha_next = self.alphas_cumprod[next_timestep]
        alpha_next = alpha_next.view(1, 1, 1, 1)
        direction = torch.sqrt(1.0 - alpha_next) * predicted_noise
        return torch.sqrt(alpha_next) * x0 + direction

    @torch.no_grad()
    def reconstruct(self, x, t_recon=200, ddim_steps=50):
        t_recon = int(max(1, min(self.num_steps - 1, t_recon)))
        ddim_steps = int(max(2, ddim_steps))

        timestep_tensor = torch.full((x.size(0),), t_recon, device=x.device, dtype=torch.long)
        noisy = self.q_sample(x, timestep_tensor, noise=torch.randn_like(x))

        timesteps = torch.linspace(float(t_recon), 0.0, steps=ddim_steps, device=x.device).round().long()
        unique_steps = []
        seen = set()
        for step in timesteps.tolist():
            if step not in seen:
                unique_steps.append(step)
                seen.add(step)
        if unique_steps[-1] != 0:
            unique_steps.append(0)

        current = noisy
        for index, step in enumerate(unique_steps):
            next_step = unique_steps[index + 1] if index + 1 < len(unique_steps) else -1
            current = self.ddim_step(current, step, next_step)
        return current.clamp(-1.0, 1.0)
