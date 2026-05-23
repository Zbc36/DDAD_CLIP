import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNetRefineNetwork(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        base_channels=32,
        fusion_mode="concat",
        fusion_groups=1
    ):
        super().__init__()
        if fusion_mode not in {"concat", "learnable_fusion"}:
            raise ValueError("Unsupported refine fusion mode: {}".format(fusion_mode))

        effective_in_channels = in_channels
        if fusion_mode == "learnable_fusion":
            if fusion_groups <= 0:
                raise ValueError("fusion_groups must be >= 1, got {}".format(fusion_groups))
            if in_channels % fusion_groups != 0:
                raise ValueError(
                    "in_channels must be divisible by fusion_groups, got {} and {}".format(
                        in_channels, fusion_groups
                    )
                )
            self.input_fusion = nn.Conv2d(
                in_channels,
                fusion_groups,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=fusion_groups
            )
            effective_in_channels = fusion_groups
        else:
            self.input_fusion = nn.Identity()

        self.enc1 = _ConvBlock(effective_in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(base_channels * 2, base_channels)
        self.out_proj = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_fusion(x)

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bottleneck)
        if dec2.shape[-2:] != enc2.shape[-2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        if dec1.shape[-2:] != enc1.shape[-2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.out_proj(dec1)
