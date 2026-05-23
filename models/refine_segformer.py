import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.segformer.configuration_segformer import SegformerConfig
    from transformers.models.segformer.modeling_segformer import SegformerForSemanticSegmentation
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:
    SegformerConfig = None
    SegformerForSemanticSegmentation = None
    _TRANSFORMERS_IMPORT_ERROR = exc


DEFAULT_SEGFORMER_B0_PRETRAINED = "nvidia/segformer-b0-finetuned-ade-512-512"


def _require_transformers():
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "SegFormer refine backbone requires the 'transformers' package. "
            "Install it with `pip install transformers safetensors`."
        ) from _TRANSFORMERS_IMPORT_ERROR


class SegFormerRefineNetwork(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        fusion_mode="concat",
        fusion_groups=1,
        pretrained=True,
        pretrained_name=DEFAULT_SEGFORMER_B0_PRETRAINED
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

        self.input_proj = nn.Conv2d(effective_in_channels, 3, kernel_size=1, stride=1, padding=0)
        self._init_input_proj(effective_in_channels)
        self.segformer = self._build_backbone(
            out_channels=out_channels,
            pretrained=pretrained,
            pretrained_name=pretrained_name
        )

    def _init_input_proj(self, in_channels):
        with torch.no_grad():
            self.input_proj.weight.zero_()
            self.input_proj.bias.zero_()
            fill_value = 1.0 / float(max(1, in_channels))
            self.input_proj.weight[:, :, 0, 0].fill_(fill_value)

    def _build_backbone(self, out_channels, pretrained, pretrained_name):
        _require_transformers()
        if pretrained:
            try:
                return SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_name,
                    num_labels=out_channels,
                    ignore_mismatched_sizes=True
                )
            except OSError as exc:
                raise OSError(
                    "Unable to load SegFormer pretrained weights from '{}'. "
                    "If this machine is offline, either set RefineSolver.pretrained=false "
                    "to train from scratch, or set RefineSolver.pretrained_name to a local "
                    "directory containing the SegFormer config and weight files.".format(pretrained_name)
                ) from exc

        config = SegformerConfig(num_labels=out_channels, num_channels=3)
        return SegformerForSemanticSegmentation(config)

    def forward(self, x):
        x = self.input_fusion(x)
        original_size = x.shape[-2:]
        pixel_values = self.input_proj(x)
        logits = self.segformer(pixel_values=pixel_values).logits
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits
