import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_prompt_sets(prompt_ensemble, normal_prompts=None, abnormal_prompts=None):
    if normal_prompts is not None and abnormal_prompts is not None:
        return tuple(normal_prompts), tuple(abnormal_prompts)

    ensemble_name = str(prompt_ensemble or "rsna_chest_xray").lower()
    if ensemble_name in {"rsna_chest_xray", "vin_chest_xray"}:
        return (
            (
                "a normal chest x-ray",
                "a clear frontal chest radiograph",
                "a chest radiograph without focal opacity",
                "a healthy chest x-ray",
                "a chest x-ray without abnormal findings",
                "a normal chest radiograph with clear lungs",
            ),
            (
                "an abnormal chest x-ray",
                "a chest x-ray with disease findings",
                "a chest radiograph with focal opacity",
                "a chest x-ray with pulmonary infiltrates",
                "a chest x-ray with lung opacity",
                "a chest radiograph showing abnormal airspace disease",
            ),
        )
    if ensemble_name == "brain_mri":
        return (
            (
                "a normal brain mri",
                "a healthy brain mri scan",
                "a brain mri without tumor",
                "a brain mri without lesion",
                "a brain mri without abnormal findings",
                "a normal intracranial mri image",
            ),
            (
                "an abnormal brain mri",
                "a brain mri with tumor",
                "a brain mri with lesion",
                "a brain mri with intracranial mass",
                "a brain mri showing abnormal findings",
                "a brain mri with abnormal tissue",
            ),
        )
    if ensemble_name in {"resc_oct", "retinal_oct", "oct_retina"}:
        return (
            (
                "a normal retinal oct image",
                "a healthy retinal oct scan",
                "a retinal oct without abnormal findings",
                "a macular oct scan without lesion",
                "a normal optical coherence tomography retina image",
                "a retinal oct scan with preserved retinal layers",
            ),
            (
                "an abnormal retinal oct image",
                "a retinal oct scan with lesion",
                "a retinal oct image with pathological findings",
                "a macular oct scan with abnormal retina",
                "an optical coherence tomography retina image with disease",
                "a retinal oct scan with disrupted retinal layers",
            ),
        )
    if ensemble_name == "lag_fundus":
        return (
            (
                "a normal fundus photograph",
                "a healthy retinal fundus image",
                "a fundus image without glaucoma",
                "a normal optic disc photograph",
                "a retinal image without abnormal findings",
                "a normal eye fundus image",
            ),
            (
                "an abnormal fundus photograph",
                "a glaucomatous fundus image",
                "a fundus image with glaucoma",
                "a retinal image with abnormal optic disc",
                "a fundus image with optic nerve damage",
                "a fundus photograph showing abnormal findings",
            ),
        )
    else:
        raise ValueError("Unsupported prompt ensemble: {}".format(prompt_ensemble))


class TokenAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _maybe_use_quick_gelu(pretrained_tag):
    return str(pretrained_tag).lower() == "openai"


def _convert_attention_mask_to_bool(mask):
    if mask is None or not torch.is_tensor(mask) or mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point():
        if torch.any(mask < 0):
            return mask < 0
        return mask != 0
    return mask != 0


def _patched_open_clip_attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
    k_x = k_x if k_x is not None else q_x
    v_x = v_x if v_x is not None else q_x

    if attn_mask is not None:
        attn_mask = _convert_attention_mask_to_bool(attn_mask)
        if torch.is_tensor(attn_mask) and attn_mask.dtype != torch.bool and attn_mask.dtype != q_x.dtype:
            attn_mask = attn_mask.to(q_x.dtype)

    return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]


def _patch_torch_multihead_attention_globally():
    if getattr(nn.MultiheadAttention, "_codex_global_bool_mask_patch", False):
        return

    original_forward = nn.MultiheadAttention.forward

    def patched_forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if attn_mask is not None:
            attn_mask = _convert_attention_mask_to_bool(attn_mask)
        if key_padding_mask is not None:
            key_padding_mask = _convert_attention_mask_to_bool(key_padding_mask)
        return original_forward(
            self,
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    nn.MultiheadAttention.forward = patched_forward
    nn.MultiheadAttention._codex_global_bool_mask_patch = True


def _patch_open_clip_attention_globally(open_clip_module):
    transformer_module = getattr(open_clip_module, "transformer", None)
    if transformer_module is None:
        return

    block_cls = getattr(transformer_module, "ResidualAttentionBlock", None)
    if block_cls is None or getattr(block_cls, "_codex_global_bool_attn_patch", False):
        return

    block_cls.attention = _patched_open_clip_attention
    block_cls._codex_global_bool_attn_patch = True


def _normalize_open_clip_attention_masks(module):
    if hasattr(module, "attn_mask"):
        mask = getattr(module, "attn_mask")
        if torch.is_tensor(mask):
            module.attn_mask = _convert_attention_mask_to_bool(mask)

    transformer = getattr(module, "transformer", None)
    if transformer is not None and hasattr(transformer, "attn_mask"):
        mask = getattr(transformer, "attn_mask")
        if torch.is_tensor(mask):
            transformer.attn_mask = _convert_attention_mask_to_bool(mask)

    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is not None:
        for block in resblocks:
            if hasattr(block, "attn_mask") and torch.is_tensor(block.attn_mask):
                block.attn_mask = _convert_attention_mask_to_bool(block.attn_mask)


class BiomedCLIPBranch(nn.Module):
    def __init__(
        self,
        image_size=336,
        backbone_name="ViT-L-14-336",
        pretrained="openai",
        feature_layers=None,
        prompt_token_count=4,
        seg_adapter_dim=512,
        det_adapter_dim=512,
        prompt_ensemble="rsna_chest_xray",
        normal_prompts=None,
        abnormal_prompts=None,
        use_dap_pooling=False,
        dap_alpha=0.5,
        dap_detach_map=True,
        dap_temperature=1.0,
        offline_pretrained=False,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.backbone_name = str(backbone_name)
        self.pretrained = str(pretrained)
        self.use_dap_pooling = bool(use_dap_pooling)
        self.dap_alpha = float(dap_alpha)
        self.dap_detach_map = bool(dap_detach_map)
        self.dap_temperature = max(1.0e-6, float(dap_temperature))
        self.offline_pretrained = bool(offline_pretrained)

        if self.offline_pretrained:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for the MVFA-style CLIP branch. Please install open_clip_torch."
            ) from exc

        _patch_torch_multihead_attention_globally()
        _patch_open_clip_attention_globally(open_clip)

        try:
            create_kwargs = dict(
                model_name=self.backbone_name,
                pretrained=self.pretrained,
            )
            if _maybe_use_quick_gelu(self.pretrained):
                create_kwargs["force_quick_gelu"] = True
            clip_model, _, _ = open_clip.create_model_and_transforms(**create_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize CLIP backbone '{}' with pretrained='{}'. "
                "Please make sure the weights are available before training. "
                "If this is the first run on this machine, set CLIP.offline_pretrained=false once "
                "to allow downloading, then set it back to true for offline runs.".format(
                    self.backbone_name,
                    self.pretrained,
                )
            ) from exc

        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        _normalize_open_clip_attention_masks(clip_model)
        _normalize_open_clip_attention_masks(clip_model.visual)

        self.clip_model = clip_model
        self.visual = clip_model.visual
        self.tokenizer = open_clip.get_tokenizer(self.backbone_name)

        if not hasattr(self.visual, "transformer") or not hasattr(self.visual.transformer, "resblocks"):
            raise RuntimeError(
                "The CLIP visual encoder for '{}' does not expose transformer.resblocks required "
                "for MVFA-style multi-level feature extraction.".format(self.backbone_name)
            )

        total_layers = len(self.visual.transformer.resblocks)
        self.feature_layers = tuple(feature_layers or [6, 12, 18, 24])
        if len(self.feature_layers) == 0:
            raise ValueError("feature_layers must not be empty.")
        for layer_idx in self.feature_layers:
            if int(layer_idx) < 1 or int(layer_idx) > total_layers:
                raise ValueError(
                    "feature layer {} is out of range for backbone '{}' with {} blocks.".format(
                        layer_idx,
                        self.backbone_name,
                        total_layers,
                    )
                )
        self.feature_layers = tuple(int(layer_idx) for layer_idx in self.feature_layers)

        self.transformer_batch_first = bool(getattr(self.visual.transformer, "batch_first", False))
        self.visual_width = self._infer_visual_width()

        self.normal_prompts, self.abnormal_prompts = _default_prompt_sets(
            prompt_ensemble,
            normal_prompts=normal_prompts,
            abnormal_prompts=abnormal_prompts,
        )
        base_normal, base_abnormal = self._encode_prompt_sets()
        self.embed_dim = int(base_normal.size(-1))

        self.seg_adapters = nn.ModuleList([
            TokenAdapter(self.visual_width, int(seg_adapter_dim), self.embed_dim)
            for _ in self.feature_layers
        ])
        self.det_adapters = nn.ModuleList([
            TokenAdapter(self.visual_width, int(det_adapter_dim), self.embed_dim)
            for _ in self.feature_layers
        ])

        initial_logit_scale = torch.tensor(1.0)
        if hasattr(self.clip_model, "logit_scale"):
            initial_logit_scale = self.clip_model.logit_scale.detach().float().clone()
        self.logit_scale = nn.Parameter(initial_logit_scale)
        self.normal_prompt_tokens = nn.Parameter(torch.zeros(int(prompt_token_count), self.embed_dim))
        self.abnormal_prompt_tokens = nn.Parameter(torch.zeros(int(prompt_token_count), self.embed_dim))

        self.register_buffer("normal_base_embeddings", base_normal, persistent=True)
        self.register_buffer("abnormal_base_embeddings", base_abnormal, persistent=True)
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _infer_visual_width(self):
        if hasattr(self.visual, "class_embedding"):
            class_embedding = self.visual.class_embedding
            if class_embedding.dim() == 1:
                return int(class_embedding.shape[0])
            return int(class_embedding.shape[-1])
        if hasattr(self.visual, "conv1"):
            return int(self.visual.conv1.weight.shape[0])
        raise RuntimeError("Unable to infer visual token width from CLIP visual encoder.")

    def _encode_text(self, prompts):
        tokenized = self.tokenizer(list(prompts))
        with torch.no_grad():
            embeddings = self.clip_model.encode_text(tokenized)
        return F.normalize(embeddings.float(), dim=-1)

    def _encode_prompt_sets(self):
        base_normal = self._encode_text(self.normal_prompts)
        base_abnormal = self._encode_text(self.abnormal_prompts)
        return base_normal, base_abnormal

    def _prompt_embeddings(self, device, dtype):
        normal_prompt = self.normal_base_embeddings.to(device=device, dtype=dtype)
        abnormal_prompt = self.abnormal_base_embeddings.to(device=device, dtype=dtype)

        normal_prompt = normal_prompt + self.normal_prompt_tokens.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)
        abnormal_prompt = abnormal_prompt + self.abnormal_prompt_tokens.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)

        normal_prompt = F.normalize(normal_prompt, dim=-1)
        abnormal_prompt = F.normalize(abnormal_prompt, dim=-1)
        return normal_prompt, abnormal_prompt

    def _prepare_image(self, image):
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError("Expected CLIP image tensor with shape [B, 3, H, W], got {}".format(tuple(image.shape)))
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                "Expected CLIP image size ({0}, {0}), got {1}. Resize the input in runtime before calling the model."
                .format(self.image_size, tuple(image.shape[-2:]))
            )

        image = image.float()
        image = (image + 1.0) / 2.0
        image = (image - self.clip_mean) / self.clip_std
        weight_dtype = self.visual.conv1.weight.dtype if hasattr(self.visual, "conv1") else image.dtype
        return image.to(dtype=weight_dtype)

    def _forward_visual_intermediates(self, image):
        x = self._prepare_image(image)
        if not hasattr(self.visual, "conv1"):
            raise RuntimeError("The selected CLIP backbone does not expose the expected ViT conv1 patch embed layer.")

        x = self.visual.conv1(x)
        grid_h, grid_w = int(x.shape[-2]), int(x.shape[-1])
        x = x.reshape(x.shape[0], x.shape[1], grid_h * grid_w).permute(0, 2, 1)

        class_embedding = self.visual.class_embedding.to(dtype=x.dtype)
        class_embedding = class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([class_embedding, x], dim=1)

        positional_embedding = self.visual.positional_embedding.to(dtype=x.dtype)
        if positional_embedding.shape[0] != x.shape[1]:
            raise RuntimeError(
                "Positional embedding length {} does not match token count {} for image size {}.".format(
                    positional_embedding.shape[0],
                    x.shape[1],
                    self.image_size,
                )
            )
        x = x + positional_embedding.unsqueeze(0)

        if hasattr(self.visual, "patch_dropout") and self.visual.patch_dropout is not None:
            x = self.visual.patch_dropout(x)
        if hasattr(self.visual, "ln_pre") and self.visual.ln_pre is not None:
            x = self.visual.ln_pre(x)

        if self.transformer_batch_first:
            sequence = x
        else:
            sequence = x.permute(1, 0, 2)

        intermediates = []
        for block_index, block in enumerate(self.visual.transformer.resblocks, start=1):
            sequence = block(sequence)
            if block_index in self.feature_layers:
                if self.transformer_batch_first:
                    intermediates.append(sequence)
                else:
                    intermediates.append(sequence.permute(1, 0, 2))

        return intermediates, (grid_h, grid_w)

    def _layer_patch_logits(self, patch_tokens, prompt_embeddings):
        similarities = torch.einsum("bnd,pd->bpn", patch_tokens, prompt_embeddings)
        return similarities.mean(dim=1)

    def _layer_global_logits(self, global_tokens, prompt_embeddings):
        similarities = torch.matmul(global_tokens, prompt_embeddings.transpose(0, 1))
        return similarities.mean(dim=1, keepdim=True)

    def _disease_aware_pool(self, patch_tokens, patch_logits):
        if patch_logits.dim() != 3:
            raise ValueError("Expected patch_logits with shape [B, 1, N], got {}".format(tuple(patch_logits.shape)))
        if self.dap_detach_map:
            patch_logits = patch_logits.detach()
        weights = torch.softmax(patch_logits.squeeze(1) / self.dap_temperature, dim=1).unsqueeze(-1)
        weighted_tokens = torch.sum(patch_tokens.float() * weights, dim=1)
        return weighted_tokens

    def forward(self, image, output_size=(128, 128), return_patch_maps=False):
        layer_tokens, patch_grid = self._forward_visual_intermediates(image)
        token_dtype = layer_tokens[0].dtype
        normal_prompt, abnormal_prompt = self._prompt_embeddings(image.device, token_dtype)

        layer_global_logits = []
        layer_patch_maps_native = []
        layer_patch_maps = []

        grid_h, grid_w = patch_grid
        for layer_index, tokens in enumerate(layer_tokens):
            patch_tokens = tokens[:, 1:, :]
            global_tokens = tokens.mean(dim=1)
            patch_embeddings = None
            patch_logits = None
            if return_patch_maps or self.use_dap_pooling:
                patch_embeddings = self.seg_adapters[layer_index](patch_tokens.float())
                patch_embeddings = F.normalize(patch_embeddings, dim=-1)
                normal_patch_logits = self._layer_patch_logits(patch_embeddings, normal_prompt.float())
                abnormal_patch_logits = self._layer_patch_logits(patch_embeddings, abnormal_prompt.float())
                patch_logits = (abnormal_patch_logits - normal_patch_logits).view(tokens.size(0), 1, grid_h, grid_w)

            pooled_global_tokens = global_tokens.float()
            if self.use_dap_pooling and patch_logits is not None:
                dap_tokens = self._disease_aware_pool(
                    patch_tokens,
                    patch_logits.view(tokens.size(0), 1, grid_h * grid_w),
                )
                pooled_global_tokens = (
                    (1.0 - self.dap_alpha) * pooled_global_tokens +
                    self.dap_alpha * dap_tokens.float()
                )

            det_tokens = self.det_adapters[layer_index](pooled_global_tokens)
            det_tokens = F.normalize(det_tokens, dim=-1)

            normal_global_logits = self._layer_global_logits(det_tokens, normal_prompt.float())
            abnormal_global_logits = self._layer_global_logits(det_tokens, abnormal_prompt.float())
            global_logit = self.logit_scale.exp().float() * (abnormal_global_logits - normal_global_logits)
            layer_global_logits.append(global_logit.float())
            if return_patch_maps and patch_logits is not None:
                layer_patch_maps_native.append(patch_logits.float())
                layer_patch_maps.append(
                    F.interpolate(patch_logits.float(), size=output_size, mode="bilinear", align_corners=False)
                )

        global_logit = torch.stack(layer_global_logits, dim=0).mean(dim=0)
        outputs = {
            "global_logit": global_logit,
            "layer_global_logits": layer_global_logits,
            "feature_layers": self.feature_layers,
        }
        if return_patch_maps:
            patch_map_native = torch.stack(layer_patch_maps_native, dim=0).mean(dim=0)
            patch_map = torch.stack(layer_patch_maps, dim=0).mean(dim=0)
            outputs.update({
                "patch_logits": patch_map_native,
                "patch_map": patch_map,
                "layer_patch_maps": layer_patch_maps,
            })
        return outputs

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self
