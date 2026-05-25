import json
import os
import time
import copy
import random
from collections import deque
from contextlib import nullcontext

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

from anomaly_data import CachedFusionDataset, MultiBranchDataset, PseudoLabelDataset
from models import AverageMeter, EntropyLossEncap, FocalLoss, get_model, load_ab
from models.clip_branch import BiomedCLIPBranch
from models.diffusion_unet import GaussianDiffusionReconstructor
from models.fusion_refine_net import FusionRefineNet, SemanticBandDecomposer
from refine_processing import (
    build_refine_input,
    fixed_three_band_decomposition,
    get_refine_runtime_cfg,
    infer_refine_in_channels,
)

matplotlib.use("Agg")


def _get_device(cfgs):
    gpu_ids = _get_gpu_ids(cfgs)
    if torch.cuda.is_available() and len(gpu_ids) > 0:
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def _get_gpu_ids(cfgs):
    exp_cfg = cfgs.get("Exp", {})
    preferred_gpu = int(exp_cfg.get("gpu", 0))
    gpu_ids = exp_cfg.get("gpu_ids")
    if gpu_ids is None:
        gpu_ids = [preferred_gpu]
    elif isinstance(gpu_ids, str):
        gpu_ids = [int(item.strip()) for item in gpu_ids.split(",") if item.strip() != ""]
    else:
        gpu_ids = [int(item) for item in gpu_ids]

    if not torch.cuda.is_available():
        return []

    available = torch.cuda.device_count()
    filtered = [gpu_id for gpu_id in gpu_ids if 0 <= gpu_id < available]
    if len(filtered) == 0:
        if 0 <= preferred_gpu < available:
            filtered = [preferred_gpu]
    elif preferred_gpu in filtered:
        filtered = [preferred_gpu] + [gpu_id for gpu_id in filtered if gpu_id != preferred_gpu]
    return filtered


def _dataset_root(dataset_name):
    dataset_key = str(dataset_name or "").lower()
    root_name = {
        "rsna": "RSNA",
        "vin": "VinCXR",
        "brain": "BrainTumor",
        "brainmri": "BrainMRI",
        "lag": "LAG",
        "resc": "RESC",
    }.get(dataset_key)
    if root_name is None:
        raise ValueError("Unsupported dataset for the multi-branch pipeline: {}".format(dataset_name))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", root_name)


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _fusion_cache_root(cfgs):
    return _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "fusion_cache"))


def _get_section(cfgs, section_name):
    return cfgs.get(section_name, {})


def _checkpoint_out_dir(cfgs):
    exp_cfg = cfgs.get("Exp", {})
    return exp_cfg.get("checkpoint_out_dir", exp_cfg["out_dir"])


def _get_ensemble_cfg(cfgs):
    ensemble_cfg = cfgs.get("Ensemble", {})
    return {
        "target_count": int(ensemble_cfg.get("target_count", 5)),
        "base_seed": int(ensemble_cfg.get("base_seed", 3407)),
        "real_score_topk_ratio": float(ensemble_cfg.get("real_score_topk_ratio", 0.01)),
    }


def _get_data_cfg(cfgs):
    data_cfg = cfgs.get("Data", {})
    if str(data_cfg.get("dataset", "rsna")).lower() not in {"rsna", "vin", "brain", "brainmri", "lag", "resc"}:
        raise ValueError("Unsupported dataset for the multi-branch pipeline: {}".format(data_cfg.get("dataset")))
    return data_cfg


def _build_multibranch_loader(
    cfgs,
    subset,
    batch_size,
    shuffle,
    synthetic_probability,
    drop_last=False,
    include_clip=True,
    num_workers_override=None,
    persistent_workers_override=None,
    prefetch_factor_override=None,
    cache_images_override=None,
    cache_clip_images_override=None,
    force_deterministic=False,
    deterministic_seed=1337,
    dataset_kwargs=None,
):
    data_cfg = _get_data_cfg(cfgs)
    num_workers = int(
        data_cfg.get("workers", 0) if num_workers_override is None else num_workers_override
    )
    pin_memory = bool(data_cfg.get("pin_memory", torch.cuda.is_available()))
    dataset = MultiBranchDataset(
        main_path=_dataset_root(data_cfg.get("dataset", "rsna")),
        subset=subset,
        image_size=int(data_cfg.get("img_size", 64)),
        clip_image_size=int(data_cfg.get("clip_img_size", 336)),
        include_clip=bool(include_clip),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        val_seed=int(data_cfg.get("val_seed", 0)),
        synthetic_probability=float(synthetic_probability),
        synthetic_mode_probs=dict(data_cfg.get("synthetic_mode_probs", {
            "copy_paste": 0.4,
            "intensity_shift": 0.2,
            "blur_or_sharpen": 0.2,
        })),
        synthetic_shape_probs=dict(data_cfg.get("synthetic_shape_probs", {
            "rectangle": 0.15,
            "ellipse": 0.25,
            "blob": 0.25,
            "multi_blob": 0.20,
            "streak": 0.15,
        })),
        cache_images=bool(
            data_cfg.get("cache_images", True) if cache_images_override is None else cache_images_override
        ),
        cache_clip_images=bool(
            data_cfg.get("cache_clip_images", data_cfg.get("cache_images", True))
            if cache_clip_images_override is None
            else cache_clip_images_override
        ),
        force_deterministic=bool(force_deterministic),
        deterministic_seed=int(deterministic_seed),
        **(dataset_kwargs or {}),
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            data_cfg.get("persistent_workers", True)
            if persistent_workers_override is None
            else persistent_workers_override
        )
        loader_kwargs["prefetch_factor"] = int(
            data_cfg.get("prefetch_factor", 2)
            if prefetch_factor_override is None
            else prefetch_factor_override
        )
    return DataLoader(
        **loader_kwargs
    )


def _get_clip_loader_overrides(cfgs):
    data_cfg = _get_data_cfg(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    default_workers = int(data_cfg.get("workers", 0))
    train_workers = int(clip_cfg.get("workers", min(default_workers, 4)))
    val_workers = int(clip_cfg.get("val_workers", min(train_workers, 2)))
    default_prefetch = int(data_cfg.get("prefetch_factor", 2))
    train_prefetch = int(clip_cfg.get("prefetch_factor", min(default_prefetch, 2)))
    val_prefetch = int(clip_cfg.get("val_prefetch_factor", train_prefetch))
    return {
        "train_workers": max(0, train_workers),
        "val_workers": max(0, val_workers),
        "train_persistent_workers": bool(
            clip_cfg.get("persistent_workers", train_workers > 0)
        ),
        "val_persistent_workers": bool(
            clip_cfg.get("val_persistent_workers", False)
        ),
        "train_prefetch_factor": max(1, train_prefetch),
        "val_prefetch_factor": max(1, val_prefetch),
        "cache_images": bool(clip_cfg.get("cache_images", data_cfg.get("cache_images", True))),
        "cache_clip_images": bool(clip_cfg.get("cache_clip_images", False)),
    }


def _get_weakclip_cfg(cfgs):
    data_cfg = _get_data_cfg(cfgs)
    weak_cfg = cfgs.get("WeakCLIP", {})
    default_ratio = float(data_cfg.get("val_ratio", 0.1))
    top_ratio = float(weak_cfg.get("pseudo_top_abnormal_ratio", 0.02))
    bottom_ratio = float(weak_cfg.get("pseudo_bottom_normal_ratio", 0.10))
    if not 0.0 < top_ratio < 1.0:
        raise ValueError("WeakCLIP.pseudo_top_abnormal_ratio must be in (0, 1), got {}".format(top_ratio))
    if not 0.0 < bottom_ratio < 1.0:
        raise ValueError("WeakCLIP.pseudo_bottom_normal_ratio must be in (0, 1), got {}".format(bottom_ratio))
    return {
        "group_key": weak_cfg.get("group_key", "auto"),
        "clean_val_ratio": float(weak_cfg.get("clean_val_ratio", default_ratio)),
        "unlabeled_val_ratio": float(weak_cfg.get("unlabeled_val_ratio", default_ratio)),
        "pseudo_top_abnormal_ratio": top_ratio,
        "pseudo_bottom_normal_ratio": bottom_ratio,
        "pseudo_loss_weight": float(weak_cfg.get("pseudo_loss_weight", 1.0)),
        "pseudo_abnormal_min_target": float(weak_cfg.get("pseudo_abnormal_min_target", 0.85)),
        "pseudo_normal_max_target": float(weak_cfg.get("pseudo_normal_max_target", 0.15)),
        "pseudo_abnormal_score_min": float(weak_cfg.get("pseudo_abnormal_score_min", 0.70)),
        "pseudo_normal_score_max": float(weak_cfg.get("pseudo_normal_score_max", 0.20)),
        "ddad_abnormal_percentile_min": float(weak_cfg.get("ddad_abnormal_percentile_min", 0.80)),
        "ddad_normal_percentile_max": float(weak_cfg.get("ddad_normal_percentile_max", 0.20)),
        "student_use_synthetic_cls": bool(weak_cfg.get("student_use_synthetic_cls", False)),
        "student_clean_loss_weight": float(weak_cfg.get("student_clean_loss_weight", 0.5)),
        "student_pseudo_loss_weight": float(weak_cfg.get("student_pseudo_loss_weight", 1.0)),
        "student_seg_syn_weight": float(weak_cfg.get("student_seg_syn_weight", 0.2)),
        "student_pseudo_loc_weight": float(weak_cfg.get("student_pseudo_loc_weight", 0.1)),
        "student_bg_suppression_weight": float(weak_cfg.get("student_bg_suppression_weight", 0.05)),
        "student_fusion_loss_weight": float(weak_cfg.get("student_fusion_loss_weight", 1.0)),
        "student_clip_aux_loss_weight": float(weak_cfg.get("student_clip_aux_loss_weight", 0.3)),
        "student_ddad_map_loss_weight": float(weak_cfg.get("student_ddad_map_loss_weight", 0.05)),
        "student_fusion_hidden_dim": int(weak_cfg.get("student_fusion_hidden_dim", 32)),
        "student_fusion_dropout": float(weak_cfg.get("student_fusion_dropout", 0.1)),
        "student_fusion_init_stage": str(weak_cfg.get("student_fusion_init_stage", "clip_student")).strip().lower(),
        "student_fusion_freeze_clip": bool(weak_cfg.get("student_fusion_freeze_clip", True)),
        "use_refine_score": bool(weak_cfg.get("use_refine_score", False)),
        "refine_checkpoint": str(weak_cfg.get("refine_checkpoint", "weak_refine_dual")).strip(),
        "use_safd_score": bool(weak_cfg.get("use_safd_score", False)),
        "safd_levels": int(weak_cfg.get("safd_levels", 3)),
        "safd_patch_size": int(weak_cfg.get("safd_patch_size", 4)),
        "safd_topk_ratio": float(weak_cfg.get("safd_topk_ratio", 0.01)),
        "safd_lambda_repulsion": float(weak_cfg.get("safd_lambda_repulsion", 1.0e-8)),
        "safd_seed": int(weak_cfg.get("safd_seed", 3407)),
        "safd_score_mode": str(weak_cfg.get("safd_score_mode", "max")).strip().lower(),
        "safd_normal_mad_eps": float(weak_cfg.get("safd_normal_mad_eps", 1.0e-4)),
        "safd_normal_reduce": str(weak_cfg.get("safd_normal_reduce", "max")).strip().lower(),
        "safd_fusion_mode": str(weak_cfg.get("safd_fusion_mode", "linear")).strip().lower(),
        "safd_apply_scope": str(weak_cfg.get("safd_apply_scope", "global")).strip().lower(),
        "topup_abnormal_safd_weight": float(weak_cfg.get("topup_abnormal_safd_weight", 0.25)),
        "topup_abnormal_base_weight": float(weak_cfg.get("topup_abnormal_base_weight", 0.75)),
        "refined_ddad_joint_weight": float(weak_cfg.get("refined_ddad_joint_weight", 0.60)),
        "safd_joint_weight": float(weak_cfg.get("safd_joint_weight", 0.20)),
        "clip_joint_weight": float(weak_cfg.get("clip_joint_weight", 0.15)),
        "localization_joint_weight": float(weak_cfg.get("localization_joint_weight", 0.05)),
        "pseudo_topup_abnormal_min_target": float(weak_cfg.get("pseudo_topup_abnormal_min_target", 0.75)),
        "pseudo_topup_abnormal_weight_scale": float(weak_cfg.get("pseudo_topup_abnormal_weight_scale", 0.5)),
        "pseudo_topup_abnormal_use_safd_gate": bool(weak_cfg.get("pseudo_topup_abnormal_use_safd_gate", False)),
        "teacher_clean_loss_weight": float(weak_cfg.get("teacher_clean_loss_weight", 1.0)),
        "teacher_synthetic_cls_weight": float(weak_cfg.get("teacher_synthetic_cls_weight", 1.0)),
        "teacher_seg_loss_weight": float(weak_cfg.get("teacher_seg_loss_weight", 0.2)),
        "teacher_bg_suppression_weight": float(weak_cfg.get("teacher_bg_suppression_weight", 0.0)),
        "checkpoint_metric": str(weak_cfg.get("checkpoint_metric", "unlabeled_val_hidden")).strip().lower(),
        "save_hidden_label_debug": bool(weak_cfg.get("save_hidden_label_debug", True)),
        "eval_checkpoint": str(weak_cfg.get("eval_checkpoint", "student")).strip().lower(),
    }


def _get_weak_refine_solver_cfg(cfgs):
    solver_cfg = cfgs.get("RefineSolver", {})
    return {
        "bs": int(solver_cfg.get("bs", 64)),
        "lr": float(solver_cfg.get("lr", 5.0e-4)),
        "weight_decay": float(solver_cfg.get("weight_decay", 0.0)),
        "num_epoch": int(solver_cfg.get("num_epoch", 250)),
        "grad_clip": solver_cfg.get("grad_clip", None),
    }


def _get_baseline_protocol_cfg(cfgs):
    data_cfg = _get_data_cfg(cfgs)
    protocol_cfg = cfgs.get("BaselineProtocol", {})
    mode = str(protocol_cfg.get("mode", "legacy")).strip().lower()
    if mode not in {"legacy", "fair"}:
        raise ValueError("BaselineProtocol.mode must be 'legacy' or 'fair', got '{}'".format(mode))
    if mode == "legacy":
        return {
            "mode": mode,
            "synthetic_train_subset": "train_normal",
            "real_train_subset": "real_train",
            "synthetic_val_subset": "synthetic_val",
            "real_val_subset": "real_val",
            "real_eval_subset": "real_test",
            "synthetic_eval_subset": "synthetic_test",
            "ddad_source": str(protocol_cfg.get("ddad_source", "legacy")).strip().lower(),
            "use_weak_dataset_kwargs": False,
            "synthetic_val_seed": 1337,
            "synthetic_eval_seed": 2337,
        }
    default_ratio = float(data_cfg.get("val_ratio", 0.1))
    ddad_source = str(protocol_cfg.get("ddad_source", "weak")).strip().lower()
    if ddad_source not in {"legacy", "weak"}:
        raise ValueError("BaselineProtocol.ddad_source must be 'legacy' or 'weak', got '{}'".format(ddad_source))
    return {
        "mode": mode,
        "synthetic_train_subset": "clean_train_normal",
        "real_train_subset": "unlabeled_train_pool",
        "synthetic_val_subset": "clean_val_normal",
        "real_val_subset": "unlabeled_val_pool",
        "real_eval_subset": "official_test",
        "synthetic_eval_subset": str(protocol_cfg.get("synthetic_eval_subset", "clean_val_normal")).strip(),
        "ddad_source": ddad_source,
        "use_weak_dataset_kwargs": True,
        "clean_val_ratio": float(protocol_cfg.get("clean_val_ratio", default_ratio)),
        "unlabeled_val_ratio": float(protocol_cfg.get("unlabeled_val_ratio", default_ratio)),
        "group_key": protocol_cfg.get("group_key", "auto"),
        "synthetic_val_seed": int(protocol_cfg.get("synthetic_val_seed", 1337)),
        "synthetic_eval_seed": int(protocol_cfg.get("synthetic_eval_seed", 2337)),
    }


def _baseline_dataset_kwargs(cfgs):
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)
    if not protocol_cfg["use_weak_dataset_kwargs"]:
        return None
    return {
        "clean_val_ratio": protocol_cfg["clean_val_ratio"],
        "unlabeled_val_ratio": protocol_cfg["unlabeled_val_ratio"],
        "group_key": protocol_cfg["group_key"],
    }


def _baseline_loader_extra_kwargs(cfgs, subset, split_role):
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)
    kwargs = {}
    if protocol_cfg["use_weak_dataset_kwargs"]:
        kwargs["dataset_kwargs"] = _baseline_dataset_kwargs(cfgs)
    if protocol_cfg["mode"] == "fair" and subset == "clean_val_normal":
        if split_role == "synthetic_val":
            kwargs["force_deterministic"] = True
            kwargs["deterministic_seed"] = protocol_cfg["synthetic_val_seed"]
        elif split_role == "synthetic_eval":
            kwargs["force_deterministic"] = True
            kwargs["deterministic_seed"] = protocol_cfg["synthetic_eval_seed"]
    return kwargs


def _weakclip_dataset_kwargs(cfgs):
    weak_cfg = _get_weakclip_cfg(cfgs)
    return {
        "clean_val_ratio": weak_cfg["clean_val_ratio"],
        "unlabeled_val_ratio": weak_cfg["unlabeled_val_ratio"],
        "group_key": weak_cfg["group_key"],
    }


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weak_ddad_mode_to_dir(mode):
    if mode not in {"weak_a", "weak_b"}:
        raise ValueError("Unsupported weak DDAD mode: {}".format(mode))
    return mode


def _weak_ddad_member_dir(cfgs, mode):
    return _ensure_dir(os.path.join(_checkpoint_out_dir(cfgs), _weak_ddad_mode_to_dir(mode)))


def _weak_ddad_member_paths(cfgs, mode, member_index):
    member_dir = _weak_ddad_member_dir(cfgs, mode)
    return {
        "last": os.path.join(member_dir, "{}_last.pth".format(member_index)),
        "best": os.path.join(member_dir, "{}.pth".format(member_index)),
    }


def _weak_refine_checkpoint_path(cfgs, refine_in=None):
    weak_cfg = _get_weakclip_cfg(cfgs)
    checkpoint_name = weak_cfg["refine_checkpoint"] or (
        "weak_refine_dual" if refine_in is None or len(refine_in) == 2 else "weak_refine_intra"
    )
    if not checkpoint_name.endswith(".pth"):
        checkpoint_name = "{}.pth".format(checkpoint_name)
    return os.path.join(_ensure_dir(os.path.join(_checkpoint_out_dir(cfgs), "refine")), checkpoint_name)


def _sorted_checkpoint_names(model_dir):
    if not os.path.exists(model_dir):
        return []
    return sorted(
        [name for name in os.listdir(model_dir) if name.endswith(".pth") and "_last" not in name],
        key=lambda item: int(os.path.splitext(item)[0]) if os.path.splitext(item)[0].isdigit() else item,
    )


def _load_weak_ddad_ensemble(cfgs, mode, requires_grad=False):
    device = _get_device(cfgs)
    model_cfg = cfgs["Model"]
    data_cfg = _get_data_cfg(cfgs)
    ensemble_cfg = _get_ensemble_cfg(cfgs)
    target_count = ensemble_cfg["target_count"]
    member_dir = _weak_ddad_member_dir(cfgs, mode)
    checkpoint_names = _sorted_checkpoint_names(member_dir)
    if len(checkpoint_names) < target_count:
        raise RuntimeError(
            "Weak DDAD ensemble '{}' incomplete: found {} checkpoints in '{}', need {}.".format(
                mode,
                len(checkpoint_names),
                member_dir,
                target_count,
            )
        )

    models = []
    for checkpoint_name in checkpoint_names[:target_count]:
        model = get_model(
            network=model_cfg["network"],
            mp=model_cfg["mp"],
            ls=model_cfg["ls"],
            img_size=int(data_cfg.get("img_size", 64)),
            mem_dim=model_cfg["mem_dim"],
            shrink_thres=model_cfg["shrink_thres"],
        )
        _load_checkpoint(os.path.join(member_dir, checkpoint_name), model, map_location=device)
        if not requires_grad:
            _freeze_module(model)
        else:
            model.train()
        models.append(model)
    return models


def _build_cached_fusion_loader(cfgs, split_name, batch_size, shuffle, drop_last=False):
    data_cfg = _get_data_cfg(cfgs)
    fusion_cfg = _get_section(cfgs, "Fusion")
    cache_dir = os.path.join(_fusion_cache_root(cfgs), split_name)
    dataset = CachedFusionDataset(cache_dir)
    num_workers = int(fusion_cfg.get("cache_loader_workers", data_cfg.get("workers", 0)))
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            fusion_cfg.get("cache_loader_persistent_workers", data_cfg.get("persistent_workers", True))
        )
        loader_kwargs["prefetch_factor"] = int(
            fusion_cfg.get("cache_loader_prefetch_factor", data_cfg.get("prefetch_factor", 2))
        )
    return DataLoader(**loader_kwargs)


def _device_is_cuda(device):
    return str(device).startswith("cuda")


def _move_tensor(tensor, device):
    return tensor.to(device, non_blocking=_device_is_cuda(device))


def _amp_enabled(section_cfg, device):
    return bool(section_cfg.get("amp", True)) and _device_is_cuda(device)


def _autocast_context(enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        except TypeError:
            return torch.amp.autocast(dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def _make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _module_forward(model, network, x):
    if network == "AE":
        return model(x), None
    if network == "AE-U":
        mean, logvar = model(x)
        return mean, torch.exp(logvar)
    if network == "MemAE":
        output = model(x)
        return output["output"], None
    raise ValueError("Unsupported DDAD network: {}".format(network))


def _freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def _save_checkpoint(path, model, optimizer=None, extra=None):
    payload = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def _load_checkpoint(path, model, optimizer=None, map_location=None):
    payload = torch.load(path, map_location=map_location)
    if "model" in payload:
        model.load_state_dict(payload["model"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload.get("extra", {})
    model.load_state_dict(payload)
    return {}


def _build_diffusion_model(cfgs, device):
    data_cfg = _get_data_cfg(cfgs)
    diff_cfg = _get_section(cfgs, "Diffusion")
    model = GaussianDiffusionReconstructor(
        in_channels=1,
        image_size=int(data_cfg.get("img_size", 64)),
        base_channels=int(diff_cfg.get("base_channels", 32)),
        time_dim=int(diff_cfg.get("time_dim", 128)),
        num_steps=int(diff_cfg.get("num_steps", 1000)),
        beta_start=float(diff_cfg.get("beta_start", 1.0e-4)),
        beta_end=float(diff_cfg.get("beta_end", 2.0e-2)),
    )
    return model.to(device)


def _build_clip_model(cfgs, device):
    clip_cfg = _get_section(cfgs, "CLIP")
    model = BiomedCLIPBranch(
        image_size=int(_get_data_cfg(cfgs).get("clip_img_size", 336)),
        backbone_name=str(clip_cfg.get("backbone_name", "ViT-L-14-336")),
        pretrained=str(clip_cfg.get("pretrained", "openai")),
        feature_layers=list(clip_cfg.get("feature_layers", [6, 12, 18, 24])),
        prompt_token_count=int(clip_cfg.get("prompt_token_count", 4)),
        seg_adapter_dim=int(clip_cfg.get("seg_adapter_dim", 512)),
        det_adapter_dim=int(clip_cfg.get("det_adapter_dim", 512)),
        prompt_ensemble=str(clip_cfg.get("prompt_ensemble", "rsna_chest_xray")),
        normal_prompts=clip_cfg.get("normal_prompts"),
        abnormal_prompts=clip_cfg.get("abnormal_prompts"),
        use_dap_pooling=bool(clip_cfg.get("use_dap_pooling", False)),
        dap_alpha=float(clip_cfg.get("dap_alpha", 0.5)),
        dap_detach_map=bool(clip_cfg.get("dap_detach_map", True)),
        dap_temperature=float(clip_cfg.get("dap_temperature", 1.0)),
        offline_pretrained=bool(clip_cfg.get("offline_pretrained", False)),
    )
    return model.to(device)


def _configure_clip_trainable_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

    for module in [model.seg_adapters, model.det_adapters]:
        for param in module.parameters():
            param.requires_grad = True

    model.normal_prompt_tokens.requires_grad = True
    model.abnormal_prompt_tokens.requires_grad = True
    model.logit_scale.requires_grad = True


def _build_fusion_model(cfgs, device):
    fusion_cfg = _get_section(cfgs, "Fusion")
    inferred_in_channels = len(_fusion_spatial_map_order())
    configured_in_channels = fusion_cfg.get("in_channels")
    if configured_in_channels is not None and int(configured_in_channels) != inferred_in_channels:
        raise ValueError(
            "Fusion in_channels mismatch: config has {}, but runtime expects {} spatial maps.".format(
                configured_in_channels,
                inferred_in_channels,
            )
        )
    model = FusionRefineNet(
        in_channels=inferred_in_channels,
        out_channels=2,
        base_channels=int(fusion_cfg.get("base_channels", 32)),
        aux_in_channels=3,
    )
    return model.to(device)


def _fusion_spatial_map_order():
    return [
        "ddad_inter_img",
        "ddad_intra_img",
        "ddad_inter_band",
        "ddad_intra_band",
        "diff_residual_img",
        "diff_inter_img",
        "diff_residual_band",
        "diff_inter_band",
        "clip_patch_map",
    ]


def _stack_maps_in_order(map_dict):
    return torch.cat([map_dict[name] for name in _fusion_spatial_map_order()], dim=1)


def _build_image_aux_from_raw_maps(ddad_map_raw, diff_map_raw, clip_global_logit, topk_ratio):
    ddad_topk_raw = _topk_mean_score(ddad_map_raw, topk_ratio).view(-1, 1)
    diff_topk_raw = _topk_mean_score(diff_map_raw, topk_ratio).view(-1, 1)
    clip_global_logit = clip_global_logit.float().view(-1, 1)
    return torch.cat([ddad_topk_raw.float(), diff_topk_raw.float(), clip_global_logit], dim=1)


def _build_fusion_inputs(features):
    return _stack_maps_in_order(features["maps"]), features["image_aux"].float()


def _combined_branch_map(branch_maps):
    normalized = [min_max_normalize_map(branch_map) for branch_map in branch_maps]
    return torch.stack(normalized, dim=0).mean(dim=0)


def _combined_branch_map_raw(branch_maps):
    return torch.stack([branch_map.float() for branch_map in branch_maps], dim=0).mean(dim=0)


def _topk_mean_score(score_map, ratio):
    if score_map.dim() != 4:
        raise ValueError("Expected score_map with rank 4, got {}".format(score_map.dim()))

    flat = score_map.float().view(score_map.size(0), -1)
    ratio = float(ratio)
    k = max(1, int(np.ceil(flat.size(1) * ratio)))
    topk_values = torch.topk(flat, k=k, dim=1).values
    return topk_values.mean(dim=1)


def _safe_model_std(tensor_stack, dim=0):
    if tensor_stack.size(dim) <= 1:
        reference = tensor_stack.select(dim, 0)
        return torch.zeros_like(reference)
    return torch.std(tensor_stack, dim=dim, unbiased=False)


def _tensor_stats(tensor):
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    if finite_mask.any():
        finite_values = detached[finite_mask]
        min_value = finite_values.min().item()
        max_value = finite_values.max().item()
    else:
        min_value = float("nan")
        max_value = float("nan")
    return {
        "shape": tuple(detached.shape),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
        "min": min_value,
        "max": max_value,
    }


def _ensure_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        stats = _tensor_stats(tensor)
        raise RuntimeError(
            "{} contains non-finite values. shape={} nan_count={} inf_count={} min={} max={}".format(
                name,
                stats["shape"],
                stats["nan_count"],
                stats["inf_count"],
                stats["min"],
                stats["max"],
            )
        )


def _safe_roc_auc_score(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    valid_mask = np.isfinite(scores)
    if valid_mask.sum() == 0:
        return 0.0
    filtered_labels = labels[valid_mask]
    filtered_scores = scores[valid_mask]
    if len(np.unique(filtered_labels)) < 2:
        return 0.0
    return float(roc_auc_score(filtered_labels, filtered_scores))




def _feature_cache_key(cache_prefix, img_id):
    return "{}:{}".format(cache_prefix, img_id)


def _split_feature_batch(features):
    batch_size = features["x64"].size(0)
    split_samples = []
    for index in range(batch_size):
        split_samples.append({
            "x64": features["x64"][index:index + 1].detach().cpu(),
            "maps": {key: value[index:index + 1].detach().cpu() for key, value in features["maps"].items()},
            "image_aux": features["image_aux"][index:index + 1].detach().cpu(),
            "ddad_map": features["ddad_map"][index:index + 1].detach().cpu(),
            "ddad_map_raw": features["ddad_map_raw"][index:index + 1].detach().cpu(),
            "diff_map": features["diff_map"][index:index + 1].detach().cpu(),
            "diff_map_raw": features["diff_map_raw"][index:index + 1].detach().cpu(),
            "clip_map": features["clip_map"][index:index + 1].detach().cpu(),
            "clip_global_logit": features["clip_global_logit"][index:index + 1].detach().cpu(),
        })
    return split_samples


def _stack_feature_samples(samples, device):
    maps = {
        key: _move_tensor(torch.cat([sample["maps"][key] for sample in samples], dim=0), device)
        for key in samples[0]["maps"].keys()
    }
    return {
        "x64": _move_tensor(torch.cat([sample["x64"] for sample in samples], dim=0), device),
        "maps": maps,
        "image_aux": _move_tensor(torch.cat([sample["image_aux"] for sample in samples], dim=0), device),
        "ddad_map": _move_tensor(torch.cat([sample["ddad_map"] for sample in samples], dim=0), device),
        "ddad_map_raw": _move_tensor(torch.cat([sample["ddad_map_raw"] for sample in samples], dim=0), device),
        "diff_map": _move_tensor(torch.cat([sample["diff_map"] for sample in samples], dim=0), device),
        "diff_map_raw": _move_tensor(torch.cat([sample["diff_map_raw"] for sample in samples], dim=0), device),
        "clip_map": _move_tensor(torch.cat([sample["clip_map"] for sample in samples], dim=0), device),
        "clip_global_logit": _move_tensor(torch.cat([sample["clip_global_logit"] for sample in samples], dim=0), device),
    }


def _get_or_compute_features(batch, cfgs, components, amp_enabled=False, feature_cache=None, cache_prefix=None):
    if feature_cache is None or cache_prefix is None:
        return _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)

    img_ids = list(batch["img_id"])
    cache_keys = [_feature_cache_key(cache_prefix, img_id) for img_id in img_ids]
    missing_keys = [cache_key for cache_key in cache_keys if cache_key not in feature_cache]

    if missing_keys:
        computed_features = _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)
        split_samples = _split_feature_batch(computed_features)
        for cache_key, sample in zip(cache_keys, split_samples):
            feature_cache[cache_key] = sample

    return _stack_feature_samples([feature_cache[cache_key] for cache_key in cache_keys], components["device"])


def _cached_fusion_path(cache_dir, img_id):
    return os.path.join(cache_dir, "{}.pt".format(img_id))


def _save_cached_fusion_batch(cache_dir, batch, fusion_inputs):
    mask_syn = batch["mask_syn"].detach().cpu()
    label_img = batch["label_img"].detach().cpu()
    image_aux = batch["image_aux"].detach().cpu()
    fusion_inputs = fusion_inputs.detach().cpu()
    for index, img_id in enumerate(batch["img_id"]):
        torch.save(
            {
                "fusion_inputs": fusion_inputs[index],
                "image_aux": image_aux[index],
                "mask_syn": mask_syn[index],
                "label_img": label_img[index],
                "img_id": str(img_id),
            },
            _cached_fusion_path(cache_dir, img_id),
        )


def _cache_fusion_split(
    cfgs,
    components,
    split_name,
    subset,
    batch_size,
    synthetic_probability,
    include_clip=True,
    force_deterministic=False,
    deterministic_seed=1337,
    dataset_kwargs=None,
):
    fusion_cfg = _get_section(cfgs, "Fusion")
    amp_enabled = _amp_enabled(fusion_cfg, components["device"])
    cache_dir = _ensure_dir(os.path.join(_fusion_cache_root(cfgs), split_name))

    for file_name in os.listdir(cache_dir):
        if file_name.endswith(".pt"):
            os.remove(os.path.join(cache_dir, file_name))

    loader = _build_multibranch_loader(
        cfgs,
        subset=subset,
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=synthetic_probability,
        drop_last=False,
        include_clip=include_clip,
        force_deterministic=force_deterministic,
        deterministic_seed=deterministic_seed,
        dataset_kwargs=dataset_kwargs,
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="cache_{}".format(split_name)):
            features = _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)
            fusion_inputs, image_aux = _build_fusion_inputs(features)
            _save_cached_fusion_batch(
                cache_dir,
                {
                    "mask_syn": batch["mask_syn"],
                    "label_img": batch["label_img"],
                    "img_id": batch["img_id"],
                    "image_aux": image_aux,
                },
                fusion_inputs,
            )

    return cache_dir


def cache_fusion_features(cfgs):
    device = _get_device(cfgs)
    fusion_cfg = _get_section(cfgs, "Fusion")
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)
    band_cfg = _get_fusion_band_cfg(cfgs)
    components = _gather_components(cfgs, device)
    train_batch_size = int(fusion_cfg.get("cache_bs", fusion_cfg.get("bs", 16)))
    eval_batch_size = int(fusion_cfg.get("cache_eval_bs", fusion_cfg.get("val_bs", train_batch_size)))
    synthetic_probability = float(fusion_cfg.get("synthetic_probability", 0.5))
    deterministic_seed = int(fusion_cfg.get("cache_seed", 3407))
    print(
        "=> Fusion cache preprocess: band_mode={} vector_dim={} band_score_topk_ratio={:.2%}".format(
            band_cfg["band_mode"],
            _fusion_vector_dim(cfgs),
            band_cfg["band_score_topk_ratio"],
        )
    )

    print("=> Caching fusion features under {}".format(_fusion_cache_root(cfgs)))
    print("=> Cache fusion feature extractors: {} device(s)".format(len(components.get("extractors", [components]))))
    _cache_fusion_split(
        cfgs,
        components,
        split_name="train_cached",
        subset=protocol_cfg["synthetic_train_subset"],
        batch_size=train_batch_size,
        synthetic_probability=synthetic_probability,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed,
        dataset_kwargs=_baseline_dataset_kwargs(cfgs),
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="synthetic_val",
        subset=protocol_cfg["synthetic_val_subset"],
        batch_size=eval_batch_size,
        synthetic_probability=synthetic_probability,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=protocol_cfg["synthetic_val_seed"] if protocol_cfg["mode"] == "fair" else deterministic_seed + 1000,
        dataset_kwargs=_baseline_dataset_kwargs(cfgs),
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="synthetic_test",
        subset=protocol_cfg["synthetic_eval_subset"],
        batch_size=eval_batch_size,
        synthetic_probability=synthetic_probability,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=protocol_cfg["synthetic_eval_seed"] if protocol_cfg["mode"] == "fair" else deterministic_seed + 2000,
        dataset_kwargs=_baseline_dataset_kwargs(cfgs),
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="real_train",
        subset=protocol_cfg["real_train_subset"],
        batch_size=train_batch_size,
        synthetic_probability=0.0,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed + 2500,
        dataset_kwargs=_baseline_dataset_kwargs(cfgs),
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="real_val",
        subset=protocol_cfg["real_val_subset"],
        batch_size=eval_batch_size,
        synthetic_probability=0.0,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed + 2750,
        dataset_kwargs=_baseline_dataset_kwargs(cfgs),
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="real_test",
        subset=protocol_cfg["real_eval_subset"],
        batch_size=eval_batch_size,
        synthetic_probability=0.0,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed + 3000,
        dataset_kwargs=_baseline_dataset_kwargs(cfgs),
    )


def _clone_frozen_module(module, device):
    cloned = copy.deepcopy(module)
    cloned.to(device)
    _freeze_module(cloned)
    return cloned


def _clone_extractor_to_device(extractor, device):
    cloned = {
        "device": device,
        "module_a": [_clone_frozen_module(model, device) for model in extractor["module_a"]],
        "module_b": [_clone_frozen_module(model, device) for model in extractor["module_b"]],
        "clip_model": _clone_frozen_module(extractor["clip_model"], device),
        "has_diffusion": bool(extractor.get("has_diffusion", False)),
    }
    if cloned["has_diffusion"]:
        cloned["diff_a_model"] = _clone_frozen_module(extractor["diff_a_model"], device)
        cloned["diff_b_model"] = _clone_frozen_module(extractor["diff_b_model"], device)
    else:
        cloned["diff_a_model"] = None
        cloned["diff_b_model"] = None
    return cloned


def _split_batch_for_extractors(batch, extractor_count):
    batch_size = int(batch["image_64"].size(0))
    if batch_size <= 0:
        return []

    chunk_count = min(max(1, int(extractor_count)), batch_size)
    base_chunk = batch_size // chunk_count
    remainder = batch_size % chunk_count
    chunks = []
    start = 0
    for chunk_index in range(chunk_count):
        chunk_size = base_chunk + (1 if chunk_index < remainder else 0)
        end = start + chunk_size
        chunk = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                chunk[key] = value[start:end]
            elif isinstance(value, (list, tuple)):
                chunk[key] = list(value[start:end])
            else:
                chunk[key] = value
        chunks.append(chunk)
        start = end
    return chunks


def _merge_feature_outputs(feature_chunks, target_device):
    if len(feature_chunks) == 1:
        single = feature_chunks[0]
        return {
            "x64": _move_tensor(single["x64"], target_device),
            "maps": {key: _move_tensor(value, target_device) for key, value in single["maps"].items()},
            "image_aux": _move_tensor(single["image_aux"], target_device),
            "ddad_map": _move_tensor(single["ddad_map"], target_device),
            "ddad_map_raw": _move_tensor(single["ddad_map_raw"], target_device),
            "diff_map": _move_tensor(single["diff_map"], target_device),
            "diff_map_raw": _move_tensor(single["diff_map_raw"], target_device),
            "clip_map": _move_tensor(single["clip_map"], target_device),
            "clip_global_logit": _move_tensor(single["clip_global_logit"], target_device),
        }

    merged_maps = {}
    map_keys = feature_chunks[0]["maps"].keys()
    for key in map_keys:
        merged_maps[key] = torch.cat([_move_tensor(chunk["maps"][key], target_device) for chunk in feature_chunks], dim=0)

    return {
        "x64": torch.cat([_move_tensor(chunk["x64"], target_device) for chunk in feature_chunks], dim=0),
        "maps": merged_maps,
        "image_aux": torch.cat([_move_tensor(chunk["image_aux"], target_device) for chunk in feature_chunks], dim=0),
        "ddad_map": torch.cat([_move_tensor(chunk["ddad_map"], target_device) for chunk in feature_chunks], dim=0),
        "ddad_map_raw": torch.cat([_move_tensor(chunk["ddad_map_raw"], target_device) for chunk in feature_chunks], dim=0),
        "diff_map": torch.cat([_move_tensor(chunk["diff_map"], target_device) for chunk in feature_chunks], dim=0),
        "diff_map_raw": torch.cat([_move_tensor(chunk["diff_map_raw"], target_device) for chunk in feature_chunks], dim=0),
        "clip_map": torch.cat([_move_tensor(chunk["clip_map"], target_device) for chunk in feature_chunks], dim=0),
        "clip_global_logit": torch.cat(
            [_move_tensor(chunk["clip_global_logit"], target_device) for chunk in feature_chunks],
            dim=0,
        ),
    }


def _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=False):
    extractors = components.get("extractors", [components])
    if len(extractors) <= 1 or int(batch["image_64"].size(0)) <= 1:
        return _compute_all_branch_features(batch, cfgs, extractors[0], amp_enabled=amp_enabled)

    batch_chunks = _split_batch_for_extractors(batch, len(extractors))
    feature_chunks = []
    for extractor, batch_chunk in zip(extractors, batch_chunks):
        feature_chunks.append(_compute_all_branch_features(batch_chunk, cfgs, extractor, amp_enabled=amp_enabled))
    return _merge_feature_outputs(feature_chunks, components["device"])


def _evaluate_diffusion_recon_loss(model, loader, device, diff_cfg, amp_enabled=False):
    losses = []
    with torch.no_grad():
        for batch in loader:
            x64 = _move_tensor(batch["image_64"], device)
            with _autocast_context(amp_enabled):
                recon = model.reconstruct(
                    x64,
                    t_recon=int(diff_cfg.get("t_recon", 200)),
                    ddim_steps=int(diff_cfg.get("ddim_steps", 50)),
                )
            losses.append(F.mse_loss(recon.float(), x64.float()).item())
    return float(np.mean(losses)) if len(losses) > 0 else 0.0


def train_diffusion_module(cfgs, mode):
    if mode not in {"diff_a", "diff_b"}:
        raise ValueError("Unsupported diffusion mode: {}".format(mode))

    device = _get_device(cfgs)
    diff_cfg = _get_section(cfgs, "Diffusion")
    data_cfg = _get_data_cfg(cfgs)
    amp_enabled = _amp_enabled(diff_cfg, device)
    subset = "train_plus_unlabeled" if mode == "diff_a" else "train_normal"
    diffusion_workers = int(diff_cfg.get("workers", min(int(data_cfg.get("workers", 0)), 8)))
    diffusion_persistent_workers = bool(diff_cfg.get("persistent_workers", diffusion_workers > 0))
    diffusion_prefetch_factor = int(diff_cfg.get("prefetch_factor", 2))
    diffusion_cache_images = bool(diff_cfg.get("cache_images", True))
    train_loader = _build_multibranch_loader(
        cfgs,
        subset=subset,
        batch_size=int(diff_cfg.get("bs", 16)),
        shuffle=True,
        synthetic_probability=0.0,
        drop_last=True,
        include_clip=False,
        num_workers_override=diffusion_workers,
        persistent_workers_override=diffusion_persistent_workers,
        prefetch_factor_override=diffusion_prefetch_factor,
        cache_images_override=diffusion_cache_images,
    )
    val_loader = _build_multibranch_loader(
        cfgs,
        subset="synthetic_val",
        batch_size=int(diff_cfg.get("val_bs", diff_cfg.get("bs", 16))),
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        include_clip=False,
        num_workers_override=diffusion_workers,
        persistent_workers_override=diffusion_persistent_workers,
        prefetch_factor_override=diffusion_prefetch_factor,
        cache_images_override=diffusion_cache_images,
    )

    print(
        "=> {} loader uses include_clip=False, workers={}, persistent_workers={}, prefetch_factor={}, cache_images={}".format(
            mode,
            diffusion_workers,
            diffusion_persistent_workers,
            diffusion_prefetch_factor,
            diffusion_cache_images,
        )
    )

    model = _build_diffusion_model(cfgs, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(diff_cfg.get("lr", 1.0e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(diff_cfg.get("weight_decay", 0.0)),
    )
    scaler = _make_grad_scaler(amp_enabled)

    out_dir = _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "diffusion"))
    writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_diffusion_{}".format(mode)))

    best_val = float("inf")
    best_path = os.path.join(out_dir, "{}_best.pth".format(mode))
    last_path = os.path.join(out_dir, "{}.pth".format(mode))
    num_epoch = int(diff_cfg.get("num_epoch", 40))

    for epoch in range(1, num_epoch + 1):
        model.train()
        losses = AverageMeter()
        start = time.time()
        for batch in train_loader:
            x64 = _move_tensor(batch["image_64"], device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(amp_enabled):
                loss = model.training_loss(x64)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.update(loss.item(), x64.size(0))

        model.eval()
        val_loss = _evaluate_diffusion_recon_loss(model, val_loader, device, diff_cfg, amp_enabled=amp_enabled)
        writer.add_scalar("train_loss", losses.avg, epoch)
        writer.add_scalar("val_recon_loss", val_loss, epoch)
        print(
            "{} Epoch[{}/{}]\tTime:{:.1f}s\tTrain Loss:{:.4f}\tVal Recon:{:.4f}\tAMP:{}".format(
                mode,
                epoch,
                num_epoch,
                time.time() - start,
                losses.avg,
                val_loss,
                amp_enabled,
            )
        )

        _save_checkpoint(last_path, model, optimizer, extra={"epoch": epoch, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(best_path, model, optimizer, extra={"epoch": epoch, "val_loss": val_loss})

    writer.close()


def _build_segmentation_probabilities(single_channel_logits):
    two_channel_logits = torch.cat([-single_channel_logits, single_channel_logits], dim=1)
    return two_channel_logits, torch.softmax(two_channel_logits, dim=1)


def _soft_dice_loss(anomaly_prob, target_mask, eps=1.0e-6):
    target = target_mask.float()
    if target.dim() == 3:
        target = target.unsqueeze(1)

    if anomaly_prob.dim() == 4 and anomaly_prob.size(1) == 2:
        anomaly_prob = anomaly_prob[:, 1:2]
    elif anomaly_prob.dim() == 3:
        anomaly_prob = anomaly_prob.unsqueeze(1)

    intersection = (anomaly_prob * target).sum(dim=(1, 2, 3))
    denominator = anomaly_prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _soft_iou_loss(anomaly_prob, target_mask, eps=1.0e-6):
    target = target_mask.float()
    if target.dim() == 3:
        target = target.unsqueeze(1)

    if anomaly_prob.dim() == 4 and anomaly_prob.size(1) == 2:
        anomaly_prob = anomaly_prob[:, 1:2]
    elif anomaly_prob.dim() == 3:
        anomaly_prob = anomaly_prob.unsqueeze(1)

    intersection = (anomaly_prob * target).sum(dim=(1, 2, 3))
    union = anomaly_prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


def _fusion_segmentation_loss(cfgs, seg_prob, target_mask, focal_loss):
    fusion_cfg = _get_section(cfgs, "Fusion")
    dice_loss_weight = float(fusion_cfg.get("dice_loss_weight", 0.5))
    iou_loss_weight = float(fusion_cfg.get("iou_loss_weight", 0.25))
    seg_loss = focal_loss(seg_prob, target_mask)
    if dice_loss_weight > 0.0:
        seg_loss = seg_loss + dice_loss_weight * _soft_dice_loss(seg_prob[:, 1:2], target_mask)
    if iou_loss_weight > 0.0:
        seg_loss = seg_loss + iou_loss_weight * _soft_iou_loss(seg_prob[:, 1:2], target_mask)
    return seg_loss


def _synthetic_pixel_metrics(score_maps, masks):
    flat_scores = np.concatenate([score.reshape(-1) for score in score_maps]) if score_maps else np.array([])
    flat_masks = np.concatenate([mask.reshape(-1) for mask in masks]) if masks else np.array([])

    if flat_scores.size == 0 or flat_masks.size == 0 or float(np.max(flat_masks)) == float(np.min(flat_masks)):
        auc = 0.0
    else:
        auc = roc_auc_score(flat_masks, flat_scores)

    dice_scores = []
    for score_map, mask in zip(score_maps, masks):
        binary = (score_map >= 0.5).astype(np.float32)
        target = mask.astype(np.float32)
        intersection = float((binary * target).sum())
        denom = float(binary.sum() + target.sum())
        dice_scores.append((2.0 * intersection + 1.0e-6) / (denom + 1.0e-6))
    dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    return auc, dice




def _load_diffusion_pair(cfgs, device):
    diff_out_dir = os.path.join(_checkpoint_out_dir(cfgs), "diffusion")
    diff_a_model = _build_diffusion_model(cfgs, device)
    diff_b_model = _build_diffusion_model(cfgs, device)
    _load_checkpoint(os.path.join(diff_out_dir, "diff_a_best.pth"), diff_a_model, map_location=device)
    _load_checkpoint(os.path.join(diff_out_dir, "diff_b_best.pth"), diff_b_model, map_location=device)
    _freeze_module(diff_a_model)
    _freeze_module(diff_b_model)
    return diff_a_model, diff_b_model


def _load_clip_model(cfgs, device):
    clip_model = _build_clip_model(cfgs, device)
    clip_dir = os.path.join(cfgs["Exp"]["out_dir"], "clip")
    _load_checkpoint(os.path.join(clip_dir, "clip_best.pth"), clip_model, map_location=device)
    _freeze_module(clip_model)
    return clip_model


def _clip_stage_dir(cfgs, stage_name):
    return _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], str(stage_name)))


def _clip_stage_checkpoint_paths(cfgs, stage_name):
    stage_dir = _clip_stage_dir(cfgs, stage_name)
    stage_name = str(stage_name)
    return {
        "dir": stage_dir,
        "best": os.path.join(stage_dir, "{}_best.pth".format(stage_name)),
        "last": os.path.join(stage_dir, "{}_last.pth".format(stage_name)),
    }


def _clip_student_fusion_checkpoint_paths(cfgs):
    return _clip_stage_checkpoint_paths(cfgs, "clip_student_fusion")


def _load_clip_stage_model(cfgs, device, stage_name):
    checkpoint_paths = _clip_stage_checkpoint_paths(cfgs, stage_name)
    if not os.path.exists(checkpoint_paths["best"]):
        raise FileNotFoundError("Expected CLIP checkpoint at {}".format(checkpoint_paths["best"]))
    clip_model = _build_clip_model(cfgs, device)
    _load_checkpoint(checkpoint_paths["best"], clip_model, map_location=device)
    _freeze_module(clip_model)
    return clip_model


def _build_pseudo_label_loader(cfgs, manifest_path, batch_size, shuffle, drop_last=False):
    data_cfg = _get_data_cfg(cfgs)
    loader_overrides = _get_clip_loader_overrides(cfgs)
    dataset = PseudoLabelDataset(
        main_path=_dataset_root(data_cfg.get("dataset", "rsna")),
        manifest_path=manifest_path,
        image_size=int(data_cfg.get("img_size", 64)),
        clip_image_size=int(data_cfg.get("clip_img_size", 336)),
        cache_images=loader_overrides["cache_images"],
        cache_clip_images=loader_overrides["cache_clip_images"],
    )
    num_workers = int(loader_overrides["train_workers"])
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(loader_overrides["train_persistent_workers"])
        loader_kwargs["prefetch_factor"] = int(loader_overrides["train_prefetch_factor"])
    return DataLoader(**loader_kwargs)


def _gather_components(cfgs, device):
    band_cfg = _get_fusion_band_cfg(cfgs)
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)
    if protocol_cfg["ddad_source"] == "weak":
        module_a = _load_weak_ddad_ensemble(cfgs, "weak_a")
        module_b = _load_weak_ddad_ensemble(cfgs, "weak_b")
    else:
        module_a, module_b = load_ab(cfgs)
    for model in module_a + module_b:
        model.to(device)
        _freeze_module(model)

    has_diffusion = bool(band_cfg["enable_diffusion_in_fusion"])
    if has_diffusion:
        diff_a_model, diff_b_model = _load_diffusion_pair(cfgs, device)
    else:
        diff_a_model, diff_b_model = None, None
    clip_model = _load_clip_model(cfgs, device)
    primary_extractor = {
        "device": device,
        "module_a": module_a,
        "module_b": module_b,
        "diff_a_model": diff_a_model,
        "diff_b_model": diff_b_model,
        "clip_model": clip_model,
        "has_diffusion": has_diffusion,
    }

    extractor_gpu_ids = _get_gpu_ids(cfgs)
    extractors = [primary_extractor]
    if len(extractor_gpu_ids) <= 1:
        return {
            **primary_extractor,
            "extractors": extractors,
            "feature_cache": {},
        }

    for gpu_id in extractor_gpu_ids[1:]:
        replica_device = torch.device("cuda:{}".format(gpu_id))
        extractors.append(_clone_extractor_to_device(primary_extractor, replica_device))

    return {
        **primary_extractor,
        "extractors": extractors,
        "feature_cache": {},
    }


def _evaluate_fusion_epoch(loader, cfgs, components, fusion_model, amp_enabled=False, feature_cache=None, cache_prefix=None):
    focal_loss = FocalLoss()
    image_loss_fn = nn.BCEWithLogitsLoss()
    image_loss_weight = float(_get_section(cfgs, "Fusion").get("image_loss_weight", 0.3))
    losses = []
    image_labels, image_scores = [], []
    pixel_maps, pixel_masks = [], []

    with torch.no_grad():
        for batch in loader:
            if "fusion_inputs" in batch:
                fusion_inputs = _move_tensor(batch["fusion_inputs"], components["device"])
                image_aux = _move_tensor(batch["image_aux"], components["device"])
            else:
                features = _get_or_compute_features(
                    batch,
                    cfgs,
                    components,
                    amp_enabled=amp_enabled,
                    feature_cache=feature_cache,
                    cache_prefix=cache_prefix,
                )
                fusion_inputs, image_aux = _build_fusion_inputs(features)
            _ensure_finite_tensor("fusion_inputs", fusion_inputs)
            _ensure_finite_tensor("fusion_image_aux", image_aux)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(fusion_inputs, image_aux)
            _ensure_finite_tensor("fusion_seg_logits", outputs["seg_logits"])
            _ensure_finite_tensor("fusion_image_logit", outputs["image_logit"])
            seg_prob, anomaly_prob = _fusion_segmentation_probabilities(cfgs, None, outputs["seg_logits"])
            seg_loss = _fusion_segmentation_loss(
                cfgs,
                seg_prob,
                _move_tensor(batch["mask_syn"], components["device"]),
                focal_loss,
            )
            image_loss = image_loss_fn(
                outputs["image_logit"].view(-1).float(),
                _move_tensor(batch["label_img"].float(), components["device"]).view(-1),
            )
            total_loss = seg_loss + image_loss_weight * image_loss
            losses.append(total_loss.item())

            pixel_maps.extend(anomaly_prob.cpu().numpy())
            pixel_masks.extend(batch["mask_syn"].cpu().numpy())
            image_labels.extend(batch["label_img"].cpu().numpy().tolist())
            image_scores.extend(torch.sigmoid(outputs["image_logit"].float()).view(-1).cpu().numpy().tolist())

    image_auc = _safe_roc_auc_score(image_labels, image_scores)
    pixel_auc, pixel_dice = _synthetic_pixel_metrics(pixel_maps, pixel_masks)
    return float(np.mean(losses)) if losses else 0.0, image_auc, pixel_auc, pixel_dice


def _evaluate_fusion_real_image_auc(
    loader,
    cfgs,
    components,
    fusion_model,
    amp_enabled=False,
    feature_cache=None,
    cache_prefix=None,
):
    labels, scores = [], []
    with torch.no_grad():
        for batch in loader:
            if "fusion_inputs" in batch:
                fusion_inputs = _move_tensor(batch["fusion_inputs"], components["device"])
                image_aux = _move_tensor(batch["image_aux"], components["device"])
            else:
                features = _get_or_compute_features(
                    batch,
                    cfgs,
                    components,
                    amp_enabled=amp_enabled,
                    feature_cache=feature_cache,
                    cache_prefix=cache_prefix,
                )
                fusion_inputs, image_aux = _build_fusion_inputs(features)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(fusion_inputs, image_aux)
            labels.extend(batch["label_img"].cpu().numpy().tolist())
            scores.extend(torch.sigmoid(outputs["image_logit"].float()).view(-1).cpu().numpy().tolist())
    return _safe_roc_auc_score(labels, scores)


def train_fusion_module(cfgs):
    device = _get_device(cfgs)
    fusion_cfg = _get_section(cfgs, "Fusion")
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)
    band_cfg = _get_fusion_band_cfg(cfgs)
    amp_enabled = _amp_enabled(fusion_cfg, device)
    batch_size = int(fusion_cfg.get("bs", 4))
    synthetic_probability = float(fusion_cfg.get("synthetic_probability", 0.5))
    grad_clip = fusion_cfg.get("grad_clip", 1.0)
    use_cached_features = bool(fusion_cfg.get("use_cached_features", True))
    val_batch_size = int(fusion_cfg.get("val_bs", batch_size))

    components = _gather_components(cfgs, device)
    print("=> Fusion feature extractors: {} device(s)".format(len(components.get("extractors", [components]))))
    print(
        "=> Fusion preprocess: band_mode={} vector_dim={} band_score_topk_ratio={:.2%}".format(
            band_cfg["band_mode"],
            _fusion_vector_dim(cfgs),
            band_cfg["band_score_topk_ratio"],
        )
    )

    if use_cached_features:
        cache_root = _fusion_cache_root(cfgs)
        required_splits = ["train_cached", "synthetic_val", "real_train", "real_val"]
        missing_splits = [
            split_name for split_name in required_splits
            if not os.path.isdir(os.path.join(cache_root, split_name))
            or len([file_name for file_name in os.listdir(os.path.join(cache_root, split_name)) if file_name.endswith(".pt")]) == 0
        ]
        if len(missing_splits) > 0:
            raise RuntimeError(
                "Fusion cache is missing split(s): {}. Please run `python main.py --config <your_config> --mode cache_fusion` first.".format(
                    ", ".join(missing_splits)
                )
            )
        train_loader = _build_cached_fusion_loader(
            cfgs,
            split_name="train_cached",
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        real_train_loader = _build_cached_fusion_loader(
            cfgs,
            split_name="real_train",
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = _build_cached_fusion_loader(
            cfgs,
            split_name="synthetic_val",
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
        )
        real_val_loader = _build_cached_fusion_loader(
            cfgs,
            split_name="real_val",
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
        )
        print("=> Fusion training uses cached features from {}".format(cache_root))
    else:
        train_loader = _build_multibranch_loader(
            cfgs,
            subset=protocol_cfg["synthetic_train_subset"],
            batch_size=batch_size,
            shuffle=True,
            synthetic_probability=synthetic_probability,
            drop_last=True,
            **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["synthetic_train_subset"], "synthetic_train"),
        )
        real_train_loader = _build_multibranch_loader(
            cfgs,
            subset=protocol_cfg["real_train_subset"],
            batch_size=batch_size,
            shuffle=True,
            synthetic_probability=0.0,
            drop_last=True,
            **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["real_train_subset"], "real_train"),
        )
        val_loader = _build_multibranch_loader(
            cfgs,
            subset=protocol_cfg["synthetic_val_subset"],
            batch_size=val_batch_size,
            shuffle=False,
            synthetic_probability=synthetic_probability,
            drop_last=False,
            **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["synthetic_val_subset"], "synthetic_val"),
        )
        real_val_loader = _build_multibranch_loader(
            cfgs,
            subset=protocol_cfg["real_val_subset"],
            batch_size=val_batch_size,
            shuffle=False,
            synthetic_probability=0.0,
            drop_last=False,
            **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["real_val_subset"], "real_val"),
        )
        print("=> Fusion training uses online feature recomputation (fallback mode)")
    print("=> Fusion baseline protocol: {} (ddad_source={})".format(protocol_cfg["mode"], protocol_cfg["ddad_source"]))

    fusion_model = _build_fusion_model(cfgs, device)
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=float(fusion_cfg.get("lr", 1.0e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(fusion_cfg.get("weight_decay", 1.0e-4)),
    )
    scaler = _make_grad_scaler(amp_enabled)
    focal_loss = FocalLoss()
    image_loss_fn = nn.BCEWithLogitsLoss()

    out_dir = _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "fusion"))
    writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_fusion"))
    best_score = -float("inf")
    best_path = os.path.join(out_dir, "fusion_best.pth")
    last_path = os.path.join(out_dir, "fusion_refine.pth")
    num_epoch = int(fusion_cfg.get("num_epoch", 30))
    image_loss_weight = float(fusion_cfg.get("image_loss_weight", 0.3))

    for epoch in range(1, num_epoch + 1):
        fusion_model.train()
        total_losses = AverageMeter()
        synthetic_losses = AverageMeter()
        real_losses = AverageMeter()
        start = time.time()
        for batch in train_loader:
            if use_cached_features:
                fusion_inputs = _move_tensor(batch["fusion_inputs"], device)
                image_aux = _move_tensor(batch["image_aux"], device)
            else:
                with torch.no_grad():
                    features = _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)
                    fusion_inputs, image_aux = _build_fusion_inputs(features)
                    fusion_inputs = fusion_inputs.detach()
                    image_aux = image_aux.detach()
            _ensure_finite_tensor("fusion_train_inputs", fusion_inputs)
            _ensure_finite_tensor("fusion_train_image_aux", image_aux)

            mask_syn = _move_tensor(batch["mask_syn"], device)
            label_img = _move_tensor(batch["label_img"].float(), device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(fusion_inputs, image_aux)
            _ensure_finite_tensor("fusion_train_seg_logits", outputs["seg_logits"])
            _ensure_finite_tensor("fusion_train_image_logit", outputs["image_logit"])
            seg_prob, _ = _fusion_segmentation_probabilities(cfgs, None, outputs["seg_logits"])
            seg_loss = _fusion_segmentation_loss(cfgs, seg_prob, mask_syn, focal_loss)
            image_loss = image_loss_fn(outputs["image_logit"].view(-1).float(), label_img.view(-1))
            loss = seg_loss + image_loss_weight * image_loss

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            total_losses.update(loss.item(), fusion_inputs.size(0))
            synthetic_losses.update(loss.item(), fusion_inputs.size(0))

        for batch in real_train_loader:
            if use_cached_features:
                fusion_inputs = _move_tensor(batch["fusion_inputs"], device)
                image_aux = _move_tensor(batch["image_aux"], device)
            else:
                with torch.no_grad():
                    features = _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)
                    fusion_inputs, image_aux = _build_fusion_inputs(features)
                    fusion_inputs = fusion_inputs.detach()
                    image_aux = image_aux.detach()
            _ensure_finite_tensor("fusion_real_train_inputs", fusion_inputs)
            _ensure_finite_tensor("fusion_real_train_image_aux", image_aux)

            label_img = _move_tensor(batch["label_img"].float(), device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(fusion_inputs, image_aux)
                image_loss = image_loss_fn(outputs["image_logit"].view(-1).float(), label_img.view(-1))

            scaler.scale(image_loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            total_losses.update(image_loss.item(), fusion_inputs.size(0))
            real_losses.update(image_loss.item(), fusion_inputs.size(0))

        fusion_model.eval()
        val_loss, synthetic_image_auc, synthetic_pixel_auc, synthetic_dice = _evaluate_fusion_epoch(
            val_loader,
            cfgs,
            components,
            fusion_model,
            amp_enabled=amp_enabled,
            feature_cache=components["feature_cache"],
            cache_prefix="synthetic_val",
        )
        real_image_auc = _evaluate_fusion_real_image_auc(
            real_val_loader,
            cfgs,
            components,
            fusion_model,
            amp_enabled=amp_enabled,
            feature_cache=components["feature_cache"],
            cache_prefix="real_val",
        )
        selection_score = real_image_auc + synthetic_pixel_auc + synthetic_dice

        writer.add_scalar("train_loss", total_losses.avg, epoch)
        writer.add_scalar("train_synthetic_loss", synthetic_losses.avg, epoch)
        writer.add_scalar("train_real_loss", real_losses.avg, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_synthetic_image_auc", synthetic_image_auc, epoch)
        writer.add_scalar("val_real_image_auc", real_image_auc, epoch)
        writer.add_scalar("val_synthetic_pixel_auc", synthetic_pixel_auc, epoch)
        writer.add_scalar("val_synthetic_dice", synthetic_dice, epoch)
        writer.add_scalar("val_selection_score", selection_score, epoch)
        print(
            "fusion Epoch[{}/{}]\tTime:{:.1f}s\tLoss:{:.4f}\tSyn Loss:{:.4f}\tReal Loss:{:.4f}\tVal Loss:{:.4f}\tReal AUC:{:.4f}\tSyn Image AUC:{:.4f}\tSyn Pixel AUC:{:.4f}\tSyn Dice:{:.4f}\tSel:{:.4f}\tAMP:{}".format(
                epoch,
                num_epoch,
                time.time() - start,
                total_losses.avg,
                synthetic_losses.avg,
                real_losses.avg,
                val_loss,
                real_image_auc,
                synthetic_image_auc,
                synthetic_pixel_auc,
                synthetic_dice,
                selection_score,
                amp_enabled,
            )
        )

        checkpoint_extra = {
            "epoch": epoch,
            "selection_score": selection_score,
            "real_image_auc": real_image_auc,
            "synthetic_image_auc": synthetic_image_auc,
            "synthetic_pixel_auc": synthetic_pixel_auc,
            "synthetic_dice": synthetic_dice,
            "val_loss": val_loss,
        }
        _save_checkpoint(last_path, fusion_model, optimizer, extra=checkpoint_extra)
        if selection_score > best_score:
            best_score = selection_score
            _save_checkpoint(best_path, fusion_model, optimizer, extra=checkpoint_extra)

    writer.close()


def _normalize_visual_map(value):
    value = np.asarray(value, dtype=np.float32)
    value = np.squeeze(value)
    return (value - value.min()) / (value.max() - value.min() + 1.0e-6)


def _to_rgb_image(image):
    image = _normalize_visual_map(image)
    return np.stack([image, image, image], axis=-1)


def _heatmap_to_rgb(score_map, cmap_name="turbo"):
    cmap = plt.get_cmap(cmap_name)
    return cmap(_normalize_visual_map(score_map))[..., :3].astype(np.float32)


def _get_fusion_visual_cfg(cfgs):
    fusion_cfg = _get_section(cfgs, "Fusion")
    return {
        "vis_threshold": float(fusion_cfg.get("vis_threshold", 0.5)),
        "min_region_area": int(fusion_cfg.get("min_region_area", 64)),
        "max_region_area_ratio": float(fusion_cfg.get("max_region_area_ratio", 0.45)),
        "peak_split_threshold_ratio": float(fusion_cfg.get("peak_split_threshold_ratio", 0.7)),
        "box_nms_iou": float(fusion_cfg.get("box_nms_iou", 0.3)),
        "max_boxes": int(fusion_cfg.get("max_boxes", 3)),
        "export_seed": int(fusion_cfg.get("export_seed", 3407)),
        "export_num_images": int(fusion_cfg.get("export_num_images", 20)),
    }


def _build_real_eval_loader(
    cfgs,
    subset,
    force_deterministic=False,
    deterministic_seed=1337,
    dataset_kwargs=None,
):
    return _build_multibranch_loader(
        cfgs,
        subset=subset,
        batch_size=1,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        force_deterministic=force_deterministic,
        deterministic_seed=deterministic_seed,
        dataset_kwargs=dataset_kwargs,
    )


def _choose_real_box_threshold(labels, scores, default_threshold=0.5):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    valid_mask = np.isfinite(scores)
    if valid_mask.sum() == 0:
        return float(default_threshold)
    labels = labels[valid_mask]
    scores = scores[valid_mask]
    if len(np.unique(labels)) < 2:
        return float(default_threshold)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    if len(thresholds) == 0:
        return float(default_threshold)
    youden_j = tpr - fpr
    best_index = int(np.argmax(youden_j))
    best_threshold = thresholds[best_index]
    if not np.isfinite(best_threshold):
        finite_thresholds = thresholds[np.isfinite(thresholds)]
        if finite_thresholds.size == 0:
            return float(default_threshold)
        best_threshold = finite_thresholds[-1]
    return float(best_threshold)


def _component_bbox(component_mask):
    ys, xs = np.where(component_mask)
    if ys.size == 0 or xs.size == 0:
        return -1, -1, -1, -1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _connected_components_from_mask(binary_mask):
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if binary_mask.ndim != 2 or not np.any(binary_mask):
        return []

    if cv2 is not None:
        binary_u8 = binary_mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
        components = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            components.append({
                "mask": labels == label_id,
                "area": area,
                "bbox_x0": x,
                "bbox_y0": y,
                "bbox_x1": x + w - 1,
                "bbox_y1": y + h - 1,
            })
        return components

    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    components = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            pixels = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not binary_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

            component_mask = np.zeros_like(binary_mask, dtype=bool)
            ys, xs = zip(*pixels)
            component_mask[list(ys), list(xs)] = True
            x0, y0, x1, y1 = _component_bbox(component_mask)
            components.append({
                "mask": component_mask,
                "area": int(len(pixels)),
                "bbox_x0": x0,
                "bbox_y0": y0,
                "bbox_x1": x1,
                "bbox_y1": y1,
            })
    return components


def _make_localization_box(region_mask, score_map):
    area = int(region_mask.sum())
    if area <= 0:
        return None
    x0, y0, x1, y1 = _component_bbox(region_mask)
    masked_scores = np.where(region_mask, score_map, -1.0)
    peak_y, peak_x = np.unravel_index(int(np.argmax(masked_scores)), masked_scores.shape)
    peak_score = float(masked_scores[peak_y, peak_x])
    mean_score = float(score_map[region_mask].mean())
    return {
        "bbox_x0": int(x0),
        "bbox_y0": int(y0),
        "bbox_x1": int(x1),
        "bbox_y1": int(y1),
        "area": area,
        "peak_score": peak_score,
        "mean_score": mean_score,
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "region_score": float(peak_score + 0.25 * mean_score),
        "_mask": region_mask,
    }


def _box_iou(box_a, box_b):
    inter_x0 = max(int(box_a["bbox_x0"]), int(box_b["bbox_x0"]))
    inter_y0 = max(int(box_a["bbox_y0"]), int(box_b["bbox_y0"]))
    inter_x1 = min(int(box_a["bbox_x1"]), int(box_b["bbox_x1"]))
    inter_y1 = min(int(box_a["bbox_y1"]), int(box_b["bbox_y1"]))
    if inter_x1 < inter_x0 or inter_y1 < inter_y0:
        return 0.0
    inter_area = float((inter_x1 - inter_x0 + 1) * (inter_y1 - inter_y0 + 1))
    area_a = float((int(box_a["bbox_x1"]) - int(box_a["bbox_x0"]) + 1) * (int(box_a["bbox_y1"]) - int(box_a["bbox_y0"]) + 1))
    area_b = float((int(box_b["bbox_x1"]) - int(box_b["bbox_x0"]) + 1) * (int(box_b["bbox_y1"]) - int(box_b["bbox_y0"]) + 1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _nms_boxes(boxes, iou_threshold, max_boxes):
    kept = []
    for box in sorted(boxes, key=lambda item: item["region_score"], reverse=True):
        if any(_box_iou(box, existing_box) >= float(iou_threshold) for existing_box in kept):
            continue
        kept.append(box)
        if len(kept) >= int(max_boxes):
            break
    return kept


def _extract_localization_boxes(
    score_map,
    threshold,
    fusion_score=None,
    real_box_threshold=None,
    min_region_area=64,
    max_region_area_ratio=0.45,
    peak_split_threshold_ratio=0.7,
    max_boxes=3,
    nms_iou_threshold=0.3,
):
    normalized_map = _normalize_visual_map(score_map)
    image_height, image_width = normalized_map.shape
    image_area = max(1, image_height * image_width)
    peak_score = float(normalized_map.max()) if normalized_map.size > 0 else 0.0
    passed_gate = real_box_threshold is None or fusion_score is None or float(fusion_score) >= float(real_box_threshold)
    if not passed_gate:
        return {
            "pred_mask": np.zeros_like(normalized_map, dtype=np.float32),
            "passed_gate": False,
            "boxes": [],
            "box_count": 0,
            "peak_score": peak_score,
            "no_box_reason": "below_real_box_threshold",
        }

    base_mask = normalized_map >= float(threshold)
    if not np.any(base_mask):
        return {
            "pred_mask": np.zeros_like(normalized_map, dtype=np.float32),
            "passed_gate": True,
            "boxes": [],
            "box_count": 0,
            "peak_score": peak_score,
            "no_box_reason": "no_region_after_vis_threshold",
        }

    max_region_area = image_area if float(max_region_area_ratio) <= 0 else int(round(image_area * float(max_region_area_ratio)))
    max_region_area = max(1, max_region_area)
    candidate_boxes = []
    for component in _connected_components_from_mask(base_mask):
        if component["area"] < int(min_region_area):
            continue
        component_mask = component["mask"]
        component_peak = float(normalized_map[component_mask].max())
        split_threshold = max(float(threshold), component_peak * float(peak_split_threshold_ratio))
        split_mask = component_mask & (normalized_map >= split_threshold)
        split_components = _connected_components_from_mask(split_mask)
        if len(split_components) > 1:
            region_masks = [split_component["mask"] for split_component in split_components]
        else:
            region_masks = [component_mask]

        for region_mask in region_masks:
            if int(region_mask.sum()) < int(min_region_area):
                continue
            box = _make_localization_box(region_mask, normalized_map)
            if box is None or box["area"] > max_region_area:
                continue
            candidate_boxes.append(box)

    if len(candidate_boxes) == 0:
        return {
            "pred_mask": np.zeros_like(normalized_map, dtype=np.float32),
            "passed_gate": True,
            "boxes": [],
            "box_count": 0,
            "peak_score": peak_score,
            "no_box_reason": "all_regions_filtered",
        }

    kept_boxes = _nms_boxes(candidate_boxes, iou_threshold=nms_iou_threshold, max_boxes=max_boxes)
    pred_mask = np.zeros_like(normalized_map, dtype=np.float32)
    serialized_boxes = []
    for rank, box in enumerate(kept_boxes, start=1):
        pred_mask[box["_mask"]] = 1.0
        serialized_boxes.append({
            "rank": int(rank),
            "bbox_x0": int(box["bbox_x0"]),
            "bbox_y0": int(box["bbox_y0"]),
            "bbox_x1": int(box["bbox_x1"]),
            "bbox_y1": int(box["bbox_y1"]),
            "area": int(box["area"]),
            "peak_score": float(box["peak_score"]),
            "mean_score": float(box["mean_score"]),
            "region_score": float(box["region_score"]),
            "peak_x": int(box["peak_x"]),
            "peak_y": int(box["peak_y"]),
        })

    return {
        "pred_mask": pred_mask,
        "passed_gate": True,
        "boxes": serialized_boxes,
        "box_count": len(serialized_boxes),
        "peak_score": peak_score,
        "no_box_reason": "" if len(serialized_boxes) > 0 else "all_regions_filtered",
    }


def _sanitize_sample_id(sample_id):
    sample_id = str(sample_id)
    for token in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        sample_id = sample_id.replace(token, "_")
    return sample_id


def _collect_real_image_scores(cfgs, components, fusion_model, subset, cache_prefix):
    fusion_cfg = _get_section(cfgs, "Fusion")
    amp_enabled = _amp_enabled(fusion_cfg, components["device"])
    loader = _build_real_eval_loader(cfgs, subset)
    labels, scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="{}_scores".format(subset)):
            features = _get_or_compute_features(
                batch,
                cfgs,
                components,
                amp_enabled=amp_enabled,
                feature_cache=components["feature_cache"],
                cache_prefix=cache_prefix,
            )
            fusion_inputs, image_aux = _build_fusion_inputs(features)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(fusion_inputs, image_aux)
            labels.extend(batch["label_img"].cpu().numpy().tolist())
            scores.extend(torch.sigmoid(outputs["image_logit"].float()).view(-1).cpu().numpy().tolist())
    return labels, scores


def _collect_real_export_samples(cfgs, components, fusion_model, subset, cache_prefix, positive_only=True):
    fusion_cfg = _get_section(cfgs, "Fusion")
    amp_enabled = _amp_enabled(fusion_cfg, components["device"])
    loader = _build_real_eval_loader(cfgs, subset)
    samples = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="{}_vis".format(subset)):
            label = int(batch["label_img"].cpu().numpy()[0])
            if positive_only and label != 1:
                continue
            features = _get_or_compute_features(
                batch,
                cfgs,
                components,
                amp_enabled=amp_enabled,
                feature_cache=components["feature_cache"],
                cache_prefix=cache_prefix,
            )
            fusion_inputs, image_aux = _build_fusion_inputs(features)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(fusion_inputs, image_aux)
            _, fusion_map = _fusion_segmentation_probabilities(cfgs, None, outputs["seg_logits"])
            samples.append({
                "img_id": batch["img_id"][0],
                "label": label,
                "image": features["x64"][0, 0].cpu().numpy(),
                "fusion_map": fusion_map[0, 0].cpu().numpy(),
                "fusion_score": float(torch.sigmoid(outputs["image_logit"].float()).view(-1).cpu().numpy()[0]),
            })
    return samples


def _render_real_vis_composite(
    save_path,
    sample,
    localization,
    real_box_threshold,
    vis_threshold,
    gt_boxes=None,
    gt_mask=None,
):
    image = sample["image"]
    fusion_map = sample["fusion_map"]
    fusion_score = float(sample["fusion_score"])
    heatmap = _heatmap_to_rgb(fusion_map)
    image_rgb = _to_rgb_image(image)
    overlay = np.clip(0.65 * image_rgb + 0.35 * heatmap, 0.0, 1.0)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    for axis in axes:
        axis.axis("off")

    axes[0].imshow(image, cmap="gray")
    if gt_mask is not None and np.any(gt_mask):
        axes[0].imshow(np.ma.masked_where(np.asarray(gt_mask) <= 0, gt_mask), cmap="Greens", alpha=0.25)
    axes[0].set_title("Original\n{}".format(sample["img_id"]), fontsize=10)

    axes[1].imshow(overlay)
    axes[1].set_title(
        "Heatmap\nscore={:.3f} peak={:.3f}".format(fusion_score, float(localization["peak_score"])),
        fontsize=10,
    )

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(heatmap, alpha=0.20)
    if gt_mask is not None and np.any(gt_mask):
        axes[2].imshow(np.ma.masked_where(np.asarray(gt_mask) <= 0, gt_mask), cmap="Greens", alpha=0.18)
    axes[2].set_title(
        "Pred boxes\ngate={} box_thr={:.3f} vis_thr={:.2f}".format(
            str(bool(localization["passed_gate"])),
            float(real_box_threshold),
            float(vis_threshold),
        ),
        fontsize=10,
    )
    if localization["box_count"] == 0:
        axes[2].text(
            0.5,
            0.5,
            "NO BOX\n{}".format(localization["no_box_reason"]),
            color="red",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="red"),
        )
    else:
        for box in localization["boxes"]:
            axes[2].add_patch(Rectangle(
                (box["bbox_x0"], box["bbox_y0"]),
                box["bbox_x1"] - box["bbox_x0"] + 1,
                box["bbox_y1"] - box["bbox_y0"] + 1,
                linewidth=2.0,
                edgecolor="red",
                facecolor="none",
            ))
            axes[2].scatter(box["peak_x"], box["peak_y"], c="cyan", s=20, marker="x")
            axes[2].text(
                box["bbox_x0"],
                max(0, box["bbox_y0"] - 2),
                "#{} p={:.2f} a={}".format(box["rank"], box["peak_score"], box["area"]),
                color="yellow",
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.45, pad=1.5),
            )

    if gt_boxes is not None:
        for gt_box in gt_boxes:
            axes[2].add_patch(Rectangle(
                (gt_box["bbox_x0"], gt_box["bbox_y0"]),
                gt_box["bbox_x1"] - gt_box["bbox_x0"] + 1,
                gt_box["bbox_y1"] - gt_box["bbox_y0"] + 1,
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
                linestyle="--",
            ))

    figure.tight_layout()
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def _evaluate_real_image_level(cfgs, components, fusion_model):
    fusion_cfg = _get_section(cfgs, "Fusion")
    ensemble_cfg = _get_ensemble_cfg(cfgs)
    amp_enabled = _amp_enabled(fusion_cfg, components["device"])
    topk_ratio = ensemble_cfg["real_score_topk_ratio"]
    loader = _build_real_eval_loader(cfgs, "real_test")
    ddad_scores, diff_scores, clip_scores, fusion_scores, labels = [], [], [], [], []
    print("=> Real image-level DDAD/Diffusion score uses raw-map top-{:.2%} mean".format(topk_ratio))

    with torch.no_grad():
        for batch in tqdm(loader, desc="real_eval"):
            features = _get_or_compute_features(
                batch,
                cfgs,
                components,
                amp_enabled=amp_enabled,
                feature_cache=components["feature_cache"],
                cache_prefix="real_test",
            )
            fusion_inputs, image_aux = _build_fusion_inputs(features)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(fusion_inputs, image_aux)
            fusion_map = torch.softmax(outputs["seg_logits"].float(), dim=1)[:, 1:2]

            labels.extend(batch["label_img"].cpu().numpy().tolist())
            ddad_scores.extend(_topk_mean_score(features["ddad_map_raw"], topk_ratio).cpu().numpy().tolist())
            diff_scores.extend(_topk_mean_score(features["diff_map_raw"], topk_ratio).cpu().numpy().tolist())
            clip_scores.extend(torch.sigmoid(features["clip_global_logit"]).view(-1).cpu().numpy().tolist())
            fusion_score = float(torch.sigmoid(outputs["image_logit"].float()).view(-1).cpu().numpy()[0])
            fusion_scores.append(fusion_score)

    metrics = {}
    for name, scores in [
        ("ddad", ddad_scores),
        ("diffusion", diff_scores),
        ("clip", clip_scores),
        ("fusion", fusion_scores),
    ]:
        metrics[name] = {
            "auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.0,
            "ap": float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else 0.0,
        }
    return metrics


def export_real_visualizations(cfgs, gt_boxes_by_img_id=None):
    device = _get_device(cfgs)
    components = _gather_components(cfgs, device)
    fusion_model = _build_fusion_model(cfgs, device)
    fusion_path = os.path.join(cfgs["Exp"]["out_dir"], "fusion", "fusion_best.pth")
    _load_checkpoint(fusion_path, fusion_model, map_location=device)
    fusion_model.eval()

    visual_cfg = _get_fusion_visual_cfg(cfgs)
    real_metrics = _evaluate_real_image_level(cfgs, components, fusion_model)
    synthetic_metrics = _evaluate_synthetic_pixel_level(cfgs, components, fusion_model)
    real_val_labels, real_val_scores = _collect_real_image_scores(
        cfgs,
        components,
        fusion_model,
        subset="real_val",
        cache_prefix="real_val_export",
    )
    real_box_threshold = _choose_real_box_threshold(
        real_val_labels,
        real_val_scores,
        default_threshold=0.5,
    )
    print("=> Real localization gating threshold (Youden J on real_val): {:.4f}".format(real_box_threshold))

    positive_samples = _collect_real_export_samples(
        cfgs,
        components,
        fusion_model,
        subset="real_test",
        cache_prefix="real_test_export",
        positive_only=True,
    )
    export_num_images = max(1, int(visual_cfg["export_num_images"]))
    rng = np.random.default_rng(int(visual_cfg["export_seed"]))
    if len(positive_samples) > export_num_images:
        selected_indices = sorted(rng.choice(len(positive_samples), size=export_num_images, replace=False).tolist())
        selected_samples = [positive_samples[index] for index in selected_indices]
    else:
        selected_samples = positive_samples

    export_root = _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "real_vis_{}".format(export_num_images)))
    composites_dir = _ensure_dir(os.path.join(export_root, "composites"))
    metadata_rows = []
    boxes_payload = {}
    for index, sample in enumerate(selected_samples, start=1):
        localization = _extract_localization_boxes(
            sample["fusion_map"],
            threshold=visual_cfg["vis_threshold"],
            fusion_score=sample["fusion_score"],
            real_box_threshold=real_box_threshold,
            min_region_area=visual_cfg["min_region_area"],
            max_region_area_ratio=visual_cfg["max_region_area_ratio"],
            peak_split_threshold_ratio=visual_cfg["peak_split_threshold_ratio"],
            max_boxes=visual_cfg["max_boxes"],
            nms_iou_threshold=visual_cfg["box_nms_iou"],
        )
        sample_gt_boxes = None if gt_boxes_by_img_id is None else gt_boxes_by_img_id.get(sample["img_id"])
        file_name = "{:02d}_{}.png".format(index, _sanitize_sample_id(sample["img_id"]))
        _render_real_vis_composite(
            os.path.join(composites_dir, file_name),
            sample,
            localization,
            real_box_threshold,
            visual_cfg["vis_threshold"],
            gt_boxes=sample_gt_boxes,
        )
        metadata_rows.append({
            "img_id": str(sample["img_id"]),
            "label": int(sample["label"]),
            "fusion_score": float(sample["fusion_score"]),
            "passed_gate": bool(localization["passed_gate"]),
            "box_count": int(localization["box_count"]),
            "peak_score": float(localization["peak_score"]),
            "no_box_reason": str(localization["no_box_reason"]),
        })
        boxes_payload[str(sample["img_id"])] = localization["boxes"]

    pd.DataFrame(
        metadata_rows,
        columns=["img_id", "label", "fusion_score", "passed_gate", "box_count", "peak_score", "no_box_reason"],
    ).to_csv(os.path.join(export_root, "metadata.csv"), index=False)
    with open(os.path.join(export_root, "boxes.json"), "w") as f:
        json.dump(boxes_payload, f, indent=2)
    with open(os.path.join(export_root, "thresholds.json"), "w") as f:
        json.dump({
            "real_box_threshold": float(real_box_threshold),
            "vis_threshold": float(visual_cfg["vis_threshold"]),
            "min_region_area": int(visual_cfg["min_region_area"]),
            "max_region_area_ratio": float(visual_cfg["max_region_area_ratio"]),
            "peak_split_threshold_ratio": float(visual_cfg["peak_split_threshold_ratio"]),
            "box_nms_iou": float(visual_cfg["box_nms_iou"]),
            "max_boxes": int(visual_cfg["max_boxes"]),
            "random_seed": int(visual_cfg["export_seed"]),
        }, f, indent=2)
    with open(os.path.join(export_root, "analysis_summary.json"), "w") as f:
        json.dump({
            "real_image_level_metrics": real_metrics,
            "synthetic_pixel_level_metrics": synthetic_metrics,
            "checkpoint": fusion_path,
            "selected_images": [str(sample["img_id"]) for sample in selected_samples],
            "config_summary": {
                "vis_threshold": float(visual_cfg["vis_threshold"]),
                "min_region_area": int(visual_cfg["min_region_area"]),
                "max_region_area_ratio": float(visual_cfg["max_region_area_ratio"]),
                "peak_split_threshold_ratio": float(visual_cfg["peak_split_threshold_ratio"]),
                "box_nms_iou": float(visual_cfg["box_nms_iou"]),
                "max_boxes": int(visual_cfg["max_boxes"]),
                "real_box_threshold": float(real_box_threshold),
            },
            "diagnosis": [
                "real_image_level_metrics use image-level scores and do not directly measure localization quality.",
                "The fusion real branch is trained with image-level BCE only and has no real pixel-level supervision.",
                "Synthetic Dice is computed against synthetic masks, not real lesion boxes or masks.",
                "Connected-component filtering and image-level gating are applied only in export_real_vis and do not change training.",
            ],
        }, f, indent=2)
    print("=> Exported {} real-test composites to {}".format(len(selected_samples), export_root))


def _evaluate_synthetic_pixel_level(cfgs, components, fusion_model):
    fusion_cfg = _get_section(cfgs, "Fusion")
    amp_enabled = _amp_enabled(fusion_cfg, components["device"])
    loader = _build_multibranch_loader(
        cfgs,
        subset="synthetic_test",
        batch_size=1,
        shuffle=False,
        synthetic_probability=float(_get_section(cfgs, "Fusion").get("synthetic_probability", 0.5)),
        drop_last=False,
    )
    collectors = {
        "ddad": {"maps": [], "masks": []},
        "diffusion": {"maps": [], "masks": []},
        "clip": {"maps": [], "masks": []},
        "fusion": {"maps": [], "masks": []},
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="synthetic_eval"):
            features = _get_or_compute_features(
                batch,
                cfgs,
                components,
                amp_enabled=amp_enabled,
                feature_cache=components["feature_cache"],
                cache_prefix="synthetic_test",
            )
            fusion_inputs, image_aux = _build_fusion_inputs(features)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(fusion_inputs, image_aux)
            _, fusion_map = _fusion_segmentation_probabilities(cfgs, None, outputs["seg_logits"])
            mask = batch["mask_syn"].cpu().numpy()

            collectors["ddad"]["maps"].extend(features["ddad_map"].cpu().numpy())
            collectors["diffusion"]["maps"].extend(features["diff_map"].cpu().numpy())
            collectors["clip"]["maps"].extend(features["clip_map"].cpu().numpy())
            collectors["fusion"]["maps"].extend(fusion_map.cpu().numpy())
            for key in collectors:
                collectors[key]["masks"].extend(mask)

    metrics = {}
    for key, value in collectors.items():
        auc, dice = _synthetic_pixel_metrics(value["maps"], value["masks"])
        metrics[key] = {"pixel_auc": auc, "dice": dice}
    return metrics


def evaluate_all_modules(cfgs):
    device = _get_device(cfgs)
    band_cfg = _get_fusion_band_cfg(cfgs)
    components = _gather_components(cfgs, device)
    fusion_model = _build_fusion_model(cfgs, device)
    fusion_path = os.path.join(cfgs["Exp"]["out_dir"], "fusion", "fusion_best.pth")
    _load_checkpoint(fusion_path, fusion_model, map_location=device)
    fusion_model.eval()
    print(
        "=> Evaluating fusion: band_mode={} vector_dim={} band_score_topk_ratio={:.2%}".format(
            band_cfg["band_mode"],
            _fusion_vector_dim(cfgs),
            band_cfg["band_score_topk_ratio"],
        )
    )

    real_metrics = _evaluate_real_image_level(cfgs, components, fusion_model)
    synthetic_metrics = _evaluate_synthetic_pixel_level(cfgs, components, fusion_model)

    results = {
        "real_image_level_metrics": real_metrics,
        "synthetic_pixel_level_metrics": synthetic_metrics,
    }

    result_path = os.path.join(cfgs["Exp"]["out_dir"], "eval_all_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for branch_name, branch_metrics in real_metrics.items():
        rows.append({
            "split": "real_image_level",
            "branch": branch_name,
            "auc": branch_metrics["auc"],
            "ap_or_dice": branch_metrics["ap"],
        })
    for branch_name, branch_metrics in synthetic_metrics.items():
        rows.append({
            "split": "synthetic_pixel_level",
            "branch": branch_name,
            "auc": branch_metrics["pixel_auc"],
            "ap_or_dice": branch_metrics["dice"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(cfgs["Exp"]["out_dir"], "eval_all_results.csv"), index=False)

    print(json.dumps(results, indent=2))


def _safe_average_precision_score(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    valid_mask = np.isfinite(scores)
    if valid_mask.sum() == 0:
        return 0.0
    filtered_labels = labels[valid_mask]
    filtered_scores = scores[valid_mask]
    if len(np.unique(filtered_labels)) < 2:
        return 0.0
    return float(average_precision_score(filtered_labels, filtered_scores))


def _classification_metrics(labels, scores):
    return {
        "auc": _safe_roc_auc_score(labels, scores),
        "ap": _safe_average_precision_score(labels, scores),
    }


def _branch_scalar_score(score_map, topk_ratio):
    return _topk_mean_score(score_map.float(), topk_ratio).view(-1, 1)


def _build_clip_model(cfgs, device):
    clip_cfg = _get_section(cfgs, "CLIP")
    model = BiomedCLIPBranch(
        image_size=int(_get_data_cfg(cfgs).get("clip_img_size", 336)),
        backbone_name=str(clip_cfg.get("backbone_name", "ViT-L-14-336")),
        pretrained=str(clip_cfg.get("pretrained", "openai")),
        feature_layers=list(clip_cfg.get("feature_layers", [6, 12, 18, 24])),
        prompt_token_count=int(clip_cfg.get("prompt_token_count", 4)),
        seg_adapter_dim=int(clip_cfg.get("seg_adapter_dim", 512)),
        det_adapter_dim=int(clip_cfg.get("det_adapter_dim", 512)),
        prompt_ensemble=str(clip_cfg.get("prompt_ensemble", "rsna_chest_xray")),
        normal_prompts=clip_cfg.get("normal_prompts"),
        abnormal_prompts=clip_cfg.get("abnormal_prompts"),
        use_dap_pooling=bool(clip_cfg.get("use_dap_pooling", False)),
        dap_alpha=float(clip_cfg.get("dap_alpha", 0.5)),
        dap_detach_map=bool(clip_cfg.get("dap_detach_map", True)),
        dap_temperature=float(clip_cfg.get("dap_temperature", 1.0)),
        offline_pretrained=bool(clip_cfg.get("offline_pretrained", False)),
    )
    return model.to(device)


def _configure_clip_trainable_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

    for module in [model.det_adapters, model.seg_adapters]:
        for param in module.parameters():
            param.requires_grad = True

    model.normal_prompt_tokens.requires_grad = True
    model.abnormal_prompt_tokens.requires_grad = True
    model.logit_scale.requires_grad = True


def _fusion_vector_dim(cfgs):
    band_cfg = _get_fusion_band_cfg(cfgs)
    clip_dim = 1 if band_cfg["include_clip_score_in_vector"] else 0
    if band_cfg["use_safd"]:
        return clip_dim + _fusion_raw_map_count(band_cfg) * band_cfg["safd_levels"]
    ddad_dim = 0
    diffusion_dim = 0
    if band_cfg["enable_ddad_in_fusion"]:
        ddad_dim = 6 if band_cfg["use_fixed_3band"] else 1
    if band_cfg["enable_diffusion_in_fusion"]:
        diffusion_dim = 6 if band_cfg["use_fixed_3band"] else 1
    return clip_dim + ddad_dim + diffusion_dim


def _get_fusion_band_cfg(cfgs):
    fusion_cfg = _get_section(cfgs, "Fusion")
    band_mode = str(fusion_cfg.get("band_mode", "none")).strip().lower()
    if band_mode not in {"none", "fixed_3band", "safd"}:
        raise ValueError("Unsupported Fusion.band_mode: {}".format(band_mode))

    include_clip_score_in_vector = bool(fusion_cfg.get("include_clip_score_in_vector", True))
    use_clip_anchor = bool(fusion_cfg.get("use_clip_anchor", True))
    clip_branch_dropout_prob = float(fusion_cfg.get("clip_branch_dropout_prob", 0.0))
    if not 0.0 <= clip_branch_dropout_prob < 1.0:
        raise ValueError(
            "Fusion.clip_branch_dropout_prob must be in [0, 1), got {}".format(clip_branch_dropout_prob)
        )

    enable_ddad_in_fusion = bool(fusion_cfg.get("enable_ddad_in_fusion", True))
    enable_diffusion_in_fusion = bool(fusion_cfg.get("enable_diffusion_in_fusion", True))
    if not include_clip_score_in_vector and not enable_ddad_in_fusion and not enable_diffusion_in_fusion:
        raise ValueError(
            "Fusion has no learnable inputs. Enable at least one of include_clip_score_in_vector, "
            "enable_ddad_in_fusion, or enable_diffusion_in_fusion."
        )
    if band_mode == "safd" and not (enable_ddad_in_fusion or enable_diffusion_in_fusion):
        raise ValueError("Fusion.band_mode='safd' requires DDAD and/or diffusion maps to be enabled.")

    band_score_topk_ratio = fusion_cfg.get("band_score_topk_ratio")
    if band_score_topk_ratio is None:
        band_score_topk_ratio = _get_ensemble_cfg(cfgs)["real_score_topk_ratio"]
    band_score_topk_ratio = float(band_score_topk_ratio)
    if not 0.0 < band_score_topk_ratio <= 1.0:
        raise ValueError(
            "Fusion.band_score_topk_ratio must be in (0, 1], got {}".format(band_score_topk_ratio)
        )

    safd_levels = int(fusion_cfg.get("safd_levels", 3))
    safd_patch_size = int(fusion_cfg.get("safd_patch_size", 4))
    safd_lambda_repulsion = float(fusion_cfg.get("safd_lambda_repulsion", 1.0e-8))
    safd_topk_ratio = fusion_cfg.get("safd_topk_ratio", band_score_topk_ratio)
    safd_topk_ratio = float(safd_topk_ratio)
    if not 0.0 < safd_topk_ratio <= 1.0:
        raise ValueError("Fusion.safd_topk_ratio must be in (0, 1], got {}".format(safd_topk_ratio))
    safd_repulsion_weight = float(fusion_cfg.get("safd_repulsion_weight", 1.0))

    return {
        "band_mode": band_mode,
        "use_fixed_3band": band_mode == "fixed_3band",
        "use_safd": band_mode == "safd",
        "include_clip_score_in_vector": include_clip_score_in_vector,
        "use_clip_anchor": use_clip_anchor,
        "clip_branch_dropout_prob": clip_branch_dropout_prob,
        "enable_ddad_in_fusion": enable_ddad_in_fusion,
        "enable_diffusion_in_fusion": enable_diffusion_in_fusion,
        "band_score_topk_ratio": band_score_topk_ratio,
        "safd_levels": safd_levels,
        "safd_patch_size": safd_patch_size,
        "safd_lambda_repulsion": safd_lambda_repulsion,
        "safd_topk_ratio": safd_topk_ratio,
        "safd_repulsion_weight": safd_repulsion_weight,
    }


def _multi_band_score_vector(score_map, ratio):
    low_band, mid_band, high_band = fixed_three_band_decomposition(score_map)
    return torch.cat(
        [
            _branch_scalar_score(low_band, ratio),
            _branch_scalar_score(mid_band, ratio),
            _branch_scalar_score(high_band, ratio),
        ],
        dim=1,
    )


def _fusion_raw_map_order():
    return [
        "ddad_inter_img",
        "ddad_intra_img",
        "diff_residual_img",
        "diff_inter_img",
    ]


def _selected_fusion_raw_map_names(band_cfg):
    selected_names = []
    full_order = _fusion_raw_map_order()
    if band_cfg["enable_ddad_in_fusion"]:
        selected_names.extend(full_order[:2])
    if band_cfg["enable_diffusion_in_fusion"]:
        selected_names.extend(full_order[2:])
    return selected_names


def _fusion_raw_map_count(band_cfg=None):
    if band_cfg is None:
        return len(_fusion_raw_map_order())
    return len(_selected_fusion_raw_map_names(band_cfg))


def _raw_fusion_map_chunks(raw_fusion_maps):
    return dict(
        zip(
            _fusion_raw_map_order(),
            torch.chunk(raw_fusion_maps.float(), chunks=_fusion_raw_map_count(), dim=1),
        )
    )


def _select_raw_fusion_maps(raw_fusion_maps, band_cfg):
    raw_map_chunks = _raw_fusion_map_chunks(raw_fusion_maps)
    selected_names = _selected_fusion_raw_map_names(band_cfg)
    if len(selected_names) == 0:
        raise ValueError("At least one raw fusion branch must be enabled.")
    return torch.cat([raw_map_chunks[name] for name in selected_names], dim=1)


def _build_fusion_vector_from_raw_maps(raw_fusion_maps, clip_global_logit, band_cfg, branch_topk_ratio):
    raw_map_chunks = _raw_fusion_map_chunks(raw_fusion_maps)
    ddad_inter_img = raw_map_chunks["ddad_inter_img"]
    ddad_intra_img = raw_map_chunks["ddad_intra_img"]
    diff_residual_img = raw_map_chunks["diff_residual_img"]
    diff_inter_img = raw_map_chunks["diff_inter_img"]
    clip_global_logit = clip_global_logit.float().view(raw_fusion_maps.size(0), 1)

    fusion_parts = []
    if band_cfg["include_clip_score_in_vector"]:
        fusion_parts.append(clip_global_logit)

    if band_cfg["enable_ddad_in_fusion"]:
        if band_cfg["use_fixed_3band"]:
            ddad_vector = torch.cat(
                [
                    _multi_band_score_vector(ddad_inter_img, band_cfg["band_score_topk_ratio"]),
                    _multi_band_score_vector(ddad_intra_img, band_cfg["band_score_topk_ratio"]),
                ],
                dim=1,
            )
        else:
            ddad_vector = 0.5 * (
                _branch_scalar_score(ddad_inter_img, branch_topk_ratio) +
                _branch_scalar_score(ddad_intra_img, branch_topk_ratio)
            )
        fusion_parts.append(ddad_vector)

    if band_cfg["enable_diffusion_in_fusion"]:
        if band_cfg["use_fixed_3band"]:
            diffusion_vector = torch.cat(
                [
                    _multi_band_score_vector(diff_residual_img, band_cfg["band_score_topk_ratio"]),
                    _multi_band_score_vector(diff_inter_img, band_cfg["band_score_topk_ratio"]),
                ],
                dim=1,
            )
        else:
            diffusion_vector = 0.5 * (
                _branch_scalar_score(diff_residual_img, branch_topk_ratio) +
                _branch_scalar_score(diff_inter_img, branch_topk_ratio)
            )
        fusion_parts.append(diffusion_vector)

    if len(fusion_parts) == 0:
        raise RuntimeError("Fusion vector would be empty. Please enable at least one fusion feature source.")

    fusion_vector = torch.cat(fusion_parts, dim=1)
    return fusion_vector.float()


def _build_fusion_model(cfgs, device):
    fusion_cfg = _get_section(cfgs, "Fusion")
    band_cfg = _get_fusion_band_cfg(cfgs)
    inferred_in_channels = _fusion_vector_dim(cfgs)
    configured_in_channels = fusion_cfg.get("in_channels")
    if isinstance(configured_in_channels, str) and configured_in_channels.strip().lower() in {"auto", "infer"}:
        configured_in_channels = None
    if configured_in_channels is not None and int(configured_in_channels) != inferred_in_channels:
        raise ValueError(
            "Fusion in_channels mismatch: config has {}, but runtime expects {}.".format(
                configured_in_channels,
                inferred_in_channels,
            )
        )
    return FusionRefineNet(
        in_channels=inferred_in_channels,
        out_channels=1,
        base_channels=int(fusion_cfg.get("base_channels", 128)),
        aux_in_channels=0,
        dropout=float(fusion_cfg.get("dropout", 0.1)),
        band_mode=band_cfg["band_mode"],
        safd_levels=band_cfg["safd_levels"],
        safd_patch_size=band_cfg["safd_patch_size"],
        safd_lambda_repulsion=band_cfg["safd_lambda_repulsion"],
        safd_topk_ratio=band_cfg["safd_topk_ratio"],
        safd_raw_in_channels=_fusion_raw_map_count(band_cfg),
        include_clip_score=band_cfg["include_clip_score_in_vector"],
    ).to(device)


def _build_fusion_inputs(cfgs, feature_source):
    band_cfg = _get_fusion_band_cfg(cfgs)
    branch_topk_ratio = _get_ensemble_cfg(cfgs)["real_score_topk_ratio"]
    raw_fusion_maps = feature_source["raw_fusion_maps"].float()
    clip_global_logit = feature_source["clip_global_logit"].float().view(raw_fusion_maps.size(0), 1)
    clip_anchor = feature_source["clip_anchor"].float().view(raw_fusion_maps.size(0), 1)

    if band_cfg["use_safd"]:
        fusion_inputs = {
            "raw_maps": _select_raw_fusion_maps(raw_fusion_maps, band_cfg),
        }
        if band_cfg["include_clip_score_in_vector"]:
            fusion_inputs["clip_global_logit"] = clip_global_logit
        if band_cfg["use_clip_anchor"]:
            fusion_inputs["clip_anchor"] = clip_anchor
        return fusion_inputs

    fusion_vector = _build_fusion_vector_from_raw_maps(
        raw_fusion_maps,
        clip_global_logit,
        band_cfg,
        branch_topk_ratio,
    )
    fusion_inputs = {
        "fusion_vector": fusion_vector,
    }
    if band_cfg["use_clip_anchor"]:
        fusion_inputs["clip_anchor"] = clip_anchor
    return fusion_inputs


def _build_cached_fusion_inputs(cfgs, batch, device):
    band_cfg = _get_fusion_band_cfg(cfgs)

    clip_anchor = batch.get("clip_anchor")
    if clip_anchor is None:
        raise RuntimeError(
            "Cached fusion sample is missing clip_anchor. Please regenerate fusion cache with the current code."
        )
    clip_anchor = _move_tensor(clip_anchor.float(), device).view(-1, 1)

    raw_fusion_maps = batch.get("raw_fusion_maps")
    clip_global_logit = batch.get("clip_global_logit")
    if raw_fusion_maps is not None and clip_global_logit is not None:
        cache_features = {
            "raw_fusion_maps": _move_tensor(raw_fusion_maps.float(), device),
            "clip_global_logit": _move_tensor(clip_global_logit.float(), device).view(-1, 1),
            "clip_anchor": clip_anchor,
        }
        return _build_fusion_inputs(cfgs, cache_features)

    if band_cfg["use_safd"]:
        raise RuntimeError(
            "Cached fusion samples do not contain raw_fusion_maps / clip_global_logit required by Fusion.band_mode='safd'. "
            "Please rerun `python main.py --config <your_config> --mode cache_fusion`."
        )

    if band_cfg["use_fixed_3band"]:
        raise RuntimeError(
            "Cached fusion samples do not contain raw_fusion_maps needed to rebuild fixed 3-band fusion inputs. "
            "Please rerun `python main.py --config <your_config> --mode cache_fusion`."
        )

    fusion_vector = batch.get("fusion_vector")
    if fusion_vector is None:
        raise RuntimeError(
            "Cached fusion sample is missing fusion_vector. Please regenerate fusion cache with the current code."
        )
    cached_inputs = {
        "fusion_vector": _move_tensor(fusion_vector.float(), device),
    }
    if band_cfg["use_clip_anchor"]:
        cached_inputs["clip_anchor"] = clip_anchor
    return cached_inputs


def _apply_fusion_clip_dropout(fusion_inputs, band_cfg):
    dropout_prob = band_cfg["clip_branch_dropout_prob"]
    if dropout_prob <= 0.0:
        return fusion_inputs

    reference = fusion_inputs.get("fusion_vector")
    if reference is None:
        reference = fusion_inputs.get("raw_maps")
    if reference is None:
        reference = fusion_inputs.get("clip_anchor")
    if reference is None:
        reference = fusion_inputs.get("clip_global_logit")
    if reference is None:
        return fusion_inputs

    keep_mask = (torch.rand(reference.size(0), 1, device=reference.device) >= dropout_prob).to(reference.dtype)
    dropped_inputs = dict(fusion_inputs)
    if band_cfg["include_clip_score_in_vector"] and "fusion_vector" in dropped_inputs:
        dropped_inputs["fusion_vector"] = dropped_inputs["fusion_vector"].clone()
        dropped_inputs["fusion_vector"][:, :1] = dropped_inputs["fusion_vector"][:, :1] * keep_mask
    if "clip_global_logit" in dropped_inputs:
        dropped_inputs["clip_global_logit"] = dropped_inputs["clip_global_logit"] * keep_mask
    if "clip_anchor" in dropped_inputs:
        dropped_inputs["clip_anchor"] = dropped_inputs["clip_anchor"] * keep_mask
    return dropped_inputs


def _compute_ddad_branch_features(x64, cfgs, module_a, module_b):
    network = cfgs["Model"]["network"]
    topk_ratio = _get_ensemble_cfg(cfgs)["real_score_topk_ratio"]
    outputs_a, outputs_b = [], []
    uncertainties = []

    for model in module_a:
        reconstruction, _ = _module_forward(model, network, x64)
        outputs_a.append(reconstruction)

    for model in module_b:
        reconstruction, uncertainty = _module_forward(model, network, x64)
        outputs_b.append(reconstruction)
        if uncertainty is not None:
            uncertainties.append(uncertainty)

    outputs_a = torch.stack(outputs_a, dim=0)
    outputs_b = torch.stack(outputs_b, dim=0)

    mu_a = outputs_a.mean(dim=0)
    mu_b = outputs_b.mean(dim=0)
    inter_img = torch.abs(mu_a - mu_b)
    intra_img = _safe_model_std(outputs_b, dim=0)

    if network == "AE-U" and len(uncertainties) > 0:
        uncertainty_map = torch.stack(uncertainties, dim=0).mean(dim=0)
        uncertainty_map = torch.sqrt(uncertainty_map + 1.0e-6)
        inter_img = inter_img / uncertainty_map
        intra_img = intra_img / uncertainty_map

    inter_score = _branch_scalar_score(inter_img, topk_ratio)
    intra_score = _branch_scalar_score(intra_img, topk_ratio)
    return {
        "score": 0.5 * (inter_score + intra_score),
        "inter_score": inter_score,
        "intra_score": intra_score,
        "inter_img": inter_img.float(),
        "intra_img": intra_img.float(),
    }


def _build_weak_refine_model(cfgs, device, refine_in):
    refine_runtime_cfg = get_refine_runtime_cfg(cfgs)
    refine_in_channels = infer_refine_in_channels(
        refine_in,
        use_fixed_3band=refine_runtime_cfg["use_fixed_3band"],
    )
    model = get_model(network="refine", in_channels=refine_in_channels, out_channels=2)
    return model.to(device), refine_runtime_cfg, refine_in_channels


def _load_weak_refine_model(cfgs, device, refine_in):
    model, refine_runtime_cfg, refine_in_channels = _build_weak_refine_model(cfgs, device, refine_in)
    checkpoint_path = _weak_refine_checkpoint_path(cfgs, refine_in)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Expected weak refine checkpoint at {}. Run mode weak_r before score_unlabeled, "
            "or set WeakCLIP.use_refine_score=false.".format(checkpoint_path)
        )
    _load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return model, refine_runtime_cfg, refine_in_channels


def _compute_refined_ddad_features(x64, cfgs, module_a, module_b, refine_net, refine_in, refine_runtime_cfg):
    ddad_features = _compute_ddad_branch_features(x64, cfgs, module_a, module_b)
    net_in = build_refine_input(
        ddad_features["inter_img"],
        ddad_features["intra_img"],
        refine_in,
        use_fixed_3band=refine_runtime_cfg["use_fixed_3band"],
    )
    logits = refine_net(net_in)
    prob_map = torch.softmax(logits, dim=1)[:, 1:, :, :].float()
    refined_score = _topk_mean_score(prob_map, refine_runtime_cfg["score_topk_ratio"]).view(-1, 1)
    outputs = dict(ddad_features)
    outputs.update({
        "refined_score": refined_score,
        "refined_map": prob_map,
    })
    return outputs


def _normalize_maps_per_sample(score_map):
    if score_map.dim() != 4:
        raise ValueError("Expected score_map with shape [B, C, H, W], got {}".format(tuple(score_map.shape)))
    flat = score_map.float().view(score_map.size(0), score_map.size(1), -1)
    mins = flat.min(dim=2).values.view(score_map.size(0), score_map.size(1), 1, 1)
    maxs = flat.max(dim=2).values.view(score_map.size(0), score_map.size(1), 1, 1)
    return (score_map.float() - mins) / (maxs - mins + 1.0e-6)


def _build_weak_safd_decomposer(cfgs, device):
    weak_cfg = _get_weakclip_cfg(cfgs)
    score_mode = weak_cfg["safd_score_mode"]
    if score_mode not in {"max", "mean", "normal_bank"}:
        raise ValueError("WeakCLIP.safd_score_mode must be 'max', 'mean', or 'normal_bank', got '{}'".format(score_mode))
    normal_reduce = weak_cfg["safd_normal_reduce"]
    if normal_reduce not in {"max", "mean"}:
        raise ValueError("WeakCLIP.safd_normal_reduce must be 'max' or 'mean', got '{}'".format(normal_reduce))
    if weak_cfg["safd_fusion_mode"] not in {"linear", "agreement"}:
        raise ValueError("WeakCLIP.safd_fusion_mode must be 'linear' or 'agreement', got '{}'".format(weak_cfg["safd_fusion_mode"]))
    if weak_cfg["safd_apply_scope"] not in {"global", "topup_only"}:
        raise ValueError("WeakCLIP.safd_apply_scope must be 'global' or 'topup_only', got '{}'".format(weak_cfg["safd_apply_scope"]))
    rng_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(weak_cfg["safd_seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed(weak_cfg["safd_seed"])
        decomposer = SemanticBandDecomposer(
            n_levels=weak_cfg["safd_levels"],
            patch_size=weak_cfg["safd_patch_size"],
            lambda_repulsion=weak_cfg["safd_lambda_repulsion"],
        )
    return decomposer.to(device).eval()


def _weak_safd_raw_maps_from_features(features):
    raw_maps = torch.cat(
        [
            features["refined_map"].float(),
            _normalize_maps_per_sample(features["inter_img"].float()),
            _normalize_maps_per_sample(features["intra_img"].float()),
        ],
        dim=1,
    )
    return raw_maps


def _compute_weak_safd_coefficients(features, safd_decomposer):
    raw_maps = _weak_safd_raw_maps_from_features(features)
    _, coeff_maps, _ = safd_decomposer(raw_maps)
    return coeff_maps.float()


def _compute_weak_safd_score(features, safd_decomposer, weak_cfg):
    raw_maps = _weak_safd_raw_maps_from_features(features)
    band_maps, _, _ = safd_decomposer(raw_maps)
    score_maps = band_maps.permute(0, 2, 1, 3, 4).contiguous()
    batch_size, channels, levels, height, width = score_maps.shape
    flattened_maps = score_maps.view(batch_size * channels * levels, 1, height, width)
    band_scores = _topk_mean_score(flattened_maps, weak_cfg["safd_topk_ratio"]).view(batch_size, channels * levels)
    if weak_cfg["safd_score_mode"] == "mean":
        safd_score = band_scores.mean(dim=1, keepdim=True)
    else:
        safd_score = band_scores.max(dim=1, keepdim=True).values
    return safd_score.float(), band_scores.float()


def _compute_weak_safd_normal_score(features, safd_decomposer, safd_bank, weak_cfg, device):
    coeff_maps = _compute_weak_safd_coefficients(features, safd_decomposer)
    median = safd_bank["median"].to(device=device, dtype=coeff_maps.dtype)
    mad = safd_bank["mad"].to(device=device, dtype=coeff_maps.dtype)
    eps = float(safd_bank.get("mad_eps", weak_cfg["safd_normal_mad_eps"]))
    if tuple(coeff_maps.shape[1:]) != tuple(median.shape):
        raise RuntimeError(
            "SAFD normal bank shape {} does not match current coefficient shape {}. "
            "Rebuild it with mode safd_bank.".format(tuple(median.shape), tuple(coeff_maps.shape[1:]))
        )
    distance = torch.abs(coeff_maps - median.unsqueeze(0)) / (mad.unsqueeze(0) + eps)
    batch_size, levels, channels, height, width = distance.shape
    flattened = distance.reshape(batch_size * levels * channels, 1, height, width)
    band_scores = _topk_mean_score(flattened, weak_cfg["safd_topk_ratio"]).view(batch_size, levels * channels)
    if weak_cfg["safd_normal_reduce"] == "mean":
        safd_score = band_scores.mean(dim=1, keepdim=True)
    else:
        safd_score = band_scores.max(dim=1, keepdim=True).values
    return safd_score.float(), band_scores.float()


def _load_weak_safd_normal_bank(cfgs, device):
    bank_path = _weakclip_safd_bank_path(cfgs)
    if not os.path.exists(bank_path):
        raise FileNotFoundError(
            "Expected SAFD normal bank at {}. Run: python main.py --config <config> --mode safd_bank".format(bank_path)
        )
    try:
        bank = torch.load(bank_path, map_location=device, weights_only=False)
    except TypeError:
        bank = torch.load(bank_path, map_location=device)
    if "median" not in bank or "mad" not in bank:
        raise RuntimeError("Invalid SAFD normal bank at {}: missing median/mad.".format(bank_path))
    return bank


def _compute_diffusion_branch_features(x64, cfgs, diff_a_model, diff_b_model):
    diff_cfg = _get_section(cfgs, "Diffusion")
    topk_ratio = _get_ensemble_cfg(cfgs)["real_score_topk_ratio"]
    t_recon = int(diff_cfg.get("t_recon", 200))
    ddim_steps = int(diff_cfg.get("ddim_steps", 50))

    recon_b = diff_b_model.reconstruct(x64, t_recon=t_recon, ddim_steps=ddim_steps)
    recon_a = diff_a_model.reconstruct(x64, t_recon=t_recon, ddim_steps=ddim_steps)

    residual_img = torch.abs(x64 - recon_b)
    inter_img = torch.abs(recon_b - recon_a)
    residual_score = _branch_scalar_score(residual_img, topk_ratio)
    inter_score = _branch_scalar_score(inter_img, topk_ratio)
    return {
        "score": 0.5 * (residual_score + inter_score),
        "residual_score": residual_score,
        "inter_score": inter_score,
        "residual_img": residual_img.float(),
        "inter_img": inter_img.float(),
    }


def _zero_diffusion_branch_features(x64):
    zero_map = torch.zeros_like(x64, dtype=torch.float32)
    zero_score = torch.zeros(x64.size(0), 1, device=x64.device, dtype=torch.float32)
    return {
        "score": zero_score,
        "residual_score": zero_score,
        "inter_score": zero_score,
        "residual_img": zero_map,
        "inter_img": zero_map,
    }


def _compute_clip_branch_features(image_224, image_64_size, clip_model, return_patch_maps=False):
    outputs = clip_model(
        image_224,
        output_size=image_64_size,
        return_patch_maps=return_patch_maps,
    )
    clip_score = outputs["global_logit"].float().view(-1, 1)
    features = {
        "score": clip_score,
        "global_logit": clip_score,
        "base_global_logit": clip_score,
    }
    if return_patch_maps and "patch_map" in outputs:
        features["patch_map"] = outputs["patch_map"].float()
    return features


def _compute_all_branch_features(batch, cfgs, components, amp_enabled=False, return_heatmaps=False):
    x64 = _move_tensor(batch["image_64"], components["device"])
    x224 = _move_tensor(batch["image_224"], components["device"])
    has_diffusion = bool(components.get("has_diffusion", False))

    with _autocast_context(amp_enabled):
        ddad_features = _compute_ddad_branch_features(
            x64,
            cfgs,
            components["module_a"],
            components["module_b"],
        )
        if has_diffusion:
            diffusion_features = _compute_diffusion_branch_features(
                x64,
                cfgs,
                components["diff_a_model"],
                components["diff_b_model"],
            )
        else:
            diffusion_features = _zero_diffusion_branch_features(x64)
        clip_features = _compute_clip_branch_features(
            x224,
            x64.shape[-2:],
            components["clip_model"],
            return_patch_maps=return_heatmaps,
        )

    raw_fusion_maps = torch.cat(
        [
            ddad_features["inter_img"].float(),
            ddad_features["intra_img"].float(),
            diffusion_features["residual_img"].float(),
            diffusion_features["inter_img"].float(),
        ],
        dim=1,
    )

    _ensure_finite_tensor("raw_fusion_maps", raw_fusion_maps)
    features = {
        "raw_fusion_maps": raw_fusion_maps.float(),
        "branch_scores": {
            "ddad": ddad_features["score"].float(),
            "diffusion": diffusion_features["score"].float(),
            "clip": clip_features["score"].float(),
        },
        "clip_anchor": clip_features["base_global_logit"].float(),
        "clip_global_logit": clip_features["global_logit"].float(),
        "clip_base_global_logit": clip_features["base_global_logit"].float(),
    }
    if return_heatmaps:
        heatmaps = {}
        if "patch_map" in clip_features:
            heatmaps["clip_patch_map"] = clip_features["patch_map"].float()
        if len(heatmaps) > 0:
            features["heatmaps"] = heatmaps
    return features


def _split_feature_batch(features):
    batch_size = features["raw_fusion_maps"].size(0)
    split_samples = []
    for index in range(batch_size):
        split_samples.append({
            "raw_fusion_maps": features["raw_fusion_maps"][index:index + 1].detach().cpu(),
            "branch_scores": {
                key: value[index:index + 1].detach().cpu()
                for key, value in features["branch_scores"].items()
            },
            "clip_anchor": features["clip_anchor"][index:index + 1].detach().cpu(),
            "clip_global_logit": features["clip_global_logit"][index:index + 1].detach().cpu(),
        })
    return split_samples


def _stack_feature_samples(samples, device):
    return {
        "raw_fusion_maps": _move_tensor(torch.cat([sample["raw_fusion_maps"] for sample in samples], dim=0), device),
        "branch_scores": {
            key: _move_tensor(torch.cat([sample["branch_scores"][key] for sample in samples], dim=0), device)
            for key in samples[0]["branch_scores"].keys()
        },
        "clip_anchor": _move_tensor(
            torch.cat([sample["clip_anchor"] for sample in samples], dim=0),
            device,
        ),
        "clip_global_logit": _move_tensor(
            torch.cat([sample["clip_global_logit"] for sample in samples], dim=0),
            device,
        ),
    }


def _get_or_compute_features(batch, cfgs, components, amp_enabled=False, feature_cache=None, cache_prefix=None):
    if feature_cache is None or cache_prefix is None:
        return _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)

    img_ids = list(batch["img_id"])
    cache_keys = [_feature_cache_key(cache_prefix, img_id) for img_id in img_ids]
    missing_keys = [cache_key for cache_key in cache_keys if cache_key not in feature_cache]

    if missing_keys:
        computed_features = _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)
        split_samples = _split_feature_batch(computed_features)
        for cache_key, sample in zip(cache_keys, split_samples):
            feature_cache[cache_key] = sample

    return _stack_feature_samples([feature_cache[cache_key] for cache_key in cache_keys], components["device"])


def _save_cached_fusion_batch(cache_dir, batch, raw_fusion_maps, clip_global_logit, clip_anchor):
    label_img = batch["label_img"].detach().cpu()
    raw_fusion_maps = raw_fusion_maps.detach().cpu()
    clip_global_logit = clip_global_logit.detach().cpu()
    clip_anchor = clip_anchor.detach().cpu()
    for index, img_id in enumerate(batch["img_id"]):
        torch.save(
            {
                "raw_fusion_maps": raw_fusion_maps[index],
                "clip_global_logit": clip_global_logit[index],
                "clip_anchor": clip_anchor[index],
                "label_img": label_img[index],
                "img_id": str(img_id),
            },
            _cached_fusion_path(cache_dir, img_id),
        )


def _cache_fusion_split(
    cfgs,
    components,
    split_name,
    subset,
    batch_size,
    synthetic_probability,
    include_clip=True,
    force_deterministic=False,
    deterministic_seed=1337,
):
    fusion_cfg = _get_section(cfgs, "Fusion")
    amp_enabled = _amp_enabled(fusion_cfg, components["device"])
    cache_dir = _ensure_dir(os.path.join(_fusion_cache_root(cfgs), split_name))

    for file_name in os.listdir(cache_dir):
        if file_name.endswith(".pt"):
            os.remove(os.path.join(cache_dir, file_name))

    loader = _build_multibranch_loader(
        cfgs,
        subset=subset,
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=synthetic_probability,
        drop_last=False,
        include_clip=include_clip,
        force_deterministic=force_deterministic,
        deterministic_seed=deterministic_seed,
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="cache_{}".format(split_name)):
            features = _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)
            _save_cached_fusion_batch(
                cache_dir,
                {
                    "label_img": batch["label_img"],
                    "img_id": batch["img_id"],
                },
                features["raw_fusion_maps"],
                features["clip_global_logit"],
                features["clip_anchor"],
            )

    return cache_dir


def _merge_feature_outputs(feature_chunks, target_device):
    if len(feature_chunks) == 1:
        single = feature_chunks[0]
        merged = {
            "raw_fusion_maps": _move_tensor(single["raw_fusion_maps"], target_device),
            "branch_scores": {
                key: _move_tensor(value, target_device) for key, value in single["branch_scores"].items()
            },
            "clip_anchor": _move_tensor(single["clip_anchor"], target_device),
            "clip_global_logit": _move_tensor(single["clip_global_logit"], target_device),
        }
        if "heatmaps" in single:
            merged["heatmaps"] = {
                key: _move_tensor(value, target_device) for key, value in single["heatmaps"].items()
            }
        return merged

    merged = {
        "raw_fusion_maps": torch.cat(
            [_move_tensor(chunk["raw_fusion_maps"], target_device) for chunk in feature_chunks],
            dim=0,
        ),
        "branch_scores": {
            key: torch.cat(
                [_move_tensor(chunk["branch_scores"][key], target_device) for chunk in feature_chunks],
                dim=0,
            )
            for key in feature_chunks[0]["branch_scores"].keys()
        },
        "clip_anchor": torch.cat(
            [_move_tensor(chunk["clip_anchor"], target_device) for chunk in feature_chunks],
            dim=0,
        ),
        "clip_global_logit": torch.cat(
            [_move_tensor(chunk["clip_global_logit"], target_device) for chunk in feature_chunks],
            dim=0,
        ),
    }
    heatmap_keys = sorted({
        key
        for chunk in feature_chunks
        if "heatmaps" in chunk
        for key in chunk["heatmaps"].keys()
    })
    if len(heatmap_keys) > 0:
        merged["heatmaps"] = {
            key: torch.cat(
                [
                    _move_tensor(chunk["heatmaps"][key], target_device)
                    for chunk in feature_chunks
                    if "heatmaps" in chunk and key in chunk["heatmaps"]
                ],
                dim=0,
            )
            for key in heatmap_keys
        }
    return merged


def _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=False, return_heatmaps=False):
    extractors = components.get("extractors", [components])
    if len(extractors) <= 1 or int(batch["image_64"].size(0)) <= 1:
        return _compute_all_branch_features(
            batch,
            cfgs,
            extractors[0],
            amp_enabled=amp_enabled,
            return_heatmaps=return_heatmaps,
        )

    batch_chunks = _split_batch_for_extractors(batch, len(extractors))
    feature_chunks = []
    for extractor, batch_chunk in zip(extractors, batch_chunks):
        feature_chunks.append(
            _compute_all_branch_features(
                batch_chunk,
                cfgs,
                extractor,
                amp_enabled=amp_enabled,
                return_heatmaps=return_heatmaps,
            )
        )
    return _merge_feature_outputs(feature_chunks, components["device"])


def _evaluate_clip_image_metrics(loader, model, device, amp_enabled=False):
    labels, scores = [], []
    with torch.no_grad():
        for batch in loader:
            image_224 = _move_tensor(batch["image_224"], device)
            with _autocast_context(amp_enabled):
                outputs = model(
                    image_224,
                    output_size=batch["image_64"].shape[-2:],
                    return_patch_maps=False,
                )
            labels.extend(batch["label_img"].cpu().numpy().tolist())
            scores.extend(torch.sigmoid(outputs["global_logit"].float()).view(-1).cpu().numpy().tolist())
    return _classification_metrics(labels, scores)


def _evaluate_clip_score_stats(loader, model, device, amp_enabled=False):
    scores = []
    with torch.no_grad():
        for batch in loader:
            image_224 = _move_tensor(batch["image_224"], device)
            with _autocast_context(amp_enabled):
                outputs = model(
                    image_224,
                    output_size=batch["image_64"].shape[-2:],
                    return_patch_maps=False,
                )
            scores.extend(torch.sigmoid(outputs["global_logit"].float()).view(-1).cpu().numpy().tolist())
    if len(scores) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    score_array = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(score_array.mean()),
        "std": float(score_array.std()),
        "min": float(score_array.min()),
        "max": float(score_array.max()),
    }


def _map_quality_metrics(score_map, apply_sigmoid=True):
    patch_tensor = score_map.detach().float()
    if patch_tensor.dim() == 4:
        patch_tensor = patch_tensor.squeeze(1)
    batch_size = patch_tensor.size(0)
    flat = patch_tensor.view(batch_size, -1)
    flat_prob = torch.sigmoid(flat) if apply_sigmoid else flat.clamp(0.0, 1.0)
    topk_ratio = 0.01
    topk_count = max(1, int(np.ceil(flat_prob.size(1) * topk_ratio)))
    topk_values = torch.topk(flat_prob, k=topk_count, dim=1).values
    topk_mean = topk_values.mean(dim=1)
    peak = flat_prob.max(dim=1).values
    foreground_ratio = (flat_prob >= 0.7).float().mean(dim=1)
    normalized_prob = flat_prob / flat_prob.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    entropy = -(normalized_prob * normalized_prob.clamp_min(1.0e-6).log()).sum(dim=1)
    entropy = entropy / np.log(flat_prob.size(1))
    localization_confidence = topk_mean * (1.0 - entropy).clamp_min(0.0)
    background_consistency = ((1.0 - topk_mean) * (1.0 - peak) * (1.0 - foreground_ratio)).clamp_min(0.0)
    return {
        "peak": peak.cpu().numpy().tolist(),
        "topk_mean": topk_mean.cpu().numpy().tolist(),
        "foreground_ratio": foreground_ratio.cpu().numpy().tolist(),
        "entropy": entropy.cpu().numpy().tolist(),
        "localization_confidence": localization_confidence.cpu().numpy().tolist(),
        "background_consistency": background_consistency.cpu().numpy().tolist(),
    }


def _clip_patch_quality_metrics(patch_map):
    return _map_quality_metrics(patch_map, apply_sigmoid=True)


def _refined_map_quality_metrics(refined_map):
    return _map_quality_metrics(refined_map, apply_sigmoid=False)


def _collect_clip_score_rows(loader, model, device, amp_enabled=False):
    rows = []
    with torch.no_grad():
        for batch in loader:
            image_224 = _move_tensor(batch["image_224"], device)
            with _autocast_context(amp_enabled):
                outputs = model(
                    image_224,
                    output_size=batch["image_64"].shape[-2:],
                    return_patch_maps=True,
                )
            batch_scores = torch.sigmoid(outputs["global_logit"].float()).view(-1).cpu().numpy().tolist()
            if "patch_map" in outputs:
                patch_quality = _clip_patch_quality_metrics(outputs["patch_map"])
            else:
                patch_quality = {
                    "peak": [0.0] * len(batch_scores),
                    "topk_mean": [0.0] * len(batch_scores),
                    "foreground_ratio": [0.0] * len(batch_scores),
                    "entropy": [1.0] * len(batch_scores),
                    "localization_confidence": [0.0] * len(batch_scores),
                    "background_consistency": [1.0] * len(batch_scores),
                }
            hidden_labels = batch.get("hidden_label")
            if hidden_labels is None:
                hidden_label_values = [-1] * len(batch_scores)
            else:
                hidden_label_values = hidden_labels.cpu().numpy().tolist()
            image_names = batch.get("image_name")
            if image_names is None:
                image_names = [str(img_id) for img_id in batch["img_id"]]
            group_ids = batch.get("group_id")
            if group_ids is None:
                group_ids = [str(img_id) for img_id in batch["img_id"]]

            for img_id, group_id, image_name, score, hidden_label, peak, topk_mean, foreground_ratio, entropy, localization_confidence, background_consistency in zip(
                batch["img_id"],
                group_ids,
                image_names,
                batch_scores,
                hidden_label_values,
                patch_quality["peak"],
                patch_quality["topk_mean"],
                patch_quality["foreground_ratio"],
                patch_quality["entropy"],
                patch_quality["localization_confidence"],
                patch_quality["background_consistency"],
            ):
                abnormal_selection_score = float(score) * float(localization_confidence)
                normal_selection_score = float(1.0 - float(score)) * float(background_consistency)
                rows.append({
                    "img_id": str(img_id),
                    "group_id": str(group_id),
                    "image_name": str(image_name),
                    "score": float(score),
                    "heatmap_peak": float(peak),
                    "heatmap_topk_mean": float(topk_mean),
                    "heatmap_foreground_ratio": float(foreground_ratio),
                    "heatmap_entropy": float(entropy),
                    "localization_confidence": float(localization_confidence),
                    "background_consistency": float(background_consistency),
                    "abnormal_selection_score": float(abnormal_selection_score),
                    "normal_selection_score": float(normal_selection_score),
                    "hidden_label": int(hidden_label),
                })
    return rows


def _save_hidden_label_debug(rows, output_path):
    debug_rows = [row for row in rows if int(row.get("hidden_label", -1)) >= 0]
    if len(debug_rows) == 0:
        return
    labels = [int(row["hidden_label"]) for row in debug_rows]
    scores = [float(row["score"]) for row in debug_rows]
    payload = {
        "count": len(debug_rows),
        "metrics": _classification_metrics(labels, scores),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def _hidden_label_metrics_from_rows(rows, score_key="score"):
    debug_rows = [row for row in rows if int(row.get("hidden_label", -1)) >= 0]
    if len(debug_rows) == 0:
        return {"auc": 0.0, "ap": 0.0, "count": 0}
    labels = [int(row["hidden_label"]) for row in debug_rows]
    scores = [float(row[score_key]) for row in debug_rows]
    metrics = _classification_metrics(labels, scores)
    metrics["count"] = len(debug_rows)
    return metrics


def _score_distribution_summary(rows):
    if len(rows) == 0:
        return {"count": 0, "quantiles": {}}
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    quantile_points = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0]
    return {
        "count": int(scores.size),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "quantiles": {
            "{:.2f}".format(quantile): float(np.quantile(scores, quantile))
            for quantile in quantile_points
        },
    }


def _weakclip_pseudo_output_dir(cfgs):
    return _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "pseudo"))


def _weakclip_safd_output_dir(cfgs):
    return _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "safd"))


def _weakclip_safd_bank_path(cfgs):
    return os.path.join(_weakclip_safd_output_dir(cfgs), "safd_normal_bank.pth")


def _teacher_unlabeled_score_paths(cfgs):
    pseudo_dir = _weakclip_pseudo_output_dir(cfgs)
    return {
        "train_scores": os.path.join(pseudo_dir, "unlabeled_train_scores.csv"),
        "val_scores": os.path.join(pseudo_dir, "unlabeled_val_scores.csv"),
        "summary": os.path.join(pseudo_dir, "score_summary.json"),
        "train_debug": os.path.join(pseudo_dir, "unlabeled_train_hidden_debug.json"),
        "val_debug": os.path.join(pseudo_dir, "unlabeled_val_hidden_debug.json"),
        "selected_manifest": os.path.join(pseudo_dir, "pseudo_selected_train.csv"),
        "selected_summary": os.path.join(pseudo_dir, "pseudo_selected_summary.json"),
    }


def _weakclip_selection_targets(score, weak_cfg):
    score = float(score)
    if score >= 0.5:
        target = max(weak_cfg["pseudo_abnormal_min_target"], score)
    else:
        target = min(weak_cfg["pseudo_normal_max_target"], score)
    return float(np.clip(target, 1.0e-4, 1.0 - 1.0e-4))


def _joint_confidence_weight(joint_score):
    return float(np.clip(0.2 + 0.8 * float(joint_score), 0.2, 1.0))


def _evaluate_hidden_label_metrics(loader, model, device, amp_enabled=False):
    rows = []
    with torch.no_grad():
        for batch in loader:
            image_224 = _move_tensor(batch["image_224"], device)
            hidden_label = batch.get("hidden_label")
            if hidden_label is None:
                continue
            hidden_label = hidden_label.view(-1)
            valid_mask = hidden_label >= 0
            if not bool(valid_mask.any()):
                continue
            with _autocast_context(amp_enabled):
                outputs = model(
                    image_224,
                    output_size=batch["image_64"].shape[-2:],
                    return_patch_maps=False,
                )
            scores = torch.sigmoid(outputs["global_logit"].float()).view(-1).detach().cpu()
            hidden_label_cpu = hidden_label.detach().cpu()
            for label, score in zip(hidden_label_cpu[valid_mask].numpy().tolist(), scores[valid_mask].numpy().tolist()):
                rows.append({"hidden_label": int(label), "score": float(score)})
    return _hidden_label_metrics_from_rows(rows)


def _weakclip_checkpoint_score(hidden_metrics, clean_stats):
    return (
        float(hidden_metrics.get("auc", 0.0)) +
        float(hidden_metrics.get("ap", 0.0)) -
        0.25 * float(clean_stats.get("mean", 0.0))
    )


def _resolve_official_eval_stage(cfgs):
    weak_cfg = _get_weakclip_cfg(cfgs)
    preferred = weak_cfg["eval_checkpoint"]
    if preferred not in {"student", "teacher"}:
        preferred = "student"
    if preferred == "student":
        student_best = _clip_stage_checkpoint_paths(cfgs, "clip_student")["best"]
        if os.path.exists(student_best):
            return "clip_student", student_best
        teacher_best = _clip_stage_checkpoint_paths(cfgs, "clip_teacher")["best"]
        return "clip_teacher", teacher_best
    teacher_best = _clip_stage_checkpoint_paths(cfgs, "clip_teacher")["best"]
    return "clip_teacher", teacher_best


def _zero_loss(reference_tensor):
    return reference_tensor.float().sum() * 0.0


def _dice_loss_from_logits(logits, target, eps=1.0e-6):
    prob = torch.sigmoid(logits.float())
    target = target.float()
    prob = prob.view(prob.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (prob * target).sum(dim=1)
    denominator = prob.sum(dim=1) + target.sum(dim=1)
    dice = 1.0 - ((2.0 * intersection + eps) / (denominator + eps))
    return dice.mean()


def _segmentation_bce_dice_loss(logits, target):
    if logits is None or logits.numel() == 0:
        if logits is None:
            return torch.tensor(0.0)
        return _zero_loss(logits)
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), target)
    dice = _dice_loss_from_logits(logits.float(), target)
    return bce + dice


def _background_suppression_loss(patch_map):
    if patch_map is None or patch_map.numel() == 0:
        if patch_map is None:
            return torch.tensor(0.0)
        return _zero_loss(patch_map)
    return torch.sigmoid(patch_map.float()).mean()


class DDADGuidedStudentFusionHead(nn.Module):
    def __init__(self, in_dim=11, hidden_dim=32, dropout=0.1):
        super().__init__()
        hidden_dim = max(int(hidden_dim), int(in_dim) * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )

    def forward(self, x):
        return self.net(x.float())


def _clip_patch_quality_tensors(patch_map):
    patch_tensor = patch_map.float()
    if patch_tensor.dim() == 4:
        patch_tensor = patch_tensor.squeeze(1)
    flat_prob = torch.sigmoid(patch_tensor).view(patch_tensor.size(0), -1)
    topk_count = max(1, int(np.ceil(flat_prob.size(1) * 0.01)))
    topk_mean = torch.topk(flat_prob, k=topk_count, dim=1).values.mean(dim=1)
    peak = flat_prob.max(dim=1).values
    foreground_ratio = (flat_prob >= 0.7).float().mean(dim=1)
    normalized_prob = flat_prob / flat_prob.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    entropy = -(normalized_prob * normalized_prob.clamp_min(1.0e-6).log()).sum(dim=1)
    entropy = entropy / float(np.log(flat_prob.size(1)))
    localization_confidence = topk_mean * (1.0 - entropy).clamp_min(0.0)
    background_consistency = ((1.0 - topk_mean) * (1.0 - peak) * (1.0 - foreground_ratio)).clamp_min(0.0)
    return {
        "peak": peak,
        "topk_mean": topk_mean,
        "foreground_ratio": foreground_ratio,
        "entropy": entropy,
        "localization_confidence": localization_confidence,
        "background_consistency": background_consistency,
    }


def _build_student_fusion_vector(
    clip_logit,
    patch_map,
    refined_features,
    manifest_abnormal_joint=None,
    manifest_normal_joint=None,
):
    clip_logit = clip_logit.float().view(-1, 1)
    clip_score = torch.sigmoid(clip_logit)
    refined_score = refined_features["refined_score"].float().view(-1, 1).detach()
    ddad_score = refined_features["score"].float().view(-1, 1).detach()
    quality = _clip_patch_quality_tensors(patch_map)
    localization_confidence = quality["localization_confidence"].view(-1, 1)
    background_consistency = quality["background_consistency"].view(-1, 1)
    if manifest_abnormal_joint is None:
        abnormal_joint = (
            0.70 * refined_score.clamp(0.0, 1.0) +
            0.20 * clip_score +
            0.10 * localization_confidence.clamp(0.0, 1.0)
        )
    else:
        abnormal_joint = manifest_abnormal_joint.float().view(-1, 1).to(device=clip_logit.device)
    if manifest_normal_joint is None:
        normal_joint = (
            0.70 * (1.0 - refined_score.clamp(0.0, 1.0)) +
            0.20 * (1.0 - clip_score) +
            0.10 * background_consistency.clamp(0.0, 1.0)
        )
    else:
        normal_joint = manifest_normal_joint.float().view(-1, 1).to(device=clip_logit.device)
    return torch.cat(
        [
            clip_logit,
            clip_score,
            refined_score.to(device=clip_logit.device),
            ddad_score.to(device=clip_logit.device),
            abnormal_joint,
            normal_joint,
            quality["peak"].view(-1, 1),
            quality["topk_mean"].view(-1, 1),
            quality["foreground_ratio"].view(-1, 1),
            localization_confidence,
            background_consistency,
        ],
        dim=1,
    )


def _run_clip_teacher_epoch(
    loader,
    model,
    device,
    optimizer,
    scaler,
    amp_enabled,
    clean_loss_weight,
    synthetic_cls_weight,
    seg_loss_weight,
    bg_weight,
):
    image_loss_fn = nn.BCEWithLogitsLoss()
    total_meter = AverageMeter()
    cls_meter = AverageMeter()
    seg_meter = AverageMeter()
    model.train()
    for batch in loader:
        image_224 = _move_tensor(batch["image_224"], device)
        label_img = _move_tensor(batch["label_img"].float(), device)
        mask_syn = _move_tensor(batch["mask_syn"].float(), device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled):
            outputs = model(
                image_224,
                output_size=batch["image_64"].shape[-2:],
                return_patch_maps=True,
            )
            logits = outputs["global_logit"].view(-1).float()
            positive_mask = mask_syn.view(mask_syn.size(0), -1).sum(dim=1) > 0
            normal_mask = ~positive_mask
            clean_cls_loss = _zero_loss(outputs["global_logit"])
            if bool(normal_mask.any()):
                clean_cls_loss = image_loss_fn(logits[normal_mask], torch.zeros_like(logits[normal_mask]))
            synthetic_cls_loss = _zero_loss(outputs["global_logit"])
            if float(synthetic_cls_weight) > 0.0 and bool(positive_mask.any()):
                synthetic_cls_loss = image_loss_fn(logits[positive_mask], label_img.view(-1)[positive_mask])
            cls_loss = float(clean_loss_weight) * clean_cls_loss + float(synthetic_cls_weight) * synthetic_cls_loss
            seg_loss = _zero_loss(outputs["global_logit"])
            if "patch_map" in outputs and bool(positive_mask.any()):
                seg_loss = _segmentation_bce_dice_loss(
                    outputs["patch_map"][positive_mask],
                    mask_syn[positive_mask],
                )
            bg_loss = _zero_loss(outputs["global_logit"])
            if "patch_map" in outputs and bool(normal_mask.any()):
                bg_loss = _background_suppression_loss(outputs["patch_map"][normal_mask])
            loss = cls_loss + float(seg_loss_weight) * seg_loss + float(bg_weight) * bg_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_meter.update(loss.item(), image_224.size(0))
        cls_meter.update(cls_loss.item(), image_224.size(0))
        seg_meter.update(float(seg_loss.item()) if torch.is_tensor(seg_loss) else float(seg_loss), image_224.size(0))
    return {
        "total": total_meter.avg,
        "cls": cls_meter.avg,
        "seg": seg_meter.avg,
    }


def _run_clip_clean_epoch(loader, model, device, optimizer, scaler, amp_enabled, clean_loss_weight, bg_weight):
    image_loss_fn = nn.BCEWithLogitsLoss()
    total_meter = AverageMeter()
    cls_meter = AverageMeter()
    bg_meter = AverageMeter()
    model.train()
    for batch in loader:
        image_224 = _move_tensor(batch["image_224"], device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled):
            outputs = model(
                image_224,
                output_size=batch["image_64"].shape[-2:],
                return_patch_maps=True,
            )
            logits = outputs["global_logit"].view(-1).float()
            clean_targets = torch.zeros_like(logits)
            cls_loss = image_loss_fn(logits, clean_targets)
            bg_loss = _background_suppression_loss(outputs.get("patch_map"))
            loss = float(clean_loss_weight) * cls_loss + float(bg_weight) * bg_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_meter.update(loss.item(), image_224.size(0))
        cls_meter.update(cls_loss.item(), image_224.size(0))
        bg_meter.update(float(bg_loss.item()) if torch.is_tensor(bg_loss) else float(bg_loss), image_224.size(0))
    return {
        "total": total_meter.avg,
        "cls": cls_meter.avg,
        "bg": bg_meter.avg,
    }


def _run_clip_synthetic_local_epoch(
    loader,
    model,
    device,
    optimizer,
    scaler,
    amp_enabled,
    seg_syn_weight,
    use_synthetic_cls=False,
):
    image_loss_fn = nn.BCEWithLogitsLoss()
    total_meter = AverageMeter()
    seg_meter = AverageMeter()
    cls_meter = AverageMeter()
    model.train()
    for batch in loader:
        image_224 = _move_tensor(batch["image_224"], device)
        label_img = _move_tensor(batch["label_img"].float(), device)
        mask_syn = _move_tensor(batch["mask_syn"].float(), device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled):
            outputs = model(
                image_224,
                output_size=batch["image_64"].shape[-2:],
                return_patch_maps=True,
            )
            seg_loss = _zero_loss(outputs["global_logit"])
            positive_mask = mask_syn.view(mask_syn.size(0), -1).sum(dim=1) > 0
            if "patch_map" in outputs and bool(positive_mask.any()):
                seg_loss = _segmentation_bce_dice_loss(
                    outputs["patch_map"][positive_mask],
                    mask_syn[positive_mask],
                )
            cls_loss = _zero_loss(outputs["global_logit"])
            if use_synthetic_cls:
                logits = outputs["global_logit"].view(-1).float()
                cls_loss = image_loss_fn(logits, label_img.view(-1))
            loss = float(seg_syn_weight) * seg_loss + cls_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_meter.update(loss.item(), image_224.size(0))
        seg_meter.update(float(seg_loss.item()) if torch.is_tensor(seg_loss) else float(seg_loss), image_224.size(0))
        cls_meter.update(float(cls_loss.item()) if torch.is_tensor(cls_loss) else float(cls_loss), image_224.size(0))
    return {
        "total": total_meter.avg,
        "seg": seg_meter.avg,
        "cls": cls_meter.avg,
    }


def _run_clip_pseudo_epoch(
    loader,
    model,
    teacher_model,
    device,
    optimizer,
    scaler,
    amp_enabled,
    pseudo_loss_weight,
    pseudo_loc_weight,
    bg_weight,
):
    total_meter = AverageMeter()
    cls_meter = AverageMeter()
    loc_meter = AverageMeter()
    bg_meter = AverageMeter()
    if loader is None:
        return {
            "total": total_meter.avg,
            "cls": cls_meter.avg,
            "loc": loc_meter.avg,
            "bg": bg_meter.avg,
        }

    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    for batch in loader:
        image_224 = _move_tensor(batch["image_224"], device)
        pseudo_target = _move_tensor(batch["pseudo_target"].float(), device).view(-1)
        pseudo_weight = _move_tensor(batch["pseudo_weight"].float(), device).view(-1)
        abnormal_mask = pseudo_target >= 0.5
        normal_mask = ~abnormal_mask

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled):
            outputs = model(
                image_224,
                output_size=batch["image_64"].shape[-2:],
                return_patch_maps=True,
            )
            logits = outputs["global_logit"].view(-1).float()
            per_sample_loss = F.binary_cross_entropy_with_logits(logits, pseudo_target, reduction="none")
            cls_loss = (per_sample_loss * pseudo_weight).mean()

            pseudo_loc_loss = _zero_loss(outputs["global_logit"])
            if teacher_model is not None and "patch_map" in outputs and bool(abnormal_mask.any()):
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        image_224[abnormal_mask],
                        output_size=batch["image_64"].shape[-2:],
                        return_patch_maps=True,
                    )
                student_patch = torch.sigmoid(outputs["patch_map"][abnormal_mask].float())
                teacher_patch = torch.sigmoid(teacher_outputs["patch_map"].float())
                pseudo_loc_loss = F.mse_loss(student_patch, teacher_patch)

            bg_loss = _zero_loss(outputs["global_logit"])
            if "patch_map" in outputs and bool(normal_mask.any()):
                bg_loss = _background_suppression_loss(outputs["patch_map"][normal_mask])

            loss = (
                float(pseudo_loss_weight) * cls_loss +
                float(pseudo_loc_weight) * pseudo_loc_loss +
                float(bg_weight) * bg_loss
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_meter.update(loss.item(), image_224.size(0))
        cls_meter.update(float(cls_loss.item()) if torch.is_tensor(cls_loss) else float(cls_loss), image_224.size(0))
        loc_meter.update(float(pseudo_loc_loss.item()) if torch.is_tensor(pseudo_loc_loss) else float(pseudo_loc_loss), image_224.size(0))
        bg_meter.update(float(bg_loss.item()) if torch.is_tensor(bg_loss) else float(bg_loss), image_224.size(0))
    return {
        "total": total_meter.avg,
        "cls": cls_meter.avg,
        "loc": loc_meter.avg,
        "bg": bg_meter.avg,
    }


def _run_clip_student_fusion_pseudo_epoch(
    loader,
    model,
    fusion_head,
    cfgs,
    module_a,
    module_b,
    refine_net,
    refine_in,
    refine_runtime_cfg,
    device,
    optimizer,
    scaler,
    amp_enabled,
    fusion_loss_weight,
    clip_aux_loss_weight,
    ddad_map_loss_weight,
    bg_weight,
):
    total_meter = AverageMeter()
    fused_meter = AverageMeter()
    clip_meter = AverageMeter()
    map_meter = AverageMeter()
    bg_meter = AverageMeter()
    image_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    clip_is_frozen = not any(param.requires_grad for param in model.parameters())
    model.eval() if clip_is_frozen else model.train()
    fusion_head.train()
    refine_net.eval()
    for batch in loader:
        image_224 = _move_tensor(batch["image_224"], device)
        x64 = _move_tensor(batch["image_64"], device)
        pseudo_target = _move_tensor(batch["pseudo_target"].float(), device).view(-1)
        pseudo_weight = _move_tensor(batch["pseudo_weight"].float(), device).view(-1)
        abnormal_mask = pseudo_target >= 0.5
        normal_mask = ~abnormal_mask
        abnormal_joint = _move_tensor(batch["abnormal_joint_score"].float(), device).view(-1)
        normal_joint = _move_tensor(batch["normal_joint_score"].float(), device).view(-1)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            refined_features = _compute_refined_ddad_features(
                x64,
                cfgs,
                module_a,
                module_b,
                refine_net,
                refine_in,
                refine_runtime_cfg,
            )
        with _autocast_context(amp_enabled):
            outputs = model(
                image_224,
                output_size=batch["image_64"].shape[-2:],
                return_patch_maps=True,
            )
            clip_logits = outputs["global_logit"].view(-1).float()
            fusion_vector = _build_student_fusion_vector(
                clip_logits,
                outputs["patch_map"],
                refined_features,
                manifest_abnormal_joint=abnormal_joint,
                manifest_normal_joint=normal_joint,
            )
            fused_logits = fusion_head(fusion_vector).view(-1)
            fused_loss = (image_loss_fn(fused_logits, pseudo_target) * pseudo_weight).mean()
            clip_aux_loss = (image_loss_fn(clip_logits, pseudo_target) * pseudo_weight).mean()
            ddad_map_loss = _zero_loss(outputs["global_logit"])
            if float(ddad_map_loss_weight) > 0.0 and "patch_map" in outputs and bool(abnormal_mask.any()):
                student_patch = torch.sigmoid(outputs["patch_map"][abnormal_mask].float())
                ddad_patch = refined_features["refined_map"][abnormal_mask].float().to(device=student_patch.device)
                ddad_map_loss = F.mse_loss(student_patch, ddad_patch)
            bg_loss = _zero_loss(outputs["global_logit"])
            if "patch_map" in outputs and bool(normal_mask.any()):
                bg_loss = _background_suppression_loss(outputs["patch_map"][normal_mask])
            loss = (
                float(fusion_loss_weight) * fused_loss +
                float(clip_aux_loss_weight) * clip_aux_loss +
                float(ddad_map_loss_weight) * ddad_map_loss +
                float(bg_weight) * bg_loss
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_meter.update(loss.item(), image_224.size(0))
        fused_meter.update(float(fused_loss.item()), image_224.size(0))
        clip_meter.update(float(clip_aux_loss.item()), image_224.size(0))
        map_meter.update(float(ddad_map_loss.item()), image_224.size(0))
        bg_meter.update(float(bg_loss.item()), image_224.size(0))
    return {
        "total": total_meter.avg,
        "fused_cls": fused_meter.avg,
        "clip_aux": clip_meter.avg,
        "ddad_map": map_meter.avg,
        "bg": bg_meter.avg,
    }


def _run_weak_ddad_train_epoch(loader, model, network, optimizer, entropy_loss_weight=0.0):
    total_meter = AverageMeter()
    rec_meter = AverageMeter()
    aux_meter = AverageMeter()
    model.train()
    entropy_criterion = EntropyLossEncap() if network == "MemAE" else None
    for batch in loader:
        x64 = _move_tensor(batch["image_64"], next(model.parameters()).device)
        optimizer.zero_grad(set_to_none=True)
        if network == "AE":
            reconstruction = model(x64)
            rec_err = (reconstruction - x64) ** 2
            rec_loss = rec_err.mean()
            aux_loss = rec_loss.new_zeros(())
            loss = rec_loss
        elif network == "AE-U":
            mean, logvar = model(x64)
            rec_err = (mean - x64) ** 2
            rec_loss = torch.mean(torch.exp(-logvar) * rec_err)
            aux_loss = torch.mean(logvar)
            loss = rec_loss + aux_loss
        elif network == "MemAE":
            output = model(x64)
            reconstruction = output["output"]
            rec_err = (reconstruction - x64) ** 2
            rec_loss = rec_err.mean()
            aux_loss = entropy_criterion(output["att"]).mean()
            loss = rec_loss + float(entropy_loss_weight) * aux_loss
        else:
            raise ValueError("Unsupported weak DDAD network: {}".format(network))

        loss.backward()
        optimizer.step()
        total_meter.update(loss.item(), x64.size(0))
        rec_meter.update(rec_loss.item(), x64.size(0))
        aux_meter.update(aux_loss.item(), x64.size(0))
    return {
        "total": total_meter.avg,
        "rec": rec_meter.avg,
        "aux": aux_meter.avg,
    }


def _evaluate_single_ddad_reconstruction(loader, model, network, device):
    rec_losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x64 = _move_tensor(batch["image_64"], device)
            if network == "AE":
                reconstruction = model(x64)
                rec_err = (reconstruction - x64) ** 2
                rec_losses.append(rec_err.mean().item())
            elif network == "AE-U":
                mean, logvar = model(x64)
                rec_err = (mean - x64) ** 2
                loss = torch.mean(torch.exp(-logvar) * rec_err) + torch.mean(logvar)
                rec_losses.append(loss.item())
            elif network == "MemAE":
                output = model(x64)
                rec_err = (output["output"] - x64) ** 2
                rec_losses.append(rec_err.mean().item())
            else:
                raise ValueError("Unsupported weak DDAD network: {}".format(network))
    if len(rec_losses) == 0:
        return 0.0
    return float(np.mean(rec_losses))


def _collect_ddad_score_rows(loader, cfgs, module_a, module_b, device):
    rows = []
    with torch.no_grad():
        for batch in loader:
            x64 = _move_tensor(batch["image_64"], device)
            ddad_features = _compute_ddad_branch_features(x64, cfgs, module_a, module_b)
            ddad_scores = ddad_features["score"].view(-1).detach().cpu().numpy().tolist()
            inter_scores = ddad_features["inter_score"].view(-1).detach().cpu().numpy().tolist()
            intra_scores = ddad_features["intra_score"].view(-1).detach().cpu().numpy().tolist()
            hidden_labels = batch.get("hidden_label")
            if hidden_labels is None:
                hidden_label_values = [-1] * len(ddad_scores)
            else:
                hidden_label_values = hidden_labels.cpu().numpy().tolist()
            image_names = batch.get("image_name")
            if image_names is None:
                image_names = [str(img_id) for img_id in batch["img_id"]]
            group_ids = batch.get("group_id")
            if group_ids is None:
                group_ids = [str(img_id) for img_id in batch["img_id"]]
            for img_id, group_id, image_name, ddad_score, inter_score, intra_score, hidden_label in zip(
                batch["img_id"],
                group_ids,
                image_names,
                ddad_scores,
                inter_scores,
                intra_scores,
                hidden_label_values,
            ):
                rows.append({
                    "img_id": str(img_id),
                    "group_id": str(group_id),
                    "image_name": str(image_name),
                    "ddad_score": float(ddad_score),
                    "ddad_inter_score": float(inter_score),
                    "ddad_intra_score": float(intra_score),
                    "hidden_label": int(hidden_label),
                })
    return rows


def _collect_refined_ddad_score_rows(loader, cfgs, module_a, module_b, refine_net, refine_in, refine_runtime_cfg, device):
    rows = []
    with torch.no_grad():
        for batch in loader:
            x64 = _move_tensor(batch["image_64"], device)
            features = _compute_refined_ddad_features(
                x64,
                cfgs,
                module_a,
                module_b,
                refine_net,
                refine_in,
                refine_runtime_cfg,
            )
            refined_scores = features["refined_score"].view(-1).detach().cpu().numpy().tolist()
            refined_quality = _refined_map_quality_metrics(features["refined_map"])
            hidden_labels = batch.get("hidden_label")
            if hidden_labels is None:
                hidden_label_values = [-1] * len(refined_scores)
            else:
                hidden_label_values = hidden_labels.cpu().numpy().tolist()
            image_names = batch.get("image_name")
            if image_names is None:
                image_names = [str(img_id) for img_id in batch["img_id"]]
            group_ids = batch.get("group_id")
            if group_ids is None:
                group_ids = [str(img_id) for img_id in batch["img_id"]]
            for img_id, group_id, image_name, refined_score, hidden_label, peak, topk_mean, foreground_ratio, entropy in zip(
                batch["img_id"],
                group_ids,
                image_names,
                refined_scores,
                hidden_label_values,
                refined_quality["peak"],
                refined_quality["topk_mean"],
                refined_quality["foreground_ratio"],
                refined_quality["entropy"],
            ):
                rows.append({
                    "img_id": str(img_id),
                    "group_id": str(group_id),
                    "image_name": str(image_name),
                    "refined_ddad_score": float(refined_score),
                    "refined_heatmap_peak": float(peak),
                    "refined_heatmap_topk_mean": float(topk_mean),
                    "refined_heatmap_foreground_ratio": float(foreground_ratio),
                    "refined_heatmap_entropy": float(entropy),
                    "hidden_label": int(hidden_label),
                })
    return rows


def _collect_safd_ddad_score_rows(loader, cfgs, module_a, module_b, refine_net, refine_in, refine_runtime_cfg, device):
    rows = []
    weak_cfg = _get_weakclip_cfg(cfgs)
    safd_decomposer = _build_weak_safd_decomposer(cfgs, device)
    safd_bank = _load_weak_safd_normal_bank(cfgs, device) if weak_cfg["safd_score_mode"] == "normal_bank" else None
    with torch.no_grad():
        for batch in loader:
            x64 = _move_tensor(batch["image_64"], device)
            features = _compute_refined_ddad_features(
                x64,
                cfgs,
                module_a,
                module_b,
                refine_net,
                refine_in,
                refine_runtime_cfg,
            )
            if weak_cfg["safd_score_mode"] == "normal_bank":
                safd_scores, band_scores = _compute_weak_safd_normal_score(
                    features, safd_decomposer, safd_bank, weak_cfg, device
                )
            else:
                safd_scores, band_scores = _compute_weak_safd_score(features, safd_decomposer, weak_cfg)
            safd_scores = safd_scores.view(-1).detach().cpu().numpy().tolist()
            band_mean = band_scores.mean(dim=1).detach().cpu().numpy().tolist()
            band_max = band_scores.max(dim=1).values.detach().cpu().numpy().tolist()
            hidden_labels = batch.get("hidden_label")
            if hidden_labels is None:
                hidden_label_values = [-1] * len(safd_scores)
            else:
                hidden_label_values = hidden_labels.cpu().numpy().tolist()
            image_names = batch.get("image_name")
            if image_names is None:
                image_names = [str(img_id) for img_id in batch["img_id"]]
            group_ids = batch.get("group_id")
            if group_ids is None:
                group_ids = [str(img_id) for img_id in batch["img_id"]]
            for img_id, group_id, image_name, safd_score, safd_band_mean, safd_band_max, hidden_label in zip(
                batch["img_id"],
                group_ids,
                image_names,
                safd_scores,
                band_mean,
                band_max,
                hidden_label_values,
            ):
                rows.append({
                    "img_id": str(img_id),
                    "group_id": str(group_id),
                    "image_name": str(image_name),
                    "safd_ddad_score": float(safd_score),
                    "safd_normal_score": float(safd_score),
                    "safd_ddad_band_mean": float(safd_band_mean),
                    "safd_ddad_band_max": float(safd_band_max),
                    "safd_normal_band_mean": float(safd_band_mean),
                    "safd_normal_band_max": float(safd_band_max),
                    "hidden_label": int(hidden_label),
                })
    return rows


def build_safd_normal_bank(cfgs, refine_in):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    loader_overrides = _get_clip_loader_overrides(cfgs)
    batch_size = int(clip_cfg.get("val_bs", clip_cfg.get("bs", 4)))
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    weak_cfg = _get_weakclip_cfg(cfgs)
    if not weak_cfg["use_refine_score"]:
        raise RuntimeError("SAFD normal bank requires WeakCLIP.use_refine_score=true.")

    loader = _build_multibranch_loader(
        cfgs,
        subset="clean_train_normal",
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        include_clip=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    module_a = _load_weak_ddad_ensemble(cfgs, "weak_a")
    module_b = _load_weak_ddad_ensemble(cfgs, "weak_b")
    refine_net, refine_runtime_cfg, _ = _load_weak_refine_model(cfgs, device, refine_in)
    refine_net.eval()
    safd_decomposer = _build_weak_safd_decomposer(cfgs, device)

    coeff_chunks = []
    with torch.no_grad():
        for batch in loader:
            x64 = _move_tensor(batch["image_64"], device)
            features = _compute_refined_ddad_features(
                x64,
                cfgs,
                module_a,
                module_b,
                refine_net,
                refine_in,
                refine_runtime_cfg,
            )
            coeff_chunks.append(_compute_weak_safd_coefficients(features, safd_decomposer).detach().cpu())
    if len(coeff_chunks) == 0:
        raise RuntimeError("No clean normal samples available to build SAFD normal bank.")

    coeffs = torch.cat(coeff_chunks, dim=0).float()
    median = coeffs.median(dim=0).values
    mad = torch.abs(coeffs - median.unsqueeze(0)).median(dim=0).values
    bank = {
        "median": median,
        "mad": mad.clamp_min(float(weak_cfg["safd_normal_mad_eps"])),
        "mad_eps": float(weak_cfg["safd_normal_mad_eps"]),
        "count": int(coeffs.size(0)),
        "shape": list(median.shape),
        "safd_levels": int(weak_cfg["safd_levels"]),
        "safd_patch_size": int(weak_cfg["safd_patch_size"]),
        "safd_topk_ratio": float(weak_cfg["safd_topk_ratio"]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    bank_path = _weakclip_safd_bank_path(cfgs)
    torch.save(bank, bank_path)
    summary_path = os.path.join(_weakclip_safd_output_dir(cfgs), "safd_normal_bank_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "checkpoint": bank_path,
            "count": bank["count"],
            "shape": bank["shape"],
            "safd_levels": bank["safd_levels"],
            "safd_patch_size": bank["safd_patch_size"],
            "safd_topk_ratio": bank["safd_topk_ratio"],
            "mad_eps": bank["mad_eps"],
        }, f, indent=2)
    print("=> Saved SAFD normal bank to {} (count={}, shape={})".format(bank_path, bank["count"], bank["shape"]))


def _evaluate_weak_refine_loader(loader, cfgs, module_a, module_b, refine_net, refine_in, refine_runtime_cfg, device):
    rows = []
    with torch.no_grad():
        refine_net.eval()
        for batch in loader:
            x64 = _move_tensor(batch["image_64"], device)
            features = _compute_refined_ddad_features(
                x64,
                cfgs,
                module_a,
                module_b,
                refine_net,
                refine_in,
                refine_runtime_cfg,
            )
            scores = features["refined_score"].view(-1).detach().cpu().numpy().tolist()
            fallback_labels = batch["label_img"].view(-1).detach().cpu().numpy().tolist()
            hidden_labels = batch.get("hidden_label")
            if hidden_labels is None:
                labels = fallback_labels
            else:
                hidden_values = hidden_labels.view(-1).detach().cpu().numpy().tolist()
                labels = [
                    int(hidden_label) if int(hidden_label) >= 0 else int(fallback_label)
                    for hidden_label, fallback_label in zip(hidden_values, fallback_labels)
                ]
            image_names = batch.get("image_name")
            if image_names is None:
                image_names = [str(img_id) for img_id in batch["img_id"]]
            group_ids = batch.get("group_id")
            if group_ids is None:
                group_ids = [str(img_id) for img_id in batch["img_id"]]
            for img_id, group_id, image_name, label, score in zip(
                batch["img_id"],
                group_ids,
                image_names,
                labels,
                scores,
            ):
                rows.append({
                    "img_id": str(img_id),
                    "group_id": str(group_id),
                    "image_name": str(image_name),
                    "label": int(label),
                    "refined_ddad_score": float(score),
                })
    metrics = _classification_metrics(
        [row["label"] for row in rows],
        [row["refined_ddad_score"] for row in rows],
    )
    metrics["count"] = len(rows)
    return metrics, rows


def _attach_rank_and_percentile(df, score_column, prefix):
    if score_column not in df.columns or len(df) == 0:
        df = df.copy()
        df["{}_rank".format(prefix)] = pd.Series(dtype=np.int64)
        df["{}_percentile".format(prefix)] = pd.Series(dtype=np.float64)
        return df
    df = df.copy()
    descending = True
    ranks = df[score_column].rank(method="average", ascending=not descending)
    if len(df) == 1:
        percentiles = np.ones(1, dtype=np.float64)
    else:
        percentiles = 1.0 - ((ranks.to_numpy(dtype=np.float64) - 1.0) / float(len(df) - 1))
    df["{}_rank".format(prefix)] = ranks.astype(np.int64)
    df["{}_percentile".format(prefix)] = percentiles.astype(np.float64)
    return df


def train_weak_ddad_module(cfgs, mode):
    if mode not in {"weak_a", "weak_b"}:
        raise ValueError("Unsupported weak DDAD mode: {}".format(mode))

    device = _get_device(cfgs)
    model_cfg = cfgs["Model"]
    data_cfg = _get_data_cfg(cfgs)
    solver_cfg = _get_section(cfgs, "Solver")
    clip_loader_overrides = _get_clip_loader_overrides(cfgs)
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    ensemble_cfg = _get_ensemble_cfg(cfgs)
    base_seed = ensemble_cfg["base_seed"]
    target_count = ensemble_cfg["target_count"]
    network = model_cfg["network"]
    batch_size = int(solver_cfg.get("bs", 64))
    num_epoch = int(solver_cfg.get("num_epoch", 250))
    lr = float(solver_cfg.get("lr", 5.0e-4))
    weight_decay = float(solver_cfg.get("weight_decay", 0.0))
    entropy_loss_weight = float(model_cfg.get("entropy_loss_weight", 0.0002))
    subset = "clean_plus_unlabeled_train_pool" if mode == "weak_a" else "clean_train_normal"

    train_loader = _build_multibranch_loader(
        cfgs,
        subset=subset,
        batch_size=batch_size,
        shuffle=True,
        synthetic_probability=0.0,
        drop_last=True,
        include_clip=False,
        num_workers_override=clip_loader_overrides["train_workers"],
        persistent_workers_override=clip_loader_overrides["train_persistent_workers"],
        prefetch_factor_override=clip_loader_overrides["train_prefetch_factor"],
        cache_images_override=clip_loader_overrides["cache_images"],
        cache_clip_images_override=False,
        dataset_kwargs=dataset_kwargs,
    )
    clean_val_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_val_normal",
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        include_clip=False,
        num_workers_override=clip_loader_overrides["val_workers"],
        persistent_workers_override=clip_loader_overrides["val_persistent_workers"],
        prefetch_factor_override=clip_loader_overrides["val_prefetch_factor"],
        cache_images_override=clip_loader_overrides["cache_images"],
        cache_clip_images_override=False,
        dataset_kwargs=dataset_kwargs,
    )

    member_dir = _weak_ddad_member_dir(cfgs, mode)
    existing_members = {
        int(os.path.splitext(name)[0])
        for name in _sorted_checkpoint_names(member_dir)
        if os.path.splitext(name)[0].isdigit()
    }
    missing_members = [member_index for member_index in range(target_count) if member_index not in existing_members]
    print("=> Existing {} members: {}/{}".format(mode, len(existing_members.intersection(set(range(target_count)))), target_count))
    if len(missing_members) == 0:
        print("=> {} ensemble already complete. Nothing to train.".format(mode))
        return

    for member_index in missing_members:
        seed = base_seed + member_index
        _set_global_seed(seed)
        model = get_model(
            network=network,
            mp=model_cfg["mp"],
            ls=model_cfg["ls"],
            img_size=int(data_cfg.get("img_size", 64)),
            mem_dim=model_cfg["mem_dim"],
            shrink_thres=model_cfg["shrink_thres"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
        checkpoint_paths = _weak_ddad_member_paths(cfgs, mode, member_index)
        writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_{}".format(mode), "member_{:02d}".format(member_index)))
        best_clean_val = float("inf")
        print("=> Training {} member {}/{} (seed={}) on subset '{}'".format(mode, member_index + 1, target_count, seed, subset))
        for epoch in range(1, num_epoch + 1):
            start = time.time()
            train_stats = _run_weak_ddad_train_epoch(
                train_loader,
                model,
                network,
                optimizer,
                entropy_loss_weight=entropy_loss_weight,
            )
            clean_val_loss = _evaluate_single_ddad_reconstruction(clean_val_loader, model, network, device)
            writer.add_scalar("train_total", train_stats["total"], epoch)
            writer.add_scalar("train_rec", train_stats["rec"], epoch)
            writer.add_scalar("train_aux", train_stats["aux"], epoch)
            writer.add_scalar("clean_val_rec", clean_val_loss, epoch)
            checkpoint_extra = {
                "epoch": epoch,
                "clean_val_rec": clean_val_loss,
                "train_total": train_stats["total"],
                "train_rec": train_stats["rec"],
                "train_aux": train_stats["aux"],
            }
            _save_checkpoint(checkpoint_paths["last"], model, optimizer, extra=checkpoint_extra)
            if clean_val_loss < best_clean_val:
                best_clean_val = clean_val_loss
                _save_checkpoint(checkpoint_paths["best"], model, optimizer, extra=checkpoint_extra)
            if epoch == 1 or epoch % 25 == 0 or epoch == num_epoch:
                print(
                    "{} member {:02d} Epoch[{}/{}]\tTime:{:.1f}s\tTrain:{:.4f}\tRec:{:.4f}\tAux:{:.4f}\tCleanVal:{:.4f}".format(
                        mode,
                        member_index,
                        epoch,
                        num_epoch,
                        time.time() - start,
                        train_stats["total"],
                        train_stats["rec"],
                        train_stats["aux"],
                        clean_val_loss,
                    )
                )
        writer.close()

    other_mode = "weak_b" if mode == "weak_a" else "weak_a"
    try:
        module_a = _load_weak_ddad_ensemble(cfgs, "weak_a")
        module_b = _load_weak_ddad_ensemble(cfgs, "weak_b")
    except RuntimeError:
        if os.path.exists(_weak_ddad_member_dir(cfgs, other_mode)):
            print("=> Waiting for both weak_a and weak_b to be complete before DDAD hidden-debug evaluation.")
        return

    unlabeled_val_loader = _build_multibranch_loader(
        cfgs,
        subset="unlabeled_val_pool",
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        include_clip=False,
        num_workers_override=clip_loader_overrides["val_workers"],
        persistent_workers_override=clip_loader_overrides["val_persistent_workers"],
        prefetch_factor_override=clip_loader_overrides["val_prefetch_factor"],
        cache_images_override=clip_loader_overrides["cache_images"],
        cache_clip_images_override=False,
        dataset_kwargs=dataset_kwargs,
    )
    ddad_rows = _collect_ddad_score_rows(unlabeled_val_loader, cfgs, module_a, module_b, device)
    ddad_metrics = _hidden_label_metrics_from_rows(ddad_rows, score_key="ddad_score")
    debug_path = os.path.join(_checkpoint_out_dir(cfgs), "weak_ddad_hidden_debug.json")
    with open(debug_path, "w") as f:
        json.dump(
            {
                "split": "unlabeled_val_pool",
                "metrics": ddad_metrics,
                "count": len(ddad_rows),
            },
            f,
            indent=2,
        )
    print("=> Saved weak DDAD hidden-debug metrics to {}".format(debug_path))


def train_weak_refine_module(cfgs, refine_in=None):
    refine_in = list(refine_in or ["inter_dis", "intra_dis"])
    device = _get_device(cfgs)
    clip_loader_overrides = _get_clip_loader_overrides(cfgs)
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    solver_cfg = _get_weak_refine_solver_cfg(cfgs)
    synthetic_probability = float(_get_section(cfgs, "CLIP").get("synthetic_probability", 0.5))
    batch_size = solver_cfg["bs"]
    num_epoch = solver_cfg["num_epoch"]
    grad_clip = solver_cfg["grad_clip"]
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    module_a = _load_weak_ddad_ensemble(cfgs, "weak_a")
    module_b = _load_weak_ddad_ensemble(cfgs, "weak_b")
    refine_net, refine_runtime_cfg, refine_in_channels = _build_weak_refine_model(cfgs, device, refine_in)
    optimizer = torch.optim.Adam(
        refine_net.parameters(),
        lr=solver_cfg["lr"],
        betas=(0.5, 0.999),
        weight_decay=solver_cfg["weight_decay"],
    )
    criterion = FocalLoss()
    train_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_train_normal",
        batch_size=batch_size,
        shuffle=True,
        synthetic_probability=synthetic_probability,
        drop_last=True,
        include_clip=False,
        num_workers_override=clip_loader_overrides["train_workers"],
        persistent_workers_override=clip_loader_overrides["train_persistent_workers"],
        prefetch_factor_override=clip_loader_overrides["train_prefetch_factor"],
        cache_images_override=clip_loader_overrides["cache_images"],
        cache_clip_images_override=False,
        dataset_kwargs=dataset_kwargs,
    )
    val_loader = _build_multibranch_loader(
        cfgs,
        subset="unlabeled_val_pool",
        batch_size=int(_get_section(cfgs, "CLIP").get("val_bs", 1)),
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        include_clip=False,
        num_workers_override=clip_loader_overrides["val_workers"],
        persistent_workers_override=clip_loader_overrides["val_persistent_workers"],
        prefetch_factor_override=clip_loader_overrides["val_prefetch_factor"],
        cache_images_override=clip_loader_overrides["cache_images"],
        cache_clip_images_override=False,
        dataset_kwargs=dataset_kwargs,
    )
    checkpoint_path = _weak_refine_checkpoint_path(cfgs, refine_in)
    writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_weak_refine"))
    best_auc = -float("inf")
    best_ap = -float("inf")
    results = []
    print(
        "=> Training weak_r refine: band_mode={} in_channels={} topk={:.2%} checkpoint={}".format(
            refine_runtime_cfg["band_mode"],
            refine_in_channels,
            refine_runtime_cfg["score_topk_ratio"],
            checkpoint_path,
        )
    )

    for epoch in range(1, num_epoch + 1):
        start = time.time()
        refine_net.train()
        loss_meter = AverageMeter()
        for batch in train_loader:
            x64 = _move_tensor(batch["image_64"], device)
            mask_syn = _move_tensor(batch["mask_syn"].long(), device)
            with torch.no_grad():
                ddad_features = _compute_ddad_branch_features(x64, cfgs, module_a, module_b)
                net_in = build_refine_input(
                    ddad_features["inter_img"],
                    ddad_features["intra_img"],
                    refine_in,
                    use_fixed_3band=refine_runtime_cfg["use_fixed_3band"],
                )
            segmentation = torch.softmax(refine_net(net_in), dim=1)
            loss = criterion(segmentation, mask_syn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(refine_net.parameters(), grad_clip)
            optimizer.step()
            loss_meter.update(loss.item(), x64.size(0))

        metrics, _ = _evaluate_weak_refine_loader(
            val_loader,
            cfgs,
            module_a,
            module_b,
            refine_net,
            refine_in,
            refine_runtime_cfg,
            device,
        )
        auc = float(metrics["auc"])
        ap = float(metrics["ap"])
        if auc > best_auc:
            best_auc = auc
        if ap > best_ap:
            best_ap = ap
        results.append({
            "epoch": epoch,
            "loss": float(loss_meter.avg),
            "auc": auc,
            "ap": ap,
            "best_auc": best_auc,
            "best_ap": best_ap,
        })
        writer.add_scalar("train_loss", loss_meter.avg, epoch)
        writer.add_scalar("unlabeled_val_auc", auc, epoch)
        writer.add_scalar("unlabeled_val_ap", ap, epoch)
        checkpoint_extra = {
            "epoch": epoch,
            "train_loss": loss_meter.avg,
            "unlabeled_val_auc": auc,
            "unlabeled_val_ap": ap,
            "best_auc": best_auc,
            "best_ap": best_ap,
            "refine_in": refine_in,
            "band_mode": refine_runtime_cfg["band_mode"],
            "score_topk_ratio": refine_runtime_cfg["score_topk_ratio"],
        }
        _save_checkpoint(checkpoint_path.replace(".pth", "_last.pth"), refine_net, optimizer, extra=checkpoint_extra)
        if auc >= best_auc:
            _save_checkpoint(checkpoint_path, refine_net, optimizer, extra=checkpoint_extra)
        if epoch == 1 or epoch % 25 == 0 or epoch == num_epoch:
            print(
                "weak_r Epoch[{}/{}]\tTime:{:.1f}s\tLoss:{:.4f}\tUnlabeledVal AUC:{:.4f}\tUnlabeledVal AP:{:.4f}\tBest AUC:{:.4f}\tBest AP:{:.4f}".format(
                    epoch,
                    num_epoch,
                    time.time() - start,
                    loss_meter.avg,
                    auc,
                    ap,
                    best_auc,
                    best_ap,
                )
            )
    writer.close()
    method_name = "weak_refine_dual" if len(refine_in) == 2 else "weak_refine_intra"
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(_checkpoint_out_dir(cfgs), "{}_results.csv".format(method_name)), index=False)
    summary = {
        "method": method_name,
        "final_auc": float(results[-1]["auc"]) if len(results) > 0 else 0.0,
        "final_ap": float(results[-1]["ap"]) if len(results) > 0 else 0.0,
        "selection_split": "unlabeled_val_pool",
        "best_auc": float(best_auc if best_auc > -float("inf") else 0.0),
        "best_ap": float(best_ap if best_ap > -float("inf") else 0.0),
        "checkpoint": checkpoint_path,
    }
    with open(os.path.join(_checkpoint_out_dir(cfgs), "{}_results.json".format(method_name)), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(_checkpoint_out_dir(cfgs), "{}_results.txt".format(method_name)), "w") as f:
        f.write(
            "{}   Final AUC:{:.3f}  Final AP:{:.3f}    Best AUC:{:.3f}  Best AP:{:.3f}\n".format(
                method_name,
                summary["final_auc"],
                summary["final_ap"],
                summary["best_auc"],
                summary["best_ap"],
            )
        )


def evaluate_weak_refine_module(cfgs, refine_in=None):
    refine_in = list(refine_in or ["inter_dis", "intra_dis"])
    device = _get_device(cfgs)
    clip_loader_overrides = _get_clip_loader_overrides(cfgs)
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    batch_size = int(_get_section(cfgs, "CLIP").get("val_bs", 1))
    module_a = _load_weak_ddad_ensemble(cfgs, "weak_a")
    module_b = _load_weak_ddad_ensemble(cfgs, "weak_b")
    refine_net, refine_runtime_cfg, refine_in_channels = _load_weak_refine_model(cfgs, device, refine_in)
    test_loader = _build_multibranch_loader(
        cfgs,
        subset="official_test",
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        include_clip=False,
        num_workers_override=clip_loader_overrides["val_workers"],
        persistent_workers_override=clip_loader_overrides["val_persistent_workers"],
        prefetch_factor_override=clip_loader_overrides["val_prefetch_factor"],
        cache_images_override=clip_loader_overrides["cache_images"],
        cache_clip_images_override=False,
        dataset_kwargs=dataset_kwargs,
    )
    print(
        "=> eval_weak_r preprocess: band_mode={} in_channels={} score_topk_ratio={:.2%}".format(
            refine_runtime_cfg["band_mode"],
            refine_in_channels,
            refine_runtime_cfg["score_topk_ratio"],
        )
    )
    metrics, rows = _evaluate_weak_refine_loader(
        test_loader,
        cfgs,
        module_a,
        module_b,
        refine_net,
        refine_in,
        refine_runtime_cfg,
        device,
    )
    output_dir = _checkpoint_out_dir(cfgs)
    rows_path = os.path.join(output_dir, "eval_weak_r_scores.csv")
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    payload = {
        "method": "weak_R_dual" if len(refine_in) == 2 else "weak_R_intra",
        "auc": float(metrics["auc"]),
        "ap": float(metrics["ap"]),
        "count": int(metrics["count"]),
        "checkpoint": _weak_refine_checkpoint_path(cfgs, refine_in),
        "scores": rows_path,
    }
    result_str = "{} - AUC:{:.3f}  AP:{:.3f}".format(payload["method"], payload["auc"], payload["ap"])
    print(result_str)
    with open(os.path.join(output_dir, "eval_weak_r_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(output_dir, "eval_weak_r_results.txt"), "w") as f:
        f.write(result_str + "\n")


def train_clip_teacher(cfgs):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    loader_overrides = _get_clip_loader_overrides(cfgs)
    amp_enabled = _amp_enabled(clip_cfg, device)
    batch_size = int(clip_cfg.get("bs", 8))
    val_batch_size = int(clip_cfg.get("val_bs", batch_size))
    synthetic_probability = float(clip_cfg.get("synthetic_probability", 0.5))
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    weak_cfg = _get_weakclip_cfg(cfgs)

    train_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_train_normal",
        batch_size=batch_size,
        shuffle=True,
        synthetic_probability=synthetic_probability,
        drop_last=True,
        num_workers_override=loader_overrides["train_workers"],
        persistent_workers_override=loader_overrides["train_persistent_workers"],
        prefetch_factor_override=loader_overrides["train_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    synthetic_val_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_val_normal",
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=synthetic_probability,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    clean_val_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_val_normal",
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    unlabeled_val_loader = _build_multibranch_loader(
        cfgs,
        subset="unlabeled_val_pool",
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )

    model = _build_clip_model(cfgs, device)
    _configure_clip_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(clip_cfg.get("lr", 1.0e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(clip_cfg.get("weight_decay", 1.0e-4)),
    )
    scaler = _make_grad_scaler(amp_enabled)

    checkpoint_paths = _clip_stage_checkpoint_paths(cfgs, "clip_teacher")
    writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_clip_teacher"))
    best_score = -float("inf")
    num_epoch = int(clip_cfg.get("num_epoch", 30))

    for epoch in range(1, num_epoch + 1):
        start = time.time()
        train_stats = _run_clip_teacher_epoch(
            train_loader,
            model,
            device,
            optimizer,
            scaler,
            amp_enabled,
            weak_cfg["teacher_clean_loss_weight"],
            weak_cfg["teacher_synthetic_cls_weight"],
            weak_cfg["teacher_seg_loss_weight"],
            weak_cfg["teacher_bg_suppression_weight"],
        )

        model.eval()
        synthetic_metrics = _evaluate_clip_image_metrics(
            synthetic_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        clean_stats = _evaluate_clip_score_stats(
            clean_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        unlabeled_hidden_metrics = _evaluate_hidden_label_metrics(
            unlabeled_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        score = _weakclip_checkpoint_score(unlabeled_hidden_metrics, clean_stats)

        writer.add_scalar("train_loss", train_stats["total"], epoch)
        writer.add_scalar("train_cls_loss", train_stats["cls"], epoch)
        writer.add_scalar("train_seg_loss", train_stats["seg"], epoch)
        writer.add_scalar("val_synthetic_auc", synthetic_metrics["auc"], epoch)
        writer.add_scalar("val_synthetic_ap", synthetic_metrics["ap"], epoch)
        writer.add_scalar("val_clean_score_mean", clean_stats["mean"], epoch)
        writer.add_scalar("val_unlabeled_hidden_auc", unlabeled_hidden_metrics["auc"], epoch)
        writer.add_scalar("val_unlabeled_hidden_ap", unlabeled_hidden_metrics["ap"], epoch)
        print(
            "clip_teacher Epoch[{}/{}]\tTime:{:.1f}s\tLoss:{:.4f}\tCls:{:.4f}\tSeg:{:.4f}\tSyn AUC:{:.4f}\tSyn AP:{:.4f}\tValHidden AUC:{:.4f}\tValHidden AP:{:.4f}\tClean Mean:{:.4f}\tAMP:{}".format(
                epoch,
                num_epoch,
                time.time() - start,
                train_stats["total"],
                train_stats["cls"],
                train_stats["seg"],
                synthetic_metrics["auc"],
                synthetic_metrics["ap"],
                unlabeled_hidden_metrics["auc"],
                unlabeled_hidden_metrics["ap"],
                clean_stats["mean"],
                amp_enabled,
            )
        )

        checkpoint_extra = {
            "epoch": epoch,
            "score": score,
            "train_total": train_stats["total"],
            "train_cls": train_stats["cls"],
            "train_seg": train_stats["seg"],
            "synthetic_auc": synthetic_metrics["auc"],
            "synthetic_ap": synthetic_metrics["ap"],
            "unlabeled_val_hidden_auc": unlabeled_hidden_metrics["auc"],
            "unlabeled_val_hidden_ap": unlabeled_hidden_metrics["ap"],
            "clean_score_mean": clean_stats["mean"],
            "clean_score_std": clean_stats["std"],
        }
        _save_checkpoint(checkpoint_paths["last"], model, optimizer, extra=checkpoint_extra)
        if score > best_score:
            best_score = score
            _save_checkpoint(checkpoint_paths["best"], model, optimizer, extra=checkpoint_extra)

    writer.close()


def score_unlabeled_with_teacher(cfgs):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    loader_overrides = _get_clip_loader_overrides(cfgs)
    amp_enabled = _amp_enabled(clip_cfg, device)
    batch_size = int(clip_cfg.get("val_bs", clip_cfg.get("bs", 4)))
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    weak_cfg = _get_weakclip_cfg(cfgs)
    output_paths = _teacher_unlabeled_score_paths(cfgs)

    train_loader = _build_multibranch_loader(
        cfgs,
        subset="unlabeled_train_pool",
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    val_loader = _build_multibranch_loader(
        cfgs,
        subset="unlabeled_val_pool",
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    model = _load_clip_stage_model(cfgs, device, "clip_teacher")
    model.eval()
    weak_module_a = _load_weak_ddad_ensemble(cfgs, "weak_a")
    weak_module_b = _load_weak_ddad_ensemble(cfgs, "weak_b")

    train_rows = _collect_clip_score_rows(train_loader, model, device, amp_enabled=amp_enabled)
    val_rows = _collect_clip_score_rows(val_loader, model, device, amp_enabled=amp_enabled)
    train_ddad_rows = _collect_ddad_score_rows(train_loader, cfgs, weak_module_a, weak_module_b, device)
    val_ddad_rows = _collect_ddad_score_rows(val_loader, cfgs, weak_module_a, weak_module_b, device)
    train_refined_rows = []
    val_refined_rows = []
    train_safd_rows = []
    val_safd_rows = []
    if weak_cfg["use_safd_score"] and not weak_cfg["use_refine_score"]:
        raise RuntimeError("WeakCLIP.use_safd_score=true requires WeakCLIP.use_refine_score=true.")
    if weak_cfg["use_refine_score"]:
        refine_in = ["inter_dis", "intra_dis"]
        refine_net, refine_runtime_cfg, _ = _load_weak_refine_model(cfgs, device, refine_in)
        train_refined_rows = _collect_refined_ddad_score_rows(
            train_loader,
            cfgs,
            weak_module_a,
            weak_module_b,
            refine_net,
            refine_in,
            refine_runtime_cfg,
            device,
        )
        val_refined_rows = _collect_refined_ddad_score_rows(
            val_loader,
            cfgs,
            weak_module_a,
            weak_module_b,
            refine_net,
            refine_in,
            refine_runtime_cfg,
            device,
        )
        if weak_cfg["use_safd_score"]:
            train_safd_rows = _collect_safd_ddad_score_rows(
                train_loader,
                cfgs,
                weak_module_a,
                weak_module_b,
                refine_net,
                refine_in,
                refine_runtime_cfg,
                device,
            )
            val_safd_rows = _collect_safd_ddad_score_rows(
                val_loader,
                cfgs,
                weak_module_a,
                weak_module_b,
                refine_net,
                refine_in,
                refine_runtime_cfg,
                device,
            )

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    train_ddad_df = pd.DataFrame(train_ddad_rows)
    val_ddad_df = pd.DataFrame(val_ddad_rows)
    train_df = train_df.merge(train_ddad_df, on=["img_id", "group_id", "image_name", "hidden_label"], how="left")
    val_df = val_df.merge(val_ddad_df, on=["img_id", "group_id", "image_name", "hidden_label"], how="left")
    if weak_cfg["use_refine_score"]:
        train_refined_df = pd.DataFrame(train_refined_rows)
        val_refined_df = pd.DataFrame(val_refined_rows)
        train_df = train_df.merge(train_refined_df, on=["img_id", "group_id", "image_name", "hidden_label"], how="left")
        val_df = val_df.merge(val_refined_df, on=["img_id", "group_id", "image_name", "hidden_label"], how="left")
    if weak_cfg["use_safd_score"]:
        train_safd_df = pd.DataFrame(train_safd_rows)
        val_safd_df = pd.DataFrame(val_safd_rows)
        train_df = train_df.merge(train_safd_df, on=["img_id", "group_id", "image_name", "hidden_label"], how="left")
        val_df = val_df.merge(val_safd_df, on=["img_id", "group_id", "image_name", "hidden_label"], how="left")
    for df in [train_df, val_df]:
        for column_name in [
            "ddad_score",
            "ddad_inter_score",
            "ddad_intra_score",
            "refined_ddad_score",
            "safd_ddad_score",
            "safd_normal_score",
            "safd_ddad_band_mean",
            "safd_ddad_band_max",
            "safd_normal_band_mean",
            "safd_normal_band_max",
            "refined_heatmap_peak",
            "refined_heatmap_topk_mean",
            "refined_heatmap_foreground_ratio",
            "refined_heatmap_entropy",
        ]:
            if column_name in df.columns:
                df[column_name] = df[column_name].fillna(0.0)

    train_df = _attach_rank_and_percentile(train_df, "ddad_score", "ddad")
    val_df = _attach_rank_and_percentile(val_df, "ddad_score", "ddad")
    train_df = _attach_rank_and_percentile(train_df, "score", "clip_score")
    val_df = _attach_rank_and_percentile(val_df, "score", "clip_score")
    train_df = _attach_rank_and_percentile(train_df, "localization_confidence", "localization_confidence")
    val_df = _attach_rank_and_percentile(val_df, "localization_confidence", "localization_confidence")
    train_df = _attach_rank_and_percentile(train_df, "background_consistency", "background_consistency")
    val_df = _attach_rank_and_percentile(val_df, "background_consistency", "background_consistency")
    ddad_percentile_column = "ddad_percentile"
    if weak_cfg["use_refine_score"]:
        train_df = _attach_rank_and_percentile(train_df, "refined_ddad_score", "refined_ddad")
        val_df = _attach_rank_and_percentile(val_df, "refined_ddad_score", "refined_ddad")
        ddad_percentile_column = "refined_ddad_percentile"
    if weak_cfg["use_safd_score"]:
        safd_score_column = "safd_normal_score" if weak_cfg["safd_score_mode"] == "normal_bank" else "safd_ddad_score"
        safd_prefix = "safd_normal" if weak_cfg["safd_score_mode"] == "normal_bank" else "safd_ddad"
        safd_percentile_column = "{}_percentile".format(safd_prefix)
        train_df = _attach_rank_and_percentile(train_df, safd_score_column, safd_prefix)
        val_df = _attach_rank_and_percentile(val_df, safd_score_column, safd_prefix)
        if weak_cfg["safd_score_mode"] == "normal_bank":
            train_df["safd_ddad_rank"] = train_df["safd_normal_rank"]
            train_df["safd_ddad_percentile"] = train_df["safd_normal_percentile"]
            val_df["safd_ddad_rank"] = val_df["safd_normal_rank"]
            val_df["safd_ddad_percentile"] = val_df["safd_normal_percentile"]
        for df in [train_df, val_df]:
            refined_percentile = df[ddad_percentile_column].astype(float).clip(0.0, 1.0)
            safd_percentile = df[safd_percentile_column].astype(float).clip(0.0, 1.0)
            df["safd_refined_agreement_score"] = np.sqrt(refined_percentile * safd_percentile)
            df["safd_normal_agreement_score"] = np.sqrt((1.0 - refined_percentile) * (1.0 - safd_percentile))
            df["abnormal_joint_score"] = (
                weak_cfg["refined_ddad_joint_weight"] * refined_percentile +
                weak_cfg["clip_joint_weight"] * df["clip_score_percentile"].astype(float) +
                weak_cfg["localization_joint_weight"] * df["localization_confidence_percentile"].astype(float)
            )
            df["normal_joint_score"] = (
                weak_cfg["refined_ddad_joint_weight"] * (1.0 - refined_percentile) +
                weak_cfg["clip_joint_weight"] * (1.0 - df["clip_score_percentile"].astype(float)) +
                weak_cfg["localization_joint_weight"] * df["background_consistency_percentile"].astype(float)
            )
            df["topup_abnormal_score"] = (
                weak_cfg["topup_abnormal_base_weight"] * df["abnormal_joint_score"].astype(float) +
                weak_cfg["topup_abnormal_safd_weight"] * df["safd_refined_agreement_score"].astype(float)
            )
    else:
        train_df["abnormal_joint_score"] = (
            0.70 * train_df[ddad_percentile_column].astype(float) +
            0.20 * train_df["clip_score_percentile"].astype(float) +
            0.10 * train_df["localization_confidence_percentile"].astype(float)
        )
        train_df["normal_joint_score"] = (
            0.70 * (1.0 - train_df[ddad_percentile_column].astype(float)) +
            0.20 * (1.0 - train_df["clip_score_percentile"].astype(float)) +
            0.10 * train_df["background_consistency_percentile"].astype(float)
        )
        val_df["abnormal_joint_score"] = (
            0.70 * val_df[ddad_percentile_column].astype(float) +
            0.20 * val_df["clip_score_percentile"].astype(float) +
            0.10 * val_df["localization_confidence_percentile"].astype(float)
        )
        val_df["normal_joint_score"] = (
            0.70 * (1.0 - val_df[ddad_percentile_column].astype(float)) +
            0.20 * (1.0 - val_df["clip_score_percentile"].astype(float)) +
            0.10 * val_df["background_consistency_percentile"].astype(float)
        )
        train_df["topup_abnormal_score"] = train_df["abnormal_joint_score"]
        val_df["topup_abnormal_score"] = val_df["abnormal_joint_score"]

    train_df.sort_values("abnormal_joint_score", ascending=False).to_csv(output_paths["train_scores"], index=False)
    val_df.sort_values("abnormal_joint_score", ascending=False).to_csv(output_paths["val_scores"], index=False)
    summary = {
        "train": _score_distribution_summary(train_rows),
        "val": _score_distribution_summary(val_rows),
        "train_ddad": _score_distribution_summary([
            {"score": row["ddad_score"]} for row in train_ddad_rows
        ]),
        "val_ddad": _score_distribution_summary([
            {"score": row["ddad_score"]} for row in val_ddad_rows
        ]),
        "train_abnormal_selection": _score_distribution_summary(
            [{"score": float(score)} for score in train_df["abnormal_joint_score"].astype(float).tolist()]
        ),
        "val_abnormal_selection": _score_distribution_summary(
            [{"score": float(score)} for score in val_df["abnormal_joint_score"].astype(float).tolist()]
        ),
        "train_normal_selection": _score_distribution_summary(
            [{"score": float(score)} for score in train_df["normal_joint_score"].astype(float).tolist()]
        ),
        "val_normal_selection": _score_distribution_summary(
            [{"score": float(score)} for score in val_df["normal_joint_score"].astype(float).tolist()]
        ),
        "train_topup_abnormal_selection": _score_distribution_summary(
            [{"score": float(score)} for score in train_df["topup_abnormal_score"].astype(float).tolist()]
        ),
        "val_topup_abnormal_selection": _score_distribution_summary(
            [{"score": float(score)} for score in val_df["topup_abnormal_score"].astype(float).tolist()]
        ),
        "clip_hidden_debug_train": _hidden_label_metrics_from_rows(train_rows),
        "clip_hidden_debug_val": _hidden_label_metrics_from_rows(val_rows),
        "ddad_hidden_debug_train": _hidden_label_metrics_from_rows(train_ddad_rows, score_key="ddad_score"),
        "ddad_hidden_debug_val": _hidden_label_metrics_from_rows(val_ddad_rows, score_key="ddad_score"),
        "refined_ddad_hidden_debug_train": _hidden_label_metrics_from_rows(train_refined_rows, score_key="refined_ddad_score") if weak_cfg["use_refine_score"] else {"auc": 0.0, "ap": 0.0, "count": 0},
        "refined_ddad_hidden_debug_val": _hidden_label_metrics_from_rows(val_refined_rows, score_key="refined_ddad_score") if weak_cfg["use_refine_score"] else {"auc": 0.0, "ap": 0.0, "count": 0},
        "safd_ddad_hidden_debug_train": _hidden_label_metrics_from_rows(train_safd_rows, score_key="safd_ddad_score") if weak_cfg["use_safd_score"] else {"auc": 0.0, "ap": 0.0, "count": 0},
        "safd_ddad_hidden_debug_val": _hidden_label_metrics_from_rows(val_safd_rows, score_key="safd_ddad_score") if weak_cfg["use_safd_score"] else {"auc": 0.0, "ap": 0.0, "count": 0},
        "safd_normal_hidden_debug_train": _hidden_label_metrics_from_rows(train_safd_rows, score_key="safd_normal_score") if weak_cfg["use_safd_score"] else {"auc": 0.0, "ap": 0.0, "count": 0},
        "safd_normal_hidden_debug_val": _hidden_label_metrics_from_rows(val_safd_rows, score_key="safd_normal_score") if weak_cfg["use_safd_score"] else {"auc": 0.0, "ap": 0.0, "count": 0},
        "joint_hidden_debug_train": _hidden_label_metrics_from_rows(train_df.to_dict("records"), score_key="abnormal_joint_score"),
        "joint_hidden_debug_val": _hidden_label_metrics_from_rows(val_df.to_dict("records"), score_key="abnormal_joint_score"),
        "selection_score_source": ddad_percentile_column,
        "use_safd_score": weak_cfg["use_safd_score"],
        "safd_score_mode": weak_cfg["safd_score_mode"],
        "safd_fusion_mode": weak_cfg["safd_fusion_mode"],
        "safd_apply_scope": weak_cfg["safd_apply_scope"],
        "joint_weights": {
            "refined_ddad": weak_cfg["refined_ddad_joint_weight"],
            "safd": 0.0,
            "clip": weak_cfg["clip_joint_weight"],
            "localization": weak_cfg["localization_joint_weight"],
        },
        "topup_weights": {
            "base": weak_cfg["topup_abnormal_base_weight"],
            "safd": weak_cfg["topup_abnormal_safd_weight"],
        },
        "selection_ratios": {
            "pseudo_top_abnormal_ratio": weak_cfg["pseudo_top_abnormal_ratio"],
            "pseudo_bottom_normal_ratio": weak_cfg["pseudo_bottom_normal_ratio"],
        },
    }
    if weak_cfg["use_refine_score"]:
        summary["train_refined_ddad"] = _score_distribution_summary([
            {"score": row["refined_ddad_score"]} for row in train_refined_rows
        ])
        summary["val_refined_ddad"] = _score_distribution_summary([
            {"score": row["refined_ddad_score"]} for row in val_refined_rows
        ])
    if weak_cfg["use_safd_score"]:
        summary["train_safd_ddad"] = _score_distribution_summary([
            {"score": row["safd_ddad_score"]} for row in train_safd_rows
        ])
        summary["val_safd_ddad"] = _score_distribution_summary([
            {"score": row["safd_ddad_score"]} for row in val_safd_rows
        ])
        summary["train_safd_normal"] = _score_distribution_summary([
            {"score": row["safd_normal_score"]} for row in train_safd_rows
        ])
        summary["val_safd_normal"] = _score_distribution_summary([
            {"score": row["safd_normal_score"]} for row in val_safd_rows
        ])
    with open(output_paths["summary"], "w") as f:
        json.dump(summary, f, indent=2)
    if weak_cfg["save_hidden_label_debug"]:
        _save_hidden_label_debug(train_rows, output_paths["train_debug"])
        _save_hidden_label_debug(val_rows, output_paths["val_debug"])
    print("=> Saved unlabeled train scores to {}".format(output_paths["train_scores"]))
    print("=> Saved unlabeled val scores to {}".format(output_paths["val_scores"]))


def select_pseudo_labels(cfgs):
    weak_cfg = _get_weakclip_cfg(cfgs)
    output_paths = _teacher_unlabeled_score_paths(cfgs)
    if not os.path.exists(output_paths["train_scores"]):
        raise FileNotFoundError(
            "Expected unlabeled train scores at {}. Run score_unlabeled first.".format(output_paths["train_scores"])
        )
    if not os.path.exists(output_paths["val_scores"]):
        raise FileNotFoundError(
            "Expected unlabeled val scores at {}. Run score_unlabeled first.".format(output_paths["val_scores"])
        )

    train_df = pd.read_csv(output_paths["train_scores"])
    val_df = pd.read_csv(output_paths["val_scores"])
    if len(train_df) == 0:
        raise RuntimeError("Unlabeled train score file is empty: {}".format(output_paths["train_scores"]))

    abnormal_count = max(1, int(np.ceil(len(train_df) * weak_cfg["pseudo_top_abnormal_ratio"])))
    normal_count = max(1, int(np.ceil(len(train_df) * weak_cfg["pseudo_bottom_normal_ratio"])))
    required_columns = {
        "ddad_score",
        "ddad_percentile",
        "abnormal_joint_score",
        "normal_joint_score",
        "localization_confidence",
        "background_consistency",
    }
    if not required_columns.issubset(set(train_df.columns)) or not required_columns.issubset(set(val_df.columns)):
        raise RuntimeError(
            "Unlabeled score files are missing CLIP/DDAD joint-score columns. Please rerun score_unlabeled with the current code."
        )
    ddad_selection_column = "ddad_percentile"
    if weak_cfg["use_refine_score"]:
        refined_required_columns = {"refined_ddad_score", "refined_ddad_percentile"}
        if (
            not refined_required_columns.issubset(set(train_df.columns)) or
            not refined_required_columns.issubset(set(val_df.columns))
        ):
            raise RuntimeError(
                "WeakCLIP.use_refine_score=true but unlabeled score files do not contain refined DDAD columns. "
                "Run weak_r and then rerun score_unlabeled."
            )
        ddad_selection_column = "refined_ddad_percentile"
    if weak_cfg["use_safd_score"]:
        safd_required_columns = {"safd_ddad_score", "safd_ddad_percentile"}
        if (
            not safd_required_columns.issubset(set(train_df.columns)) or
            not safd_required_columns.issubset(set(val_df.columns))
        ):
            raise RuntimeError(
                "WeakCLIP.use_safd_score=true but unlabeled score files do not contain SAFD columns. "
                "Rerun score_unlabeled with the current code."
            )

    val_loc_threshold = float(np.quantile(val_df["localization_confidence"].astype(float).to_numpy(), 0.80))
    val_bg_threshold = float(np.quantile(val_df["background_consistency"].astype(float).to_numpy(), 0.80))

    abnormal_mask = (
        (train_df["score"].astype(float) >= weak_cfg["pseudo_abnormal_score_min"]) &
        (train_df["localization_confidence"].astype(float) >= val_loc_threshold) &
        (train_df[ddad_selection_column].astype(float) >= weak_cfg["ddad_abnormal_percentile_min"])
    )
    strict_abnormal_candidate_count = int(abnormal_mask.sum())
    pseudo_abnormal = train_df[abnormal_mask].copy()
    pseudo_abnormal = pseudo_abnormal.nlargest(abnormal_count, "abnormal_joint_score")
    pseudo_abnormal["pseudo_source"] = "strict_abnormal"
    abnormal_topup_count = 0
    if len(pseudo_abnormal) < abnormal_count:
        fill_count = abnormal_count - len(pseudo_abnormal)
        selected_abnormal_ids = set(pseudo_abnormal["img_id"].astype(str).tolist())
        abnormal_fill_pool = train_df[
            ~train_df["img_id"].astype(str).isin(selected_abnormal_ids)
        ].copy()
        topup_score_column = "topup_abnormal_score" if "topup_abnormal_score" in abnormal_fill_pool.columns else "abnormal_joint_score"
        abnormal_fill = abnormal_fill_pool.nlargest(fill_count, topup_score_column).copy()
        abnormal_topup_count = int(len(abnormal_fill))
        if len(abnormal_fill) > 0:
            abnormal_fill["pseudo_source"] = "topup_abnormal"
            pseudo_abnormal = pd.concat([pseudo_abnormal, abnormal_fill], axis=0, ignore_index=True)
            pseudo_abnormal = pseudo_abnormal.drop_duplicates(subset=["img_id"], keep="first")

    normal_mask = (
        (train_df["score"].astype(float) <= weak_cfg["pseudo_normal_score_max"]) &
        (train_df["background_consistency"].astype(float) >= val_bg_threshold) &
        (train_df[ddad_selection_column].astype(float) <= weak_cfg["ddad_normal_percentile_max"])
    )
    pseudo_normal = train_df[normal_mask].copy()
    pseudo_normal = pseudo_normal.nlargest(normal_count, "normal_joint_score")
    if len(pseudo_normal) == 0:
        pseudo_normal = train_df.nlargest(normal_count, "normal_joint_score").copy()
    pseudo_normal["pseudo_source"] = "strict_normal"

    pseudo_abnormal["pseudo_kind"] = "abnormal"
    pseudo_normal["pseudo_kind"] = "normal"
    selected_df = pd.concat([pseudo_abnormal, pseudo_normal], axis=0, ignore_index=True)
    selected_df = selected_df.drop_duplicates(subset=["img_id"], keep="first").reset_index(drop=True)

    pseudo_targets, pseudo_weights = [], []
    for row in selected_df.itertuples():
        if str(row.pseudo_kind) == "abnormal":
            if str(getattr(row, "pseudo_source", "")) == "topup_abnormal":
                target_floor = weak_cfg["pseudo_topup_abnormal_min_target"]
            else:
                target_floor = weak_cfg["pseudo_abnormal_min_target"]
            target = _weakclip_selection_targets(max(float(row.score), target_floor), weak_cfg)
            weight = _joint_confidence_weight(float(row.abnormal_joint_score))
            if str(getattr(row, "pseudo_source", "")) == "topup_abnormal":
                weight = max(0.2, weight * weak_cfg["pseudo_topup_abnormal_weight_scale"])
                if weak_cfg["pseudo_topup_abnormal_use_safd_gate"] and hasattr(row, "safd_refined_agreement_score"):
                    safd_gate = 0.5 + 0.5 * float(row.safd_refined_agreement_score)
                    weight = max(0.2, weight * safd_gate)
        else:
            target = _weakclip_selection_targets(min(float(row.score), weak_cfg["pseudo_normal_max_target"]), weak_cfg)
            weight = _joint_confidence_weight(float(row.normal_joint_score))
        pseudo_targets.append(target)
        pseudo_weights.append(weight)
    selected_df["teacher_score"] = selected_df["score"].astype(float)
    selected_df["pseudo_target"] = pseudo_targets
    selected_df["pseudo_weight"] = pseudo_weights
    manifest_columns = [
        "img_id",
        "group_id",
        "image_name",
        "teacher_score",
        "ddad_score",
        "ddad_rank",
        "ddad_percentile",
        "refined_ddad_score",
        "refined_ddad_rank",
        "refined_ddad_percentile",
        "safd_ddad_score",
        "safd_ddad_rank",
        "safd_ddad_percentile",
        "safd_ddad_band_mean",
        "safd_ddad_band_max",
        "safd_normal_score",
        "safd_normal_rank",
        "safd_normal_percentile",
        "safd_normal_band_mean",
        "safd_normal_band_max",
        "safd_refined_agreement_score",
        "safd_normal_agreement_score",
        "abnormal_joint_score",
        "topup_abnormal_score",
        "normal_joint_score",
        "heatmap_peak",
        "heatmap_topk_mean",
        "heatmap_foreground_ratio",
        "heatmap_entropy",
        "localization_confidence",
        "clip_score_percentile",
        "localization_confidence_percentile",
        "background_consistency_percentile",
        "background_consistency",
        "refined_heatmap_peak",
        "refined_heatmap_topk_mean",
        "refined_heatmap_foreground_ratio",
        "refined_heatmap_entropy",
        "pseudo_target",
        "pseudo_weight",
        "pseudo_kind",
        "pseudo_source",
        "hidden_label",
    ]
    selected_df = selected_df[[column for column in manifest_columns if column in selected_df.columns]]
    selected_df.to_csv(output_paths["selected_manifest"], index=False)

    summary = {
        "selection": {
            "pseudo_top_abnormal_ratio": weak_cfg["pseudo_top_abnormal_ratio"],
            "pseudo_bottom_normal_ratio": weak_cfg["pseudo_bottom_normal_ratio"],
            "val_localization_threshold": val_loc_threshold,
            "val_background_threshold": val_bg_threshold,
            "pseudo_abnormal_score_min": weak_cfg["pseudo_abnormal_score_min"],
            "pseudo_normal_score_max": weak_cfg["pseudo_normal_score_max"],
            "ddad_abnormal_percentile_min": weak_cfg["ddad_abnormal_percentile_min"],
            "ddad_normal_percentile_max": weak_cfg["ddad_normal_percentile_max"],
            "ddad_selection_column": ddad_selection_column,
            "target_abnormal_count": int(abnormal_count),
            "strict_abnormal_candidate_count": strict_abnormal_candidate_count,
            "abnormal_topup_count": abnormal_topup_count,
            "selected_count": int(len(selected_df)),
            "selected_abnormal_count": int((selected_df["pseudo_kind"] == "abnormal").sum()),
            "selected_normal_count": int((selected_df["pseudo_kind"] == "normal").sum()),
            "pseudo_source_counts": {
                str(key): int(value)
                for key, value in selected_df["pseudo_source"].value_counts().to_dict().items()
            } if "pseudo_source" in selected_df.columns else {},
            "topup_abnormal_min_target": weak_cfg["pseudo_topup_abnormal_min_target"],
            "topup_abnormal_weight_scale": weak_cfg["pseudo_topup_abnormal_weight_scale"],
            "topup_abnormal_use_safd_gate": weak_cfg["pseudo_topup_abnormal_use_safd_gate"],
            "topup_abnormal_safd_weight": weak_cfg["topup_abnormal_safd_weight"],
            "topup_abnormal_base_weight": weak_cfg["topup_abnormal_base_weight"],
        },
        "unlabeled_val_distribution": _score_distribution_summary(val_df.to_dict("records")),
        "selected_train_distribution": _score_distribution_summary(
            [
                {"score": score}
                for score in selected_df["teacher_score"].astype(float).tolist()
            ]
        ),
    }
    if weak_cfg["save_hidden_label_debug"] and "hidden_label" in selected_df.columns:
        debug_df = selected_df[selected_df["hidden_label"] >= 0]
        if len(debug_df) > 0:
            summary["selected_hidden_debug"] = {
                "count": int(len(debug_df)),
                "teacher_metrics": _classification_metrics(
                    debug_df["hidden_label"].astype(int).tolist(),
                    debug_df["teacher_score"].astype(float).tolist(),
                ),
                "ddad_metrics": _classification_metrics(
                    debug_df["hidden_label"].astype(int).tolist(),
                    debug_df["ddad_score"].astype(float).tolist(),
                ),
                "refined_ddad_metrics": _classification_metrics(
                    debug_df["hidden_label"].astype(int).tolist(),
                    debug_df["refined_ddad_score"].astype(float).tolist(),
                ) if "refined_ddad_score" in debug_df.columns else None,
                "safd_ddad_metrics": _classification_metrics(
                    debug_df["hidden_label"].astype(int).tolist(),
                    debug_df["safd_ddad_score"].astype(float).tolist(),
                ) if "safd_ddad_score" in debug_df.columns else None,
                "safd_normal_metrics": _classification_metrics(
                    debug_df["hidden_label"].astype(int).tolist(),
                    debug_df["safd_normal_score"].astype(float).tolist(),
                ) if "safd_normal_score" in debug_df.columns else None,
                "abnormal_joint_metrics": _classification_metrics(
                    debug_df["hidden_label"].astype(int).tolist(),
                    debug_df["abnormal_joint_score"].astype(float).tolist(),
                ),
            }
            if "pseudo_source" in debug_df.columns:
                source_debug = {}
                for source_name, source_df in debug_df.groupby("pseudo_source"):
                    source_debug[str(source_name)] = {
                        "count": int(len(source_df)),
                        "positive_count": int(source_df["hidden_label"].astype(int).sum()),
                        "positive_rate": float(source_df["hidden_label"].astype(int).mean()),
                        "mean_pseudo_weight": float(source_df["pseudo_weight"].astype(float).mean()),
                    }
                summary["selected_hidden_debug"]["by_pseudo_source"] = source_debug
    with open(output_paths["selected_summary"], "w") as f:
        json.dump(summary, f, indent=2)
    print("=> Saved pseudo-label manifest to {}".format(output_paths["selected_manifest"]))


def train_clip_student(cfgs):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    loader_overrides = _get_clip_loader_overrides(cfgs)
    amp_enabled = _amp_enabled(clip_cfg, device)
    batch_size = int(clip_cfg.get("bs", 8))
    val_batch_size = int(clip_cfg.get("val_bs", batch_size))
    synthetic_probability = float(clip_cfg.get("synthetic_probability", 0.5))
    weak_cfg = _get_weakclip_cfg(cfgs)
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    output_paths = _teacher_unlabeled_score_paths(cfgs)
    if not os.path.exists(output_paths["selected_manifest"]):
        raise FileNotFoundError(
            "Expected pseudo-label manifest at {}. Run select_pseudo first.".format(output_paths["selected_manifest"])
        )

    synthetic_train_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_train_normal",
        batch_size=batch_size,
        shuffle=True,
        synthetic_probability=synthetic_probability,
        drop_last=True,
        num_workers_override=loader_overrides["train_workers"],
        persistent_workers_override=loader_overrides["train_persistent_workers"],
        prefetch_factor_override=loader_overrides["train_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    clean_train_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_train_normal",
        batch_size=batch_size,
        shuffle=True,
        synthetic_probability=0.0,
        drop_last=True,
        num_workers_override=loader_overrides["train_workers"],
        persistent_workers_override=loader_overrides["train_persistent_workers"],
        prefetch_factor_override=loader_overrides["train_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    pseudo_loader = _build_pseudo_label_loader(
        cfgs,
        output_paths["selected_manifest"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    synthetic_val_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_val_normal",
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=synthetic_probability,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    clean_val_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_val_normal",
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    unlabeled_val_loader = _build_multibranch_loader(
        cfgs,
        subset="unlabeled_val_pool",
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )

    model = _build_clip_model(cfgs, device)
    teacher_checkpoint = _clip_stage_checkpoint_paths(cfgs, "clip_teacher")["best"]
    _load_checkpoint(teacher_checkpoint, model, map_location=device)
    _configure_clip_trainable_parameters(model)
    teacher_model = _load_clip_stage_model(cfgs, device, "clip_teacher")
    teacher_model.eval()
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(clip_cfg.get("lr", 1.0e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(clip_cfg.get("weight_decay", 1.0e-4)),
    )
    scaler = _make_grad_scaler(amp_enabled)

    checkpoint_paths = _clip_stage_checkpoint_paths(cfgs, "clip_student")
    writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_clip_student"))
    best_score = -float("inf")
    num_epoch = int(clip_cfg.get("num_epoch", 30))

    for epoch in range(1, num_epoch + 1):
        start = time.time()
        clean_train_stats = _run_clip_clean_epoch(
            clean_train_loader,
            model,
            device,
            optimizer,
            scaler,
            amp_enabled,
            weak_cfg["student_clean_loss_weight"],
            weak_cfg["student_bg_suppression_weight"],
        )
        synthetic_local_stats = _run_clip_synthetic_local_epoch(
            synthetic_train_loader,
            model,
            device,
            optimizer,
            scaler,
            amp_enabled,
            weak_cfg["student_seg_syn_weight"],
            use_synthetic_cls=weak_cfg["student_use_synthetic_cls"],
        )
        pseudo_stats = _run_clip_pseudo_epoch(
            pseudo_loader,
            model,
            teacher_model,
            device,
            optimizer,
            scaler,
            amp_enabled,
            weak_cfg["student_pseudo_loss_weight"],
            weak_cfg["student_pseudo_loc_weight"],
            weak_cfg["student_bg_suppression_weight"],
        )

        model.eval()
        synthetic_metrics = _evaluate_clip_image_metrics(
            synthetic_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        clean_stats = _evaluate_clip_score_stats(
            clean_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        unlabeled_hidden_metrics = _evaluate_hidden_label_metrics(
            unlabeled_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        score = _weakclip_checkpoint_score(unlabeled_hidden_metrics, clean_stats)

        writer.add_scalar("train_clean_total", clean_train_stats["total"], epoch)
        writer.add_scalar("train_clean_cls", clean_train_stats["cls"], epoch)
        writer.add_scalar("train_clean_bg", clean_train_stats["bg"], epoch)
        writer.add_scalar("train_synthetic_local_total", synthetic_local_stats["total"], epoch)
        writer.add_scalar("train_synthetic_local_seg", synthetic_local_stats["seg"], epoch)
        writer.add_scalar("train_synthetic_local_cls", synthetic_local_stats["cls"], epoch)
        writer.add_scalar("train_pseudo_total", pseudo_stats["total"], epoch)
        writer.add_scalar("train_pseudo_cls", pseudo_stats["cls"], epoch)
        writer.add_scalar("train_pseudo_loc", pseudo_stats["loc"], epoch)
        writer.add_scalar("train_pseudo_bg", pseudo_stats["bg"], epoch)
        writer.add_scalar("val_synthetic_auc", synthetic_metrics["auc"], epoch)
        writer.add_scalar("val_synthetic_ap", synthetic_metrics["ap"], epoch)
        writer.add_scalar("val_clean_score_mean", clean_stats["mean"], epoch)
        writer.add_scalar("val_unlabeled_hidden_auc", unlabeled_hidden_metrics["auc"], epoch)
        writer.add_scalar("val_unlabeled_hidden_ap", unlabeled_hidden_metrics["ap"], epoch)
        print(
            "clip_student Epoch[{}/{}]\tTime:{:.1f}s\tClean:{:.4f}\tSynLocal:{:.4f}\tPseudo:{:.4f}\tPseudoLoc:{:.4f}\tSyn AUC:{:.4f}\tSyn AP:{:.4f}\tValHidden AUC:{:.4f}\tValHidden AP:{:.4f}\tClean Mean:{:.4f}\tAMP:{}".format(
                epoch,
                num_epoch,
                time.time() - start,
                clean_train_stats["total"],
                synthetic_local_stats["total"],
                pseudo_stats["total"],
                pseudo_stats["loc"],
                synthetic_metrics["auc"],
                synthetic_metrics["ap"],
                unlabeled_hidden_metrics["auc"],
                unlabeled_hidden_metrics["ap"],
                clean_stats["mean"],
                amp_enabled,
            )
        )

        checkpoint_extra = {
            "epoch": epoch,
            "score": score,
            "train_clean_total": clean_train_stats["total"],
            "train_clean_cls": clean_train_stats["cls"],
            "train_clean_bg": clean_train_stats["bg"],
            "train_synthetic_local_total": synthetic_local_stats["total"],
            "train_synthetic_local_seg": synthetic_local_stats["seg"],
            "train_synthetic_local_cls": synthetic_local_stats["cls"],
            "train_pseudo_total": pseudo_stats["total"],
            "train_pseudo_cls": pseudo_stats["cls"],
            "train_pseudo_loc": pseudo_stats["loc"],
            "train_pseudo_bg": pseudo_stats["bg"],
            "synthetic_auc": synthetic_metrics["auc"],
            "synthetic_ap": synthetic_metrics["ap"],
            "unlabeled_val_hidden_auc": unlabeled_hidden_metrics["auc"],
            "unlabeled_val_hidden_ap": unlabeled_hidden_metrics["ap"],
            "clean_score_mean": clean_stats["mean"],
            "clean_score_std": clean_stats["std"],
        }
        _save_checkpoint(checkpoint_paths["last"], model, optimizer, extra=checkpoint_extra)
        if score > best_score:
            best_score = score
            _save_checkpoint(checkpoint_paths["best"], model, optimizer, extra=checkpoint_extra)

    writer.close()


def _save_clip_student_fusion_checkpoint(path, model, fusion_head, optimizer=None, extra=None):
    payload = {
        "model": {
            "clip": model.state_dict(),
            "fusion_head": fusion_head.state_dict(),
        }
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def _load_clip_student_fusion_checkpoint(path, model, fusion_head, optimizer=None, map_location=None):
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model", payload)
    if "clip" not in state or "fusion_head" not in state:
        raise RuntimeError("Invalid clip_student_fusion checkpoint: {}".format(path))
    model.load_state_dict(state["clip"])
    fusion_head.load_state_dict(state["fusion_head"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("extra", {})


def train_clip_student_fusion(cfgs):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    loader_overrides = _get_clip_loader_overrides(cfgs)
    amp_enabled = _amp_enabled(clip_cfg, device)
    batch_size = int(clip_cfg.get("bs", 8))
    val_batch_size = int(clip_cfg.get("val_bs", batch_size))
    synthetic_probability = float(clip_cfg.get("synthetic_probability", 0.5))
    weak_cfg = _get_weakclip_cfg(cfgs)
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    output_paths = _teacher_unlabeled_score_paths(cfgs)
    if not os.path.exists(output_paths["selected_manifest"]):
        raise FileNotFoundError(
            "Expected pseudo-label manifest at {}. Run select_pseudo first.".format(output_paths["selected_manifest"])
        )

    pseudo_loader = _build_pseudo_label_loader(
        cfgs,
        output_paths["selected_manifest"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    synthetic_val_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_val_normal",
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=synthetic_probability,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    clean_val_loader = _build_multibranch_loader(
        cfgs,
        subset="clean_val_normal",
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )

    module_a = _load_weak_ddad_ensemble(cfgs, "weak_a")
    module_b = _load_weak_ddad_ensemble(cfgs, "weak_b")
    refine_in = ["inter_dis", "intra_dis"]
    refine_net, refine_runtime_cfg, _ = _load_weak_refine_model(cfgs, device, refine_in)
    refine_net.eval()

    model = _build_clip_model(cfgs, device)
    init_stage = weak_cfg["student_fusion_init_stage"]
    if init_stage not in {"clip_student", "clip_teacher"}:
        raise ValueError(
            "WeakCLIP.student_fusion_init_stage must be 'clip_student' or 'clip_teacher', got '{}'".format(
                init_stage
            )
        )
    init_checkpoint = _clip_stage_checkpoint_paths(cfgs, init_stage)["best"]
    if not os.path.exists(init_checkpoint):
        raise FileNotFoundError(
            "Expected {} checkpoint at {} for clip_student_fusion initialization.".format(
                init_stage,
                init_checkpoint,
            )
        )
    _load_checkpoint(init_checkpoint, model, map_location=device)
    if weak_cfg["student_fusion_freeze_clip"]:
        _freeze_module(model)
    else:
        _configure_clip_trainable_parameters(model)
    fusion_head = DDADGuidedStudentFusionHead(
        in_dim=11,
        hidden_dim=int(weak_cfg.get("student_fusion_hidden_dim", 32)),
        dropout=float(weak_cfg.get("student_fusion_dropout", 0.1)),
    ).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad] + list(fusion_head.parameters())
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=float(clip_cfg.get("lr", 1.0e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(clip_cfg.get("weight_decay", 1.0e-4)),
    )
    scaler = _make_grad_scaler(amp_enabled)

    fusion_loss_weight = float(weak_cfg.get("student_fusion_loss_weight", 1.0))
    clip_aux_loss_weight = float(weak_cfg.get("student_clip_aux_loss_weight", 0.3))
    ddad_map_loss_weight = float(weak_cfg.get("student_ddad_map_loss_weight", 0.05))
    bg_weight = float(weak_cfg.get("student_bg_suppression_weight", 0.05))
    effective_clip_aux_loss_weight = 0.0 if weak_cfg["student_fusion_freeze_clip"] else clip_aux_loss_weight

    checkpoint_paths = _clip_student_fusion_checkpoint_paths(cfgs)
    writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_clip_student_fusion"))
    best_score = -float("inf")
    num_epoch = int(clip_cfg.get("num_epoch", 30))
    print(
        "=> clip_student_fusion init_stage={} freeze_clip={} clip_aux_weight={:.4f} ddad_map_loss_weight={:.4f}".format(
            init_stage,
            weak_cfg["student_fusion_freeze_clip"],
            effective_clip_aux_loss_weight,
            ddad_map_loss_weight,
        )
    )

    for epoch in range(1, num_epoch + 1):
        start = time.time()
        train_stats = _run_clip_student_fusion_pseudo_epoch(
            pseudo_loader,
            model,
            fusion_head,
            cfgs,
            module_a,
            module_b,
            refine_net,
            refine_in,
            refine_runtime_cfg,
            device,
            optimizer,
            scaler,
            amp_enabled,
            fusion_loss_weight,
            effective_clip_aux_loss_weight,
            ddad_map_loss_weight,
            bg_weight,
        )
        model.eval()
        fusion_head.eval()
        synthetic_metrics = _evaluate_clip_image_metrics(
            synthetic_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        clean_stats = _evaluate_clip_score_stats(
            clean_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        score = (
            float(synthetic_metrics["ap"]) -
            float(train_stats["total"]) -
            0.25 * float(clean_stats["mean"])
        )
        writer.add_scalar("train_total", train_stats["total"], epoch)
        writer.add_scalar("train_fused_cls", train_stats["fused_cls"], epoch)
        writer.add_scalar("train_clip_aux", train_stats["clip_aux"], epoch)
        writer.add_scalar("train_ddad_map", train_stats["ddad_map"], epoch)
        writer.add_scalar("train_bg", train_stats["bg"], epoch)
        writer.add_scalar("val_synthetic_auc", synthetic_metrics["auc"], epoch)
        writer.add_scalar("val_synthetic_ap", synthetic_metrics["ap"], epoch)
        writer.add_scalar("val_clean_score_mean", clean_stats["mean"], epoch)
        print(
            "clip_student_fusion Epoch[{}/{}]\tTime:{:.1f}s\tLoss:{:.4f}\tFused:{:.4f}\tClipAux:{:.4f}\tDDADMap:{:.4f}\tBg:{:.4f}\tSyn AUC:{:.4f}\tSyn AP:{:.4f}\tClean Mean:{:.4f}\tAMP:{}".format(
                epoch,
                num_epoch,
                time.time() - start,
                train_stats["total"],
                train_stats["fused_cls"],
                train_stats["clip_aux"],
                train_stats["ddad_map"],
                train_stats["bg"],
                synthetic_metrics["auc"],
                synthetic_metrics["ap"],
                clean_stats["mean"],
                amp_enabled,
            )
        )
        checkpoint_extra = {
            "epoch": epoch,
            "score": score,
            "train_total": train_stats["total"],
            "train_fused_cls": train_stats["fused_cls"],
            "train_clip_aux": train_stats["clip_aux"],
            "train_ddad_map": train_stats["ddad_map"],
            "train_bg": train_stats["bg"],
            "synthetic_auc": synthetic_metrics["auc"],
            "synthetic_ap": synthetic_metrics["ap"],
            "clean_score_mean": clean_stats["mean"],
            "selection_metric": "synthetic_ap - train_total - 0.25 * clean_score_mean",
            "student_fusion_init_stage": init_stage,
            "student_fusion_init_checkpoint": init_checkpoint,
            "student_fusion_freeze_clip": weak_cfg["student_fusion_freeze_clip"],
            "student_ddad_map_loss_weight": ddad_map_loss_weight,
            "student_clip_aux_loss_weight": clip_aux_loss_weight,
            "student_effective_clip_aux_loss_weight": effective_clip_aux_loss_weight,
            "student_fusion_loss_weight": fusion_loss_weight,
        }
        _save_clip_student_fusion_checkpoint(
            checkpoint_paths["last"], model, fusion_head, optimizer=optimizer, extra=checkpoint_extra
        )
        if score > best_score:
            best_score = score
            _save_clip_student_fusion_checkpoint(
                checkpoint_paths["best"], model, fusion_head, optimizer=optimizer, extra=checkpoint_extra
            )

    writer.close()


def _collect_student_fusion_eval_rows(
    loader,
    cfgs,
    model,
    module_a,
    module_b,
    refine_net,
    refine_in,
    refine_runtime_cfg,
    device,
    amp_enabled=False,
):
    rows = []
    with torch.no_grad():
        model.eval()
        refine_net.eval()
        for batch in loader:
            image_224 = _move_tensor(batch["image_224"], device)
            x64 = _move_tensor(batch["image_64"], device)
            refined_features = _compute_refined_ddad_features(
                x64,
                cfgs,
                module_a,
                module_b,
                refine_net,
                refine_in,
                refine_runtime_cfg,
            )
            with _autocast_context(amp_enabled):
                outputs = model(
                    image_224,
                    output_size=batch["image_64"].shape[-2:],
                    return_patch_maps=True,
                )
            clip_logits = outputs["global_logit"].float().view(-1)
            clip_scores = torch.sigmoid(clip_logits).detach().cpu().numpy().tolist()
            clip_logits_list = clip_logits.detach().cpu().numpy().tolist()
            refined_scores = refined_features["refined_score"].view(-1).detach().cpu().numpy().tolist()
            ddad_scores = refined_features["score"].view(-1).detach().cpu().numpy().tolist()
            quality = _clip_patch_quality_metrics(outputs["patch_map"])
            labels = batch["label_img"].view(-1).detach().cpu().numpy().tolist()
            image_names = batch.get("image_name")
            if image_names is None:
                image_names = [str(img_id) for img_id in batch["img_id"]]
            group_ids = batch.get("group_id")
            if group_ids is None:
                group_ids = [str(img_id) for img_id in batch["img_id"]]
            for index, (img_id, group_id, image_name, label) in enumerate(zip(
                batch["img_id"],
                group_ids,
                image_names,
                labels,
            )):
                rows.append({
                    "img_id": str(img_id),
                    "group_id": str(group_id),
                    "image_name": str(image_name),
                    "label": int(label),
                    "clip_logit": float(clip_logits_list[index]),
                    "clip_score": float(clip_scores[index]),
                    "refined_ddad_score": float(refined_scores[index]),
                    "ddad_score": float(ddad_scores[index]),
                    "heatmap_peak": float(quality["peak"][index]),
                    "heatmap_topk_mean": float(quality["topk_mean"][index]),
                    "heatmap_foreground_ratio": float(quality["foreground_ratio"][index]),
                    "localization_confidence": float(quality["localization_confidence"][index]),
                    "background_consistency": float(quality["background_consistency"][index]),
                })
    return rows


def _score_student_fusion_rows(rows, fusion_head, device):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, {"auc": 0.0, "ap": 0.0}
    df = _attach_rank_and_percentile(df, "refined_ddad_score", "refined_ddad")
    df = _attach_rank_and_percentile(df, "clip_score", "clip_score")
    df = _attach_rank_and_percentile(df, "localization_confidence", "localization_confidence")
    df = _attach_rank_and_percentile(df, "background_consistency", "background_consistency")
    df["abnormal_joint_score"] = (
        0.70 * df["refined_ddad_percentile"].astype(float) +
        0.20 * df["clip_score_percentile"].astype(float) +
        0.10 * df["localization_confidence_percentile"].astype(float)
    )
    df["normal_joint_score"] = (
        0.70 * (1.0 - df["refined_ddad_percentile"].astype(float)) +
        0.20 * (1.0 - df["clip_score_percentile"].astype(float)) +
        0.10 * df["background_consistency_percentile"].astype(float)
    )
    feature_columns = [
        "clip_logit",
        "clip_score",
        "refined_ddad_score",
        "ddad_score",
        "abnormal_joint_score",
        "normal_joint_score",
        "heatmap_peak",
        "heatmap_topk_mean",
        "heatmap_foreground_ratio",
        "localization_confidence",
        "background_consistency",
    ]
    features = torch.tensor(df[feature_columns].astype(float).to_numpy(), dtype=torch.float32)
    fusion_scores = []
    fusion_head.eval()
    with torch.no_grad():
        for start in range(0, features.size(0), 256):
            batch_features = features[start:start + 256].to(device)
            logits = fusion_head(batch_features).view(-1)
            fusion_scores.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    df["fusion_score"] = fusion_scores
    metrics = _classification_metrics(
        df["label"].astype(int).tolist(),
        df["fusion_score"].astype(float).tolist(),
    )
    metrics["count"] = int(len(df))
    return df, metrics


def evaluate_clip_student_fusion_official(cfgs):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    loader_overrides = _get_clip_loader_overrides(cfgs)
    amp_enabled = _amp_enabled(clip_cfg, device)
    batch_size = int(clip_cfg.get("val_bs", clip_cfg.get("bs", 4)))
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    checkpoint_path = _clip_student_fusion_checkpoint_paths(cfgs)["best"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Could not find clip_student_fusion checkpoint at {}. Run mode clip_student_fusion first.".format(
                checkpoint_path
            )
        )
    official_test_loader = _build_multibranch_loader(
        cfgs,
        subset="official_test",
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    model = _build_clip_model(cfgs, device)
    weak_cfg = _get_weakclip_cfg(cfgs)
    fusion_head = DDADGuidedStudentFusionHead(
        in_dim=11,
        hidden_dim=int(weak_cfg.get("student_fusion_hidden_dim", 32)),
        dropout=float(weak_cfg.get("student_fusion_dropout", 0.1)),
    ).to(device)
    checkpoint_extra = _load_clip_student_fusion_checkpoint(checkpoint_path, model, fusion_head, map_location=device)
    model.eval()
    fusion_head.eval()
    refine_in = ["inter_dis", "intra_dis"]
    module_a = _load_weak_ddad_ensemble(cfgs, "weak_a")
    module_b = _load_weak_ddad_ensemble(cfgs, "weak_b")
    refine_net, refine_runtime_cfg, _ = _load_weak_refine_model(cfgs, device, refine_in)
    rows = _collect_student_fusion_eval_rows(
        official_test_loader,
        cfgs,
        model,
        module_a,
        module_b,
        refine_net,
        refine_in,
        refine_runtime_cfg,
        device,
        amp_enabled=amp_enabled,
    )
    scored_df, fusion_metrics = _score_student_fusion_rows(rows, fusion_head, device)
    labels = scored_df["label"].astype(int).tolist() if len(scored_df) > 0 else []
    results = {
        "checkpoint": checkpoint_path,
        "checkpoint_extra": {
            "epoch": checkpoint_extra.get("epoch"),
            "score": checkpoint_extra.get("score"),
            "student_fusion_init_stage": checkpoint_extra.get("student_fusion_init_stage"),
            "student_fusion_init_checkpoint": checkpoint_extra.get("student_fusion_init_checkpoint"),
            "student_fusion_freeze_clip": checkpoint_extra.get("student_fusion_freeze_clip"),
            "student_ddad_map_loss_weight": checkpoint_extra.get("student_ddad_map_loss_weight"),
            "student_clip_aux_loss_weight": checkpoint_extra.get("student_clip_aux_loss_weight"),
            "student_effective_clip_aux_loss_weight": checkpoint_extra.get("student_effective_clip_aux_loss_weight"),
            "student_fusion_loss_weight": checkpoint_extra.get("student_fusion_loss_weight"),
            "selection_metric": checkpoint_extra.get("selection_metric"),
        },
        "official_test_metrics": {
            "clip_student_raw": _classification_metrics(labels, scored_df["clip_score"].astype(float).tolist()) if len(scored_df) > 0 else {"auc": 0.0, "ap": 0.0},
            "refined_ddad": _classification_metrics(labels, scored_df["refined_ddad_score"].astype(float).tolist()) if len(scored_df) > 0 else {"auc": 0.0, "ap": 0.0},
            "ddad_guided_student_fusion": fusion_metrics,
        },
    }
    output_dir = cfgs["Exp"]["out_dir"]
    result_path = os.path.join(output_dir, "eval_clip_student_fusion_results.json")
    rows_path = os.path.join(output_dir, "eval_clip_student_fusion_scores.csv")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    scored_df.to_csv(rows_path, index=False)
    print(json.dumps(results, indent=2))


def evaluate_clip_official(cfgs):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    loader_overrides = _get_clip_loader_overrides(cfgs)
    amp_enabled = _amp_enabled(clip_cfg, device)
    batch_size = int(clip_cfg.get("val_bs", clip_cfg.get("bs", 4)))
    dataset_kwargs = _weakclip_dataset_kwargs(cfgs)
    stage_name, checkpoint_path = _resolve_official_eval_stage(cfgs)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Could not find a checkpoint for eval_clip_official. Expected {}".format(checkpoint_path)
        )

    official_test_loader = _build_multibranch_loader(
        cfgs,
        subset="official_test",
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        dataset_kwargs=dataset_kwargs,
    )
    results = {
        "official_test_metrics": {},
        "checkpoint": checkpoint_path,
        "stage": stage_name,
    }
    csv_rows = []
    for eval_stage_name, eval_checkpoint in [
        ("teacher", _clip_stage_checkpoint_paths(cfgs, "clip_teacher")["best"]),
        ("student", _clip_stage_checkpoint_paths(cfgs, "clip_student")["best"]),
    ]:
        if not os.path.exists(eval_checkpoint):
            continue
        model = _build_clip_model(cfgs, device)
        _load_checkpoint(eval_checkpoint, model, map_location=device)
        model.eval()
        official_metrics = _evaluate_clip_image_metrics(
            official_test_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        results["official_test_metrics"][eval_stage_name] = {
            "clip": official_metrics,
        }
        csv_rows.append(
            {
                "split": "official_test",
                "branch": "clip",
                "auc": official_metrics["auc"],
                "ap": official_metrics["ap"],
                "stage": eval_stage_name,
            }
        )
    result_path = os.path.join(cfgs["Exp"]["out_dir"], "eval_clip_official_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(csv_rows).to_csv(os.path.join(cfgs["Exp"]["out_dir"], "eval_clip_official_results.csv"), index=False)
    print(json.dumps(results, indent=2))


def evaluate_clip_only(cfgs):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)
    amp_enabled = _amp_enabled(clip_cfg, device)
    batch_size = int(clip_cfg.get("val_bs", clip_cfg.get("bs", 4)))
    loader_overrides = _get_clip_loader_overrides(cfgs)

    real_test_loader = _build_multibranch_loader(
        cfgs,
        subset=protocol_cfg["real_eval_subset"],
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["real_eval_subset"], "real_eval"),
    )
    synthetic_test_loader = _build_multibranch_loader(
        cfgs,
        subset=protocol_cfg["synthetic_eval_subset"],
        batch_size=batch_size,
        shuffle=False,
        synthetic_probability=float(clip_cfg.get("synthetic_probability", 0.5)),
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["synthetic_eval_subset"], "synthetic_eval"),
    )

    model = _load_clip_model(cfgs, device)
    model.eval()

    real_metrics = _evaluate_clip_image_metrics(
        real_test_loader,
        model,
        device,
        amp_enabled=amp_enabled,
    )
    synthetic_metrics = _evaluate_clip_image_metrics(
        synthetic_test_loader,
        model,
        device,
        amp_enabled=amp_enabled,
    )

    results = {
        "baseline_protocol": protocol_cfg["mode"],
        "real_image_level_metrics": {
            "clip": real_metrics,
        },
        "synthetic_image_level_metrics": {
            "clip": synthetic_metrics,
        },
    }

    result_path = os.path.join(cfgs["Exp"]["out_dir"], "eval_clip_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    rows = [
        {
            "split": "real_image_level_metrics",
            "branch": "clip",
            "auc": real_metrics["auc"],
            "ap": real_metrics["ap"],
        },
        {
            "split": "synthetic_image_level_metrics",
            "branch": "clip",
            "auc": synthetic_metrics["auc"],
            "ap": synthetic_metrics["ap"],
        },
    ]
    pd.DataFrame(rows).to_csv(os.path.join(cfgs["Exp"]["out_dir"], "eval_clip_results.csv"), index=False)
    print(json.dumps(results, indent=2))


def train_clip_module(cfgs):
    device = _get_device(cfgs)
    clip_cfg = _get_section(cfgs, "CLIP")
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)
    loader_overrides = _get_clip_loader_overrides(cfgs)
    amp_enabled = _amp_enabled(clip_cfg, device)
    batch_size = int(clip_cfg.get("bs", 8))
    synthetic_probability = float(clip_cfg.get("synthetic_probability", 0.5))
    val_batch_size = int(clip_cfg.get("val_bs", batch_size))
    print(
        "=> CLIP loader config: train_workers={} val_workers={} "
        "train_persistent={} val_persistent={} train_prefetch={} val_prefetch={} "
        "cache_images={} cache_clip_images={}".format(
            loader_overrides["train_workers"],
            loader_overrides["val_workers"],
            loader_overrides["train_persistent_workers"],
            loader_overrides["val_persistent_workers"],
            loader_overrides["train_prefetch_factor"],
            loader_overrides["val_prefetch_factor"],
            loader_overrides["cache_images"],
            loader_overrides["cache_clip_images"],
        )
    )

    train_loader = _build_multibranch_loader(
        cfgs,
        subset=protocol_cfg["synthetic_train_subset"],
        batch_size=batch_size,
        shuffle=True,
        synthetic_probability=synthetic_probability,
        drop_last=True,
        num_workers_override=loader_overrides["train_workers"],
        persistent_workers_override=loader_overrides["train_persistent_workers"],
        prefetch_factor_override=loader_overrides["train_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["synthetic_train_subset"], "synthetic_train"),
    )
    real_train_loader = _build_multibranch_loader(
        cfgs,
        subset=protocol_cfg["real_train_subset"],
        batch_size=batch_size,
        shuffle=True,
        synthetic_probability=0.0,
        drop_last=True,
        num_workers_override=loader_overrides["train_workers"],
        persistent_workers_override=loader_overrides["train_persistent_workers"],
        prefetch_factor_override=loader_overrides["train_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["real_train_subset"], "real_train"),
    )
    val_loader = _build_multibranch_loader(
        cfgs,
        subset=protocol_cfg["synthetic_val_subset"],
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=synthetic_probability,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["synthetic_val_subset"], "synthetic_val"),
    )
    real_val_loader = _build_multibranch_loader(
        cfgs,
        subset=protocol_cfg["real_val_subset"],
        batch_size=val_batch_size,
        shuffle=False,
        synthetic_probability=0.0,
        drop_last=False,
        num_workers_override=loader_overrides["val_workers"],
        persistent_workers_override=loader_overrides["val_persistent_workers"],
        prefetch_factor_override=loader_overrides["val_prefetch_factor"],
        cache_images_override=loader_overrides["cache_images"],
        cache_clip_images_override=loader_overrides["cache_clip_images"],
        **_baseline_loader_extra_kwargs(cfgs, protocol_cfg["real_val_subset"], "real_val"),
    )
    print("=> CLIP baseline protocol: {}".format(protocol_cfg["mode"]))

    model = _build_clip_model(cfgs, device)
    _configure_clip_trainable_parameters(model)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(clip_cfg.get("lr", 1.0e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(clip_cfg.get("weight_decay", 1.0e-4)),
    )
    scaler = _make_grad_scaler(amp_enabled)
    image_loss_fn = nn.BCEWithLogitsLoss()

    out_dir = _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "clip"))
    writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_clip"))
    best_score = -float("inf")
    best_path = os.path.join(out_dir, "clip_best.pth")
    last_path = os.path.join(out_dir, "clip_branch.pth")
    num_epoch = int(clip_cfg.get("num_epoch", 30))

    for epoch in range(1, num_epoch + 1):
        model.train()
        synthetic_losses = AverageMeter()
        real_losses = AverageMeter()
        start = time.time()

        for batch in train_loader:
            image_224 = _move_tensor(batch["image_224"], device)
            label_img = _move_tensor(batch["label_img"].float(), device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(amp_enabled):
                outputs = model(
                    image_224,
                    output_size=batch["image_64"].shape[-2:],
                    return_patch_maps=False,
                )
                loss = image_loss_fn(outputs["global_logit"].view(-1).float(), label_img.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            synthetic_losses.update(loss.item(), image_224.size(0))

        for batch in real_train_loader:
            image_224 = _move_tensor(batch["image_224"], device)
            label_img = _move_tensor(batch["label_img"].float(), device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(amp_enabled):
                outputs = model(
                    image_224,
                    output_size=batch["image_64"].shape[-2:],
                    return_patch_maps=False,
                )
                loss = image_loss_fn(outputs["global_logit"].view(-1).float(), label_img.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            real_losses.update(loss.item(), image_224.size(0))

        model.eval()
        synthetic_metrics = _evaluate_clip_image_metrics(
            val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        real_metrics = _evaluate_clip_image_metrics(
            real_val_loader,
            model,
            device,
            amp_enabled=amp_enabled,
        )
        score = real_metrics["auc"] + real_metrics["ap"]

        writer.add_scalar("train_synthetic_loss", synthetic_losses.avg, epoch)
        writer.add_scalar("train_real_loss", real_losses.avg, epoch)
        writer.add_scalar("val_synthetic_auc", synthetic_metrics["auc"], epoch)
        writer.add_scalar("val_synthetic_ap", synthetic_metrics["ap"], epoch)
        writer.add_scalar("val_real_auc", real_metrics["auc"], epoch)
        writer.add_scalar("val_real_ap", real_metrics["ap"], epoch)
        print(
            "clip Epoch[{}/{}]\tTime:{:.1f}s\tSyn Loss:{:.4f}\tReal Loss:{:.4f}\tReal AUC:{:.4f}\tReal AP:{:.4f}\tSyn AUC:{:.4f}\tSyn AP:{:.4f}\tAMP:{}".format(
                epoch,
                num_epoch,
                time.time() - start,
                synthetic_losses.avg,
                real_losses.avg,
                real_metrics["auc"],
                real_metrics["ap"],
                synthetic_metrics["auc"],
                synthetic_metrics["ap"],
                amp_enabled,
            )
        )

        checkpoint_extra = {
            "epoch": epoch,
            "score": score,
            "real_auc": real_metrics["auc"],
            "real_ap": real_metrics["ap"],
            "synthetic_auc": synthetic_metrics["auc"],
            "synthetic_ap": synthetic_metrics["ap"],
        }
        _save_checkpoint(last_path, model, optimizer, extra=checkpoint_extra)
        if score > best_score:
            best_score = score
            _save_checkpoint(best_path, model, optimizer, extra=checkpoint_extra)

    writer.close()


def cache_fusion_features(cfgs):
    device = _get_device(cfgs)
    fusion_cfg = _get_section(cfgs, "Fusion")
    band_cfg = _get_fusion_band_cfg(cfgs)
    components = _gather_components(cfgs, device)
    train_batch_size = int(fusion_cfg.get("cache_bs", fusion_cfg.get("bs", 16)))
    eval_batch_size = int(fusion_cfg.get("cache_eval_bs", fusion_cfg.get("val_bs", train_batch_size)))
    synthetic_probability = float(fusion_cfg.get("synthetic_probability", 0.5))
    deterministic_seed = int(fusion_cfg.get("cache_seed", 3407))

    print("=> Caching fusion branch primitives under {}".format(_fusion_cache_root(cfgs)))
    print("=> Fusion band mode: {}".format(band_cfg["band_mode"]))
    print("=> Cache diffusion enabled: {}".format(components.get("has_diffusion", False)))
    print("=> Cache fusion feature extractors: {} device(s)".format(len(components.get("extractors", [components]))))
    _cache_fusion_split(
        cfgs,
        components,
        split_name="train_cached",
        subset="train_normal",
        batch_size=train_batch_size,
        synthetic_probability=synthetic_probability,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed,
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="synthetic_val",
        subset="synthetic_val",
        batch_size=eval_batch_size,
        synthetic_probability=synthetic_probability,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed + 1000,
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="synthetic_test",
        subset="synthetic_test",
        batch_size=eval_batch_size,
        synthetic_probability=synthetic_probability,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed + 2000,
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="real_train",
        subset="real_train",
        batch_size=train_batch_size,
        synthetic_probability=0.0,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed + 2500,
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="real_val",
        subset="real_val",
        batch_size=eval_batch_size,
        synthetic_probability=0.0,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed + 2750,
    )
    _cache_fusion_split(
        cfgs,
        components,
        split_name="real_test",
        subset="real_test",
        batch_size=eval_batch_size,
        synthetic_probability=0.0,
        include_clip=True,
        force_deterministic=True,
        deterministic_seed=deterministic_seed + 3000,
    )


def _detach_fusion_forward_inputs(fusion_inputs):
    return {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in fusion_inputs.items()
    }


def _ensure_fusion_forward_inputs_finite(name, fusion_inputs):
    for key in ["fusion_vector", "raw_maps", "clip_global_logit", "clip_anchor"]:
        if key in fusion_inputs:
            _ensure_finite_tensor("{}_{}".format(name, key), fusion_inputs[key])


def _evaluate_fusion_image_metrics(
    loader,
    cfgs,
    components,
    fusion_model,
    amp_enabled=False,
    feature_cache=None,
    cache_prefix=None,
):
    image_loss_fn = nn.BCEWithLogitsLoss()
    band_cfg = _get_fusion_band_cfg(cfgs)
    losses = []
    labels, scores = [], []

    with torch.no_grad():
        for batch in loader:
            if "raw_fusion_maps" in batch or "fusion_vector" in batch:
                fusion_inputs = _build_cached_fusion_inputs(cfgs, batch, components["device"])
            else:
                features = _get_or_compute_features(
                    batch,
                    cfgs,
                    components,
                    amp_enabled=amp_enabled,
                    feature_cache=feature_cache,
                    cache_prefix=cache_prefix,
                )
                fusion_inputs = _build_fusion_inputs(cfgs, features)

            _ensure_fusion_forward_inputs_finite("fusion_eval", fusion_inputs)
            label_img = _move_tensor(batch["label_img"].float(), components["device"])
            with _autocast_context(amp_enabled):
                outputs = fusion_model(**fusion_inputs)
                image_logit = outputs["image_logit"].view(-1).float()
                loss = image_loss_fn(image_logit, label_img.view(-1))
                loss = loss + band_cfg["safd_repulsion_weight"] * outputs.get("safd_repulsion_loss", image_logit.new_tensor(0.0))
            losses.append(loss.item())
            labels.extend(batch["label_img"].cpu().numpy().tolist())
            scores.extend(torch.sigmoid(image_logit).cpu().numpy().tolist())

    metrics = _classification_metrics(labels, scores)
    return float(np.mean(losses)) if losses else 0.0, metrics


def train_fusion_module(cfgs):
    device = _get_device(cfgs)
    fusion_cfg = _get_section(cfgs, "Fusion")
    band_cfg = _get_fusion_band_cfg(cfgs)
    amp_enabled = _amp_enabled(fusion_cfg, device)
    batch_size = int(fusion_cfg.get("bs", 4))
    synthetic_probability = float(fusion_cfg.get("synthetic_probability", 0.5))
    grad_clip = fusion_cfg.get("grad_clip", 1.0)
    use_cached_features = bool(fusion_cfg.get("use_cached_features", True))
    val_batch_size = int(fusion_cfg.get("val_bs", batch_size))

    components = _gather_components(cfgs, device)
    print("=> Fusion feature extractors: {} device(s)".format(len(components.get("extractors", [components]))))
    print("=> Fusion band mode: {}".format(band_cfg["band_mode"]))
    print(
        "=> Fusion inputs: clip_in_vector={} clip_anchor={} clip_dropout={} ddad={} diffusion={}".format(
            band_cfg["include_clip_score_in_vector"],
            band_cfg["use_clip_anchor"],
            band_cfg["clip_branch_dropout_prob"],
            band_cfg["enable_ddad_in_fusion"],
            band_cfg["enable_diffusion_in_fusion"],
        )
    )

    if use_cached_features:
        cache_root = _fusion_cache_root(cfgs)
        required_splits = ["train_cached", "synthetic_val", "real_train", "real_val"]
        missing_splits = [
            split_name for split_name in required_splits
            if not os.path.isdir(os.path.join(cache_root, split_name))
            or len([file_name for file_name in os.listdir(os.path.join(cache_root, split_name)) if file_name.endswith(".pt")]) == 0
        ]
        if len(missing_splits) > 0:
            raise RuntimeError(
                "Fusion cache is missing split(s): {}. Please run `python main.py --config <your_config> --mode cache_fusion` first.".format(
                    ", ".join(missing_splits)
                )
            )
        train_loader = _build_cached_fusion_loader(
            cfgs,
            split_name="train_cached",
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        real_train_loader = _build_cached_fusion_loader(
            cfgs,
            split_name="real_train",
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = _build_cached_fusion_loader(
            cfgs,
            split_name="synthetic_val",
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
        )
        real_val_loader = _build_cached_fusion_loader(
            cfgs,
            split_name="real_val",
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
        )
        print("=> Fusion training uses cached branch primitives from {}".format(cache_root))
    else:
        train_loader = _build_multibranch_loader(
            cfgs,
            subset="train_normal",
            batch_size=batch_size,
            shuffle=True,
            synthetic_probability=synthetic_probability,
            drop_last=True,
        )
        real_train_loader = _build_multibranch_loader(
            cfgs,
            subset="real_train",
            batch_size=batch_size,
            shuffle=True,
            synthetic_probability=0.0,
            drop_last=True,
        )
        val_loader = _build_multibranch_loader(
            cfgs,
            subset="synthetic_val",
            batch_size=val_batch_size,
            shuffle=False,
            synthetic_probability=synthetic_probability,
            drop_last=False,
        )
        real_val_loader = _build_multibranch_loader(
            cfgs,
            subset="real_val",
            batch_size=val_batch_size,
            shuffle=False,
            synthetic_probability=0.0,
            drop_last=False,
        )
        print("=> Fusion training uses online feature recomputation (fallback mode)")

    fusion_model = _build_fusion_model(cfgs, device)
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=float(fusion_cfg.get("lr", 1.0e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(fusion_cfg.get("weight_decay", 1.0e-4)),
    )
    scaler = _make_grad_scaler(amp_enabled)
    image_loss_fn = nn.BCEWithLogitsLoss()

    out_dir = _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "fusion"))
    writer = SummaryWriter(os.path.join(cfgs["Exp"]["out_dir"], "log_fusion"))
    best_score = -float("inf")
    best_path = os.path.join(out_dir, "fusion_best.pth")
    last_path = os.path.join(out_dir, "fusion_refine.pth")
    num_epoch = int(fusion_cfg.get("num_epoch", 30))

    for epoch in range(1, num_epoch + 1):
        fusion_model.train()
        total_losses = AverageMeter()
        synthetic_losses = AverageMeter()
        real_losses = AverageMeter()
        start = time.time()

        for batch in train_loader:
            if "raw_fusion_maps" in batch or "fusion_vector" in batch:
                fusion_inputs = _build_cached_fusion_inputs(cfgs, batch, device)
            else:
                with torch.no_grad():
                    features = _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)
                    fusion_inputs = _detach_fusion_forward_inputs(_build_fusion_inputs(cfgs, features))
            fusion_inputs = _apply_fusion_clip_dropout(fusion_inputs, band_cfg)
            _ensure_fusion_forward_inputs_finite("fusion_train", fusion_inputs)
            label_img = _move_tensor(batch["label_img"].float(), device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(**fusion_inputs)
                image_logit = outputs["image_logit"].view(-1).float()
                loss = image_loss_fn(image_logit, label_img.view(-1))
                loss = loss + band_cfg["safd_repulsion_weight"] * outputs.get("safd_repulsion_loss", image_logit.new_tensor(0.0))
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            batch_items = label_img.size(0)
            total_losses.update(loss.item(), batch_items)
            synthetic_losses.update(loss.item(), batch_items)

        for batch in real_train_loader:
            if "raw_fusion_maps" in batch or "fusion_vector" in batch:
                fusion_inputs = _build_cached_fusion_inputs(cfgs, batch, device)
            else:
                with torch.no_grad():
                    features = _compute_all_branch_features_multi_device(batch, cfgs, components, amp_enabled=amp_enabled)
                    fusion_inputs = _detach_fusion_forward_inputs(_build_fusion_inputs(cfgs, features))
            fusion_inputs = _apply_fusion_clip_dropout(fusion_inputs, band_cfg)
            _ensure_fusion_forward_inputs_finite("fusion_real_train", fusion_inputs)
            label_img = _move_tensor(batch["label_img"].float(), device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(**fusion_inputs)
                image_logit = outputs["image_logit"].view(-1).float()
                loss = image_loss_fn(image_logit, label_img.view(-1))
                loss = loss + band_cfg["safd_repulsion_weight"] * outputs.get("safd_repulsion_loss", image_logit.new_tensor(0.0))
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            batch_items = label_img.size(0)
            total_losses.update(loss.item(), batch_items)
            real_losses.update(loss.item(), batch_items)

        fusion_model.eval()
        val_loss, synthetic_metrics = _evaluate_fusion_image_metrics(
            val_loader,
            cfgs,
            components,
            fusion_model,
            amp_enabled=amp_enabled,
            feature_cache=components["feature_cache"],
            cache_prefix="synthetic_val",
        )
        real_val_loss, real_metrics = _evaluate_fusion_image_metrics(
            real_val_loader,
            cfgs,
            components,
            fusion_model,
            amp_enabled=amp_enabled,
            feature_cache=components["feature_cache"],
            cache_prefix="real_val",
        )
        selection_score = real_metrics["auc"] + real_metrics["ap"]

        writer.add_scalar("train_loss", total_losses.avg, epoch)
        writer.add_scalar("train_synthetic_loss", synthetic_losses.avg, epoch)
        writer.add_scalar("train_real_loss", real_losses.avg, epoch)
        writer.add_scalar("val_synthetic_loss", val_loss, epoch)
        writer.add_scalar("val_real_loss", real_val_loss, epoch)
        writer.add_scalar("val_synthetic_auc", synthetic_metrics["auc"], epoch)
        writer.add_scalar("val_synthetic_ap", synthetic_metrics["ap"], epoch)
        writer.add_scalar("val_real_auc", real_metrics["auc"], epoch)
        writer.add_scalar("val_real_ap", real_metrics["ap"], epoch)
        writer.add_scalar("val_selection_score", selection_score, epoch)
        print(
            "fusion Epoch[{}/{}]\tTime:{:.1f}s\tLoss:{:.4f}\tSyn Loss:{:.4f}\tReal Loss:{:.4f}\tVal Syn Loss:{:.4f}\tVal Real Loss:{:.4f}\tReal AUC:{:.4f}\tReal AP:{:.4f}\tSyn AUC:{:.4f}\tSyn AP:{:.4f}\tSel:{:.4f}\tAMP:{}".format(
                epoch,
                num_epoch,
                time.time() - start,
                total_losses.avg,
                synthetic_losses.avg,
                real_losses.avg,
                val_loss,
                real_val_loss,
                real_metrics["auc"],
                real_metrics["ap"],
                synthetic_metrics["auc"],
                synthetic_metrics["ap"],
                selection_score,
                amp_enabled,
            )
        )

        checkpoint_extra = {
            "epoch": epoch,
            "selection_score": selection_score,
            "real_auc": real_metrics["auc"],
            "real_ap": real_metrics["ap"],
            "synthetic_auc": synthetic_metrics["auc"],
            "synthetic_ap": synthetic_metrics["ap"],
        }
        _save_checkpoint(last_path, fusion_model, optimizer, extra=checkpoint_extra)
        if selection_score > best_score:
            best_score = selection_score
            _save_checkpoint(best_path, fusion_model, optimizer, extra=checkpoint_extra)

    writer.close()


def _evaluate_image_level_split(cfgs, components, fusion_model, subset, cache_prefix, is_real_split=False):
    fusion_cfg = _get_section(cfgs, "Fusion")
    amp_enabled = _amp_enabled(fusion_cfg, components["device"])
    has_diffusion = bool(components.get("has_diffusion", False))
    if is_real_split:
        real_loader_kwargs = _baseline_loader_extra_kwargs(cfgs, subset, split_role="real_eval")
        loader = _build_real_eval_loader(cfgs, subset, **real_loader_kwargs)
    else:
        loader = _build_multibranch_loader(
            cfgs,
            subset=subset,
            batch_size=1,
            shuffle=False,
            synthetic_probability=float(fusion_cfg.get("synthetic_probability", 0.5)),
            drop_last=False,
            **_baseline_loader_extra_kwargs(cfgs, subset, split_role="synthetic_eval"),
        )

    labels, ddad_scores, clip_scores, fusion_scores = [], [], [], []
    diffusion_scores = [] if has_diffusion else None
    with torch.no_grad():
        for batch in tqdm(loader, desc="{}_eval".format(subset)):
            features = _get_or_compute_features(
                batch,
                cfgs,
                components,
                amp_enabled=amp_enabled,
                feature_cache=components["feature_cache"],
                cache_prefix=cache_prefix,
            )
            fusion_inputs = _build_fusion_inputs(cfgs, features)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(**fusion_inputs)
            image_score = torch.sigmoid(outputs["image_logit"].float()).view(-1).cpu().numpy().tolist()

            labels.extend(batch["label_img"].cpu().numpy().tolist())
            ddad_scores.extend(features["branch_scores"]["ddad"].view(-1).cpu().numpy().tolist())
            if has_diffusion and diffusion_scores is not None:
                diffusion_scores.extend(features["branch_scores"]["diffusion"].view(-1).cpu().numpy().tolist())
            clip_scores.extend(torch.sigmoid(features["clip_global_logit"].view(-1)).cpu().numpy().tolist())
            fusion_scores.extend(image_score)

    metrics = {
        "ddad": _classification_metrics(labels, ddad_scores),
        "clip": _classification_metrics(labels, clip_scores),
        "fusion": _classification_metrics(labels, fusion_scores),
    }
    if has_diffusion and diffusion_scores is not None:
        metrics["diffusion"] = _classification_metrics(labels, diffusion_scores)
    return metrics


def evaluate_all_modules(cfgs):
    device = _get_device(cfgs)
    components = _gather_components(cfgs, device)
    fusion_model = _build_fusion_model(cfgs, device)
    fusion_path = os.path.join(cfgs["Exp"]["out_dir"], "fusion", "fusion_best.pth")
    _load_checkpoint(fusion_path, fusion_model, map_location=device)
    fusion_model.eval()
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)

    real_subset = protocol_cfg["real_eval_subset"]
    synthetic_subset = protocol_cfg["synthetic_eval_subset"]
    real_metrics = _evaluate_image_level_split(
        cfgs,
        components,
        fusion_model,
        real_subset,
        "{}".format(real_subset),
        is_real_split=True,
    )
    synthetic_metrics = _evaluate_image_level_split(
        cfgs,
        components,
        fusion_model,
        synthetic_subset,
        "{}".format(synthetic_subset),
        is_real_split=False,
    )
    results = {
        "baseline_protocol": protocol_cfg["mode"],
        "real_image_level_metrics": real_metrics,
        "synthetic_image_level_metrics": synthetic_metrics,
    }

    result_path = os.path.join(cfgs["Exp"]["out_dir"], "eval_all_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for split_name, split_metrics in results.items():
        if not isinstance(split_metrics, dict):
            continue
        for branch_name, branch_metrics in split_metrics.items():
            rows.append({
                "split": split_name,
                "branch": branch_name,
                "auc": branch_metrics["auc"],
                "ap_or_dice": branch_metrics["ap"],
            })
    pd.DataFrame(rows).to_csv(os.path.join(cfgs["Exp"]["out_dir"], "eval_all_results.csv"), index=False)
    print(json.dumps(results, indent=2))


def _gt_boxes_from_mask(binary_mask):
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if binary_mask.ndim != 2 or not np.any(binary_mask):
        return []
    return [
        {
            "bbox_x0": int(component["bbox_x0"]),
            "bbox_y0": int(component["bbox_y0"]),
            "bbox_x1": int(component["bbox_x1"]),
            "bbox_y1": int(component["bbox_y1"]),
        }
        for component in _connected_components_from_mask(binary_mask)
    ]


def export_real_visualizations(cfgs, gt_boxes_by_img_id=None):
    device = _get_device(cfgs)
    components = _gather_components(cfgs, device)
    fusion_model = _build_fusion_model(cfgs, device)
    fusion_path = os.path.join(cfgs["Exp"]["out_dir"], "fusion", "fusion_best.pth")
    _load_checkpoint(fusion_path, fusion_model, map_location=device)
    fusion_model.eval()
    protocol_cfg = _get_baseline_protocol_cfg(cfgs)

    visual_cfg = _get_fusion_visual_cfg(cfgs)
    real_eval_subset = protocol_cfg["real_eval_subset"]
    real_val_subset = protocol_cfg["real_val_subset"]
    real_metrics = _evaluate_image_level_split(
        cfgs,
        components,
        fusion_model,
        real_eval_subset,
        "{}_export".format(real_eval_subset),
        is_real_split=True,
    )
    amp_enabled = _amp_enabled(_get_section(cfgs, "Fusion"), components["device"])

    real_val_labels, real_val_scores = [], []
    real_val_loader = _build_real_eval_loader(
        cfgs,
        real_val_subset,
        **_baseline_loader_extra_kwargs(cfgs, real_val_subset, split_role="real_val"),
    )
    with torch.no_grad():
        for batch in tqdm(real_val_loader, desc="{}_scores".format(real_val_subset)):
            features = _compute_all_branch_features_multi_device(
                batch,
                cfgs,
                components,
                amp_enabled=amp_enabled,
                return_heatmaps=False,
            )
            fusion_inputs = _build_fusion_inputs(cfgs, features)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(**fusion_inputs)
            real_val_labels.extend(batch["label_img"].cpu().numpy().tolist())
            real_val_scores.extend(torch.sigmoid(outputs["image_logit"].float()).view(-1).cpu().numpy().tolist())

    real_box_threshold = _choose_real_box_threshold(real_val_labels, real_val_scores, default_threshold=0.5)
    print(
        "=> Real localization gating threshold (Youden J on {}): {:.4f}".format(
            real_val_subset,
            real_box_threshold,
        )
    )

    export_loader = _build_real_eval_loader(
        cfgs,
        real_eval_subset,
        **_baseline_loader_extra_kwargs(cfgs, real_eval_subset, split_role="real_eval"),
    )
    positive_samples = []
    real_mask_maps, real_masks = [], []
    with torch.no_grad():
        for batch in tqdm(export_loader, desc="{}_vis".format(real_eval_subset)):
            label = int(batch["label_img"].cpu().numpy()[0])
            if label != 1:
                continue
            features = _compute_all_branch_features_multi_device(
                batch,
                cfgs,
                components,
                amp_enabled=amp_enabled,
                return_heatmaps=True,
            )
            fusion_inputs = _build_fusion_inputs(cfgs, features)
            with _autocast_context(amp_enabled):
                outputs = fusion_model(**fusion_inputs)

            image = batch["image_64"][0, 0].cpu().numpy()
            clip_patch_map = np.zeros_like(image, dtype=np.float32)
            if "heatmaps" in features and "clip_patch_map" in features["heatmaps"]:
                clip_patch_map = features["heatmaps"]["clip_patch_map"][0, 0].detach().cpu().numpy().astype(np.float32)
            clip_patch_map = _normalize_visual_map(clip_patch_map)
            fusion_score = float(torch.sigmoid(outputs["image_logit"].float()).view(-1).cpu().numpy()[0])

            has_real_mask = bool(batch["has_real_mask"][0].item()) if "has_real_mask" in batch else False
            real_mask = batch["mask_real"][0, 0].cpu().numpy().astype(np.float32) if has_real_mask else None
            if has_real_mask and real_mask is not None:
                real_mask_maps.append(clip_patch_map)
                real_masks.append(real_mask)

            positive_samples.append({
                "img_id": str(batch["img_id"][0]),
                "label": label,
                "image": image,
                "fusion_map": clip_patch_map,
                "fusion_score": fusion_score,
                "has_real_mask": has_real_mask,
                "real_mask": real_mask,
            })

    export_num_images = max(1, int(visual_cfg["export_num_images"]))
    rng = np.random.default_rng(int(visual_cfg["export_seed"]))
    if len(positive_samples) > export_num_images:
        selected_indices = sorted(rng.choice(len(positive_samples), size=export_num_images, replace=False).tolist())
        selected_samples = [positive_samples[index] for index in selected_indices]
    else:
        selected_samples = positive_samples

    pixel_auc, pixel_dice = _synthetic_pixel_metrics(real_mask_maps, real_masks)
    export_root = _ensure_dir(os.path.join(cfgs["Exp"]["out_dir"], "real_vis_{}".format(export_num_images)))
    composites_dir = _ensure_dir(os.path.join(export_root, "composites"))

    metadata_rows = []
    boxes_payload = {}
    for index, sample in enumerate(selected_samples, start=1):
        localization = _extract_localization_boxes(
            sample["fusion_map"],
            threshold=visual_cfg["vis_threshold"],
            fusion_score=sample["fusion_score"],
            real_box_threshold=real_box_threshold,
            min_region_area=visual_cfg["min_region_area"],
            max_region_area_ratio=visual_cfg["max_region_area_ratio"],
            peak_split_threshold_ratio=visual_cfg["peak_split_threshold_ratio"],
            max_boxes=visual_cfg["max_boxes"],
            nms_iou_threshold=visual_cfg["box_nms_iou"],
        )
        sample_gt_boxes = None
        if gt_boxes_by_img_id is not None:
            sample_gt_boxes = gt_boxes_by_img_id.get(sample["img_id"])
        elif sample["has_real_mask"] and sample["real_mask"] is not None:
            sample_gt_boxes = _gt_boxes_from_mask(sample["real_mask"] > 0.5)
        file_name = "{:02d}_{}.png".format(index, _sanitize_sample_id(sample["img_id"]))
        _render_real_vis_composite(
            os.path.join(composites_dir, file_name),
            sample,
            localization,
            real_box_threshold,
            visual_cfg["vis_threshold"],
            gt_boxes=sample_gt_boxes,
            gt_mask=sample["real_mask"],
        )
        metadata_rows.append({
            "img_id": str(sample["img_id"]),
            "label": int(sample["label"]),
            "fusion_score": float(sample["fusion_score"]),
            "passed_gate": bool(localization["passed_gate"]),
            "box_count": int(localization["box_count"]),
            "peak_score": float(localization["peak_score"]),
            "has_real_mask": bool(sample["has_real_mask"]),
            "no_box_reason": str(localization["no_box_reason"]),
        })
        boxes_payload[str(sample["img_id"])] = {
            "pred_boxes": localization["boxes"],
            "gt_boxes": sample_gt_boxes or [],
        }

    pd.DataFrame(
        metadata_rows,
        columns=["img_id", "label", "fusion_score", "passed_gate", "box_count", "peak_score", "has_real_mask", "no_box_reason"],
    ).to_csv(os.path.join(export_root, "metadata.csv"), index=False)
    with open(os.path.join(export_root, "boxes.json"), "w") as f:
        json.dump(boxes_payload, f, indent=2)
    with open(os.path.join(export_root, "thresholds.json"), "w") as f:
        json.dump({
            "real_box_threshold": float(real_box_threshold),
            "vis_threshold": float(visual_cfg["vis_threshold"]),
            "min_region_area": int(visual_cfg["min_region_area"]),
            "max_region_area_ratio": float(visual_cfg["max_region_area_ratio"]),
            "peak_split_threshold_ratio": float(visual_cfg["peak_split_threshold_ratio"]),
            "box_nms_iou": float(visual_cfg["box_nms_iou"]),
            "max_boxes": int(visual_cfg["max_boxes"]),
            "random_seed": int(visual_cfg["export_seed"]),
        }, f, indent=2)
    with open(os.path.join(export_root, "analysis_summary.json"), "w") as f:
        json.dump({
            "real_image_level_metrics": real_metrics,
            "real_mask_pixel_metrics": {
                "clip_patch_auc": float(pixel_auc),
                "clip_patch_dice": float(pixel_dice),
                "num_positive_samples_with_masks": int(len(real_masks)),
            },
            "checkpoint": fusion_path,
            "selected_images": [str(sample["img_id"]) for sample in selected_samples],
            "config_summary": {
                "vis_threshold": float(visual_cfg["vis_threshold"]),
                "min_region_area": int(visual_cfg["min_region_area"]),
                "max_region_area_ratio": float(visual_cfg["max_region_area_ratio"]),
                "peak_split_threshold_ratio": float(visual_cfg["peak_split_threshold_ratio"]),
                "box_nms_iou": float(visual_cfg["box_nms_iou"]),
                "max_boxes": int(visual_cfg["max_boxes"]),
                "real_box_threshold": float(real_box_threshold),
            },
            "notes": [
                "The current classification-first fusion model has no real-mask supervision during training.",
                "Localization visualization uses CLIP patch heatmaps gated by the fusion image-level score.",
                "Real-mask pixel metrics are diagnostic outputs only and do not change training.",
            ],
        }, f, indent=2)
    print("=> Exported {} real-test composites to {}".format(len(selected_samples), export_root))
