import yaml
from trainer import *
from test import evaluate, test_rec, evaluate_r
from argparse import ArgumentParser
from multibranch_runtime import (
    cache_fusion_features,
    build_safd_normal_bank,
    evaluate_clip_only,
    evaluate_clip_official,
    evaluate_all_modules,
    evaluate_weak_refine_module,
    export_real_visualizations,
    score_unlabeled_with_teacher,
    select_pseudo_labels,
    train_clip_module,
    train_clip_student,
    train_clip_student_fusion,
    train_clip_teacher,
    train_diffusion_module,
    train_fusion_module,
    train_weak_ddad_module,
    train_weak_refine_module,
    evaluate_clip_student_fusion_official,
)

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def _primary_gpu_id(cfgs):
    exp_cfg = cfgs.get("Exp", {})
    preferred = int(exp_cfg.get("gpu", 0))
    gpu_ids = exp_cfg.get("gpu_ids")
    if gpu_ids is None:
        return preferred
    if isinstance(gpu_ids, str):
        parsed = [int(item.strip()) for item in gpu_ids.split(",") if item.strip() != ""]
    else:
        parsed = [int(item) for item in gpu_ids]
    if preferred in parsed:
        return preferred
    return parsed[0] if len(parsed) > 0 else preferred


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config', type=str, default="config/RSNA_AE.yaml")  # config file
    parser.add_argument('--mode', dest='mode', type=str, default=None,
                        help="e.g., a, b, r, eval, eval_r, test, diff_a, diff_b, clip, clip_teacher, weak_a, weak_b, weak_r, eval_weak_r, safd_bank, score_unlabeled, select_pseudo, clip_student, clip_student_fusion, eval_clip_official, eval_clip_student_fusion, cache_fusion, fusion, eval_all, export_real_vis")
    parser.add_argument('--refine', dest='refine_in', type=str, default="dual", help="dual, intra")

    """
    Description of modes.
    a: train one network in the Unknown Distribution Module (UDM) on normal+unlabeled data
    b: train one network in the Normative Distribution Module (NDM) on normal data
    r: train the ASR-Net (after finishing the training of UDM and NDM).
    eval: evaluate the reconstruction ensemble and DDAD (without ASR-Net).
    eval_r: evaluate the DDAD with ASR-Net
    test: test the reconstruction baselines
    """

    opt = parser.parse_args()

    with open(opt.config, "r") as f:
        cfgs = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.cuda.set_device(_primary_gpu_id(cfgs))

    out_dir = cfgs["Exp"]["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if opt.refine_in == "dual":
        refine_in = ["inter_dis", "intra_dis"]  # For R_{dual}
    elif opt.refine_in == "intra":
        refine_in = ["intra_dis"]  # For R_{intra}
    else:
        raise Exception("Invalid refine in: {}".format(opt.refine_in))

    if opt.mode == "a" or opt.mode == "b":
        module = "UDM" if opt.mode == "a" else "NDM"
        print("=> Training the {}".format(module))
        train_module_ab(cfgs, opt)
    elif opt.mode == "r":
        print("=> Training the ASR-Net ...")
        if len(refine_in) == 2:
            print("=> Refine dual discrepancies")
        else:
            print("=> Refine intra-discrepancy")
        train_refine(cfgs, refine_in)
    elif opt.mode == "eval":
        print("=> Evaluating DDAD without ASR ...")
        evaluate(cfgs)
    elif opt.mode == "eval_r":
        print("=> Evaluating DDAD with ASR ...")
        evaluate_r(cfgs, refine_in)
    elif opt.mode == "test":
        print("=> Testing the reconstruction models ...")
        test_rec(cfgs)
    elif opt.mode == "diff_a" or opt.mode == "diff_b":
        print("=> Training the diffusion branch ({}) ...".format(opt.mode))
        train_diffusion_module(cfgs, opt.mode)
    elif opt.mode == "clip":
        print("=> Training the CLIP branch ...")
        train_clip_module(cfgs)
    elif opt.mode == "clip_teacher":
        print("=> Training the weakly-supervised CLIP teacher ...")
        train_clip_teacher(cfgs)
    elif opt.mode == "weak_a":
        print("=> Training the weak DDAD branch A on clean normal + unlabeled pool ...")
        train_weak_ddad_module(cfgs, "weak_a")
    elif opt.mode == "weak_b":
        print("=> Training the weak DDAD branch B on clean normal ...")
        train_weak_ddad_module(cfgs, "weak_b")
    elif opt.mode == "weak_r":
        print("=> Training the weak DDAD refine branch ...")
        train_weak_refine_module(cfgs, refine_in)
    elif opt.mode == "eval_weak_r":
        print("=> Evaluating the weak DDAD refine branch ...")
        evaluate_weak_refine_module(cfgs, refine_in)
    elif opt.mode == "safd_bank":
        print("=> Building the SAFD clean-normal bank ...")
        build_safd_normal_bank(cfgs, refine_in)
    elif opt.mode == "score_unlabeled":
        print("=> Scoring unlabeled pools with the CLIP teacher ...")
        score_unlabeled_with_teacher(cfgs)
    elif opt.mode == "select_pseudo":
        print("=> Selecting pseudo labels from the unlabeled train pool ...")
        select_pseudo_labels(cfgs)
    elif opt.mode == "clip_student":
        print("=> Fine-tuning the CLIP student with pseudo labels ...")
        train_clip_student(cfgs)
    elif opt.mode == "clip_student_fusion":
        print("=> Fine-tuning the DDAD-guided CLIP student fusion model ...")
        train_clip_student_fusion(cfgs)
    elif opt.mode == "eval_clip":
        print("=> Evaluating the CLIP branch ...")
        evaluate_clip_only(cfgs)
    elif opt.mode == "eval_clip_official":
        print("=> Evaluating the weakly-supervised CLIP model on official_test ...")
        evaluate_clip_official(cfgs)
    elif opt.mode == "eval_clip_student_fusion":
        print("=> Evaluating the DDAD-guided CLIP student fusion model on official_test ...")
        evaluate_clip_student_fusion_official(cfgs)
    elif opt.mode == "cache_fusion":
        print("=> Caching fusion inputs from DDAD + Diffusion + CLIP ...")
        cache_fusion_features(cfgs)
    elif opt.mode == "fusion":
        print("=> Training the multi-branch fusion network ...")
        train_fusion_module(cfgs)
    elif opt.mode == "eval_all":
        print("=> Evaluating DDAD + Diffusion + CLIP + Fusion ...")
        evaluate_all_modules(cfgs)
    elif opt.mode == "export_real_vis":
        print("=> Exporting real-test localization visualizations ...")
        export_real_visualizations(cfgs)
    else:
        raise Exception("Invalid mode: {}".format(opt.mode))
