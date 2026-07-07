import argparse
import json
import math
import os
import random
import time
from typing import Dict, List

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from skimage.filters.thresholding import threshold_otsu
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.Evaluation import Evaluation
from utils.data import Data_Loader, to_normalization

    COMPASSModel,
    prototype_distillation_loss,
    weighted_prototype_distillation_loss,
    structural_teacher_distillation_loss,
    js_divergence_from_probs,
)

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def minmax01(arr: np.ndarray):
    arr = arr.astype(np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax - amin < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)

def to_uint8_01(arr01: np.ndarray):
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0).astype(np.uint8)

class Timer:
    def __init__(self):
        self.t0 = None

    def start(self):
        self.t0 = time.time()

    def stop(self):
        return 0.0 if self.t0 is None else (time.time() - self.t0)

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cosine_rampup(current_epoch: int, total_epochs: int, start_value: float, end_value: float):
    if total_epochs <= 1:
        return end_value
    ratio = (current_epoch - 1) / (total_epochs - 1)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    value = end_value + 0.5 * (start_value - end_value) * (1.0 + math.cos(math.pi * ratio))
    return value

@torch.no_grad()
def normalized_l2_per_sample(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x1 = x1 / (x1.norm(dim=-1, keepdim=True) + eps)
    x2 = x2 / (x2.norm(dim=-1, keepdim=True) + eps)
    return torch.norm(x1 - x2, p=2, dim=-1)

@torch.no_grad()
def pooled_feature_distance(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    p1 = feat1.mean(dim=[2, 3])
    p2 = feat2.mean(dim=[2, 3])
    return normalized_l2_per_sample(p1, p2)

@torch.no_grad()
def flattened_feature_distance(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    B = feat1.size(0)
    f1 = feat1.view(B, -1)
    f2 = feat2.view(B, -1)
    return normalized_l2_per_sample(f1, f2)

@torch.no_grad()
def prototype_reliability(p1: torch.Tensor, p2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Estimate the reliability of prototype discrepancy using assignment entropy.

    p1, p2: [B, K], prototype probability distributions of the two dates.
    return: [B], reliability in [0, 1].

    Low-entropy prototype assignments are treated as more reliable. This is
    label-free and is consistent with the confidence definition used during
    confidence-aware cross-modal distillation.
    """
    if p1.dim() != 2 or p2.dim() != 2:
        raise ValueError(f"prototype_reliability expects [B,K], got {tuple(p1.shape)} and {tuple(p2.shape)}")
    if p1.shape != p2.shape:
        raise ValueError(f"prototype_reliability shape mismatch: {tuple(p1.shape)} vs {tuple(p2.shape)}")

    k = p1.shape[-1]
    max_ent = math.log(max(k, 2))

    ent1 = -(p1.clamp_min(eps) * torch.log(p1.clamp_min(eps))).sum(dim=-1)
    ent2 = -(p2.clamp_min(eps) * torch.log(p2.clamp_min(eps))).sum(dim=-1)

    conf1 = (1.0 - ent1 / max_ent).clamp(0.0, 1.0)
    conf2 = (1.0 - ent2 / max_ent).clamp(0.0, 1.0)
    return 0.5 * (conf1 + conf2)

@torch.no_grad()
def structural_reliability(struct_layers: List[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """
    Estimate the reliability of structural discrepancy using cross-level consistency.

    struct_layers: list of [B] structural discrepancy tensors from different levels.
    return: [B], reliability in [0, 1].

    If different levels produce consistent structural evidence, the coefficient
    of variation is small and the reliability is high. If shallow/deep structural
    cues disagree, the reliability is suppressed.
    """
    if len(struct_layers) == 0:
        raise ValueError("structural_reliability received an empty struct_layers list.")

    stack = torch.stack(struct_layers, dim=0)  # [L, B]
    mean = stack.mean(dim=0)
    std = stack.std(dim=0, unbiased=False)
    cv = std / (mean.abs() + eps)
    return torch.exp(-cv).clamp(0.0, 1.0)

@torch.no_grad()
def adaptive_proto_struct_weights(
    proto_probs_t1: torch.Tensor,
    proto_probs_t2: torch.Tensor,
    struct_layers: List[torch.Tensor],
    eps: float = 1e-6,
):
    """
    Generate label-free adaptive weights for prototype-structural fusion.

    return:
        w_proto, w_struct, r_proto, r_struct, all with shape [B].
    """
    r_proto = prototype_reliability(proto_probs_t1, proto_probs_t2)
    r_struct = structural_reliability(struct_layers)
    denom = r_proto + r_struct + eps
    w_proto = r_proto / denom
    w_struct = r_struct / denom
    return w_proto, w_struct, r_proto, r_struct

def train_one_epoch(
    model,
    loader,
    opt,
    device,
    epoch: int,
    total_epochs: int,
    t_teacher: float,
    t_student: float,
    w_cm: float = 1.0,
    w_sm: float = 0.25,
    w_struct: float = 0.50,
    w_coco: float = 0.05,
    center_m: float = 0.9,
    teacher_m: float = 0.996,
    student_chunk_size: int = 128,
    conf_power: float = 1.0,
):
    """
    No-Aug training version.

    Original Scheme2 uses four augmented views:
        t1_v1, t1_v2, t2_v1, t2_v2.
    This version only uses raw t1 and raw t2 from the loader.

    Loss design:
        L_sm: teacher-student same-input self-distillation.
        L_cm: bidirectional cross-modal prototype alignment.
        L_struct: structural teacher distillation for graph-manifold commonality.
        L_coco: local/global compact-view consistency.

    This clean version follows the current setting w_sem = 0 and w_rec = 0:
    semantic-teacher distillation and reconstruction stabilization are removed.
    """
    model.train()

    w_cm_epoch = cosine_rampup(epoch, total_epochs, w_cm * 0.35, w_cm)
    w_struct_epoch = cosine_rampup(epoch, total_epochs, w_struct * 0.35, w_struct)

    running = {
        "loss": 0.0,
        "loss_cm": 0.0,
        "loss_sm": 0.0,
        "loss_struct": 0.0,
        "loss_coco": 0.0,
        "mean_conf": 0.0,
    }

    with tqdm(total=len(loader), desc=f"Train (COMPASS) [Epoch {epoch}]", ncols=170) as t:
        for batch in loader:
            # Original dataset output: (t1, t2, label).
            t1, t2 = batch[0], batch[1]
            t1 = t1.float().to(device, non_blocking=True)
            t2 = t2.float().to(device, non_blocking=True)

            B = t1.size(0)
            num_chunks = math.ceil(B / student_chunk_size)

            with torch.no_grad():
                teacher_out_t1 = model.teacher_forward_dual(t1, T=t_teacher, return_logits=True, return_features=True)
                teacher_out_t2 = model.teacher_forward_dual(t2, T=t_teacher, return_logits=True, return_features=True)

                q_t1 = teacher_out_t1["probs"]
                q_t2 = teacher_out_t2["probs"]
                logit_t1 = teacher_out_t1["logits"]
                logit_t2 = teacher_out_t2["logits"]

                conf_t1 = model.batch_confidence_from_probs(q_t1)
                conf_t2 = model.batch_confidence_from_probs(q_t2)
                cm_w_t1_from_t2 = conf_t2.pow(conf_power).detach()
                cm_w_t2_from_t1 = conf_t1.pow(conf_power).detach()
                mean_conf = 0.5 * (conf_t1.mean() + conf_t2.mean())

            opt.zero_grad(set_to_none=True)
            batch_vals = {"loss": 0.0, "cm": 0.0, "sm": 0.0, "struct": 0.0, "coco": 0.0}

            for start in range(0, B, student_chunk_size):
                end = min(start + student_chunk_size, B)

                q1 = q_t1[start:end]
                q2 = q_t2[start:end]
                w_cm_from_q2 = cm_w_t1_from_t2[start:end]
                w_cm_from_q1 = cm_w_t2_from_t1[start:end]

                t1_chunk = t1[start:end]
                t2_chunk = t2[start:end]

                s_t1 = model.student_forward_logits(
                    t1_chunk, branch="t1", T=t_student, return_feat=True
                )
                s_t2 = model.student_forward_logits(
                    t2_chunk, branch="t2", T=t_student, return_feat=True
                )

                p_t1 = s_t1["log_probs"]
                p_t2 = s_t2["log_probs"]

                # Same-modal self-distillation without augmented positive views.
                loss_sm = 0.5 * (prototype_distillation_loss(p_t1, q1) + prototype_distillation_loss(p_t2, q2))

                # Cross-modal distillation keeps only two direct directions.
                loss_cm = 0.5 * (
                    weighted_prototype_distillation_loss(p_t1, q2, w_cm_from_q2)
                    + weighted_prototype_distillation_loss(p_t2, q1, w_cm_from_q1)
                )

                # DualTeacher structural distillation:
                # the structural EMA teacher provides privileged graph-manifold targets
                # and a teacher-derived unchanged mask for reliable structural alignment.
                t_struct1 = [x[start:end] for x in teacher_out_t1["struct_teacher_feats"]]
                t_struct2 = [x[start:end] for x in teacher_out_t2["struct_teacher_feats"]]
                loss_struct = structural_teacher_distillation_loss(
                    s_t1["feats"], s_t2["feats"], t_struct1, t_struct2
                )

                loss_coco = 0.5 * (s_t1.get("coco_loss", p_t1.new_tensor(0.0)) + s_t2.get("coco_loss", p_t1.new_tensor(0.0)))

                loss = w_sm * loss_sm + w_cm_epoch * loss_cm + w_struct_epoch * loss_struct + w_coco * loss_coco
                (loss / num_chunks).backward()

                batch_vals["loss"] += float(loss.item())
                batch_vals["cm"] += float(loss_cm.item())
                batch_vals["sm"] += float(loss_sm.item())
                batch_vals["struct"] += float(loss_struct.item())
                batch_vals["coco"] += float(loss_coco.item())

            nn.utils.clip_grad_norm_(model.student_backbone.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(model.student_head.parameters(), max_norm=5.0)
            opt.step()

            with torch.no_grad():
                model.update_teacher(m=teacher_m)
                model.update_center_logits(torch.cat([logit_t1, logit_t2], dim=0), momentum=center_m)

            for k in batch_vals:
                batch_vals[k] /= max(1, num_chunks)

            running["loss"] += batch_vals["loss"]
            running["loss_cm"] += batch_vals["cm"]
            running["loss_sm"] += batch_vals["sm"]
            running["loss_struct"] += batch_vals["struct"]
            running["loss_coco"] += batch_vals["coco"]
            running["mean_conf"] += float(mean_conf.item())

            t.set_postfix({
                "loss": f"{batch_vals['loss']:.5f}",
                "cm": f"{batch_vals['cm']:.5f}",
                "sm": f"{batch_vals['sm']:.5f}",
                "struct": f"{batch_vals['struct']:.5f}",
                "conf": f"{float(mean_conf.item()):.3f}",
                "w_cm": f"{w_cm_epoch:.3f}",
                "w_struct": f"{w_struct_epoch:.3f}",
            })
            t.update(1)

    denom = max(1, len(loader))
    return {
        "loss": running["loss"] / denom,
        "loss_cm": running["loss_cm"] / denom,
        "loss_sm": running["loss_sm"] / denom,
        "loss_struct": running["loss_struct"] / denom,
        "loss_coco": running["loss_coco"] / denom,
        "mean_conf": running["mean_conf"] / denom,
        "w_cm_epoch": w_cm_epoch,
        "w_struct_epoch": w_struct_epoch,
    }

def to_3ch_image(img: np.ndarray, fallback_gray: np.ndarray) -> np.ndarray:
    if img is None:
        base = np.stack([fallback_gray, fallback_gray, fallback_gray], axis=2)
        return base
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    return np.stack([fallback_gray, fallback_gray, fallback_gray], axis=2)

def overlay_fp_fn(base_bgr: np.ndarray, fp_mask255: np.uint8, fn_mask255: np.uint8,
                  alpha_fp: float = 0.45, alpha_fn: float = 0.45):
    fp_layer = np.zeros_like(base_bgr)
    fn_layer = np.zeros_like(base_bgr)
    fp_layer[fp_mask255 > 0] = (0, 0, 255)
    fn_layer[fn_mask255 > 0] = (255, 0, 0)
    fp_vis = cv2.addWeighted(base_bgr, 1.0, fp_layer, alpha_fp, 0)
    fn_vis = cv2.addWeighted(base_bgr, 1.0, fn_layer, alpha_fn, 0)
    combined = cv2.addWeighted(fp_vis, 1.0, fn_layer, alpha_fn, 0)
    return combined, fp_vis, fn_vis

def overlay_tp_fp_fn(base_bgr: np.ndarray,
                     tp_mask255: np.uint8, fp_mask255: np.uint8, fn_mask255: np.uint8,
                     alpha_tp: float = 0.45, alpha_fp: float = 0.45, alpha_fn: float = 0.45):
    tp_layer = np.zeros_like(base_bgr)
    fp_layer = np.zeros_like(base_bgr)
    fn_layer = np.zeros_like(base_bgr)
    tp_layer[tp_mask255 > 0] = (0, 255, 0)
    fp_layer[fp_mask255 > 0] = (0, 0, 255)
    fn_layer[fn_mask255 > 0] = (255, 0, 0)
    tp_vis = cv2.addWeighted(base_bgr, 1.0, tp_layer, alpha_tp, 0)
    fp_vis = cv2.addWeighted(base_bgr, 1.0, fp_layer, alpha_fp, 0)
    fn_vis = cv2.addWeighted(base_bgr, 1.0, fn_layer, alpha_fn, 0)
    combined = cv2.addWeighted(base_bgr, 1.0, tp_layer, alpha_tp, 0)
    combined = cv2.addWeighted(combined, 1.0, fp_layer, alpha_fp, 0)
    combined = cv2.addWeighted(combined, 1.0, fn_layer, alpha_fn, 0)
    return combined, tp_vis, fp_vis, fn_vis

@torch.no_grad()
def validate_change_map(
    model,
    loader,
    o_h,
    o_w,
    gt,
    vis_dir,
    device,
    x1_full=None,
    x2_full=None,
    t_student_infer: float = 0.1,
    threshold_mode: str = "otsu",
    manual_thr: float = 66.3,
    percentile_thr: float = 85.0,
):
    model.eval()

    feat_scores, proto_scores, struct_scores = [], [], []
    proto_weight_scores, struct_weight_scores = [], []
    proto_reliability_scores, struct_reliability_scores = [], []

    with tqdm(total=len(loader), desc='Validate (COMPASS)', ncols=170, colour='cyan') as t:
        for (t1, t2, _) in loader:
            t1 = t1.to(device).float()
            t2 = t2.to(device).float()
            out1 = model.inference_forward(t1, T_student=t_student_infer)
            out2 = model.inference_forward(t2, T_student=t_student_infer)

            struct_layers = []
            for f1, f2 in zip(out1["feats"], out2["feats"]):

                local1 = F.avg_pool2d(f1, 3, 1, 1)
                local2 = F.avg_pool2d(f2, 3, 1, 1)
                lap1 = f1 - local1
                lap2 = f2 - local2
                d_struct = 0.5 * pooled_feature_distance(local1, local2) + 0.5 * pooled_feature_distance(lap1, lap2)
                struct_layers.append(d_struct)

            struct_score = sum(struct_layers) / len(struct_layers)
            proto_score = js_divergence_from_probs(out1["proto_probs"], out2["proto_probs"])

            # Adaptive Prototype-Structural Reliability Fusion (APSRF).
            # feat_score is still saved for ablation/visualization, but it is
            # deliberately excluded from the final change score to avoid raw
            # feature-distance responses caused by modality gaps.
            w_proto, w_struct, r_proto, r_struct = adaptive_proto_struct_weights(
                out1["proto_probs"], out2["proto_probs"], struct_layers
            )

            proto_scores.extend(proto_score.cpu().numpy().tolist())
            struct_scores.extend(struct_score.cpu().numpy().tolist())
            proto_weight_scores.extend(w_proto.cpu().numpy().tolist())
            struct_weight_scores.extend(w_struct.cpu().numpy().tolist())
            proto_reliability_scores.extend(r_proto.cpu().numpy().tolist())
            struct_reliability_scores.extend(r_struct.cpu().numpy().tolist())
            t.update(1)

    proto_raw = np.array(proto_scores, dtype=np.float32).reshape(o_h, o_w)
    struct_raw = np.array(struct_scores, dtype=np.float32).reshape(o_h, o_w)

    w_proto_map = np.array(proto_weight_scores, dtype=np.float32).reshape(o_h, o_w)
    w_struct_map = np.array(struct_weight_scores, dtype=np.float32).reshape(o_h, o_w)
    r_proto_map = np.array(proto_reliability_scores, dtype=np.float32).reshape(o_h, o_w)
    r_struct_map = np.array(struct_reliability_scores, dtype=np.float32).reshape(o_h, o_w)

    # Normalize branch discrepancy maps before fusion. This avoids a misleading
    # dominance caused purely by different numeric scales of JS divergence and
    # structural L2 distance. The adaptive weights themselves remain raw [0,1]
    # reliability-normalized values.

    proto_map = minmax01(proto_raw)
    struct_map = minmax01(struct_raw)
    final_map = w_proto_map * proto_map + w_struct_map * struct_map
    cmi = minmax01(final_map)

    proto_u8 = to_uint8_01(proto_map)
    struct_u8 = to_uint8_01(struct_map)
    cmi_u8 = to_uint8_01(cmi)
    w_proto_u8 = to_uint8_01(w_proto_map)
    w_struct_u8 = to_uint8_01(w_struct_map)
    r_proto_u8 = to_uint8_01(r_proto_map)
    r_struct_u8 = to_uint8_01(r_struct_map)

    cv2.imwrite(os.path.join(vis_dir, 'CMI_proto_js.png'), proto_u8)
    cv2.imwrite(os.path.join(vis_dir, 'CMI_structure_multi.png'), struct_u8)
    cv2.imwrite(os.path.join(vis_dir, 'Weight_proto_APSRF.png'), w_proto_u8)
    cv2.imwrite(os.path.join(vis_dir, 'Weight_structure_APSRF.png'), w_struct_u8)
    cv2.imwrite(os.path.join(vis_dir, 'Reliability_proto_APSRF.png'), r_proto_u8)
    cv2.imwrite(os.path.join(vis_dir, 'Reliability_structure_APSRF.png'), r_struct_u8)
    cv2.imwrite(os.path.join(vis_dir, 'CMI_CMSDL_COMPASS_APSRF_.png'), cmi_u8)

    if threshold_mode.lower() == "otsu":
        thr = float(threshold_otsu(cmi_u8))
    elif threshold_mode.lower() == "percentile":
        thr = float(np.percentile(cmi_u8, percentile_thr))
    else:
        thr = float(manual_thr)

    bcm = (cmi_u8 > thr).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(vis_dir, 'BCM_CMSDL_COMPASS_APSRF_.png'), bcm)

    evaler = Evaluation(gt.astype(np.uint8), bcm)
    OA, KC, AA = evaler.Classification_indicators()
    P, R, F1 = evaler.ObjectExtract_indicators()

    FPR, TPR, _ = metrics.roc_curve((gt > 127).astype(int).flatten(), cmi.flatten())
    AUC = metrics.auc(FPR, TPR)

    pred = (bcm > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    FP = ((pred == 1) & (gt_bin == 0)).astype(np.uint8) * 255
    FN = ((pred == 0) & (gt_bin == 1)).astype(np.uint8) * 255
    TP = ((pred == 1) & (gt_bin == 1)).astype(np.uint8) * 255

    base_raw = x2_full if x2_full is not None else x1_full
    base_bgr = to_3ch_image(base_raw, cmi_u8)
    H, W = base_bgr.shape[:2]
    if (FP.shape[0] != H) or (FP.shape[1] != W):
        FP = cv2.resize(FP, (W, H), interpolation=cv2.INTER_NEAREST)
        FN = cv2.resize(FN, (W, H), interpolation=cv2.INTER_NEAREST)
        TP = cv2.resize(TP, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay_all_2, overlay_fp, overlay_fn = overlay_fp_fn(base_bgr, FP, FN)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_FN.png"), overlay_all_2)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_only.png"), overlay_fp)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FN_only.png"), overlay_fn)

    overlay_all_3, overlay_tp, overlay_fp2, overlay_fn2 = overlay_tp_fp_fn(base_bgr, TP, FP, FN)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_TP_FP_FN.png"), overlay_all_3)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_TP_only.png"), overlay_tp)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_only_v2.png"), overlay_fp2)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FN_only_v2.png"), overlay_fn2)

    print(f'[VAL][COMPASS-APSRF-] AUC={AUC:.4f} OA={OA:.2f} KC={KC:.2f} F1={F1:.2f} thr={thr:.2f} mode={threshold_mode}')

    outputs = {
        "proto_u8": proto_u8,
        "struct_u8": struct_u8,
        "w_proto_u8": w_proto_u8,
        "w_struct_u8": w_struct_u8,
        "r_proto_u8": r_proto_u8,
        "r_struct_u8": r_struct_u8,
        "cmi_u8": cmi_u8,
        "bcm_u8": bcm,
    }
    stats = {
        "AUC": float(AUC),
        "OA": float(OA),
        "KC": float(KC),
        "F1": float(F1),
        "threshold": float(thr),
    }
    return outputs, stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='yellow', type=str)
    parser.add_argument('--t1_path', default='./data/Yellow/yellow_A_1.bmp', type=str)
    parser.add_argument('--t2_path', default='./data/Yellow/yellow_A_2.bmp', type=str)
    parser.add_argument('--gt_path', default='./data/Yellow/yellow_A_gt.bmp', type=str)
    # parser.add_argument('--data_name', default='bastrop', type=str)
    # parser.add_argument('--t1_path', default='./data/Texas/data.mat', type=str)
    # parser.add_argument('--t2_path', default='./data/Texas/im2.bmp', type=str)
    # parser.add_argument('--gt_path', default='./data/Texas/reference.bmp', type=str)

    parser.add_argument('--patch_size', default=11, type=int)
    parser.add_argument('--test_ps', default=11, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--vision_path', default='./CGSD/yellow_A/', type=str)

    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--head_hidden', default=512, type=int)
    parser.add_argument('--head_bottleneck', default=256, type=int)
    parser.add_argument('--t_teacher', default=0.04, type=float)
    parser.add_argument('--t_student', default=0.1, type=float)
    parser.add_argument('--teacher_m', default=0.996, type=float)
    parser.add_argument('--center_m', default=0.9, type=float)

    parser.add_argument('--w_cm', default=1.20, type=float)
    parser.add_argument('--w_sm', default=0.20, type=float)
    parser.add_argument('--w_struct', default=0.5, type=float)
    parser.add_argument('--w_coco', default=0.05, type=float, help='Weight of CoCo local/global compact-view consistency loss.')
    parser.add_argument('--student_chunk_size', default=128, type=int)
    parser.add_argument('--conf_power', default=1.0, type=float)

    parser.add_argument('--threshold_mode', default='otsu', type=str, choices=['otsu', 'manual', 'percentile'])
    parser.add_argument('--manual_thr', default=66.3, type=float)
    parser.add_argument('--percentile_thr', default=84.0, type=float)
    parser.add_argument('--seed', default=2025, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.vision_path)

    train_dataset = Data_Loader(args.data_name, args.t1_path, args.t2_path, args.gt_path,
                                patch_size=args.patch_size, mode='train', transform=T.ToTensor())
    test_dataset = Data_Loader(args.data_name, args.t1_path, args.t2_path, args.gt_path,
                               patch_size=args.test_ps, mode='test', transform=T.ToTensor())

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              worker_init_fn=_seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 6, shuffle=False, num_workers=0,
                             worker_init_fn=_seed_worker, generator=g)

    if args.data_name == 'bastrop':
        mat = io.loadmat(args.t1_path)
        x1 = mat['t1_L5'][:, :, 3]
        x2 = mat["t2_ALI"][:, :, 5]
        x1 = to_normalization(x1)[..., np.newaxis]
        x2 = to_normalization(x2)[..., np.newaxis]
        gt = (mat["ROI_1"] * 255).astype(np.uint8)
        x1_full = (x1 * 255).astype(np.uint8)
        x2_full = (x2 * 255).astype(np.uint8)
        x1_full = np.repeat(x1_full, 3, axis=2)
        x2_full = np.repeat(x2_full, 3, axis=2)
    else:
        x1_full = cv2.imread(args.t1_path)
        x2_full = cv2.imread(args.t2_path)
        gt_img = cv2.imread(args.gt_path)
        if gt_img is None:
            raise FileNotFoundError(f'Cannot read gt_path: {args.gt_path}')
        gt = gt_img[:, :, 0].astype(np.uint8)

    o_h, o_w = gt.shape[:2]

    model = COMPASSModel(
        in_channels=3,
        feat_dim=64,
        head_hidden=args.head_hidden,
        head_bottleneck=args.head_bottleneck,
        out_dim=args.out_dim,
        teacher_momentum=args.teacher_m,
    ).to(device)

    opt = optim.RMSprop([
        {"params": model.student_backbone.parameters(), "lr": args.lr},
        {"params": model.student_head.parameters(), "lr": args.lr},
    ], lr=args.lr, weight_decay=1e-5, momentum=0.9)

    history: List[Dict[str, float]] = []
    best_auc = -1.0
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model, loader=train_loader, opt=opt, device=device,
            epoch=epoch, total_epochs=args.epochs,
            t_teacher=args.t_teacher, t_student=args.t_student,
            w_cm=args.w_cm, w_sm=args.w_sm, w_struct=args.w_struct, w_coco=args.w_coco,
            center_m=args.center_m, teacher_m=args.teacher_m,
            student_chunk_size=args.student_chunk_size, conf_power=args.conf_power,
        )

        print(
            f"[EPOCH {epoch}] loss={train_stats['loss']:.5f} cm={train_stats['loss_cm']:.5f} "
            f"sm={train_stats['loss_sm']:.5f} struct={train_stats['loss_struct']:.5f} coco={train_stats['loss_coco']:.5f} "
            f"conf={train_stats['mean_conf']:.4f} w_cm={train_stats['w_cm_epoch']:.4f} w_struct={train_stats['w_struct_epoch']:.4f}"
        )

        ep_dir = os.path.join(args.vision_path, str(epoch))
        ensure_dir(ep_dir)
        outputs, metrics_dict = validate_change_map(
            model=model, loader=test_loader, o_h=o_h, o_w=o_w, gt=gt,
            vis_dir=ep_dir, device=device, x1_full=x1_full, x2_full=x2_full,
            t_student_infer=args.t_student, threshold_mode=args.threshold_mode,
            manual_thr=args.manual_thr, percentile_thr=args.percentile_thr,
        )

        io.savemat(os.path.join(ep_dir, 'CMSDL_COMPASS_APSRF__outputs.mat'), {
            'proto_u8': outputs['proto_u8'],
            'struct_u8': outputs['struct_u8'],
            'w_proto_u8': outputs['w_proto_u8'],
            'w_struct_u8': outputs['w_struct_u8'],
            'r_proto_u8': outputs['r_proto_u8'],
            'r_struct_u8': outputs['r_struct_u8'],
            'cmi_u8': outputs['cmi_u8'],
            'bcm_u8': outputs['bcm_u8'],
        })

        record = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_loss_cm": float(train_stats["loss_cm"]),
            "train_loss_sm": float(train_stats["loss_sm"]),
            "train_loss_struct": float(train_stats["loss_struct"]),
            "train_loss_coco": float(train_stats["loss_coco"]),
            "mean_conf": float(train_stats["mean_conf"]),
            "w_cm_epoch": float(train_stats["w_cm_epoch"]),
            "w_struct_epoch": float(train_stats["w_struct_epoch"]),
            **{k: float(v) for k, v in metrics_dict.items()},
        }
        history.append(record)

        with open(os.path.join(ep_dir, 'metrics_CMSDL_COMPASS_APSRF_.json'), 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        torch.save(model.state_dict(), os.path.join(ep_dir, f'cmsdl_localfeats_only_noxmas_epoch{epoch}.pth'))

        if metrics_dict["AUC"] > best_auc:
            best_auc = metrics_dict["AUC"]
            torch.save(model.state_dict(), os.path.join(args.vision_path, 'best_auc_cmsdl_localfeats_only_noxmas.pth'))
        if metrics_dict["F1"] > best_f1:
            best_f1 = metrics_dict["F1"]
            torch.save(model.state_dict(), os.path.join(args.vision_path, 'best_f1_cmsdl_localfeats_only_noxmas.pth'))

    with open(os.path.join(args.vision_path, 'history_all.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print('Training finished.')

if __name__ == '__main__':
    main()