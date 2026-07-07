"""
COMPASS compact prototype-structural graph distillation backbone.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Basic blocks
# -----------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.dw = ConvBNReLU(in_ch, in_ch, k=k, p=p, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
class GraphStructureBlock(nn.Module):
    """
    Lightweight local graph-manifold approximation.
    It avoids explicit graph construction and instead uses:
    1) local spatial aggregation,
    2) feature-similarity gating,
    3) Laplacian-style residual.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        groups = max(1, channels // 8)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.pre(x)
        local_mean = F.avg_pool2d(x0, kernel_size=3, stride=1, padding=1)
        lap = x0 - local_mean

        x_norm = F.normalize(x0, dim=1)
        local_norm = F.normalize(local_mean, dim=1)
        sim = (x_norm * local_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        sim = 0.5 * (sim + 1.0)

        out = self.fuse(torch.cat([x0, local_mean, lap.abs(), sim], dim=1))
        return out + x

class CompactPrototypeLearning(nn.Module):
    """
    Prototype-structural compactness learning adapted to dense feature maps.

    The official implementation treats the embedding matrix Z as samples x
    channels, iteratively estimates a low-rank subspace with an EM-like update,
    reconstructs Z in that compact subspace, and injects the reconstruction back
    through a residual path. Here, spatial locations from local/global feature
    maps are treated as graph nodes.
    """
    def __init__(self, rank: int = 32, stage_num: int = 3, beta: float = 1.0, lamd: float = 1.0):
        super().__init__()
        self.rank = int(rank)
        self.stage_num = int(stage_num)
        self.beta = float(beta)
        self.lamd = float(lamd)
        self.register_buffer("mu_seed", torch.empty(1, self.rank))
        nn.init.normal_(self.mu_seed, mean=0.0, std=math.sqrt(2.0 / max(1, self.rank)))
        self.mu_seed.data = self.mu_seed.data / (self.mu_seed.data.norm(dim=0, keepdim=True) + 1e-6)

    def forward(self, embds: torch.Tensor) -> torch.Tensor:
        # embds: [N, C]. This follows the official Structural convention where the
        # low-rank basis mu has shape [N, K] and responsibilities have [C, K].
        if embds.dim() != 2:
            raise ValueError(f"CompactPrototypeLearning expects [N, C], got {tuple(embds.shape)}")
        n_nodes, n_dim = embds.shape
        if n_nodes <= 1 or n_dim <= 1:
            return embds

        embd_nor = embds / (embds.norm(dim=0, keepdim=True) + 1e-6)
        work = embd_nor
        mu = self.mu_seed.to(embds.device, embds.dtype).repeat(n_nodes, 1)

        with torch.no_grad():
            for _ in range(max(1, self.stage_num)):
                # E-step: channel-to-subspace responsibility, [C, K].
                z = torch.mm(work.transpose(0, 1), mu) / max(self.lamd, 1e-6)
                z = F.softmax(z, dim=1)
                z = z / (z.sum(dim=0, keepdim=True) + 1e-6)
                # M-step: update low-rank subspace, [N, K].
                mu = torch.mm(work, z)
                mu = mu / (mu.norm(dim=0, keepdim=True) + 1e-6)

        # Low-rank feature reconstruction. Gradient flows through embd_nor, as
        # in official Structural, while EM variables are treated as tuned targets.
        recon = torch.mm(mu, z.transpose(0, 1))
        return self.beta * recon + embd_nor

class LocalGlobalConsistency(nn.Module):
    """Distribution-level local/global consistency with an anchor queue.

    This mirrors official Structural's SampleSimilarities + symmetric KL design, but
    keeps the queue small and layer-local for patch-wise MCD training.
    """
    def __init__(self, channels: int, queue_size: int = 128, temperature: float = 0.05):
        super().__init__()
        self.channels = int(channels)
        self.queue_size = int(queue_size)
        self.temperature = float(temperature)
        self.register_buffer("local_memory", F.normalize(torch.randn(self.queue_size, channels), dim=1))
        self.register_buffer("global_memory", F.normalize(torch.randn(self.queue_size, channels), dim=1))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, local_prototypes: torch.Tensor, global_prototypes: torch.Tensor):
        n = min(local_prototypes.shape[0], self.queue_size, 128)
        if n <= 0:
            return
        idx = torch.randperm(local_prototypes.shape[0], device=local_prototypes.device)[:n]
        l = F.normalize(local_prototypes[idx].detach(), dim=1)
        g = F.normalize(global_prototypes[idx].detach(), dim=1)
        ptr = int(self.ptr.item())
        end = ptr + n
        if end <= self.queue_size:
            self.local_memory[ptr:end].copy_(l)
            self.global_memory[ptr:end].copy_(g)
        else:
            first = self.queue_size - ptr
            self.local_memory[ptr:].copy_(l[:first])
            self.global_memory[ptr:].copy_(g[:first])
            self.local_memory[:end - self.queue_size].copy_(l[first:])
            self.global_memory[:end - self.queue_size].copy_(g[first:])
        self.ptr.fill_(end % self.queue_size)

    def forward(self, local_prototypes: torch.Tensor, global_prototypes: torch.Tensor, update: bool = True) -> torch.Tensor:
        if local_prototypes.shape[0] < 2:
            return local_prototypes.new_tensor(0.0)
        qn = min(local_prototypes.shape[0], 512)
        qidx = torch.randperm(local_prototypes.shape[0], device=local_prototypes.device)[:qn]
        lq = F.normalize(local_prototypes[qidx], dim=1)
        gq = F.normalize(global_prototypes[qidx], dim=1)

        lmem = F.normalize(self.local_memory.detach(), dim=1)
        gmem = F.normalize(self.global_memory.detach(), dim=1)
        logits_l = torch.mm(lq, lmem.t()) / max(self.temperature, 1e-6)
        logits_g = torch.mm(gq, gmem.t()) / max(self.temperature, 1e-6)

        p_l = F.log_softmax(logits_l, dim=1)
        p_g = F.log_softmax(logits_g, dim=1)
        t_l = F.softmax(logits_l.detach(), dim=1)
        t_g = F.softmax(logits_g.detach(), dim=1)
        loss = 0.5 * (F.kl_div(p_l, t_g, reduction="batchmean") + F.kl_div(p_g, t_l, reduction="batchmean"))
        if update:
            self._dequeue_and_enqueue(local_prototypes, global_prototypes)
        return loss

class StructuralConsistencyBlock(nn.Module):
    """
    Prototype-structural-inspired dense graph block for MCD.

    It contains the three Structural components: (1) local/global graph-filter views,
    (2) EM-style low-rank compactness reconstruction, and (3) anchor-distribution
    consistency between local and global compact embeddings.
    """
    def __init__(
        self,
        channels: int,
        rank: int = 32,
        stage_num: int = 3,
        beta: float = 1.0,
        alpha: float = 0.2,
        filter_steps: int = 3,
        queue_size: int = 128,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.filter_steps = int(filter_steps)
        self.local_mlp = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.global_mlp = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.compact = CompactPrototypeLearning(rank=rank, stage_num=stage_num, beta=beta, lamd=1.0)
        self.consistency = LocalGlobalConsistency(channels, queue_size=queue_size, temperature=temperature)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.refine = DepthwiseSeparableConv(channels, channels, 3, 1)
        self.last_consistency_loss: Optional[torch.Tensor] = None

    @staticmethod
    def _avg_filter(x: torch.Tensor, kernel: int) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)

    def _local_filter(self, x: torch.Tensor) -> torch.Tensor:
        # Generalized Laplacian smoothing approximation on local image graph.
        h = x
        for _ in range(max(1, self.filter_steps)):
            h = 0.5 * h + 0.5 * self._avg_filter(h, 3)
        return h

    def _global_filter(self, x: torch.Tensor) -> torch.Tensor:
        # PPR-style diffusion approximation. Smaller alpha gives more global
        # integration, consistent with Structural's graph diffusion discussion.
        h = x
        for _ in range(max(1, self.filter_steps * 2)):
            h = self.alpha * x + (1.0 - self.alpha) * self._avg_filter(h, 5)
        return h

    def _to_prototypes(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])

    def _from_prototypes(self, prototypes: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        b, c, h, w = ref.shape
        return prototypes.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_mlp(self._local_filter(x))
        global_ = self.global_mlp(self._global_filter(x))

        lt = self._to_prototypes(local)
        gt = self._to_prototypes(global_)
        all_prototypes = torch.cat([lt, gt], dim=0)
        compact_prototypes = self.compact(all_prototypes)
        lt_c, gt_c = torch.split(compact_prototypes, lt.shape[0], dim=0)

        # Structural distribution consistency, only used as an auxiliary loss in the
        # student branch. It is still computed in forward so the backbone can
        # expose it to the training script.
        if self.training and torch.is_grad_enabled():
            self.last_consistency_loss = self.consistency(lt_c, gt_c, update=True)
        else:
            self.last_consistency_loss = x.new_tensor(0.0)

        local_c = self._from_prototypes(lt_c, local)
        global_c = self._from_prototypes(gt_c, global_)
        fused = 0.5 * (local_c + global_c)
        gate = self.fuse_gate(torch.cat([x, local_c, global_c], dim=1))
        return x + gate * self.refine(fused)

class CompactGraphBackbone(nn.Module):
    """Local structural backbone with Conv/Graph/Structural stages only.

    This variant stops at local_feats=[f1,f2,f3,f4]. It removes the semantic
    context path and dense feature fusion, and uses out=local_feats[-1].
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c1 = out_channels // 2
        c2 = out_channels // 2
        c3 = out_channels
        c4 = out_channels

        self.conv1 = ConvBNReLU(in_channels, c1, 3, 1)
        self.conv2 = ConvBNReLU(c1, c2, 3, 1)
        self.conv3 = ConvBNReLU(c2, c3, 3, 1)
        self.conv4 = ConvBNReLU(c3, c4, 3, 1)

        self.gm1 = GraphStructureBlock(c1)
        self.gm2 = GraphStructureBlock(c2)
        self.gm3 = GraphStructureBlock(c3)
        self.gm4 = GraphStructureBlock(c4)

        # Prototype-structural blocks: local/global graph filtering + EM low-rank
        # compactness + anchor-distribution consistency.
        self.compact1 = StructuralConsistencyBlock(c1, rank=max(8, c1 // 2))
        self.compact2 = StructuralConsistencyBlock(c2, rank=max(8, c2 // 2))
        self.compact3 = StructuralConsistencyBlock(c3, rank=max(16, c3 // 2))
        self.compact4 = StructuralConsistencyBlock(c4, rank=max(16, c4 // 2))

        self.out_channels_list = [c1, c2, c3, c4]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f1 = self.compact1(self.gm1(self.conv1(x)))
        f2 = self.compact2(self.gm2(self.conv2(f1)))
        f3 = self.compact3(self.gm3(self.conv3(f2)))
        f4 = self.compact4(self.gm4(self.conv4(f3)))
        local_feats = [f1, f2, f3, f4]
        out = local_feats[-1]

        if return_features:
            return out, local_feats
        return out

    def get_coco_loss(self) -> torch.Tensor:
        losses = []
        for blk in [self.compact1, self.compact2, self.compact3, self.compact4]:
            if getattr(blk, "last_consistency_loss", None) is not None:
                losses.append(blk.last_consistency_loss)
        if not losses:
            return next(self.parameters()).new_tensor(0.0)
        return torch.stack(losses).mean()

class PrototypeProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
        out_dim: int = 256,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, bottleneck_dim)
        self.act2 = nn.GELU()
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def _mlp(self, x_flat: torch.Tensor) -> torch.Tensor:
        z = self.act1(self.fc1(x_flat))
        z = self.act2(self.fc2(z))
        z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        z = self._mlp(x)
        return self.last_layer(z)

class COMPASSNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 64,
        head_hidden: int = 512,
        head_bottleneck: int = 256,
        out_dim: int = 256,
        teacher_momentum: float = 0.996,
    ):
        super().__init__()
        self.student_backbone = CompactGraphBackbone(in_channels=in_channels, out_channels=feat_dim)
        self.prototype_teacher_backbone = CompactGraphBackbone(in_channels=in_channels, out_channels=feat_dim)

        self.prototype_student_head = PrototypeProjectionHead(
            in_dim=feat_dim,
            hidden_dim=head_hidden,
            bottleneck_dim=head_bottleneck,
            out_dim=out_dim,
        )
        self.prototype_teacher_head = PrototypeProjectionHead(
            in_dim=feat_dim,
            hidden_dim=head_hidden,
            bottleneck_dim=head_bottleneck,
            out_dim=out_dim,
        )

        self._init_teacher()
        for p in self.prototype_teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.prototype_teacher_head.parameters():
            p.requires_grad = False

        self.teacher_momentum = teacher_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def _init_teacher(self):
        for s, t in zip(self.student_backbone.parameters(), self.prototype_teacher_backbone.parameters()):
            t.data.copy_(s.data)
        for s, t in zip(self.prototype_student_head.parameters(), self.prototype_teacher_head.parameters()):
            t.data.copy_(s.data)

    @torch.no_grad()
    def update_teacher(self, m: Optional[float] = None):
        if m is None:
            m = self.teacher_momentum
        for s, t in zip(self.student_backbone.parameters(), self.prototype_teacher_backbone.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)
        for s, t in zip(self.prototype_student_head.parameters(), self.prototype_teacher_head.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)

    @torch.no_grad()
    def update_center_logits(self, teacher_logits: torch.Tensor, momentum: float = 0.9):
        batch_center = teacher_logits.detach().mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)

    @torch.no_grad()
    def teacher_forward(
        self,
        x: torch.Tensor,
        T: float = 0.04,
        return_logits: bool = False,
        return_features: bool = False,
    ) -> Dict[str, object]:
        feat, feats = self.prototype_teacher_backbone(x, return_features=True)
        logits = self.prototype_teacher_head(feat)
        centered_logits = (logits - self.center) / T
        probs = F.softmax(centered_logits, dim=-1)
        out: Dict[str, object] = {"probs": probs}
        if return_logits:
            out["logits"] = logits
        if return_features:
            out["feat"] = feat
            out["feats"] = feats
        return out

    def student_forward_logits(
        self,
        x: torch.Tensor,
        branch: str = "",
        T: float = 0.1,
        return_feat: bool = False,
    ) -> Dict[str, object]:
        feat, feats = self.student_backbone(x, return_features=True)
        coco_loss = self.student_backbone.get_coco_loss() if hasattr(self.student_backbone, "get_coco_loss") else feat.new_tensor(0.0)
        logits = self.prototype_student_head(feat) / T
        log_probs = F.log_softmax(logits, dim=-1)
        out: Dict[str, object] = {"log_probs": log_probs, "coco_loss": coco_loss}
        if return_feat:
            out["feat"] = feat
            out["feats"] = feats
        return out

    @torch.no_grad()
    def inference_forward(self, x: torch.Tensor, T_student: float = 0.1) -> Dict[str, object]:
        feat, feats = self.student_backbone(x, return_features=True)
        proto_logits = self.prototype_student_head(feat) / T_student
        proto_probs = F.softmax(proto_logits, dim=-1)
        return {
            "feat": feat,
            "feats": feats,
            "proto_logits": proto_logits,
            "proto_probs": proto_probs,
        }

    @staticmethod
    @torch.no_grad()
    def batch_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)

    @staticmethod
    @torch.no_grad()
    def batch_confidence_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        K = probs.shape[-1]
        ent = COMPASSNetwork.batch_entropy(probs, eps=eps)
        max_ent = math.log(K + 1e-12)
        return (1.0 - ent / max_ent).clamp(0.0, 1.0)

# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def dino_loss(student_logp: torch.Tensor, teacher_p: torch.Tensor) -> torch.Tensor:
    return torch.sum(-teacher_p.detach() * student_logp, dim=-1).mean()

def weighted_dino_loss(
    student_logp: torch.Tensor,
    teacher_p: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    per_sample = torch.sum(-teacher_p.detach() * student_logp, dim=-1)
    if sample_weight is None:
        return per_sample.mean()
    w = sample_weight.detach()
    w = w / (w.sum() + 1e-6)
    return torch.sum(per_sample * w)

@torch.no_grad()
def js_divergence_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def normalize_to_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def soft_unchanged_mask(f1: torch.Tensor, f2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = (f1 - f2).abs().mean(dim=1, keepdim=True)
    diff = normalize_to_01(diff, eps=eps)
    scale = diff.mean(dim=(-2, -1), keepdim=True).detach() + eps
    mask = torch.exp(-diff / scale)
    return mask.clamp(0.05, 1.0)

def masked_smooth_l1(f1: torch.Tensor, f2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(f1, f2, reduction="none").mean(dim=1, keepdim=True)
    return (loss * mask).sum() / (mask.sum() + 1e-6)

def masked_covariance_loss(f1: torch.Tensor, f2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    def _masked_cov(feat: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        w = m.expand_as(feat)
        x = feat * w
        x = x.flatten(2)
        w_flat = w.flatten(2)
        denom = w_flat.sum(dim=2, keepdim=True) + 1e-6
        mean = x.sum(dim=2, keepdim=True) / denom
        x = x - mean
        cov = torch.bmm(x, x.transpose(1, 2)) / denom.squeeze(-1).unsqueeze(-1)
        return cov

    c1 = _masked_cov(f1, mask)
    c2 = _masked_cov(f2, mask)
    return F.mse_loss(c1, c2)

# Backward-compatible aliases.
COMPASSNetworkNoAugPrototype = COMPASSNetwork
COMPASSNetwork = COMPASSNetwork

# -----------------------------------------------------------------------------
# DualTeacher extension: structural teacher + semantic teacher
# -----------------------------------------------------------------------------
class COMPASSNetwork(COMPASSNetwork):
    """
    Clean training-time teacher version for unsupervised cross-modal change detection.

    The original StructuralOfficial model contained prototype, structural, and semantic
    EMA teachers. Since the current experiment sets w_sem = 0, the semantic EMA
    teacher and its loss are removed. The class name is kept for backward
    compatibility with existing training scripts.

    Active teachers:
      1) Prototype EMA teacher: stable prototype-level self-distillation target.
      2) Structural EMA teacher: graph-manifold/commonality feature target.

    During inference only the student backbone/head are used.
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 64,
        head_hidden: int = 512,
        head_bottleneck: int = 256,
        out_dim: int = 256,
        teacher_momentum: float = 0.996,
        structural_teacher_momentum: float = 0.996,
    ):
        super().__init__(
            in_channels=in_channels,
            feat_dim=feat_dim,
            head_hidden=head_hidden,
            head_bottleneck=head_bottleneck,
            out_dim=out_dim,
            teacher_momentum=teacher_momentum,
        )
        self.structural_prototype_teacher_backbone = CompactGraphBackbone(in_channels=in_channels, out_channels=feat_dim)
        self.structural_teacher_momentum = structural_teacher_momentum
        self._init_extra_teachers()
        for p in self.structural_prototype_teacher_backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _init_extra_teachers(self):
        for s, t in zip(self.student_backbone.parameters(), self.structural_prototype_teacher_backbone.parameters()):
            t.data.copy_(s.data)

    @torch.no_grad()
    def update_teacher(self, m: Optional[float] = None):
        # Update prototype teacher inherited from the base class.
        super().update_teacher(m=m)
        ms = self.structural_teacher_momentum if m is None else m
        for s, t in zip(self.student_backbone.parameters(), self.structural_prototype_teacher_backbone.parameters()):
            t.data.mul_(ms).add_(s.data, alpha=1.0 - ms)

    @torch.no_grad()
    def teacher_forward_dual(
        self,
        x: torch.Tensor,
        T: float = 0.04,
        return_logits: bool = False,
        return_features: bool = False,
    ) -> Dict[str, object]:
        # Prototype teacher: keeps the original EMA Prototype-style target.
        proto_feat, proto_feats = self.prototype_teacher_backbone(x, return_features=True)
        logits = self.prototype_teacher_head(proto_feat)
        centered_logits = (logits - self.center) / T
        probs = F.softmax(centered_logits, dim=-1)

        # Structural teacher: graph-manifold local features are used as
        # privileged structural targets.
        _, struct_feats = self.structural_prototype_teacher_backbone(x, return_features=True)

        out: Dict[str, object] = {"probs": probs}
        if return_logits:
            out["logits"] = logits
        if return_features:
            out["feat"] = proto_feat
            out["feats"] = proto_feats
            out["struct_teacher_feats"] = struct_feats
        return out

# -----------------------------------------------------------------------------
# DualTeacher losses
# -----------------------------------------------------------------------------
def _pooled_norm_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pa = F.normalize(a.mean(dim=[2, 3]), dim=-1)
    pb = F.normalize(b.detach().mean(dim=[2, 3]), dim=-1)
    return F.mse_loss(pa, pb)

def structural_teacher_distillation_loss(
    s_feats1: List[torch.Tensor],
    s_feats2: List[torch.Tensor],
    t_struct_feats1: List[torch.Tensor],
    t_struct_feats2: List[torch.Tensor],
) -> torch.Tensor:
    """
    Structural teacher distillation.

    The structural teacher supplies privileged graph-manifold evidence. It has
    two roles:
      1) same-input target: student structural features should follow EMA
         structural features;
      2) cross-time commonality target: the teacher-derived unchanged mask tells
         where cross-modal structural alignment is reliable.
    """
    loss = 0.0
    n = len(s_feats1)
    for s1, s2, t1, t2 in zip(s_feats1, s_feats2, t_struct_feats1, t_struct_feats2):
        same = 0.5 * (_pooled_norm_mse(s1, t1) + _pooled_norm_mse(s2, t2))
        with torch.no_grad():
            mask = soft_unchanged_mask(t1.detach(), t2.detach())
        cross = 0.5 * masked_smooth_l1(s1, s2, mask) + 0.5 * masked_covariance_loss(s1, s2, mask)
        loss = loss + same + cross
    return loss / float(max(1, n))

# Backward-compatible aliases for this specific variant.
COMPASSNetworkDT = COMPASSNetwork
COMPASSNetworkDualTeacher = COMPASSNetwork
