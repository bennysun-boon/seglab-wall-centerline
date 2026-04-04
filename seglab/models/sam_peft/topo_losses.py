"""Topology-aware losses: Dice, clDice, soft skeletonization."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary masks."""
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable skeletonization."""
    skel = torch.zeros_like(img)
    for _ in range(iters):
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
        img = soft_erode(img)
    return skel


def cldice_loss(pred: torch.Tensor, target: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Centerline Dice loss."""
    skel_pred = soft_skeletonize(pred, iters)
    skel_true = soft_skeletonize(target, iters)

    tprec = (skel_pred * target).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    tsens = (skel_true * pred).sum(dim=(1, 2, 3)) / (skel_true.sum(dim=(1, 2, 3)) + eps)
    cldice = (2 * tprec * tsens + eps) / (tprec + tsens + eps)
    return 1 - cldice.mean()


def boundary_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable boundary Dice using Laplacian edges."""
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred.device, dtype=pred.dtype)
    lap = lap.view(1, 1, 3, 3)
    pred_e = F.conv2d(pred, lap, padding=1).abs()
    tgt_e = F.conv2d(target, lap, padding=1).abs()
    return dice_loss(pred_e, tgt_e, eps=eps)


class BorderEDTLoss(nn.Module):
    """Band-masked smooth-L1 + optional Sobel gradient loss for border EDT regression.

    The EDT target encodes unsigned distance to the nearest paving boundary,
    normalised to [0, 1] and truncated (pixels beyond the truncation radius are
    clamped to 1.0).  Supervising on the clamped plateau provides no useful
    gradient signal, so a band mask excludes those pixels.

    Sobel kernels are registered as persistent buffers so they are created once,
    live on the correct device, and survive dtype changes under AMP without
    requiring dtype=pred.dtype casts on every forward call.

    Args:
        band_threshold:  Pixels with target < this are supervised.  Default 0.98
                         excludes the clamped plateau at max truncation.
        sobel_weight:    Weight for Sobel gradient loss.  Penalises flat trough
                         predictions by matching the spatial gradient slope —
                         producing a sharper boundary trough and more precise
                         threshold-based border recovery at inference.
    """

    def __init__(self, band_threshold: float = 0.98, sobel_weight: float = 0.0):
        super().__init__()
        self.band_threshold = band_threshold
        self.sobel_weight   = sobel_weight

        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                seg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if target.ndim == 3:
            target = target.unsqueeze(1)

        # Band mask: exclude saturated plateau (far background / deep interior)
        band_mask = (target < self.band_threshold).float()

        # Seg mask: EDT only meaningful inside or near paving regions.
        # Use a dilated version so boundary pixels just outside the mask are included.
        if seg_mask is not None:
            if seg_mask.ndim == 3:
                seg_mask = seg_mask.unsqueeze(1)
            # Dilate by ~5px to include pixels just outside the boundary
            seg_dilated = F.max_pool2d(seg_mask.float(), kernel_size=11, stride=1, padding=5)
            band_mask = band_mask * seg_dilated

        n_band = band_mask.sum().clamp(min=1.0)

        loss = F.smooth_l1_loss(
            pred * band_mask, target * band_mask, reduction="sum"
        ) / n_band

        if self.sobel_weight > 0.0:
            sx = self.sobel_x.to(dtype=pred.dtype)
            sy = self.sobel_y.to(dtype=pred.dtype)
            pred_gx = F.conv2d(pred,   sx, padding=1)
            pred_gy = F.conv2d(pred,   sy, padding=1)
            tgt_gx  = F.conv2d(target, sx, padding=1)
            tgt_gy  = F.conv2d(target, sy, padding=1)
            sobel_loss = (
                ((pred_gx - tgt_gx) * band_mask).abs().sum() +
                ((pred_gy - tgt_gy) * band_mask).abs().sum()
            ) / (2.0 * n_band)
            loss = loss + self.sobel_weight * sobel_loss

        return loss


def border_edt_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    band_threshold: float = 0.98,
    sobel_weight: float = 0.0,
) -> torch.Tensor:
    """Functional wrapper around BorderEDTLoss for one-off / test calls."""
    fn = BorderEDTLoss(band_threshold=band_threshold, sobel_weight=sobel_weight)
    fn = fn.to(device=pred.device)
    return fn(pred, target)


# ── Polygon head losses (vmap + voff) ─────────────────────────────────────────

def vmap_loss(pred: torch.Tensor, target: torch.Tensor,
              seg_mask: Optional[torch.Tensor] = None,
              focal_weight: float = 10.0, dice_weight: float = 0.5,
              alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """FocalDice loss for the vertex heatmap head.

    Vertices occupy ~0.1% of pixels — standard BCE collapses to predicting zeros.
    Focal loss emphasises hard positives; dice provides overlap-level supervision.

    Args:
        pred:         (B, 1, H, W) raw logits
        target:       (B, 1, H, W) Gaussian heatmap targets in [0, 1]
        seg_mask:     (B, 1, H, W) binary seg mask — restrict supervision to
                      inside/near paving regions (vertices can't be in background)
        focal_weight: multiplier for focal term (default 10 from SAMPolyBuild)
        dice_weight:  multiplier for dice term
        alpha:        focal positive class weight
        gamma:        focal focusing parameter
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)

    if seg_mask is not None:
        if seg_mask.ndim == 3:
            seg_mask = seg_mask.unsqueeze(1)
        seg_mask = seg_mask.float()

    # Focal loss with soft targets
    prob = torch.sigmoid(pred)
    ce   = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt   = prob * target + (1 - prob) * (1 - target)
    focal_factor = alpha * target + (1 - alpha) * (1 - target)
    focal = focal_factor * (1 - pt) ** gamma * ce

    if seg_mask is not None:
        n = seg_mask.sum().clamp(min=1.0)
        l_focal = (focal * seg_mask).sum() / n
    else:
        l_focal = focal.mean()

    # Dice loss restricted to seg mask
    if seg_mask is not None:
        l_dice = dice_loss(prob * seg_mask, target * seg_mask)
    else:
        l_dice = dice_loss(prob, target)

    return focal_weight * l_focal + dice_weight * l_dice


# ── Frame Field losses (Girard et al., CVPR 2021) ─────────────────────────────

def _ff_poly_abs_sq(ff: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Compute |f(e^{iθ}; c0, c2)|² using real trig arithmetic.

    The polynomial f(z) = z⁴ + c₂z² + c₀ with z = e^{iθ} expands to:
        real = cos4θ + c2_re·cos2θ - c2_im·sin2θ + c0_re
        imag = sin4θ + c2_re·sin2θ + c2_im·cos2θ + c0_im
        |f|² = real² + imag²

    Works natively in bf16/fp16/fp32 — no torch.complex or upcasting needed.

    ff:    (B, 4, H, W) — [c0_re, c0_im, c2_re, c2_im], tanh-bounded
    theta: (B, 1, H, W) — angle in [0, π)
    Returns: (B, 1, H, W) non-negative scalar per pixel
    """
    c0_re, c0_im = ff[:, 0:1], ff[:, 1:2]
    c2_re, c2_im = ff[:, 2:3], ff[:, 3:4]

    cos2 = torch.cos(2.0 * theta)
    sin2 = torch.sin(2.0 * theta)
    cos4 = torch.cos(4.0 * theta)
    sin4 = torch.sin(4.0 * theta)

    real = cos4 + c2_re * cos2 - c2_im * sin2 + c0_re
    imag = sin4 + c2_re * sin2 + c2_im * cos2 + c0_im

    return real ** 2 + imag ** 2


def ff_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice on binary edge predictions.

    Edges are ~5-10% of pixels — no focal loss needed (unlike vmap).

    Args:
        pred:   (B, 1, H, W) raw logits from edge_branch
        target: (B, 1, H, W) or (B, H, W) float binary edge map in [0, 1]
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    d    = dice_loss(torch.sigmoid(pred), target)
    return bce + d


def ff_align_loss(ff: torch.Tensor, theta: torch.Tensor,
                  edge_mask: torch.Tensor) -> torch.Tensor:
    """Align frame field to GT tangent directions at edge pixels.

    Evaluates |f(e^{iθ}; c0, c2)|² at each edge pixel, where
    f(z) = z^4 + c2*z^2 + c0 should → 0 when z aligns with the edge tangent.

    Args:
        ff:        (B, 4, H, W) — [Re(c0), Im(c0), Re(c2), Im(c2)], tanh output
        theta:     (B, 1, H, W) or (B, H, W) GT tangent angle in [0, π)
        edge_mask: (B, 1, H, W) float, 1 at edge pixels
    """
    if theta.ndim == 3:
        theta = theta.unsqueeze(1)
    if edge_mask.ndim == 3:
        edge_mask = edge_mask.unsqueeze(1)

    poly_sq = _ff_poly_abs_sq(ff, theta)
    n = edge_mask.sum().clamp(min=1.0)
    return (edge_mask * poly_sq).sum() / n


def ff_align90_loss(ff: torch.Tensor, theta: torch.Tensor,
                    edge_mask: torch.Tensor) -> torch.Tensor:
    """Align frame field to the perpendicular direction at edge pixels.

    Prevents the field from collapsing to a single-direction (line) field
    by also aligning to the direction 90° off the tangent.
    """
    if theta.ndim == 3:
        theta = theta.unsqueeze(1)
    if edge_mask.ndim == 3:
        edge_mask = edge_mask.unsqueeze(1)

    theta_perp = theta + (torch.pi / 2)
    poly_sq = _ff_poly_abs_sq(ff, theta_perp)
    n = edge_mask.sum().clamp(min=1.0)
    return (edge_mask * poly_sq).sum() / n


def ff_smooth_loss(ff: torch.Tensor,
                   edge_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Laplacian penalty — penalises curvature in c0 and c2 (lydorn spec).

    Uses the same 3x3 Laplacian kernel as lydorn's LaplacianPenalty:
        [[0.5, 1.0, 0.5],
         [1.0, -6., 1.0],
         [0.5, 1.0, 0.5]] / 12

    Smoothness is only enforced in non-edge regions (lydorn spec:
    avg_penalty = mean(penalty * gt_edges_inv)). At edge pixels the frame
    field is allowed to be discontinuous.

    Args:
        ff:        (B, 4, H, W) frame field coefficients
        edge_mask: (B, 1, H, W) float, 1 at GT edge pixels. If None,
                   smoothness is applied everywhere (legacy behaviour).
    """
    # Laplacian kernel (lydorn: frame_field_utils.LaplacianPenalty)
    kernel = ff.new_tensor([[0.5, 1.0, 0.5],
                            [1.0, -6., 1.0],
                            [0.5, 1.0, 0.5]]) / 12.0
    kernel = kernel[None, None, :, :].expand(4, -1, -1, -1)  # (4,1,3,3)
    penalty = torch.abs(F.conv2d(ff, kernel, padding=1, groups=4))  # (B,4,H,W)

    if edge_mask is not None:
        if edge_mask.ndim == 3:
            edge_mask = edge_mask.unsqueeze(1)
        non_edge = (1.0 - edge_mask).clamp(0.0, 1.0)
        return torch.mean(penalty * non_edge)

    return penalty.mean()


def ff_interior_coupling_loss(ff: torch.Tensor,
                               seg_logits: torch.Tensor) -> torch.Tensor:
    """Couple frame field direction to seg gradient (Lint_align from FFL).

    The spatial gradient of the interior seg map points perpendicular to
    boundaries — this provides free supervision for the frame field without
    needing GT theta at non-edge pixels. Gradients backprop only through ff.

    Args:
        ff:         (B, 4, H, W) frame field coefficients
        seg_logits: (B, 1, H, W) seg logits — detached, used as signal source only
    """
    seg_prob = torch.sigmoid(seg_logits.detach())

    gx = F.pad(seg_prob[:, :, :, 1:] - seg_prob[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(seg_prob[:, :, 1:, :] - seg_prob[:, :, :-1, :], (0, 0, 0, 1))
    grad_mag = (gx ** 2 + gy ** 2).sqrt()
    grad_dir = torch.atan2(gy, gx)

    poly_sq = _ff_poly_abs_sq(ff, grad_dir)
    return (grad_mag * poly_sq).mean()


def ff_edge_coupling_loss(edge_pred: torch.Tensor,
                           seg_logits: torch.Tensor) -> torch.Tensor:
    """Couple edge prediction to seg gradient magnitude (Lint_edge from FFL).

    Edge prediction should match where the interior map has strong gradients.
    Weighted by max(1 - seg_prob, grad_mag) to focus on boundary + background.
    """
    seg_prob = torch.sigmoid(seg_logits.detach())
    gx = F.pad(seg_prob[:, :, :, 1:] - seg_prob[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(seg_prob[:, :, 1:, :] - seg_prob[:, :, :-1, :], (0, 0, 0, 1))
    grad_mag  = (gx ** 2 + gy ** 2).sqrt()
    edge_prob = torch.sigmoid(edge_pred)
    weight    = torch.max(1.0 - seg_prob, grad_mag)
    return (weight * (grad_mag - edge_prob).abs()).mean()


def voff_loss(pred: torch.Tensor, target: torch.Tensor,
              vmask: torch.Tensor) -> torch.Tensor:
    """Masked L1 loss for the vertex offset head.

    Supervises only pixels within the vmap supervision radius (vmask > 0).
    Inverse-frequency weights balance the contribution of vertex-dense tiles
    vs sparse tiles.

    Args:
        pred:   (B, 2, H, W) raw logits — sigmoid maps to offset in [-1, 1]
        target: (B, 2, H, W) normalised offsets in [-1, 1] (dx/radius, dy/radius)
        vmask:  (B, 1, H, W) or (B, H, W) float supervision mask (1 = supervised)
    """
    if vmask.ndim == 3:
        vmask = vmask.unsqueeze(1)  # (B, 1, H, W)

    pred_off = torch.sigmoid(pred) * 2.0 - 1.0   # remap [0,1] → [-1,1]
    loss = (pred_off - target).abs()

    # Inverse-frequency weighting per sample so sparse tiles aren't drowned out
    w = vmask.mean(dim=[-2, -1], keepdim=True).clamp(min=1e-6)
    loss = loss * vmask / w

    return loss.mean()

