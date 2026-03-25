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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            target = target.unsqueeze(1)

        band_mask = (target < self.band_threshold).float()
        n_band    = band_mask.sum().clamp(min=1.0)

        loss = F.smooth_l1_loss(
            pred * band_mask, target * band_mask, reduction="sum"
        ) / n_band

        if self.sobel_weight > 0.0:
            # Cast buffers to pred's dtype — handles bf16/fp16 AMP forward passes.
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

