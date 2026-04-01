"""Model builders and Lightning modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf

from seglab.metrics.segmentation import SegmentationMeter, boundary_f_score
from seglab.metrics.topology import cldice_score
from seglab.metrics.calibration import expected_calibration_error
from seglab.models.sam_peft.topo_losses import (
    dice_loss,
    cldice_loss,
    boundary_loss,
    BorderEDTLoss,
    vmap_loss,
    voff_loss,
    ff_edge_loss,
    ff_align_loss,
    ff_align90_loss,
    ff_smooth_loss,
    ff_interior_coupling_loss,
    ff_edge_coupling_loss,
)


class LitBinarySeg(pl.LightningModule):
    """Generic Lightning module for binary segmentation."""

    def __init__(self, net: nn.Module, cfg: Any) -> None:
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))  # type: ignore

        # Detect if auxiliary heads are active
        self._has_junction_head = (
            hasattr(net, "junction_head") and net.junction_head is not None
        )
        self._has_centerline_head = (
            hasattr(net, "centerline_head") and net.centerline_head is not None
        )
        self._has_border_edt_head = (
            hasattr(net, "border_edt_head") and net.border_edt_head is not None
        )
        self._has_poly_head = (
            hasattr(net, "poly_head") and net.poly_head is not None
        )
        self._has_frame_field_head = (
            hasattr(net, "frame_field_head") and net.frame_field_head is not None
        )

        # Freeze existing params if configured (for transfer learning)
        # Only new head(s) and explicitly unfrozen layers remain trainable
        if getattr(cfg, "freeze_existing", False):
            trainable_keywords = []
            if self._has_junction_head:
                trainable_keywords.append("junction_head")
            if self._has_centerline_head:
                trainable_keywords.append("centerline_head")
            # Allow config to unfreeze specific layers (e.g., output_upscaling)
            unfreeze_keywords = list(getattr(cfg, "unfreeze_keywords", []))
            trainable_keywords.extend(unfreeze_keywords)
            if self._has_border_edt_head:
                trainable_keywords.append("border_edt_head")
            if self._has_poly_head:
                trainable_keywords.append("poly_head")
            if self._has_frame_field_head:
                trainable_keywords.append("frame_field_head")
            if trainable_keywords:
                for name, param in net.named_parameters():
                    if not any(kw in name for kw in trainable_keywords):
                        param.requires_grad = False

        # BorderEDTLoss owns Sobel buffers — instantiated here so Lightning moves
        # them to the correct device when the module is moved.
        if self._has_border_edt_head:
            band_threshold = getattr(getattr(cfg, "loss", {}), "border_edt_band_threshold", 0.98)
            sobel_weight   = getattr(getattr(cfg, "loss", {}), "border_edt_sobel_weight",   0.1)
            self.border_edt_criterion = BorderEDTLoss(
                band_threshold=band_threshold,
                sobel_weight=sobel_weight,
            )
        else:
            self.border_edt_criterion = None

        self.train_meter = SegmentationMeter()
        self.val_meter = SegmentationMeter()
        self.test_meter = SegmentationMeter()
        self._test_run_dir: Optional[Path] = None
        self._test_pred_dir: Optional[Path] = None
        self._test_fig_dir: Optional[Path] = None
        self._test_image_counter: int = 0
        self._test_pixel_probs: List[np.ndarray] = []
        self._test_pixel_targets: List[np.ndarray] = []
        self._test_pixel_count: int = 0
        self._test_bf_sum: float = 0.0
        self._test_bf_n: int = 0
        self._test_cldice_sum: float = 0.0
        self._test_cldice_n: int = 0
        self._test_image_metrics: List[Dict[str, float]] = []
        self._best_samples: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._worst_samples: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))

    def setup(self, stage: str) -> None:
        # Compile the encoder here — after Lightning has loaded transfer weights
        # (doing it in __init__ causes key mismatches with torch.compile's _orig_mod prefix)
        net = self.net
        if getattr(net, "_compile_encoder", False) and hasattr(torch, "compile"):
            if not hasattr(net.sam.image_encoder, "_orig_mod"):  # not already compiled
                net.sam.image_encoder = torch.compile(net.sam.image_encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        out = self.net(x)
        if isinstance(out, (tuple, list)) and len(out) > 1:
            # Multi-head: first element is always seg logits
            results = list(out)
            if results[0].ndim == 3:
                results[0] = results[0].unsqueeze(1)
            return tuple(results)
        # Backward compatible: single-head
        logits = out
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        return logits

    def _compute_seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_f = target.float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, target_f)
        dice = dice_loss(torch.sigmoid(logits), target_f)

        loss = (
            self.cfg.loss.bce_weight * bce
            + self.cfg.loss.dice_weight * dice
        )

        if getattr(self.cfg.loss, "cldice_weight", 0.0) > 0:
            loss = loss + self.cfg.loss.cldice_weight * cldice_loss(
                torch.sigmoid(logits), target_f
            )
        if getattr(self.cfg.loss, "boundary_weight", 0.0) > 0:
            loss = loss + self.cfg.loss.boundary_weight * boundary_loss(
                torch.sigmoid(logits), target_f
            )
        return loss

    def _compute_junction_loss(self, junction_logits: torch.Tensor,
                               junction_target: torch.Tensor) -> torch.Tensor:
        """Weighted BCE + Dice loss on junction heatmap predictions.

        Weighted BCE upweights positive (junction) pixels to counter the extreme
        class imbalance (~0.1% positive). Dice provides overlap-based optimization.
        """
        target_f = junction_target.float()
        if target_f.ndim == 3:
            target_f = target_f.unsqueeze(1)

        pred = torch.sigmoid(junction_logits)

        # Weighted BCE: pos_weight upscales the gradient for positive pixels
        pos_weight_val = getattr(self.cfg.loss, "junction_pos_weight", 50.0)
        pos_weight = torch.tensor([pos_weight_val], device=junction_logits.device)
        bce = F.binary_cross_entropy_with_logits(
            junction_logits, target_f, pos_weight=pos_weight
        )

        d = dice_loss(pred, target_f)

        bce_w = getattr(self.cfg.loss, "junction_bce_weight", 1.0)
        dice_w = getattr(self.cfg.loss, "junction_dice_weight", 1.0)

        return bce_w * bce + dice_w * d

    @staticmethod
    def _sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Sobel gradients (Gx, Gy) of a (B, 1, H, W) tensor."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return gx, gy

    def _compute_centerline_loss(self, centerline_pred: torch.Tensor,
                                  centerline_target: torch.Tensor,
                                  mask: torch.Tensor) -> torch.Tensor:
        """Smooth-L1 regression loss + masked Sobel gradient loss on distance transform.

        Three components:
        1. Wall pixels: smooth-L1 between predicted and target distance values.
        2. Background pixels: L1 penalty pushing predictions to zero.
        3. Sobel gradient loss (wall pixels only): penalizes flat ridge predictions
           by matching the gradient (slope) of the predicted DT to the target DT.
           Masked to wall pixels so high-pass Sobel doesn't amplify background noise.
        """
        target_f = centerline_target.float()
        if target_f.ndim == 3:
            target_f = target_f.unsqueeze(1)
        mask_f = mask.float()
        if mask_f.ndim == 3:
            mask_f = mask_f.unsqueeze(1)

        wall_mask = (mask_f > 0.5).float()
        bg_mask = 1.0 - wall_mask

        # 1. Wall pixels: smooth-L1 on distance values
        n_wall = wall_mask.sum().clamp(min=1.0)
        wall_loss = F.smooth_l1_loss(
            centerline_pred * wall_mask, target_f * wall_mask, reduction="sum"
        ) / n_wall

        # 2. Background pixels: push predictions to zero
        n_bg = bg_mask.sum().clamp(min=1.0)
        bg_loss = (centerline_pred * bg_mask).abs().sum() / n_bg

        bg_weight = getattr(self.cfg.loss, "centerline_bg_weight", 2.0)
        sobel_weight = getattr(self.cfg.loss, "centerline_sobel_weight", 0.0)

        loss = wall_loss + bg_weight * bg_loss

        # 3. Sobel gradient loss — masked strictly to wall pixels
        if sobel_weight > 0.0:
            pred_gx, pred_gy = self._sobel_gradients(centerline_pred)
            tgt_gx, tgt_gy = self._sobel_gradients(target_f)
            # Mask both gradient maps to wall pixels only
            sobel_loss = (
                ((pred_gx - tgt_gx) * wall_mask).abs().sum() +
                ((pred_gy - tgt_gy) * wall_mask).abs().sum()
            ) / (2.0 * n_wall)
            loss = loss + sobel_weight * sobel_loss

        return loss

    def _compute_border_edt_loss(
        self,
        edt_pred: torch.Tensor,
        edt_target: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Delegate to the BorderEDTLoss module (Sobel buffers on correct device)."""
        if edt_target.ndim == 3:
            edt_target = edt_target.unsqueeze(1)
        return self.border_edt_criterion(edt_pred, edt_target, seg_mask=seg_mask)  # type: ignore[union-attr]

    def _compute_poly_loss(
        self,
        poly_pred: dict,
        vmap_target: torch.Tensor,
        voff_target: torch.Tensor,
        vmask: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FocalDice vmap + masked-L1 voff losses."""
        if vmap_target.ndim == 3:
            vmap_target = vmap_target.unsqueeze(1)
        if voff_target.ndim == 3:
            voff_target = voff_target.unsqueeze(1)

        vmap_w      = getattr(self.cfg.loss, "vmap_weight", 1.25)
        voff_w      = getattr(self.cfg.loss, "voff_weight", 5.0)
        focal_alpha = getattr(self.cfg.loss, "vmap_focal_alpha", 0.95)
        focal_gamma = getattr(self.cfg.loss, "vmap_focal_gamma", 2.0)

        l_vmap = vmap_loss(poly_pred["vmap"], vmap_target, seg_mask=seg_mask,
                           alpha=focal_alpha, gamma=focal_gamma)
        l_voff = voff_loss(poly_pred["voff"], voff_target, vmask)
        return vmap_w * l_vmap + voff_w * l_voff

    def _compute_ff_loss(
        self,
        ff_pred: dict,
        edge_target: torch.Tensor,
        theta_target: torch.Tensor,
        seg_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Combine 6 frame field losses with configurable weights."""
        if edge_target.ndim == 3:
            edge_target = edge_target.unsqueeze(1)
        if theta_target.ndim == 3:
            theta_target = theta_target.unsqueeze(1)

        edge_pred = ff_pred["edge"]
        ff_coeff  = ff_pred["ff"]

        w_edge     = getattr(self.cfg.loss, "ff_edge_weight",         0.5)
        w_align    = getattr(self.cfg.loss, "ff_align_weight",        1.0)
        w_align90  = getattr(self.cfg.loss, "ff_align90_weight",      0.2)
        w_smooth   = getattr(self.cfg.loss, "ff_smooth_weight",       0.005)
        w_int_ff   = getattr(self.cfg.loss, "ff_int_coupling_weight", 0.2)
        w_int_edge = getattr(self.cfg.loss, "ff_int_edge_weight",     0.2)

        l_edge      = ff_edge_loss(edge_pred, edge_target)
        l_align     = ff_align_loss(ff_coeff, theta_target, edge_target)
        l_align90   = ff_align90_loss(ff_coeff, theta_target, edge_target)
        l_smooth    = ff_smooth_loss(ff_coeff)
        l_int_ff    = ff_interior_coupling_loss(ff_coeff, seg_logits)
        l_int_edge  = ff_edge_coupling_loss(edge_pred, seg_logits)

        return (w_edge * l_edge + w_align * l_align + w_align90 * l_align90
                + w_smooth * l_smooth + w_int_ff * l_int_ff + w_int_edge * l_int_edge)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch["image"]
        y = batch["mask"]
        out = self(x)

        # Unpack multi-head outputs based on which heads are active
        junction_logits  = None
        centerline_pred  = None
        border_edt_pred  = None
        poly_pred        = None
        frame_field_pred = None
        if isinstance(out, tuple):
            seg_logits = out[0]
            idx = 1
            if self._has_junction_head:
                junction_logits = out[idx]; idx += 1
            if self._has_centerline_head:
                centerline_pred = out[idx]; idx += 1
            if self._has_border_edt_head:
                border_edt_pred = out[idx]; idx += 1
            if self._has_poly_head:
                poly_pred = out[idx]; idx += 1
            if self._has_frame_field_head:
                frame_field_pred = out[idx]; idx += 1
        else:
            seg_logits = out

        # Segmentation loss: compute if shared features are unfrozen (to preserve seg quality)
        freeze_existing = getattr(self.cfg, "freeze_existing", False)
        has_aux = self._has_junction_head or self._has_centerline_head
        has_unfreeze = len(list(getattr(self.cfg, "unfreeze_keywords", []))) > 0
        if freeze_existing and has_aux and not has_unfreeze:
            # Fully frozen: skip seg loss but keep on compute graph
            seg_loss = 0.0 * seg_logits.sum()
        else:
            seg_loss = self._compute_seg_loss(seg_logits, y)

        loss = seg_loss
        self.log(f"{stage}/seg_loss", seg_loss, on_step=False, on_epoch=True)

        # Junction heatmap loss
        has_junction_target = "junction_heatmap" in batch
        if junction_logits is not None and has_junction_target:
            junction_target = batch["junction_heatmap"]
            junction_loss = self._compute_junction_loss(junction_logits, junction_target)
            junction_weight = getattr(self.cfg.loss, "junction_weight", 1.0)
            loss = loss + junction_weight * junction_loss
            self.log(f"{stage}/junction_loss", junction_loss, on_step=False, on_epoch=True)

            # Log junction dice for monitoring/early-stopping
            with torch.no_grad():
                pred = torch.sigmoid(junction_logits)
                target_f = junction_target.float()
                if target_f.ndim == 3:
                    target_f = target_f.unsqueeze(1)
                inter = (pred * target_f).sum()
                union = pred.sum() + target_f.sum()
                junc_dice = (2.0 * inter + 1e-6) / (union + 1e-6)
            self.log(f"{stage}/junction_dice", junc_dice, on_step=False, on_epoch=True, prog_bar=(stage == "val"))

        # Border EDT loss (paving) — restricted to inside/near seg mask
        has_border_edt_target = "border_edt" in batch
        if border_edt_pred is not None and has_border_edt_target:
            border_edt_target = batch["border_edt"]
            seg_prob = torch.sigmoid(seg_logits).detach()
            edt_loss = self._compute_border_edt_loss(
                border_edt_pred, border_edt_target, seg_mask=seg_prob
            )
            border_edt_weight = getattr(self.cfg.loss, "border_edt_weight", 3.0)
            loss = loss + border_edt_weight * edt_loss
            self.log(f"{stage}/border_edt_loss", edt_loss, on_step=False, on_epoch=True)

            with torch.no_grad():
                target_f = border_edt_target.float()
                if target_f.ndim == 3:
                    target_f = target_f.unsqueeze(1)
                band_mask = target_f < (self.border_edt_criterion.band_threshold if self.border_edt_criterion else 0.98)
                if band_mask.sum() > 0:
                    mae = (border_edt_pred[band_mask] - target_f[band_mask]).abs().mean()
                else:
                    mae = torch.tensor(0.0, device=x.device)
            self.log(f"{stage}/border_edt_mae", mae, on_step=False, on_epoch=True,
                     prog_bar=(stage == "val"))

        # Polygon vertex head loss (vmap + voff) — restricted to inside seg mask
        has_poly_target = "vmap" in batch and "voff" in batch
        if poly_pred is not None and has_poly_target:
            seg_prob = torch.sigmoid(seg_logits).detach()
            pl = self._compute_poly_loss(
                poly_pred, batch["vmap"], batch["voff"],
                batch.get("vmask", torch.ones_like(batch["vmap"])),
                seg_mask=seg_prob,
            )
            loss = loss + pl
            self.log(f"{stage}/poly_loss", pl, on_step=False, on_epoch=True)
            with torch.no_grad():
                vmap_prob = torch.sigmoid(poly_pred["vmap"])
                vmap_gt   = batch["vmap"].float()
                if vmap_gt.ndim == 3:
                    vmap_gt = vmap_gt.unsqueeze(1)
                tp = (vmap_prob > 0.5) & (vmap_gt > 0.5)
                fp = (vmap_prob > 0.5) & (vmap_gt <= 0.5)
                fn = (vmap_prob <= 0.5) & (vmap_gt > 0.5)
                vmap_f1 = (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-6))
            self.log(f"{stage}/vmap_f1", vmap_f1, on_step=False, on_epoch=True,
                     prog_bar=(stage == "val"))

        # Frame field loss (edge + alignment + smoothness + coupling)
        has_ff_target = "edge" in batch and "theta" in batch
        if frame_field_pred is not None and has_ff_target:
            ff_loss = self._compute_ff_loss(
                frame_field_pred, batch["edge"], batch["theta"], seg_logits
            )
            ff_weight = getattr(self.cfg.loss, "frame_field_weight", 1.0)
            loss = loss + ff_weight * ff_loss
            self.log(f"{stage}/ff_loss", ff_loss, on_step=False, on_epoch=True)

            with torch.no_grad():
                # ff_angle_error: mean angular deviation at edge pixels (degrees)
                # Primary early-stopping metric — measures directional quality
                edge_mask = batch["edge"].float()
                if edge_mask.ndim == 3:
                    edge_mask = edge_mask.unsqueeze(1)
                theta_gt = batch["theta"].float()
                if theta_gt.ndim == 3:
                    theta_gt = theta_gt.unsqueeze(1)
                n_edge = edge_mask.sum().clamp(min=1.0)
                if n_edge > 10:
                    # Recover predicted angle: find θ that minimises |f(e^{iθ})|²
                    # Approximation: use gradient direction from c2 (dominant term)
                    # c2 = -(u² + v²) ≈ -2u² for u≈v, so angle(u) ≈ atan2(-Im(c2), -Re(c2))/2
                    ff = frame_field_pred["ff"]
                    c2_re = ff[:, 2:3]; c2_im = ff[:, 3:4]
                    pred_angle = (torch.atan2(-c2_im, -c2_re) / 2.0) % torch.pi
                    diff = (pred_angle - theta_gt).abs()
                    diff = torch.min(diff, torch.pi - diff)
                    ff_angle_err = (diff * edge_mask).sum() / n_edge * (180.0 / torch.pi)
                else:
                    ff_angle_err = torch.tensor(90.0, device=seg_logits.device)
            self.log(f"{stage}/ff_angle_error", ff_angle_err, on_step=False, on_epoch=True,
                     prog_bar=(stage == "val"))

            # Edge F1 — sanity gate (should climb fast and stay high)
            with torch.no_grad():
                edge_prob_pred = frame_field_pred["edge"]
                tp = ((edge_prob_pred > 0.5) & (edge_mask > 0.5)).sum().float()
                fp = ((edge_prob_pred > 0.5) & (edge_mask <= 0.5)).sum().float()
                fn = ((edge_prob_pred <= 0.5) & (edge_mask > 0.5)).sum().float()
                edge_f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            self.log(f"{stage}/edge_f1", edge_f1, on_step=False, on_epoch=True)

        # Centerline distance transform loss
        has_centerline_target = "distance_transform" in batch
        if centerline_pred is not None and has_centerline_target:
            centerline_target = batch["distance_transform"]
            centerline_loss = self._compute_centerline_loss(centerline_pred, centerline_target, y)
            centerline_weight = getattr(self.cfg.loss, "centerline_weight", 1.0)
            loss = loss + centerline_weight * centerline_loss
            self.log(f"{stage}/centerline_loss", centerline_loss, on_step=False, on_epoch=True)

            # Log centerline MAE on wall pixels for monitoring
            with torch.no_grad():
                target_f = centerline_target.float()
                if target_f.ndim == 3:
                    target_f = target_f.unsqueeze(1)
                mask_f = y.float()
                if mask_f.ndim == 3:
                    mask_f = mask_f.unsqueeze(1)
                wall_mask = (mask_f > 0.5)
                if wall_mask.sum() > 0:
                    mae = (centerline_pred[wall_mask] - target_f[wall_mask]).abs().mean()
                else:
                    mae = torch.tensor(0.0, device=x.device)
            self.log(f"{stage}/centerline_mae", mae, on_step=False, on_epoch=True, prog_bar=(stage == "val"))

        probs = torch.sigmoid(seg_logits)
        meter = getattr(self, f"{stage}_meter")
        meter.update(probs.detach(), y.detach())
        self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["image"]
        y = batch["mask"]
        out = self(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = self._compute_seg_loss(logits, y)
        probs = torch.sigmoid(logits)
        self.test_meter.update(probs.detach(), y.detach())
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Save predictions (PNG) + gather samples for PR/calibration + qualitative grids.
        pred_dir = self._test_pred_dir
        max_pixels = int(getattr(getattr(self.cfg, "artifacts", {}), "sample_pixels", 200_000))
        pixels_per_image = int(getattr(getattr(self.cfg, "artifacts", {}), "sample_pixels_per_image", 5_000))
        max_qual = int(getattr(getattr(self.cfg, "artifacts", {}), "max_qual", 8))
        save_preds = bool(getattr(getattr(self.cfg, "artifacts", {}), "save_test_preds", True))

        probs_cpu = probs.detach().cpu().float().numpy()
        y_cpu = y.detach().cpu().float().numpy().astype(np.uint8)
        x_cpu = x.detach().cpu().float().numpy()
        pred_bin = (probs_cpu > 0.5).astype(np.uint8)

        if save_preds and pred_dir is not None:
            from PIL import Image

            for b in range(pred_bin.shape[0]):
                idx = self._test_image_counter + b
                Image.fromarray(pred_bin[b, 0] * 255).save(
                    pred_dir / f"{idx:06d}.png",
                    format="PNG",
                    compress_level=9,
                    optimize=True,
                )

        # Per-image metrics (for best/worst selection and optional CSV)
        for b in range(pred_bin.shape[0]):
            pr = pred_bin[b, 0]
            gt = y_cpu[b]
            tp = float((pr * gt).sum())
            fp = float((pr * (1 - gt)).sum())
            fn = float(((1 - pr) * gt).sum())
            eps = 1e-6
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            dice = 2 * tp / (2 * tp + fp + fn + eps)
            iou = tp / (tp + fp + fn + eps)

            self._test_image_metrics.append(
                {
                    "index": float(self._test_image_counter + b),
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                }
            )

            # Boundary F-score (tolerance=1 by default)
            try:
                bf = boundary_f_score(pr, gt, tol=int(getattr(getattr(self.cfg, "artifacts", {}), "bf_tol", 1)))
                self._test_bf_sum += float(bf)
                self._test_bf_n += 1
                self._test_image_metrics[-1]["bfscore"] = float(bf)
            except Exception:
                pass

            # clDice only for retina datasets by default
            if getattr(self.cfg.dataset, "type", "") == "hf_retina":
                try:
                    cld = float(cldice_score(pr[None, None, ...], gt[None, ...]))
                    self._test_cldice_sum += cld
                    self._test_cldice_n += 1
                    self._test_image_metrics[-1]["cldice"] = cld
                except Exception:
                    pass

            # Keep best/worst qualitative samples
            sample = (dice, x_cpu[b], gt, pr)
            self._best_samples.append(sample)
            self._best_samples = sorted(self._best_samples, key=lambda t: t[0], reverse=True)[:max_qual]
            self._worst_samples.append(sample)
            self._worst_samples = sorted(self._worst_samples, key=lambda t: t[0])[:max_qual]

            # Pixel sampling for PR/ECE/reliability
            if self._test_pixel_count < max_pixels:
                flat_p = probs_cpu[b, 0].reshape(-1)
                flat_t = gt.reshape(-1).astype(np.uint8)
                remaining = max_pixels - self._test_pixel_count
                k = int(min(pixels_per_image, remaining, flat_p.size))
                if k > 0:
                    idxs = self._rng.integers(0, flat_p.size, size=k, endpoint=False)
                    self._test_pixel_probs.append(flat_p[idxs])
                    self._test_pixel_targets.append(flat_t[idxs])
                    self._test_pixel_count += k

        self._test_image_counter += int(pred_bin.shape[0])

    def on_train_epoch_end(self) -> None:
        self.train_meter.log(self, prefix="train")
        self.train_meter.reset()

    def on_validation_epoch_end(self) -> None:
        self.val_meter.log(self, prefix="val")
        self.val_meter.reset()

    def on_test_epoch_end(self) -> None:
        self.test_meter.log(self, prefix="test")
        self.test_meter.reset()
        # Aggregate BFScore/clDice means
        if self._test_bf_n > 0:
            self.log("test/bfscore", float(self._test_bf_sum / max(self._test_bf_n, 1)))
        if self._test_cldice_n > 0:
            self.log("test/cldice", float(self._test_cldice_sum / max(self._test_cldice_n, 1)))

        # Calibration + PR curves from sampled pixels
        if self._test_pixel_probs:
            probs_np = np.concatenate(self._test_pixel_probs, axis=0)
            targets_np = np.concatenate(self._test_pixel_targets, axis=0)
            try:
                ece = expected_calibration_error(
                    torch.from_numpy(probs_np), torch.from_numpy(targets_np)
                )
                self.log("test/ece", float(ece))
            except Exception:
                pass

            save_figures = bool(getattr(getattr(self.cfg, "artifacts", {}), "save_figures", True))
            if save_figures and self._test_fig_dir is not None:
                try:
                    from seglab.utils.viz import plot_pr_curve, plot_reliability_diagram, save_qualitative_grid

                    plot_pr_curve(probs_np, targets_np, self._test_fig_dir / "pr_curve.png")
                    plot_reliability_diagram(
                        probs_np, targets_np, self._test_fig_dir / "reliability.png", n_bins=15
                    )

                    if self._best_samples:
                        imgs = np.stack([s[1] for s in self._best_samples], axis=0)
                        gts = np.stack([s[2] for s in self._best_samples], axis=0)
                        prs = np.stack([s[3] for s in self._best_samples], axis=0)
                        save_qualitative_grid(imgs, gts, prs, self._test_fig_dir / "qual_best.png")
                    if self._worst_samples:
                        imgs = np.stack([s[1] for s in self._worst_samples], axis=0)
                        gts = np.stack([s[2] for s in self._worst_samples], axis=0)
                        prs = np.stack([s[3] for s in self._worst_samples], axis=0)
                        save_qualitative_grid(imgs, gts, prs, self._test_fig_dir / "qual_worst.png")
                except Exception as e:
                    self.print(f"[warn] failed to create test figures: {e}")

        # Save per-image test metrics CSV (if possible)
        if self._test_run_dir is not None and self._test_image_metrics:
            try:
                import pandas as pd

                pd.DataFrame(self._test_image_metrics).to_csv(
                    self._test_run_dir / "test_metrics_per_image.csv", index=False
                )
            except Exception:
                pass

        # Reset artifact buffers
        self._test_pixel_probs.clear()
        self._test_pixel_targets.clear()
        self._test_pixel_count = 0
        self._test_bf_sum = 0.0
        self._test_bf_n = 0
        self._test_cldice_sum = 0.0
        self._test_cldice_n = 0
        self._test_image_metrics.clear()
        self._best_samples.clear()
        self._worst_samples.clear()

    def on_test_start(self) -> None:
        run_dir = Path(self.trainer.default_root_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._test_run_dir = run_dir

        # Save per-image binary predictions in the run directory.
        pred_dir = run_dir / "preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        self._test_pred_dir = pred_dir

        # Save figures under global figures/ with the same run id.
        fig_root = Path(getattr(getattr(self.cfg, "paths", {}), "figures_dir", "figures"))
        # Robust run id (supports optional nested tags).
        try:
            seed_root = (
                Path(getattr(getattr(self.cfg, "paths", {}), "results_dir", "results"))
                / str(getattr(self.cfg, "experiment", "default"))
                / str(getattr(self.cfg.dataset, "name", "dataset"))
                / str(getattr(self.cfg.model, "name", "model"))
                / f"seed{int(getattr(self.cfg, 'seed', 0))}"
            )
            run_id = "-".join(run_dir.relative_to(seed_root).parts)
        except Exception:
            run_id = run_dir.name
        fig_dir = (
            fig_root
            / str(getattr(self.cfg, "experiment", "default"))
            / str(getattr(self.cfg.dataset, "name", "dataset"))
            / str(getattr(self.cfg.model, "name", "model"))
            / f"seed{int(getattr(self.cfg, 'seed', 0))}"
            / run_id
        )
        fig_dir.mkdir(parents=True, exist_ok=True)
        self._test_fig_dir = fig_dir

        self._test_image_counter = 0

    def configure_optimizers(self):
        opt_name = self.cfg.optimizer.get("name", "adamw").lower()
        lr = self.cfg.optimizer.get("lr", 1e-4)
        wd = self.cfg.optimizer.get("weight_decay", 1e-4)
        if opt_name == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        sched_cfg = self.cfg.get("scheduler", {})
        sched_name = sched_cfg.get("name", "none")

        if sched_name == "warmup_flat":
            # Linear warmup from warmup_start_lr → lr over warmup_epochs,
            # then constant lr for remaining epochs.
            warmup_epochs = int(sched_cfg.get("warmup_epochs", 4))
            warmup_start = float(sched_cfg.get("warmup_start_lr", 1e-6))
            # LinearLR multiplies the base lr by start_factor → end_factor
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=warmup_start / lr,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            flat = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=self.cfg.trainer.max_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, flat], milestones=[warmup_epochs]
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        if sched_name == "warmup_cosine":
            # Linear warmup then cosine annealing.
            warmup_epochs = int(sched_cfg.get("warmup_epochs", 4))
            warmup_start = float(sched_cfg.get("warmup_start_lr", 1e-6))
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=warmup_start / lr,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.cfg.trainer.max_epochs - warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": scheduler}

        return opt


# Import model builders to register them.
from .unet_smp import build_unet  # noqa: E402,F401
from .deeplabv3p_smp import build_deeplabv3p  # noqa: E402,F401
from .segformer_hf import build_segformer  # noqa: E402,F401
from .mask2former_d2 import build_mask2former  # noqa: E402,F401
from .sam_peft.model import build_sam_topolora  # noqa: E402,F401
