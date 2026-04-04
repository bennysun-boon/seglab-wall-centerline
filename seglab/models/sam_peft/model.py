"""SAM-based PEFT model with optional LoRA and topology regularization."""

from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from seglab.models import LitBinarySeg
from seglab.models.sam_peft.adapters import ConvAdapter
from seglab.models.sam_peft.lora import inject_lora
from seglab.models.sam_peft.sam_loader import load_sam
from seglab.utils.registry import register_model


class FrameFieldHead(nn.Module):
    """Frame field head for polygon extraction (Girard et al., CVPR 2021).

    Predicts two outputs from the mask decoder's 32ch upscaled features:
      edge: (B, 1, H, W) — binary boundary probability (sigmoid)
      ff:   (B, 4, H, W) — frame field coefficients [Re(c0), Im(c0), Re(c2), Im(c2)] (tanh)

    The frame field f(z; c0, c2) = z^4 + c2*z^2 + c0 encodes two orthogonal
    directions per pixel. At straight edges: u ∥ edge, v ⊥ edge. At corners:
    u and v diverge to capture both tangent directions simultaneously.

    Edge branch is computed first and concatenated with the embedding before
    the frame field branch — the field is conditioned on edge predictions.

    Replaces both BorderEDTHead and PavingPolyHead in the pipeline.
    in_channels = 33 (32 embedding + 1 seg_prob).
    """

    def __init__(self, in_channels: int = 33, hidden_dim: int = 256):
        super().__init__()

        # Edge branch: operates at 256×256 input resolution, outputs 1ch edge map
        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )
        # FF branch: conditioned on features + edge_prob at 256×256
        self.ff_branch = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_dim, 3, padding=1),  # +1 for edge
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 4, 1),  # Re(c0), Im(c0), Re(c2), Im(c2)
        )

    def forward(self, features: torch.Tensor,
                output_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        # Both branches operate at 256×256 (feature resolution), then upsample
        edge_logits = self.edge_branch(features)          # (B, 1, 256, 256)
        edge_prob   = torch.sigmoid(edge_logits)

        ff_input    = torch.cat([features, edge_prob], dim=1)   # (B, 34, 256, 256)
        ff_coeff    = 2.0 * torch.tanh(self.ff_branch(ff_input))  # (B, 4, 256, 256) in [-2, 2] (lydorn spec)

        edge_out = F.interpolate(edge_prob, size=output_size, mode="bilinear", align_corners=False)
        ff_out   = F.interpolate(ff_coeff,  size=output_size, mode="bilinear", align_corners=False)
        return {"edge": edge_out, "ff": ff_out}


class PavingPolyHead(nn.Module):
    """Predict polygon vertex heatmap (vmap) and offset vectors (voff).

    Outputs:
        vmap:  (B, 1, H, W) logits — sigmoid → vertex Gaussian heatmap
        voff:  (B, 2, H, W) logits — sigmoid*2-1 → (dx/r, dy/r) offset to
               nearest vertex, normalised by supervision radius r

    Uses separate Upsample+Conv upscaling branches for vmap and voff to avoid
    feature interference. Avoids ConvTranspose2d checkerboard artefacts.
    Same 32ch @ 256×256 feature tap as BorderEDTHead.
    """

    def __init__(self, in_channels: int = 32, hidden_dim: int = 128):
        super().__init__()
        # in_channels = 32 (embedding) + 1 (seg_prob) + 1 (edt_pred) = 34 when cascaded

        def _up_branch(out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.GELU(),
                nn.Conv2d(hidden_dim // 2, out_ch, 1),
            )

        self.vmap_up = _up_branch(1)
        self.voff_up = _up_branch(2)

    def forward(self, upscaled_embedding: torch.Tensor,
                output_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        vmap = F.interpolate(
            self.vmap_up(upscaled_embedding), size=output_size,
            mode="bilinear", align_corners=False,
        )
        voff = F.interpolate(
            self.voff_up(upscaled_embedding), size=output_size,
            mode="bilinear", align_corners=False,
        )
        return {"vmap": vmap, "voff": voff}


class BorderEDTHead(nn.Module):
    """Predict unsigned border EDT from mask decoder's upscaled features.

    Optionally conditioned on seg_prob (pass as extra channel) so the head
    knows which regions are paving before predicting boundary distances.

    Outputs the distance of every pixel to the nearest paving boundary,
    normalised to [0, 1].  Boundary pixels → 0; deep interior / far background
    → 1 (clamped at the truncation radius set during preprocessing).
    """

    def __init__(self, in_channels: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (edt_full_res, edt_256) — edt_256 for downstream cascade."""
        edt_256 = torch.sigmoid(self.net(features))   # (B, 1, 256, 256)
        edt_full = F.interpolate(edt_256, size=output_size, mode="bilinear", align_corners=False)
        return edt_full, edt_256


class CenterlineDistHead(nn.Module):
    """Predict distance transform regression from mask decoder's upscaled features.

    The ridge (local maxima) of the predicted distance map IS the centerline.
    Peak values encode wall half-width. Same feature tap point as JunctionHeatmapHead
    (32ch @ 256×256 from mask decoder output_upscaling).
    """

    def __init__(self, in_channels: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, upscaled_embedding: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            upscaled_embedding: (B, 32, 256, 256) from mask decoder's output_upscaling
            output_size: (H, W) target spatial size

        Returns:
            Distance transform prediction (B, 1, H, W), sigmoid-activated to [0, 1]
        """
        x = self.net(upscaled_embedding)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class JunctionHeatmapHead(nn.Module):
    """Predict junction/endpoint heatmaps from mask decoder's upscaled features.

    Taps into the 256×256 upscaled embeddings (32ch) from SAM's mask decoder
    instead of the raw 64×64 encoder output, giving 4× better spatial resolution
    for precise keypoint localization.

    Uses a deeper 4-conv architecture to compensate for the lower channel count
    of the upscaled features (32ch vs 256ch from encoder).
    """

    def __init__(self, in_channels: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, upscaled_embedding: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            upscaled_embedding: (B, 32, 256, 256) from mask decoder's output_upscaling
            output_size: (H, W) target spatial size

        Returns:
            Heatmap logits (B, 1, H, W)
        """
        x = self.net(upscaled_embedding)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class SAMPEFTNet(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.sam = load_sam(cfg.model.sam_type, cfg.model.sam_checkpoint)

        # Freeze everything
        for p in self.sam.parameters():
            p.requires_grad = False

        # LoRA injection into image encoder
        if cfg.model.lora.enabled:
            inject_lora(
                self.sam.image_encoder,
                target_keywords=cfg.model.lora.target_keywords,
                r=cfg.model.lora.r,
                alpha=cfg.model.lora.alpha,
                dropout=cfg.model.lora.dropout,
            )

        # Conv adapter on image embeddings (SAM vit-b outputs 256 channels)
        self.adapter: Optional[nn.Module] = None
        if cfg.model.adapter.enabled:
            self.adapter = ConvAdapter(
                channels=256,
                hidden_dim=cfg.model.adapter.hidden_dim,
                kernel_size=cfg.model.adapter.kernel_size,
            )

        if cfg.model.get("train_mask_decoder", True):
            for p in self.sam.mask_decoder.parameters():
                p.requires_grad = True

        # Optional junction heatmap head (taps into mask decoder's upscaled features)
        self.junction_head: Optional[JunctionHeatmapHead] = None
        junc_cfg = cfg.model.get("junction_head", {})
        if junc_cfg.get("enabled", False):
            self.junction_head = JunctionHeatmapHead(
                in_channels=32,  # mask decoder output_upscaling produces 32ch at 256×256
                hidden_dim=junc_cfg.get("hidden_dim", 128),
            )

        # Optional centerline distance transform head (same feature tap point)
        self.centerline_head: Optional[CenterlineDistHead] = None
        cl_cfg = cfg.model.get("centerline_head", {})
        if cl_cfg.get("enabled", False):
            self.centerline_head = CenterlineDistHead(
                in_channels=32,
                hidden_dim=cl_cfg.get("hidden_dim", 256),
            )

        # Optional border EDT head — conditioned on seg_prob (+1ch)
        self.border_edt_head: Optional[BorderEDTHead] = None
        edt_cfg = cfg.model.get("border_edt_head", {})
        has_edt = edt_cfg.get("enabled", False)
        if has_edt:
            self.border_edt_head = BorderEDTHead(
                in_channels=33,   # 32 embedding + 1 seg_prob
                hidden_dim=edt_cfg.get("hidden_dim", 256),
            )

        # Optional polygon vertex head — conditioned on seg_prob + edt_pred (+2ch)
        self.poly_head: Optional[PavingPolyHead] = None
        poly_cfg = cfg.model.get("poly_head", {})
        if poly_cfg.get("enabled", False):
            poly_in = 32 + (1 if has_edt else 0) + 1   # +1 seg, +1 edt (if present)
            self.poly_head = PavingPolyHead(
                in_channels=poly_in,
                hidden_dim=poly_cfg.get("hidden_dim", 128),
            )

        # Optional frame field head — replaces BorderEDTHead + PavingPolyHead
        # Conditioned on seg_prob (+1ch) → in_channels = 33
        self.frame_field_head: Optional[FrameFieldHead] = None
        ff_cfg = cfg.model.get("frame_field_head", {})
        if ff_cfg.get("enabled", False):
            self.frame_field_head = FrameFieldHead(
                in_channels=33,
                hidden_dim=ff_cfg.get("hidden_dim", 256),
            )

        self._compile_encoder = getattr(cfg.model, "compile_encoder", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # SAM expects inputs normalized with its own pixel_mean/std and padded to img_size.
        # Our datasets use ImageNet normalization; invert it back to [0, 255] RGB first.
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_01 = (x * imagenet_std) + imagenet_mean
        x_255 = (x_01.clamp(0, 1) * 255.0).to(dtype=torch.float32)

        img_size = int(getattr(self.sam.image_encoder, "img_size", 1024))
        if x_255.shape[-1] != img_size or x_255.shape[-2] != img_size:
            x_255 = F.interpolate(x_255, size=(img_size, img_size), mode="bilinear", align_corners=False)

        x_sam = self.sam.preprocess(x_255)
        encoder_frozen = not any(p.requires_grad for p in self.sam.image_encoder.parameters())
        _enc_ctx = torch.no_grad() if encoder_frozen else contextlib.nullcontext()
        with _enc_ctx:
            image_embeddings = self.sam.image_encoder(x_sam)
        if self.adapter is not None:
            image_embeddings = self.adapter(image_embeddings)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )

        has_aux = (self.junction_head is not None or self.centerline_head is not None
                   or self.border_edt_head is not None or self.poly_head is not None
                   or self.frame_field_head is not None)
        if has_aux:
            # Run mask decoder internals manually to access upscaled embeddings
            low_res_masks, upscaled_embedding = self._decode_with_upscaled(
                image_embeddings, sparse_embeddings, dense_embeddings
            )
            masks = F.interpolate(low_res_masks, size=x.shape[-2:], mode="bilinear", align_corners=False)

            # seg_prob at 256×256 — used to condition EDT and poly heads
            seg_prob_256 = torch.sigmoid(low_res_masks)  # (B, 1, 256, 256)

            results = [masks]
            if self.junction_head is not None:
                results.append(self.junction_head(upscaled_embedding, output_size=x.shape[-2:]))
            if self.centerline_head is not None:
                results.append(self.centerline_head(upscaled_embedding, output_size=x.shape[-2:]))
            if self.border_edt_head is not None:
                edt_features = torch.cat([upscaled_embedding, seg_prob_256], dim=1)  # 33ch
                edt_full, edt_256 = self.border_edt_head(edt_features, output_size=x.shape[-2:])
                results.append(edt_full)
            else:
                edt_256 = None
            if self.poly_head is not None:
                poly_parts = [upscaled_embedding, seg_prob_256]
                if edt_256 is not None:
                    poly_parts.append(edt_256)
                poly_features = torch.cat(poly_parts, dim=1)  # 33 or 34ch
                results.append(self.poly_head(poly_features, output_size=x.shape[-2:]))
            if self.frame_field_head is not None:
                ff_features = torch.cat([upscaled_embedding, seg_prob_256], dim=1)  # 33ch
                results.append(self.frame_field_head(ff_features, output_size=x.shape[-2:]))
            return tuple(results)

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = F.interpolate(low_res_masks, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return masks

    def _decode_with_upscaled(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run mask decoder and return both masks and upscaled embeddings (B, 32, 256, 256)."""
        decoder = self.sam.mask_decoder
        image_pe = self.sam.prompt_encoder.get_dense_pe()

        # Replicate mask decoder's predict_masks logic
        output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + decoder.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = decoder.output_upscaling(src)  # (B, 32, 256, 256)

        hyper_in_list = []
        for i in range(decoder.num_mask_tokens):
            hyper_in_list.append(decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Select single mask output (multimask_output=False)
        masks = masks[:, 0:1, :, :]

        return masks, upscaled_embedding


@register_model("sam_topolora")
def build_sam_topolora(cfg: Any) -> LitBinarySeg:
    net = SAMPEFTNet(cfg)
    return LitBinarySeg(net, cfg)
