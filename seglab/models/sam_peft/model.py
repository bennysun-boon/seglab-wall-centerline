"""SAM-based PEFT model with optional LoRA and topology regularization."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from seglab.models import LitBinarySeg
from seglab.models.sam_peft.adapters import ConvAdapter
from seglab.models.sam_peft.lora import inject_lora
from seglab.models.sam_peft.sam_loader import load_sam
from seglab.utils.registry import register_model


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
        image_embeddings = self.sam.image_encoder(x_sam)
        if self.adapter is not None:
            image_embeddings = self.adapter(image_embeddings)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        has_aux = self.junction_head is not None or self.centerline_head is not None
        if has_aux:
            # Run mask decoder internals manually to access upscaled embeddings
            low_res_masks, upscaled_embedding = self._decode_with_upscaled(
                image_embeddings, sparse_embeddings, dense_embeddings
            )
            masks = F.interpolate(low_res_masks, size=x.shape[-2:], mode="bilinear", align_corners=False)

            results = [masks]
            if self.junction_head is not None:
                results.append(self.junction_head(upscaled_embedding, output_size=x.shape[-2:]))
            if self.centerline_head is not None:
                results.append(self.centerline_head(upscaled_embedding, output_size=x.shape[-2:]))
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
