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


class JunctionHeatmapHead(nn.Module):
    """Lightweight head that predicts junction/endpoint heatmaps from SAM image embeddings."""

    def __init__(self, in_channels: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, image_embeddings: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            image_embeddings: (B, C, H_emb, W_emb) from SAM encoder
            output_size: (H, W) target spatial size

        Returns:
            Heatmap logits (B, 1, H, W)
        """
        x = self.act(self.bn1(self.conv1(image_embeddings)))
        x = self.conv2(x)
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

        # Optional junction heatmap head
        self.junction_head: Optional[JunctionHeatmapHead] = None
        junc_cfg = cfg.model.get("junction_head", {})
        if junc_cfg.get("enabled", False):
            self.junction_head = JunctionHeatmapHead(
                in_channels=256,
                hidden_dim=junc_cfg.get("hidden_dim", 128),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = F.interpolate(low_res_masks, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.junction_head is not None:
            junction_heatmap = self.junction_head(image_embeddings, output_size=x.shape[-2:])
            return masks, junction_heatmap

        return masks


@register_model("sam_topolora")
def build_sam_topolora(cfg: Any) -> LitBinarySeg:
    net = SAMPEFTNet(cfg)
    return LitBinarySeg(net, cfg)
