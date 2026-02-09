"""Wall centerline dataset loader for tiled floor plan data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import random

import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from seglab.data.transforms import build_transforms
from seglab.data.splits import make_split_indices
from seglab.utils.registry import register_dataset
from torch.utils.data import Subset


class WallCenterlineDataset(Dataset):
    """Dataset for wall centerline segmentation from tiled floor plans."""

    def __init__(
        self,
        root: str | Path,
        split_indices: list[int],
        transform: Optional[Any] = None,
    ):
        """
        Args:
            root: Path to processed data directory
            split_indices: List of tile indices for this split
            transform: Albumentations transform
        """
        self.root = Path(root)
        self.transform = transform

        # Load tile metadata
        metadata_path = self.root / "tile_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                "Run preprocessing/create_tiled_dataset.py first."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        all_tiles = metadata["tiles"]
        self.tiles = [all_tiles[i] for i in split_indices if i < len(all_tiles)]

        print(f"WallCenterlineDataset: {len(self.tiles)} tiles")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tile_meta = self.tiles[idx]
        tile_id = tile_meta["tile_id"]

        # Load image and mask
        image_path = self.root / "images" / f"{tile_id}.png"
        mask_path = self.root / "masks" / f"{tile_id}.png"

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarize mask (in case of compression artifacts)
        mask = (mask > 127).astype(np.uint8)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {
            "image": image,
            "mask": mask,
        }


@register_dataset("wall_centerline")
class WallCenterlineDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for wall centerline dataset."""

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.root = cfg.dataset.root
        self.size = cfg.dataset.size
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers

        # Load metadata to get total count
        metadata_path = Path(self.root) / "tile_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.total_tiles = len(metadata["tiles"])

        # Create splits
        cache_dir = Path(cfg.paths.cache_dir) / "splits"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.splits = make_split_indices(
            self.total_tiles,
            cfg.seed,
            val_ratio=cfg.dataset.get("val_ratio", 0.15),
            cache_path=cache_dir / f"wall_centerline_seed{cfg.seed}.json",
        )

        self.train_transform = build_transforms(
            size=self.size,
            train=True,
            sar=cfg.dataset.get("sar", False),
            aug=cfg.dataset.get("aug", {}),
        )
        self.test_transform = build_transforms(
            size=self.size,
            train=False,
        )

    def setup(self, stage: Optional[str] = None):
        """Create train/val/test datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = WallCenterlineDataset(
                root=self.root,
                split_indices=self.splits["train"],
                transform=self.train_transform,
            )
            self.val_dataset = WallCenterlineDataset(
                root=self.root,
                split_indices=self.splits["val"],
                transform=self.test_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WallCenterlineDataset(
                root=self.root,
                split_indices=self.splits["test"],
                transform=self.test_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
