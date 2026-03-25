"""Paving region dataset loader for tiled civil-plan data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from seglab.data.transforms import build_transforms
from seglab.data.splits import make_split_indices_by_group
from seglab.utils.registry import register_dataset


class PavingDataset(Dataset):
    """Dataset for paving region segmentation from tiled civil-plan PDFs.

    Loads three targets per tile:
      - mask       : binary filled paving regions (seg head target)
      - border_edt : unsigned border EDT normalised to [0, 1] (BorderEDTHead target)
    """

    def __init__(
        self,
        root: str | Path,
        split_indices: list[int],
        transform: Optional[Any] = None,
        border_edt: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.border_edt = border_edt

        metadata_path = self.root / "tile_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                "Run preprocessing/create_tiled_dataset_paving.py first."
            )

        with open(metadata_path) as f:
            metadata = json.load(f)

        all_tiles = metadata["tiles"]
        self.tiles = [all_tiles[i] for i in split_indices if i < len(all_tiles)]

        self.has_border_edt = (
            border_edt and (self.root / "distance_transforms").is_dir()
        )

        print(
            f"PavingDataset: {len(self.tiles)} tiles"
            + (", with border EDT" if self.has_border_edt else "")
        )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tile_id = self.tiles[idx]["tile_id"]

        image = np.array(Image.open(self.root / "images" / f"{tile_id}.png").convert("RGB"))
        mask  = np.array(Image.open(self.root / "masks"  / f"{tile_id}.png").convert("L"))
        mask  = (mask > 127).astype(np.uint8)

        border_edt = None
        if self.has_border_edt:
            edt_path = self.root / "distance_transforms" / f"{tile_id}.png"
            if edt_path.exists():
                try:
                    border_edt = (
                        np.array(Image.open(edt_path).convert("L")).astype(np.float32) / 255.0
                    )
                except Exception:
                    pass  # fall through to zeros below
            if border_edt is None:
                # Missing or corrupted EDT — zeros in same dtype/shape as a real EDT
                # (float32 [0,1], same H×W as mask; band mask suppresses these in loss)
                border_edt = np.zeros_like(mask, dtype=np.float32)

        if self.transform is not None:
            kwargs = {"image": image, "mask": mask}
            if border_edt is not None:
                kwargs["border_edt"] = border_edt
            transformed = self.transform(**kwargs)
            image = transformed["image"]
            mask  = transformed["mask"]
            if "border_edt" in transformed:
                border_edt = transformed["border_edt"]

        result: Dict[str, Any] = {"image": image, "mask": mask}
        if border_edt is not None:
            result["border_edt"] = border_edt
        elif self.has_border_edt:
            # Always include key when head is active — prevents collate KeyError
            result["border_edt"] = np.zeros_like(mask, dtype=np.float32)
        return result


@register_dataset("paving")
class PavingDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for paving region dataset.

    Splits are done by source PDF (group-based) to prevent tile overlap
    leakage between train/val/test.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.root        = cfg.dataset.root
        self.size        = cfg.dataset.size
        self.batch_size  = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.border_edt  = bool(cfg.dataset.get("border_edt", True))

        metadata_path = Path(self.root) / "tile_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        self.all_tiles = metadata["tiles"]

        cache_dir = Path(cfg.paths.cache_dir) / "splits"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.splits = make_split_indices_by_group(
            self.all_tiles,
            seed=cfg.seed,
            group_key="source_pdf",
            val_ratio=cfg.dataset.get("val_ratio", 0.15),
            test_ratio=cfg.dataset.get("test_ratio", 0.15),
            cache_path=cache_dir / f"paving_seed{cfg.seed}.json",
        )

        meta = self.splits.get("_metadata", {})
        if meta:
            print(
                f"Paving splits (by PDF): "
                f"train={meta['train_tiles']} tiles / {meta['train_groups']} PDFs, "
                f"val={meta['val_tiles']} / {meta['val_groups']}, "
                f"test={meta['test_tiles']} / {meta['test_groups']}"
            )

        self.train_transform = build_transforms(
            size=self.size,
            train=True,
            aug=cfg.dataset.get("aug", {}),
            border_edt=self.border_edt,
        )
        self.eval_transform = build_transforms(
            size=self.size,
            train=False,
            border_edt=self.border_edt,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = PavingDataset(
                root=self.root,
                split_indices=self.splits["train"],
                transform=self.train_transform,
                border_edt=self.border_edt,
            )
            self.val_dataset = PavingDataset(
                root=self.root,
                split_indices=self.splits["val"],
                transform=self.eval_transform,
                border_edt=self.border_edt,
            )
        if stage in ("test", None):
            self.test_dataset = PavingDataset(
                root=self.root,
                split_indices=self.splits["test"],
                transform=self.eval_transform,
                border_edt=self.border_edt,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
