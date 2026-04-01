"""Paving region dataset loader for tiled civil-plan data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from seglab.data.transforms import build_transforms
from seglab.data.splits import make_split_indices_by_group
from seglab.utils.registry import register_dataset


# ── Frame field target generation (on-the-fly, zero extra disk) ──────────────

def _build_edge_and_theta(
    polygons: list[np.ndarray], H: int, W: int, thickness: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise polygon edges and compute per-pixel tangent angles.

    Args:
        polygons:  list of (N,2) float32 arrays in tile pixel coords [x,y]
        H, W:      tile size
        thickness: rasterisation thickness in pixels

    Returns:
        edge:  (H, W) float32 binary map
        theta: (H, W) float32 tangent angle in [0, π) — defined where edge>0
    """
    edge = np.zeros((H, W), np.float32)
    sin2 = np.zeros((H, W), np.float64)
    cos2 = np.zeros((H, W), np.float64)

    for poly in polygons:
        n = len(poly)
        if n < 2:
            continue
        for i in range(n):
            p0 = poly[i]
            p1 = poly[(i + 1) % n]
            dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
            angle = math.atan2(dy, dx) % math.pi
            scratch = np.zeros((H, W), np.uint8)
            cv2.line(scratch,
                     (int(round(p0[0])), int(round(p0[1]))),
                     (int(round(p1[0])), int(round(p1[1]))),
                     color=1, thickness=thickness)
            mask = scratch > 0
            edge[mask] = 1.0
            sin2[mask] += math.sin(2.0 * angle)
            cos2[mask] += math.cos(2.0 * angle)

    theta = np.zeros((H, W), np.float32)
    has_edge = edge > 0
    if has_edge.any():
        theta[has_edge] = (np.arctan2(sin2[has_edge], cos2[has_edge]) / 2.0) % math.pi

    return edge, theta


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
        frame_field: bool = False,
        annotations_path: Optional[str | Path] = None,
        ff_thickness: int = 2,
    ):
        self.root = Path(root)
        self.transform = transform
        self.border_edt = border_edt
        self.frame_field = frame_field
        self._ff_thickness = ff_thickness

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
        # Read voff supervision radius from metadata so decoding matches preprocessing
        self._voff_radius = float(
            metadata.get("processing_params", {}).get("voff_radius", 12.0)
        )

        self.has_border_edt = (
            border_edt and (self.root / "distance_transforms").is_dir()
        )
        self.has_poly = (self.root / "vmap").is_dir() and not self.frame_field

        # Pre-scan file existence once at init to avoid per-sample stat calls in __getitem__
        self._edt_missing: set = set()
        self._poly_npy: dict = {}   # subdir -> set of stems with .npy
        self._poly_png: dict = {}   # subdir -> set of stems with .png only (fallback)
        if self.has_border_edt:
            edt_dir = self.root / "distance_transforms"
            edt_existing = {p.stem for p in edt_dir.glob("*.png")}
            self._edt_missing = {t["tile_id"] for t in self.tiles if t["tile_id"] not in edt_existing}
        if self.has_poly:
            # Per-channel scan: dict[subdir] -> set of stems with .npy / .png
            self._poly_npy: dict[str, set[str]] = {}
            self._poly_png: dict[str, set[str]] = {}
            for subdir in ("vmap", "voff_x", "voff_y"):
                d = self.root / subdir
                if d.is_dir():
                    npy_stems = {p.stem for p in d.glob("*.npy")}
                    png_stems = {p.stem for p in d.glob("*.png")}
                    self._poly_npy[subdir] = npy_stems
                    self._poly_png[subdir] = png_stems - npy_stems

        # Load polygon annotations for on-the-fly frame field target generation
        self._img_anns: dict = {}
        if self.frame_field and annotations_path is not None:
            with open(annotations_path) as f:
                ann_data = json.load(f)
            for ann in ann_data["annotations"]:
                self._img_anns.setdefault(ann["image_id"], []).append(ann)

        print(
            f"PavingDataset: {len(self.tiles)} tiles"
            + (", with border EDT" if self.has_border_edt else "")
            + (", with frame field" if self.frame_field else "")
        )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tile_id = self.tiles[idx]["tile_id"]

        # cv2 is ~3x faster than PIL for large PNGs
        img_bgr = cv2.imread(str(self.root / "images" / f"{tile_id}.png"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imread(str(self.root / "masks" / f"{tile_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask_raw > 127).astype(np.uint8)

        border_edt = None
        if self.has_border_edt:
            if tile_id not in self._edt_missing:
                try:
                    edt_raw = cv2.imread(
                        str(self.root / "distance_transforms" / f"{tile_id}.png"),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    border_edt = edt_raw.astype(np.float32) / 255.0
                except Exception:
                    pass
            if border_edt is None:
                border_edt = np.zeros_like(mask, dtype=np.float32)

        # Load vmap/voff targets — use pre-scanned sets, no per-sample stat calls
        poly_targets: Dict[str, Any] = {}
        if self.has_poly:
            for key, subdir in [("vmap", "vmap"), ("voff_x", "voff_x"), ("voff_y", "voff_y")]:
                npy_set = self._poly_npy.get(subdir, set())
                png_set = self._poly_png.get(subdir, set())
                if tile_id in npy_set:
                    try:
                        poly_targets[key] = np.load(
                            self.root / subdir / f"{tile_id}.npy"
                        ).astype(np.float32)
                    except Exception:
                        pass
                elif tile_id in png_set:
                    try:
                        if key in ("voff_x", "voff_y"):
                            arr16 = cv2.imread(
                                str(self.root / subdir / f"{tile_id}.png"),
                                cv2.IMREAD_UNCHANGED,
                            ).astype(np.float32)
                            poly_targets[key] = (arr16 - 32768.0) / self._voff_radius
                        else:
                            arr = cv2.imread(
                                str(self.root / subdir / f"{tile_id}.png"),
                                cv2.IMREAD_GRAYSCALE,
                            )
                            poly_targets[key] = arr.astype(np.float32) / 255.0
                    except Exception:
                        pass

        # Frame field targets — built on-the-fly from polygon annotations
        ff_targets: Dict[str, Any] = {}
        if self.frame_field and self._img_anns:
            tile_meta = self.tiles[idx]
            img_id = tile_meta["source_image_id"]
            tile_x = tile_meta["tile_x"]
            tile_y = tile_meta["tile_y"]
            img_w  = tile_meta.get("render_width",  tile_meta.get("img_w", 3024))
            img_h  = tile_meta.get("render_height", tile_meta.get("img_h", 2160))
            H, W   = mask.shape[:2]
            polygons = []
            for ann in self._img_anns.get(img_id, []):
                mtype = ann.get("measurement_type", "").upper()
                seg   = ann.get("segmentation", [[]])[0]
                if mtype not in ("POLYGON", "POLYLINE") or len(seg) < 2:
                    continue
                verts = np.array([[p["x"] * img_w, p["y"] * img_h] for p in seg],
                                 dtype=np.float32)
                if not np.isfinite(verts).all():
                    continue
                local = verts.copy()
                local[:, 0] -= tile_x
                local[:, 1] -= tile_y
                in_tile = ((local[:, 0] >= 0) & (local[:, 0] < W) &
                           (local[:, 1] >= 0) & (local[:, 1] < H))
                if in_tile.any():
                    local[:, 0] = np.clip(local[:, 0], 0, W - 1)
                    local[:, 1] = np.clip(local[:, 1], 0, H - 1)
                    polygons.append(local)
            edge, theta = _build_edge_and_theta(polygons, H, W, self._ff_thickness)
            ff_targets["edge"] = edge
            ff_targets["theta"] = theta

        if self.transform is not None:
            kwargs = {"image": image, "mask": mask}
            if border_edt is not None:
                kwargs["border_edt"] = border_edt
            kwargs.update(poly_targets)
            kwargs.update(ff_targets)
            transformed = self.transform(**kwargs)
            image = transformed["image"]
            mask  = transformed["mask"]
            if "border_edt" in transformed:
                border_edt = transformed["border_edt"]
            for key in poly_targets:
                if key in transformed:
                    poly_targets[key] = transformed[key]
            for key in ff_targets:
                if key in transformed:
                    ff_targets[key] = transformed[key]

        result: Dict[str, Any] = {"image": image, "mask": mask}
        if border_edt is not None:
            result["border_edt"] = border_edt
        elif self.has_border_edt:
            result["border_edt"] = np.zeros_like(mask, dtype=np.float32)

        # Frame field targets
        if ff_targets:
            result["edge"]  = ff_targets.get("edge",  np.zeros_like(mask, dtype=np.float32))
            result["theta"] = ff_targets.get("theta", np.zeros_like(mask, dtype=np.float32))

        # Combine voff_x and voff_y into a single (2, H, W) voff tensor
        if "vmap" in poly_targets:
            result["vmap"] = poly_targets["vmap"]
            vx = poly_targets.get("voff_x", np.zeros_like(poly_targets["vmap"]))
            vy = poly_targets.get("voff_y", np.zeros_like(poly_targets["vmap"]))
            result["voff"] = np.stack([vx, vy], axis=0)   # (2, H, W)
            # vx may be a Tensor after transform — handle both
            import torch as _torch
            if isinstance(vx, _torch.Tensor):
                result["vmask"] = (vx.abs() > 0.01).float().numpy()
            else:
                result["vmask"] = (np.abs(vx) > 0.01).astype(np.float32)

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
        self.border_edt       = bool(cfg.dataset.get("border_edt", True))
        self.frame_field      = bool(cfg.dataset.get("frame_field", False))
        self.annotations_path = cfg.dataset.get("annotations_path", None)
        self.ff_thickness     = int(cfg.dataset.get("ff_thickness", 2))
        self.prefetch_factor  = int(cfg.dataset.get("prefetch_factor", 4))

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

        has_poly = (Path(cfg.dataset.root) / "vmap").is_dir() and not self.frame_field
        self.train_transform = build_transforms(
            size=self.size, train=True,
            aug=cfg.dataset.get("aug", {}),
            border_edt=self.border_edt,
            vmap=has_poly, voff_x=has_poly, voff_y=has_poly,
            edge=self.frame_field, theta=self.frame_field,
        )
        self.eval_transform = build_transforms(
            size=self.size, train=False,
            border_edt=self.border_edt,
            vmap=has_poly, voff_x=has_poly, voff_y=has_poly,
            edge=self.frame_field, theta=self.frame_field,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = PavingDataset(
                root=self.root,
                split_indices=self.splits["train"],
                transform=self.train_transform,
                border_edt=self.border_edt,
                frame_field=self.frame_field,
                annotations_path=self.annotations_path,
                ff_thickness=self.ff_thickness,
            )
            self.val_dataset = PavingDataset(
                root=self.root,
                split_indices=self.splits["val"],
                transform=self.eval_transform,
                border_edt=self.border_edt,
                frame_field=self.frame_field,
                annotations_path=self.annotations_path,
                ff_thickness=self.ff_thickness,
            )
        if stage in ("test", None):
            self.test_dataset = PavingDataset(
                root=self.root,
                split_indices=self.splits["test"],
                transform=self.eval_transform,
                border_edt=self.border_edt,
                frame_field=self.frame_field,
                annotations_path=self.annotations_path,
                ff_thickness=self.ff_thickness,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )