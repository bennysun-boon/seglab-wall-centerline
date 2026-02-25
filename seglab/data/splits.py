"""Deterministic train/val/test splits with caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def make_split_indices(
    n: int, seed: int, val_ratio: float = 0.2, cache_path: str | Path | None = None
) -> Dict[str, List[int]]:
    """Return indices for train/val given n examples."""
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            train_idx = data.get("train")
            val_idx = data.get("val")
            if isinstance(train_idx, list) and isinstance(val_idx, list):
                all_idx = train_idx + val_idx
                if all_idx and max(all_idx) < n:
                    return {"train": train_idx, "val": val_idx}
            # Cache is incompatible (e.g., dataset length changed); fall through to recompute.

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n).tolist()
    val_n = int(n * val_ratio)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    splits = {"train": train_idx, "val": val_idx}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(splits))
    return splits


def make_split_indices_with_test(
    n: int, seed: int, val_ratio: float = 0.15, test_ratio: float = 0.15, cache_path: str | Path | None = None
) -> Dict[str, List[int]]:
    """Return indices for train/val/test given n examples.

    Args:
        n: Total number of examples
        seed: Random seed for reproducibility
        val_ratio: Fraction of data to use for validation (default: 0.15)
        test_ratio: Fraction of data to use for test (default: 0.15)
        cache_path: Optional path to cache the split indices

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing index lists
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            train_idx = data.get("train")
            val_idx = data.get("val")
            test_idx = data.get("test")
            if isinstance(train_idx, list) and isinstance(val_idx, list) and isinstance(test_idx, list):
                all_idx = train_idx + val_idx + test_idx
                if all_idx and max(all_idx) < n and len(all_idx) == n:
                    return {"train": train_idx, "val": val_idx, "test": test_idx}
            # Cache is incompatible (e.g., dataset length changed); fall through to recompute.

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n).tolist()

    # Calculate split sizes
    val_n = int(n * val_ratio)
    test_n = int(n * test_ratio)
    train_n = n - val_n - test_n

    # Split: [val | test | train]
    val_idx = perm[:val_n]
    test_idx = perm[val_n:val_n + test_n]
    train_idx = perm[val_n + test_n:]

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(splits))
    return splits


def make_split_indices_by_group(
    tiles: List[Dict],
    seed: int,
    group_key: str = "source_pdf",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    cache_path: str | Path | None = None,
) -> Dict[str, List[int]]:
    """Return indices for train/val/test by grouping tiles (e.g., by source PDF).

    This ensures that all tiles from the same group (e.g., same PDF) stay together
    in the same split, preventing data leakage from overlapping tiles.

    Args:
        tiles: List of tile metadata dicts, each with a group identifier
        seed: Random seed for reproducibility
        group_key: Key in tile dict to group by (e.g., 'source_pdf')
        val_ratio: Fraction of groups for validation (default: 0.15)
        test_ratio: Fraction of groups for test (default: 0.15)
        cache_path: Optional path to cache the split indices

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing tile index lists
    """
    n = len(tiles)

    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            train_idx = data.get("train")
            val_idx = data.get("val")
            test_idx = data.get("test")
            if isinstance(train_idx, list) and isinstance(val_idx, list) and isinstance(test_idx, list):
                all_idx = train_idx + val_idx + test_idx
                if all_idx and max(all_idx) < n and len(set(all_idx)) == n:
                    return {"train": train_idx, "val": val_idx, "test": test_idx}
            # Cache is incompatible; fall through to recompute.

    # Group tiles by their source identifier
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, tile in enumerate(tiles):
        group_id = tile.get(group_key)
        if group_id is None:
            raise ValueError(f"Tile {idx} missing '{group_key}' field")
        groups[group_id].append(idx)

    # Get sorted group IDs for reproducibility
    group_ids = sorted(groups.keys())
    n_groups = len(group_ids)

    # Shuffle groups
    rng = np.random.RandomState(seed)
    perm_groups = rng.permutation(group_ids).tolist()

    # Calculate split sizes (by number of groups)
    val_n = int(n_groups * val_ratio)
    test_n = int(n_groups * test_ratio)

    # Split groups
    val_groups = perm_groups[:val_n]
    test_groups = perm_groups[val_n:val_n + test_n]
    train_groups = perm_groups[val_n + test_n:]

    # Expand groups to tile indices
    train_idx = []
    val_idx = []
    test_idx = []

    for group_id in train_groups:
        train_idx.extend(groups[group_id])
    for group_id in val_groups:
        val_idx.extend(groups[group_id])
    for group_id in test_groups:
        test_idx.extend(groups[group_id])

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    # Add metadata about the split
    splits["_metadata"] = {
        "total_tiles": n,
        "total_groups": n_groups,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_tiles": len(train_idx),
        "val_tiles": len(val_idx),
        "test_tiles": len(test_idx),
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(splits, indent=2))

    return splits
