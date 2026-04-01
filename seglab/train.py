"""Training entrypoint.

Usage:
  python -m seglab.train --config configs/base.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader, Subset

# Enable TensorFloat-32 for faster FP32 training on A100
torch.set_float32_matmul_precision('high')

from seglab.data import HFRetinaDataset, HFKvasirDataset, SLSSDDDataset, build_transforms
from seglab.data.splits import make_split_indices, make_split_indices_with_test, make_split_indices_by_group
from seglab.utils import (
    seed_everything,
    load_config,
    save_config,
    make_run_dir,
    collect_env_info,
    get_model,
)
from seglab.utils.env import load_dotenv
from seglab.utils.io import copy_code_snapshot
from seglab.callbacks import MLflowCheckpointUploader, MLflowMetricsLogger


def _merge_cfg(base_cfg: DictConfig, dataset_name: str, model_name: str) -> DictConfig:
    cfg = base_cfg.copy()
    ds_cfg_path = Path("configs/datasets") / f"{dataset_name}.yaml"
    if ds_cfg_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(ds_cfg_path))
    mdl_cfg_path = Path("configs/models") / f"{model_name}.yaml"
    if mdl_cfg_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(mdl_cfg_path))
    return cfg


def build_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    cache_dir = Path(cfg.paths.cache_dir)
    ds_type = cfg.dataset.type
    size = cfg.dataset.size

    tf_train = build_transforms(size=size, train=True, sar=cfg.dataset.get("sar", False), aug=cfg.dataset.get("aug"))
    tf_eval = build_transforms(size=size, train=False, sar=cfg.dataset.get("sar", False))

    if ds_type == "hf_retina":
        name = cfg.dataset.get("hf_name") or cfg.dataset.name
        train_full = HFRetinaDataset(
            name=name,
            split=cfg.dataset.train_split,
            transforms=tf_train,
            cache_dir=cache_dir / "hf",
            image_key=cfg.dataset.get("image_key", "image"),
            mask_key=cfg.dataset.get("mask_key", "label"),
        )
        splits = make_split_indices(
            len(train_full),
            cfg.seed,
            val_ratio=cfg.dataset.val_ratio,
            cache_path=cache_dir / "splits" / f"{cfg.dataset.name}_seed{cfg.seed}.json",
        )
        train_ds = Subset(train_full, splits["train"])
        val_ds = Subset(train_full, splits["val"])
        test_ds = HFRetinaDataset(
            name=name,
            split=cfg.dataset.test_split,
            transforms=tf_eval,
            cache_dir=cache_dir / "hf",
            image_key=cfg.dataset.get("image_key", "image"),
            mask_key=cfg.dataset.get("mask_key", "label"),
        )
    elif ds_type == "hf_kvasir":
        train_full = HFKvasirDataset(
            split=cfg.dataset.train_split,
            transforms=tf_train,
            cache_dir=cache_dir / "hf",
            hf_name=cfg.dataset.get("hf_name"),
            image_key=cfg.dataset.get("image_key", "image"),
            mask_key=cfg.dataset.get("mask_key", "mask"),
        )
        splits = make_split_indices(
            len(train_full),
            cfg.seed,
            val_ratio=cfg.dataset.val_ratio,
            cache_path=cache_dir / "splits" / f"{cfg.dataset.name}_seed{cfg.seed}.json",
        )
        train_ds = Subset(train_full, splits["train"])
        val_ds = Subset(train_full, splits["val"])
        test_ds = HFKvasirDataset(
            split=cfg.dataset.test_split,
            transforms=tf_eval,
            cache_dir=cache_dir / "hf",
            hf_name=cfg.dataset.get("hf_name"),
            image_key=cfg.dataset.get("image_key", "image"),
            mask_key=cfg.dataset.get("mask_key", "mask"),
        )
    elif ds_type == "sl_ssdd":
        root = cfg.dataset.root
        def _read_list(p: str) -> list[str]:
            return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]

        train_files = _read_list(cfg.dataset.train_list)
        val_files = _read_list(cfg.dataset.val_list)
        test_files = _read_list(cfg.dataset.test_list)
        train_ds = SLSSDDDataset(root=root, split_files=train_files, transforms=tf_train)
        val_ds = SLSSDDDataset(root=root, split_files=val_files, transforms=tf_eval)
        test_ds = SLSSDDDataset(root=root, split_files=test_files, transforms=tf_eval)
    elif ds_type == "wall_centerline":
        from seglab.data.wall_centerline import WallCenterlineDataset

        # Load metadata to get tiles
        metadata_path = Path(cfg.dataset.root) / "tile_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        tiles = metadata["tiles"]

        junction_heatmap = bool(cfg.dataset.get("junction_heatmap", False))
        distance_transform = bool(cfg.dataset.get("distance_transform", False))
        if junction_heatmap or distance_transform:
            tf_train = build_transforms(size=size, train=True, sar=cfg.dataset.get("sar", False),
                                        aug=cfg.dataset.get("aug"), junction_heatmap=junction_heatmap,
                                        distance_transform=distance_transform)
            tf_eval = build_transforms(size=size, train=False, sar=cfg.dataset.get("sar", False),
                                       junction_heatmap=junction_heatmap,
                                       distance_transform=distance_transform)

        # POC mode: train and validate on all tiles (no holdout)
        val_on_train = bool(cfg.dataset.get("val_on_train", False))

        if val_on_train:
            all_indices = list(range(len(tiles)))
            splits = {"train": all_indices, "val": all_indices, "test": []}
            print(f"\n{'='*60}")
            print(f"POC MODE (val_on_train): all {len(tiles)} tiles for train+val")
            print(f"{'='*60}\n")
        else:
            # Create train/val/test splits by PDF (prevents data leakage from overlapping tiles)
            # All tiles from the same PDF will be in the same split
            force_train = list(cfg.dataset.get("force_train_pdfs", []))
            splits = make_split_indices_by_group(
                tiles,
                cfg.seed,
                group_key="source_pdf",
                val_ratio=cfg.dataset.get("val_ratio", 0.15),
                test_ratio=cfg.dataset.get("test_ratio", 0.15),
                cache_path=cache_dir / "splits" / f"wall_centerline_by_pdf_seed{cfg.seed}.json",
                force_train_groups=force_train if force_train else None,
            )

            # Print split information
            if "_metadata" in splits:
                meta = splits["_metadata"]
                print(f"\n{'='*60}")
                print(f"DATASET SPLITS (by PDF to prevent data leakage)")
                print(f"{'='*60}")
                print(f"Total PDFs: {meta['total_groups']}")
                print(f"  Train: {meta['train_groups']} PDFs ({meta['train_tiles']} tiles)")
                print(f"  Val:   {meta['val_groups']} PDFs ({meta['val_tiles']} tiles)")
                print(f"  Test:  {meta['test_groups']} PDFs ({meta['test_tiles']} tiles)")
                print(f"{'='*60}\n")

        train_ds = WallCenterlineDataset(cfg.dataset.root, splits["train"], tf_train,
                                         junction_heatmap=junction_heatmap,
                                         distance_transform=distance_transform)
        val_ds = WallCenterlineDataset(cfg.dataset.root, splits["val"], tf_eval,
                                       junction_heatmap=junction_heatmap,
                                       distance_transform=distance_transform)

        # Test set: use separate test_root if provided (e.g., data_full for POC evaluation)
        test_root = cfg.dataset.get("test_root", None)
        if test_root:
            test_metadata_path = Path(test_root) / "tile_metadata.json"
            with open(test_metadata_path, "r") as f:
                test_metadata = json.load(f)
            test_tiles = test_metadata["tiles"]

            # Use the test split from the external dataset
            test_splits = make_split_indices_by_group(
                test_tiles,
                cfg.seed,
                group_key="source_pdf",
                val_ratio=cfg.dataset.get("test_root_val_ratio", 0.15),
                test_ratio=cfg.dataset.get("test_root_test_ratio", 0.15),
                cache_path=cache_dir / "splits" / f"wall_centerline_test_root_seed{cfg.seed}.json",
            )
            print(f"Test set from external root: {test_root}")
            if "_metadata" in test_splits:
                meta = test_splits["_metadata"]
                print(f"  Test: {meta['test_groups']} PDFs ({meta['test_tiles']} tiles)")
            test_ds = WallCenterlineDataset(test_root, test_splits["test"], tf_eval,
                                             junction_heatmap=junction_heatmap,
                                             distance_transform=distance_transform)
        else:
            test_ds = WallCenterlineDataset(cfg.dataset.root, splits["test"], tf_eval,
                                            junction_heatmap=junction_heatmap,
                                            distance_transform=distance_transform)
    elif ds_type == "paving":
        from seglab.data.paving import PavingDataset

        metadata_path = Path(cfg.dataset.root) / "tile_metadata.json"
        with open(metadata_path) as f:
            tiles = json.load(f)["tiles"]

        border_edt = bool(cfg.dataset.get("border_edt", True))
        tf_train = build_transforms(size=size, train=True, aug=cfg.dataset.get("aug"),
                                    border_edt=border_edt)
        tf_eval  = build_transforms(size=size, train=False, border_edt=border_edt)

        splits = make_split_indices_by_group(
            tiles, cfg.seed,
            group_key="source_pdf",
            val_ratio=cfg.dataset.get("val_ratio", 0.15),
            test_ratio=cfg.dataset.get("test_ratio", 0.15),
            cache_path=cache_dir / "splits" / f"paving_by_pdf_seed{cfg.seed}.json",
        )
        if "_metadata" in splits:
            meta = splits["_metadata"]
            print(f"\n{'='*60}")
            print(f"PAVING SPLITS (by PDF)")
            print(f"  Train: {meta['train_groups']} PDFs ({meta['train_tiles']} tiles)")
            print(f"  Val:   {meta['val_groups']} PDFs ({meta['val_tiles']} tiles)")
            print(f"  Test:  {meta['test_groups']} PDFs ({meta['test_tiles']} tiles)")
            print(f"{'='*60}\n")

        train_ds = PavingDataset(cfg.dataset.root, splits["train"], tf_train, border_edt=border_edt)
        val_ds   = PavingDataset(cfg.dataset.root, splits["val"],   tf_eval,  border_edt=border_edt)
        test_ds  = PavingDataset(cfg.dataset.root, splits["test"],  tf_eval,  border_edt=border_edt)

    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    dl_kwargs = dict(batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)
    return train_loader, val_loader, test_loader


def run_experiment(cfg: DictConfig, tag: Optional[str] = None) -> Path:
    # Load .env from project root
    import os
    from dotenv import load_dotenv as dotenv_load
    from pathlib import Path

    # Try multiple .env locations
    env_paths = [
        Path("/home/bensunshine/ml-training-platform/wall_centerline/TopoLoRA‑SAM/.env"),
        Path(__file__).parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            dotenv_load(env_path, override=True)
            print(f"✅ Loaded .env from: {env_path}")
            break

    # Setup MLflow credentials
    if 'MLFLOW_USERNAME' in os.environ:
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ['MLFLOW_USERNAME']
    if 'MLFLOW_PASSWORD' in os.environ:
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ['MLFLOW_PASSWORD']

    seed_everything(int(cfg.seed), deterministic=cfg.trainer.get("deterministic", True))

    run_dir = make_run_dir(
        cfg.paths.results_dir,
        cfg.experiment,
        cfg.dataset.name,
        cfg.model.name,
        int(cfg.seed),
        tag=tag,
    )
    ckpt_dir = Path(cfg.paths.checkpoints_dir) / cfg.experiment / cfg.dataset.name / cfg.model.name / f"seed{cfg.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, run_dir)

    # Debug: Print actual accumulate_grad_batches value
    actual_accumulate = cfg.trainer.get("accumulate_grad_batches", 1)
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION CHECK")
    print(f"{'='*60}")
    print(f"accumulate_grad_batches: {actual_accumulate}")
    print(f"batch_size: {cfg.dataset.batch_size}")
    print(f"Effective batch size: {actual_accumulate * cfg.dataset.batch_size}")
    print(f"Expected steps per epoch: {18486 // (actual_accumulate * cfg.dataset.batch_size)}")
    print(f"{'='*60}\n")

    env_info = collect_env_info()
    (run_dir / "env.json").write_text(json.dumps(env_info, indent=2))
    copy_code_snapshot(run_dir)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    model_builder = get_model(cfg.model.name)
    lit_module = model_builder(cfg)

    # Transfer learning: load model weights only (fresh optimizer, epoch 0)
    transfer_ckpt = cfg.get("transfer_ckpt", None)
    if transfer_ckpt:
        print(f"\n{'='*60}")
        print(f"TRANSFER LEARNING")
        print(f"{'='*60}")
        print(f"Loading weights from: {transfer_ckpt}")
        ckpt = torch.load(transfer_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        # Filter out keys with shape mismatches (e.g. heads whose in_channels changed)
        model_state = lit_module.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k not in model_state or model_state[k].shape == v.shape}
        skipped = [k for k, v in state_dict.items()
                   if k in model_state and model_state[k].shape != v.shape]
        if skipped:
            print(f"  Skipped {len(skipped)} keys with shape mismatch (head architecture changed): {skipped}")
        missing, unexpected = lit_module.load_state_dict(filtered, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (expected for fresh components)")
        print(f"Model weights loaded. Optimizer and scheduler will start fresh.")
        print(f"{'='*60}\n")

    trainable_params = sum(p.numel() for p in lit_module.parameters() if p.requires_grad)
    (run_dir / "trainable_params.txt").write_text(str(trainable_params))

    csv_logger = CSVLogger(save_dir=str(run_dir), name="logs")
    loggers = [csv_logger]
    mlflow_logger = None

    # MLflow logger
    if cfg.logging.get("mlflow", False):
        try:
            from pytorch_lightning.loggers import MLFlowLogger
            import os

            tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'https://mlflow-212833769695.us-east5.run.app/')
            experiment_name = os.environ.get('EXPERIMENT_NAME', cfg.logging.get('mlflow_experiment', 'wall-semantic-segmentation'))
            run_name = os.environ.get('RUN_NAME', f"{cfg.model.name}_{cfg.dataset.name}_seed{cfg.seed}")

            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                run_name=run_name,
            )
            loggers.append(mlflow_logger)
            print(f"✅ MLflow logging enabled: {experiment_name}")
            print(f"   Tracking URI: {tracking_uri}")
            print(f"   Run name: {run_name}")
        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")
            mlflow_logger = None

    if cfg.logging.get("wandb", False):
        try:
            from pytorch_lightning.loggers import WandbLogger

            loggers.append(
                WandbLogger(
                    project=cfg.logging.project,
                    entity=cfg.logging.get("entity"),
                    tags=list(cfg.logging.get("tags", [])),
                    save_dir=str(run_dir),
                )
            )
        except Exception as e:
            print(f"[warn] wandb logging requested but unavailable: {e}")

    # Choose monitoring metric based on which head is actively training
    has_junction = cfg.model.get("junction_head", {}).get("enabled", False)
    has_centerline = cfg.model.get("centerline_head", {}).get("enabled", False)
    freeze_existing = cfg.get("freeze_existing", False)

    # Allow explicit override from config
    monitor_metric = cfg.trainer.get("monitor", None)
    monitor_mode = cfg.trainer.get("monitor_mode", None)

    if monitor_metric is None:
        if has_centerline and freeze_existing:
            monitor_metric = "val/centerline_mae"
            monitor_mode = "min"
        elif has_junction and freeze_existing:
            monitor_metric = "val/junction_dice"
            monitor_mode = "max"
        else:
            monitor_metric = "val/dice"
            monitor_mode = "max"
    if monitor_mode is None:
        monitor_mode = "max"

    ckpt_metric_name = monitor_metric.replace("/", "_")
    ckpt_filename = "{epoch}-{" + monitor_metric + ":.4f}"

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        filename=ckpt_filename,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Early stopping: stop if no improvement over min_delta for N epochs
    early_stop_patience = cfg.trainer.get("early_stop_patience", 5)
    early_stop_min_delta = cfg.trainer.get("early_stop_min_delta", 0.001)
    early_stop_cb = EarlyStopping(
        monitor=monitor_metric,
        patience=early_stop_patience,
        mode=monitor_mode,
        verbose=True,
        min_delta=early_stop_min_delta,
    )

    # Custom progress bar for cleaner log files
    progress_bar = TQDMProgressBar(refresh_rate=100)  # Update every 100 steps instead of every step

    # Setup callbacks list
    callbacks = [checkpoint_cb, lr_monitor, early_stop_cb, progress_bar]
    print(f"✅ Early stopping enabled (patience={early_stop_patience} epochs)")
    print(f"✅ Progress bar refresh rate: 100 steps (cleaner logs)")

    # Add MLflow callbacks if enabled
    if mlflow_logger is not None:
        mlflow_uploader = MLflowCheckpointUploader(mlflow_logger)
        mlflow_metrics = MLflowMetricsLogger(mlflow_logger)
        callbacks.extend([mlflow_uploader, mlflow_metrics])
        print(f"✅ MLflow checkpoint auto-upload enabled (uploads after each validation)")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.get("deterministic", True),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0.0),
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(run_dir),
    )

    # Resume from checkpoint if specified (full resume, NOT transfer learning)
    # For transfer learning use transfer_ckpt instead (weights only, fresh optimizer)
    ckpt_path = cfg.get("ckpt_path", None) if not transfer_ckpt else None
    if ckpt_path:
        print(f"\nResuming training from checkpoint: {ckpt_path}")

    trainer.fit(lit_module, train_loader, val_loader, ckpt_path=ckpt_path)
    best_path = checkpoint_cb.best_model_path
    (run_dir / "best_ckpt.txt").write_text(best_path)

    # Final checkpoint upload (if MLflow callbacks weren't used)
    if cfg.logging.get("mlflow", False) and best_path and mlflow_logger is None:
        try:
            import mlflow
            print(f"\n📦 Uploading final checkpoint to MLflow/GCS...")
            mlflow.log_artifact(best_path, "models")
            print(f"✅ Checkpoint uploaded: {Path(best_path).name}")
        except Exception as e:
            print(f"⚠️  Could not upload checkpoint: {e}")
    elif mlflow_logger is not None:
        print(f"\n✅ Training complete! Best checkpoint already uploaded to MLflow/GCS")
        print(f"   Checkpoint: {Path(best_path).name}")

    # Final test on best checkpoint
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_results = trainer.test(lit_module, dataloaders=test_loader, ckpt_path="best")

    if test_results:
        test_metrics = test_results[0]

        # Save to JSON
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

        # Log test metrics to MLflow (matching UNet style)
        if cfg.logging.get("mlflow", False) and mlflow_logger is not None:
            try:
                import mlflow

                # Log all test metrics to MLflow
                with mlflow.start_run(run_id=mlflow_logger.run_id):
                    mlflow.log_metrics({
                        key: value for key, value in test_metrics.items()
                        if isinstance(value, (int, float))
                    })

                print("\n" + "="*60)
                print("Test Evaluation Complete!")
                print("="*60)
                print(f"Test metrics logged to MLflow")
                for key, value in test_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                print("="*60)

            except Exception as e:
                print(f"⚠️  Could not log test metrics to MLflow: {e}")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dataset", default=None, help="Dataset config name (e.g., drive).")
    parser.add_argument("--model", default=None, help="Model config name (e.g., unet).")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    dataset_name = args.dataset or base_cfg.dataset.name
    model_name = args.model or base_cfg.model.name
    cfg = _merge_cfg(base_cfg, dataset_name, model_name)
    # Re-merge experiment config so it takes precedence over defaults
    cfg = OmegaConf.merge(cfg, base_cfg)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))

    run_experiment(cfg)


if __name__ == "__main__":
    main()
