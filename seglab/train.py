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
from seglab.data.splits import make_split_indices
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

        # Load metadata to get total count
        metadata_path = Path(cfg.dataset.root) / "tile_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        total_tiles = len(metadata["tiles"])

        # Create splits (make_split_indices only returns train/val, not test)
        splits = make_split_indices(
            total_tiles,
            cfg.seed,
            val_ratio=cfg.dataset.get("val_ratio", 0.15),
            cache_path=cache_dir / "splits" / f"wall_centerline_seed{cfg.seed}.json",
        )

        train_ds = WallCenterlineDataset(cfg.dataset.root, splits["train"], tf_train)
        val_ds = WallCenterlineDataset(cfg.dataset.root, splits["val"], tf_eval)
        # Use validation set as test set (common practice for single dataset)
        test_ds = WallCenterlineDataset(cfg.dataset.root, splits["val"], tf_eval)
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

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/dice",
        mode="max",
        save_top_k=1,
        filename="{epoch}-{val_dice:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Early stopping: stop if no improvement for 5 epochs
    early_stop_patience = cfg.trainer.get("early_stop_patience", 5)
    early_stop_cb = EarlyStopping(
        monitor="val/dice",
        patience=early_stop_patience,
        mode="max",
        verbose=True,
        min_delta=0.001,  # Minimum change to qualify as improvement
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
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(run_dir),
    )

    # Resume from checkpoint if specified
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path:
        print(f"\n🔄 Resuming training from checkpoint: {ckpt_path}")

    trainer.fit(lit_module, train_loader, val_loader, ckpt_path=ckpt_path)
    best_path = checkpoint_cb.best_model_path
    (run_dir / "best_ckpt.txt").write_text(best_path)

    # Final checkpoint upload (if MLflow callbacks weren't used)
    if cfg.logging.get("mlflow", False) and best_path and mlflow_logger is None:
        try:
            import mlflow
            print(f"\n📦 Uploading final checkpoint to MLflow/GCS...")
            mlflow.log_artifact(best_path, artifact_path="model")
            print(f"✅ Checkpoint uploaded: {Path(best_path).name}")
        except Exception as e:
            print(f"⚠️  Could not upload checkpoint: {e}")
    elif mlflow_logger is not None:
        print(f"\n✅ Training complete! Best checkpoint already uploaded to MLflow/GCS")
        print(f"   Checkpoint: {Path(best_path).name}")

    # Final test on best checkpoint
    test_results = trainer.test(lit_module, dataloaders=test_loader, ckpt_path="best")
    if test_results:
        (run_dir / "test_metrics.json").write_text(json.dumps(test_results[0], indent=2))
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
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))

    run_experiment(cfg)


if __name__ == "__main__":
    main()
