"""Custom PyTorch Lightning callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class MLflowCheckpointUploader(Callback):
    """
    Upload checkpoints to MLflow/GCS immediately after each validation when a new best is saved.

    This ensures checkpoints are safely stored even if training crashes or is interrupted.
    """

    def __init__(self, mlflow_logger: Optional[pl.loggers.MLFlowLogger] = None):
        """
        Args:
            mlflow_logger: MLflow logger instance for uploading artifacts
        """
        super().__init__()
        self.mlflow_logger = mlflow_logger
        self._last_uploaded_path: Optional[str] = None

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when validation ends. Uploads checkpoint if it's a new best.

        This is called after validation but before the next training epoch starts.
        """
        if not self.mlflow_logger:
            return

        # Find the ModelCheckpoint callback
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                checkpoint_callback = callback
                break

        if not checkpoint_callback:
            return

        # Check if a checkpoint was saved
        best_model_path = checkpoint_callback.best_model_path

        if not best_model_path:
            return

        # Only upload if it's a new checkpoint (different from last uploaded)
        if best_model_path == self._last_uploaded_path:
            return

        # Upload to MLflow/GCS
        try:
            import mlflow

            checkpoint_name = Path(best_model_path).name
            current_epoch = trainer.current_epoch

            # Get current metrics for printing
            checkpoint_callback = None
            for callback in trainer.callbacks:
                if isinstance(callback, pl.callbacks.ModelCheckpoint):
                    checkpoint_callback = callback
                    break

            metric_value = ""
            if checkpoint_callback and checkpoint_callback.best_model_score is not None:
                metric_value = f" ({checkpoint_callback.monitor}: {checkpoint_callback.best_model_score:.4f})"

            # Log best model to MLflow/GCS using the logger's run context
            run_id = self.mlflow_logger.run_id
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(best_model_path, "models")
            print(f"✓ Saved best model{metric_value} - uploaded to GCS")

            # Track this as the last uploaded checkpoint
            self._last_uploaded_path = best_model_path

        except Exception as e:
            print(f"⚠️  Failed to upload checkpoint to MLflow: {e}")
            import traceback
            traceback.print_exc()


class MLflowMetricsLogger(Callback):
    """
    Enhanced MLflow logging with additional metrics and artifacts.
    """

    def __init__(self, mlflow_logger: Optional[pl.loggers.MLFlowLogger] = None):
        super().__init__()
        self.mlflow_logger = mlflow_logger

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log training configuration at the start."""
        if not self.mlflow_logger:
            return

        try:
            import mlflow

            # Log model architecture info
            total_params = sum(p.numel() for p in pl_module.parameters())
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

            run_id = self.mlflow_logger.run_id
            with mlflow.start_run(run_id=run_id):
                mlflow.log_param("total_parameters", total_params)
                mlflow.log_param("trainable_parameters", trainable_params)
                mlflow.log_param("trainable_percentage", f"{100 * trainable_params / total_params:.2f}%")

        except Exception as e:
            print(f"⚠️  Failed to log training info: {e}")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log additional validation metrics."""
        if not self.mlflow_logger:
            return

        try:
            import mlflow

            # Log epoch number explicitly
            run_id = self.mlflow_logger.run_id
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("epoch", trainer.current_epoch, step=trainer.global_step)

        except Exception as e:
            pass  # Silently fail for non-critical logging
