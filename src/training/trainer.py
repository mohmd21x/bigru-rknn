"""Generic training loop with checkpointing and early stopping."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.data.dataset import resolve_manifest_for_split
from src.evaluation.metrics import compute_classification_metrics
from src.models.base import BaseFallModel
from src.training.losses import build_loss


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields in a dataloader batch to ``device``."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def resolve_class_weights(
    config: dict[str, Any],
    repo_root: Path,
) -> torch.Tensor | None:
    """Build ``CrossEntropyLoss`` class weights from config or train manifest."""
    training_cfg = config.get("training", {})
    class_weights_cfg = training_cfg.get("class_weights", "auto")

    if class_weights_cfg is None or class_weights_cfg is False:
        return None

    if isinstance(class_weights_cfg, (list, tuple)):
        weights = [float(value) for value in class_weights_cfg]
        if len(weights) != NUM_CLASSES:
            raise ValueError(
                f"class_weights must have length {NUM_CLASSES}, got {len(weights)}"
            )
        return torch.tensor(weights, dtype=torch.float32)

    if class_weights_cfg != "auto":
        raise ValueError(
            f"class_weights must be 'auto', a list, or null; got {class_weights_cfg!r}"
        )

    manifest = resolve_manifest_for_split(config, "train", repo_root)
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for row in manifest:
        counts[row.label_id] += 1.0

    total = counts.sum()
    if total <= 0:
        return None

    # Inverse-frequency weights normalized to sum to num_classes.
    weights = total / (NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


class Trainer:
    """Train a fall-detection model with validation, checkpointing, and early stopping."""

    def __init__(
        self,
        model: BaseFallModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        device: torch.device,
        repo_root: Path | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]

        training_cfg = config.get("training", {})
        self.epochs = int(training_cfg.get("epochs", 50))
        self.gradient_clip = float(training_cfg.get("gradient_clip", 1.0))
        self.early_stopping_patience = int(training_cfg.get("early_stopping_patience", 10))

        run_name = str(config.get("run_name", model.model_name))
        checkpoint_root = training_cfg.get("checkpoint_dir", "checkpoints")
        self.run_dir = self._resolve_path(checkpoint_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        class_weights = resolve_class_weights(config, self.repo_root)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = build_loss(config, class_weights).to(device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=float(training_cfg.get("lr", 1e-4)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.history_path = self.run_dir / "history.csv"
        self._init_history_file()

        config_copy_path = self.run_dir / "config.yaml"
        with config_copy_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def _resolve_path(self, path: Path | str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.repo_root / path

    def _init_history_file(self) -> None:
        if self.history_path.is_file():
            return
        with self.history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "train_macro_f1",
                    "val_loss",
                    "val_accuracy",
                    "val_macro_f1",
                    "val_fall_f1",
                    "lr",
                ]
            )

    def _append_history_row(self, epoch: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any]) -> None:
        current_lr = self.optimizer.param_groups[0]["lr"]
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['accuracy']:.6f}",
                    f"{train_metrics['macro_f1']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['accuracy']:.6f}",
                    f"{val_metrics['macro_f1']:.6f}",
                    f"{val_metrics['per_class'][CLASS_NAMES[1]]['f1']:.6f}",
                    f"{current_lr:.8f}",
                ]
            )

    def train(self) -> Path:
        """Run the full training loop and return the best checkpoint path."""
        best_path = self.run_dir / "best.pt"

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()
            self.scheduler.step(val_metrics["loss"])
            self._append_history_row(epoch, train_metrics, val_metrics)

            val_f1 = val_metrics["per_class"][CLASS_NAMES[1]]["f1"]
            print(
                f"Epoch {epoch}/{self.epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_fall_f1={val_f1:.4f}"
            )

            self._save_checkpoint(self.run_dir / "last.pt", epoch, val_metrics)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(best_path, epoch, val_metrics)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(
                        f"Early stopping after {epoch} epochs "
                        f"(best val fall F1={self.best_val_f1:.4f} at epoch {self.best_epoch})"
                    )
                    break

        if not best_path.is_file():
            raise RuntimeError("Training finished without saving a best checkpoint")

        return best_path

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_metrics: dict[str, Any],
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_f1": self.best_val_f1,
                "val_metrics": val_metrics,
                "config": self.config,
                "model_name": self.model.model_name,
            },
            path,
        )

    def _train_epoch(self) -> dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        all_labels: list[int] = []
        all_preds: list[int] = []

        progress = tqdm(self.train_loader, desc="train", leave=False)
        for batch in progress:
            batch = move_batch_to_device(batch, self.device)
            labels = batch["label"]
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            else:
                labels = labels.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch)
            loss = self.criterion(logits, labels)
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * labels.size(0)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        num_samples = max(len(all_labels), 1)
        metrics = compute_classification_metrics(
            np.asarray(all_labels, dtype=np.int64),
            np.asarray(all_preds, dtype=np.int64),
        )
        metrics["loss"] = total_loss / num_samples
        return metrics

    @torch.no_grad()
    def _validate_epoch(self) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_labels: list[int] = []
        all_preds: list[int] = []

        for batch in tqdm(self.val_loader, desc="val", leave=False):
            batch = move_batch_to_device(batch, self.device)
            labels = batch["label"]
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            else:
                labels = labels.to(self.device, dtype=torch.long)

            logits = self.model(batch)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=-1)

            total_loss += float(loss.item()) * labels.size(0)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

        num_samples = max(len(all_labels), 1)
        metrics = compute_classification_metrics(
            np.asarray(all_labels, dtype=np.int64),
            np.asarray(all_preds, dtype=np.int64),
        )
        metrics["loss"] = total_loss / num_samples
        return metrics


def load_checkpoint(
    checkpoint_path: Path | str,
    model: BaseFallModel,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    """Restore model weights (and optionally optimizer state) from a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
