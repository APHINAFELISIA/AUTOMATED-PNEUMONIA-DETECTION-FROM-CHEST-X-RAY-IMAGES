"""
Training script for the ConvNeXt-LSTM pneumonia classifier.

Features
--------
- CrossEntropyLoss + AdamW optimizer
- CosineAnnealingLR learning-rate schedule
- Mixed-precision training (AMP) on CUDA via ``torch.amp.autocast``
- Early stopping on validation loss (patience = 5)
- Best-model checkpointing to ``saved_models/best_model.pth``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import ConvNeXtLSTMClassifier, build_model

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Container for metrics computed during one train or validation pass."""

    loss: float
    accuracy: float
    num_samples: int


@dataclass
class TrainingHistory:
    """Stores per-epoch metrics for logging and optional analysis."""

    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


class EarlyStopping:
    """
    Halt training when validation loss stops improving.

    Parameters
    ----------
    patience:
        Number of consecutive epochs without improvement before stopping.
    min_delta:
        Minimum decrease in validation loss considered as improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Update early-stopping state with the latest validation loss.

        Returns
        -------
        bool
            True if validation loss improved (new best), else False.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_one_epoch(
    model: ConvNeXtLSTMClassifier,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    epoch: int,
) -> EpochMetrics:
    """
    Run one full training epoch.

    Parameters
    ----------
    model:
        Classifier in training mode.
    train_loader:
        Training DataLoader yielding ``(images [B,3,H,W], labels [B])``.
    criterion:
        Loss function (CrossEntropyLoss).
    optimizer:
        AdamW optimizer.
    device:
        Compute device.
    scaler:
        Gradient scaler for mixed-precision training.
    use_amp:
        Whether AMP is enabled (CUDA only).
    epoch:
        Current epoch index (1-based) for progress display.

    Returns
    -------
    EpochMetrics
        Average loss and accuracy over the training set.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    amp_device_type = device.type if device.type in {"cuda", "cpu"} else "cpu"

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch:02d} [Train]",
        leave=False,
        unit="batch",
    )

    for images, labels in progress:
        # images: [B, 3, 224, 224], labels: [B]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(amp_device_type, enabled=use_amp):
            logits = model(images)          # [B, num_classes]
            loss = criterion(logits, labels)  # scalar

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_loss = loss.item()
        batch_correct = (torch.argmax(logits, dim=1) == labels).sum().item()

        running_loss += batch_loss * batch_size
        running_correct += batch_correct
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{batch_loss:.4f}",
            acc=f"{batch_correct / batch_size:.4f}",
        )

    avg_loss = running_loss / total_samples
    avg_accuracy = running_correct / total_samples

    return EpochMetrics(loss=avg_loss, accuracy=avg_accuracy, num_samples=total_samples)


@torch.no_grad()
def validate(
    model: ConvNeXtLSTMClassifier,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> EpochMetrics:
    """
    Evaluate the model on the validation set.

    Returns
    -------
    EpochMetrics
        Average validation loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    amp_device_type = device.type if device.type in {"cuda", "cpu"} else "cpu"

    progress = tqdm(
        val_loader,
        desc=f"Epoch {epoch:02d} [Val]  ",
        leave=False,
        unit="batch",
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        with torch.amp.autocast(amp_device_type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_loss = loss.item()
        batch_correct = (torch.argmax(logits, dim=1) == labels).sum().item()

        running_loss += batch_loss * batch_size
        running_correct += batch_correct
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{batch_loss:.4f}",
            acc=f"{batch_correct / batch_size:.4f}",
        )

    avg_loss = running_loss / total_samples
    avg_accuracy = running_correct / total_samples

    return EpochMetrics(loss=avg_loss, accuracy=avg_accuracy, num_samples=total_samples)


def save_checkpoint(
    model: ConvNeXtLSTMClassifier,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    val_metrics: EpochMetrics,
    checkpoint_path: Path,
    history: TrainingHistory,
) -> None:
    """Persist the best model weights and training metadata."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_metrics.loss,
        "val_accuracy": val_metrics.accuracy,
        "class_names": config.CLASS_NAMES,
        "history": {
            "train_loss": history.train_loss,
            "train_accuracy": history.train_accuracy,
            "val_loss": history.val_loss,
            "val_accuracy": history.val_accuracy,
            "learning_rates": history.learning_rates,
        },
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(
        "Saved best checkpoint -> %s (val_loss=%.4f, val_acc=%.4f)",
        checkpoint_path,
        val_metrics.loss,
        val_metrics.accuracy,
    )


def train(
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
    checkpoint_path: Path | None = None,
) -> TrainingHistory:
    """
    Full training pipeline with early stopping and best-model saving.

    Parameters
    ----------
    num_epochs:
        Maximum training epochs. Defaults to ``config.NUM_EPOCHS``.
    learning_rate:
        AdamW learning rate. Defaults to ``config.LEARNING_RATE``.
    batch_size:
        Mini-batch size. Defaults to ``config.BATCH_SIZE``.
    patience:
        Early-stopping patience. Defaults to ``config.EARLY_STOPPING_PATIENCE``.
    checkpoint_path:
        Output path for the best weights. Defaults to ``config.BEST_MODEL_PATH``.

    Returns
    -------
    TrainingHistory
        Per-epoch training and validation metrics.
    """
    num_epochs = num_epochs if num_epochs is not None else config.NUM_EPOCHS
    learning_rate = learning_rate if learning_rate is not None else config.LEARNING_RATE
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    patience = patience if patience is not None else config.EARLY_STOPPING_PATIENCE
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH

    device = config.DEVICE
    use_amp = config.USE_AMP and device.type == "cuda"

    config.set_seed()
    logger.info("Training on device: %s | AMP enabled: %s", device, use_amp)

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    model = build_model(device=device, pretrained=True, freeze_backbone=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stopping = EarlyStopping(patience=patience)
    history = TrainingHistory()

    best_val_loss = float("inf")
    training_start = time.time()

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
        )

        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history.train_loss.append(train_metrics.loss)
        history.train_accuracy.append(train_metrics.accuracy)
        history.val_loss.append(val_metrics.loss)
        history.val_accuracy.append(val_metrics.accuracy)
        history.learning_rates.append(current_lr)

        epoch_bar.set_postfix(
            train_loss=f"{train_metrics.loss:.4f}",
            val_loss=f"{val_metrics.loss:.4f}",
            val_acc=f"{val_metrics.accuracy:.4f}",
            lr=f"{current_lr:.2e}",
        )

        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f | lr=%.2e",
            epoch,
            num_epochs,
            train_metrics.loss,
            train_metrics.accuracy,
            val_metrics.loss,
            val_metrics.accuracy,
            current_lr,
        )

        improved = early_stopping.step(val_metrics.loss)
        if improved:
            best_val_loss = val_metrics.loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_metrics=val_metrics,
                checkpoint_path=checkpoint_path,
                history=history,
            )

        if early_stopping.should_stop:
            logger.info(
                "Early stopping triggered after %d epochs without val-loss improvement.",
                patience,
            )
            break

    elapsed = time.time() - training_start
    logger.info(
        "Training finished in %.1f min | best val_loss=%.4f",
        elapsed / 60.0,
        best_val_loss,
    )

    return history


def main() -> None:
    """Entry point for command-line training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        train()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Ensure the dataset is organized under dataset/train, dataset/val, "
            "and dataset/test with NORMAL and PNEUMONIA subfolders."
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
