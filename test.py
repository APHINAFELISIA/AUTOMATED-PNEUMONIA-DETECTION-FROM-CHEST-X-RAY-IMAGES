"""
Test-set evaluation for the ConvNeXt-LSTM pneumonia classifier.

Computes clinical classification metrics, saves diagnostic plots, and writes
a text report to ``results/evaluation_report.txt``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import ConvNeXtLSTMClassifier, load_checkpoint

logger = logging.getLogger(__name__)

# PNEUMONIA is the positive class (label index 1) for clinical metrics.
POSITIVE_CLASS_IDX: int = config.CLASS_TO_IDX["PNEUMONIA"]
POSITIVE_CLASS_NAME: str = "PNEUMONIA"


@dataclass
class EvaluationMetrics:
    """Container for binary classification metrics on the test set."""

    accuracy: float
    precision: float
    sensitivity: float
    specificity: float
    f1_score: float
    roc_auc: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    num_samples: int

    def to_report(self) -> str:
        """Format metrics as a human-readable text report."""
        lines = [
            "=" * 60,
            "  ConvNeXt-LSTM Pneumonia Detection — Test Evaluation",
            "=" * 60,
            "",
            f"Checkpoint : {config.BEST_MODEL_PATH}",
            f"Test split : {config.TEST_DIR}",
            f"Samples    : {self.num_samples}",
            f"Positive   : {POSITIVE_CLASS_NAME} (class index {POSITIVE_CLASS_IDX})",
            "",
            "-" * 60,
            "  Classification Metrics",
            "-" * 60,
            f"  Accuracy     : {self.accuracy:.4f}  ({self.accuracy * 100:.2f}%)",
            f"  Precision    : {self.precision:.4f}",
            f"  Sensitivity  : {self.sensitivity:.4f}  (Recall / TPR)",
            f"  Specificity  : {self.specificity:.4f}  (TNR)",
            f"  F1-Score     : {self.f1_score:.4f}",
            f"  ROC-AUC      : {self.roc_auc:.4f}",
            "",
            "-" * 60,
            "  Confusion Matrix Counts",
            "-" * 60,
            f"  True Positives  (TP) : {self.true_positives}",
            f"  True Negatives  (TN) : {self.true_negatives}",
            f"  False Positives (FP) : {self.false_positives}",
            f"  False Negatives (FN) : {self.false_negatives}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


@torch.no_grad()
def collect_predictions(
    model: ConvNeXtLSTMClassifier,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the test set and gather labels and probabilities.

    Parameters
    ----------
    model:
        Trained classifier in evaluation mode.
    test_loader:
        Test DataLoader yielding ``(images [B,3,H,W], labels [B])``.
    device:
        Compute device.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - ``y_true``    : ground-truth labels, shape ``[N]``
        - ``y_pred``    : predicted class indices, shape ``[N]``
        - ``y_prob_pos``: PNEUMONIA probability, shape ``[N]``
    """
    model.eval()

    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    progress = tqdm(test_loader, desc="Testing", unit="batch")

    for images, labels in progress:
        # images: [B, 3, 224, 224], labels: [B]
        images = images.to(device, non_blocking=True)

        logits = model(images)                          # [B, 2]
        probabilities = torch.softmax(logits, dim=1)    # [B, 2]
        predictions = torch.argmax(probabilities, dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(predictions.cpu().numpy())
        all_probs.append(probabilities[:, POSITIVE_CLASS_IDX].cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob_pos = np.concatenate(all_probs)

    return y_true, y_pred, y_prob_pos


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_pos: np.ndarray,
) -> EvaluationMetrics:
    """
    Compute binary classification metrics treating PNEUMONIA as positive.

    Parameters
    ----------
    y_true:
        Ground-truth labels ``[N]`` with values in ``{0, 1}``.
    y_pred:
        Predicted labels ``[N]``.
    y_prob_pos:
        Predicted probability of the positive (PNEUMONIA) class ``[N]``.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS_IDX, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob_pos)

    return EvaluationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        f1_score=float(f1),
        roc_auc=float(roc_auc),
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        num_samples=int(len(y_true)),
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
    """Save a annotated confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix -> %s", save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    metrics: EvaluationMetrics,
    save_path: Path,
) -> None:
    """Save ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos, pos_label=POSITIVE_CLASS_IDX)
    roc_auc_value = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        linewidth=2.5,
        label=f"ROC curve (AUC = {roc_auc_value:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve -> %s", save_path)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    metrics: EvaluationMetrics,
    save_path: Path,
) -> None:
    """Save precision-recall curve with F1 reference."""
    precision_vals, recall_vals, _ = precision_recall_curve(
        y_true,
        y_prob_pos,
        pos_label=POSITIVE_CLASS_IDX,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        recall_vals,
        precision_vals,
        color="#d62728",
        linewidth=2.5,
        label=f"PR curve (F1 @ threshold = {metrics.f1_score:.4f})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curve -> %s", save_path)


def save_evaluation_report(
    metrics: EvaluationMetrics,
    report_path: Path,
) -> None:
    """Write formatted metrics to a plain-text report file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = metrics.to_report()
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved evaluation report -> %s", report_path)
    print(report_text)


def evaluate(
    checkpoint_path: Path | None = None,
    batch_size: int | None = None,
) -> EvaluationMetrics:
    """
    Full test-set evaluation pipeline.

    Parameters
    ----------
    checkpoint_path:
        Model checkpoint path. Defaults to ``config.BEST_MODEL_PATH``.
    batch_size:
        Test mini-batch size. Defaults to ``config.BATCH_SIZE``.

    Returns
    -------
    EvaluationMetrics
        Computed classification metrics on the held-out test set.
    """
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    device = config.DEVICE

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Run train.py first to produce best_model.pth."
        )

    config.set_seed()
    logger.info("Evaluating on device: %s", device)

    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    model = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        pretrained_backbone=False,
    )

    y_true, y_pred, y_prob_pos = collect_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    metrics = compute_metrics(y_true, y_pred, y_prob_pos)

    plot_confusion_matrix(y_true, y_pred, config.CONFUSION_MATRIX_PATH)
    plot_roc_curve(y_true, y_prob_pos, metrics, config.ROC_CURVE_PATH)
    plot_precision_recall_curve(y_true, y_prob_pos, metrics, config.PR_CURVE_PATH)
    save_evaluation_report(metrics, config.EVALUATION_REPORT_PATH)

    return metrics


def main() -> None:
    """Entry point for command-line evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        evaluate()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
