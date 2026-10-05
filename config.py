"""
Central configuration module for the ConvNeXt-LSTM pneumonia detection pipeline.

All paths, hyperparameters, and hardware settings are defined here so that
training, evaluation, and inference scripts share a single source of truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Project root and directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATASET_DIR: Path = PROJECT_ROOT / "dataset"
TRAIN_DIR: Path = DATASET_DIR / "train"
VAL_DIR: Path = DATASET_DIR / "val"
TEST_DIR: Path = DATASET_DIR / "test"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
SAVED_MODELS_DIR: Path = PROJECT_ROOT / "saved_models"

BEST_MODEL_PATH: Path = SAVED_MODELS_DIR / "best_model.pth"
EVALUATION_REPORT_PATH: Path = RESULTS_DIR / "evaluation_report.txt"


def ensure_directories() -> None:
    """Create all required project directories if they do not already exist."""
    for directory in (
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        RESULTS_DIR,
        SAVED_MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hardware / device selection
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA (NVIDIA GPU) -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = get_device()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (including CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN can slow training; enable for strict reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Class labels (Kaggle Chest X-Ray Pneumonia dataset)
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {idx: name for name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES: int = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Image preprocessing (ImageNet statistics)
# ---------------------------------------------------------------------------
IMAGE_SIZE: int = 224
IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
CONVNEXT_FEATURE_DIM: int = 768          # ConvNeXt-Tiny output channels
CONVNEXT_SPATIAL_SIZE: int = 7           # Spatial map: [B, 768, 7, 7]
SEQUENCE_LENGTH: int = CONVNEXT_SPATIAL_SIZE ** 2  # 49 spatial tokens

LSTM_HIDDEN_SIZE: int = 256
LSTM_NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.3                # Applied between stacked LSTM layers
CLASSIFIER_DROPOUT: float = 0.5          # Applied before the final linear head


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4               # AdamW L2 regularization
EARLY_STOPPING_PATIENCE: int = 5

# Mixed-precision training (CUDA only; disabled on CPU/MPS for stability)
USE_AMP: bool = DEVICE.type == "cuda"

# DataLoader workers (0 on Windows avoids multiprocessing spawn issues)
NUM_WORKERS: int = 0 if DEVICE.type == "cpu" else 2
PIN_MEMORY: bool = DEVICE.type == "cuda"


# ---------------------------------------------------------------------------
# Data augmentation (training set only)
# ---------------------------------------------------------------------------
@dataclass
class AugmentationConfig:
    """Hyperparameters for torchvision training transforms."""

    rotation_degrees: float = 15.0
    horizontal_flip_prob: float = 0.5
    scale_range: tuple[float, float] = (0.9, 1.1)
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    hue: float = 0.05


AUGMENTATION: AugmentationConfig = AugmentationConfig()


# ---------------------------------------------------------------------------
# Evaluation and visualization
# ---------------------------------------------------------------------------
CONFUSION_MATRIX_PATH: Path = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH: Path = RESULTS_DIR / "roc_curve.png"
PR_CURVE_PATH: Path = RESULTS_DIR / "precision_recall_curve.png"

GRADCAM_ALPHA: float = 0.45              # Heatmap overlay transparency
VISUALIZATION_SAMPLES_PER_CLASS: int = 4
VISUALIZATION_GRID_PATH: Path = RESULTS_DIR / "gradcam_visual_grid.png"


# ---------------------------------------------------------------------------
# Aggregated config object (optional convenience for scripts)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Snapshot of all runtime settings for logging and checkpoint metadata."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    device: torch.device = field(default_factory=get_device)
    seed: int = SEED
    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    image_size: int = IMAGE_SIZE
    num_classes: int = NUM_CLASSES
    lstm_hidden_size: int = LSTM_HIDDEN_SIZE
    lstm_num_layers: int = LSTM_NUM_LAYERS
    use_amp: bool = USE_AMP
    class_names: list[str] = field(default_factory=lambda: CLASS_NAMES.copy())


def get_config() -> Config:
    """Return a fresh Config instance reflecting current module-level settings."""
    return Config()


# Initialize directories on import so downstream scripts can rely on them.
ensure_directories()
