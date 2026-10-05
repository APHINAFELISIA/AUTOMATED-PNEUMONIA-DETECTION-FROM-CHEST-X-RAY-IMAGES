"""
Dataset loading utilities for the Chest X-Ray Pneumonia classification pipeline.

Expects the Kaggle-style folder layout::

    dataset/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

Each split directory must contain one sub-folder per class with image files inside.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def _is_image_file(path: Path) -> bool:
    """Return True if the path points to a supported image file."""
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def _resolve_class_dir(split_dir: Path, class_name: str) -> Path | None:
    """
    Locate a class subdirectory inside a split folder.

    Matching is case-insensitive so both ``NORMAL`` and ``normal`` are accepted.
    """
    if not split_dir.is_dir():
        return None

    target = class_name.lower()
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def _collect_image_paths(split_dir: Path) -> list[tuple[Path, int]]:
    """
    Scan a split directory and collect ``(image_path, label_index)`` pairs.

    Parameters
    ----------
    split_dir:
        Root directory for a data split (e.g. ``dataset/train``).

    Returns
    -------
    list[tuple[Path, int]]
        Sorted list of image paths with integer class labels.
    """
    samples: list[tuple[Path, int]] = []

    for class_name, label_idx in config.CLASS_TO_IDX.items():
        class_dir = _resolve_class_dir(split_dir, class_name)
        if class_dir is None:
            logger.warning(
                "Class folder '%s' not found under %s", class_name, split_dir
            )
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if _is_image_file(image_path):
                samples.append((image_path, label_idx))

    return samples


def _build_transforms(is_training: bool) -> transforms.Compose:
    """
    Build torchvision transforms for training or evaluation.

    Training applies geometric and photometric augmentation followed by
    ImageNet normalization. Validation and test use deterministic resizing only.

    Parameters
    ----------
    is_training:
        When True, enable random augmentation; otherwise use eval transforms.

    Returns
    -------
    transforms.Compose
        Composed transform pipeline producing tensors of shape ``[3, H, W]``.
    """
    normalize = transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
    aug = config.AUGMENTATION

    common_tail = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),  # PIL -> float tensor [3, 224, 224] in [0, 1]
        normalize,              # per-channel standardization
    ]

    if is_training:
        return transforms.Compose(
            [
                # Ensure 3-channel RGB regardless of source bit depth / mode.
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                transforms.RandomRotation(degrees=aug.rotation_degrees),
                transforms.RandomHorizontalFlip(p=aug.horizontal_flip_prob),
                transforms.RandomAffine(
                    degrees=0,
                    translate=None,
                    scale=aug.scale_range,
                    shear=0.0,
                ),
                transforms.ColorJitter(
                    brightness=aug.brightness,
                    contrast=aug.contrast,
                    saturation=aug.saturation,
                    hue=aug.hue,
                ),
                *common_tail,
            ]
        )

    return transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            *common_tail,
        ]
    )


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray pneumonia classification.

    Each item is a tuple ``(image_tensor, label)`` where:

    - ``image_tensor`` has shape ``[3, image_size, image_size]``
    - ``label`` is an integer in ``{0, 1}`` (NORMAL=0, PNEUMONIA=1)
    """

    def __init__(
        self,
        root_dir: Path | str,
        transform: transforms.Compose | None = None,
        is_training: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        root_dir:
            Path to a split directory (``train``, ``val``, or ``test``).
        transform:
            Optional custom transform. If omitted, a default pipeline is built
            from ``config`` based on ``is_training``.
        is_training:
            Whether to apply training augmentations when ``transform`` is None.
        """
        self.root_dir = Path(root_dir)
        self.is_training = is_training
        self.transform = transform or _build_transforms(is_training=is_training)
        self.samples: list[tuple[Path, int]] = _collect_image_paths(self.root_dir)

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in '{self.root_dir}'. "
                f"Expected class subfolders: {config.CLASS_NAMES}"
            )

        label_counts = {name: 0 for name in config.CLASS_NAMES}
        for _, label_idx in self.samples:
            label_counts[config.IDX_TO_CLASS[label_idx]] += 1

        logger.info(
            "Loaded %d samples from %s | %s",
            len(self.samples),
            self.root_dir,
            ", ".join(f"{k}={v}" for k, v in label_counts.items()),
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and transform a single sample.

        Parameters
        ----------
        index:
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``image`` tensor with shape ``[3, H, W]`` and scalar ``label`` tensor.
        """
        image_path, label = self.samples[index]

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")  # force RGB before transforms

        # After transform: image -> [3, 224, 224]
        image_tensor: torch.Tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

    def get_sample_path(self, index: int) -> Path:
        """Return the filesystem path for a dataset index (useful for visualization)."""
        return self.samples[index][0]

    def get_sample_label(self, index: int) -> int:
        """Return the integer class label for a dataset index."""
        return self.samples[index][1]


def get_dataloaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Parameters
    ----------
    batch_size:
        Mini-batch size. Defaults to ``config.BATCH_SIZE``.
    num_workers:
        Number of DataLoader worker processes. Defaults to ``config.NUM_WORKERS``.
    pin_memory:
        Whether to pin memory for faster GPU transfer. Defaults to ``config.PIN_MEMORY``.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``

        Each batch yields:

        - ``images``: float tensor ``[B, 3, 224, 224]``
        - ``labels``: long tensor ``[B]``
    """
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    pin_memory = pin_memory if pin_memory is not None else config.PIN_MEMORY

    train_dataset = ChestXRayDataset(
        root_dir=config.TRAIN_DIR,
        is_training=True,
    )
    val_dataset = ChestXRayDataset(
        root_dir=config.VAL_DIR,
        is_training=False,
    )
    test_dataset = ChestXRayDataset(
        root_dir=config.TEST_DIR,
        is_training=False,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    logger.info(
        "DataLoaders ready | train=%d, val=%d, test=%d batches (batch_size=%d)",
        len(train_loader),
        len(val_loader),
        len(test_loader),
        batch_size,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.set_seed()

    try:
        train_loader, val_loader, test_loader = get_dataloaders()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Place Kaggle images under dataset/train, dataset/val, and dataset/test "
            "with NORMAL and PNEUMONIA subfolders in each split."
        )
        raise SystemExit(1) from exc

    images, labels = next(iter(train_loader))
    # images: [B, 3, 224, 224], labels: [B]
    print(f"Train batch shape : images={tuple(images.shape)}, labels={tuple(labels.shape)}")
    print(f"Image value range : min={images.min():.3f}, max={images.max():.3f}")
    print(f"Label examples    : {labels[:8].tolist()}")
