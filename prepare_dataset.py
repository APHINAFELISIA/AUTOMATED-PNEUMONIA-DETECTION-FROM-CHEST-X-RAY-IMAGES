"""
Prepare the Kaggle Chest X-Ray Pneumonia dataset for training.

The raw Kaggle archive contains only ``train/`` and ``test/`` splits. This script
creates a stratified validation split from the training set and organizes images
into the project layout expected by ``dataset.py``::

    dataset/
        train/   NORMAL, PNEUMONIA  (~85% of Kaggle train)
        val/     NORMAL, PNEUMONIA  (~15% of Kaggle train)
        test/    NORMAL, PNEUMONIA  (original Kaggle test, unchanged)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
from pathlib import Path

import config

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def _find_class_dir(split_dir: Path, class_name: str) -> Path | None:
    """Locate a class folder using case-insensitive matching."""
    if not split_dir.is_dir():
        return None
    target = class_name.lower()
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def collect_images(class_dir: Path) -> list[Path]:
    """Recursively collect supported image files from a class directory."""
    return sorted(
        path for path in class_dir.rglob("*") if _is_image(path)
    )


def stratified_split(
    image_paths: list[Path],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    """
    Split image paths into train and validation subsets.

    Parameters
    ----------
    image_paths:
        All images for one class.
    val_ratio:
        Fraction assigned to validation (e.g. 0.15 = 15%).
    rng:
        Seeded random generator for reproducibility.
    """
    shuffled = image_paths.copy()
    rng.shuffle(shuffled)

    if len(shuffled) < 2:
        return shuffled, []

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)

    val_paths = shuffled[:val_count]
    train_paths = shuffled[val_count:]
    return train_paths, val_paths


def _link_or_copy(source: Path, destination: Path, use_symlink: bool) -> None:
    """Copy or symlink a single file to the destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if use_symlink:
        os.symlink(source.resolve(), destination)
    else:
        shutil.copy2(source, destination)


def _clear_split_dir(split_dir: Path) -> None:
    """Remove existing class subfolders inside a split directory."""
    if not split_dir.exists():
        split_dir.mkdir(parents=True, exist_ok=True)
        return

    for child in split_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)


def prepare_dataset(
    source_dir: Path,
    val_ratio: float = 0.15,
    seed: int = config.SEED,
    use_symlink: bool = False,
    clear_existing: bool = True,
) -> dict[str, dict[str, int]]:
    """
    Build train / val / test folders from the raw Kaggle ``chest_xray`` directory.

    Parameters
    ----------
    source_dir:
        Path to the Kaggle ``chest_xray`` folder (contains ``train/`` and ``test/``).
    val_ratio:
        Fraction of the original training set reserved for validation.
    seed:
        Random seed for the stratified train/val split.
    use_symlink:
        If True, create symlinks instead of copying files (saves disk space).
    clear_existing:
        If True, remove existing contents of ``dataset/train``, ``val``, ``test``.

    Returns
    -------
    dict
        Counts per split and class, e.g.
        ``{"train": {"NORMAL": 1200, ...}, "val": {...}, "test": {...}}``.
    """
    source_dir = source_dir.resolve()
    kaggle_train = source_dir / "train"
    kaggle_test = source_dir / "test"

    if not kaggle_train.is_dir() or not kaggle_test.is_dir():
        raise FileNotFoundError(
            f"Expected Kaggle layout not found under '{source_dir}'.\n"
            "The folder must contain 'train/' and 'test/' subdirectories, each "
            "with NORMAL/ and PNEUMONIA/ class folders.\n"
            "Example: chest_xray/train/NORMAL, chest_xray/test/PNEUMONIA"
        )

    if clear_existing:
        for split_dir in (config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR):
            _clear_split_dir(split_dir)

    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    # --- Split original Kaggle train into project train + val ---
    for class_name in config.CLASS_NAMES:
        class_dir = _find_class_dir(kaggle_train, class_name)
        if class_dir is None:
            raise FileNotFoundError(
                f"Class folder '{class_name}' not found in {kaggle_train}"
            )

        images = collect_images(class_dir)
        if not images:
            raise FileNotFoundError(f"No images found in {class_dir}")

        train_paths, val_paths = stratified_split(images, val_ratio, rng)

        for split_name, paths, target_root in (
            ("train", train_paths, config.TRAIN_DIR),
            ("val", val_paths, config.VAL_DIR),
        ):
            dest_class_dir = target_root / class_name
            for src in paths:
                dest = dest_class_dir / src.name
                _link_or_copy(src, dest, use_symlink=use_symlink)
            counts[split_name][class_name] = len(paths)

        logger.info(
            "%s: %d train, %d val (from %d total)",
            class_name,
            len(train_paths),
            len(val_paths),
            len(images),
        )

    # --- Copy Kaggle test set unchanged ---
    for class_name in config.CLASS_NAMES:
        class_dir = _find_class_dir(kaggle_test, class_name)
        if class_dir is None:
            raise FileNotFoundError(
                f"Class folder '{class_name}' not found in {kaggle_test}"
            )

        images = collect_images(class_dir)
        dest_class_dir = config.TEST_DIR / class_name

        for src in images:
            dest = dest_class_dir / src.name
            _link_or_copy(src, dest, use_symlink=use_symlink)

        counts["test"][class_name] = len(images)
        logger.info("%s: %d test", class_name, len(images))

    return counts


def print_summary(counts: dict[str, dict[str, int]], val_ratio: float) -> None:
    """Print a formatted summary of the prepared dataset."""
    print()
    print("=" * 60)
    print("  Dataset Preparation Complete")
    print("=" * 60)
    print(f"  Output directory : {config.DATASET_DIR}")
    print(f"  Validation ratio : {val_ratio:.0%} (from Kaggle train)")
    print("-" * 60)

    for split in ("train", "val", "test"):
        split_counts = counts[split]
        total = sum(split_counts.values())
        detail = ", ".join(f"{k}={v}" for k, v in split_counts.items())
        print(f"  {split:<6} : {total:>5} images  ({detail})")

    grand_total = sum(sum(c.values()) for c in counts.values())
    print("-" * 60)
    print(f"  Total  : {grand_total} images")
    print("=" * 60)
    print()
    print("Next step: python train.py")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Kaggle Chest X-Ray data for ConvNeXt-LSTM training.",
    )
    parser.add_argument(
        "source_dir",
        type=str,
        help="Path to the Kaggle 'chest_xray' folder (contains train/ and test/).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of Kaggle train reserved for validation (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed for the train/val split (default: 42).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files (saves disk space).",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not remove existing dataset/train, val, test contents first.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        counts = prepare_dataset(
            source_dir=Path(args.source_dir),
            val_ratio=args.val_ratio,
            seed=args.seed,
            use_symlink=args.symlink,
            clear_existing=not args.no_clear,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    print_summary(counts, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
