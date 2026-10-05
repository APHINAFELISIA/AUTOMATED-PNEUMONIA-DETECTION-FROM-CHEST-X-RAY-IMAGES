"""
Research-grade multi-panel Grad-CAM visualization grid generator.

Automatically samples NORMAL and PNEUMONIA cases from the test set, runs
inference with Grad-CAM, and saves a publication-ready comparison figure.
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

import config
from dataset import VALID_IMAGE_EXTENSIONS, _resolve_class_dir
from gradcam import GradCAM, load_image_tensor
from model import ConvNeXtLSTMClassifier, load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Container for a single visualized test case."""

    image_path: Path
    true_label: str
    predicted_label: str
    confidence: float
    is_correct: bool
    original_rgb: np.ndarray
    overlay_rgb: np.ndarray
    cam: np.ndarray


def _collect_class_images(split_dir: Path, class_name: str) -> list[Path]:
    """Return all image paths for a given class within a data split."""
    class_dir = _resolve_class_dir(split_dir, class_name)
    if class_dir is None:
        return []

    images = [
        path
        for path in sorted(class_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    return images


def sample_test_cases(
    samples_per_class: int | None = None,
    split_dir: Path | None = None,
    seed: int | None = None,
) -> list[tuple[Path, int]]:
    """
    Randomly sample balanced test cases from each class.

    Parameters
    ----------
    samples_per_class:
        Number of images per class. Defaults to
        ``config.VISUALIZATION_SAMPLES_PER_CLASS``.
    split_dir:
        Data split directory. Defaults to ``config.TEST_DIR``.
    seed:
        Random seed for reproducible sampling.

    Returns
    -------
    list[tuple[Path, int]]
        List of ``(image_path, label_index)`` tuples.
    """
    samples_per_class = (
        samples_per_class
        if samples_per_class is not None
        else config.VISUALIZATION_SAMPLES_PER_CLASS
    )
    split_dir = split_dir or config.TEST_DIR
    seed = seed if seed is not None else config.SEED

    rng = random.Random(seed)
    sampled: list[tuple[Path, int]] = []

    for class_name, label_idx in config.CLASS_TO_IDX.items():
        candidates = _collect_class_images(split_dir, class_name)
        if not candidates:
            logger.warning("No images found for class '%s' in %s", class_name, split_dir)
            continue

        count = min(samples_per_class, len(candidates))
        selected = rng.sample(candidates, k=count)
        sampled.extend((path, label_idx) for path in selected)

        logger.info(
            "Sampled %d/%d images for class '%s'",
            count,
            len(candidates),
            class_name,
        )

    if not sampled:
        raise FileNotFoundError(
            f"No test images found under '{split_dir}'. "
            "Ensure NORMAL and PNEUMONIA subfolders contain images."
        )

    return sampled


def process_sample(
    model: ConvNeXtLSTMClassifier,
    grad_cam: GradCAM,
    image_path: Path,
    true_label_idx: int,
    device: torch.device | None = None,
) -> SampleResult:
    """
    Run inference and Grad-CAM on a single sampled image.

    Parameters
    ----------
    model:
        Trained classifier in evaluation mode.
    grad_cam:
        Reusable GradCAM instance (hooks persist across calls).
    image_path:
        Path to the chest X-ray image.
    true_label_idx:
        Ground-truth class index.
    device:
        Compute device.

    Returns
    -------
    SampleResult
        Visualization data for one grid cell.
    """
    device = device or config.DEVICE
    true_label = config.IDX_TO_CLASS[true_label_idx]

    input_tensor, original_rgb = load_image_tensor(image_path, device=device)
    explanation = grad_cam.explain(
        input_tensor=input_tensor,
        original_rgb=original_rgb,
        target_class=None,
    )

    predicted_label = explanation["class_name"]
    is_correct = predicted_label == true_label

    return SampleResult(
        image_path=image_path,
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=explanation["confidence"],
        is_correct=is_correct,
        original_rgb=original_rgb,
        overlay_rgb=explanation["overlay"],
        cam=explanation["cam"],
    )


def _title_color(is_correct: bool) -> str:
    """Return green for correct predictions, red for incorrect."""
    return "#2ca02c" if is_correct else "#d62728"


def create_visual_grid(
    results: list[SampleResult],
    save_path: Path | None = None,
    title: str = "ConvNeXt-LSTM Grad-CAM — Clinical Test Cases",
) -> Path:
    """
    Render a research-grade multi-panel grid and save to disk.

    Layout: 4 columns per sample row — Original | Grad-CAM | Original | Grad-CAM ...
    Rows are grouped by class (NORMAL cases first, then PNEUMONIA).

    Parameters
    ----------
    results:
        Processed sample results from ``process_sample()``.
    save_path:
        Output figure path. Defaults to ``config.VISUALIZATION_GRID_PATH``.
    title:
        Figure suptitle.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    save_path = save_path or config.VISUALIZATION_GRID_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Group results by true label for ordered display
    grouped: dict[str, list[SampleResult]] = {name: [] for name in config.CLASS_NAMES}
    for result in results:
        grouped[result.true_label].append(result)

    ordered_results: list[SampleResult] = []
    row_labels: list[str] = []
    for class_name in config.CLASS_NAMES:
        for result in grouped[class_name]:
            ordered_results.append(result)
            row_labels.append(class_name)

    num_rows = len(ordered_results)
    num_cols = 2  # Original + Grad-CAM overlay

    fig = plt.figure(figsize=(8, 3.2 * num_rows))
    gs = GridSpec(
        num_rows,
        num_cols,
        figure=fig,
        wspace=0.05,
        hspace=0.35,
    )

    for row_idx, result in enumerate(ordered_results):
        # Column 0: Original X-ray
        ax_orig = fig.add_subplot(gs[row_idx, 0])
        ax_orig.imshow(result.original_rgb, cmap="gray" if result.original_rgb.ndim == 2 else None)
        ax_orig.set_ylabel(
            result.true_label,
            fontsize=11,
            fontweight="bold",
            rotation=0,
            labelpad=40,
            va="center",
        )
        ax_orig.set_title("Original", fontsize=10)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        # Column 1: Grad-CAM overlay
        ax_cam = fig.add_subplot(gs[row_idx, 1])
        ax_cam.imshow(result.overlay_rgb)
        status = "Correct" if result.is_correct else "Incorrect"
        color = _title_color(result.is_correct)
        ax_cam.set_title(
            f"Grad-CAM | Pred: {result.predicted_label} "
            f"({result.confidence:.1%}) — {status}",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])

        # Subtle filename annotation
        fig.text(
            0.5,
            1.0 - (row_idx / num_rows) - 0.01,
            result.image_path.name,
            ha="center",
            va="top",
            fontsize=7,
            color="gray",
            transform=fig.transFigure,
        )

    correct_count = sum(1 for r in ordered_results if r.is_correct)
    accuracy = correct_count / len(ordered_results) if ordered_results else 0.0

    fig.suptitle(
        f"{title}\n"
        f"Samples: {len(ordered_results)} | "
        f"Correct: {correct_count}/{len(ordered_results)} "
        f"({accuracy:.1%})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Saved visual grid -> %s", save_path)
    return save_path


def generate_visualization_grid(
    checkpoint_path: Path | None = None,
    samples_per_class: int | None = None,
    split_dir: Path | None = None,
    save_path: Path | None = None,
    seed: int | None = None,
) -> Path:
    """
    Full pipeline: sample test cases, run Grad-CAM, save multi-panel grid.

    Parameters
    ----------
    checkpoint_path:
        Model checkpoint. Defaults to ``config.BEST_MODEL_PATH``.
    samples_per_class:
        Images per class to visualize.
    split_dir:
        Data split to sample from (default: test set).
    save_path:
        Output figure path.
    seed:
        Random seed for sampling.

    Returns
    -------
    Path
        Path to the saved visualization grid.
    """
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Run train.py first."
        )

    config.set_seed(seed)
    device = config.DEVICE

    sampled_cases = sample_test_cases(
        samples_per_class=samples_per_class,
        split_dir=split_dir,
        seed=seed,
    )

    model = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        pretrained_backbone=False,
    )

    grad_cam = GradCAM(model)
    results: list[SampleResult] = []

    try:
        for image_path, label_idx in sampled_cases:
            logger.info("Processing %s (true=%s)", image_path.name, config.IDX_TO_CLASS[label_idx])
            result = process_sample(
                model=model,
                grad_cam=grad_cam,
                image_path=image_path,
                true_label_idx=label_idx,
                device=device,
            )
            results.append(result)
    finally:
        grad_cam.remove_hooks()

    output_path = create_visual_grid(results, save_path=save_path)

    print()
    print("=" * 55)
    print("  Grad-CAM Visual Grid — Summary")
    print("=" * 55)
    for result in results:
        status = "OK" if result.is_correct else "MISS"
        print(
            f"  [{status}] True={result.true_label:<10} "
            f"Pred={result.predicted_label:<10} "
            f"Conf={result.confidence:.1%}  "
            f"({result.image_path.name})"
        )
    print("-" * 55)
    print(f"  Saved grid: {output_path}")
    print("=" * 55)
    print()

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate a multi-panel Grad-CAM visualization grid.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.BEST_MODEL_PATH),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=str(config.TEST_DIR),
        help="Directory to sample images from (default: test set).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=config.VISUALIZATION_SAMPLES_PER_CLASS,
        help="Number of images to sample per class.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.VISUALIZATION_GRID_PATH),
        help="Output path for the visualization grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed for reproducible sampling.",
    )
    return parser


def main() -> None:
    """Entry point for command-line grid generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        generate_visualization_grid(
            checkpoint_path=Path(args.checkpoint),
            samples_per_class=args.samples_per_class,
            split_dir=Path(args.split_dir),
            save_path=Path(args.output),
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
