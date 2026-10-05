"""
Single-image inference script for the ConvNeXt-LSTM pneumonia classifier.

Loads the best checkpoint, predicts the class with confidence, generates a
Grad-CAM heatmap, and saves a side-by-side original vs overlay comparison.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from gradcam import GradCAM, load_image_tensor, save_side_by_side
from model import load_checkpoint

logger = logging.getLogger(__name__)


def predict_single_image(
    image_path: str | Path,
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
    target_class: int | None = None,
    show_plot: bool = False,
) -> dict:
    """
    Run inference and Grad-CAM explanation on a single chest X-ray image.

    Parameters
    ----------
    image_path:
        Path to the input chest X-ray image.
    checkpoint_path:
        Model checkpoint path. Defaults to ``config.BEST_MODEL_PATH``.
    output_path:
        Path to save the side-by-side visualization. Defaults to
        ``results/predictions/<image_stem>_gradcam.png``.
    target_class:
        Optional class index for Grad-CAM targeting. Defaults to predicted class.
    show_plot:
        If True, display the comparison figure interactively.

    Returns
    -------
    dict
        Prediction and Grad-CAM results including ``class_name``, ``confidence``,
        ``probabilities``, ``overlay``, and ``output_path``.
    """
    image_path = Path(image_path)
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Run train.py first."
        )

    device = config.DEVICE
    config.set_seed()

    logger.info("Loading model from %s", checkpoint_path)
    model = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        pretrained_backbone=False,
    )

    input_tensor, original_rgb = load_image_tensor(image_path, device=device)

    grad_cam = GradCAM(model)
    try:
        explanation = grad_cam.explain(
            input_tensor=input_tensor,
            original_rgb=original_rgb,
            target_class=target_class,
        )
    finally:
        grad_cam.remove_hooks()

    if output_path is None:
        output_dir = config.RESULTS_DIR / "predictions"
        output_path = output_dir / f"{image_path.stem}_gradcam.png"
    else:
        output_path = Path(output_path)

    save_side_by_side(
        original_rgb=original_rgb,
        overlay_rgb=explanation["overlay"],
        save_path=output_path,
        title_left="Original X-Ray",
        title_right=(
            f"Grad-CAM: {explanation['class_name']} "
            f"({explanation['confidence']:.1%})"
        ),
    )

    _print_prediction_summary(image_path, explanation)

    if show_plot:
        _display_comparison(original_rgb, explanation)

    result = {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "class_idx": explanation["class_idx"],
        "class_name": explanation["class_name"],
        "confidence": explanation["confidence"],
        "probabilities": explanation["probabilities"],
        "overlay": explanation["overlay"],
        "cam": explanation["cam"],
    }
    return result


def _print_prediction_summary(image_path: Path, explanation: dict) -> None:
    """Print a formatted prediction summary to the console."""
    probabilities = explanation["probabilities"]
    print()
    print("=" * 55)
    print("  Pneumonia Detection — Inference Result")
    print("=" * 55)
    print(f"  Image       : {image_path.name}")
    print(f"  Prediction  : {explanation['class_name']}")
    print(f"  Confidence  : {explanation['confidence']:.2%}")
    print("-" * 55)
    for idx, class_name in enumerate(config.CLASS_NAMES):
        print(f"  P({class_name:<10}): {probabilities[idx]:.4f}")
    print("=" * 55)
    print()


def _display_comparison(original_rgb: np.ndarray, explanation: dict) -> None:
    """Display original and Grad-CAM overlay in an interactive matplotlib window."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original X-Ray", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(explanation["overlay"])
    axes[1].set_title(
        f"Grad-CAM: {explanation['class_name']} ({explanation['confidence']:.1%})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    fig.tight_layout()
    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Predict pneumonia from a single chest X-ray with Grad-CAM.",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to a chest X-ray image file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.BEST_MODEL_PATH),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the side-by-side Grad-CAM figure.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        choices=list(range(config.NUM_CLASSES)),
        help="Class index for Grad-CAM (default: predicted class).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the comparison plot interactively.",
    )
    return parser


def main() -> None:
    """Entry point for command-line single-image prediction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = predict_single_image(
            image_path=args.image_path,
            checkpoint_path=Path(args.checkpoint),
            output_path=Path(args.output) if args.output else None,
            target_class=args.target_class,
            show_plot=args.show,
        )
        logger.info("Saved visualization -> %s", result["output_path"])
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
