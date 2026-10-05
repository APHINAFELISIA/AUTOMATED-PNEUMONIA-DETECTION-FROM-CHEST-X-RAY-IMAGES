"""
Native Grad-CAM implementation for the ConvNeXt-LSTM pneumonia classifier.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights spatial regions
that most influence the model's prediction by combining forward activations from
the ConvNeXt feature extractor with backward gradients of the target class score.

Hooks are registered on ``model.feature_extractor``, which outputs a tensor of
shape ``[B, 768, 7, 7]`` for 224×224 inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from model import ConvNeXtLSTMClassifier

logger = logging.getLogger(__name__)


def get_inference_transform() -> transforms.Compose:
    """
    Build deterministic preprocessing for inference and Grad-CAM.

    Returns
    -------
    transforms.Compose
        Pipeline producing tensors of shape ``[3, image_size, image_size]``.
    """
    return transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
        ]
    )


def load_image_tensor(
    image_path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load an image from disk and return a normalized tensor plus RGB uint8 array.

    Parameters
    ----------
    image_path:
        Path to a chest X-ray image file.
    device:
        Target device for the tensor. Defaults to ``config.DEVICE``.

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        - ``input_tensor`` : shape ``[1, 3, H, W]``, ImageNet-normalized
        - ``original_rgb`` : shape ``[H, W, 3]``, uint8 for visualization
    """
    device = device or config.DEVICE
    transform = get_inference_transform()

    with Image.open(image_path) as pil_image:
        rgb_image = pil_image.convert("RGB")
        original_rgb = np.array(
            rgb_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        )
        input_tensor = transform(rgb_image).unsqueeze(0)  # [1, 3, 224, 224]

    return input_tensor.to(device), original_rgb


def tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a single image tensor back to an RGB uint8 numpy array.

    Parameters
    ----------
    tensor:
        Normalized image tensor with shape ``[3, H, W]`` or ``[1, 3, H, W]``.

    Returns
    -------
    np.ndarray
        RGB image with shape ``[H, W, 3]`` and dtype ``uint8``.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    mean = torch.tensor(config.IMAGE_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(config.IMAGE_STD, device=tensor.device).view(3, 1, 1)

    # Denormalize: [3, H, W] -> [H, W, 3]
    image = tensor * std + mean
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    return (image * 255.0).astype(np.uint8)


class GradCAM:
    """
    Hook-based Grad-CAM for ``ConvNeXtLSTMClassifier``.

    Forward and backward hooks are attached to ``model.feature_extractor`` to
    capture activation maps ``[B, 768, 7, 7]`` and their gradients during a
    targeted backward pass on the selected class logit.

    Parameters
    ----------
    model:
        Trained ConvNeXt-LSTM classifier (eval mode recommended).
    target_layer:
        Submodule to hook. Defaults to ``model.feature_extractor``.
    """

    def __init__(
        self,
        model: ConvNeXtLSTMClassifier,
        target_layer: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer or model.feature_extractor

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )
        self._backward_handle = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Store feature-map activations from the forward pass."""
        # output shape: [B, 768, 7, 7]
        self.activations = output

    def _backward_hook(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Store gradients flowing back into the feature maps."""
        # grad_output[0] shape: [B, 768, 7, 7]
        self.gradients = grad_output[0]

    def remove_hooks(self) -> None:
        """Detach forward and backward hooks."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass

    def _resolve_target_class(
        self,
        logits: torch.Tensor,
        target_class: int | None,
    ) -> int:
        """Use the argmax prediction when no explicit target class is given."""
        if target_class is not None:
            return target_class
        return int(torch.argmax(logits, dim=1).item())

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """
        Compute a raw Grad-CAM heatmap for a single input tensor.

        Parameters
        ----------
        input_tensor:
            Normalized input with shape ``[1, 3, H, W]``.
        target_class:
            Class index to explain. If None, uses the predicted class.

        Returns
        -------
        tuple[np.ndarray, int, torch.Tensor]
            - ``cam``       : normalized heatmap, shape ``[H, W]``, values in [0, 1]
            - ``class_idx`` : explained class index
            - ``logits``    : model output, shape ``[1, num_classes]``
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        self.model.eval()
        self.activations = None
        self.gradients = None

        # Forward pass: populates activations via hook
        logits = self.model(input_tensor)  # [1, num_classes]
        class_idx = self._resolve_target_class(logits, target_class)

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Verify that target_layer is model.feature_extractor."
            )

        activations = self.activations[0]  # [768, 7, 7]
        gradients = self.gradients[0]      # [768, 7, 7]

        # Global-average-pool gradients across spatial dims -> channel weights [768]
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [768, 1, 1]

        # Weighted combination of activation maps -> [7, 7]
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to input spatial resolution [224, 224]
        cam_resized = cv2.resize(
            cam_np,
            (config.IMAGE_SIZE, config.IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        return cam_resized, class_idx, logits.detach()

    def generate_heatmap(
        self,
        cam: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the OpenCV Jet colormap to a normalized CAM.

        Parameters
        ----------
        cam:
            Normalized heatmap with shape ``[H, W]`` in [0, 1].

        Returns
        -------
        np.ndarray
            BGR color heatmap with shape ``[H, W, 3]``, dtype ``uint8``.
        """
        heatmap_uint8 = np.uint8(255.0 * cam)
        return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    def overlay_on_image(
        self,
        original_rgb: np.ndarray,
        cam: np.ndarray,
        alpha: float | None = None,
    ) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap onto an RGB image using OpenCV Jet colormap.

        Parameters
        ----------
        original_rgb:
            Base image, shape ``[H, W, 3]``, RGB uint8.
        cam:
            Normalized heatmap, shape ``[H, W]``.
        alpha:
            Heatmap blend weight. Defaults to ``config.GRADCAM_ALPHA``.

        Returns
        -------
        np.ndarray
            Blended RGB image, shape ``[H, W, 3]``, dtype ``uint8``.
        """
        alpha = alpha if alpha is not None else config.GRADCAM_ALPHA

        if original_rgb.shape[:2] != cam.shape[:2]:
            cam = cv2.resize(
                cam,
                (original_rgb.shape[1], original_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        heatmap_bgr = self.generate_heatmap(cam)
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        overlay_bgr = cv2.addWeighted(
            original_bgr,
            1.0 - alpha,
            heatmap_bgr,
            alpha,
            0.0,
        )

        return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray | None = None,
        target_class: int | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end Grad-CAM explanation for a single image tensor.

        Parameters
        ----------
        input_tensor:
            Normalized tensor with shape ``[1, 3, H, W]``.
        original_rgb:
            Optional RGB uint8 image for overlay. If None, denormalized from tensor.
        target_class:
            Class index to explain. Defaults to predicted class.
        alpha:
            Overlay transparency.

        Returns
        -------
        dict
            Keys: ``cam``, ``heatmap``, ``overlay``, ``class_idx``,
            ``class_name``, ``confidence``, ``probabilities``, ``logits``.
        """
        cam, class_idx, logits = self.generate_cam(input_tensor, target_class)
        probabilities = torch.softmax(logits, dim=1)[0]

        if original_rgb is None:
            original_rgb = tensor_to_rgb_uint8(input_tensor)

        heatmap_bgr = self.generate_heatmap(cam)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay = self.overlay_on_image(original_rgb, cam, alpha=alpha)

        return {
            "cam": cam,
            "heatmap": heatmap_rgb,
            "overlay": overlay,
            "class_idx": class_idx,
            "class_name": config.IDX_TO_CLASS[class_idx],
            "confidence": float(probabilities[class_idx].item()),
            "probabilities": probabilities.cpu().numpy(),
            "logits": logits.cpu().numpy(),
        }


def explain_image(
    model: ConvNeXtLSTMClassifier,
    image_path: str | Path,
    target_class: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper: load an image path and return a Grad-CAM explanation.

    Parameters
    ----------
    model:
        Trained classifier.
    image_path:
        Path to a chest X-ray image.
    target_class:
        Optional class index to explain.
    device:
        Compute device.

    Returns
    -------
    dict
        Grad-CAM explanation dictionary from ``GradCAM.explain()``.
    """
    device = device or config.DEVICE
    input_tensor, original_rgb = load_image_tensor(image_path, device=device)

    grad_cam = GradCAM(model)
    try:
        return grad_cam.explain(
            input_tensor=input_tensor,
            original_rgb=original_rgb,
            target_class=target_class,
        )
    finally:
        grad_cam.remove_hooks()


def save_side_by_side(
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    save_path: str | Path,
    title_left: str = "Original",
    title_right: str = "Grad-CAM Overlay",
) -> None:
    """
    Save a side-by-side comparison of the original image and Grad-CAM overlay.

    Parameters
    ----------
    original_rgb:
        Original image, shape ``[H, W, 3]``.
    overlay_rgb:
        Grad-CAM overlay, shape ``[H, W, 3]``.
    save_path:
        Output file path (``.png`` recommended).
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title(title_left, fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(title_right, fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Grad-CAM comparison -> %s", save_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from model import load_checkpoint

    parser = argparse.ArgumentParser(description="Grad-CAM demo for a single X-ray.")
    parser.add_argument("image_path", type=str, help="Path to a chest X-ray image.")
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        choices=list(range(config.NUM_CLASSES)),
        help="Class index to explain (default: predicted class).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.RESULTS_DIR / "gradcam_demo.png"),
        help="Output path for side-by-side visualization.",
    )
    args = parser.parse_args()

    model = load_checkpoint(device=config.DEVICE, pretrained_backbone=False)
    result = explain_image(
        model=model,
        image_path=args.image_path,
        target_class=args.target_class,
    )

    input_tensor, original_rgb = load_image_tensor(args.image_path)
    save_side_by_side(
        original_rgb=original_rgb,
        overlay_rgb=result["overlay"],
        save_path=args.output,
        title_right=f"Grad-CAM ({result['class_name']}, {result['confidence']:.2%})",
    )

    print(f"Predicted class : {result['class_name']}")
    print(f"Confidence      : {result['confidence']:.4f}")
    print(f"Saved overlay   : {args.output}")
