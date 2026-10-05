"""
Hybrid ConvNeXt-Tiny + LSTM classifier for chest X-ray pneumonia detection.

Architecture overview
---------------------
1. ConvNeXt-Tiny backbone (pretrained on ImageNet) extracts spatial features.
2. The ``[B, 768, 7, 7]`` feature map is reshaped into a sequence of 49 tokens.
3. A stacked LSTM models spatial dependencies across the token sequence.
4. A dropout-regularized linear head produces class logits.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

import config

logger = logging.getLogger(__name__)


class ConvNeXtLSTMClassifier(nn.Module):
    """
    Hybrid ConvNeXt-LSTM binary classifier.

    Parameters
    ----------
    num_classes:
        Number of output classes (default: 2 for NORMAL vs PNEUMONIA).
    lstm_hidden_size:
        Hidden dimension of the LSTM layers.
    lstm_num_layers:
        Number of stacked LSTM layers.
    lstm_dropout:
        Dropout between LSTM layers (active only when ``lstm_num_layers > 1``).
    classifier_dropout:
        Dropout applied to the final LSTM hidden state before classification.
    pretrained:
        If True, load ImageNet-pretrained ConvNeXt-Tiny weights.
    freeze_backbone:
        If True, disable gradient updates for the ConvNeXt feature extractor.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        lstm_hidden_size: int = config.LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = config.LSTM_NUM_LAYERS,
        lstm_dropout: float = config.LSTM_DROPOUT,
        classifier_dropout: float = config.CLASSIFIER_DROPOUT,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.feature_dim = config.CONVNEXT_FEATURE_DIM
        self.sequence_length = config.SEQUENCE_LENGTH

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)

        # Spatial feature extractor; Grad-CAM hooks attach to this module.
        self.feature_extractor: nn.Sequential = convnext.features

        if freeze_backbone:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False
            logger.info("ConvNeXt backbone frozen — only LSTM + classifier will train.")

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,          # 768 channels per spatial token
            hidden_size=lstm_hidden_size,         # 256
            num_layers=lstm_num_layers,           # 2
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.classifier_dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

        self._initialize_classifier_weights()

    def _initialize_classifier_weights(self) -> None:
        """Initialize the linear head with Xavier uniform weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input batch of chest X-ray images with shape ``[B, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Unnormalized class logits with shape ``[B, num_classes]``.
        """
        # Backbone: [B, 3, 224, 224] -> [B, 768, 7, 7]
        spatial_features = self.feature_extractor(x)

        batch_size = spatial_features.size(0)

        # Sequence reshape: [B, 768, 7, 7] -> [B, 49, 768]
        #   flatten spatial dims (7*7=49), then permute to (batch, seq_len, channels)
        sequence = spatial_features.flatten(2).permute(0, 2, 1)

        # LSTM: [B, 49, 768] -> lstm_out [B, 49, 256]
        lstm_out, _ = self.lstm(sequence)

        # Final sequential hidden state from the last spatial token: [B, 256]
        final_hidden = lstm_out[:, -1, :]

        # Regularize and classify: [B, 256] -> [B, num_classes]
        dropped = self.classifier_dropout(final_hidden)
        logits = self.classifier(dropped)

        return logits

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Count model parameters.

        Parameters
        ----------
        trainable_only:
            If True, count only parameters with ``requires_grad=True``.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return class probabilities via softmax.

        Parameters
        ----------
        x:
            Input tensor with shape ``[B, 3, H, W]``.

        Returns
        -------
        torch.Tensor
            Probability distribution with shape ``[B, num_classes]``.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_model(
    device: torch.device | None = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> ConvNeXtLSTMClassifier:
    """
    Factory function to construct and optionally move the model to a device.

    Parameters
    ----------
    device:
        Target device. Defaults to ``config.DEVICE``.
    pretrained:
        Whether to load ImageNet-pretrained ConvNeXt-Tiny weights.
    freeze_backbone:
        Whether to freeze the ConvNeXt feature extractor.

    Returns
    -------
    ConvNeXtLSTMClassifier
        Initialized model in evaluation mode on the requested device.
    """
    device = device or config.DEVICE

    model = ConvNeXtLSTMClassifier(
        num_classes=config.NUM_CLASSES,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_dropout=config.LSTM_DROPOUT,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    model = model.to(device)

    total_params = model.get_num_parameters(trainable_only=False)
    trainable_params = model.get_num_parameters(trainable_only=True)
    logger.info(
        "Model built on %s | total params: %s | trainable: %s",
        device,
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    return model


def load_checkpoint(
    checkpoint_path: str | Any = None,
    device: torch.device | None = None,
    pretrained_backbone: bool = False,
) -> ConvNeXtLSTMClassifier:
    """
    Load a trained model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path:
        Path to ``.pth`` checkpoint. Defaults to ``config.BEST_MODEL_PATH``.
    device:
        Target device. Defaults to ``config.DEVICE``.
    pretrained_backbone:
        If False, the backbone is initialized without ImageNet weights before
        loading the checkpoint state (avoids downloading weights at inference).

    Returns
    -------
    ConvNeXtLSTMClassifier
        Model with loaded weights in evaluation mode.
    """
    device = device or config.DEVICE
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH

    model = build_model(
        device=device,
        pretrained=pretrained_backbone,
        freeze_backbone=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = build_model(device=config.DEVICE)
    model.eval()

    dummy_input = torch.randn(
        2,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=config.DEVICE,
    )

    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)

    print(f"Input shape        : {tuple(dummy_input.shape)}")       # [2, 3, 224, 224]
    print(f"Logits shape       : {tuple(logits.shape)}")             # [2, 2]
    print(f"Probabilities shape: {tuple(probabilities.shape)}")     # [2, 2]
    print(f"Sample probabilities: {probabilities[0].cpu().tolist()}")
