"""EmotionCNN — lightweight CNN for mel-spectrogram emotion classification.

Architecture:
    Input:  (batch, 1, 128, 128)  ← grayscale mel-spectrogram
    Block1: Conv2d(1→32,3x3)  → BN → ReLU → MaxPool(2x2)
    Block2: Conv2d(32→64,3x3) → BN → ReLU → MaxPool(2x2)
    Block3: Conv2d(64→128,3x3)→ BN → ReLU → MaxPool(2x2)
    GlobalAvgPool → Flatten → (128,)
    FC: Linear(128→256) → ReLU → Dropout(0.5) → Linear(256→5)
    Output: (batch, 5) logits
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


EMOTIONS: list[str] = ["happy", "sad", "angry", "fear", "neutral"]
NUM_CLASSES: int = len(EMOTIONS)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionCNN(nn.Module):
    """CNN that classifies emotion from a (1, 128, 128) mel-spectrogram."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1, 32),    # → (32, 64, 64)
            _ConvBlock(32, 64),   # → (64, 32, 32)
            _ConvBlock(64, 128),  # → (128, 16, 16)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # → (128, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)


def load_cnn_model(
    path: str | Path,
    device: torch.device | Literal["cpu", "cuda"] | None = None,
) -> EmotionCNN:
    """Load a saved EmotionCNN checkpoint.

    Args:
        path: Path to the ``.pt`` checkpoint saved by ``train_cnn.py``.
        device: Target device.  Auto-selects CUDA if available when *None*.

    Returns:
        EmotionCNN in eval mode on the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # Support both raw state-dict and wrapped {"model_state": ..., ...} dicts
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    num_classes = checkpoint.get("num_classes", NUM_CLASSES) if isinstance(checkpoint, dict) else NUM_CLASSES

    model = EmotionCNN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
