"""Model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .constants import LABELS


def build_model(num_labels: int = len(LABELS), dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_labels),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model


def _build_for_state_dict(state_dict: dict[str, torch.Tensor], dropout: float, pretrained: bool) -> nn.Module:
    has_plain_head = "fc.weight" in state_dict and "fc.bias" in state_dict
    if has_plain_head:
        return build_model(dropout=0.0, pretrained=pretrained)
    return build_model(dropout=dropout, pretrained=pretrained)


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dropout: float = 0.2,
    pretrained: bool = False,
) -> nn.Module:
    """Load a VisionTagger-compatible ResNet-18 checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)!r}")

    model = _build_for_state_dict(state, dropout=dropout, pretrained=pretrained).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
