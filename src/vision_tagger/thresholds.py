"""Decision threshold utilities."""

from __future__ import annotations

from pathlib import Path

import torch


def load_thresholds(path: str | Path | None, num_labels: int, device: str | torch.device | None = None) -> torch.Tensor | None:
    if path is None:
        return None
    threshold_path = Path(path)
    if not threshold_path.exists():
        return None
    thresholds = torch.load(threshold_path, map_location=device or "cpu")
    if isinstance(thresholds, torch.Tensor) and thresholds.numel() == num_labels:
        return thresholds.view(-1)
    raise ValueError(f"Threshold file {threshold_path} does not contain {num_labels} values")


def apply_thresholds(probabilities: torch.Tensor, threshold: float = 0.5, thresholds: torch.Tensor | None = None) -> torch.Tensor:
    if thresholds is not None:
        return (probabilities >= thresholds.to(probabilities.device).view(1, -1)).float()
    return (probabilities >= threshold).float()


def find_per_class_thresholds(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Grid-search per-class thresholds that maximize validation F1."""
    thresholds = torch.zeros(probabilities.shape[1])
    coarse_grid = torch.linspace(0.05, 0.95, 37)
    for class_idx in range(probabilities.shape[1]):
        best_threshold = torch.tensor(0.5)
        best_f1 = -1.0
        y_true = targets[:, class_idx]
        for threshold in coarse_grid:
            y_pred = (probabilities[:, class_idx] >= threshold).float()
            tp = ((y_pred == 1) & (y_true == 1)).sum().float()
            fp = ((y_pred == 1) & (y_true == 0)).sum().float()
            fn = ((y_pred == 0) & (y_true == 1)).sum().float()
            f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[class_idx] = best_threshold
    return thresholds
