"""Multi-label evaluation metrics."""

from __future__ import annotations

import torch


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, loss: float | None = None) -> dict[str, float]:
    predictions = predictions.float()
    targets = targets.float()

    exact_match = (predictions == targets).all(dim=1).float().mean().item()
    hamming_acc = (predictions == targets).float().mean().item()

    intersection = (predictions * targets).sum(dim=1)
    union = ((predictions + targets) > 0).float().sum(dim=1)
    mean_iou = torch.where(union > 0, intersection / union, torch.ones_like(union)).mean().item()

    tp = ((predictions == 1) & (targets == 1)).sum().float()
    fp = ((predictions == 1) & (targets == 0)).sum().float()
    fn = ((predictions == 0) & (targets == 1)).sum().float()

    precision_micro = (tp / (tp + fp + 1e-8)).item()
    recall_micro = (tp / (tp + fn + 1e-8)).item()
    f1_micro = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    tp_c = ((predictions == 1) & (targets == 1)).sum(dim=0).float()
    fp_c = ((predictions == 1) & (targets == 0)).sum(dim=0).float()
    fn_c = ((predictions == 0) & (targets == 1)).sum(dim=0).float()
    precision_c = tp_c / (tp_c + fp_c + 1e-8)
    recall_c = tp_c / (tp_c + fn_c + 1e-8)
    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-8)

    metrics = {
        "exact_match": exact_match,
        "hamming_acc": hamming_acc,
        "mean_iou": mean_iou,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "macro_f1": f1_c.mean().item(),
    }
    if loss is not None:
        metrics["loss"] = loss
    return metrics


def per_class_f1(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = predictions.float()
    targets = targets.float()
    tp = ((predictions == 1) & (targets == 1)).sum(dim=0).float()
    fp = ((predictions == 1) & (targets == 0)).sum(dim=0).float()
    fn = ((predictions == 0) & (targets == 1)).sum(dim=0).float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)
