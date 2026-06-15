"""Inference helpers for images and folders."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .constants import IMAGE_SIZE, LABELS
from .data import eval_transform
from .model import load_model
from .thresholds import apply_thresholds, load_thresholds


def preprocess_image(path: str | Path, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return eval_transform(image_size)(image).unsqueeze(0)


def predict_tensor(
    model: torch.nn.Module,
    batch: torch.Tensor,
    threshold: float = 0.5,
    thresholds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.sigmoid(logits)
        predictions = apply_thresholds(probabilities, threshold=threshold, thresholds=thresholds)
    return predictions.cpu(), probabilities.cpu()


def predict_image(
    image_path: str | Path,
    model_path: str | Path,
    thresholds_path: str | Path | None = None,
    threshold: float = 0.5,
    image_size: int = IMAGE_SIZE,
    device: str | torch.device | None = None,
) -> list[dict[str, float | bool | str]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device=device)
    thresholds = load_thresholds(thresholds_path, len(LABELS), device=device)
    batch = preprocess_image(image_path, image_size=image_size).to(device)
    predictions, probabilities = predict_tensor(model, batch, threshold=threshold, thresholds=thresholds)
    return format_prediction(predictions[0], probabilities[0])


def format_prediction(prediction: torch.Tensor, probabilities: torch.Tensor) -> list[dict[str, float | bool | str]]:
    rows = []
    for label, predicted, probability in zip(LABELS, prediction.tolist(), probabilities.tolist(), strict=True):
        rows.append(
            {
                "label": label,
                "probability": round(float(probability), 4),
                "predicted": bool(predicted),
            }
        )
    return sorted(rows, key=lambda row: row["probability"], reverse=True)
