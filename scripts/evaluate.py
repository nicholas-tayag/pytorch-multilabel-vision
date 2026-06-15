#!/usr/bin/env python3
"""Evaluate a VisionTagger checkpoint on a directory-encoded dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_tagger.config import load_config
from vision_tagger.constants import IMAGE_SIZE, LABELS
from vision_tagger.data import DirectoryMultilabelDataset, eval_transform
from vision_tagger.metrics import compute_metrics
from vision_tagger.model import load_model
from vision_tagger.thresholds import apply_thresholds, load_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisionTagger.")
    parser.add_argument("--data", type=Path, default=None, help="Evaluation dataset directory.")
    parser.add_argument("--model", type=Path, default=None, help="Checkpoint path.")
    parser.add_argument("--thresholds", type=Path, default=None, help="Optional per-class thresholds.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})
    training_cfg = config.get("training", {})

    data_dir = args.data or Path(data_cfg.get("test_dir", "project_test_data"))
    model_path = args.model or Path(inference_cfg.get("model_path", "artifacts/best_cnn_model.pth"))
    thresholds_path = args.thresholds or inference_cfg.get("thresholds_path")
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 32))
    image_size = int(data_cfg.get("image_size", IMAGE_SIZE))
    threshold = float(inference_cfg.get("threshold", 0.5))

    dataset = DirectoryMultilabelDataset(data_dir, transform=eval_transform(image_size))
    if len(dataset) == 0:
        raise ValueError(f"No valid evaluation samples found in {data_dir}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)
    thresholds = load_thresholds(thresholds_path, len(LABELS), device=device)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_n = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            preds = apply_thresholds(probabilities, threshold=threshold, thresholds=thresholds)
            total_loss += criterion(logits, labels).item() * images.size(0)
            total_n += images.size(0)
            predictions.append(preds.cpu())
            targets.append(labels.cpu())

    metrics = compute_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        loss=total_loss / max(1, total_n),
    )

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
