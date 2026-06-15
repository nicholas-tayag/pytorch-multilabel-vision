#!/usr/bin/env python3
"""Config-driven training entrypoint for VisionTagger."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_tagger.config import load_config
from vision_tagger.constants import LABELS
from vision_tagger.data import TensorSubset, load_tensor_dataset, multilabel_train_val_split, tensor_eval_transform, train_transform
from vision_tagger.metrics import compute_metrics
from vision_tagger.model import build_model
from vision_tagger.thresholds import apply_thresholds, find_per_class_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VisionTagger.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--data", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    torch.manual_seed(int(data_cfg.get("seed", 42)))
    data_dir = args.data or Path(data_cfg.get("train_dir", "aggregated"))
    image_size = int(data_cfg.get("image_size", 128))
    output_dir = Path(training_cfg.get("output_dir", "runs/vision_tagger"))
    output_dir.mkdir(parents=True, exist_ok=True)

    images, targets = load_tensor_dataset(data_dir, image_size=image_size)
    images = images.float() / 255.0
    train_idx, val_idx = multilabel_train_val_split(
        targets,
        val_frac=float(data_cfg.get("val_frac", 0.15)),
        seed=int(data_cfg.get("seed", 42)),
    )

    train_ds = TensorSubset(images, targets, train_idx, transform=train_transform(image_size))
    val_ds = TensorSubset(images, targets, val_idx, transform=tensor_eval_transform())
    batch_size = int(training_cfg.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        num_labels=len(LABELS),
        dropout=float(model_cfg.get("dropout", 0.2)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    freeze_epochs = int(training_cfg.get("freeze_epochs", 0))
    if freeze_epochs > 0:
        for name, parameter in model.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False

    train_targets = targets[train_idx]
    pos = train_targets.sum(dim=0)
    neg = train_targets.shape[0] - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(neg / (pos + 1e-6)).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 2e-3)),
    )

    epochs = int(training_cfg.get("epochs", 50))
    warmup_epochs = 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs
        if epoch < warmup_epochs
        else 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / max(1, epochs - warmup_epochs))),
    )

    best_val_loss = float("inf")
    patience = int(training_cfg.get("early_stop_patience", 5))
    stale_epochs = 0

    for epoch in range(epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = True

        model.train()
        train_loss = 0.0
        train_n = 0
        for images_batch, labels_batch in train_loader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(images_batch)
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images_batch.size(0)
            train_n += images_batch.size(0)

        model.eval()
        val_loss = 0.0
        val_n = 0
        val_predictions = []
        val_targets = []
        val_probabilities = []
        with torch.no_grad():
            for images_batch, labels_batch in val_loader:
                images_batch = images_batch.to(device)
                labels_batch = labels_batch.to(device)
                logits = model(images_batch)
                probabilities = torch.sigmoid(logits)
                preds = apply_thresholds(probabilities)
                val_loss += loss_fn(logits, labels_batch).item() * images_batch.size(0)
                val_n += images_batch.size(0)
                val_predictions.append(preds.cpu())
                val_targets.append(labels_batch.cpu())
                val_probabilities.append(probabilities.cpu())

        scheduler.step()
        avg_val_loss = val_loss / max(1, val_n)
        metrics = compute_metrics(torch.cat(val_predictions), torch.cat(val_targets), loss=avg_val_loss)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss / max(1, train_n):.4f} "
            f"val_loss={avg_val_loss:.4f} "
            f"hamming={metrics['hamming_acc']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stale_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_cnn_model.pth")
            thresholds = find_per_class_thresholds(torch.cat(val_probabilities), torch.cat(val_targets))
            torch.save(thresholds, output_dir / "cnn_thresholds.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after {patience} stale epochs")
                break

    print(f"saved best checkpoint and thresholds to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
