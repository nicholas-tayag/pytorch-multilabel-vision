import argparse
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from load_data import load_data

LABELS = [
    "pen", "paper", "book", "clock", "phone", "laptop",
    "chair", "desk", "bottle", "keychain", "backpack", "calculator"
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TensorSet(Dataset):
    def __init__(self, x, y, idx, tfm=None):
        self.x = x
        self.y = y
        self.idx = idx
        self.tfm = tfm

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.x[j]
        y = self.y[j]
        if self.tfm is not None:
            x = self.tfm(x)
        return x, y


def build_model(num_labels=12, dropout=0.0):
    m = models.resnet18(weights=None)
    if dropout > 0:
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(m.fc.in_features, num_labels),
        )
    else:
        m.fc = nn.Linear(m.fc.in_features, num_labels)
    return m


def multilabel_train_val_split(y, val_frac=0.15, seed=42):
    n = y.shape[0]
    n_val = max(1, int(n * val_frac))
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g)
    val_mask = torch.zeros(n, dtype=torch.bool)

    pos = y.sum(dim=0).long()
    target = torch.clamp((pos.float() * val_frac).round().long(), min=1)
    current = torch.zeros_like(target)

    for i in order:
        if val_mask.sum().item() >= n_val:
            break
        labels = y[i].bool()
        if ((current < target) & labels).any():
            val_mask[i] = True
            current += labels.long()

    if val_mask.sum().item() < n_val:
        need = n_val - val_mask.sum().item()
        remaining = order[~val_mask[order]]
        extra = remaining[:need]
        val_mask[extra] = True

    train_idx = torch.arange(n)[~val_mask]
    val_idx = torch.arange(n)[val_mask]
    return train_idx, val_idx


def split_train_val_test(y, test_frac=0.15, val_frac=0.15, seed=42):
    remaining_idx, test_idx = multilabel_train_val_split(y, val_frac=test_frac, seed=seed)
    y_remaining = y[remaining_idx]
    adj_val_frac = val_frac / max(1e-8, (1.0 - test_frac))
    train_idx_rel, val_idx_rel = multilabel_train_val_split(y_remaining, val_frac=adj_val_frac, seed=seed)
    train_idx = remaining_idx[train_idx_rel]
    val_idx = remaining_idx[val_idx_rel]
    return train_idx, val_idx, test_idx


def evaluate_epoch(model, loader, device, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    avg_loss = total_loss / max(1, total_n)
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    per_class_acc = (preds == labels).float().mean(dim=0)
    exact_match = (preds == labels).all(dim=1).float().mean().item()
    hamming = (preds == labels).float().mean().item()

    tp = ((preds == 1) & (labels == 1)).sum(dim=0).float()
    fp = ((preds == 1) & (labels == 0)).sum(dim=0).float()
    fn = ((preds == 0) & (labels == 1)).sum(dim=0).float()

    per_class_prec = tp / (tp + fp + 1e-8)
    per_class_rec = tp / (tp + fn + 1e-8)
    per_class_f1 = 2 * per_class_prec * per_class_rec / (per_class_prec + per_class_rec + 1e-8)
    macro_f1 = per_class_f1.mean().item()

    return avg_loss, per_class_acc, per_class_f1, exact_match, hamming, macro_f1


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Scratch ResNet-18 baseline (no aug, no pos_weight)")
    parser.add_argument("--data_dir", type=str, default="aggregated")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_path", type=str, default="baseline_split.pt")
    parser.add_argument("--save_path", type=str, default="baseline_resnet18_scratch.pth")
    args = parser.parse_args()

    set_seed(args.seed)

    x_all, y_all = load_data(args.data_dir)
    x_all = x_all.float() / 255.0

    if args.split_path and os.path.exists(args.split_path):
        split = torch.load(args.split_path, map_location="cpu")
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
    else:
        train_idx, val_idx, test_idx = split_train_val_test(
            y_all, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed
        )
        if args.split_path:
            torch.save(
                {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx},
                args.split_path
            )

    print(f"train size: {len(train_idx)} | val size: {len(val_idx)} | test size: {len(test_idx)}")

    tfm = transforms.Normalize(mean=MEAN, std=STD)
    tr_ds = TensorSet(x_all, y_all, train_idx, tfm=tfm)
    va_ds = TensorSet(x_all, y_all, val_idx, tfm=tfm)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_labels=len(LABELS), dropout=0.0).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_n = 0

        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            train_loss += loss.item() * x.size(0)
            train_n += x.size(0)

        train_loss /= max(1, train_n)

        val_loss, per_class_acc, per_class_f1, exact_match, hamming, macro_f1 = evaluate_epoch(
            model, va_loader, device, loss_fn, threshold=0.5
        )

        if macro_f1 > best_metric:
            best_metric = macro_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.save_path)

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"exact_match={exact_match:.4f} "
            f"hamming={hamming:.4f} "
            f"macro_f1={macro_f1:.4f}"
        )

    print(f"best checkpoint at epoch {best_epoch} with macro_f1={best_metric:.4f}")
    print(f"saved: {args.save_path}")


if __name__ == "__main__":
    main()
