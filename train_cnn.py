import io
import contextlib
import math
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from load_data import load_data

LABELS = ["pen", "paper", "book", "clock", "phone", "laptop", "chair", "desk", "bottle", "keychain", "backpack", "calculator"]

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

def build_model(num_labels=12):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_labels)
    return m

class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = (1 - p_t) ** self.gamma * bce
        return loss.mean()

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

def save_checkpoint(model, opt, scheduler, epoch, best_metric, best_metric_value, best_epoch, epochs_since_improve, path):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": scheduler.state_dict(),
        "best_metric": best_metric,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "epochs_since_improve": epochs_since_improve,
    }
    torch.save(ckpt, path)

def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device)
    except Exception:
        return None

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

    return avg_loss, per_class_acc, per_class_prec, per_class_rec, per_class_f1, exact_match, hamming, macro_f1

def find_thresholds(model, loader, device, num_labels):
    model.eval()
    probs_all = []
    labels_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu()
            probs_all.append(probs)
            labels_all.append(y)
    probs = torch.cat(probs_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    thresholds = torch.zeros(num_labels)
    grid = torch.linspace(0.05, 0.95, 19)
    for c in range(num_labels):
        best_t = 0.5
        best_f1 = -1.0
        y_true = labels[:, c]
        for t in grid:
            y_pred = (probs[:, c] >= t).float()
            tp = ((y_pred == 1) & (y_true == 1)).sum().float()
            fp = ((y_pred == 1) & (y_true == 0)).sum().float()
            fn = ((y_pred == 0) & (y_true == 1)).sum().float()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[c] = best_t
    return thresholds

def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        x_all, y_all = load_data("aggregated")

    x_all = x_all.float() / 255.0

    image_size = 128
    use_focal = True
    focal_gamma = 2.0
    base_lr = 3e-4
    weight_decay = 5e-4
    warmup_epochs = 2
    num_epochs = 80
    batch_size = 32
    checkpoint_metric = "macro_f1"
    checkpoint_dir = "checkpoints"
    checkpoint_every = 5
    early_stop_patience = 10
    resume_from = None
    pin_memory = torch.cuda.is_available()

    train_idx, val_idx = multilabel_train_val_split(y_all, val_frac=0.15, seed=42)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.1),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    val_tfm = transforms.Compose([
        transforms.Normalize(mean=mean, std=std),
    ])

    tr_ds = TensorSet(x_all, y_all, train_idx, train_tfm)
    va_ds = TensorSet(x_all, y_all, val_idx, val_tfm)

    train_labels = y_all[train_idx].int()
    combo_counts = {}
    for row in train_labels:
        key = tuple(row.tolist())
        combo_counts[key] = combo_counts.get(key, 0) + 1

    weights = []
    for row in train_labels:
        key = tuple(row.tolist())
        weights.append(1.0 / combo_counts[key])

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=2, pin_memory=pin_memory)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_labels=len(LABELS)).to(device)

    y_train = y_all[train_idx]
    pos = y_train.sum(dim=0)
    neg = y_train.shape[0] - pos
    pos_weight = neg / (pos + 1e-6)
    pos_weight = pos_weight.to(device)

    if use_focal:
        loss_fn = FocalLoss(pos_weight=pos_weight, gamma=focal_gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 0
    best_metric = -float("inf")
    best_metric_value = None
    best_epoch = -1
    epochs_since_improve = 0

    if resume_from is not None and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from, device)
        if ckpt is not None and isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["opt_state"])
            scheduler.load_state_dict(ckpt["sched_state"])
            start_epoch = ckpt["epoch"] + 1
            best_metric = ckpt.get("best_metric", best_metric)
            best_metric_value = ckpt.get("best_metric_value", best_metric_value)
            best_epoch = ckpt.get("best_epoch", best_epoch)
            epochs_since_improve = ckpt.get("epochs_since_improve", 0)
            print(f"resuming from epoch {start_epoch}")
        else:
            print("resume checkpoint missing model_state, skipping resume")

    train_loss_hist = []
    val_loss_hist = []
    exact_match_hist = []
    hamming_hist = []
    macro_f1_hist = []
    per_class_acc_hist = []
    per_class_f1_hist = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_n = 0

        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            train_loss += loss.item() * x.size(0)
            train_n += x.size(0)

        train_loss = train_loss / max(1, train_n)

        val_loss, per_class_acc, per_class_prec, per_class_rec, per_class_f1, exact_match, hamming, macro_f1 = evaluate_epoch(
            model, va_loader, device, loss_fn, threshold=0.5
        )

        scheduler.step()

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        exact_match_hist.append(exact_match)
        hamming_hist.append(hamming)
        macro_f1_hist.append(macro_f1)
        per_class_acc_hist.append(per_class_acc)
        per_class_f1_hist.append(per_class_f1)

        if checkpoint_metric == "val_loss":
            current_metric = -val_loss
            display_metric = val_loss
        elif checkpoint_metric == "macro_f1":
            current_metric = macro_f1
            display_metric = macro_f1
        elif checkpoint_metric == "exact_match":
            current_metric = exact_match
            display_metric = exact_match
        elif checkpoint_metric == "hamming":
            current_metric = hamming
            display_metric = hamming
        else:
            raise ValueError(f"unknown checkpoint_metric: {checkpoint_metric}")

        if current_metric > best_metric:
            best_metric = current_metric
            best_metric_value = display_metric
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save(model.state_dict(), "best_cnn_model.pth")
            save_checkpoint(
                model,
                opt,
                scheduler,
                epoch,
                best_metric,
                best_metric_value,
                best_epoch,
                epochs_since_improve,
                os.path.join(checkpoint_dir, "best.pt"),
            )
        else:
            epochs_since_improve += 1

        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                model,
                opt,
                scheduler,
                epoch,
                best_metric,
                best_metric_value,
                best_epoch,
                epochs_since_improve,
                os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt"),
            )

        save_checkpoint(
            model,
            opt,
            scheduler,
            epoch,
            best_metric,
            best_metric_value,
            best_epoch,
            epochs_since_improve,
            os.path.join(checkpoint_dir, "last.pt"),
        )

        print(
            f"epoch {epoch + 1}/{num_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"exact_match={exact_match:.4f} "
            f"hamming={hamming:.4f} "
            f"macro_f1={macro_f1:.4f}"
        )

        if early_stop_patience is not None and epochs_since_improve >= early_stop_patience:
            print(f"early stop at epoch {epoch + 1} with no improvement for {early_stop_patience} epochs")
            break

    if best_epoch >= 0:
        model.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
        print(f"best checkpoint at epoch {best_epoch} with {checkpoint_metric}={best_metric_value:.4f}")

    epochs = list(range(1, len(train_loss_hist) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss_hist, label="train")
    plt.plot(epochs, val_loss_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curves.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, exact_match_hist, label="exact_match")
    plt.plot(epochs, hamming_hist, label="hamming")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("multi-label accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("multilabel_accuracy.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, macro_f1_hist, label="macro_f1")
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.title("macro f1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("macro_f1.png")
    plt.close()

    per_class_matrix = torch.stack(per_class_acc_hist, dim=0).numpy()
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(LABELS):
        plt.plot(epochs, per_class_matrix[:, i], label=label)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("per-class accuracy")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("per_class_accuracy.png")
    plt.close()



    per_class_f1_matrix = torch.stack(per_class_f1_hist, dim=0).numpy()
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(LABELS):
        plt.plot(epochs, per_class_f1_matrix[:, i], label=label)
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.title("per-class f1")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("per_class_f1.png")
    plt.close()

    final_per_class = per_class_acc_hist[-1]
    final_per_class_f1 = per_class_f1_hist[-1]
    print("final per-class accuracy:")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {final_per_class[i].item():.4f}")

    print("final per-class f1:")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {final_per_class_f1[i].item():.4f}")

    thresholds = find_thresholds(model, va_loader, device, len(LABELS))

    torch.save(model.state_dict(), "cnn_model.pth")
    torch.save(thresholds, "cnn_thresholds.pt")
    print("cnn_model.pth")

if __name__ == "__main__":
    main()
