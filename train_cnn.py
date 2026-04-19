import io
import contextlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

    n = x_all.shape[0]
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=g)
    n_val = max(1, int(n * 0.15))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tfm = transforms.Compose([
        transforms.Normalize(mean=mean, std=std),
    ])

    tr_ds = TensorSet(x_all, y_all, train_idx, train_tfm)
    va_ds = TensorSet(x_all, y_all, val_idx, val_tfm)

    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_labels=len(LABELS)).to(device)

    y_train = y_all[train_idx]
    pos = y_train.sum(dim=0)
    neg = y_train.shape[0] - pos
    pos_weight = neg / (pos + 1e-6)
    pos_weight = pos_weight.to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    for _ in range(30):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item() * x.size(0)
                val_n += x.size(0)
        scheduler.step(val_loss / max(1, val_n))

    thresholds = find_thresholds(model, va_loader, device, len(LABELS))

    torch.save(model.state_dict(), "cnn_model.pth")
    torch.save(thresholds, "cnn_thresholds.pt")
    print("cnn_model.pth")

if __name__ == "__main__":
    main()
