import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from load_data import load_data
from train_cnn import build_model

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


def load_thresholds(path, num_labels, device):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    t = torch.load(p, map_location=device)
    if isinstance(t, torch.Tensor) and t.numel() == num_labels:
        return t.view(-1).to(device)
    return None


def compute_metrics(preds, labels):
    preds = preds.float()
    labels = labels.float()

    exact_match = (preds == labels).all(dim=1).float().mean().item()
    hamming_acc = (preds == labels).float().mean().item()

    intersection = (preds * labels).sum(dim=1)
    union = ((preds + labels) > 0).float().sum(dim=1)
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    mean_iou = iou.mean().item()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    precision_micro = (tp / (tp + fp + 1e-8)).item()
    recall_micro = (tp / (tp + fn + 1e-8)).item()
    f1_micro = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    tp_c = ((preds == 1) & (labels == 1)).sum(dim=0).float()
    fp_c = ((preds == 1) & (labels == 0)).sum(dim=0).float()
    fn_c = ((preds == 0) & (labels == 1)).sum(dim=0).float()
    prec_c = tp_c / (tp_c + fp_c + 1e-8)
    rec_c = tp_c / (tp_c + fn_c + 1e-8)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
    macro_f1 = f1_c.mean().item()

    return {
        "exact_match": exact_match,
        "hamming_acc": hamming_acc,
        "mean_iou": mean_iou,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "macro_f1": macro_f1,
    }


def predict_model(model, loader, device, threshold=0.5, thresholds=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            if thresholds is not None:
                thr = thresholds.view(1, -1)
                preds = (probs >= thr).float()
            else:
                preds = (probs >= threshold).float()

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return preds, labels


def eval_baselines(y_train, y_test):
    combo_counts = {}
    for row in y_train:
        key = tuple(row.tolist())
        combo_counts[key] = combo_counts.get(key, 0) + 1
    mode_key = max(combo_counts.items(), key=lambda kv: kv[1])[0]
    mode_vec = torch.tensor(mode_key, dtype=y_test.dtype).view(1, -1)
    mode_preds = mode_vec.repeat(y_test.shape[0], 1)
    mode_metrics = compute_metrics(mode_preds, y_test)

    return mode_metrics


def eval_model_path(model_path, loader, device, thresholds, threshold, label, dropout):
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        print(f"[warn] {label} path not found: {model_path}")
        return None

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    has_plain_fc = "fc.weight" in state and "fc.bias" in state
    has_dropout_fc = "fc.1.weight" in state and "fc.1.bias" in state
    if has_plain_fc and not has_dropout_fc:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        model = model.to(device)
    else:
        model = build_model(num_labels=len(LABELS), dropout=dropout, use_pretrained=False).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        model = model.to(device)
        model.load_state_dict(state)

    preds, labels = predict_model(model, loader, device, threshold=threshold, thresholds=thresholds)
    metrics = compute_metrics(preds, labels)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate baselines and models on test split")
    parser.add_argument("--data_dir", type=str, default="aggregated", help="dataset directory")
    parser.add_argument("--test_frac", type=float, default=0.15, help="test split fraction")
    parser.add_argument("--val_frac", type=float, default=0.15, help="val split fraction")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--threshold", type=float, default=0.5, help="global threshold if no per-class thresholds")
    parser.add_argument("--thresholds_path", type=str, default="cnn_thresholds.pt", help="per-class thresholds path")
    parser.add_argument("--baseline_model_path", type=str, default=None, help="Path to ResNet baseline model")
    parser.add_argument("--baseline_dropout", type=float, default=0.2, help="Dropout for baseline model head")
    parser.add_argument("--our_model_path", type=str, default=None, help="Path to our best model")
    parser.add_argument("--our_dropout", type=float, default=0.2, help="Dropout for our model head")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_all, y_all = load_data(args.data_dir)
    x_all = x_all.float() / 255.0

    train_idx, val_idx, test_idx = split_train_val_test(
        y_all, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed
    )

    tfm = transforms.Normalize(mean=MEAN, std=STD)
    test_ds = TensorSet(x_all, y_all, test_idx, tfm=tfm)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    print(f"train size: {len(train_idx)} | val size: {len(val_idx)} | test size: {len(test_idx)}")

    mode_metrics = eval_baselines(y_train, y_test)

    thresholds = load_thresholds(args.thresholds_path, len(LABELS), device)
    if thresholds is None:
        print("[info] using global threshold:", args.threshold)
    else:
        print("[info] loaded per-class thresholds from:", args.thresholds_path)

    baseline_metrics = eval_model_path(
        args.baseline_model_path, test_loader, device, thresholds, args.threshold, "baseline", args.baseline_dropout
    )
    our_metrics = eval_model_path(
        args.our_model_path, test_loader, device, thresholds, args.threshold, "our", args.our_dropout
    )


    def print_metrics(name, metrics):
        if metrics is None:
            return
        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print_metrics("mode_labelset_baseline", mode_metrics)
    print_metrics("baseline_model", baseline_metrics)
    print_metrics("our_model", our_metrics)


if __name__ == "__main__":
    main()
