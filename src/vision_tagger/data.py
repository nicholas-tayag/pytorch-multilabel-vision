"""Dataset utilities for directory-encoded multi-label image data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

from .constants import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABELS,
    LABEL_TO_INDEX,
    VALID_LABELS,
)

IMAGE_RE = re.compile(r"^img\S*\.png$", re.IGNORECASE)


def parse_label_folder(folder_name: str, separator: str = "_") -> list[str] | None:
    """Return labels encoded by a folder name, or None if the name is invalid."""
    labels = [part.strip().lower() for part in folder_name.split(separator) if part]
    if not labels:
        return None
    if len(labels) != len(set(labels)):
        return None
    if any(label not in VALID_LABELS for label in labels):
        return None
    return labels


def encode_labels(labels: Iterable[str]) -> torch.Tensor:
    """Convert an iterable of label names into a 12-dimensional multi-hot vector."""
    target = torch.zeros(len(LABELS), dtype=torch.float32)
    for label in labels:
        target[LABEL_TO_INDEX[label]] = 1.0
    return target


def eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.1),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]
    )


def tensor_eval_transform() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class DirectoryMultilabelDataset(Dataset):
    """Loads images from folders whose names encode multi-label targets."""

    def __init__(self, root: str | Path, transform=None, separator: str = "_"):
        self.root = Path(root)
        self.transform = transform
        self.separator = separator
        self.classes = LABELS
        self.samples: list[tuple[Path, torch.Tensor]] = []

        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue
            labels = parse_label_folder(subdir.name, separator=separator)
            if labels is None:
                continue
            target = encode_labels(labels)
            for path in sorted(subdir.iterdir()):
                if path.is_file() and IMAGE_RE.match(path.name):
                    self.samples.append((path, target.clone()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, target = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class TensorSubset(Dataset):
    """Dataset view over in-memory image and target tensors."""

    def __init__(self, images: torch.Tensor, targets: torch.Tensor, indices: torch.Tensor, transform=None):
        self.images = images
        self.targets = targets
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_idx = self.indices[idx]
        image = self.images[source_idx]
        target = self.targets[source_idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def load_tensor_dataset(root: str | Path, image_size: int = IMAGE_SIZE, shuffle: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the training layout into image and label tensors."""
    root_path = Path(root)
    images = []
    targets = []
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue
        labels = parse_label_folder(subdir.name)
        if labels is None:
            continue
        target = encode_labels(labels)
        for image_path in sorted(subdir.iterdir()):
            if not image_path.is_file() or not IMAGE_RE.match(image_path.name):
                continue
            image = read_image(str(image_path.resolve()))
            if image.shape[0] != 3 or image.shape[1] != image_size or image.shape[2] != image_size:
                image = transforms.functional.resize(image, [image_size, image_size])
            images.append(image)
            targets.append(target.clone())

    if not images:
        raise ValueError(f"No valid images found in {root_path}")

    image_tensor = torch.stack(images)
    target_tensor = torch.stack(targets)
    if shuffle:
        order = torch.randperm(len(image_tensor))
        image_tensor = image_tensor[order]
        target_tensor = target_tensor[order]
    return image_tensor, target_tensor


def multilabel_train_val_split(targets: torch.Tensor, val_frac: float = 0.15, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a validation split that keeps each label represented when possible."""
    n = targets.shape[0]
    n_val = max(1, int(n * val_frac))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    val_mask = torch.zeros(n, dtype=torch.bool)

    positives = targets.sum(dim=0).long()
    desired = torch.clamp((positives.float() * val_frac).round().long(), min=1)
    current = torch.zeros_like(desired)

    for index in order:
        if val_mask.sum().item() >= n_val:
            break
        row_labels = targets[index].bool()
        if ((current < desired) & row_labels).any():
            val_mask[index] = True
            current += row_labels.long()

    if val_mask.sum().item() < n_val:
        remaining = order[~val_mask[order]]
        val_mask[remaining[: n_val - val_mask.sum().item()]] = True

    all_indices = torch.arange(n)
    return all_indices[~val_mask], all_indices[val_mask]


def split_train_val_test(
    targets: torch.Tensor,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remaining_idx, test_idx = multilabel_train_val_split(targets, val_frac=test_frac, seed=seed)
    adjusted_val_frac = val_frac / max(1e-8, 1.0 - test_frac)
    train_rel, val_rel = multilabel_train_val_split(targets[remaining_idx], val_frac=adjusted_val_frac, seed=seed)
    return remaining_idx[train_rel], remaining_idx[val_rel], test_idx
