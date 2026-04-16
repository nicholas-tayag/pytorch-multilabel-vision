# Load packages we need
import sys
import os
import time

import numpy as np
import pandas as pd
import sklearn

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.io import read_image

from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
import re

from sklearn.model_selection import train_test_split

from torch.nn import Linear, Conv2d, MaxPool2d, Dropout, Flatten, ReLU
from sklearn.model_selection import train_test_split

from pathlib import Path

LABEL_ORDER = ["pen", "paper", "book", "clock", "phone", "laptop", "chair", "desk", "bottle", "keychain", "backpack", "calculator"]
VALID_LABELS = set(LABEL_ORDER)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_ORDER)}
IMG_RE = re.compile(r"^img(\S+)\.png$", re.IGNORECASE)
def load_data(directory):
    starting_point = Path(directory)
    data = []
    labels = []
    print(f"Starting to load data from {starting_point}")
    for subdir in starting_point.iterdir():
        if not subdir.is_dir():
            print(f"Skipping {subdir} because it is not a directory.")
            continue
        subdir_labels = subdir.name.split('_')
        if not all(label in VALID_LABELS for label in subdir_labels):
            print(f"Skipping directory {subdir} due to invalid labels: {subdir_labels}")
            continue
        target = torch.zeros(12, dtype=torch.float32)
        for i, label in enumerate(LABEL_ORDER):
            if label in subdir_labels:
                target[i] = 1.0
        for img_file in subdir.iterdir():
            if img_file.is_file():
                match = IMG_RE.match(img_file.name)
                if match:
                    img_path = img_file.resolve()
                    img = read_image(str(img_path))
                    if img.shape[0] != 3 or img.shape[1] != 128 or img.shape[2] != 128:
                        print(f"{img_file} had an invalid invalid image shape: {img.shape}")
                        img = transforms.functional.resize(img, [128, 128])
                    #NOTE: I currently left the actual data as uint8 to save memory, but we might want to convert it to float later
                    data.append(img)
                    labels.append(target.clone())
                else:
                    print(f"Skipping file {img_file} due to invalid name: {img_file.name}")
    images_tensor = torch.stack(data)
    labels_tensor = torch.stack(labels)
    perm = torch.randperm(len(images_tensor))
    images_tensor = images_tensor[perm]
    labels_tensor = labels_tensor[perm]
    return images_tensor, labels_tensor
