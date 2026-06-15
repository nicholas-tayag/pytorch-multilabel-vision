"""Shared project constants."""

LABELS = [
    "pen",
    "paper",
    "book",
    "clock",
    "phone",
    "laptop",
    "chair",
    "desk",
    "bottle",
    "keychain",
    "backpack",
    "calculator",
]

LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
VALID_LABELS = set(LABELS)

IMAGE_SIZE = 128
IMAGE_PATTERN = "img*.png"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
