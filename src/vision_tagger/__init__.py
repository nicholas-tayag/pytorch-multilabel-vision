"""VisionTagger package for multi-label desk object recognition."""

from .constants import IMAGE_SIZE, LABELS
from .inference import predict_image
from .model import build_model, load_model

__all__ = ["IMAGE_SIZE", "LABELS", "build_model", "load_model", "predict_image"]
