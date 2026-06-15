#!/usr/bin/env python3
"""Run VisionTagger predictions for one image or a folder of images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_tagger.config import load_config
from vision_tagger.constants import IMAGE_SIZE
from vision_tagger.inference import predict_image


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    images: list[Path] = []
    for pattern in patterns:
        images.extend(sorted(path.glob(pattern)))
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict desk objects in image(s).")
    parser.add_argument("--image", type=Path, required=True, help="Image file or folder of images.")
    parser.add_argument("--model", type=Path, default=None, help="Checkpoint path.")
    parser.add_argument("--thresholds", type=Path, default=None, help="Optional per-class thresholds file.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--threshold", type=float, default=None, help="Global threshold when no thresholds file is used.")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    inference_cfg = config.get("inference", {})
    data_cfg = config.get("data", {})

    model_path = args.model or Path(inference_cfg.get("model_path", "artifacts/best_cnn_model.pth"))
    thresholds_path = args.thresholds or inference_cfg.get("thresholds_path")
    threshold = args.threshold if args.threshold is not None else float(inference_cfg.get("threshold", 0.5))
    image_size = args.image_size or int(data_cfg.get("image_size", IMAGE_SIZE))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. See docs/ARTIFACTS.md for artifact setup."
        )

    results = []
    for image_path in iter_images(args.image):
        rows = predict_image(
            image_path=image_path,
            model_path=model_path,
            thresholds_path=thresholds_path,
            threshold=threshold,
            image_size=image_size,
        )
        results.append({"image": str(image_path), "predictions": rows})

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for result in results:
        print(f"\n{result['image']}")
        print("label        probability  predicted")
        print("-----------  -----------  ---------")
        for row in result["predictions"]:
            print(f"{row['label']:<11}  {row['probability']:<11.4f}  {row['predicted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
