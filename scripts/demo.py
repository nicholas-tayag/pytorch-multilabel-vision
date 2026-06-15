#!/usr/bin/env python3
"""Launch a local VisionTagger Gradio demo."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_tagger.config import load_config
from vision_tagger.inference import predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VisionTagger local demo.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Install demo dependencies with `uv sync --extra demo`.") from exc

    args = parse_args()
    config = load_config(args.config)
    inference_cfg = config.get("inference", {})
    data_cfg = config.get("data", {})
    model_path = args.model or Path(inference_cfg.get("model_path", "artifacts/best_cnn_model.pth"))
    thresholds_path = args.thresholds or inference_cfg.get("thresholds_path")
    threshold = float(inference_cfg.get("threshold", 0.5))
    image_size = int(data_cfg.get("image_size", 128))

    def classify(image):
        if image is None:
            return []
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            image.save(handle.name)
            rows = predict_image(
                handle.name,
                model_path=model_path,
                thresholds_path=thresholds_path,
                threshold=threshold,
                image_size=image_size,
            )
        return [[row["label"], row["probability"], row["predicted"]] for row in rows]

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Desk image"),
        outputs=gr.Dataframe(headers=["Label", "Probability", "Predicted"], label="VisionTagger predictions"),
        title="VisionTagger",
        description="Upload a desk or study-space image to inspect multi-label object predictions.",
    )
    demo.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
