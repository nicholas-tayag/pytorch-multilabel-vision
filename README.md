# PyTorch Multilabel Vision

PyTorch Multilabel Vision is a computer vision training pipeline for detecting
multiple objects in a single image. It uses ResNet-18 transfer learning,
multi-label loss functions, class-imbalance weighting, per-class threshold
tuning, and baseline comparisons across 12 object categories.

## Highlights

- Multi-label image classification for 12 object categories.
- Custom directory-based dataset loader for folder names such as
  `pen_paper_book`.
- ResNet-18 transfer learning with a dropout classification head.
- Training-time augmentation with crop, flip, rotation, color jitter, and random
  erasing.
- Class-imbalance handling with `BCEWithLogitsLoss(pos_weight=...)`.
- Validation metrics for exact match, hamming accuracy, macro F1, and per-class
  F1.
- Per-class threshold search to improve multi-label decision quality.
- Scratch ResNet and mode-labelset baselines for comparison.

## Labels

The model predicts any combination of:

```text
pen, paper, book, clock, phone, laptop,
chair, desk, bottle, keychain, backpack, calculator
```

## Dataset Layout

Training and evaluation data should use this structure:

```text
aggregated/
  pen/
    img001.png
  pen_paper/
    img002.png
  laptop_phone_book/
    img003.png
```

Each folder name is split on `_`, and each valid label becomes one active entry
in the target vector. Images are loaded as RGB tensors and normalized to
ImageNet statistics for ResNet-18.

## Install

Install uv and sync dependencies:

```bash
pip install uv
uv sync
```

## Train

Run the main transfer-learning model:

```bash
python train_cnn.py
```

Run the scratch ResNet-18 baseline:

```bash
python train_baseline.py
```

## Evaluate

Evaluate a trained checkpoint on a compatible test directory:

```bash
python eval.py \
  --model_path best_cnn_model.pth \
  --test_data project_test_data \
  --group_id 0 \
  --project_title "Multilabel Object Recognition"
```

For local baseline comparison:

```bash
python eval_baselines.py \
  --baseline_model_path baseline_resnet18_scratch.pth \
  --our_model_path best_cnn_model.pth
```

## Current Results

The saved training run selected epoch 12 as the best checkpoint by validation
loss. The final logged validation metrics were approximately:

| Metric | Value |
| --- | ---: |
| Hamming accuracy | 0.92 |
| Exact match accuracy | 0.49 |
| Macro F1 | 0.75 |

Training plots are available in `figures/`.

## Roadmap

- Add single-image inference CLI.
- Move hyperparameters into config files.
- Add a lightweight sample dataset for smoke tests.
- Add model-card documentation.
- Export the trained model to ONNX.
- Add a small web or Gradio demo for interactive predictions.
