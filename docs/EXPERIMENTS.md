# Experiment Notes

## Main Comparison

VisionTagger tracks three baselines:

| Approach | Purpose | Notes |
| --- | --- | --- |
| Mode labelset baseline | Sanity check | Always predicts the most common label combination from the training split. |
| Scratch ResNet-18 | Architecture baseline | Same family of model, but no pretrained visual features. |
| Transfer-learned ResNet-18 | Main model | Uses pretrained ImageNet features, augmentation, class weighting, and threshold tuning. |

## Logged Transfer-Learning Run

The saved run selected epoch 12 as the best checkpoint by validation loss:

| Metric | Approximate Value |
| --- | ---: |
| Validation loss | 0.53 |
| Exact match accuracy | 0.49 |
| Hamming accuracy | 0.92 |
| Macro F1 | 0.75 |

## Per-Class Thresholds

Thresholds were tuned on validation predictions to maximize per-class F1. This
matters because each label appears with a different frequency, so a single 0.5
cutoff is not always the best decision rule.

The logged thresholds were:

| Label | Threshold |
| --- | ---: |
| pen | 0.5600 |
| paper | 0.3800 |
| book | 0.6650 |
| clock | 0.7500 |
| phone | 0.7550 |
| laptop | 0.8300 |
| chair | 0.7500 |
| desk | 0.7850 |
| bottle | 0.7300 |
| keychain | 0.9100 |
| backpack | 0.9225 |
| calculator | 0.7950 |

## Next Experiment Questions

- Does a personal desk-photo validation set expose different errors than the
  original dataset?
- How much does threshold tuning help compared with a global 0.5 threshold?
- Which augmentations improve generalization versus adding noise?
- Can a smaller model keep most of the quality while running faster on CPU?
