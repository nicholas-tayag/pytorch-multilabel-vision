# Model Card: VisionTagger ResNet-18

## Model Summary

VisionTagger is a multi-label image classifier for desk and study-space object
recognition. It predicts 12 independent object labels from a single image using
a ResNet-18 backbone and sigmoid outputs.

## Intended Use

- Personal desk inventory experiments.
- Learning-oriented multi-label computer vision workflows.
- Local inference demos for images with the expected object categories.

It is not intended for safety-critical inventory, surveillance, or general
object detection outside the listed labels.

## Labels

```text
pen, paper, book, clock, phone, laptop,
chair, desk, bottle, keychain, backpack, calculator
```

## Training Approach

- ResNet-18 transfer learning.
- Dropout classification head with 12 outputs.
- Binary cross-entropy with logits.
- Class-imbalance weighting from training label counts.
- Image augmentation: random resized crop, horizontal flip, rotation, color
  jitter, and random erasing.
- Per-class threshold tuning on validation predictions.

## Evaluation

Approximate metrics from the logged transfer-learning run:

| Metric | Value |
| --- | ---: |
| Exact match accuracy | 0.49 |
| Hamming accuracy | 0.92 |
| Macro F1 | 0.75 |

These are validation metrics from the available run log, not claims of
production performance.

## Limitations

- The original dataset is private and not included in this public repo.
- The label set is small and desk-specific.
- The model is image-level multi-label classification, not bounding-box object
  detection.
- Performance may drop on lighting, camera angles, or object styles not covered
  by the training data.

## Ethical and Privacy Notes

The project should be used with images the user owns or has permission to
process. Desk images can contain private documents, screens, or identifying
information, so local inference is preferred by default.
