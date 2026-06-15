import torch

from vision_tagger.metrics import compute_metrics
from vision_tagger.thresholds import apply_thresholds


def test_apply_thresholds_returns_binary_predictions_with_expected_shape():
    probabilities = torch.tensor([[0.2, 0.7, 0.9], [0.8, 0.1, 0.4]])
    thresholds = torch.tensor([0.5, 0.5, 0.8])
    predictions = apply_thresholds(probabilities, thresholds=thresholds)
    assert predictions.shape == probabilities.shape
    assert predictions.tolist() == [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]


def test_compute_metrics_includes_standard_multilabel_scores():
    predictions = torch.tensor([[1, 0, 1], [0, 1, 0]])
    labels = torch.tensor([[1, 0, 0], [0, 1, 0]])
    metrics = compute_metrics(predictions, labels)
    assert metrics["hamming_acc"] == torch.tensor(5 / 6).item()
    assert "exact_match" in metrics
    assert "macro_f1" in metrics
    assert "mean_iou" in metrics
