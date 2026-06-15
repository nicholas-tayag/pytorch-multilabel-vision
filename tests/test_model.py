import torch

from vision_tagger.model import build_model


def test_model_output_shape():
    model = build_model(num_labels=12, dropout=0.0, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 128, 128))
    assert output.shape == (2, 12)
