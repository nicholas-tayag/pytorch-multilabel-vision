from pathlib import Path

from PIL import Image

from vision_tagger.constants import LABEL_TO_INDEX
from vision_tagger.data import DirectoryMultilabelDataset, encode_labels, parse_label_folder


def test_parse_label_folder_accepts_multilabel_names():
    assert parse_label_folder("pen_paper") == ["pen", "paper"]


def test_parse_label_folder_rejects_invalid_and_duplicate_labels():
    assert parse_label_folder("pen_pen") is None
    assert parse_label_folder("pen_mouse") is None


def test_encode_labels_returns_expected_multihot_vector():
    target = encode_labels(["pen", "paper"])
    assert target[LABEL_TO_INDEX["pen"]].item() == 1.0
    assert target[LABEL_TO_INDEX["paper"]].item() == 1.0
    assert target.sum().item() == 2.0


def test_directory_dataset_skips_invalid_files_and_folders(tmp_path: Path):
    valid_dir = tmp_path / "pen_paper"
    valid_dir.mkdir()
    Image.new("RGB", (128, 128), color="white").save(valid_dir / "img001.png")
    Image.new("RGB", (128, 128), color="white").save(valid_dir / "notes.png")

    invalid_dir = tmp_path / "pen_mouse"
    invalid_dir.mkdir()
    Image.new("RGB", (128, 128), color="white").save(invalid_dir / "img002.png")

    dataset = DirectoryMultilabelDataset(tmp_path)
    assert len(dataset) == 1
    _, target = dataset[0]
    assert target.sum().item() == 2.0
