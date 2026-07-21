"""Tests for the Roboflow-COCO -> ImageFolder converter, on a tiny synthetic
COCO export so no real dataset is needed."""

import json

import cv2
import numpy as np

from parking.data.coco_to_rois import _class_from_category, convert, find_archive_root


def _make_fake_coco(root, split="train", n_empty=3, n_occupied=2):
    d = root / split
    d.mkdir(parents=True)
    # one 100x100 frame with distinguishable boxes
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (50, 60, 70)
    cv2.imwrite(str(d / "f0.jpg"), frame)

    cats = [
        {"id": 0, "name": "spaces"},
        {"id": 1, "name": "space-empty"},
        {"id": 2, "name": "space-occupied"},
    ]
    anns, aid = [], 0
    for _ in range(n_empty):
        anns.append({"id": aid, "image_id": 0, "category_id": 1, "bbox": [5, 5, 20, 30], "area": 600, "iscrowd": 0})
        aid += 1
    for _ in range(n_occupied):
        anns.append({"id": aid, "image_id": 0, "category_id": 2, "bbox": [40, 40, 25, 25], "area": 625, "iscrowd": 0})
        aid += 1
    coco = {
        "images": [{"id": 0, "file_name": "f0.jpg", "height": 100, "width": 100}],
        "categories": cats,
        "annotations": anns,
    }
    (d / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return root


def test_class_mapping():
    assert _class_from_category("space-empty") == "empty"
    assert _class_from_category("space-occupied") == "occupied"
    assert _class_from_category("spaces") is None  # supercategory skipped


def test_find_archive_root_accepts_parent(tmp_path):
    _make_fake_coco(tmp_path / "archive", "train")
    assert find_archive_root(tmp_path).name == "archive"          # parent
    assert find_archive_root(tmp_path / "archive").name == "archive"  # itself


def test_convert_builds_imagefolder_and_maps_valid_to_val(tmp_path):
    root = tmp_path / "archive"
    _make_fake_coco(root, "train", n_empty=3, n_occupied=2)
    _make_fake_coco(root, "valid", n_empty=1, n_occupied=4)
    out = tmp_path / "out"

    manifest = convert(src=root, out=out)

    # class dirs exactly empty/occupied (train.py asserts this); valid -> val
    assert (out / "train" / "empty").is_dir() and (out / "train" / "occupied").is_dir()
    assert (out / "val" / "empty").is_dir()
    assert not (out / "valid").exists()

    assert manifest["counts"]["train"] == {"empty": 3, "occupied": 2}
    assert manifest["counts"]["val"] == {"empty": 1, "occupied": 4}
    assert len(list((out / "train" / "empty").glob("*.jpg"))) == 3

    # crops are real, correctly-sized images
    crop = cv2.imread(str(next((out / "train" / "empty").glob("*.jpg"))))
    assert crop is not None and crop.shape[:2] == (30, 20)  # bbox h=30, w=20


def test_limit_per_class_caps_output(tmp_path):
    root = tmp_path / "archive"
    _make_fake_coco(root, "train", n_empty=10, n_occupied=10)
    out = tmp_path / "out"
    m = convert(src=root, out=out, limit_per_class=4)
    assert m["counts"]["train"] == {"empty": 4, "occupied": 4}
