"""Tests for the PKLot prep script, exercised on a tiny synthetic tree so no
real dataset is needed."""

from pathlib import Path

import pytest

from parking.data.prepare_pklot import choose_split, collect_lot, discover_lots, prepare


def _make_fake_pklot(root: Path) -> Path:
    """Build <root>/PKLotSegmented/<lot>/<weather>/<date>/{Empty,Occupied}/*.jpg."""
    seg = root / "PKLotSegmented"
    layout = {
        "PUC": ("Sunny", 2, 3),      # (weather, n_empty, n_occupied)
        "UFPR04": ("Rainy", 1, 2),
        "UFPR05": ("Cloudy", 2, 1),
    }
    for lot, (weather, n_e, n_o) in layout.items():
        for cls, n in (("Empty", n_e), ("Occupied", n_o)):
            d = seg / lot / weather / "2013-01-01" / cls
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff")  # minimal jpeg-ish bytes
    return root


def test_discover_and_collect(tmp_path):
    root = _make_fake_pklot(tmp_path)
    seg = root / "PKLotSegmented"
    lots = discover_lots(seg)
    assert set(lots) == {"PUC", "UFPR04", "UFPR05"}

    puc = collect_lot(lots["PUC"], weathers=None)
    assert len(puc["empty"]) == 2 and len(puc["occupied"]) == 3


def test_choose_split_auto_holds_out_puc(tmp_path):
    lots = discover_lots(_make_fake_pklot(tmp_path) / "PKLotSegmented")
    train, val = choose_split(lots, None, None)
    assert val == ["PUC"]
    assert set(train) == {"UFPR04", "UFPR05"}


def test_choose_split_rejects_overlap(tmp_path):
    lots = discover_lots(_make_fake_pklot(tmp_path) / "PKLotSegmented")
    with pytest.raises(SystemExit):
        choose_split(lots, train_lots=["PUC"], val_lots=["PUC"])


def test_prepare_builds_imagefolder(tmp_path):
    root = _make_fake_pklot(tmp_path)
    out = tmp_path / "out"
    manifest = prepare(
        src=root, out=out,
        train_lots=["UFPR04", "UFPR05"], val_lots=["PUC"],
        link="copy",
    )
    # class dirs must be exactly empty/occupied (train.py asserts this)
    assert (out / "train" / "empty").is_dir()
    assert (out / "train" / "occupied").is_dir()
    assert (out / "val" / "empty").is_dir()

    # counts: train = UFPR04(1e,2o)+UFPR05(2e,1o) = 3 empty, 3 occupied; val = PUC(2e,3o)
    assert manifest["counts"]["train"] == {"empty": 3, "occupied": 3}
    assert manifest["counts"]["val"] == {"empty": 2, "occupied": 3}
    assert len(list((out / "val" / "occupied").glob("*.jpg"))) == 3
    assert (out / "split_manifest.json").is_file()


def test_prepare_weather_filter_and_limit(tmp_path):
    root = _make_fake_pklot(tmp_path)
    out = tmp_path / "out"
    # only Rainy exists in UFPR04; filtering to Sunny should yield 0 there
    m = prepare(src=root, out=out, train_lots=["UFPR04"], val_lots=["PUC"],
                weather=["Sunny"], link="copy")
    assert m["counts"]["train"] == {"empty": 0, "occupied": 0}
    assert m["counts"]["val"]["occupied"] == 3  # PUC is Sunny
