import numpy as np

from parking.classifiers import EdgeDensityClassifier
from parking.detectors import occupancy_from_detections
from parking.detectors.base import Detection
from parking.geometry import Spot
from parking.types import Occupancy


def test_edge_density_smooth_roi_is_empty():
    clf = EdgeDensityClassifier()
    smooth = np.full((96, 96, 3), 128, dtype=np.uint8)  # flat asphalt-ish
    (pred,) = clf.classify_batch([smooth])
    assert pred.label is Occupancy.EMPTY
    assert 0.0 <= pred.confidence <= 1.0


def test_edge_density_busy_roi_is_occupied():
    clf = EdgeDensityClassifier()
    rng = np.random.default_rng(0)
    busy = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)  # lots of edges
    (pred,) = clf.classify_batch([busy])
    assert pred.label is Occupancy.OCCUPIED


def test_empty_batch_returns_empty_list():
    assert EdgeDensityClassifier().classify_batch([]) == []


def test_matching_box_over_spot_marks_occupied():
    spots = [
        Spot("A1", [[0, 0], [100, 0], [100, 100], [0, 100]]),
        Spot("A2", [[200, 0], [300, 0], [300, 100], [200, 100]]),
    ]
    dets = [Detection(bbox=(5, 5, 95, 95), score=0.9, label="car")]  # covers A1 only
    preds = occupancy_from_detections(spots, dets, coverage_thresh=0.3)
    assert preds[0].label is Occupancy.OCCUPIED
    assert preds[1].label is Occupancy.EMPTY
