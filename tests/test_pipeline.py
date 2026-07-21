import numpy as np

from parking.classifiers import EdgeDensityClassifier
from parking.geometry import LotConfig, Spot
from parking.output import draw_overlay
from parking.pipeline import ParkingPipeline
from parking.types import FrameResult


def _lot():
    return LotConfig(
        camera_id="test",
        image_size=(200, 200),
        roi_size=(64, 64),
        spots=[
            Spot("A1", [[10, 10], [90, 10], [90, 90], [10, 90]]),
            Spot("A2", [[110, 10], [190, 10], [190, 90], [10 + 100, 90]]),
        ],
    )


def test_pipeline_end_to_end_still_image():
    lot = _lot()
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)  # all smooth -> all empty
    pipe = ParkingPipeline(lot, EdgeDensityClassifier(), smoothing_k=1)
    result = pipe.process_frame(frame)

    assert isinstance(result, FrameResult)
    assert result.total == 2
    assert result.available == 2
    # structured output is JSON-serializable and keyed by spot id
    d = result.to_dict()
    assert set(d["spots"]) == {"A1", "A2"}


def test_overlay_runs_and_preserves_shape():
    lot = _lot()
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    pipe = ParkingPipeline(lot, EdgeDensityClassifier(), smoothing_k=1)
    result = pipe.process_frame(frame)
    annotated = draw_overlay(frame, lot.spots, result)
    assert annotated.shape == frame.shape
