import numpy as np

from parking.geometry import RoiExtractor, Spot, order_corners


def test_order_corners_canonical_regardless_of_click_order():
    # A square, points given in a scrambled order.
    scrambled = [[10, 10], [10, 110], [110, 110], [110, 10]]
    ordered = order_corners(scrambled)
    expected = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32)
    assert np.allclose(ordered, expected)


def test_roi_extractor_shapes_and_order():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    spots = [
        Spot("A1", [[10, 10], [60, 10], [60, 60], [10, 60]]),
        Spot("A2", [[70, 10], [120, 10], [120, 60], [70, 60]]),
    ]
    ext = RoiExtractor(spots, roi_size=(48, 32))
    rois = ext.extract(frame)
    assert len(rois) == 2
    assert rois[0].shape == (32, 48, 3)  # (h, w, c)


def test_roi_warp_recovers_content():
    # Put a white block inside a spot; warped ROI should be mostly white.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:60, 20:60] = 255
    spot = Spot("A1", [[20, 20], [60, 20], [60, 60], [20, 60]])
    roi = RoiExtractor([spot], roi_size=(40, 40)).extract(frame)[0]
    assert roi.mean() > 200
