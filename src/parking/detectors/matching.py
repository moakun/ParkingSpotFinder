"""Assign detected car boxes to spots by geometric overlap (Approach B).

The occupancy signal we use is *intersection-over-spot* (how much of the spot
polygon a car covers), not symmetric IoU — a tall vehicle that overflows its
box shouldn't be penalized, and we care whether the spot is filled, not whether
the box matches the spot exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from parking.detectors.base import Detection
from parking.geometry import Spot
from parking.types import Occupancy, Prediction


def _poly_box_intersection_area(poly: np.ndarray, box) -> float:
    x1, y1, x2, y2 = box
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    area, _ = cv2.intersectConvexConvex(poly.astype(np.float32), rect)
    return float(area)


def occupancy_from_detections(
    spots: Sequence[Spot],
    detections: Sequence[Detection],
    coverage_thresh: float = 0.30,
) -> list[Prediction]:
    """One :class:`Prediction` per spot, in spot order.

    A spot is ``OCCUPIED`` if the best-covering detection fills at least
    ``coverage_thresh`` of the spot polygon.
    """
    preds: list[Prediction] = []
    for spot in spots:
        poly = spot.polygon.astype(np.float32)
        spot_area = max(cv2.contourArea(poly), 1e-6)
        best_cov = 0.0
        for det in detections:
            cov = _poly_box_intersection_area(poly, det.bbox) / spot_area
            best_cov = max(best_cov, cov)
        if best_cov >= coverage_thresh:
            preds.append(Prediction(Occupancy.OCCUPIED, min(best_cov, 1.0)))
        else:
            preds.append(Prediction(Occupancy.EMPTY, 1.0 - min(best_cov, 1.0)))
    return preds
