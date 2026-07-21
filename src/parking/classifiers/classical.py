"""Classical floor: edge-density threshold (plan §4, model stage 1).

No training. The intuition: a parked car fills the ROI with edges (windows,
trim, wheels, plates); empty asphalt is smooth and nearly edge-free. So the
fraction of Canny-edge pixels in the ROI separates occupied from empty
surprisingly well, and it tells you how hard the problem actually is before
you reach for a CNN. If the CNN can't beat this, the bug is in the CNN.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from parking.classifiers.base import Classifier
from parking.types import Occupancy, Prediction


class EdgeDensityClassifier(Classifier):
    """Threshold on Canny edge density.

    Parameters
    ----------
    threshold:
        Edge-pixel fraction above which a spot is called occupied.
    softness:
        Scale for mapping (density - threshold) to a confidence via a logistic;
        smaller -> more decisive.
    canny:
        (low, high) hysteresis thresholds for Canny.
    """

    name = "edge-density"

    def __init__(
        self,
        threshold: float = 0.12,
        softness: float = 0.05,
        canny: tuple[int, int] = (50, 150),
    ):
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.canny = canny

    def _density(self, roi: np.ndarray) -> float:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, self.canny[0], self.canny[1])
        return float(np.count_nonzero(edges)) / float(edges.size)

    def classify_batch(self, rois: Sequence[np.ndarray]) -> list[Prediction]:
        preds: list[Prediction] = []
        for roi in rois:
            density = self._density(roi)
            margin = (density - self.threshold) / max(self.softness, 1e-6)
            p_occupied = 1.0 / (1.0 + math.exp(-margin))
            if p_occupied >= 0.5:
                preds.append(Prediction(Occupancy.OCCUPIED, p_occupied))
            else:
                preds.append(Prediction(Occupancy.EMPTY, 1.0 - p_occupied))
        return preds
