"""Detector interface for Approach B.

A detector consumes a full BGR frame and returns car bounding boxes. This is
deliberately minimal so that any backend — an off-the-shelf YOLO, or a custom
small-object detector like MEISCF — can implement it without the rest of the
pipeline caring which one is running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object in image coordinates."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    label: str = "car"


class Detector(ABC):
    """Base class for car detectors (Approach B)."""

    name: str = "detector"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for one frame."""
        raise NotImplementedError

    def warmup(self) -> None:
        return None

    @staticmethod
    def filter_vehicles(
        dets: Sequence[Detection],
        classes: Sequence[str] = ("car", "truck", "bus", "motorcycle"),
        min_score: float = 0.25,
    ) -> list[Detection]:
        """Keep only vehicle detections above a score threshold."""
        allowed = {c.lower() for c in classes}
        return [d for d in dets if d.label.lower() in allowed and d.score >= min_score]
