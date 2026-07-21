"""Spot geometry and ROI extraction.

Each spot is a quadrilateral (4 points) in image coordinates, annotated once
per camera view and stored in a JSON config. For angled cameras we
perspective-warp each ROI to a canonical rectangle so the classifier always
sees consistent geometry (plan §2, §3).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Spot:
    """A single parking spot as an ordered quadrilateral."""

    id: str
    polygon: np.ndarray  # (4, 2) float32, image coordinates

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.float32).reshape(4, 2)


@dataclass
class LotConfig:
    """All spots for one camera view, plus its capture/ROI metadata."""

    camera_id: str
    image_size: tuple[int, int]  # (width, height) the polygons were drawn on
    roi_size: tuple[int, int]  # (width, height) each ROI is warped to
    spots: list[Spot]


def order_corners(pts: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return the 4 points ordered top-left, top-right, bottom-right, bottom-left.

    Robust to whatever order the annotator clicked in, so the perspective warp
    is always consistent. Uses the classic sum/diff trick:
    TL has the smallest x+y, BR the largest; TR the smallest y-x, BL the largest.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array(
        [
            pts[np.argmin(s)],  # top-left
            pts[np.argmin(d)],  # top-right
            pts[np.argmax(s)],  # bottom-right
            pts[np.argmax(d)],  # bottom-left
        ],
        dtype=np.float32,
    )


def load_lot(path: str | Path) -> LotConfig:
    """Load a lot geometry JSON produced by the annotation tool."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    spots = [Spot(id=str(s["id"]), polygon=s["polygon"]) for s in data["spots"]]
    return LotConfig(
        camera_id=data.get("camera_id", "unknown"),
        image_size=tuple(data.get("image_size", (0, 0))),  # type: ignore[arg-type]
        roi_size=tuple(data.get("roi_size", (96, 96))),  # type: ignore[arg-type]
        spots=spots,
    )


def save_lot(cfg: LotConfig, path: str | Path) -> None:
    """Serialize a lot geometry back to JSON."""
    data = {
        "camera_id": cfg.camera_id,
        "image_size": list(cfg.image_size),
        "roi_size": list(cfg.roi_size),
        "spots": [
            {"id": s.id, "polygon": s.polygon.astype(float).round(2).tolist()}
            for s in cfg.spots
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


class RoiExtractor:
    """Crops and perspective-warps each spot ROI to a canonical rectangle.

    Precomputes one perspective transform per spot at construction, so the
    per-frame cost is just N ``warpPerspective`` calls. Output ROIs are
    returned in the same order as ``spots``.
    """

    def __init__(self, spots: Sequence[Spot], roi_size: tuple[int, int] = (96, 96)):
        self.spots = list(spots)
        self.roi_w, self.roi_h = int(roi_size[0]), int(roi_size[1])
        dst = np.array(
            [
                [0, 0],
                [self.roi_w - 1, 0],
                [self.roi_w - 1, self.roi_h - 1],
                [0, self.roi_h - 1],
            ],
            dtype=np.float32,
        )
        self._transforms: list[np.ndarray] = [
            cv2.getPerspectiveTransform(order_corners(s.polygon), dst)
            for s in self.spots
        ]

    def extract(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return one warped ROI (roi_h, roi_w, 3) per spot, in spot order."""
        return [
            cv2.warpPerspective(frame, M, (self.roi_w, self.roi_h))
            for M in self._transforms
        ]
