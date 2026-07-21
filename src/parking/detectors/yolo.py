"""YOLO detector adapter (Approach B) — reference / baseline detector.

Thin wrapper over Ultralytics YOLO so there's a working detector to validate
the Approach B path and to benchmark a custom detector (e.g. MEISCF) against on
the *same* spot-matching and eval harness. Requires the optional ``detect``
extra: ``pip install ultralytics``.
"""

from __future__ import annotations

import numpy as np

from parking.detectors.base import Detection, Detector


class YOLODetector(Detector):
    """Ultralytics YOLO adapter (YOLOv8/v11 weights)."""

    name = "yolo"

    def __init__(self, weights: str = "yolo11n.pt", device: str | None = None, score_thresh: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "YOLODetector needs ultralytics: pip install 'parking-spot-finder[detect]'"
            ) from e
        self.model = YOLO(weights)
        self.device = device
        self.score_thresh = score_thresh
        self._names = self.model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, device=self.device, conf=self.score_thresh, verbose=False)[0]
        dets: list[Detection] = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = self._names[int(box.cls[0])]
            dets.append(Detection(bbox=(x1, y1, x2, y2), score=float(box.conf[0]), label=label))
        return dets
