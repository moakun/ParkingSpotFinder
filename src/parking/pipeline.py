"""The Approach A pipeline: source -> ROI crop+warp -> classify -> smooth -> output.

This is the orchestration layer. It stays model-agnostic: any
:class:`~parking.classifiers.base.Classifier` plugs in, and temporal smoothing
is applied only for streams (video/webcam), matching the plan (§2, §5).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from parking.classifiers.base import Classifier
from parking.geometry import LotConfig, RoiExtractor
from parking.temporal import SmoothingBank
from parking.types import FrameResult, SpotResult


class ParkingPipeline:
    """Per-frame occupancy pipeline for a single fixed camera.

    Parameters
    ----------
    lot:
        Spot geometry for this camera view.
    classifier:
        Any occupied/empty classifier.
    smoothing_k:
        Frames of agreement required to flip a spot's state. Set to 0/1 to
        disable smoothing (e.g. for still images).
    """

    def __init__(self, lot: LotConfig, classifier: Classifier, smoothing_k: int = 5):
        self.lot = lot
        self.classifier = classifier
        self.extractor = RoiExtractor(lot.spots, lot.roi_size)
        self.spot_ids = [s.id for s in lot.spots]
        self.smoother: SmoothingBank | None = (
            SmoothingBank(self.spot_ids, k=smoothing_k) if smoothing_k and smoothing_k > 1 else None
        )

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        rois = self.extractor.extract(frame)
        preds = self.classifier.classify_batch(rois)  # one batched forward pass
        if self.smoother is not None:
            results = self.smoother.update(self.spot_ids, preds)
        else:
            results = [
                SpotResult(spot_id=sid, state=p.label, raw=p)
                for sid, p in zip(self.spot_ids, preds)
            ]
        return FrameResult(results=results)

    def run(self, source) -> Iterator[tuple[np.ndarray, FrameResult]]:
        """Yield ``(frame, result)`` for each frame from an iterable source."""
        self.classifier.warmup()
        for frame in source:
            yield frame, self.process_frame(frame)
