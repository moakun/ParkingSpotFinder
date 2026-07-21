"""MEISCF detector adapter (Approach B) — INTEGRATION STUB.

This is the single place your MEISCF small-object detector plugs into the
pipeline. Once wired, MEISCF is interchangeable with the YOLO adapter and the
rest of the code is unaffected: the pipeline only sees the :class:`Detector`
interface (car boxes in image coordinates).

To finish this adapter I need to know a few things about MEISCF (ask/answer in
the project thread), then fill in the three TODOs below:

  1. How is it loaded?   e.g. a torch ``state_dict`` + model def, a TorchScript
     ``.pt``, or an exported ``.onnx`` (run via onnxruntime).
  2. Input contract:     expected size, color order (RGB/BGR), normalization,
     NCHW vs NHWC, letterbox/pad?
  3. Output contract:    box format (xyxy vs xywh vs cxcywh), absolute pixels vs
     normalized, score/class layout, and whether NMS is already applied.

Keeping MEISCF behind this adapter is deliberate: swapping detectors (to
benchmark MEISCF vs. YOLOv11 on the *same* spot-matching + eval harness) then
costs one CLI flag, and the head-to-head comparison is apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parking.detectors.base import Detection, Detector


class MEISCFDetector(Detector):
    """Adapter wrapping a MEISCF model behind the :class:`Detector` interface."""

    name = "meiscf"

    def __init__(
        self,
        weights: str | Path,
        device: str | None = None,
        input_size: tuple[int, int] = (640, 640),
        score_thresh: float = 0.25,
    ):
        self.weights = str(weights)
        self.device = device
        self.input_size = input_size
        self.score_thresh = score_thresh
        # TODO(1) — load the MEISCF model here (torch / torchscript / onnxruntime).
        # self.model = _load_meiscf(self.weights, self.device)
        raise NotImplementedError(
            "MEISCFDetector is a stub. Provide MEISCF's load format and I/O "
            "contract (see module docstring) and I'll fill in _preprocess, "
            "the forward call, and _postprocess."
        )

    def _preprocess(self, frame: np.ndarray):
        # TODO(2) — resize/letterbox to self.input_size, color-convert,
        # normalize, and lay out as the model expects (NCHW/NHWC).
        raise NotImplementedError

    def _postprocess(self, raw_output, orig_shape) -> list[Detection]:
        # TODO(3) — decode boxes to absolute xyxy in the ORIGINAL frame's
        # coordinates, apply NMS if needed, wrap each as Detection(bbox, score,
        # label="car"). Scale back if you letterboxed in _preprocess.
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list[Detection]:
        x = self._preprocess(frame)
        raw = self.model(x)  # type: ignore[attr-defined]
        dets = self._postprocess(raw, frame.shape[:2])
        return [d for d in dets if d.score >= self.score_thresh]
