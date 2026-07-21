"""Small CNN classifier (plan §4, model stage 2 — the production model).

MobileNetV3-Small backbone, 2-class head (empty/occupied), ~96x96 ROI input.
Small enough for real-time on CPU, strong enough for the cross-lot case.

Batching (plan §4): all ROIs from a frame are stacked into one
``(N, 3, H, W)`` tensor and pushed through a **single** forward pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from parking.classifiers.base import Classifier
from parking.types import Occupancy, Prediction

# Class index convention: 0 = empty, 1 = occupied. Keep this consistent with
# training (see parking/train.py) and any exported weights.
_CLASSES = (Occupancy.EMPTY, Occupancy.OCCUPIED)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(arch: str = "mobilenetv3_small", num_classes: int = 2):
    """Construct the backbone with a fresh 2-class head. Import torch lazily so
    the rest of the package works without it."""
    import torchvision

    if arch in {"mobilenetv3_small", "mobilenetv3", "mobilenet"}:
        import torch.nn as nn

        model = torchvision.models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unknown arch {arch!r}")


class CNNClassifier(Classifier):
    """Torch CNN classifier with batched inference.

    Parameters
    ----------
    weights:
        Path to a checkpoint (state_dict or ``{'model': state_dict}``). If
        ``None`` the head is randomly initialized — fine for wiring up the
        pipeline, but predictions are meaningless until you train (see
        ``parking/train.py``).
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` to auto-select.
    input_size:
        (width, height) the ROI is resized to before the network.
    """

    name = "mobilenetv3-small"

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | None = None,
        arch: str = "mobilenetv3_small",
        input_size: tuple[int, int] = (96, 96),
    ):
        import torch

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_size = input_size
        self.trained = weights is not None

        self.model = build_backbone(arch, num_classes=len(_CLASSES))
        if weights is not None:
            state = torch.load(weights, map_location="cpu")
            state = state.get("model", state) if isinstance(state, dict) else state
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def _to_tensor(self, rois: Sequence[np.ndarray]):
        """Stack BGR ROIs into a normalized (N, 3, H, W) float tensor."""
        w, h = self.input_size
        batch = np.empty((len(rois), h, w, 3), dtype=np.float32)
        for i, roi in enumerate(rois):
            if roi.shape[1] != w or roi.shape[0] != h:
                roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
            batch[i] = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        t = self.torch.from_numpy(batch).to(self.device)
        t = t.permute(0, 3, 1, 2) / 255.0
        return (t - self._mean) / self._std

    def classify_batch(self, rois: Sequence[np.ndarray]) -> list[Prediction]:
        if not rois:
            return []
        with self.torch.inference_mode():
            x = self._to_tensor(rois)
            logits = self.model(x)  # ONE forward pass for the whole frame
            probs = self.torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        conf = conf.cpu().numpy()
        idx = idx.cpu().numpy()
        return [Prediction(_CLASSES[int(i)], float(c)) for i, c in zip(idx, conf)]

    def warmup(self) -> None:
        w, h = self.input_size
        dummy = [np.zeros((h, w, 3), dtype=np.uint8)]
        self.classify_batch(dummy)
