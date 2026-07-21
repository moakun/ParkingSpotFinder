"""The classifier interface every model plugs into.

A classifier takes a *batch* of ROIs and returns one :class:`Prediction` per
ROI. Batching is not optional: the plan's key perf note (§4) is to stack all
ROIs from a frame into a single tensor and run **one** forward pass, never a
Python loop over spots. The interface enforces that shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from parking.types import Prediction


class Classifier(ABC):
    """Base class for all occupied/empty classifiers.

    Implementations must be safe to call with an empty batch (return ``[]``)
    and must return predictions in the same order as the input ROIs.
    """

    #: human-readable id, handy for logging/overlays
    name: str = "classifier"

    @abstractmethod
    def classify_batch(self, rois: Sequence[np.ndarray]) -> list[Prediction]:
        """Classify a batch of BGR ROIs. Returns one prediction per ROI."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: run a dummy batch so the first real frame isn't slow
        (JIT/CUDA init). No-op by default."""
        return None
