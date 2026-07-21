"""Shared data types used across the pipeline.

Kept dependency-light (only enum/dataclasses) so every module can import
these without pulling in torch/opencv.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Occupancy(str, Enum):
    """State of a single parking spot.

    Subclasses ``str`` so values serialize directly to JSON as
    ``"empty"`` / ``"occupied"`` / ``"unknown"``.
    """

    EMPTY = "empty"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Prediction:
    """A single-frame classifier output for one ROI.

    ``confidence`` is the model's confidence *in ``label``* (0..1), not a
    fixed "probability of occupied". Downstream smoothing may use it as an
    EMA signal.
    """

    label: Occupancy
    confidence: float


@dataclass
class SpotResult:
    """Per-spot outcome for one frame.

    ``state`` is what the system reports (after temporal smoothing, if any);
    ``raw`` is the unsmoothed classifier prediction for this frame.
    """

    spot_id: str
    state: Occupancy
    raw: Prediction


@dataclass
class FrameResult:
    """Everything the pipeline produces for one frame."""

    results: list[SpotResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def available(self) -> int:
        return sum(1 for r in self.results if r.state is Occupancy.EMPTY)

    @property
    def occupied(self) -> int:
        return sum(1 for r in self.results if r.state is Occupancy.OCCUPIED)

    def to_dict(self) -> dict:
        """JSON-serializable snapshot — the structured output the plan (§9)
        wants the UI to consume, separate from the CV core."""
        return {
            "total": self.total,
            "available": self.available,
            "occupied": self.occupied,
            "spots": {
                r.spot_id: {
                    "state": r.state.value,
                    "confidence": round(r.raw.confidence, 4),
                }
                for r in self.results
            },
        }
