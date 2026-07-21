"""Temporal smoothing (§5): reject per-frame flicker on video streams."""

from parking.temporal.smoothing import SmoothingBank, SpotState

__all__ = ["SmoothingBank", "SpotState"]
