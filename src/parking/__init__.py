"""Parking Lot Spot Finder.

Detect occupied vs. available parking spaces from images and video.

The main pipeline (Approach A in the project plan) is ROI classification:
fixed spot polygons -> crop+warp each ROI -> classify occupied/empty ->
temporal smoothing (video) -> overlay + counts. Object detection
(Approach B) lives under `parking.detectors` as the moving-camera fallback.
"""

from parking.types import Occupancy, Prediction, SpotResult, FrameResult

__all__ = ["Occupancy", "Prediction", "SpotResult", "FrameResult"]
__version__ = "0.1.0"
