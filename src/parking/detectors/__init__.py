"""Approach B — object detection (plan §1, the moving/handheld-camera fallback).

NOT the primary path. The main pipeline (Approach A) classifies fixed spot
ROIs and needs no car detector at all. Build this only when the camera can't
be fixed, so spot polygons can't be pre-annotated.

A detector here answers "where are the cars"; :mod:`parking.detectors.matching`
then assigns those boxes to spots by IoU. Any detector — YOLO, or a custom
model such as MEISCF — plugs in by implementing :class:`Detector`.
"""

from parking.detectors.base import Detection, Detector
from parking.detectors.matching import occupancy_from_detections

__all__ = ["Detection", "Detector", "occupancy_from_detections"]
