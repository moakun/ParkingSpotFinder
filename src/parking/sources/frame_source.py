"""Source abstraction: one interface over image file, video file, and webcam.

Frames flow through the rest of the pipeline identically (BGR ``np.ndarray``);
only the source object differs. This is stage 1 of the pipeline in the plan (§2).
"""

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import cv2
import numpy as np

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FrameSource(ABC):
    """Yields BGR frames. ``is_stream`` distinguishes a live/continuous feed
    (video, webcam) from a finite set of stills (temporal smoothing only makes
    sense for streams)."""

    is_stream: bool = False

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]: ...

    def release(self) -> None:  # pragma: no cover - trivial default
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class ImageSource(FrameSource):
    """One or more still images. Accepts a single path or a glob pattern."""

    is_stream = False

    def __init__(self, spec: str):
        if any(ch in spec for ch in "*?[") and not os.path.exists(spec):
            self.paths = sorted(glob.glob(spec))
        else:
            self.paths = [spec]
        if not self.paths:
            raise FileNotFoundError(f"No images matched: {spec!r}")

    def __iter__(self) -> Iterator[np.ndarray]:
        for p in self.paths:
            frame = cv2.imread(p, cv2.IMREAD_COLOR)
            if frame is None:
                raise IOError(f"Failed to read image: {p!r}")
            yield frame


class VideoSource(FrameSource):
    """A video file. Iterates frames until EOF."""

    is_stream = True

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {path!r}")

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 0.0

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self.cap.release()


class WebcamSource(FrameSource):
    """A live webcam by device index. Runs until iteration is broken."""

    is_stream = True

    def __init__(self, index: int = 0):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open webcam index {index}")

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self.cap.release()


def open_source(spec: str | int) -> FrameSource:
    """Factory: pick the right source from a spec.

    - ``int`` or an all-digit string  -> webcam by index
    - a path with an image extension   -> :class:`ImageSource` (also globs)
    - anything else                    -> :class:`VideoSource`
    """
    if isinstance(spec, int):
        return WebcamSource(spec)
    if isinstance(spec, str) and spec.isdigit():
        return WebcamSource(int(spec))

    ext = os.path.splitext(str(spec))[1].lower()
    if ext in _IMAGE_EXTS or any(ch in str(spec) for ch in "*?["):
        return ImageSource(str(spec))
    return VideoSource(str(spec))
