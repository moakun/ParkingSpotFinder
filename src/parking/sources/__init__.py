"""Frame sources (§2 stage 1): one interface over image / video / webcam."""

from parking.sources.frame_source import (
    FrameSource,
    ImageSource,
    VideoSource,
    WebcamSource,
    open_source,
)

__all__ = [
    "FrameSource",
    "ImageSource",
    "VideoSource",
    "WebcamSource",
    "open_source",
]
