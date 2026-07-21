"""Draw the availability overlay: green polygon = empty, red = occupied, plus a
live count banner. Kept purely presentational — it consumes the structured
:class:`FrameResult` the pipeline emits and returns an annotated copy."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from parking.geometry import Spot
from parking.types import FrameResult, Occupancy

_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_GRAY = (160, 160, 160)
_COLORS = {
    Occupancy.EMPTY: _GREEN,
    Occupancy.OCCUPIED: _RED,
    Occupancy.UNKNOWN: _GRAY,
}


def draw_overlay(
    frame: np.ndarray,
    spots: Sequence[Spot],
    result: FrameResult,
    *,
    alpha: float = 0.35,
    show_ids: bool = True,
    show_conf: bool = False,
) -> np.ndarray:
    """Return a copy of ``frame`` with spot polygons and a count banner drawn."""
    out = frame.copy()
    fill = out.copy()
    by_id = {r.spot_id: r for r in result.results}

    for spot in spots:
        r = by_id.get(spot.id)
        color = _COLORS[r.state] if r else _GRAY
        poly = spot.polygon.astype(np.int32)
        cv2.fillPoly(fill, [poly], color)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2)

    cv2.addWeighted(fill, alpha, out, 1 - alpha, 0, dst=out)

    if show_ids or show_conf:
        for spot in spots:
            r = by_id.get(spot.id)
            centroid = spot.polygon.mean(axis=0).astype(int)
            label = spot.id if show_ids else ""
            if show_conf and r:
                label = f"{label} {r.raw.confidence:.2f}".strip()
            if label:
                cv2.putText(
                    out, label, (centroid[0] - 12, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

    _draw_banner(out, result)
    return out


def _draw_banner(out: np.ndarray, result: FrameResult) -> None:
    text = f"Available: {result.available} / {result.total}"
    cv2.rectangle(out, (0, 0), (270, 40), (0, 0, 0), -1)
    cv2.putText(
        out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )
