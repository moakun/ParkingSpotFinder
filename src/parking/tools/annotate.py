"""Spot annotation tool (``psf-annotate``) — milestone M0.

Click the 4 corners of each parking spot on a reference frame; every 4th click
closes a spot. Saves the polygons to a lot geometry JSON that the rest of the
pipeline consumes.

Controls
--------
  left click   add a corner (4 per spot)
  u            undo last point (or last spot if none pending)
  s            save and quit
  q / ESC      quit without saving

Usage
-----
    psf-annotate reference.jpg --out configs/my_lot.json --roi 96 96
    psf-annotate lot.mp4       --out configs/my_lot.json   # uses the first frame
"""

from __future__ import annotations

import argparse
import string
import sys

import cv2
import numpy as np

from parking.geometry import LotConfig, Spot, save_lot
from parking.sources import open_source


def _spot_id(index: int) -> str:
    """A1, A2, ... A26, B1, ... — readable, stable ids."""
    letter = string.ascii_uppercase[index // 26]
    return f"{letter}{index % 26 + 1}"


class _Annotator:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.spots: list[list[tuple[int, int]]] = []
        self.pending: list[tuple[int, int]] = []

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending.append((x, y))
            if len(self.pending) == 4:
                self.spots.append(self.pending)
                self.pending = []

    def undo(self):
        if self.pending:
            self.pending.pop()
        elif self.spots:
            self.pending = self.spots.pop()
            self.pending.pop()

    def render(self) -> np.ndarray:
        canvas = self.frame.copy()
        for i, poly in enumerate(self.spots):
            pts = np.array(poly, np.int32)
            cv2.polylines(canvas, [pts], True, (0, 200, 0), 2)
            c = pts.mean(0).astype(int)
            cv2.putText(canvas, _spot_id(i), (c[0] - 10, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for p in self.pending:
            cv2.circle(canvas, p, 4, (0, 165, 255), -1)
        cv2.putText(canvas, f"spots: {len(self.spots)}  (s=save q=quit u=undo)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas


def _first_frame(source_spec: str) -> np.ndarray:
    with open_source(source_spec) as src:
        for frame in src:
            return frame
    raise IOError(f"No frames read from {source_spec!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="psf-annotate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="reference image or video (first frame used)")
    p.add_argument("--out", required=True, help="output lot geometry JSON")
    p.add_argument("--camera-id", default="cam1")
    p.add_argument("--roi", nargs=2, type=int, default=(96, 96), metavar=("W", "H"))
    args = p.parse_args(argv)

    frame = _first_frame(args.source)
    ann = _Annotator(frame)
    win = "psf-annotate"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, ann.on_mouse)

    saved = False
    while True:
        cv2.imshow(win, ann.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u"):
            ann.undo()
        if key == ord("s"):
            saved = True
            break
    cv2.destroyAllWindows()

    if not saved or not ann.spots:
        print("Nothing saved.", file=sys.stderr)
        return 1

    h, w = frame.shape[:2]
    cfg = LotConfig(
        camera_id=args.camera_id,
        image_size=(w, h),
        roi_size=(int(args.roi[0]), int(args.roi[1])),
        spots=[Spot(id=_spot_id(i), polygon=np.array(poly, np.float32)) for i, poly in enumerate(ann.spots)],
    )
    save_lot(cfg, args.out)
    print(f"Saved {len(cfg.spots)} spots -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
