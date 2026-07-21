"""Command-line runner (``psf-run``): source + geometry -> overlay + counts.

Examples
--------
Still image with the no-training classical baseline::

    psf-run assets/lot.jpg --config configs/example_lot.json --show

Video with the trained CNN and temporal smoothing, saving an annotated clip::

    psf-run lot.mp4 --config configs/my_lot.json \
        --classifier cnn --weights weights/mobilenet.pt --save outputs/annotated.mp4
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from parking.classifiers import build_classifier
from parking.geometry import load_lot
from parking.output import draw_overlay
from parking.pipeline import ParkingPipeline
from parking.sources import open_source
from parking.sources.frame_source import VideoSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psf-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="image path / glob, video path, or webcam index")
    p.add_argument("--config", required=True, help="lot geometry JSON (from psf-annotate)")
    p.add_argument("--classifier", default="classical", help="classical | cnn")
    p.add_argument("--weights", default=None, help="checkpoint for --classifier cnn")
    p.add_argument("--k", type=int, default=5, help="smoothing frames (streams only; 1 disables)")
    p.add_argument("--show", action="store_true", help="display a live window")
    p.add_argument("--save", default=None, help="write annotated image/video here")
    p.add_argument("--json-out", default=None, help="write per-frame availability JSON here")
    p.add_argument("--conf", action="store_true", help="draw per-spot confidence")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    lot = load_lot(args.config)
    clf_kwargs = {}
    if args.classifier.lower() in {"cnn", "mobilenet", "mobilenetv3"}:
        clf_kwargs = {"weights": args.weights, "input_size": lot.roi_size}
    classifier = build_classifier(args.classifier, **clf_kwargs)

    source = open_source(args.source)
    k = args.k if source.is_stream else 1
    pipeline = ParkingPipeline(lot, classifier, smoothing_k=k)

    writer = None
    json_records: list[dict] = []
    last_result = None

    try:
        for frame, result in pipeline.run(source):
            last_result = result
            annotated = draw_overlay(frame, lot.spots, result, show_conf=args.conf)

            if args.json_out is not None:
                json_records.append(result.to_dict())

            if args.save is not None:
                writer = _write(args.save, annotated, writer, source)

            if args.show:
                cv2.imshow("parking-spot-finder", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if args.json_out is not None:
        payload = json_records if source.is_stream else (json_records[-1] if json_records else {})
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if last_result is not None:
        print(f"Available: {last_result.available} / {last_result.total}")
    return 0


def _write(path: str, frame, writer, source):
    """Lazily create the right writer (image vs video) on first frame."""
    if not source.is_stream:
        cv2.imwrite(path, frame)
        return None
    if writer is None:
        h, w = frame.shape[:2]
        fps = source.fps if isinstance(source, VideoSource) and source.fps > 0 else 20.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    writer.write(frame)
    return writer


if __name__ == "__main__":
    sys.exit(main())
