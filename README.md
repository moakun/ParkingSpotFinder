# Parking Lot Spot Finder

Detect **occupied** vs **available** parking spaces from images and video.

See [`parking-spot-finder-plan.md`](parking-spot-finder-plan.md) for the full design rationale. This README is the practical entry point.

---

## Architecture in one breath

Two framings exist; they are **not** equal (plan §1):

- **Approach A — ROI classification (primary).** Define each spot's polygon once per fixed camera, crop+warp each ROI, and classify it occupied/empty with a tiny CNN. Handles empty spots natively, real-time on CPU, **no car detector needed.**
- **Approach B — object detection (fallback).** Only when the camera moves/can't be pre-annotated: detect cars, assign boxes to spots by overlap. This is the **only** place a detector (YOLO, or a custom model like **MEISCF**) is involved.

The code builds A as the main pipeline and keeps B fully scaffolded behind a one-flag detector interface.

```
source ─▶ spot geometry ─▶ ROI crop+warp ─▶ classifier ─▶ temporal smoothing ─▶ overlay + counts
(img/vid)  (JSON polygons)  (batched tensor)  (occ/empty)   (video only)         (+ structured JSON)
```

---

## Install

```bash
python -m pip install -e .          # core (Approach A)
python -m pip install -e ".[dev]"   # + pytest
python -m pip install -e ".[detect]"  # + ultralytics/onnxruntime for Approach B
```

Requires Python ≥ 3.10. Torch is only needed for the CNN classifier; the classical baseline runs on OpenCV + NumPy alone.

---

## Quickstart

**1. Annotate spots once** (click 4 corners per spot; `s` saves, `u` undoes, `q` quits):

```bash
psf-annotate path/to/reference.jpg --out configs/my_lot.json --roi 96 96
```

**2. Run it.** The classical baseline needs no training — good first smoke test:

```bash
# still image
psf-run path/to/lot.jpg --config configs/my_lot.json --show

# video, trained CNN, temporal smoothing, save annotated clip + per-frame JSON
psf-run lot.mp4 --config configs/my_lot.json \
    --classifier cnn --weights weights/mobilenet.pt \
    --k 5 --save outputs/annotated.mp4 --json-out outputs/availability.json

# live webcam (device 0)
psf-run 0 --config configs/my_lot.json --classifier cnn --weights weights/mobilenet.pt --show
```

Structured output (`--json-out`) is the contract for any UI (plan §9) — the CV core just emits state:

```json
{"total": 3, "available": 1, "occupied": 2,
 "spots": {"A1": {"state": "occupied", "confidence": 0.99}, "A2": {"state": "empty", "confidence": 0.92}}}
```

**3. Prep PKLot, then train the CNN** (milestone M2). First reshape PKLot's
`<lot>/<weather>/<date>/{Empty,Occupied}` tree into an ImageFolder split — a
whole lot is **held out for validation** (plan §3), so train/val never share a
lot:

```bash
# hold out PUC for val, train on UFPR04+UFPR05; hardlink avoids duplicating ~1GB
python -m parking.data.prepare_pklot --src data/PKLot --out data/pklot --link hardlink

python -m parking.train --data data/pklot --epochs 8 --out weights/mobilenet.pt
```

`prepare_pklot` supports `--train-lots/--val-lots` (explicit split), `--weather`
(robustness slices, §6), `--limit-per-class` (smoke subsets), and `--dry-run`.
It writes a `split_manifest.json` recording exactly which lots went where.

**Alternative source — Roboflow PKLot (COCO detection export).** If you have the
Roboflow `train/valid/test` + `_annotations.coco.json` export instead, crop its
labeled space boxes into the same ImageFolder layout:

```bash
python -m parking.data.coco_to_rois --src archive --out data/pklot_roi
python -m parking.train --data data/pklot_roi --epochs 8 --out weights/mobilenet.pt
```

Caveat: that export carries **no lot identity**, so its split mixes lots and is
*not* a cross-lot split (plan §3) — fine for getting a classifier training fast,
but use `prepare_pklot` on the canonical PKLotSegmented for honest cross-lot eval.

---

## Project layout

```
src/parking/
  types.py              shared dataclasses (Occupancy, Prediction, FrameResult)
  sources/              image / video / webcam behind one interface
  geometry/             spot polygons, ordering, ROI crop + perspective warp
  classifiers/          base interface + classical baseline + MobileNetV3 CNN
  detectors/            Approach B: Detector interface, box→spot matching,
                        yolo.py (reference), meiscf.py (your model — stub)
  temporal/             per-spot hysteresis smoothing (anti-flicker)
  output/               green/red overlay + count banner
  data/                 PKLot -> ImageFolder split-by-lot prep
  pipeline.py           orchestration (Approach A)
  train.py              CNN training + cross-lot eval
  cli.py                psf-run
  tools/annotate.py     psf-annotate (M0)
configs/                lot geometry JSON
tests/                  deterministic core (13 tests, no data/weights needed)
```

---

## The model layer is pluggable

Every classifier implements `Classifier.classify_batch(rois) -> [Prediction]`; every detector implements `Detector.detect(frame) -> [Detection]`. Both **batch a whole frame in one forward pass** (plan §4 perf note). Swapping models is a CLI flag, so head-to-head benchmarks (e.g. **MEISCF vs YOLOv11** on the same matching + eval harness) are apples-to-apples.

**MEISCF integration point:** [`src/parking/detectors/meiscf.py`](src/parking/detectors/meiscf.py) is a documented stub with the exact load / input / output contract it needs. Fill the three TODOs and it drops into Approach B interchangeably with YOLO. (If MEISCF is better used as the ROI *classifier* for Approach A, it can instead subclass `Classifier` — see [open question](#open-questions).)

---

## Milestones (plan §8)

| | | status |
|---|---|---|
| M0 | data pipeline + spot-annotation tool | ✅ `psf-annotate`, source abstraction |
| M1 | classical baseline on PKLot | ✅ `EdgeDensityClassifier` (the floor) |
| M2 | CNN classifier + **cross-lot** eval | ◻ prep (`prepare_pklot`) + arch + `train.py` ready; needs data + weights |
| M3 | video pipeline + temporal smoothing | ✅ pipeline + hysteresis; needs real clips |
| M4 | capture target lot, fine-tune, ship overlay + counts | ◻ |
| M5 | (optional) phone/web UI consuming the JSON | ◻ separate from CV core |

---

## Data

- **PKLot** — ~695k labeled spot images; the training workhorse.
- **CNRPark-EXT** — real-world shadows/occlusion; the *hard* generalization test.
- **Own capture** — target lot; final fine-tune + honest eval.

Split rule: **train and test on different lots.** Same-lot accuracy is trivially >99% and misleading. `train.py` reports cross-lot accuracy **and** the false-*available* rate — the expensive error that sends a driver to a full spot (plan §6).

---

## Open questions

Wiring MEISCF needs a couple of decisions — see the summary in the project thread. Until then, MEISCF lives behind its adapter stub and the rest of the pipeline is fully runnable with the classical baseline and the (trainable) CNN.
