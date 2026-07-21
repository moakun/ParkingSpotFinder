# Parking Lot Spot Finder — Project Plan

Detect **occupied** vs **available** parking spaces from images and video.

---

## 1. Core architectural decision (read this first)

There are two ways to frame this problem. They are not equally good, and the choice dictates the whole pipeline.

| | **A. ROI classification** (recommended) | **B. Object detection** (fallback) |
|---|---|---|
| Idea | Define spot polygons once, classify each ROI occupied/empty | Detect cars with YOLO, match boxes to spots by IoU |
| Fixed camera | Excellent | Overkill |
| Moving / handheld camera | Fails (geometry breaks) | Required |
| Handles empty spots natively | Yes — a spot is a first-class object | No — absence of a car ≠ a known empty spot |
| Cost | Tiny CNN, ~real-time on CPU | Full detector per frame |
| Weak point | Camera drift misaligns ROIs | Occlusion, spot geometry still needed |

**Decision:** build **A** as the main pipeline. It's cheaper, handles the empty-spot case directly, and the parking-CV literature (PKLot, CNRPark) hits >99% same-lot accuracy with it. Keep **B** documented as the path for unconstrained cameras — but don't build it first.

The one honest caveat: same-lot accuracy is trivially high and misleading. The real difficulty is **cross-lot / cross-weather generalization**, and the plan evaluates for that explicitly (§6).

---

## 2. Pipeline stages

```
source ─▶ spot geometry ─▶ ROI crop+warp ─▶ classifier ─▶ temporal smoothing ─▶ output
(img/vid)   (JSON polygons)   (batched tensor)  (occ/empty)   (video only)      (overlay+counts)
```

1. **Source abstraction** — one interface over image / video file / webcam stream. Frames flow through identically; only the source object differs.
2. **Spot geometry** — each spot stored as a quadrilateral (4 points) per camera, in a JSON config. Annotated *once* per camera view.
3. **ROI extraction** — crop each polygon; for angled cameras, perspective-warp to a canonical rectangle so the classifier sees consistent geometry.
4. **Classifier** — per-ROI occupied/empty + confidence.
5. **Temporal smoothing** — video only; suppresses per-frame flicker (§5).
6. **Output** — overlay (green/red), live available-count, optional JSON/HTTP endpoint.

---

## 3. Data

- **PKLot** — ~695k labeled spot images, 3 lots, sunny/rainy/overcast. The training workhorse.
- **CNRPark-EXT** — real-world, includes shadows and inter-vehicle occlusion. Use as the *hard* generalization test set.
- **Own capture** — target lot images + video, for final fine-tune and honest eval. This is what actually matters if you deploy it anywhere.

Split rule: **train and test on different lots.** Testing on held-out crops of the *same* lot inflates numbers and hides the failure mode.

---

## 4. Models

Build in this order — each stage is a baseline the next must beat.

1. **Classical floor** — grayscale ROI → edge/gradient density threshold, or HOG + linear SVM. Fast to build, tells you how hard the problem actually is. If a CNN can't beat this, something's wrong with the CNN, not the problem.
2. **Main — small CNN** — mAlexNet-style or MobileNetV3-Small backbone, ~96×96 ROI input. This is the production model. Small enough for real-time on CPU, strong enough for the cross-lot case.
3. Skip anything heavier unless cross-lot eval demands it. A ResNet50 here is wasted compute.

**Perf note that matters:** stack all ROIs from one frame into a single `(N, C, H, W)` tensor and run **one** forward pass — never loop per spot. With 100 spots that's a 100× reduction in inference calls and the difference between real-time and slideshow.

---

## 5. Temporal logic (video)

Frame-by-frame classification flickers: a pedestrian crossing an empty spot, a car mid-park, a shadow. Never flip a spot's state on a single frame.

Per-spot state machine with **hysteresis** — require N consecutive agreeing frames (or an EMA of confidence past a threshold) before changing state:

```python
class SpotState:
    def __init__(self, k=5):
        self.state = "empty"
        self.k = k          # frames of agreement needed to flip
        self.streak = 0
        self.pending = None

    def update(self, pred):          # pred: "occupied" | "empty"
        if pred == self.state:
            self.streak, self.pending = 0, None
            return self.state
        if pred == self.pending:
            self.streak += 1
        else:
            self.pending, self.streak = pred, 1
        if self.streak >= self.k:
            self.state, self.streak, self.pending = pred, 0, None
        return self.state
```

Tune `k` against frame rate: at 30 fps, `k=5` ≈ 0.17s of confirmation — enough to reject transients, fast enough to feel live.

---

## 6. Evaluation

- **Primary metric:** per-spot accuracy, **reported per-lot and cross-lot** — not a single blended number.
- **Cost-weighted view:** a false *available* (says empty, is full) is the expensive error — it sends a driver to an occupied spot. Track that cell of the confusion matrix specifically; it's the one that ruins trust.
- **Robustness slices:** break accuracy out by weather and by day/night. A model that's 99% sunny and 70% rainy is a 70% model in practice.
- **Latency / FPS** on the actual target hardware, measured end-to-end (crop + warp + inference + smoothing), not just the model's forward pass.

---

## 7. Known hard parts

- **Camera drift** — a few pixels of movement misaligns every ROI. Either lock the mount, or add periodic re-calibration / mild geometric robustness in the crop.
- **Occlusion & angle** — tall vehicles overflow their ROI or hide the spot behind them. Steeper camera angle helps; a top-down view helps most.
- **Domain shift** — night, rain, snow, low sun. This is where naive models collapse; it's why §6 slices by condition.
- **Cross-lot generalization** — the actual research-grade difficulty. Budget iteration here, not on squeezing same-lot accuracy past 99%.

---

## 8. Milestones

- **M0** — data pipeline + a spot-annotation tool (click 4 corners → JSON).
- **M1** — classical baseline on PKLot. Establishes the floor.
- **M2** — CNN classifier + **cross-lot** eval. The core deliverable.
- **M3** — video pipeline + temporal smoothing on real clips.
- **M4** — capture target lot, fine-tune, ship the overlay + count output.

---

## 9. Stack

Python · OpenCV (I/O, warp, overlay) · PyTorch (classifier) · ONNX (export, if deploying to edge/browser).

Given your RN/Expo background, a phone or web frontend consuming the availability JSON is a natural M5 — but keep it out of the core CV pipeline. The vision system should output structured state; the UI is a separate consumer.
