"""Standalone evaluation for the trained classifier (plan §6).

Loads a checkpoint and evaluates it on an ImageFolder split (default: the
held-out ``val`` lot), reporting cross-lot accuracy, the confusion matrix,
per-class precision/recall, and the two operational error rates:

  * false-available (predicts empty, is occupied) — the expensive error; it
    sends a driver to a full spot.
  * false-occupied  (predicts occupied, is empty) — wastes a real free spot.

If the crop filenames encode weather (as ``prepare_pklot`` writes them, e.g.
``PUC__Sunny__...jpg``), accuracy is also broken out by weather — the §6
robustness slice.

Runs as its own process with a single DataLoader, so it avoids the
multi-loader teardown that can trip the in-training final eval on Windows.

Examples
--------
    python -m parking.eval --data data/pklot_segmented --weights weights/mobilenet.pt
    python -m parking.eval --data-dir data/pklot_segmented/val --weights weights/mobilenet.pt --workers 8
    # if you ever hit a DataLoader worker error, add: --workers 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

_WEATHERS = ("Sunny", "Cloudy", "Rainy")


def parse_weather(path: str) -> str | None:
    """Extract Sunny/Cloudy/Rainy from a crop filename, or None if absent."""
    name = os.path.basename(path).lower()
    for w in _WEATHERS:
        if w.lower() in name:
            return w
    return None


def compute_metrics(preds, labels) -> dict:
    """Confusion counts + rates. Convention: occupied=1 is the positive class,
    empty=0. Pure NumPy so it's unit-testable without torch."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    n = int(len(labels))
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n_occ, n_emp = tp + fn, tn + fp
    return {
        "n": n,
        "acc": (tp + tn) / max(n, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision_occupied": tp / max(tp + fp, 1),
        "recall_occupied": tp / max(tp + fn, 1),
        "false_available_rate": fn / max(n_occ, 1),  # says empty, is full (§6)
        "false_occupied_rate": fp / max(n_emp, 1),   # says full, is empty
    }


def _print_report(m: dict, split_dir: str, weights: str) -> None:
    print(f"\nEvaluated {m['n']} imgs from {split_dir}")
    print(f"weights: {weights}\n")
    print(f"  accuracy           {m['acc']:.4f}")
    print(f"  precision(occ)     {m['precision_occupied']:.4f}")
    print(f"  recall(occ)        {m['recall_occupied']:.4f}")
    print(f"  false-available    {m['false_available_rate']:.4f}  (says empty, is full - the expensive error)")
    print(f"  false-occupied     {m['false_occupied_rate']:.4f}  (says full, is empty)")
    print("\n  confusion (rows=true, cols=pred):")
    print(f"                pred:empty   pred:occ")
    print(f"    true:empty  {m['tn']:>10}  {m['fp']:>9}")
    print(f"    true:occ    {m['fn']:>10}  {m['tp']:>9}")


def build_eval_transform(input_size):
    from torchvision import transforms

    w, h = input_size
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def run(weights, split_dir, input_size=(96, 96), batch_size=256, workers=8, device=None):
    """Load the checkpoint, run inference over split_dir, return (preds, labels,
    sample_paths) as NumPy arrays / list."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    from parking.classifiers.cnn import build_backbone

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = ImageFolder(split_dir, transform=build_eval_transform(input_size))
    assert ds.classes == ["empty", "occupied"], (
        f"class dirs must be exactly ['empty','occupied']; got {ds.classes}"
    )

    model = build_backbone("mobilenetv3_small", num_classes=2)
    state = torch.load(weights, map_location="cpu")
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.to(device).eval()

    loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    if workers > 0:
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(ds, **loader_kwargs)

    n_batches = len(loader)
    all_preds = np.empty(len(ds), dtype=np.int64)
    pos = 0
    print(f"device={device}  {len(ds)} imgs  ({n_batches} batches)", flush=True)
    with torch.inference_mode():
        for bi, (x, _) in enumerate(loader, 1):
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            all_preds[pos:pos + len(pred)] = pred
            pos += len(pred)
            if bi % 200 == 0 or bi == n_batches:
                print(f"  [{bi}/{n_batches}]", flush=True)

    labels = np.array([lbl for _, lbl in ds.samples], dtype=np.int64)
    paths = [p for p, _ in ds.samples]
    return all_preds, labels, paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="parking.eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="checkpoint from parking.train")
    p.add_argument("--data", default=None, help="ImageFolder root; evaluates <root>/<split>")
    p.add_argument("--split", default="val", help="split under --data (default: val)")
    p.add_argument("--data-dir", default=None, help="evaluate this dir directly (overrides --data/--split)")
    p.add_argument("--input", nargs=2, type=int, default=(96, 96), metavar=("W", "H"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8, help="loader workers; use 0 if you hit a worker error")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = p.parse_args(argv)

    if args.data_dir:
        split_dir = args.data_dir
    elif args.data:
        split_dir = str(Path(args.data) / args.split)
    else:
        p.error("provide --data (with --split) or --data-dir")

    preds, labels, paths = run(
        args.weights, split_dir, tuple(args.input), args.batch_size, args.workers, args.device
    )

    m = compute_metrics(preds, labels)
    _print_report(m, split_dir, args.weights)

    # §6 weather robustness slice, when filenames carry weather (prepare_pklot).
    weathers = np.array([parse_weather(pp) or "" for pp in paths])
    if any(weathers != ""):
        print("\n  by weather:")
        for w in _WEATHERS:
            mask = weathers == w
            if mask.any():
                acc_w = float((preds[mask] == labels[mask]).mean())
                print(f"    {w:<7} {acc_w:.4f}  ({int(mask.sum())} imgs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
