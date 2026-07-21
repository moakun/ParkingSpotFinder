"""Train the CNN classifier (milestone M2).

Expects an ImageFolder layout with class dirs named to match the pipeline's
label convention (alphabetical -> index): ``empty`` = 0, ``occupied`` = 1.

    <data_root>/
        train/{empty,occupied}/*.jpg
        val/{empty,occupied}/*.jpg     # IMPORTANT: a *different lot* than train

The split rule from the plan (§3) is not optional: train and val must be
different lots, or the numbers are inflated and hide the real failure mode.
Run this to produce a checkpoint, then pass it to ``psf-run --classifier cnn
--weights ...``.

    python -m parking.train --data data/pklot --epochs 8 --out weights/mobilenet.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="parking.train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="ImageFolder root with train/ and val/")
    p.add_argument("--out", default="weights/mobilenet.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--input", nargs=2, type=int, default=(96, 96), metavar=("W", "H"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    import time

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    from parking.classifiers.cnn import build_backbone

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w, h = args.input
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),  # weather/lighting robustness (§6)
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), norm])

    root = Path(args.data)
    train_ds = ImageFolder(root / "train", transform=train_tf)
    val_ds = ImageFolder(root / "val", transform=eval_tf)
    assert train_ds.classes == ["empty", "occupied"], (
        f"class dirs must be exactly ['empty','occupied']; got {train_ds.classes}"
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_backbone("mobilenetv3_small", num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    n_batches = len(train_dl)
    print(f"device={device}  classes={train_ds.classes}", flush=True)
    print(f"train={len(train_ds)} imgs ({n_batches} batches/epoch)  val={len(val_ds)} imgs  "
          f"batch={args.batch_size} lr={args.lr} epochs={args.epochs}", flush=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for bi, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
            if bi % 200 == 0 or bi == n_batches:
                ips = seen / max(time.time() - t0, 1e-6)
                print(f"  epoch {epoch} [{bi}/{n_batches}]  loss {running / seen:.4f}  "
                      f"{ips:.0f} img/s", flush=True)

        print(f"  epoch {epoch}: evaluating on {len(val_ds)} held-out val imgs...", flush=True)
        acc, false_available = _evaluate(model, val_dl, device)
        print(f"epoch {epoch:2d}  cross-lot val acc {acc:.4f}  false-available {false_available:.4f}", flush=True)
        if acc >= best_acc:
            best_acc = acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"           new best -> saved {args.out}", flush=True)

    print(f"best cross-lot acc {best_acc:.4f} -> {args.out}", flush=True)
    return 0


def _evaluate(model, loader, device):
    """Return (accuracy, false-available rate).

    false-available = predicted empty(0) but truly occupied(1) — the expensive
    error the plan (§6) says to track: it sends a driver to a full spot.
    """
    import torch

    model.eval()
    correct = total = false_avail = truly_occupied = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            occ = y == 1
            truly_occupied += occ.sum().item()
            false_avail += ((pred == 0) & occ).sum().item()
    acc = correct / max(total, 1)
    fa = false_avail / max(truly_occupied, 1)
    return acc, fa


if __name__ == "__main__":
    raise SystemExit(main())
