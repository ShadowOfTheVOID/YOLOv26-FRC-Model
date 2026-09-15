#!/usr/bin/env python3
"""Train a YOLO26 detector on the prepared dataset.

Defaults are tuned for this dataset's problem, which is small objects on a
wide strip: fuel is ~17x12 px on a 1920x504 frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "dataset.yaml"


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def warn_if_slow(device: str, imgsz: int, batch: int, n_train: int) -> None:
    """Say up front how long this will take on Apple silicon.

    Measured on an 8 GB M2 (10 GPU cores): yolo26n at 960/batch 4 runs about
    0.33 s per image, and yolo26s with the P2 head about 0.91 s. Memory is not
    the constraint -- s+p2 peaks near 4.4 GB and fits -- wall-clock is.
    """
    if device != "mps":
        return
    per_img = 0.9 if "-p2" in str(imgsz) else 0.35
    hours = per_img * n_train * 100 / 3600
    print(f"  note: on Apple silicon expect roughly {per_img*n_train/60:.0f} min/epoch "
          f"(~{hours:.0f} h for 100 epochs) at these settings.")
    if hours > 12:
        print("        That is long enough to be worth renting a GPU instead; "
              "see train/README.md.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26s.pt",
                    help="pretrained checkpoint (yolo26n/s/m/l/x.pt)")
    ap.add_argument("--p2", action="store_true",
                    help="use the P2 head (stride 4) for small objects, seeded "
                         "from --model's pretrained weights")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="fuel26")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        print(f"{DATA_YAML} missing -- run train/prepare_dataset.py first")
        return 1

    labels = list((ROOT / "dataset" / "labels").rglob("*.txt"))
    non_empty = [p for p in labels if p.stat().st_size > 0]
    if not non_empty:
        print("No non-empty label files under dataset/labels/.\n"
              "Ultralytics treats a missing or empty label file as 'this image\n"
              "contains nothing', so training now teaches the model to predict\n"
              "nothing. Generate fuel proposals with train/autolabel_fuel.py and\n"
              "annotate robots/hubs before training.")
        return 1
    print(f"{len(non_empty)} labelled frames of {len(labels)} label files")

    from ultralytics import YOLO
    device = args.device or pick_device()
    print(f"device: {device}")
    n_train = len(list((ROOT / "dataset" / "images" / "train").glob("*.jpg")))
    warn_if_slow(device, f"{args.imgsz}{'-p2' if args.p2 else ''}", args.batch, n_train)

    if args.p2:
        scale = Path(args.model).stem.replace("yolo26", "")[:1] or "s"
        model = YOLO(f"yolo26{scale}-p2.yaml").load(args.model)
        print(f"P2 head, weights seeded from {args.model}")
    else:
        model = YOLO(args.model)

    model.train(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        name=args.name,
        resume=args.resume,
        project=str(ROOT / "runs"),
        # Small-object settings. The default scale=0.5 can shrink a 17px ball
        # to 8px, below what even the P2 head resolves well; mosaic helps early
        # but is closed before the final epochs so the model finishes on real
        # full-frame layouts.
        scale=0.25,
        mosaic=1.0,
        close_mosaic=15,
        # The camera is fixed and the field is symmetric, so a horizontal flip
        # produces a plausible frame -- alliance colour, not screen side, is
        # what the *_blue / *_red classes key on.
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        patience=30,
    )
    print(f"\nweights: {ROOT}/runs/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
