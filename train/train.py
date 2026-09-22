#!/usr/bin/env python3
"""Train a YOLO26 detector on the prepared dataset.

Defaults are tuned for this dataset's problem, which is small objects on a
wide strip: fuel is ~17x12 px on a 1920x504 frame.

There are two models worth training from one labelling effort, and they want
different settings:

  * **the scouting detector** -- all five classes, over harvested broadcast
    frames. Accuracy is what matters; it runs offline through `run.py detect`
    and can take as long as it likes.
  * **the counting model** -- fuel and the two hubs only, built by
    `train/subset_classes.py`, for `run.py count` at a scrimmage. There,
    SPEED is a correctness property: a model too slow for the camera misses
    balls between the frames it sees, the score comes out low, and with no FMS
    nothing else will ever notice. Train it smaller, and measure it with
    `train/benchmark.py` before the event rather than discovering it there.

`--data` points at either. `--export` writes an ONNX beside the weights, which
`run.py detect` and `run.py count` both load in place of a `.pt`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "dataset.yaml"


def dataset_root(data_yaml: Path) -> Path:
    """The dataset directory a data.yaml describes.

    Derived from the yaml's own location rather than assumed to be
    `ROOT/dataset`, so a derived set -- `dataset-fuel/`, from
    subset_classes.py -- has its labels counted instead of the five-class
    one's. That check is the only thing standing between a typo and a hundred
    epochs against an empty label set.
    """
    return Path(data_yaml).resolve().parent


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
    ap.add_argument("--data", type=Path, default=DATA_YAML,
                    help="dataset.yaml to train against (default the "
                         "five-class one; train/subset_classes.py writes the "
                         "counting one)")
    ap.add_argument("--export", default="",
                    help="also export the finished weights, e.g. onnx. "
                         "`run.py count` loads the result in place of a .pt, "
                         "and on a CPU box ONNX is usually the difference "
                         "between keeping up with the camera and not.")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"{args.data} missing -- run train/prepare_dataset.py first"
              + ("" if args.data == DATA_YAML else
                 ", then train/subset_classes.py"))
        return 1

    # Off the yaml's own directory, not ROOT/dataset: pointed at a derived set
    # this used to count the five-class one's labels and report a healthy
    # number while training against nothing.
    labels = list((dataset_root(args.data) / "labels").rglob("*.txt"))
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
        data=str(args.data),
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
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    print(f"\nweights: {best}")

    if args.export:
        # Exported from `best`, not from the in-memory model, which after
        # training is the LAST epoch rather than the best one -- exporting that
        # would quietly ship a worse model than the file beside it.
        print(f"exporting {args.export} ...")
        out = YOLO(str(best)).export(format=args.export, imgsz=args.imgsz)
        print(f"exported: {out}")
        print(f"  run.py count --weights {out}  loads this in place of the .pt")

    names = getattr(model, "names", None)
    if names and len(names) <= 3:
        print(f"\n{len(names)} classes ({', '.join(names.values())}) -- this "
              f"looks like a counting model.\n"
              f"Measure it before the event, not at it:\n"
              f"    python3 train/benchmark.py --weights {best} "
              f"--imgsz {args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
