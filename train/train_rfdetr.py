#!/usr/bin/env python3
"""Train RF-DETR on a dataset prepared for YOLO26, as the bake-off against it.

Why a second architecture at all: RF-DETR (Roboflow, Apache-2.0 -- no AGPL
question when an app bundles it) is a DETR with a DINOv2 backbone, which is
meant to fine-tune well from few labels -- what week 1 of a new game gives
us. Like YOLO26 it has no NMS, which matters when game pieces arrive touching
in a burst (NMS deletes boxes that overlap heavily). Whether it beats YOLO26
on ~15 px fuel is NOT known: DETR-style models have historically been weaker
on tiny objects. Compare both on held-out matches with train/benchmark.py
before trusting either.

The only work here is layout. Ours (prepare_dataset.py, subset_classes.py):

    dataset/dataset.yaml   images/{train,val}/   labels/{train,val}/

RF-DETR's YOLO reader wants:

    data.yaml   train/{images,labels}/   valid/{images,labels}/

`rfdetr_view()` builds the second as symlinks to the first, so nothing is
copied and the labels stay the single source of truth.

Resolution must be a multiple of patch_size * num_windows (32 for nano,
small, medium). The default 960 matches train.py's, chosen there because fuel
is ~17x12 px; RF-DETR's memory grows faster with resolution than YOLO's, so
lower --batch before lowering --imgsz.

    .venv-train/bin/pip install rfdetr
    .venv-train/bin/python train/train_rfdetr.py --data dataset-scout/dataset.yaml \
        --size small --epochs 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_classes import yaml_names  # noqa: E402  (stdlib-only helper)

ROOT = Path(__file__).resolve().parent.parent
SIZES = ("nano", "small", "medium", "base", "large")
SPLITS = {"train": "train", "val": "valid"}  # ours -> RF-DETR's


def rfdetr_view(data_yaml: Path, out: Path) -> list:
    """Symlink our images/<split>, labels/<split> into RF-DETR's layout under
    `out`, write its data.yaml, and return the class names."""
    src = Path(data_yaml).resolve().parent
    names = yaml_names(Path(data_yaml).read_text())
    if not names:
        raise SystemExit(f"{data_yaml}: no class names found")
    out.mkdir(parents=True, exist_ok=True)
    for ours, theirs in SPLITS.items():
        for kind in ("images", "labels"):
            target = src / kind / ours
            if not target.is_dir():
                raise SystemExit(f"{target} is missing -- run prepare_dataset.py first")
            link = out / theirs / kind
            link.parent.mkdir(exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target, target_is_directory=True)
    (out / "data.yaml").write_text(
        "train: train/images\nval: valid/images\n"
        f"nc: {len(names)}\nnames: [{', '.join(names)}]\n")
    return names


def size_from_pretrain(name: str) -> str:
    """'rf-detr-nano.pth' -> 'nano'; the checkpoint records which pretrained
    weights it started from, and that is the only place its size is written."""
    stem = Path(str(name)).name.lower()
    for size in SIZES:
        if f"-{size}" in stem:
            return size
    return "base"


def load_rfdetr(weights, size: str = ""):
    """A trained RF-DETR plus its class names, 0-based like our label files.

    Two traps this exists for. RF-DETR's `model.class_names` property returns
    {1: first, 2: second, ...} -- 1-based, a COCO habit -- while a model
    trained from YOLO labels predicts class_id 0 for the first class. Indexing
    one with the other shifted every class by one: on the smoke-test model a
    red robot came out as robot_blue, which the bumper reader would then have
    matched against the wrong alliance. And the size and resolution must match
    training, so both are read from the checkpoint rather than passed in.
    """
    import torch
    import rfdetr
    args = torch.load(str(weights), map_location="cpu", weights_only=False)["args"]
    size = size or size_from_pretrain(getattr(args, "pretrain_weights", ""))
    cls = getattr(rfdetr, "RFDETR" + size.capitalize())
    model = cls(pretrain_weights=str(weights), resolution=int(args.resolution))
    return model, list(args.class_names)


def check_resolution(size: str, imgsz: int) -> None:
    step = 56 if size in ("base", "large") else 32  # patch 14 x 4 / patch 16 x 2
    if imgsz % step:
        raise SystemExit(f"--imgsz {imgsz} must be a multiple of {step} for RF-DETR {size}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "dataset" / "dataset.yaml")
    ap.add_argument("--size", choices=SIZES, default="small")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="effective batch = batch x grad-accum (RF-DETR's recipe is 16)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps; default: RF-DETR picks")
    ap.add_argument("--name", default="rfdetr")
    ap.add_argument("--resume", default="", help="checkpoint .pth to resume from")
    args = ap.parse_args()

    check_resolution(args.size, args.imgsz)
    view = ROOT / "runs" / args.name / "dataset"
    names = rfdetr_view(args.data, view)
    print(f"RF-DETR {args.size} @ {args.imgsz}: classes {names}; dataset view {view}")

    import rfdetr  # late: the layout check above runs without torch
    cls = getattr(rfdetr, "RFDETR" + args.size.capitalize())
    kw = {"resolution": args.imgsz}
    if args.device:
        kw["device"] = args.device
    model = cls(**kw)
    model.train(dataset_dir=str(view), dataset_file="yolo",
                epochs=args.epochs, batch_size=args.batch,
                grad_accum_steps=args.grad_accum, num_workers=args.workers,
                output_dir=str(ROOT / "runs" / args.name),
                class_names=names, resume=args.resume or None,
                **({"device": args.device} if args.device else {}))
    print(f"weights: {ROOT / 'runs' / args.name}/checkpoint_best_total.pth")


if __name__ == "__main__":
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    main()
