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

`rfdetr_view()` builds the second beside the dataset (`<dataset>-rfdetr/`)
as real folders holding one link per file, so nothing is copied and the
labels stay the single source of truth. Two things ruled out simpler views:

  * linking whole folders: RF-DETR 1.12 resolves each split folder and
    refuses one that lands outside the dataset root ("YOLO split directory
    resolves outside the dataset root"); a folder of file links stays inside.
  * putting the view under runs/: the Colab notebook keeps runs/ in Google
    Drive, which cannot store links at all.

Resolution must be a multiple of patch_size * num_windows (32 for nano,
small, medium). The default 960 matches train.py's, chosen there because fuel
is ~17x12 px; RF-DETR's memory grows faster with resolution than YOLO's, so
lower --batch before lowering --imgsz.

    .venv-train/bin/pip install "rfdetr[train]>=1.12"   # 1.5 also works
    .venv-train/bin/python train/train_rfdetr.py --data dataset-scout/dataset.yaml \
        --size small --epochs 50
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_classes import yaml_names  # noqa: E402  (stdlib-only helper)

ROOT = Path(__file__).resolve().parent.parent
SIZES = ("nano", "small", "medium", "base", "large")
SPLITS = {"train": "train", "val": "valid"}  # ours -> RF-DETR's


def view_dir(data_yaml: Path) -> Path:
    """Where the RF-DETR view of a dataset goes: beside it, on the same disk."""
    src = Path(data_yaml).resolve().parent
    return src.parent / (src.name + "-rfdetr")


def rfdetr_view(data_yaml: Path, out: Path) -> list:
    """Link every file of our images/<split>, labels/<split> into RF-DETR's
    layout under `out` (real folders, one link per file), write its
    data.yaml, and return the class names. Re-running refreshes the links."""
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
            folder = out / theirs / kind
            if folder.is_symlink():          # a view from an older version
                folder.unlink()
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            for f in target.iterdir():
                if f.is_file():
                    (folder / f.name).symlink_to(f.resolve())
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


def checkpoint_names(ckpt: dict) -> list:
    """Class names, 0-based, from either checkpoint format: RF-DETR 1.5 keeps
    an argparse Namespace under "args", 1.12 (Lightning) a dict."""
    args = ckpt.get("args")
    names = args.get("class_names") if isinstance(args, dict) else getattr(args, "class_names", None)
    if not names:
        raise SystemExit("checkpoint records no class names -- was it trained by train_rfdetr.py?")
    return list(names)


def load_rfdetr(weights, size: str = ""):
    """A trained RF-DETR plus its class names, 0-based like our label files.

    Two traps this exists for. RF-DETR's `model.class_names` property returns
    {1: first, 2: second, ...} -- 1-based, a COCO habit -- while a model
    trained from YOLO labels predicts class_id 0 for the first class. Indexing
    one with the other shifted every class by one: on the smoke-test model a
    red robot came out as robot_blue, which the bumper reader would then have
    matched against the wrong alliance. And the size and resolution must match
    training, so they come from the checkpoint rather than being passed in:
    1.12 has RFDETR.from_checkpoint for that; 1.5 records the pretrained
    weights' name and the resolution in its args.
    """
    import torch
    import rfdetr
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    names = checkpoint_names(ckpt)
    if hasattr(rfdetr, "RFDETR") and hasattr(rfdetr.RFDETR, "from_checkpoint") and not size:
        return rfdetr.RFDETR.from_checkpoint(str(weights), trust_checkpoint=True), names
    args = ckpt["args"]
    get = args.get if isinstance(args, dict) else (lambda k, d=None: getattr(args, k, d))
    mc = ckpt.get("model_config") or {}
    size = size or size_from_pretrain(get("pretrain_weights", "") or mc.get("pretrain_weights", ""))
    resolution = int(get("resolution", 0) or mc.get("resolution", 0) or 560)
    cls = getattr(rfdetr, "RFDETR" + size.capitalize())
    return cls(pretrain_weights=str(weights), resolution=resolution), names


def detection_names(class_ids, names: list, library_names=None) -> list:
    """A class name per RF-DETR detection, or None for one to drop.

    RF-DETR 1.12 names its detections itself (detections.data["class_name"])
    and, at low thresholds, returns the head's extra no-object slot as
    "__background__" -- on the smoke-test model, id 2 of a 2-class head, 42 of
    300 boxes at threshold 0.01. Indexed as a class it would have become a
    class called "2", and the bumper reader would have asked Moondream to read
    a bumper on nothing. 1.5 has no names in the data; its ids are 0-based
    into the checkpoint's list, and an id past the end is the same slot.
    """
    out = []
    for i, c in enumerate(class_ids):
        if library_names is not None and len(library_names) > i:
            n = str(library_names[i])
            out.append(None if n.startswith("__") else n)
        else:
            out.append(names[int(c)] if 0 <= int(c) < len(names) else None)
    return out


QUERY_PARAMS = ("refpoint_embed.weight", "query_feat.weight")


def expand_queries(src, dst, target: int, noise: float = 0.05, seed: int = 0) -> int:
    """Write a copy of pretrained weights with `target` query slots per group.

    Why: RF-DETR small answers with at most 300 objects per image, and
    Einstein frames average ~170 fuel with the crowded ones past 300
    (82,581 boxes in 487 validation frames, 2026-11-16). Asking for 900
    queries, RF-DETR 1.12's loader refuses the 300-query pretrained weights
    ("size mismatch for refpoint_embed.weight: [3900, 4] vs [11700, 4]");
    it can only shrink the count, never grow it.

    So each training group's 300 pretrained slots are tiled up to `target`,
    keeping the per-group layout the loader expects (group g owns rows
    g*n .. (g+1)*n). Copies of a query would answer identically, so the
    copies' reference points get gaussian noise of `noise` x that tensor's
    spread; the originals are unchanged. The checkpoint's args record the
    new count, so the loader takes it as it is. Returns the group count.
    """
    import torch
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    args = ck.get("args")
    get = (lambda k: args.get(k)) if isinstance(args, dict) else (lambda k: getattr(args, k, None))
    n = int(get("num_queries") or 300)
    model = ck["model"]
    rows = model[QUERY_PARAMS[0]].shape[0]
    groups = int(get("group_detr") or rows // n)
    if rows != n * groups:
        raise SystemExit(f"{src}: {rows} query rows is not {n} queries x {groups} groups")
    gen = torch.Generator().manual_seed(seed)
    for name in list(model):
        if not name.endswith(QUERY_PARAMS):
            continue
        t = model[name].reshape(groups, n, -1)
        reps = -(-target // n)
        big = t.repeat(1, reps, 1)[:, :target].clone()
        if name.endswith("refpoint_embed.weight") and target > n:
            big[:, n:] += torch.randn(big[:, n:].shape, generator=gen) * noise * float(t.std())
        model[name] = big.reshape(groups * target, -1)
    if isinstance(args, dict):
        args.update(num_queries=target, num_select=target, group_detr=groups)
    elif args is not None:
        for k, v in (("num_queries", target), ("num_select", target), ("group_detr", groups)):
            setattr(args, k, v)
    torch.save(ck, str(dst))
    return groups


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
    ap.add_argument("--queries", type=int, default=0,
                    help="most objects one image can hold (RF-DETR small: 300). Einstein frames "
                         "average ~170 fuel and the crowded ones pass 300, so 900 for fuel")
    ap.add_argument("--scheduler", choices=("default", "cosine"), default="default",
                    help="cosine decays the learning rate over the run; the default step "
                         "schedule drops it at epoch 100, so a 50-epoch run never decays")
    args = ap.parse_args()

    check_resolution(args.size, args.imgsz)
    view = view_dir(args.data)
    names = rfdetr_view(args.data, view)
    print(f"RF-DETR {args.size} @ {args.imgsz}: classes {names}; dataset view {view}")

    import rfdetr  # late: the layout check above runs without torch
    cls = getattr(rfdetr, "RFDETR" + args.size.capitalize())
    kw = {"resolution": args.imgsz}
    train_kw = {}
    if args.queries:     # also on --resume: the model must be built with the checkpoint's count
        # A DETR answers with a fixed number of queries; an image with more
        # objects than that cannot be fully boxed however well it learns.
        # Start from pretrained weights grown to that many (expand_queries).
        base = cls(**kw)        # downloads the size's pretrained weights if needed
        src = Path(str(base.model_config.pretrain_weights))
        dst = ROOT / "runs" / args.name / f"{src.stem}-q{args.queries}.pth"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            g = expand_queries(src, dst, args.queries)
            print(f"pretrained queries 300 -> {args.queries} per group ({g} groups): {dst}")
        del base
        kw.update(num_queries=args.queries, num_select=args.queries, pretrain_weights=str(dst))
        train_kw["eval_max_dets"] = max(args.queries, 500)
    if args.scheduler == "cosine":
        train_kw["lr_scheduler"] = "cosine"
    if args.device:
        kw["device"] = args.device
    model = cls(**kw)
    model.train(dataset_dir=str(view), dataset_file="yolo",
                epochs=args.epochs, batch_size=args.batch,
                grad_accum_steps=args.grad_accum, num_workers=args.workers,
                output_dir=str(ROOT / "runs" / args.name),
                class_names=names, resume=args.resume or None, **train_kw,
                **({"device": args.device} if args.device else {}))
    print(f"weights: {ROOT / 'runs' / args.name}/checkpoint_best_total.pth")


if __name__ == "__main__":
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    main()
