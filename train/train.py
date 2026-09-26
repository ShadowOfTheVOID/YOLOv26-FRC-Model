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


def repoint_dataset(data_yaml: Path) -> None:
    """Point the yaml's `path:` at wherever the dataset actually is now.

    prepare_dataset.py and subset_classes.py write an absolute `path:`, because
    Ultralytics resolves a relative one against its own datasets directory
    rather than the yaml's -- so a dataset built on a Mac says
    `path: /Users/.../dataset-fuel`, and on any other machine Ultralytics
    stops with "images not found" after the GPU has already been set up.
    The Colab and Kaggle notebooks each rewrote the line themselves; nothing
    else did, and the first run on an AMD droplet hit exactly that.

    The yaml's own directory is the right answer whenever it holds the images,
    so use it. Rewritten in place and said out loud, so the file on disk
    matches what was trained.
    """
    text = data_yaml.read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("path:"):
            continue
        current = Path(line.split(":", 1)[1].strip())
        here = data_yaml.resolve().parent
        if current.exists() or not (here / "images").is_dir():
            return
        lines[i] = f"path: {here}"
        data_yaml.write_text("\n".join(lines) + "\n")
        print(f"  dataset moved: {current} -> {here} (rewrote {data_yaml.name})")
        return


def share_by_file() -> None:
    """Pass dataloader tensors between processes by file, not by descriptor.

    PyTorch's default on Linux hands each tensor a worker produces to the
    trainer as an open file descriptor. A batch here is a lot of tensors --
    one measured batch carried 6,454 fuel labels across 16 images -- and 16
    workers each keep several batches in flight, so a busy batch pushes the
    process past the container's open-file limit (often 1024). What that
    looks like is not an error about files: it is "received 0 items of
    ancdata" and "Pin memory thread exited unexpectedly", four epochs into a
    run that was fine, on the MI300X droplet.

    The file_system strategy shares through /dev/shm instead, which the limit
    does not touch. Linux only; macOS already uses its own mechanism.
    """
    import platform
    if platform.system() != "Linux":
        return
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
    find_shm_manager_libs()


def find_shm_manager_libs() -> None:
    """Let torch_shm_manager find the ROCm libraries the trainer already has.

    file_system sharing runs torch/bin/torch_shm_manager as a separate
    program. On the second MI300X droplet's image (torch 2.12+rocm7.14) it
    could not load librocm-openblas.so.0 -- the library ships inside the
    venv at _rocm_sdk_core/lib/host-math/lib, which Python's torch finds and
    a separately started binary does not -- so every dataloader worker died
    with "no response from torch_shm_manager" before epoch 1. Putting that
    directory on LD_LIBRARY_PATH here is inherited by the helper when the
    workers start it.
    """
    import glob
    import os
    import site
    roots = site.getsitepackages() + [site.getusersitepackages()]
    dirs = sorted({os.path.dirname(p) for root in roots for p in glob.glob(
        os.path.join(root, "_rocm_sdk_*", "lib", "**", "librocm-openblas.so*"),
        recursive=True)})
    if dirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([current] if current else []))


def batch_arg(text: str):
    """`--batch 32`, `--batch 0.70`, `--batch -1`.

    Ultralytics reads a float in (0, 1) as a fraction of GPU memory to fill
    and -1 as "work it out", which is how you use a card whose memory you
    have not measured against. argparse with type=int rejected both.
    """
    value = float(text)
    return int(value) if value.is_integer() else value


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> None:
    """Say which chip this is about to run on, and shout if it is the CPU.

    ROCm reports AMD hardware through `torch.cuda`, so an MI300X and an H100
    both come back as device "0" and nothing in the log distinguishes a
    192 GB accelerator from a fallback. The failure that matters is quieter
    still: `pip install ultralytics` inside a ROCm image pulls torch from
    PyPI, which is the CUDA build, and it overwrites the ROCm one. Then
    `torch.cuda.is_available()` is False, this picks "cpu", and a run that
    should take an hour takes a week without ever saying why. Print the name
    so the first ten lines of output answer "am I on the GPU".
    """
    import torch
    if device == "cpu":
        print("  WARNING: training on the CPU.")
        if getattr(torch.version, "hip", None):
            print("           This is a ROCm build of torch but no GPU is "
                  "visible -- check `rocm-smi`, and that the container was "
                  "started with --device=/dev/kfd --device=/dev/dri.")
        else:
            print(f"           torch {torch.__version__} has no GPU support "
                  f"(cuda={torch.version.cuda}, hip=None). On an AMD box this "
                  f"usually means a PyPI wheel replaced the ROCm one; see "
                  f"deploy/AMD_DEVCLOUD.md.")
        return
    if device == "mps":
        return
    for idx in [d for d in device.split(",") if d.strip().isdigit()]:
        try:
            name = torch.cuda.get_device_name(int(idx))
            mem = torch.cuda.get_device_properties(int(idx)).total_memory
        except Exception:
            continue
        print(f"  gpu {idx}: {name} ({mem / 1024**3:.0f} GB)")
    if getattr(torch.version, "hip", None):
        print(f"  ROCm/HIP {torch.version.hip}")


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
    ap.add_argument("--batch", type=batch_arg, default=4,
                    help="images per step. An integer, or a fraction of GPU "
                         "memory (0.70), or -1 to autodetect. 4 suits an 8 GB "
                         "laptop; a 192 GB MI300X wants far more -- see "
                         "deploy/AMD_DEVCLOUD.md for why bigger is not always "
                         "better on a dataset this small")
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader workers. The default keeps a laptop "
                         "responsive; on a many-core GPU box JPEG decoding "
                         "becomes the bottleneck and this is what fixes it")
    ap.add_argument("--cache", default="",
                    help="'ram' or 'disk' to cache decoded images. 'ram' is "
                         "the single biggest speedup on a box with memory to "
                         "spare (~1.4 GB per 1000 frames at 1920x504)")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="disable mixed precision. Reach for this if the loss "
                         "goes NaN or mAP stays at zero -- a known ROCm "
                         "symptom, and it costs speed rather than accuracy")
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

    repoint_dataset(args.data)

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
    share_by_file()
    device = args.device or pick_device()
    print(f"device: {device}")
    describe_device(device)
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
        workers=args.workers,
        cache=args.cache or False,
        amp=args.amp,
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
