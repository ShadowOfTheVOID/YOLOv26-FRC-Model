#!/usr/bin/env python3
"""Derive a smaller dataset from the labelled one, for a model with a job.

The five classes exist for the scouting detector, which wants to know where
the robots are. `tbavid/count.py` does not: it looks at fuel and at the two
hubs, and a robot box is something it ignores on every frame of every match.

That is not free. Two extra classes are two more things the head predicts, two
more sources of confusion for the ones that matter, and -- the part that counts
at a scrimmage -- work done on every frame for an output nothing reads. A
counter that cannot keep up with the camera misses balls silently, so a model
that spends a third of its attention on robots is not a neutral choice.

So: label once, derive twice. This reads the built dataset and writes a second
one next to it with only the classes named, their indices remapped to the new
order, and images that still have a box kept.

    python3 train/subset_classes.py --classes fuel,hub_blue,hub_red

## The remap is the whole risk

A label file is `<class index> cx cy w h`. Drop `robot_blue` (index 1) and
`hub_blue` stops being index 3 and becomes index 1. Get that wrong and nothing
fails -- the model trains happily, on fuel labelled as hubs, and the first sign
is a scoreboard counting nonsense at a scrimmage where nothing else is
counting. `remap_labels` below is pure and tested for exactly that reason.

## Frames with nothing left

An image whose only boxes were robots has no labels in the subset. Ultralytics
reads an empty label file as "this image contains nothing", which is a real and
useful thing to train on -- background -- but an image that is *mostly* an
unlabelled hub is not background, it is a missing label. Keeping every such
frame would teach the model that hubs are not there. So a frame that loses all
of its boxes is dropped by default, and `--keep-empty` puts it back for
whoever wants genuine negatives.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset"

# The canonical order, matching prepare_dataset.CLASSES and tbavid.detect.
CLASSES = ["fuel", "robot_blue", "robot_red", "hub_blue", "hub_red"]

# What tbavid/count.py actually reads.
COUNTING = ["fuel", "hub_blue", "hub_red"]


def index_map(keep: Sequence[str],
              source: Sequence[str] = CLASSES) -> Dict[int, int]:
    """{old index: new index} for the kept classes, in the order given.

    Keyed on the SOURCE order, because that is what the label files on disk
    were written against.
    """
    unknown = [c for c in keep if c not in source]
    if unknown:
        raise SystemExit(f"not classes in this dataset: {', '.join(unknown)}\n"
                         f"  known: {', '.join(source)}")
    return {source.index(c): i for i, c in enumerate(keep)}


def remap_labels(text: str, mapping: Dict[int, int]) -> Tuple[str, int]:
    """Rewrite one label file's class indices. Returns (text, boxes kept).

    A row whose class is not kept is dropped. A row that is malformed is also
    dropped rather than passed through: a label file is machine-written, so a
    line that does not parse is corruption, and carrying it into a second
    dataset would hide where it came from.
    """
    out: List[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls not in mapping:
            continue
        out.append(" ".join([str(mapping[cls])] + parts[1:]))
    return ("\n".join(out) + "\n" if out else ""), len(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default=",".join(COUNTING),
                    help="comma-separated, in the order the new model should "
                         f"have them (default: {','.join(COUNTING)}, which is "
                         "what tbavid/count.py reads)")
    ap.add_argument("--src", type=Path, default=SRC, help="the built dataset")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write it (default: <src>-<first class>)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep frames that lose all their boxes, as background "
                         "negatives. Off by default: a frame whose hub is now "
                         "unlabelled is a missing label, not a negative.")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking")
    args = ap.parse_args()

    keep = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not keep:
        raise SystemExit("--classes cannot be empty")
    mapping = index_map(keep)
    out = args.out or args.src.parent / f"{args.src.name}-{keep[0]}"

    if not (args.src / "images").exists():
        raise SystemExit(f"no dataset at {args.src} -- run "
                         f"train/prepare_dataset.py first")

    if out.exists():
        shutil.rmtree(out)
    kept = {"train": 0, "val": 0}
    dropped = {"train": 0, "val": 0}
    boxes = 0

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((args.src / "images" / split).glob("*.jpg")):
            label = args.src / "labels" / split / f"{img.stem}.txt"
            text = label.read_text() if label.exists() else ""
            new_text, n = remap_labels(text, mapping)
            if n == 0 and not args.keep_empty:
                dropped[split] += 1
                continue
            boxes += n
            dst = out / "images" / split / img.name
            src_real = img.resolve()
            if args.copy:
                shutil.copy2(src_real, dst)
            else:
                try:
                    dst.symlink_to(src_real)
                except OSError:
                    shutil.copy2(src_real, dst)
            (out / "labels" / split / f"{img.stem}.txt").write_text(new_text)
            kept[split] += 1

    yaml = out / "dataset.yaml"
    yaml.write_text(
        "# Generated by train/subset_classes.py\n"
        f"# Derived from {args.src} -- class indices are REMAPPED, so a model\n"
        "# trained here does not share indices with the five-class one.\n"
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(keep))
    )

    print(f"classes: {', '.join(f'{i}:{n}' for i, n in enumerate(keep))}")
    print(f"train={kept['train']}  val={kept['val']}  boxes={boxes}")
    total_dropped = sum(dropped.values())
    if total_dropped:
        print(f"dropped {total_dropped} frame(s) that had no box left"
              f"{' (--keep-empty keeps them as negatives)' if not args.keep_empty else ''}")
    if not kept["val"]:
        print("\n!! no val frames survived. Every metric from this dataset is "
              "meaningless;\n   label some frames that contain these classes "
              "in the val split first.")
    print(f"\nwrote {yaml}")
    print(f"train it with:  python3 train/train.py --data {yaml} "
          f"--name count26")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
