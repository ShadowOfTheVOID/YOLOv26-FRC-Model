#!/usr/bin/env python3
"""Remove frames that are not the main camera: crowd shots, replays, closeups.

The harvester cuts each video to main-camera footage, and some frames survive
that are not. A crowd shot is the costly one. It has people in yellow shirts
in it, so the fuel labeller proposes boxes on them -- 49 on one measured frame
-- and every quality check in autolabel_fuel.py passes it, because those checks
ask whether the visible yellow ended up inside a box and on that frame it did.
Nothing there can know the yellow is a T-shirt.

Colour cannot answer this and neither can shape. What does: **the camera does
not move within a match**. Every field frame of a match is nearly the same
picture, so the match's own median frame is what the field looks like, and a
frame that differs wildly from it is a different shot rather than a bad one.

The score is the mean absolute difference from that median, on a 64x24
greyscale thumbnail, turned into a modified z-score against the match's own
spread. That makes the threshold mean the same thing on a dim broadcast as a
bright one -- "unlike this match's other frames" rather than any fixed number
of grey levels.

    python3 train/drop_offcamera.py --dry-run     # list what would go
    python3 train/drop_offcamera.py               # move them to dataset/skipped/

Run it BEFORE autolabel_fuel.py: labelling a crowd shot writes a file full of
boxes on spectators, and that file is what training reads.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
THUMB = (64, 24)

# 2026nhdur_qm7_TcNs-iAbOa0_000003.jpg -> 2026nhdur_qm7_TcNs-iAbOa0
FRAME_RE = re.compile(r"^(.*)_(\d+)$")


def match_of(path: Path) -> str:
    m = FRAME_RE.match(path.stem)
    return m.group(1) if m else path.stem


def scores(files: list) -> list:
    """Modified z-score of each frame's distance from the match median.

    Median and MAD rather than mean and standard deviation, because the
    outliers being looked for are exactly what would drag a mean: a match with
    six crowd shots in it would raise its own bar until they looked normal.
    """
    thumbs = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        thumbs.append(None if img is None else
                      cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32))
    good = [t for t in thumbs if t is not None]
    if len(good) < 8:
        return [0.0] * len(files)      # too few frames to know what normal is
    median = np.median(np.stack(good), axis=0)
    raw = [float(np.abs(t - median).mean()) if t is not None else 0.0 for t in thumbs]
    centre = float(np.median(raw))
    mad = float(np.median([abs(r - centre) for r in raw])) or 1e-6
    return [0.6745 * (r - centre) / mad for r in raw]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="modified z-score above which a frame is a different "
                         "shot. Lower drops more")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be moved and change nothing")
    ap.add_argument("--out", type=Path, default=DATASET / "skipped" / "offcamera")
    args = ap.parse_args()

    total = moved = 0
    for split in ("train", "val"):
        images = sorted((DATASET / "images" / split).glob("*.jpg"))
        if not images:
            continue
        by_match = {}
        for f in images:
            by_match.setdefault(match_of(f), []).append(f)
        for match, files in sorted(by_match.items()):
            total += len(files)
            zs = scores(files)
            odd = [(f, z) for f, z in zip(files, zs) if z > args.threshold]
            if not odd:
                continue
            print(f"{match}  ({split})  {len(odd)} of {len(files)} frames are a "
                  f"different shot:")
            for f, z in sorted(odd, key=lambda p: -p[1])[:6]:
                print(f"    z={z:6.1f}  {f.name}")
            if len(odd) > 6:
                print(f"    ... and {len(odd) - 6} more")
            if args.dry_run:
                moved += len(odd)
                continue
            dest = args.out / split
            dest.mkdir(parents=True, exist_ok=True)
            for f, _ in odd:
                f.rename(dest / f.name)
                lab = DATASET / "labels" / split / f"{f.stem}.txt"
                if lab.exists():
                    lab.unlink()
                moved += 1

    if not total:
        print("no images under dataset/images -- run prepare_dataset.py first")
        return 1
    verb = "would move" if args.dry_run else "moved"
    print(f"\n{verb} {moved} of {total} frames"
          + ("" if args.dry_run else f" to {args.out}"))
    if moved and not args.dry_run:
        print("Run train/autolabel_fuel.py next; anything already labelled here "
              "had its label removed with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
