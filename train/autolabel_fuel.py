#!/usr/bin/env python3
"""Propose YOLO boxes for fuel balls by colour, so you never hand-draw 127k of them.

A frame holds roughly 240 balls. Annotating those by hand across even one
match is not a real option, and it doesn't have to be: fuel is a uniform
saturated yellow sphere on a grey field, which classical CV separates cleanly.
Measured on the calibration match, isolated balls come out at a median
area/bbox fill of 0.79 -- a circle is 0.785 -- so the fill ratio itself tells
you whether a blob is one ball or several stuck together.

These are PROPOSALS. Review them in a labelling tool before training; the
dense mid-field pile is deliberately left unlabelled (see --keep-clumps).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
FUEL_CLASS = 0

# HSV gate for fuel yellow (OpenCV hue is 0-179).
HSV_LO = (18, 90, 90)
HSV_HI = (38, 255, 255)


def detect(img: np.ndarray, min_area: int, max_area: int, min_fill: float,
           keep_clumps: bool) -> list:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 5 or h < 4:
            continue
        fill = area / float(w * h)
        clump = area > max_area or fill < min_fill
        if clump and not keep_clumps:
            # Balls heaped together can't be separated into individual boxes
            # by thresholding, and a box drawn round the whole heap teaches the
            # detector that "fuel" is a big amorphous blob. Skipping is safer.
            continue
        boxes.append((x, y, w, h))
    return boxes


def to_yolo(boxes, W: int, H: int) -> str:
    lines = []
    for x, y, w, h in boxes:
        lines.append(f"{FUEL_CLASS} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                     f"{w / W:.6f} {h / H:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-area", type=int, default=60)
    ap.add_argument("--max-area", type=int, default=3000)
    ap.add_argument("--min-fill", type=float, default=0.62,
                    help="area/bbox below this is a merged clump (circle = 0.785)")
    ap.add_argument("--keep-clumps", action="store_true",
                    help="also box merged heaps (usually makes the dataset worse)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames")
    ap.add_argument("--preview", type=Path, default=None,
                    help="write one annotated frame here and stop")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing label files (default: skip them, so "
                         "hand-corrected labels are never clobbered)")
    args = ap.parse_args()

    images = sorted(p for split in ("train", "val")
                    for p in (DATASET / "images" / split).glob("*.jpg"))
    if not images:
        print(f"no images under {DATASET}/images -- run prepare_dataset.py first")
        return 1

    if args.preview:
        src = images[len(images) // 2]
        img = cv2.imread(str(src))
        boxes = detect(img, args.min_area, args.max_area, args.min_fill,
                       args.keep_clumps)
        for x, y, w, h in boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(img, f"{len(boxes)} fuel proposals", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), img)
        print(f"{src.name}: {len(boxes)} proposals -> {args.preview}")
        return 0

    if args.limit:
        images = images[:args.limit]

    written = skipped = total = 0
    for src in images:
        split = src.parent.name
        dst = DATASET / "labels" / split / f"{src.stem}.txt"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = detect(img, args.min_area, args.max_area, args.min_fill,
                       args.keep_clumps)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(to_yolo(boxes, W, H))
        written += 1
        total += len(boxes)

    print(f"wrote {written} label files ({skipped} skipped as already present)")
    if written:
        print(f"{total} fuel proposals, {total/written:.0f} per frame")
    print("\nThese are proposals for class 0 (fuel) only. robot_blue/robot_red/"
          "hub_blue/hub_red still need a human.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
