#!/usr/bin/env python3
"""Propose YOLO boxes for fuel balls by colour, so you never hand-draw 127k of them.

A frame holds roughly 240 balls. Annotating those by hand across even one
match is not a real option, and it doesn't have to be: fuel is a uniform
saturated yellow sphere on a grey field, which classical CV separates cleanly.
Measured on the calibration match, isolated balls come out at a median
area/bbox fill of 0.79 -- a circle is 0.785 -- so the fill ratio itself tells
you whether a blob is one ball or several stuck together.

## Touching balls are the whole recall problem

Thresholding finds yellow; it does not find *balls*. Two balls resting against
each other are one connected component with a fill ratio around 0.6, and the
fill gate that correctly rejects a heap rejects that pair as well. On a real
frame most fuel is in loose groups of two to five, so gating alone proposes a
small fraction of what is there -- 92 of several hundred on the frame this was
tuned against.

So a component that fails the gate is not discarded outright, and which tool
recovers it depends on how big it is.

A few balls stuck together are cut apart on their outline: a distance
transform peaks once per ball centre, and a watershed from those peaks follows
the seams. Each piece is sized against the frame's own median ball, so a cut
that produced nonsense is dropped rather than labelled.

A cluster is a different problem, and the one that costs the most. A ball in
the middle of a cluster is surrounded by yellow, so the mask has no seam there
and no amount of cutting finds it -- which is why the heaps stayed bare while
the pairs came back. What is still visible is each ball's own shading, the
dark crescent where the next one meets it, so those are found with a Hough
circle search over the gradient, with the radius range pinned to the band's
measured ball. On a hex-packed cluster of 30, that is the difference between
0 proposals and 30.

These are PROPOSALS. Preview them before you commit to 200k of them:

    python3 train/autolabel_fuel.py --preview /tmp/check.jpg

Red boxes came through the gate, orange ones came out of a split, and the
caption counts both plus the heaps that were skipped.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
FUEL_CLASS = 0

# HSV gate for fuel yellow (OpenCV hue is 0-179). Overridable per run: stadium
# lighting and broadcast colour grading move this more than you would like.
HSV_LO = (18, 90, 90)
HSV_HI = (38, 255, 255)

# How far into a blob a distance-transform peak has to sit, as a fraction of
# one ball's radius, before it counts as a ball centre. Too low merges two
# centres into one seed; too high finds no seed in a ball that is half hidden.
PEAK_FRAC = 0.55

# Border left around a component so watershed has somewhere to flood from and
# Hough has gradient on both sides of a rim ball's edge.
PAD = 3


def component_mask(labels: np.ndarray, idx: int, box: tuple) -> np.ndarray:
    """One component, padded, as a 0/1 mask."""
    x, y, w, h = box
    sub = np.zeros((h + 2 * PAD, w + 2 * PAD), np.uint8)
    sub[PAD:PAD + h, PAD:PAD + w] = (labels[y:y + h, x:x + w] == idx).astype(np.uint8)
    return sub


def hsv_arg(text: str) -> tuple:
    """'18,90,90' -> (18, 90, 90)."""
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("want three comma-separated numbers, e.g. 18,90,90")
    return tuple(parts)


def split_clump(labels: np.ndarray, idx: int, box: tuple, img: np.ndarray,
                unit_area: float, min_area: int) -> list:
    """Cut one merged blob into per-ball boxes, or return nothing.

    The blob is padded so it has background all round it -- watershed needs
    somewhere to flood from, and a component touching its own bounding box has
    none. Seeds are the distance transform's plateau above PEAK_FRAC of a ball
    radius: one per ball centre, because a ball is convex and its centre is the
    furthest point from the blob's edge.

    Everything after the watershed is a sanity gate. A region smaller than a
    third of a ball is a sliver off the seam; one bigger than two and a half
    balls is two balls the split failed to separate. Both are dropped: a
    missing proposal costs recall, a wrong one costs the model.
    """
    x, y, w, h = box
    pad = PAD
    sub = component_mask(labels, idx, box)

    radius = math.sqrt(unit_area / math.pi)
    dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
    peaks = (dist > PEAK_FRAC * radius).astype(np.uint8)
    n_seeds, seeds = cv2.connectedComponents(peaks)
    if n_seeds <= 2:
        # One seed means the distance transform sees a single ball, and the
        # gate already had its say. Splitting it would only invent boxes.
        return []

    markers = seeds.astype(np.int32) + 1      # 1 = background, 2.. = seeds
    markers[(sub == 1) & (peaks == 0)] = 0    # unknown: the seams
    markers[sub == 0] = 1

    crop = img[y:y + h, x:x + w]
    colour = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    cv2.watershed(colour, markers)

    out = []
    for m in range(2, n_seeds + 1):
        ys, xs = np.nonzero(markers == m)
        if xs.size == 0:
            continue
        area = int(xs.size)
        if area < max(min_area * 0.5, unit_area * 0.35) or area > unit_area * 2.5:
            continue
        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max()) - bx + 1, int(ys.max()) - by + 1
        if bw < 5 or bh < 4:
            continue
        out.append((x + bx - pad, y + by - pad, bw, bh))
    return out


def unit_areas(singles: list, height: int, bands: int = 6) -> list:
    """The area of one ball, per horizontal band of the frame.

    Not one number for the whole frame: the camera looks down the field, so the
    same ball is ~25 px across at the near rail and ~12 px at the far wall. A
    single global median calls every near ball a merged pair and every far pair
    a ball. Bands follow the perspective cheaply, since on this camera depth is
    essentially the y coordinate.

    A band with too few singles to measure borrows the frame's median rather
    than inventing one.
    """
    overall = float(np.median([a for *_, a in singles])) if singles else 0.0
    step = max(height // bands, 1)
    out = []
    for b in range(bands):
        lo, hi = b * step, (b + 1) * step if b < bands - 1 else height
        areas = [a for x, y, w, h, a in singles if lo <= y + h / 2 < hi]
        out.append(float(np.median(areas)) if len(areas) >= 5 else overall)
    return out


def unit_at(units: list, y: int, height: int) -> float:
    step = max(height // len(units), 1)
    return units[min(int(y) // step, len(units) - 1)]


def hough_split(img: np.ndarray, sub: np.ndarray, box: tuple, pad: int,
                unit_area: float) -> list:
    """Find balls inside a clump by their circular shading, not by its outline.

    The distance transform cuts a blob where it is narrow, which finds the
    seams on the rim of a group and nothing at all in the middle of one: a ball
    surrounded by other balls has no edge, so no distance minimum, so no seam
    to cut. That is why a heap stayed unlabelled however the gate was tuned,
    and it is most of the fuel on a real frame -- the field's clusters hold
    more balls than its open floor.

    What is still visible in the middle of a cluster is each ball's own
    shading: a dark crescent where the next ball meets it. Hough sees that,
    because it works on gradients rather than on the mask. The radius range is
    taken from the band's measured ball, so the accumulator is not free to
    invent a circle the size of a hub.
    """
    x, y, w, h = box
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        return []
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grey = cv2.copyMakeBorder(grey, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    grey = cv2.medianBlur(grey, 3)

    radius = math.sqrt(unit_area / math.pi)
    circles = cv2.HoughCircles(
        grey, cv2.HOUGH_GRADIENT, dp=1,
        minDist=max(radius * 1.2, 4.0),
        param1=90, param2=max(int(radius * 0.9), 9),
        minRadius=max(int(radius * 0.55), 3),
        maxRadius=max(int(radius * 1.6), 6))
    if circles is None:
        return []

    out = []
    for cx, cy, r in np.round(circles[0]).astype(int):
        if not (0 <= cy < sub.shape[0] and 0 <= cx < sub.shape[1]) or not sub[cy, cx]:
            continue          # a circle centred off the yellow is not a ball
        bx, by = cx - r, cy - r
        bw = bh = 2 * r
        if bw < 5 or bh < 4:
            continue
        out.append((x + bx - pad, y + by - pad, bw, bh))
    return out


def detect(img: np.ndarray, min_area: int, max_area: int, min_fill: float,
           keep_clumps: bool, split: bool = True, max_split: int = 8,
           hsv_lo: tuple = HSV_LO, hsv_hi: tuple = HSV_HI,
           merge_factor: float = 1.6, hough: bool = True) -> tuple:
    """-> (gated boxes, boxes recovered by splitting, heaps skipped).

    Three return values rather than one list because the preview needs to
    distinguish them and so do you: a frame where most proposals came out of
    splits is a frame worth looking at before trusting.

    Two things get a component sent to the splitter. The obvious one is failing
    the fill gate -- a ragged outline is several balls. The one that matters
    more is being too BIG for its part of the frame: two balls side by side
    fill their bounding box to 0.82, sail through a gate set at 0.62, and get
    labelled as one ball twice the size of its neighbours. Fill cannot see
    that; area against the local median can.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    singles, clumps = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 5 or h < 4:
            continue
        fill = area / float(w * h)
        if area <= max_area and fill >= min_fill:
            singles.append((int(x), int(y), int(w), int(h), int(area), i))
        else:
            clumps.append((i, int(x), int(y), int(w), int(h), int(area)))

    if keep_clumps:
        # The old escape hatch: box the heap as one thing. Still usually makes
        # the dataset worse, still here for whoever wants to see it.
        return ([(x, y, w, h) for x, y, w, h, _, _ in singles],
                [(x, y, w, h) for _, x, y, w, h, _ in clumps], 0)

    if not split or not singles:
        return [(x, y, w, h) for x, y, w, h, _, _ in singles], [], len(clumps)

    H = img.shape[0]
    units = unit_areas([(x, y, w, h, a) for x, y, w, h, a, _ in singles], H)

    gated, recovered, heaps = [], [], 0

    # Second look at everything that passed: anything well over the local ball
    # size is a merge the fill gate did not notice. A failed split keeps the
    # original box -- it passed the gate, and dropping it trades a slightly
    # wrong label for no label at all.
    for x, y, w, h, area, i in singles:
        unit = unit_at(units, y + h // 2, H)
        if unit and area > merge_factor * unit:
            pieces = split_clump(labels, i, (x, y, w, h), img, unit, min_area)
            if pieces:
                recovered.extend(pieces)
                continue
        gated.append((x, y, w, h))

    for i, x, y, w, h, area in clumps:
        unit = unit_at(units, y + h // 2, H)
        if not unit:
            heaps += 1
            continue
        pieces = []
        if area <= max_split * unit:
            # A few balls stuck together: the seams are on the outline, and the
            # watershed cut is tighter than a circle fit.
            pieces = split_clump(labels, i, (x, y, w, h), img, unit, min_area)
        if not pieces and hough:
            # A cluster. Its interior balls have no outline to cut, so find
            # them by their shading instead.
            pieces = hough_split(img, component_mask(labels, i, (x, y, w, h)),
                                 (x, y, w, h), PAD, unit)
        if pieces:
            recovered.extend(pieces)
        else:
            heaps += 1
    return gated, recovered, heaps


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
    ap.add_argument("--no-split", dest="split", action="store_false",
                    help="don't try to cut merged blobs apart -- gate only, "
                         "which is what this did before and finds a fraction "
                         "of the fuel on a busy frame")
    ap.add_argument("--merge-factor", type=float, default=1.6,
                    help="a blob more than this many times the local ball area "
                         "is treated as a merge and split, even if it passed "
                         "the fill gate -- which a side-by-side pair does")
    ap.add_argument("--max-split", type=int, default=8,
                    help="biggest blob the watershed cut is used on, in "
                         "ball-areas. Bigger clusters go to the circle finder, "
                         "which is what --no-hough turns off")
    ap.add_argument("--no-hough", dest="hough", action="store_false",
                    help="don't look for balls inside a cluster by their "
                         "shading. Leaves every heap unlabelled, which is most "
                         "of the fuel on a busy frame")
    ap.add_argument("--keep-clumps", action="store_true",
                    help="box merged heaps whole (usually makes the dataset worse)")
    ap.add_argument("--hsv-lo", type=hsv_arg, default=HSV_LO,
                    help=f"lower HSV gate (default {','.join(map(str, HSV_LO))}). "
                         "Lower the S and V if a dim broadcast is losing balls "
                         "in shadow")
    ap.add_argument("--hsv-hi", type=hsv_arg, default=HSV_HI,
                    help=f"upper HSV gate (default {','.join(map(str, HSV_HI))})")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames")
    ap.add_argument("--preview", type=Path, default=None,
                    help="write one annotated frame here and stop")
    ap.add_argument("--preview-frame", default=None,
                    help="which frame to preview (filename or substring); "
                         "default is the middle one")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing label files (default: skip them, so "
                         "hand-corrected labels are never clobbered)")
    args = ap.parse_args()

    images = sorted(p for split in ("train", "val")
                    for p in (DATASET / "images" / split).glob("*.jpg"))
    if not images:
        print(f"no images under {DATASET}/images -- run prepare_dataset.py first")
        return 1

    def run(img):
        return detect(img, args.min_area, args.max_area, args.min_fill,
                      args.keep_clumps, args.split, args.max_split,
                      args.hsv_lo, args.hsv_hi, args.merge_factor, args.hough)

    if args.preview:
        src = images[len(images) // 2]
        if args.preview_frame:
            matches = [p for p in images if args.preview_frame in p.name]
            if not matches:
                print(f"no frame matching {args.preview_frame!r}")
                return 1
            src = matches[0]
        img = cv2.imread(str(src))
        gated, recovered, heaps = run(img)
        for x, y, w, h in gated:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        for x, y, w, h in recovered:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), 1)
        cv2.putText(img, f"{len(gated)} gated + {len(recovered)} split = "
                         f"{len(gated) + len(recovered)}   ({heaps} heaps skipped)",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), img)
        print(f"{src.name}: {len(gated)} gated, {len(recovered)} from splits, "
              f"{heaps} heaps skipped -> {args.preview}")
        return 0

    if args.limit:
        images = images[:args.limit]

    written = skipped = total = from_splits = heaps_total = 0
    for src in images:
        split_dir = src.parent.name
        dst = DATASET / "labels" / split_dir / f"{src.stem}.txt"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        H, W = img.shape[:2]
        gated, recovered, heaps = run(img)
        boxes = gated + recovered
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(to_yolo(boxes, W, H))
        written += 1
        total += len(boxes)
        from_splits += len(recovered)
        heaps_total += heaps

    print(f"wrote {written} label files ({skipped} skipped as already present)")
    if written:
        print(f"{total} fuel proposals, {total/written:.0f} per frame "
              f"({from_splits} of them recovered from merged blobs, "
              f"{heaps_total/written:.1f} heaps skipped per frame)")
    print("\nThese are proposals for class 0 (fuel) only. robot_blue/robot_red/"
          "hub_blue/hub_red come from train/autolabel_objects.py, which appends "
          "to these files rather than replacing them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
