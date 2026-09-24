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

## The two the mask cannot see at all

A ball a robot is carrying reads V=103 at its highlight and V=58 at its rim,
against a gate whose floor is 90 -- so it survives as a 7x7 dot, below
`--min-area`, and a robot holding four balls contributes nothing. Lowering the
gate to reach it takes in every yellow banner in the stands. Instead the
highlight is used as a seed and grown to the band's ball size, kept only if
what it covers is mostly yellow under a much looser gate. Hough does not work
here: a hopper bar cuts the circular gradient, and on a synthetic hopper it
found one ball of four at its loosest setting.

Reflections are the same problem inverted. The floor is glossy, so most balls
near it come with a mirrored copy below -- same hue, same size, and no colour
gate can tell them apart. Left in they roughly double the count in exactly the
places where counting matters. A reflection is dimmer than what casts it and
sits almost directly beneath it, so a proposal with a brighter one above it,
within `--reach` ball-heights and aligned to half a width, is dropped.

These are PROPOSALS. Preview them before you commit to 200k of them:

    python3 train/autolabel_fuel.py --preview /tmp/check.jpg

Red boxes came through the gate, orange ones were recovered from a cluster or
out of shade, and `--show-dropped` adds in green what the reflection filter
removed -- worth looking at, because a filter eating real balls and one working
correctly produce the same count.
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

# The same hue with the brightness and saturation floors dropped, for fuel in
# shade -- inside a hopper, under a ramp. Far too loose to threshold on: it
# takes in banners, shirts and the yellow in the stands. Nothing is proposed
# from it on its own; it only confirms what a highlight has already seeded.
LOOSE_LO = (15, 55, 45)
LOOSE_HI = (40, 255, 255)

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


def box_stats(hsv: np.ndarray, box: tuple) -> tuple:
    """(mean saturation, mean value) over a proposal, clipped to the frame."""
    x, y, w, h = box
    H, W = hsv.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    patch = hsv[y0:y1, x0:x1]
    return float(patch[..., 1].mean()), float(patch[..., 2].mean())


def drop_reflections(hsv: np.ndarray, boxes: list, v_ratio: float,
                     reach: float) -> tuple:
    """Separate real balls from their reflections in the floor.

    The field surface is glossy, so most balls near it come with a mirrored
    copy a short way below: same hue, same rough size, and a colour gate has no
    way to tell the two apart. Left in, they roughly double the count in
    exactly the places counting matters, and they teach the detector that a
    smear on the floor is fuel.

    What separates them is that a reflection is DIMMER than the ball casting
    it, and sits almost directly under it. So a proposal is dropped when
    another proposal sits above it, within `reach` ball-heights, horizontally
    aligned to within half a width, and this one's mean brightness is below
    `v_ratio` of that one's. Two real balls stacked in frame are lit alike and
    survive; a reflection is never as bright as its source.
    """
    scored = [(b, box_stats(hsv, b)) for b in boxes]
    order = sorted(range(len(scored)), key=lambda i: scored[i][0][1])  # top first
    keep, dropped = [], []
    for pos, i in enumerate(order):
        (x, y, w, h), (_, v) = scored[i]
        cx = x + w / 2
        reflection = False
        for j in order[:pos]:
            (ax, ay, aw, ah), (_, av) = scored[j]
            if ay + ah > y + h:
                continue
            gap = y - (ay + ah)
            if gap < -ah * 0.25 or gap > ah * reach:
                continue
            if abs((ax + aw / 2) - cx) > max(aw, w) * 0.5:
                continue
            if v < av * v_ratio:
                reflection = True
                break
        (dropped if reflection else keep).append((x, y, w, h))
    return keep, dropped


def rescue_pass(img: np.ndarray, hsv: np.ndarray, strict: np.ndarray,
                taken: list, units: list, loose_lo: tuple, loose_hi: tuple,
                min_cover: float, close_k: int, ball_sat: float,
                sat_ratio: float) -> list:
    """Find the balls a robot is carrying, which the colour gate cannot see.

    Measure a ball in a hopper and the reason is obvious: its bright side
    reads V=103 and its rim V=58, against a gate whose floor is 90. What
    survives the threshold is a 7x7 dot of its highlight -- below `--min-area`,
    so dropped, so a robot holding four balls contributes nothing. These are
    not incidental balls. For a counting model they are the ones that decide
    whether a score is attributed at all.

    Lowering the gate to reach them is not an option: at V=45 the mask takes in
    every yellow banner in the stands. But the highlight IS reliable, and it is
    surrounded by the same ball at lower brightness. So each unclaimed
    highlight is grown into a ball-sized box for its band, and kept only if
    that box is mostly loose-yellow.

    Why not Hough here, when Hough is what cracked the clusters: a hopper has
    bars across it, and a bar cuts the circular gradient a circle finder needs.
    It does not cut the highlight. Measured on a synthetic hopper -- four balls
    at 45% brightness behind 2 px bars -- Hough found one of four at its
    loosest setting and none at a usable one.

    ## Saturation, not brightness, is what says "fuel"

    Shading scales a pixel's VALUE and leaves its SATURATION alone: fuel
    measures S=191 in arena light and S=191 in a hopper, while the tan of the
    arena wall measures S=68 at any brightness. An absolute floor low enough
    for a shaded ball is therefore also low enough for the wall, the rail and
    every washed-out surface in the frame -- which is exactly what it proposed
    boxes on. So the test is relative: a rescued box has to be about as
    saturated as the balls this frame has already found, at any brightness.
    """
    H, W = strict.shape
    loose = cv2.inRange(hsv, np.array(loose_lo, np.uint8), np.array(loose_hi, np.uint8))
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if close_k > 1:
        # Bridge the bars. A strut thinner than a ball's radius does not divide
        # one ball into two, however much the mask says otherwise.
        loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE,
                                 np.ones((close_k, close_k), np.uint8))

    claimed = np.zeros((H, W), np.uint8)
    for x, y, w, h in taken:
        claimed[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = 1

    n, _, stats, centroids = cv2.connectedComponentsWithStats(strict, 8)
    out = []
    for i in range(1, n):
        cx, cy = (int(round(v)) for v in centroids[i])
        if not (0 <= cy < H and 0 <= cx < W):
            continue
        if claimed[cy, cx] or not loose[cy, cx]:
            continue
        unit = unit_at(units, cy, H)
        if not unit:
            continue
        radius = math.sqrt(unit / math.pi)
        if stats[i, cv2.CC_STAT_AREA] > unit * 1.2:
            continue              # big enough to be a ball in its own right
        # A highlight sits off-centre, toward the light, so a box centred on it
        # is a box half off the ball. Re-centre on the yellow around it.
        win = int(radius * 1.5)
        wx0, wy0 = max(cx - win, 0), max(cy - win, 0)
        wx1, wy1 = min(cx + win, W), min(cy + win, H)
        ys, xs = np.nonzero(loose[wy0:wy1, wx0:wx1])
        if xs.size:
            cx, cy = int(xs.mean()) + wx0, int(ys.mean()) + wy0

        x0, y0 = max(int(cx - radius), 0), max(int(cy - radius), 0)
        x1, y1 = min(int(cx + radius), W), min(int(cy + radius), H)
        if x1 - x0 < 5 or y1 - y0 < 4:
            continue
        patch = loose[y0:y1, x0:x1]
        if patch.size == 0 or (patch > 0).mean() < min_cover:
            continue              # a highlight on something that is not a ball
        sat, _ = box_stats(hsv, (x0, y0, x1 - x0, y1 - y0))
        if ball_sat and sat < ball_sat * sat_ratio:
            continue              # the wall, the rail, a washed-out reflection
        claimed[y0:y1, x0:x1] = 1
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def detect(img: np.ndarray, min_area: int, max_area: int, min_fill: float,
           keep_clumps: bool, split: bool = True, max_split: int = 8,
           hsv_lo: tuple = HSV_LO, hsv_hi: tuple = HSV_HI,
           merge_factor: float = 1.6, hough: bool = True,
           rescue: bool = True, reflections: bool = True,
           loose_lo: tuple = LOOSE_LO, loose_hi: tuple = LOOSE_HI,
           min_cover: float = 0.6, close_k: int = 5, sat_ratio: float = 0.7,
           v_ratio: float = 0.82, reach: float = 1.6) -> tuple:
    """-> (gated, split out of clusters, rescued from shade, heaps, dropped).

    Four buckets rather than one list because when a proposal is wrong, the
    first question is which pass made it -- and a preview that colours them
    alike cannot answer that.

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
                [(x, y, w, h) for _, x, y, w, h, _ in clumps], [], 0, [])

    if not split or not singles:
        return ([(x, y, w, h) for x, y, w, h, _, _ in singles], [], [],
                len(clumps), [])

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

    rescued = []
    if rescue:
        # Everything so far is built on the strict mask. This is the pass that
        # looks where the strict mask is blind -- shade, and behind bars. What
        # a ball looks like on THIS frame is measured from the ones already
        # found rather than assumed from a constant.
        found = gated + recovered
        ball_sat = float(np.median([box_stats(hsv, b)[0] for b in found])) if found else 0.0
        rescued = rescue_pass(img, hsv, mask, found, units, loose_lo, loose_hi,
                              min_cover, close_k, ball_sat, sat_ratio)

    dropped = []
    if reflections:
        gated, drop_a = drop_reflections(hsv, gated, v_ratio, reach)
        recovered, drop_b = drop_reflections(hsv, recovered, v_ratio, reach)
        rescued, drop_c = drop_reflections(hsv, rescued, v_ratio, reach)
        dropped = drop_a + drop_b + drop_c
    return gated, recovered, rescued, heaps, dropped


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
    ap.add_argument("--no-rescue", dest="rescue", action="store_false",
                    help="don't look for balls the colour gate cannot see -- "
                         "the ones in shade inside a robot's hopper")
    ap.add_argument("--no-reflections", dest="reflections", action="store_false",
                    help="keep proposals that look like a ball's reflection in "
                         "the floor instead of dropping them")
    ap.add_argument("--loose-lo", type=hsv_arg, default=LOOSE_LO,
                    help=f"lower gate for the rescue pass (default "
                         f"{','.join(map(str, LOOSE_LO))}). Too loose to "
                         f"threshold on alone -- nothing is proposed from it "
                         f"without a highlight to seed it")
    ap.add_argument("--loose-hi", type=hsv_arg, default=LOOSE_HI,
                    help=f"upper gate for the rescue pass (default "
                         f"{','.join(map(str, LOOSE_HI))})")
    ap.add_argument("--close", dest="close_k", type=int, default=5,
                    help="kernel that bridges an occluder in the rescue pass. "
                         "A hopper bar is thinner than a ball; this is what "
                         "stops it reading as two half balls")
    ap.add_argument("--sat-ratio", type=float, default=0.7,
                    help="how saturated a rescued ball must be, as a fraction "
                         "of the balls already found on that frame. Shade "
                         "changes brightness and leaves saturation alone, so "
                         "this separates a ball in a hopper from the wall")
    ap.add_argument("--min-cover", type=float, default=0.6,
                    help="fraction of a rescued circle that must be yellow. "
                         "The guard against round things in the crowd")
    ap.add_argument("--v-ratio", type=float, default=0.82,
                    help="a proposal dimmer than this fraction of the one "
                         "above it is its reflection. Raise to drop more")
    ap.add_argument("--reach", type=float, default=1.6,
                    help="how far below a ball, in ball-heights, its "
                         "reflection can sit")
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
    ap.add_argument("--show-dropped", action="store_true",
                    help="draw what the reflection filter removed, in green")
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
                      args.hsv_lo, args.hsv_hi, args.merge_factor, args.hough,
                      args.rescue, args.reflections, args.loose_lo,
                      args.loose_hi, args.min_cover, args.close_k, args.sat_ratio,
                      args.v_ratio, args.reach)

    if args.preview:
        src = images[len(images) // 2]
        if args.preview_frame:
            matches = [p for p in images if args.preview_frame in p.name]
            if not matches:
                print(f"no frame matching {args.preview_frame!r}")
                return 1
            src = matches[0]
        img = cv2.imread(str(src))
        gated, recovered, rescued, heaps, dropped = run(img)
        for x, y, w, h in gated:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        for x, y, w, h in recovered:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), 1)
        for x, y, w, h in rescued:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 1)
        if args.show_dropped:
            # Green is what was thrown away. Look at this before believing the
            # count: a reflection filter that is eating real balls looks
            # exactly like one that is working, from the count alone.
            for x, y, w, h in dropped:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, f"red {len(gated)} gated + orange {len(recovered)} split "
                         f"+ cyan {len(rescued)} rescued = "
                         f"{len(gated) + len(recovered) + len(rescued)}   "
                         f"(green {len(dropped)} reflections, {heaps} heaps skipped)",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), img)
        print(f"{src.name}\n"
              f"  red    {len(gated):4}  through the colour gate\n"
              f"  orange {len(recovered):4}  split out of clusters\n"
              f"  cyan   {len(rescued):4}  rescued from shade (hoppers)\n"
              f"  green  {len(dropped):4}  dropped as reflections\n"
              f"         {heaps:4}  heaps left unlabelled\n"
              f"-> {args.preview}")
        return 0

    if args.limit:
        images = images[:args.limit]

    written = skipped = total = from_splits = heaps_total = reflections_total = 0
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
        gated, recovered, rescued, heaps, dropped = run(img)
        boxes = gated + recovered + rescued
        reflections_total += len(dropped)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(to_yolo(boxes, W, H))
        written += 1
        total += len(boxes)
        from_splits += len(recovered) + len(rescued)
        heaps_total += heaps

    print(f"wrote {written} label files ({skipped} skipped as already present)")
    if written:
        print(f"{total} fuel proposals, {total/written:.0f} per frame "
              f"({from_splits} of them recovered from clusters and shade, "
              f"{reflections_total} reflections dropped, "
              f"{heaps_total/written:.1f} heaps skipped per frame)")
    print("\nThese are proposals for class 0 (fuel) only. robot_blue/robot_red/"
          "hub_blue/hub_red come from train/autolabel_objects.py, which appends "
          "to these files rather than replacing them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
