"""Work out what to crop off each broadcast.

Two things get removed, both found the same way: burned-in graphics are
pixel-for-pixel static, so a per-row temporal variance map over main-camera
frames gives them away while the field rows churn with moving robots and balls.

1. The score banner across the top.
2. The bottom panel of a split-screen layout. The 2026 Championship broadcast
   shows the elevated field camera on top and a low side camera underneath,
   separated by a static textured divider, for the entire match -- so the
   "side camera" is a region of every frame, not a shot to cut away. The
   divider reads as a band of near-zero variance in the lower half, and
   everything below it goes.

Two thresholds do the work. A strict one (1% of the field's variance) only
fires on real graphics, which is what identifies the divider; a loose one (8%)
then expands outward to the band's true edges. Using the loose threshold alone
would false-positive on the dark, barely-moving bleacher rows inside the main
camera's own panel.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ffm import FFMPEG, even, grab_frame

PROBE_W, PROBE_H = 480, 180
CROPDETECT_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

STRICT_FACTOR = 0.01      # only burned-in graphics are this still
LOOSE_FACTOR = 0.08       # "not really moving"

# A split divider is a hard graphic boundary that never moves, so in the
# temporal median image it shows up as a horizontal edge far stronger than
# anything the field produces. Measured: 116 on the district feed, 74 on the
# Championship feed, against a lower-half typical of well under 20.
EDGE_MIN_ABS = 40.0
EDGE_MIN_RATIO = 3.0

# Overshoot both bands slightly. A dozen extra pixels of empty bleacher costs
# nothing; a leftover strip of scoreboard or divider is what we came to remove.
BANNER_MARGIN_ROWS = 3
SPLIT_MARGIN_ROWS = 3

MIN_SPLIT_START = 0.45    # a split divider lives in the lower half of the frame
MIN_STRICT_RUN = 2        # rows, at PROBE_H resolution
MAX_EXPAND_ROWS = 20      # don't let the loose expansion run away

# Sanity ceilings: past these the detection is wrong, not the broadcast weird.
MAX_BANNER_FRAC = 0.35
MAX_BOTTOM_FRAC = 0.50


def sample_times(keep_ranges: List[Tuple[float, float]], n: int = 40) -> List[float]:
    """Spread n sample points across the kept (main-camera) footage."""
    spans = [(a, b) for a, b in keep_ranges if b > a]
    total = sum(b - a for a, b in spans)
    if total <= 0:
        return []
    times = []
    for i in range(n):
        target = total * (i + 0.5) / n
        acc = 0.0
        for a, b in spans:
            if acc + (b - a) >= target:
                times.append(a + (target - acc))
                break
            acc += b - a
    return times


class Profile:
    """Per-row signals used to find the overlay bands."""

    def __init__(self, var_med: np.ndarray, var_p25: np.ndarray,
                 edge: np.ndarray, field_ref: float):
        # Two column statistics, because the two bands fail differently.
        # Median survives narrow changing digits and, being the stricter of the
        # two, stops the top walk before it wanders into dark static bleachers
        # that are really main-camera content. The 25th percentile is needed
        # lower down, where a caption box covering nearly half the width would
        # otherwise hide the divider from a median.
        self.var_med = var_med
        self.var_p25 = var_p25
        self.edge = edge
        self.field_ref = field_ref


def row_profile(path: Path, times: List[float], pre_filter: str = ""
                ) -> Optional[Profile]:
    frames = []
    for t in times:
        arr = grab_frame(path, t, PROBE_W, PROBE_H, pre_filter=pre_filter)
        if arr is not None:
            frames.append(arr.astype(np.float32).mean(axis=2))
    if len(frames) < 6:
        return None

    stack = np.stack(frames)
    pixvar = stack.var(axis=0) / (255.0 ** 2)
    # 25th percentile across columns, not the median. Score digits and a match
    # clock occupy few columns so a median survives them, but a burned-in
    # caption box can cover nearly half the width and drag the median up until
    # the divider underneath it stops reading as static.
    var_p25 = np.percentile(pixvar, 25, axis=1)
    var_med = np.median(pixvar, axis=1)

    median_img = np.median(stack, axis=0)
    grad = np.abs(np.diff(median_img, axis=0))
    grad = np.vstack([grad, grad[-1:]])
    edge = np.percentile(grad, 60, axis=1)   # 60th pct => must span most of the width

    field_ref = float(np.median(var_med[int(0.40 * PROBE_H):int(0.90 * PROBE_H)]))
    if field_ref <= 0:
        return None
    return Profile(var_med, var_p25, edge, field_ref)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True spans as (start, end_exclusive)."""
    out = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def detect_top(prof: Profile, floor_px: int = 0, height: int = 0) -> Optional[float]:
    """Fraction of height taken by the static top overlay band.

    `floor_px` lets the scoreboard reader veto a too-shallow result: if it
    located a fuel counter, the banner demonstrably extends below that box, and
    a crop above it would both leave scoreboard in the training frame and slice
    the digits the OCR needs.
    """
    loose = prof.var_med < prof.field_ref * LOOSE_FACTOR
    r = 0
    while r < PROBE_H and loose[r]:
        r += 1
    rows = r + BANNER_MARGIN_ROWS if r else 0
    if rows / PROBE_H > MAX_BANNER_FRAC:
        rows = 0                      # runaway walk; fall back to config/floor

    if floor_px and height:
        rows = max(rows, int(np.ceil(floor_px / height * PROBE_H)))

    frac = rows / PROBE_H
    if frac <= 0:
        return None
    return None if frac > MAX_BANNER_FRAC else frac


def detect_bottom(prof: Profile) -> Optional[float]:
    """Fraction of height below a split-screen divider (0.0 if there isn't one).

    Anchored on the topmost strong horizontal edge in the lower half. Staticness
    alone is not enough: a burned-in caption sitting across the divider lifts
    its variance right out of the static range, which is exactly how a district
    feed kept its entire side-camera panel. The edge survives the caption
    because the caption only spans part of the width.
    """
    floor = int(MIN_SPLIT_START * PROBE_H)
    lower = prof.edge[floor:]
    if lower.size == 0:
        return 0.0
    typical = float(np.median(lower))
    thresh = max(EDGE_MIN_ABS, typical * EDGE_MIN_RATIO)

    hits = np.where(lower > thresh)[0]
    if hits.size == 0:
        return _detect_bottom_static(prof)      # no hard edge: fall back

    start = floor + int(hits[0])
    cut = max(start - SPLIT_MARGIN_ROWS, 0)
    frac = (PROBE_H - cut) / PROBE_H
    return None if frac > MAX_BOTTOM_FRAC else frac


def _detect_bottom_static(prof: Profile) -> float:
    """Older staticness route, kept for feeds whose divider has no hard edge."""
    strict = prof.var_p25 < prof.field_ref * STRICT_FACTOR
    loose = prof.var_p25 < prof.field_ref * LOOSE_FACTOR
    floor = int(MIN_SPLIT_START * PROBE_H)
    cands = [(a, b) for a, b in _runs(strict) if a >= floor and b - a >= MIN_STRICT_RUN]
    if not cands:
        return 0.0
    start = cands[0][0]
    limit = max(start - MAX_EXPAND_ROWS, 0)
    while start > limit and loose[start - 1]:
        start -= 1
    cut = max(start - SPLIT_MARGIN_ROWS, 0)
    frac = (PROBE_H - cut) / PROBE_H
    return 0.0 if frac > MAX_BOTTOM_FRAC else frac


def detect_letterbox(path: Path, start: float, width: int, height: int
                     ) -> Optional[Tuple[int, int, int, int]]:
    """ffmpeg cropdetect over a 20s window -> (w, h, x, y) content box."""
    proc = subprocess.run(
        [FFMPEG, "-v", "info", "-ss", f"{max(start, 0):.3f}", "-t", "20",
         "-i", str(path), "-vf", "cropdetect=24:2:0", "-an", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    found = CROPDETECT_RE.findall(proc.stderr or "")
    if not found:
        return None
    w, h, x, y = (int(v) for v in found[-1])
    if w <= 0 or h <= 0 or w > width or h > height:
        return None
    if w * h < 0.5 * width * height:
        return None
    return w, h, x, y


def resolve_crop(path: Path, analysis: dict, cfg: dict,
                 banner_floor_px: int = 0, event_key: str = "") -> Dict:
    """Final crop box for one video, as ffmpeg w:h:x:y plus provenance."""
    width, height = analysis["width"], analysis["height"]
    keep = [(a, b) for a, b in analysis.get("keep_ranges", [])] or \
           [(0.0, analysis["duration"])]
    conf = cfg["crop"]
    notes = []

    x, y, w, h = 0, 0, width, height
    box = detect_letterbox(path, keep[0][0], width, height)
    if box and (box[0] < width or box[1] < height):
        w, h, x, y = box
        notes.append(f"letterbox {width}x{height} -> {w}x{h}+{x}+{y}")

    # Measure the row profile inside the letterbox, so the fractions below
    # refer to the same rectangle we're about to trim.
    pre = f"crop={w}:{h}:{x}:{y}" if (w, h, x, y) != (width, height, 0, 0) else ""
    # Per-event escape hatch: some feeds will always need a human number.
    override = (conf.get("overrides") or {}).get(event_key) or {}

    top_frac = bottom_frac = None
    if conf.get("auto_detect_banner", True) or conf.get("auto_detect_split", True):
        prof = row_profile(path, sample_times(keep, 40), pre_filter=pre)
        if prof:
            if conf.get("auto_detect_banner", True):
                top_frac = detect_top(prof, banner_floor_px, h)
            if conf.get("auto_detect_split", True):
                bottom_frac = detect_bottom(prof)

    if "top" in override:
        top_frac = float(override["top"])
        notes.append(f"crop.overrides[{event_key}] top={top_frac:.3f}")
    if "bottom" in override:
        bottom_frac = float(override["bottom"])
        notes.append(f"crop.overrides[{event_key}] bottom={bottom_frac:.3f}")

    if top_frac is None:
        top_frac = float(conf.get("top", 0.0))
        notes.append(f"banner auto-detect inconclusive, using top={top_frac:.3f}")
    elif "top" not in override:
        extra = f" (floored by scoreboard at {banner_floor_px}px)" if banner_floor_px else ""
        notes.append(f"banner auto-detected at top={top_frac:.3f}{extra}")

    if bottom_frac is None:
        bottom_frac = float(conf.get("bottom", 0.0))
        notes.append(f"split auto-detect inconclusive, using bottom={bottom_frac:.3f}")
    elif bottom_frac > 0:
        notes.append(f"split-screen side camera detected, cropping bottom="
                     f"{bottom_frac:.3f}")
    else:
        notes.append("no split-screen divider found, keeping full height")

    top = int(round(h * top_frac))
    bottom = int(round(h * bottom_frac))
    left = int(round(w * float(conf.get("left", 0.0))))
    right = int(round(w * float(conf.get("right", 0.0))))

    x += left
    y += top
    w = even(w - left - right)
    h = even(h - top - bottom)

    if w < 160 or h < 120:
        raise ValueError(f"crop collapsed to {w}x{h}; check config.json crop values")

    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "top_frac": round(top_frac, 4), "bottom_frac": round(bottom_frac, 4),
            "notes": notes, "filter": f"crop={w}:{h}:{x}:{y}"}


def cropcheck(path: Path, crop: Dict, out_dir: Path, times: List[float]) -> List[Path]:
    """Write proof frames: the full frame with the box drawn, and the crop."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, t in enumerate(times):
        boxed = out_dir / f"cropcheck_{i:02d}_full.jpg"
        cropped = out_dir / f"cropcheck_{i:02d}_cropped.jpg"
        subprocess.run(
            [FFMPEG, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path),
             "-frames:v", "1",
             "-vf", (f"drawbox=x={crop['x']}:y={crop['y']}:w={crop['w']}:h={crop['h']}"
                     f":color=red@1.0:t=5,scale=960:-2"),
             "-q:v", "3", str(boxed)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [FFMPEG, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path),
             "-frames:v", "1", "-vf", f"{crop['filter']},scale=960:-2",
             "-q:v", "3", str(cropped)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        written.extend(p for p in (boxed, cropped) if p.exists())
    return written
