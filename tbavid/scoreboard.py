"""Read the broadcast scoreboard as a label source.

The score banner gets cropped out of the training images -- you don't want a
detector answering "did it score?" by reading the scoreboard -- but the banner
is the only free ground truth for when fuel actually went in, so it gets read
before it is discarded.

Each alliance has a cumulative fuel counter rendered as "N / M". N is what
matters: it steps up every time fuel scores, so the difference between two
reads is the number of balls scored in that interval. (The big centre numbers
are points, not balls -- they include the autonomous bonus, which is why they
sit a constant offset above the counter.) M is a target threshold and does
step mid-match, so nothing here may assume it is constant.

Locating the counters is automatic: inside a band that is otherwise
pixel-static, the only things that vary over a match are the digits. Connected
components of the variance map give candidate boxes, OCR over a handful of
timestamps says which ones read as monotonically non-decreasing integers, and
the box's own background colour says which alliance it belongs to.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ffm import FFMPEG, grab_frame

PROBE_W = 960
# Probe rows are derived from the search height so that one row is always
# ~2 source pixels. A fixed row count silently changes vertical resolution
# with the size of the search band: widening the search from the banner to the
# top 30% shrank a counter box from 28 px to 25, clipping the digits and
# turning a clean 726 into 1722.
PROBE_PX_PER_ROW = 2
PROBE_H_MIN, PROBE_H_MAX = 64, 320


def probe_rows(search_px: int) -> int:
    return max(PROBE_H_MIN, min(PROBE_H_MAX, int(search_px / PROBE_PX_PER_ROW)))
VAR_FACTOR = 0.05          # of peak variance, to call a pixel "changing"
SMALL_STEP = 8             # increments this size are taken on a single read
BOOTSTRAP_N = 5            # early reads used to seed the series
DILATE_X = 7               # probe px, joins the digits of one number
MIN_COMPONENT_PX = 30

# Candidate boxes, in source pixels.
MIN_BOX_W, MIN_BOX_H = 40, 14
MAX_BOX_H_FRAC = 0.45      # of the banner height
BOX_PAD = 4

# Searched, not anchored: a box can clip a neighbouring graphic that OCRs as
# a stray leading slash ("/129/360"), and an anchored pattern would throw away
# an otherwise perfect read.
PAIR_RE = re.compile(r"(\d{1,4})\s*/\s*(\d{1,5})")
INT_RE = re.compile(r"^\D*(\d{1,5})\D*$")

# The preprocessing that read every sampled frame correctly in testing;
# plain greyscale upscaled 4x, no thresholding (contrast stretching turned
# 129 into 1129 on one frame).
OCR_FILTER = "format=gray,scale=iw*4:ih*4:flags=lanczos"
OCR_ARGS = ["--psm", "7", "-c", "tessedit_char_whitelist=0123456789/"]


def available() -> bool:
    return shutil.which("tesseract") is not None


# -- locating candidate boxes ---------------------------------------------
def _banner_frames(path: Path, width: int, banner_h: int, times: List[float],
                   rows: int) -> List[np.ndarray]:
    frames = []
    need = PROBE_W * rows
    for t in times:
        out = subprocess.run(
            [FFMPEG, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
             "-frames:v", "1",
             "-vf", f"crop={width}:{banner_h}:0:0,scale={PROBE_W}:{rows}",
             "-pix_fmt", "gray", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        if len(out) >= need:
            frames.append(np.frombuffer(out[:need], dtype=np.uint8)
                          .reshape(rows, PROBE_W).astype(np.float32) / 255.0)
    return frames


def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected component labelling (no scipy in this environment)."""
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    n = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or lab[y, x]:
                continue
            n += 1
            lab[y, x] = n
            q = deque([(y, x)])
            while q:
                cy, cx = q.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = n
                            q.append((ny, nx))
    return lab, n


def tighten(comp_mask: np.ndarray, raw_mask: np.ndarray):
    """Extent of a component measured on the undilated mask.

    Split out so the icon-inflation regression is testable without ffmpeg.
    """
    tight = comp_mask & (raw_mask > 0)
    if not tight.any():
        return np.where(comp_mask)
    return np.where(tight)


def find_candidates(path: Path, width: int, banner_h: int,
                    times: List[float]) -> List[Tuple[int, int, int, int]]:
    """Boxes inside the banner whose pixels change over the match."""
    rows = probe_rows(banner_h)
    frames = _banner_frames(path, width, banner_h, times, rows)
    if len(frames) < 6:
        return []
    var = np.stack(frames).var(axis=0)
    mask = var > var.max() * VAR_FACTOR

    # Dilate horizontally only: joins digits within one number without welding
    # together numbers that sit on different rows of the banner.
    dil = mask.copy()
    for s in range(1, DILATE_X + 1):
        dil[:, s:] |= mask[:, :-s]
        dil[:, :-s] |= mask[:, s:]

    lab, n = _label(dil)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < MIN_COMPONENT_PX:
            continue
        # Re-measure the extent on the UNDILATED mask. Dilation is what joins
        # the digits of one number, but it also reaches ~14 source px outward
        # and swallows whatever sits beside them -- the alliance icon left of
        # the counter got pulled in and OCR'd as a leading "1", turning 129
        # into 1129 and inflating the whole match.
        comp = np.zeros(mask.shape, dtype=bool)
        comp[ys, xs] = True
        ys, xs = tighten(comp, mask)
        # Pad by at least one probe row's worth of source pixels, so a box is
        # never trimmed by the quantisation of the probe grid itself.
        pad_y = BOX_PAD + int(np.ceil(banner_h / rows))
        x0 = int(xs.min() / PROBE_W * width) - BOX_PAD
        x1 = int(np.ceil(xs.max() / PROBE_W * width)) + BOX_PAD
        y0 = int(ys.min() / rows * banner_h) - pad_y
        y1 = int(np.ceil(ys.max() / rows * banner_h)) + pad_y
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, banner_h)
        w, h = x1 - x0, y1 - y0
        if w < MIN_BOX_W or h < MIN_BOX_H or h > banner_h * MAX_BOX_H_FRAC:
            continue
        boxes.append((x0, y0, w, h))
    return boxes


# -- OCR -------------------------------------------------------------------
def _crop_filter(box: Tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    return f"crop={w}:{h}:{x}:{y},{OCR_FILTER}"


def ocr_batch(images: List[Path]) -> List[str]:
    """OCR many files in one tesseract invocation.

    Spawning tesseract per image dominates the runtime at a few hundred reads
    per match; batch mode reads a list file and separates pages with form feeds.
    """
    if not images:
        return []
    listing = images[0].parent / "_ocr_list.txt"
    listing.write_text("\n".join(str(p) for p in images))
    proc = subprocess.run(["tesseract", str(listing), "stdout"] + OCR_ARGS,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    pages = proc.stdout.split("\f")
    out = [p.strip().replace(" ", "") for p in pages]
    listing.unlink(missing_ok=True)
    # tesseract emits a trailing empty page
    return (out + [""] * len(images))[:len(images)]


def parse_pair(text: str) -> Optional[Tuple[int, Optional[int]]]:
    """"173/360" -> (173, 360); "173" -> (173, None); anything else -> None."""
    t = text.strip().replace(" ", "")
    m = PAIR_RE.search(t)
    if m:
        return int(m.group(1)), int(m.group(2))
    if "/" in t:
        # A slash with only one number beside it is ambiguous: OCR that drops a
        # small numerator leaves "/100", and reading that 100 as the count
        # poisoned a whole match -- it locked the monotonic filter above every
        # real value that followed. Discard rather than guess which side it is.
        return None
    m = INT_RE.match(t)
    if m:
        return int(m.group(1)), None
    return None


def parse_value(text: str) -> Optional[int]:
    pair = parse_pair(text)
    return pair[0] if pair else None


def plausible_step(prev: int, value: int) -> bool:
    """Reject OCR blowups like 712 -> 7317 while allowing real bulk scoring.

    Only meaningful on the densely sampled series, where consecutive reads are
    a fraction of a second apart. A volley can add a couple of dozen balls at
    once, so the bound is generous; what it catches is a spurious extra digit,
    which multiplies the value by roughly ten.
    """
    return value <= prev * 3 + 100


def _multi_crop(seek_args: List[str], path: Path, boxes: List,
                out_specs: List[List[str]], pre_vf: str = "") -> None:
    """Cut every box out of one decode pass.

    Seeking into the source once per (box, timestamp) is what makes naive
    scoreboard extraction unusable -- 48 candidates over 8 samples is 384
    seeks into a large AV1 file. `split` gives every box its own branch off a
    single decode instead.
    """
    n = len(boxes)
    branches = "".join(f"[s{i}]" for i in range(n))
    head = f"{pre_vf}," if pre_vf else ""
    parts = [f"[0:v]{head}split={n}{branches}"] if n > 1 else [f"[0:v]{head}null[s0]"]
    for i, box in enumerate(boxes):
        parts.append(f"[s{i}]{_crop_filter(box)}[o{i}]")
    cmd = [FFMPEG, "-v", "error", "-y"] + seek_args + ["-i", str(path),
           "-filter_complex", ";".join(parts)]
    for i, spec in enumerate(out_specs):
        cmd += ["-map", f"[o{i}]"] + spec
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _extract_candidate_grid(path: Path, boxes: List, times: List[float],
                            out_dir: Path) -> Dict[int, List[Path]]:
    """{box index: [one crop per timestamp]}, one decode per timestamp."""
    out_dir.mkdir(parents=True, exist_ok=True)
    grid: Dict[int, List[Path]] = {i: [] for i in range(len(boxes))}
    for ti, t in enumerate(times):
        paths = [out_dir / f"cand{bi:03d}_{ti:03d}.png" for bi in range(len(boxes))]
        _multi_crop(["-ss", f"{t:.3f}"], path, boxes,
                    [["-frames:v", "1", str(p)] for p in paths])
        for bi, p in enumerate(paths):
            if p.exists():
                grid[bi].append(p)
    return grid


def _extract_series_crops(path: Path, boxes: Dict[str, tuple], start: float,
                          end: float, fps: float, out_dir: Path
                          ) -> Dict[str, List[Path]]:
    """One ffmpeg pass covering every counter for the whole timeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = list(boxes)
    for tag in tags:
        for stale in out_dir.glob(f"{tag}_*.png"):
            stale.unlink()
    _multi_crop(["-ss", f"{start:.3f}", "-to", f"{end:.3f}"], path,
                [boxes[t] for t in tags],
                [[str(out_dir / f"{t}_%05d.png")] for t in tags],
                pre_vf=f"fps={fps}")
    return {t: sorted(out_dir.glob(f"{t}_*.png")) for t in tags}


# -- identification --------------------------------------------------------
def _alliance_of(path: Path, box, t: float) -> Optional[str]:
    """Blue or red, from the box's own background colour."""
    x, y, w, h = box
    arr = grab_frame(path, t, 8, 4, pre_filter=f"crop={w}:{h}:{x}:{y}")
    if arr is None:
        return None
    r, g, b = arr.reshape(-1, 3).mean(axis=0)
    if b > r * 1.25 and b > g:
        return "blue"
    if r > b * 1.25 and r > g:
        return "red"
    return None


def identify_counters(path: Path, candidates: List, times: List[float],
                      work: Path) -> Dict[str, Tuple[int, int, int, int]]:
    """Pick the per-alliance fuel counters out of the candidate boxes.

    A counter reads as an integer (optionally "N / M"), never goes backwards,
    and actually moves over the match. That trio is enough to reject the match
    clock, the period indicator and every stray graphic.
    """
    found: Dict[str, Tuple] = {}
    best: Dict[str, int] = {}
    grid = _extract_candidate_grid(path, candidates, times, work)

    for idx, box in enumerate(candidates):
        crops = grid.get(idx) or []
        if len(crops) < len(times) - 1:
            continue
        pairs = [parse_pair(t) for t in ocr_batch(crops)]
        for c in crops:
            c.unlink(missing_ok=True)
        good = [p for p in pairs if p is not None]
        if len(good) < len(times) - 1:
            continue
        # The fuel counter renders as "N / M"; the big centre number is points,
        # a bare integer, and is also monotonic -- the slash is what separates
        # them. The "period 4 / 6" indicator also has one, but sits on a
        # neutral background and so fails the alliance-colour test below.
        if sum(1 for _, den in good if den is not None) < len(good) - 1:
            continue
        values = [n for n, _ in good]
        # Tolerate a single backwards step: these samples are ~20s apart, so
        # one bad read shouldn't disqualify an otherwise clean counter. No
        # plausibility bound here -- over a 20s gap a jump from 0 to 129 is a
        # real volley, not a misread. That guard belongs on the dense series.
        if sum(1 for a, b in zip(values, values[1:]) if b < a) > 1:
            continue
        span = max(values) - min(values)
        if span <= 0:
            continue                      # never moved: static graphic
        alliance = _alliance_of(path, box, times[len(times) // 2])
        if alliance is None:
            continue
        if span > best.get(alliance, 0):
            best[alliance] = span
            found[alliance] = box
    return found


# -- series ----------------------------------------------------------------
def clean_series(values: List[Optional[int]], times: List[float]
                 ) -> List[Tuple[float, int]]:
    """Monotonic step function from noisy OCR reads.

    Values never go down, because fuel counts don't. Large jumps need two
    consecutive reads to agree, which discards single-frame misreads (a stray
    digit turning 129 into 1129). Small increments are taken on a single read:
    demanding confirmation for those loses the counter entirely whenever it
    moves faster than the sample rate, which is exactly when an alliance is
    scoring hardest.
    """
    parsed = [(t, v) for t, v in zip(times, values) if v is not None]
    if not parsed:
        return []

    # Seed from the minimum of the first few reads. The series is monotonic, so
    # its earliest true value is the smallest one around; taking the literal
    # first read instead would let an OCR blowup anchor the whole match high
    # and reject every genuine value after it as "backwards".
    head = parsed[:BOOTSTRAP_N]
    current = min(v for _, v in head)
    seed_t = next(t for t, v in head if v == current)
    out: List[Tuple[float, int]] = [(seed_t, current)]
    pending: Optional[int] = None

    for t, v in parsed:
        if t <= seed_t:
            continue
        if v == current:
            pending = None
            continue
        if v < current or not plausible_step(current, v):
            pending = None            # counters never decrease, nor decuple
            continue
        if v - current <= SMALL_STEP:
            current = v
            out.append((t, v))
            pending = None
            continue
        if pending == v:
            current = v
            out.append((t, v))
            pending = None
        else:
            pending = v
    return out


def read_counters(path: Path, boxes: Dict[str, tuple], start: float, end: float,
                  fps: float, work: Path) -> Dict[str, List[Tuple[float, int]]]:
    sets = _extract_series_crops(path, boxes, start, end, fps, work)
    out = {}
    for tag, crops in sets.items():
        if not crops:
            out[tag] = []
            continue
        times = [start + i / fps for i in range(len(crops))]
        values = [parse_value(v) for v in ocr_batch(crops)]
        for c in crops:
            c.unlink(missing_ok=True)
        out[tag] = clean_series(values, times)
    return out


def to_events(series: Dict[str, List[Tuple[float, int]]]) -> List[Dict]:
    """Flatten per-alliance step functions into scoring events."""
    events = []
    for alliance, points in series.items():
        prev = None
        for t, v in points:
            if prev is not None and v > prev:
                events.append({"t": round(t, 3), "alliance": alliance,
                               "balls": v - prev, "total": v})
            prev = v
    events.sort(key=lambda e: e["t"])
    return events


# Searched independently of the crop, because the crop depends on the answer.
SEARCH_FRAC = 0.30
BANNER_FLOOR_MARGIN = 12   # source px below the lowest counter box


def locate(path: Path, analysis: dict, cfg: dict, work: Path) -> Dict:
    """Find the per-alliance fuel counters.

    Runs BEFORE the crop is fixed and over a generous top 30% of the frame.
    Deriving the search band from the crop instead is circular, and got it
    wrong in practice: a district banner whose counters sit on the same row as
    a changing clock had the crop land above them, so the reader saw half-
    digits and found nothing, while the frames kept a strip of scoreboard.
    """
    if not available():
        return {"error": "tesseract not installed "
                         "(apt install tesseract-ocr / brew install tesseract)"}
    keep = [tuple(r) for r in analysis["keep_ranges"]]
    if not keep:
        return {"error": "no kept footage"}

    from .crop import sample_times
    search_px = int(analysis["height"] * SEARCH_FRAC)
    candidates = find_candidates(path, analysis["width"], search_px,
                                 sample_times(keep, 40))
    if not candidates:
        return {"error": "no changing regions found in the banner"}

    work.mkdir(parents=True, exist_ok=True)
    counters = identify_counters(path, candidates, sample_times(keep, 8), work)
    if not counters:
        return {"error": f"none of {len(candidates)} candidate regions "
                         f"behaved like a fuel counter",
                "candidates": len(candidates)}
    return {"counters": {a: list(map(int, b)) for a, b in counters.items()},
            "candidates": len(candidates)}


def banner_floor_px(located: Dict) -> int:
    """Lowest pixel any identified counter occupies, plus a margin."""
    boxes = (located or {}).get("counters") or {}
    if not boxes:
        return 0
    return max(y + h for _, y, _, h in boxes.values()) + BANNER_FLOOR_MARGIN


def read(path: Path, analysis: dict, located: Dict, cfg: dict, work: Path) -> Dict:
    """Read the located counters across the match."""
    if located.get("error"):
        return dict(located)
    counters = {a: tuple(b) for a, b in located["counters"].items()}
    keep = [tuple(r) for r in analysis["keep_ranges"]]
    start, end = keep[0][0], keep[-1][1]
    # Keep reading past the end of the main-camera shot. Fuel in flight at the
    # buzzer keeps landing after the broadcast cuts away, and the banner stays
    # on screen while it does -- stopping at the cut undercounted a fast blue
    # alliance by 29 of 755 against TBA's official breakdown. Frames where the
    # banner has gone simply fail to parse and are ignored.
    end = min(end + float(cfg.get("score_tail_s", 20)), analysis["duration"])
    fps = float(cfg.get("score_sample_fps", 5))
    series = read_counters(path, counters, start, end, fps, work)
    return {
        "counters": {a: list(b) for a, b in counters.items()},
        "candidates": located.get("candidates"),
        "sample_fps": fps,
        "series": {a: [[round(t, 3), v] for t, v in s] for a, s in series.items()},
        "final": {a: (s[-1][1] if s else None) for a, s in series.items()},
        "events": to_events(series),
    }


def extract(path: Path, analysis: dict, crop_box: dict, cfg: dict,
            work: Path) -> Dict:
    """locate + read, for re-running the scoreboard on an existing video."""
    return read(path, analysis, locate(path, analysis, cfg, work), cfg, work)
