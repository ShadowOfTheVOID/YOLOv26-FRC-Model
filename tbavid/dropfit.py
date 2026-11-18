"""The one-ball size from a drop test, and Measure's estimate. No OpenCV.

Why the ball size comes from a drop test and not from Measure. The colour
counter's count depends steeply on the one-ball pixel area: on Watchtower's
2026-10-10 recording, half the right area counted 302 balls against 112 and
double counted 46, and measuring lone balls mid-match gave 150-230 px
against the right 693 -- that event's ~9x overcount was a wrong area. So
someone drops a known number of balls into each hub while the blobs near
every outline are recorded, and the counter is replayed over those blobs at
log-spaced areas; the area kept is the one whose count is nearest what was
dropped. Measure (still balls, median lone-ball area) only draws the ruler
while an outline is drawn and is held up against the fit.

Ported from Watchtower's vision/dropfit.py (arnan-bajaj/watchtower-fms, MIT),
which fits one hub per camera. Here a camera can hold several outlines and
exit lines, on one hub or both, and has one ball area: every zone is
replayed at the same area, each hub combined as the setup combines it
(sum / max / median), and the error is summed over the hubs that were
counted by hand.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from . import hubcount as HC

AREA_LO, AREA_HI = 30.0, 30000.0
COARSE, FINE = 25, 17        # 33% apart over the whole range, then 4% apart around the best
RECORD_BOX_AREA = 4000.0     # record a generous box: the area is not known yet (as measure())
MAX_FRAMES = 60 * 60 * 3     # 3 minutes at 60 fps
MEASURE_MIN_BLOBS = 8        # fewer lone round blobs than this is no measurement


def record_box(poly, w: int, h: int):
    return HC.region_of(poly, RECORD_BOX_AREA, w, h)


def _counter(z: Dict, area: float, blur: float):
    """A zone dict as the page saves it -> its counter at this area."""
    if z.get("line"):
        return HC.ExitLineCounter(z["line"], z["out"], area, blur)
    return HC.CrossingCounter([tuple(p) for p in z["outline"]], area, blur)


def zone_poly(z: Dict) -> List:
    """What a zone's search box is built from (as the counter's own poly)."""
    return _counter(z, 1.0, 0.0).poly


def replay(zones: Sequence[Dict], frames: Sequence[Dict[str, list]], area: float,
           w: int, h: int, blur: float = HC.DEFAULT_BLUR,
           combine: Dict[str, str] = None) -> Dict[str, int]:
    """Each hub's count over these frames at this ball area, as the live
    counter would report it. A zone's blobs are kept only inside the box the
    live counter would search at that area."""
    per_hub: Dict[str, List[int]] = {}
    for z in zones:
        c = _counter(z, area, blur)
        x0, y0, x1, y1 = HC.region_of(c.poly, area, w, h)
        for f in frames:
            c.update([b for b in f.get(z["name"], ()) if x0 <= b[0] < x1 and y0 <= b[1] < y1])
        per_hub.setdefault(z["hub"], []).append(c.reported)
    combine = combine or {}
    return {hub: HC.combined(n, combine.get(hub, "sum")) for hub, n in per_hub.items()}


def _logspace(lo: float, hi: float, steps: int) -> List[float]:
    k = (hi / lo) ** (1 / (steps - 1))
    return [lo * k ** i for i in range(steps)]


def fit(zones: Sequence[Dict], frames: Sequence[Dict[str, list]], hand: Dict[str, int],
        w: int, h: int, blur: float = HC.DEFAULT_BLUR, combine: Dict[str, str] = None) -> Dict:
    """{"area", "got", "hand", "lo", "hi"}: a ball area whose replayed counts are
    nearest the hand counts (got: per hub), from the range of areas (lo-hi)
    that count the same. ValueError if nothing crossed any outline.

    Inside that range the area kept is a quarter of the way up (log space).
    A one-ball blob of true area A counts once from about 0.6A (below that it
    counts 2) to 3.4A (smaller than the noise floor); a two-ball clump counts
    2 only up to about 1.2A. Single drops alone give about 0.6A-3.4A, quarter
    point 0.9A; drops with clumps give about 0.6A-1.2A, quarter point 0.7A.
    Both land where clumps and singles both count right; the middle (1.4A)
    would not. (Watchtower's reasoning, unchanged.)"""
    hand = {k: int(v) for k, v in hand.items() if v is not None and str(v) != ""}
    hubs = [k for k in hand if any(z["hub"] == k for z in zones)]
    if not hubs or sum(hand[k] for k in hubs) < 1:
        raise ValueError("type how many balls went into each hub (at least one)")

    def err(counts: Dict[str, int]) -> int:
        return sum(abs(counts.get(k, 0) - hand[k]) for k in hubs)

    def search(areas):
        return [(a, replay(zones, frames, a, w, h, blur, combine)) for a in areas]

    coarse = search(_logspace(AREA_LO, AREA_HI, COARSE))
    if not any(sum(c.values()) for _, c in coarse):
        raise ValueError("Nothing crossed the outline. Check that the outline is around the "
                         "hub's opening and the balls fell through it in the picture.")
    best = min(err(c) for _, c in coarse)
    idx = [i for i, (_, c) in enumerate(coarse) if err(c) == best]
    lo = coarse[max(0, idx[0] - 1)][0]
    hi = coarse[min(COARSE - 1, idx[-1] + 1)][0]
    pts = sorted(search(_logspace(lo, hi, FINE)) + coarse, key=lambda p: p[0])
    errs = [err(c) for _, c in pts]
    best = min(errs)
    runs, cur = [], []                       # runs of neighbouring areas at the best error
    for i, e in enumerate(errs):
        if e == best:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    run = max(runs, key=len)
    lo, hi = pts[run[0]][0], pts[run[-1]][0]
    area = lo * (hi / lo) ** 0.25
    got = replay(zones, frames, area, w, h, blur, combine)
    return {"area": round(area, 1), "got": {k: got.get(k, 0) for k in hubs},
            "hand": {k: hand[k] for k in hubs}, "lo": round(lo, 1), "hi": round(hi, 1)}


def estimate(areas: Sequence[float], frames: int) -> Dict:
    """Measure: one ball's area from the lone round blobs seen
    (hubcount.isolated_ball_areas, per frame). Only an estimate: it draws
    the ruler and is checked against the drop test; never the counted size."""
    if len(areas) < MEASURE_MIN_BLOBS:
        raise ValueError(f"Only {len(areas)} lone balls seen in {frames} frames. Put a few balls "
                         f"in view, apart from each other and not moving, and measure again.")
    s = sorted(areas)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    return {"area": round(med, 1), "blobs": len(s), "frames": frames}


def agrees(measured: float, fitted: float) -> bool:
    """Is Measure's estimate within a factor of 2 of the drop test's fit?
    Outside it, one of them saw something else (shirts, a pile), and the
    page says so rather than pick one."""
    return bool(measured and fitted) and 0.5 <= measured / fitted <= 2.0
