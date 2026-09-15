"""Join exported frames to the scoreboard timeline.

Frames come out of the cleaned video, whose clock is the kept ranges
concatenated together; the scoreboard was read off the original broadcast.
Everything here exists to map between those two timelines and emit one CSV row
per frame.
"""
from __future__ import annotations

import bisect
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def source_time(t_clean: float, keep_ranges: Sequence[Sequence[float]]) -> float:
    """Cleaned-video timestamp -> timestamp in the original broadcast."""
    acc = 0.0
    for a, b in keep_ranges:
        span = b - a
        if t_clean < acc + span:
            return a + (t_clean - acc)
        acc += span
    return keep_ranges[-1][1] if keep_ranges else t_clean


class Timeline:
    """Cumulative fuel counts over source time, with windowed lookups."""

    def __init__(self, series: Dict[str, List[Sequence[float]]]):
        self.times: Dict[str, List[float]] = {}
        self.values: Dict[str, List[int]] = {}
        for alliance, points in (series or {}).items():
            self.times[alliance] = [float(t) for t, _ in points]
            self.values[alliance] = [int(v) for _, v in points]

    @property
    def alliances(self) -> List[str]:
        return sorted(self.times)

    def total_at(self, alliance: str, t: float) -> Optional[int]:
        times = self.times.get(alliance)
        if not times:
            return None
        i = bisect.bisect_right(times, t) - 1
        return self.values[alliance][i] if i >= 0 else 0

    def scored_between(self, alliance: str, t0: float, t1: float) -> Optional[int]:
        lo, hi = self.total_at(alliance, t0), self.total_at(alliance, t1)
        if lo is None or hi is None:
            return None
        return max(hi - lo, 0)


def write_frame_labels(dest: Path, frames: List[Dict], keep_ranges,
                       series: Dict, cfg: dict) -> int:
    """One CSV row per exported frame.

    The counter updates a beat after the fuel actually lands, so the useful
    column is the forward window: "how much did this alliance score in the
    next `score_window_s` seconds". `score_latency_s` shifts that window if
    you calibrate the lag for a given broadcast.
    """
    timeline = Timeline(series)
    alliances = timeline.alliances or ["blue", "red"]
    window = float(cfg.get("score_window_s", 1.5))
    latency = float(cfg.get("score_latency_s", 0.0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["frame", "t_clean", "t_source"]
    for a in alliances:
        cols += [f"{a}_total", f"{a}_scored_next", f"{a}_scored_prev"]

    with dest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for frame in frames:
            tc = frame["t_clean"]
            ts = source_time(tc, keep_ranges)
            row = [frame["file"], f"{tc:.3f}", f"{ts:.3f}"]
            for a in alliances:
                base = ts + latency
                total = timeline.total_at(a, base)
                nxt = timeline.scored_between(a, base, base + window)
                prev = timeline.scored_between(a, base - window, base)
                row += ["" if total is None else total,
                        "" if nxt is None else nxt,
                        "" if prev is None else prev]
            writer.writerow(row)
    return len(frames)
