"""Scout a live stream: read the scoreboard as it happens, keep no video.

This is not the harvest path and it is deliberately not built on it. `pull` and
`stream` exist to make a training set: they download a broadcast, crop it,
sample frames and fill a disk, and everything they produce is for a detector
that does not exist yet. None of that is any use to somebody who wants to know
how an alliance is scoring this afternoon.

So this reads the one thing in a broadcast that is already ground truth -- the
burned-in fuel counter -- off a live feed, and throws every frame away the
moment it has been read.

    ./run.py live --url <twitch or youtube live> --event 2026caclv --match qm14

## What it keeps

Nothing. Each pass records a few seconds to a temp file, OCRs the two counter
boxes out of it, and deletes it. Peak disk is one chunk. There are no frames in
`data/frames/`, no cleaned video in `data/videos/`, and nothing enters the
dataset -- the only thing written is the scoring rows, which is what a scouting
app reads.

## What it can tell you, and what it cannot

It gives the **scoring timeline, per alliance**: when fuel went in, to the
second, for the match on the field right now. Nothing else in either of these
repos produces that live, and it is the number that answers "is this alliance
front-loading or finishing strong" rather than "how much did they get".

It cannot tell you **which robot**. The counter says an alliance scored and
never which of its three did -- the same ceiling `SCOUTING.md` describes, and
no amount of live-ness moves it. Per-robot needs a trained detector and a
tracker, and neither exists yet.

## Why the counter state has to survive a chunk boundary

`scoreboard.clean_series` builds a monotonic series from one call's worth of
reads, and it seeds from the minimum of the first few -- correct for a whole
match read in one pass, wrong here. Called once per chunk it would re-seed
every few seconds, so a counter sitting at 120 would be re-anchored at 120 and
the step across the boundary would be lost, or worse, a chunk whose first read
misfires would anchor high and reject the rest of the match.

`Counter` below is that same rule made stateful, and `tests/test_pipeline.py`
pins the two together: fed one chunk, it must agree with `clean_series` read
for read. Two implementations of one rule drift otherwise.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import scoreboard
from .download import YTDLP, cookie_args
from .ffm import FFMPEG, probe
from .scoreboard import BOOTSTRAP_N, SMALL_STEP, parse_value, plausible_step

# Long enough that OCR has several reads of every change, short enough that the
# readout is not minutes behind the field. A chunk is also the unit of loss: if
# ffmpeg drops the connection mid-chunk, that chunk is what has to be re-read.
CHUNK_S = 20.0

# The warm-up recording used to find the counter boxes. Needs to be long enough
# for `find_candidates` to see the digits actually change -- it identifies a
# counter by the fact that it moves, so a still banner tells it nothing.
BOOTSTRAP_S = 90.0

# ffmpeg gets this long to produce a chunk before it is treated as a stall.
# Generous: a live HLS input buffers, and killing a slow read loses the window.
READ_TIMEOUT_MULT = 3.0


class Counter:
    """One alliance's fuel counter, read in pieces.

    The same monotonic rule as `scoreboard.clean_series` -- values never go
    down, a big jump needs two consecutive reads to agree, a small one is taken
    on a single read -- with the state held across calls instead of rebuilt per
    call.
    """

    def __init__(self):
        self.current: Optional[int] = None
        self.pending: Optional[int] = None
        self.points: List[Tuple[float, int]] = []

    def feed(self, values: List[Optional[int]], times: List[float]
             ) -> List[Tuple[float, int]]:
        """Fold one chunk's raw reads in. Returns only the NEW points."""
        parsed = [(t, v) for t, v in zip(times, values) if v is not None]
        fresh: List[Tuple[float, int]] = []
        if not parsed:
            return fresh

        if self.current is None:
            # Seed from the smallest of the first few reads, not the literal
            # first: the series is monotonic, so its earliest true value is the
            # smallest one around, and an OCR blowup taken as the seed would
            # anchor the match high and reject every real value after it.
            head = parsed[:BOOTSTRAP_N]
            self.current = min(v for _, v in head)
            seed_t = next(t for t, v in head if v == self.current)
            self.points.append((seed_t, self.current))
            fresh.append((seed_t, self.current))
            parsed = [(t, v) for t, v in parsed if t > seed_t]

        for t, v in parsed:
            if v == self.current:
                self.pending = None
                continue
            if v < self.current or not plausible_step(self.current, v):
                self.pending = None
                continue
            if v - self.current <= SMALL_STEP:
                self.current = v
                self.points.append((t, v))
                fresh.append((t, v))
                self.pending = None
                continue
            if self.pending == v:
                self.current = v
                self.points.append((t, v))
                fresh.append((t, v))
                self.pending = None
            else:
                self.pending = v
        return fresh


def resolve(url: str, cfg: Optional[dict] = None) -> Optional[str]:
    """A direct media URL ffmpeg can open, out of a page URL.

    `yt-dlp -g` rather than letting ffmpeg at the page: a Twitch or YouTube
    watch URL is HTML, and ffmpeg has no idea what to do with it. For a live
    stream what comes back is an HLS manifest, which ffmpeg reads as it grows.
    """
    proc = subprocess.run(
        [YTDLP, "-g", "--no-playlist", "--no-warnings"]
        + cookie_args(cfg or {}) + [url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return None
    # Video first when yt-dlp splits the streams; audio is of no use here.
    lines = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
    return lines[0] if lines else None


def record(media_url: str, seconds: float, dest: Path) -> Optional[Path]:
    """Capture `seconds` of a live input to a file, for reading and deleting.

    Stream copy, no re-encode: this is read once by the OCR and then unlinked,
    so spending CPU on it would be spending it twice over.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    try:
        subprocess.run(
            [FFMPEG, "-v", "error", "-y",
             # Live inputs stall; without this a dead connection hangs forever.
             "-rw_timeout", str(int(seconds * READ_TIMEOUT_MULT * 1_000_000)),
             "-i", media_url, "-t", f"{seconds:.2f}",
             "-c", "copy", "-an", str(dest)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=seconds * READ_TIMEOUT_MULT + 30)
    except subprocess.TimeoutExpired:
        dest.unlink(missing_ok=True)
        return None
    if not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        return None
    return dest


def raw_reads(clip: Path, boxes: Dict[str, tuple], fps: float, work: Path
              ) -> Dict[str, Tuple[List[Optional[int]], List[float]]]:
    """OCR the counter boxes out of one clip, uncleaned.

    Reaches `scoreboard._extract_series_crops` rather than `read_counters`
    because that one cleans as it goes, and cleaning is what has to be stateful
    here. The crop-and-OCR plumbing underneath is the same tested code either
    way; only the monotonic pass is done by `Counter` instead.
    """
    info = probe(clip) or {}
    dur = float(info.get("duration") or 0.0)
    if dur <= 0:
        return {}
    sets = scoreboard._extract_series_crops(clip, boxes, 0.0, dur, fps, work)
    out = {}
    for tag, crops in sets.items():
        if not crops:
            out[tag] = ([], [])
            continue
        times = [i / fps for i in range(len(crops))]
        values = [parse_value(v) for v in scoreboard.ocr_batch(crops)]
        for c in crops:
            c.unlink(missing_ok=True)
        out[tag] = (values, times)
    return out


class Watcher:
    """A live feed, its counter boxes, and the series read off them so far."""

    def __init__(self, media_url: str, cfg: dict, work: Path,
                 chunk_s: float = CHUNK_S):
        self.media_url = media_url
        self.cfg = cfg
        self.work = work
        self.chunk_s = chunk_s
        self.boxes: Dict[str, tuple] = {}
        self.counters: Dict[str, Counter] = {}
        self.elapsed = 0.0
        self.chunks = 0
        self.misses = 0

    def bootstrap(self, seconds: float = BOOTSTRAP_S) -> Dict:
        """Find the fuel counters on this feed. Needs the digits to move.

        `find_candidates` identifies a counter by the fact that it changes over
        the sample, so this has to run while a match is actually being played.
        Pointed at a field between matches it will correctly find nothing, and
        says so rather than locking onto the match clock.
        """
        clip = record(self.media_url, seconds, self.work / "_bootstrap.mp4")
        if clip is None:
            return {"error": "could not read the stream at all -- is it live?"}
        try:
            info = probe(clip) or {}
            if not info.get("width"):
                return {"error": "no video in what the stream returned"}
            dur = float(info.get("duration") or 0.0)
            analysis = {"width": info["width"], "height": info["height"],
                        "duration": dur, "keep_ranges": [(0.0, dur)]}
            located = scoreboard.locate(clip, analysis, self.cfg, self.work)
            if located.get("error"):
                return located
            self.boxes = {a: tuple(b) for a, b in located["counters"].items()}
            self.counters = {a: Counter() for a in self.boxes}
            return {"counters": {a: list(b) for a, b in self.boxes.items()},
                    "width": info["width"], "height": info["height"]}
        finally:
            # The point of the whole module: the video does not survive being
            # read, not even the warm-up.
            clip.unlink(missing_ok=True)

    def step(self) -> Dict:
        """Read one chunk. Returns what changed, and keeps no video."""
        fps = float(self.cfg.get("score_sample_fps", 5))
        clip = record(self.media_url, self.chunk_s, self.work / "_chunk.mp4")
        if clip is None:
            self.misses += 1
            return {"error": "chunk unreadable", "misses": self.misses}
        try:
            reads = raw_reads(clip, self.boxes, fps, self.work)
        finally:
            clip.unlink(missing_ok=True)

        events = []
        for tag, (values, times) in reads.items():
            shifted = [self.elapsed + t for t in times]
            for t, v in self.counters[tag].feed(values, shifted):
                prev = [p for p in self.counters[tag].points if p[0] < t]
                if prev:
                    events.append({"t": round(t, 2), "alliance": tag,
                                   "balls": v - prev[-1][1], "total": v})
        self.elapsed += self.chunk_s
        self.chunks += 1
        events.sort(key=lambda e: e["t"])
        return {"events": events, "totals": self.totals(),
                "elapsed": round(self.elapsed, 1)}

    def totals(self) -> Dict[str, Optional[int]]:
        return {a: c.current for a, c in self.counters.items()}

    def series(self) -> Dict[str, List[List[float]]]:
        return {a: [[round(t, 3), v] for t, v in c.points]
                for a, c in self.counters.items()}


def watch(media_url: str, cfg: dict, work: Path, on_step: Callable,
          chunk_s: float = CHUNK_S, max_s: float = 0.0,
          bootstrap_s: float = BOOTSTRAP_S) -> Dict:
    """Bootstrap, then read chunk after chunk until told to stop.

    `on_step` is called with each chunk's result and returns False to stop, so
    the caller decides what ends a session -- a match ending, a key press, a
    schedule running out -- without this module knowing about any of them.
    """
    w = Watcher(media_url, cfg, work, chunk_s=chunk_s)
    boot = w.bootstrap(bootstrap_s)
    if boot.get("error"):
        return {"error": boot["error"], "watcher": w}
    started = time.monotonic()
    while True:
        result = w.step()
        if on_step(result, w) is False:
            break
        if max_s and time.monotonic() - started > max_s:
            break
        # Consecutive failures mean the stream has gone, not that a chunk was
        # unlucky. Three is enough to tell those apart at this chunk size.
        if w.misses >= 3:
            return {"error": "the stream stopped answering", "watcher": w,
                    "series": w.series(), "totals": w.totals()}
    return {"watcher": w, "series": w.series(), "totals": w.totals(),
            "boxes": {a: list(b) for a, b in w.boxes.items()}}
