"""Split a broadcast into shots and work out which ones are the main camera.

FRC broadcasts cut between a fixed elevated field camera and side / field-level
cameras, replays, and crowd shots. The main camera is fixed, so every main-cam
shot produces a near-identical downscaled signature -- they collapse into one
big cluster, and that cluster owns most of the runtime. Everything else is
smaller and visually distinct, so "keep the longest-total-duration cluster"
separates them without needing a single labelled frame.

Cuts are found from our own frame-difference signal rather than ffmpeg's
`scene` filter. That filter is histogram-based and scored a real cut in
testing at 0.037 -- nowhere near a usable threshold -- because two shots of
the same venue can share a histogram. A mean-absolute-difference signal with
an adaptive threshold catches those, and the same decode pass feeds the shot
signatures, so the whole analysis costs one pass over the video.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ffm import FFMPEG, probe, write_thumb

DEC_W, DEC_H = 64, 36     # decode resolution for diffs and signatures
DEC_FPS = 10.0            # cut-time precision of 1/DEC_FPS


def decode_strip(path: Path, fps: float = DEC_FPS) -> Optional[np.ndarray]:
    """Decode the whole video once at low res. Returns (N, H, W, 3) uint8."""
    nbytes = DEC_W * DEC_H * 3
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path),
         "-vf", f"fps={fps},scale={DEC_W}:{DEC_H}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    n = len(proc.stdout) // nbytes
    if n < 2:
        return None
    arr = np.frombuffer(proc.stdout[:n * nbytes], dtype=np.uint8)
    return arr.reshape((n, DEC_H, DEC_W, 3))


def find_cuts(strip: np.ndarray, sigma: float, min_delta: float,
              fps: float = DEC_FPS, min_gap_s: float = 0.5) -> List[float]:
    """Cut timestamps from spikes in frame-to-frame difference.

    The threshold is derived from the video's own noise floor (median + k*MAD)
    so a grainy stream and a clean one both get sensible treatment, with an
    absolute floor so a near-static video doesn't shatter into fake shots.
    """
    f = strip.astype(np.float32) / 255.0
    d = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))    # (N-1,)
    if d.size == 0:
        return []

    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    robust_sigma = mad * 1.4826
    thresh = max(med + sigma * robust_sigma, min_delta)

    over = np.where(d > thresh)[0]
    if over.size == 0:
        return []

    # Collapse each run of over-threshold frames to its single strongest frame:
    # a dissolve or wipe spans several frames but is still one cut.
    min_gap = max(int(round(min_gap_s * fps)), 1)
    cuts: List[float] = []
    run = [over[0]]
    for idx in over[1:]:
        if idx - run[-1] <= min_gap:
            run.append(idx)
        else:
            cuts.append(float(max(run, key=lambda i: d[i]) + 1) / fps)
            run = [idx]
    cuts.append(float(max(run, key=lambda i: d[i]) + 1) / fps)
    return cuts


def build_shots(cuts: List[float], duration: float) -> List[Tuple[float, float]]:
    bounds = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] > 0.05]


def shot_signature(strip: np.ndarray, start: float, end: float,
                   fps: float = DEC_FPS) -> Optional[np.ndarray]:
    """Average the middle 50% of a shot's frames into one 0..1 vector.

    Trimming to the middle avoids dissolve frames at either boundary, which
    are a blend of two cameras and would smear the signature.
    """
    span = end - start
    lo = int(round((start + span * 0.25) * fps))
    hi = int(round((start + span * 0.75) * fps))
    lo = max(0, min(lo, len(strip) - 1))
    hi = max(lo + 1, min(hi, len(strip)))
    window = strip[lo:hi]
    if window.size == 0:
        return None
    return (window.astype(np.float32) / 255.0).mean(axis=0).ravel()


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def cluster_shots(sigs: List[np.ndarray], durations: List[float],
                  cluster_dist: float, absorb_dist: float) -> List[int]:
    """Greedy duration-weighted clustering. Returns a cluster id per shot.

    Seeding from the longest shots first anchors clusters on the footage we're
    most confident about instead of on a half-second cutaway.
    """
    order = sorted(range(len(sigs)), key=lambda i: -durations[i])
    centroids: List[np.ndarray] = []
    weights: List[float] = []
    assignment = [-1] * len(sigs)

    for idx in order:
        sig = sigs[idx]
        best, best_d = -1, float("inf")
        for cid, centroid in enumerate(centroids):
            d = distance(sig, centroid)
            if d < best_d:
                best, best_d = cid, d
        if best >= 0 and best_d <= cluster_dist:
            w = weights[best]
            dw = max(durations[idx], 1e-6)
            centroids[best] = (centroids[best] * w + sig * dw) / (w + dw)
            weights[best] = w + dw
            assignment[idx] = best
        else:
            centroids.append(sig.copy())
            weights.append(max(durations[idx], 1e-6))
            assignment[idx] = len(centroids) - 1

    return _absorb(assignment, centroids, weights, absorb_dist)


def _absorb(assignment, centroids, weights, absorb_dist) -> List[int]:
    """Fold near-duplicate clusters into the dominant one.

    Slow exposure or zoom drift on the main camera can otherwise split it into
    two clusters, halving its apparent runtime share.
    """
    if len(centroids) < 2:
        return assignment
    winner = max(range(len(centroids)), key=lambda c: weights[c])
    remap = {c: winner for c in range(len(centroids))
             if c != winner
             and distance(centroids[c], centroids[winner]) <= absorb_dist}
    return [remap.get(c, c) for c in assignment] if remap else assignment


def merge_ranges(ranges: List[Tuple[float, float]], gap: float = 0.10
                 ) -> List[Tuple[float, float]]:
    """Fuse back-to-back kept shots so the render filter graph stays small."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    out = [list(ranges[0])]
    for start, end in ranges[1:]:
        if start - out[-1][1] <= gap:
            out[-1][1] = end
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def analyze(path: Path, cfg: dict, thumb_dir: Optional[Path] = None,
            thumb_prefix: str = "") -> Dict:
    """Full shot analysis for one downloaded video."""
    info = probe(path)
    if not info or info["duration"] <= 0:
        return {"error": "could not probe video"}
    duration = info["duration"]

    strip = decode_strip(path)
    if strip is None:
        return {"error": "could not decode video"}

    cuts = find_cuts(strip, cfg["cut_sigma"], cfg["cut_min_delta"])
    raw_shots = build_shots(cuts, duration)

    shots: List[Dict] = []
    sigs: List[np.ndarray] = []
    durations: List[float] = []
    for start, end in raw_shots:
        sig = shot_signature(strip, start, end)
        if sig is None:
            continue
        shots.append({"start": round(start, 3), "end": round(end, 3),
                      "duration": round(end - start, 3)})
        sigs.append(sig)
        durations.append(end - start)

    if not shots:
        return {"error": "no analysable shots"}

    assignment = cluster_shots(sigs, durations, cfg["shot_cluster_dist"],
                               cfg["shot_absorb_dist"])

    totals: Dict[int, float] = {}
    for cid, dur in zip(assignment, durations):
        totals[cid] = totals.get(cid, 0.0) + dur
    main_cluster = max(totals, key=lambda c: totals[c])

    for shot, cid in zip(shots, assignment):
        shot["cluster"] = int(cid)
        shot["is_main"] = cid == main_cluster
    _apply_min_shot(shots, cfg["min_shot_s"])

    if thumb_dir is not None:
        for i, shot in enumerate(shots):
            name = f"{thumb_prefix}shot{i:03d}.jpg"
            if write_thumb(path, shot["start"] + shot["duration"] / 2,
                           thumb_dir / name):
                shot["thumb"] = name

    return finalize(shots, duration, cfg, info, len(cuts), len(set(assignment)),
                    int(main_cluster))


def _apply_min_shot(shots: List[Dict], min_shot: float) -> None:
    """Set `keep` on main-camera shots whose contiguous run is long enough.

    The duration test runs on the merged run, not the individual shot. A cut
    detector that over-segments a long main-camera stretch would otherwise see
    a string of short pieces and discard all of them, punching holes straight
    through the footage we most want.
    """
    i = 0
    while i < len(shots):
        if not shots[i]["is_main"]:
            shots[i]["keep"] = False
            i += 1
            continue
        j = i
        while j + 1 < len(shots) and shots[j + 1]["is_main"]:
            j += 1
        run = shots[i:j + 1]
        keep = sum(s["duration"] for s in run) >= min_shot
        for s in run:
            s["keep"] = keep
        i = j + 1


def finalize(shots: List[Dict], duration: float, cfg: dict, info: dict,
             n_cuts: int, n_clusters: int, main_cluster: int) -> Dict:
    """Derive keep_ranges + coverage from per-shot keep flags.

    Split out from analyze() so the review UI can re-derive them after a human
    flips some shots, without re-running detection.
    """
    inset = float(cfg.get("edge_inset_s", 0.15))
    kept = []
    for s in shots:
        if not s.get("keep"):
            continue
        # Pull in from both ends: cut times are quantised to the decode fps,
        # so the true boundary can be a frame or two either side.
        a, b = s["start"] + inset, s["end"] - inset
        if b - a > 0.2:
            kept.append((a, b))

    coverage = sum(b - a for a, b in kept) / duration if duration else 0.0
    return {
        "duration": round(duration, 3),
        "width": info["width"],
        "height": info["height"],
        "fps": round(info["fps"], 3),
        "n_cuts": n_cuts,
        "n_shots": len(shots),
        "n_clusters": n_clusters,
        "main_cluster": main_cluster,
        "coverage": round(coverage, 4),
        "shots": shots,
        "keep_ranges": [[round(a, 3), round(b, 3)] for a, b in merge_ranges(kept)],
    }
