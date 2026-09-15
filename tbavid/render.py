"""Render main-camera-only cropped video, then sample frames out of it."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ffm import FFMPEG

# dHash grid: (W-1)*H bits. The classic 9x8/64-bit hash is far too coarse for
# a 1920x504 field strip -- each cell covers 213x63 px, so a moving ball is
# invisible to it and consecutive real frames score a median distance of only
# 3. At 17x16 the median rises to 18, which cleanly separates "genuinely
# different" from "dead-time still" and keeps dedupe to its actual job.
HASH_W, HASH_H = 17, 16


def build_filter(ranges: List[Tuple[float, float]], crop_filter: str) -> str:
    """trim/concat the kept ranges, then crop once at the end."""
    if len(ranges) == 1:
        start, end = ranges[0]
        return (f"[0:v]trim=start={start:.3f}:end={end:.3f},"
                f"setpts=PTS-STARTPTS,{crop_filter}[out]")

    parts = []
    labels = []
    for i, (start, end) in enumerate(ranges):
        parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},"
                     f"setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(ranges)}:v=1:a=0[vc]")
    parts.append(f"[vc]{crop_filter}[out]")
    return ";".join(parts)


def render_clean(src: Path, dest: Path, ranges: List[Tuple[float, float]],
                 crop_filter: str, cfg: dict) -> Tuple[bool, str]:
    if not ranges:
        return False, "no kept ranges"
    dest.parent.mkdir(parents=True, exist_ok=True)
    graph = build_filter(ranges, crop_filter)
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-i", str(src),
         "-filter_complex", graph, "-map", "[out]",
         "-c:v", "libx264", "-preset", "medium",
         "-crf", str(cfg["render_crf"]), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(dest)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0 or not dest.exists():
        return False, (proc.stderr or "").strip()[-400:]
    return True, ""


def frames_filter(ranges: List[Tuple[float, float]], crop_filter: str,
                  fps: float, extra: str = "") -> str:
    """Same trim/concat/crop graph as the renderer, ending in a frame rate."""
    graph = build_filter(ranges, crop_filter)
    tail = f",fps={fps}" + (f",{extra}" if extra else "")
    # build_filter ends with [out]; splice the rate onto that last stage.
    return graph[: graph.rfind("[out]")] + tail + "[out]"


def export_frames_direct(src: Path, out_dir: Path, stem: str,
                         ranges: List[Tuple[float, float]], crop_filter: str,
                         cfg: dict) -> Dict:
    """Frames straight from the source, skipping the intermediate video.

    When `keep_clean` is false the cleaned .mp4 is deleted the moment its
    frames exist, so encoding it is pure waste -- an x264 pass plus a second
    decode, about 86 s of a 190 s match on an M2 and proportionally worse on
    slower hardware. Trim, concat, crop and sample in one pass instead.
    """
    if not ranges:
        return {"written": 0, "error": "no kept ranges"}
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / f".staging_{stem}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    fps = cfg["sample_fps"]
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-i", str(src),
         "-filter_complex", frames_filter(ranges, crop_filter, fps),
         "-map", "[out]", "-q:v", str(cfg["jpeg_quality"]),
         str(staging / "f_%06d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    staged = sorted(staging.glob("f_*.jpg"))
    if not staged:
        shutil.rmtree(staging, ignore_errors=True)
        return {"written": 0, "error": (proc.stderr or "").strip()[-300:]}

    hashes = []
    threshold = int(cfg.get("dedupe_hamming", 0) or 0)
    if threshold > 0:
        hashes = _dhashes_filtered(src, ranges, crop_filter, fps)
    return _finish(staged, hashes, threshold, staging, out_dir, stem, fps, cfg)


def _dhashes_filtered(src: Path, ranges, crop_filter: str, fps: float) -> List[int]:
    graph = frames_filter(ranges, crop_filter, fps, f"scale={HASH_W}:{HASH_H}")
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(src), "-filter_complex", graph,
         "-map", "[out]", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return _hashes_from_bytes(proc.stdout)


# -- frame sampling --------------------------------------------------------
def _dhashes(path: Path, fps: float) -> List[int]:
    """One 64-bit perceptual hash per sampled frame, in export order.

    Second decode pass rather than a split filter graph: the cleaned videos are
    short, and keeping the two ffmpeg invocations independent makes a failure
    in either one obvious.
    """
    nbytes = HASH_W * HASH_H
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path),
         "-vf", f"fps={fps},scale={HASH_W}:{HASH_H}",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return _hashes_from_bytes(proc.stdout)


def _hashes_from_bytes(data: bytes) -> List[int]:
    nbytes = HASH_W * HASH_H
    out = []
    for off in range(0, len(data) - nbytes + 1, nbytes):
        g = np.frombuffer(data[off:off + nbytes], dtype=np.uint8).reshape(HASH_H, HASH_W)
        bits = (g[:, 1:] > g[:, :-1]).ravel()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        out.append(value)
    return out


def export_frames(video: Path, out_dir: Path, stem: str, cfg: dict) -> Dict:
    """Sample JPGs at sample_fps, drop near-duplicates, write into out_dir.

    Frames stay at the cleaned video's native resolution -- fuel balls are only
    ~25px across at 1080p, and pre-shrinking here would cost detail that
    training at a larger imgsz can otherwise exploit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / f".staging_{stem}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    fps = cfg["sample_fps"]
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-i", str(video),
         "-vf", f"fps={fps}", "-q:v", str(cfg["jpeg_quality"]),
         str(staging / "f_%06d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    staged = sorted(staging.glob("f_*.jpg"))
    if proc.returncode != 0 and not staged:
        shutil.rmtree(staging, ignore_errors=True)
        return {"written": 0, "error": (proc.stderr or "").strip()[-300:]}

    hashes = _dhashes(video, fps) if int(cfg.get("dedupe_hamming", 0) or 0) > 0 else []
    return _finish(staged, hashes, int(cfg.get("dedupe_hamming", 0) or 0),
                   staging, out_dir, stem, fps, cfg)


def _finish(staged, hashes, threshold, staging, out_dir, stem, fps, cfg) -> Dict:
    """Dedupe, cap and name the staged frames. Shared by both export paths."""
    keep_idx = list(range(len(staged)))
    duplicates = 0
    if threshold > 0 and len(hashes) >= len(staged):
        keep_idx = []
        last = None
        for i in range(len(staged)):
            h = hashes[i]
            if last is None or bin(h ^ last).count("1") >= threshold:
                keep_idx.append(i)
                last = h
            else:
                duplicates += 1

    cap = int(cfg.get("max_frames_per_video", 0) or 0)
    if cap and len(keep_idx) > cap:
        sel = np.linspace(0, len(keep_idx) - 1, cap).round().astype(int)
        keep_idx = [keep_idx[i] for i in sorted(set(sel.tolist()))]

    # staged[i] is the frame the fps filter emitted at t = i/fps on the
    # concatenated (kept-ranges) timeline; callers map that to broadcast time.
    frames = []
    for n, i in enumerate(keep_idx, start=1):
        name = f"{stem}_{n:06d}.jpg"
        staged[i].replace(out_dir / name)
        frames.append({"file": name, "t_clean": round(i / fps, 4)})

    shutil.rmtree(staging, ignore_errors=True)
    return {"written": len(frames), "sampled": len(staged),
            "duplicates": duplicates, "frames": frames}
