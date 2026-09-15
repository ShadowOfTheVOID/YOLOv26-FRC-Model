"""Thin ffmpeg/ffprobe helpers shared by the shot, crop and render stages."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def require_tools() -> None:
    missing = [n for n, p in (("ffmpeg", shutil.which("ffmpeg")),
                              ("ffprobe", shutil.which("ffprobe")),
                              ("yt-dlp", shutil.which("yt-dlp"))) if not p]
    if missing:
        raise SystemExit(
            "Missing required tools: " + ", ".join(missing) +
            "\n  macOS:  brew install ffmpeg yt-dlp"
            "\n  Debian: apt install ffmpeg  +  see deploy/DEBIAN.md for yt-dlp"
        )


def run(cmd: List[str], capture_stderr: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
        text=capture_stderr,
    )


def probe(path: Path) -> Optional[dict]:
    """Return {duration, width, height, fps} for the first video stream."""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
         "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        info = json.loads(proc.stdout)
        stream = info["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None

    num, _, den = (stream.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return {
        "duration": float(info.get("format", {}).get("duration") or 0.0),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
    }


def grab_frame(path: Path, timestamp: float, width: int, height: int,
               pre_filter: str = "") -> Optional[np.ndarray]:
    """Decode a single frame at `timestamp`, scaled to width x height RGB.

    -ss before -i is the fast seek: ffmpeg jumps to the nearest keyframe
    instead of decoding from zero, which matters when we're sampling dozens
    of frames out of a 10-minute video. `pre_filter` runs before the scale,
    so callers measuring row profiles can strip letterboxing first and keep
    their row fractions meaningful.
    """
    nbytes = width * height * 3
    vf = f"{pre_filter},scale={width}:{height}" if pre_filter else f"scale={width}:{height}"
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{max(timestamp, 0):.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", vf,
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0 or len(proc.stdout) < nbytes:
        return None
    arr = np.frombuffer(proc.stdout[:nbytes], dtype=np.uint8)
    return arr.reshape((height, width, 3))


def write_thumb(path: Path, timestamp: float, dest: Path, width: int = 320) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-ss", f"{max(timestamp, 0):.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(dest)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0 and dest.exists()


def even(n: int) -> int:
    """H.264 needs even dimensions."""
    return int(n) - (int(n) % 2)
