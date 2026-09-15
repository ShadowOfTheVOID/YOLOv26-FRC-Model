"""yt-dlp wrapper with a duration guard."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from . import ledger as L

YTDLP = shutil.which("yt-dlp") or "yt-dlp"

UNAVAILABLE_MARKERS = (
    "video unavailable", "private video", "has been removed",
    "not available in your country", "sign in to confirm",
    "members-only", "this live event", "account associated with this video",
    "age-restricted", "video has been terminated",
)


def watch_url(yt_key: str) -> str:
    return f"https://www.youtube.com/watch?v={yt_key}"


def cookie_args(cfg: dict) -> list:
    """`--cookies FILE` when configured, else nothing.

    YouTube's bot detection treats datacenter IPs harshly, so a download that
    works from a home connection can fail from a cloud VM. Presenting cookies
    from a signed-in browser is the usual way through; it is not guaranteed and
    the cookies expire, which is why this stays opt-in rather than a default.
    """
    path = (cfg or {}).get("ytdlp_cookies")
    if path and Path(path).exists():
        return ["--cookies", str(path)]
    return []


def probe_remote(yt_key: str, cfg: Optional[dict] = None
                 ) -> Tuple[Optional[dict], Optional[str]]:
    """Fetch metadata only. Returns (info, failure_status)."""
    proc = subprocess.run(
        [YTDLP, "-J", "--no-playlist", "--no-warnings"]
        + cookie_args(cfg) + [watch_url(yt_key)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").lower()
        if any(m in err for m in UNAVAILABLE_MARKERS):
            return None, L.UNAVAILABLE
        return None, L.FAILED
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, L.FAILED


def download(yt_key: str, dest_dir: Path, cfg: dict) -> Tuple[Optional[Path], Optional[str], dict]:
    """Download one video. Returns (path, failure_status, meta).

    Probes first so a full-day event stream -- TBA does sometimes link one --
    is rejected before it eats the disk rather than after.
    """
    info, fail = probe_remote(yt_key, cfg)
    if fail:
        return None, fail, {}

    duration = float(info.get("duration") or 0)
    meta = {"title": info.get("title", ""), "duration": duration}

    if duration > cfg["max_duration_s"]:
        return None, L.TOO_LONG, meta
    if duration and duration < cfg["min_duration_s"]:
        return None, L.TOO_SHORT, meta

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(dest_dir / f"{yt_key}.%(ext)s")
    height = cfg["max_height"]
    # YouTube serves AV1 for some uploads and H.264 for others. AV1 has no
    # hardware decode before roughly 2020 and is slow in software, so on older
    # machines every later stage -- shot analysis, crop probing, OCR, frame
    # export -- pays for it repeatedly. Asking for H.264 first costs a little
    # file size and saves far more decode time.
    fmt = (f"bv*[height<={height}][vcodec^=avc1]/bv*[height<={height}]"
           f"/b[height<={height}]/bv*/b") if cfg.get("prefer_h264") else \
          f"bv*[height<={height}]/b[height<={height}]/bv*/b"

    proc = subprocess.run(
        [YTDLP,
         "-f", fmt,
         "--no-playlist", "--no-warnings", "--no-part",
         "--retries", "5", "--fragment-retries", "5"]
        + cookie_args(cfg) + ["-o", out_tmpl, watch_url(yt_key)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    matches = sorted(dest_dir.glob(f"{yt_key}.*"))
    if proc.returncode != 0 or not matches:
        for stray in matches:
            stray.unlink(missing_ok=True)
        err = (proc.stderr or "").lower()
        status = L.UNAVAILABLE if any(m in err for m in UNAVAILABLE_MARKERS) else L.FAILED
        return None, status, meta

    return matches[0], None, meta
