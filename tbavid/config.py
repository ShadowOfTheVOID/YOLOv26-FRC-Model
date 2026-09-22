"""Config loading and project paths."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Point the bulk data somewhere else without moving the code:
#   export TBAVID_DATA=/Volumes/Scratch/tbavid
# Video and frames are ~330 MB per match, so this is the difference between
# filling a laptop disk and not.
DATA = Path(os.environ.get("TBAVID_DATA") or (ROOT / "data")).expanduser()
RAW_DIR = DATA / "raw"
VIDEO_DIR = DATA / "videos"
FRAME_DIR = DATA / "frames"
REVIEW_DIR = DATA / "review"
THUMB_DIR = REVIEW_DIR / "thumbs"
LABEL_DIR = DATA / "labels"
OCR_WORK = DATA / ".ocr_work"

STATE = ROOT / "state"
CACHE_DIR = STATE / "tba_cache"
LEDGER_PATH = STATE / "seen.json"
MANIFEST_PATH = REVIEW_DIR / "manifest.json"

ALL_DIRS = (RAW_DIR, VIDEO_DIR, FRAME_DIR, REVIEW_DIR, THUMB_DIR, LABEL_DIR,
            OCR_WORK, STATE, CACHE_DIR)

DEFAULTS = {
    "season": 2026,
    "max_duration_s": 900,
    "min_duration_s": 60,
    "max_height": 1080,
    "prefer_h264": True,
    "min_shot_s": 1.5,
    "cut_sigma": 6.0,
    "cut_min_delta": 0.06,
    "edge_inset_s": 0.15,
    "shot_cluster_dist": 0.12,
    "shot_absorb_dist": 0.20,
    "reject_threshold": 0.40,
    "crop": {
        "auto_detect_banner": True,
        "auto_detect_split": True,
        # Which broadcast layout profile to use: "auto" picks one from the
        # event's TBA district, its key and the video title (see formats.py).
        # A name here forces it for the whole run; `formats` does it per event.
        "format": "auto",
        "formats": {},
        "overrides": {},
        "top": 0.16,
        "bottom": 0.0,
        "left": 0.0,
        "right": 0.0,
    },
    "render_crf": 18,
    "sample_fps": 3,
    "jpeg_quality": 2,
    "dedupe_hamming": 6,
    "score_labels": True,
    "score_sample_fps": 5,
    "score_window_s": 1.5,
    "score_tail_s": 20,
    "score_latency_s": 0.0,
    "max_frames_per_video": 0,
    "ytdlp_cookies": "",
    # Official competition only: no offseason or preseason events, and no
    # practice matches. Set false (or pass --include-noncompetitive) to widen
    # the catalogue to every event of the season.
    "competitive_only": True,
    "keep_raw": True,
    "keep_clean": True,
    "tba_min_interval_s": 0.15,
}


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = None) -> dict:
    """DEFAULTS overlaid with config.json (one level of nesting merged)."""
    path = path or (ROOT / "config.json")
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if path.exists():
        user = json.loads(path.read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def tba_key() -> str:
    """TBA auth key from $TBA_AUTH_KEY, falling back to a gitignored .env."""
    key = os.environ.get("TBA_AUTH_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, val = line.partition("=")
            if name.strip() == "TBA_AUTH_KEY":
                return val.strip().strip("'\"")
    raise SystemExit(
        "No TBA API key found.\n"
        "  Get one at https://www.thebluealliance.com/account then either:\n"
        "    export TBA_AUTH_KEY=...\n"
        "  or write it to .env (gitignored) as:\n"
        "    TBA_AUTH_KEY=..."
    )
