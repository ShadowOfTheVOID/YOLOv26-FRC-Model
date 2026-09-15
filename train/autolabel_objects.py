#!/usr/bin/env python3
"""Propose robot and hub boxes.

Robots: alliance bumpers are saturated blue/red, but so are the ramps, the
alliance walls and half the crowd. What separates a robot from all of that is
that a robot MOVES, so the gate is colour AND departure from the temporal
median background, then a shape filter -- a bumper reads as a wide, low
rectangle, not a tall blob of someone's shirt.

Hubs: static for the whole event, because the camera is. Rather than guess at
them every frame, record the two boxes once per event
(`run.py db hub --event ... --alliance blue --box x,y,w,h`) and this replays
them into every frame of every match at that event.

Robot boxes are PROPOSALS with real false positives (people in alliance
colours near the rail). Correct them on a subset, train, then let the detector
label the rest -- that bootstrap is cheaper than fighting the colour gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # train/ scripts import the tbavid package
DATASET = ROOT / "dataset"
CLASSES = {"fuel": 0, "robot_blue": 1, "robot_red": 2, "hub_blue": 3, "hub_red": 4}

BLUE = [((100, 120, 60), (130, 255, 255))]
RED = [((0, 120, 60), (8, 255, 255)), ((170, 120, 60), (180, 255, 255))]

MIN_AREA = 500
MIN_W, MIN_H = 22, 10
MAX_ASPECT_TALL = 1.4      # bumpers are wider than tall; shirts and people are not


def background(video: Path, samples: int = 40) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(n - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"could not read {video}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def robots(img: np.ndarray, bg: np.ndarray, roi_top: float, roi_bottom: float) -> list:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = img.shape[0]
    moving = cv2.morphologyEx(
        (cv2.absdiff(img, bg).max(axis=2) > 40).astype(np.uint8) * 255,
        cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    out = []
    for cls, ranges in (("robot_blue", BLUE), ("robot_red", RED)):
        mask = np.zeros(img.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(cv2.bitwise_and(mask, moving), cv2.MORPH_CLOSE,
                                np.ones((15, 15), np.uint8))
        k, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, k):
            x, y, w, h, area = stats[i]
            if area < MIN_AREA or w < MIN_W or h < MIN_H:
                continue
            if h > w * MAX_ASPECT_TALL:
                continue                       # standing person, not a bumper
            cy = (y + h / 2) / H
            if cy < roi_top or cy > roi_bottom:
                continue                       # crowd above, rail below
            out.append((cls, int(x), int(y), int(w), int(h)))
    return out


def hub_boxes(event_key: str) -> list:
    from tbavid.db import connect
    con = connect()
    row = con.execute("SELECT hub_blue, hub_red FROM events WHERE event_key=?",
                      (event_key,)).fetchone()
    con.close()
    if not row:
        return []
    out = []
    for cls, raw in (("hub_blue", row["hub_blue"]), ("hub_red", row["hub_red"])):
        if raw:
            x, y, w, h = json.loads(raw)
            out.append((cls, x, y, w, h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="cleaned video for one match (default: all in db)")
    ap.add_argument("--roi-top", type=float, default=0.05)
    ap.add_argument("--roi-bottom", type=float, default=0.80,
                    help="ignore detections below this fraction (the near rail)")
    ap.add_argument("--preview", type=Path)
    ap.add_argument("--append", action="store_true",
                    help="append to existing label files instead of skipping them")
    args = ap.parse_args()

    from tbavid.db import connect
    con = connect()
    rows = con.execute(
        "SELECT match_key, event_key, video_id, clean_path FROM matches "
        "WHERE clean_path IS NOT NULL AND status='ok'").fetchall()
    con.close()
    if args.video:
        rows = [r for r in rows if r["clean_path"] == args.video]
    if not rows:
        print("no matches with a cleaned video -- run `run.py db build` first")
        return 1

    total = 0
    for r in rows:
        video = Path(r["clean_path"])
        if not video.exists():
            print(f"  ! missing {video}")
            continue
        bg = background(video)
        hubs = hub_boxes(r["event_key"])
        if not hubs:
            print(f"  {r['event_key']}: no hub geometry recorded "
                  f"(run.py db hub --event {r['event_key']} --alliance blue --box x,y,w,h)")

        images = sorted(p for split in ("train", "val")
                        for p in (DATASET / "images" / split).glob(f"{r['video_id']}_*.jpg"))
        if args.preview and images:
            img = cv2.imread(str(images[len(images) // 2]))
            for cls, x, y, w, h in robots(img, bg, args.roi_top, args.roi_bottom) + hubs:
                col = (255, 0, 0) if "blue" in cls else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
                cv2.putText(img, cls, (x, max(y - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.preview), img)
            print(f"preview -> {args.preview}")
            return 0

        for src in images:
            img = cv2.imread(str(src))
            if img is None:
                continue
            H, W = img.shape[:2]
            boxes = robots(img, bg, args.roi_top, args.roi_bottom) + hubs
            lines = [f"{CLASSES[c]} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}"
                     for c, x, y, w, h in boxes]
            dst = DATASET / "labels" / src.parent.name / f"{src.stem}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            existing = dst.read_text() if (dst.exists() and args.append) else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            dst.write_text(existing + "\n".join(lines) + ("\n" if lines else ""))
            total += len(boxes)
        print(f"  {r['match_key']}: {len(images)} frames")

    print(f"\n{total} robot/hub proposals written")
    print("Robot boxes are proposals -- people in alliance colours near the field "
          "still get through. Review before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
