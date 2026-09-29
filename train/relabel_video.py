#!/usr/bin/env python3
"""Label match video with the detector itself, plus the balls it cannot see.

The bootstrap step of train/README.md -- train, relabel with the detector,
correct, retrain -- for the two gaps the released models were measured to
have (deploy/HUB_FEED.md, 2026-09-29):

- **Balls in flight were never labelled.** autolabel_fuel's learned field line
  left every ball above it unboxed, so the model was taught that a ball high
  against the crowd is background. On Einstein 4 it found 84% of moving
  ball-sized yellow blobs near the hubs on the halved frame.
- **It has only seen one broadcast** (2026nhdur), never an Einstein frame.

Each sampled frame is labelled like this:

1. Fuel from `--fuel` on six overlapping 640 px tiles at FULL resolution. At
   imgsz 960 a 1080p frame is halved and an 18 px ball arrives as 9 px; on
   tiles the same model found 93% of moving balls against 84%. Boxes at
   conf >= 0.4 are kept; 0.1-0.4 only if the box is at least 35% yellow.
2. Balls in flight the detector still missed: a ball-sized yellow blob that
   moved since the frame two frames earlier, with no box on it, gets one.
3. Anything else yellow and ball-like -- a compact blob neither boxed nor
   moving -- is painted grey in the saved image. Left in unboxed it would
   teach "this is not fuel", the mistake being fixed. Large irregular
   yellow (crowd shirts) is left alone: those are the negatives the model
   needs.
4. Robots from `--robots` on the full frame at imgsz 960, conf >= 0.5 (robot
   detection was checked good by eye on qm7).

Each frame is saved whole (for full-frame inference at 960, what `run.py
detect` and `shots` do) and as up to `--tiles` crops centred on balls in
flight, 960 px wide at full resolution (see SAVE_TILE). Frames with more
than `--max-grey` balls' worth of greyed yellow are skipped: an image that
is half paint teaches little.

Splits are per VIDEO, never per frame (prepare_dataset.py's rule): name the
split with each video. Keep a scored match out of every split to test on.

    python3 train/relabel_video.py --out dataset_einstein \\
        --video e4.mp4:train:6 --video e5.mp4:train:6 --video e1.mp4:val:6 \\
        --fuel models/fuel_best.pt --robots models/fuel_withBotbest.pt

`VIDEO:SPLIT:START` -- START is when the match clock starts, in video
seconds; frames are taken every `--every` s for `--length` s after it.
Classes are fuel, robot_blue, robot_red (fuel_withBotbest's order);
subset_classes.py derives a fuel-only set. `--preview N` draws N frames with
each box coloured by where it came from.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Box = Tuple[float, float, float, float]
NAMES = ["fuel", "robot_blue", "robot_red"]
TILE = 640               # detection tiles: full resolution at imgsz 640
# Saved flight tiles. Ultralytics resizes every training image so its long
# side is imgsz (960 in train.py), so a 640 px tile would be enlarged 1.5x and
# its balls no longer be at full resolution. A 960 px wide tile loads at
# scale 1; whole frames load at 0.5 -- the model sees both scales.
SAVE_TILE = 960
KEEP_CONF = 0.4          # detector boxes kept outright
CHECK_CONF = 0.1         # kept from here if the box is yellow enough
YELLOW_FRAC = 0.35
ROBOT_CONF = 0.5
MOVE_FRAC = 0.5          # share of a blob's pixels not yellow 2 frames ago


def tiles_for(w: int, h: int, size: int = TILE) -> List[Tuple[int, int]]:
    """Top-left corners of overlapping size x size tiles covering the frame."""
    def starts(total: int) -> List[int]:
        n = max(1, math.ceil(total / size))
        if n == 1:
            return [0]
        step = (total - size) / (n - 1)
        return [int(round(i * step)) for i in range(n)]
    return [(x, y) for y in starts(h) for x in starts(w)]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nms(boxes: Sequence[Tuple[Box, float]], thr: float = 0.5) -> List[Tuple[Box, float]]:
    """Tiles overlap, so one ball can be found twice; keep the surer box."""
    out: List[Tuple[Box, float]] = []
    for b, c in sorted(boxes, key=lambda x: -x[1]):
        if all(iou(b, k) < thr for k, _ in out):
            out.append((b, c))
    return out


def to_yolo(cls: int, b: Box, w: int, h: int) -> str:
    cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
    return f"{cls} {cx:.6f} {cy:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}"


def clip_labels(labels: Sequence[Tuple[int, Box, str]], x0: int, y0: int,
                tw: int, th: int) -> Tuple[List[Tuple[int, Box, str]], List[Box]]:
    """Labels inside a tile, shifted. A box less than half inside is returned
    as a region to grey: a cut-off ball with no label is a missing label."""
    keep, grey = [], []
    for cls, b, src in labels:
        c = (max(b[0], x0), max(b[1], y0), min(b[2], x0 + tw), min(b[3], y0 + th))
        if c[2] <= c[0] or c[3] <= c[1]:
            continue
        frac = (c[2] - c[0]) * (c[3] - c[1]) / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        shifted = (c[0] - x0, c[1] - y0, c[2] - x0, c[3] - y0)
        if frac >= 0.5:
            keep.append((cls, shifted, src))
        elif cls == 0:
            grey.append(shifted)
    return keep, grey


# Fuel yellow, strict: measured on Einstein 4, 32 confident fuel boxes had hue
# 27-30 and saturation 159-255 (5th-95th percentile). hubcount's LOOSE gate
# (S >= 55, H >= 15) also passes the Einstein split-screen's stone border
# (H 13-19, S 69-149), which confirmed 1273 junk boxes on one frame. Shade
# lowers value, not saturation, so V stays low.
FUEL_LO = (22, 120, 70)
FUEL_HI = (36, 255, 255)


def fuel_mask(img):
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(FUEL_LO, np.uint8), np.array(FUEL_HI, np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def label_frame(frame, prev, fuel_model, robot_model) -> Dict:
    """Every label and grey region for one frame (full-frame pixels).

    Grey is per PIXEL: yellow outside every fuel and robot box. Covering a
    clump's bounding box with one box was tried first and painted over whole
    clumps that were 80% labelled (112 regions, 522 balls' worth on one
    Einstein 4 frame).
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    corners = tiles_for(w, h)
    rs = fuel_model.predict([frame[y:y + TILE, x:x + TILE] for x, y in corners],
                            imgsz=TILE, conf=CHECK_CONF, max_det=1500, verbose=False)
    dets = []
    for (x0, y0), r in zip(corners, rs):
        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            dets.append(((b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), float(c)))
    dets = nms(dets)

    mask = fuel_mask(frame)
    pmask = fuel_mask(prev) if prev is not None else None

    labels: List[Tuple[int, Box, str]] = []
    for b, c in dets:
        if c >= KEEP_CONF:
            labels.append((0, b, "model"))
            continue
        x0, y0, x1, y1 = (int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3])))
        if x1 > x0 and y1 > y0 and (mask[y0:y1, x0:x1] > 0).mean() >= YELLOW_FRAC:
            labels.append((0, b, "model+colour"))
    sides = sorted(max(b[2] - b[0], b[3] - b[1]) for _, b, _ in labels)
    ball = sides[len(sides) // 2] if sides else 18.0
    one = math.pi / 4 * ball * ball

    robots: List[Tuple[int, Box, str]] = []
    r = robot_model.predict(frame, imgsz=960, conf=ROBOT_CONF, verbose=False)[0]
    for b, k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
        if int(k) in (1, 2):
            robots.append((int(k), tuple(b), "robot"))

    boxed = np.zeros((h, w), np.uint8)
    for _, b, _ in labels + robots:
        boxed[max(0, int(b[1])):int(math.ceil(b[3])), max(0, int(b[0])):int(math.ceil(b[2]))] = 1
    loose = (mask > 0) & (boxed == 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(loose.astype(np.uint8))
    grey = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < 0.3 * one:
            continue                       # specks: too small to teach anything
        comp = lab[y:y + bh, x:x + bw] == i
        moving = False
        if pmask is not None:
            was = pmask[y:y + bh, x:x + bw] > 0
            moving = (comp & ~was).sum() >= MOVE_FRAC * comp.sum()
        aspect = max(bw, bh) / max(1, min(bw, bh))
        fill = area / max(1, bw * bh)
        if moving and area <= 3.0 * one and fill >= 0.3 and aspect <= 3.0:
            pad = max(0.0, (0.8 * ball - min(bw, bh)) / 2)
            labels.append((0, (x - pad, y - pad, x + bw + pad, y + bh + pad), "motion"))
        elif fill >= 0.25 or area <= 3.0 * one:
            grey[y:y + bh, x:x + bw][comp] = 1
        # else: large and irregular -- crowd shirts, signage. Left in as the
        # negatives the model needs to stop calling them fuel.
    grey = cv2.dilate(grey, np.ones((5, 5), np.uint8))
    return {"labels": labels + robots, "grey": grey, "grey_balls": float(grey.sum()) / one,
            "ball_px": ball}


def paint_grey(img, mask=None, boxes: Sequence[Box] = (), pad: int = 2):
    out = img.copy()
    h, w = out.shape[:2]
    if mask is not None:
        out[mask > 0] = 118
    for b in boxes:
        x0, y0 = max(0, int(b[0]) - pad), max(0, int(b[1]) - pad)
        x1, y1 = min(w, int(math.ceil(b[2])) + pad), min(h, int(math.ceil(b[3])) + pad)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 118
    return out


def pick_tiles(labels, w: int, h: int, n: int, tw: int, th: int) -> List[Tuple[int, int]]:
    """Up to n tw x th tiles centred on balls in flight, not overlapping much."""
    flying = [b for cls, b, src in labels if cls == 0 and src == "motion"]
    out: List[Tuple[int, int]] = []
    for b in sorted(flying, key=lambda b: b[1]):          # highest first
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0 = int(min(max(cx - tw / 2, 0), w - tw))
        y0 = int(min(max(cy - th / 2, 0), h - th))
        if all(abs(x0 - a) > tw / 2 or abs(y0 - c) > th / 2 for a, c in out):
            out.append((x0, y0))
        if len(out) >= n:
            break
    return out


COLOURS = {"model": (0, 200, 0), "model+colour": (255, 255, 0), "motion": (255, 0, 255),
           "robot": (0, 128, 255)}


def draw(img, labels):
    """Boxes coloured by source: green detector, cyan detector confirmed by
    colour, magenta added by motion (balls in flight), orange/red robots.
    Greyed yellow is already painted into `img`."""
    import cv2
    out = img.copy()
    for cls, b, src in labels:
        colour = COLOURS[src] if cls == 0 else ((255, 80, 0) if cls == 1 else (0, 0, 255))
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colour, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", action="append", required=True, metavar="VIDEO:SPLIT:START")
    ap.add_argument("--fuel", default="models/fuel_best.pt")
    ap.add_argument("--robots", default="models/fuel_withBotbest.pt")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between frames")
    ap.add_argument("--length", type=float, default=166.0, help="seconds after START")
    ap.add_argument("--tiles", type=int, default=2, help="flight tiles per frame")
    ap.add_argument("--max-grey", type=float, default=12.0,
                    help="skip frames with more greyed yellow than this many balls")
    ap.add_argument("--preview", type=int, default=12)
    ap.add_argument("--rows", default="",
                    help="Y0:Y1 -- label only these rows. Einstein broadcasts are "
                         "split-screen: the main camera is rows 0:700, a second "
                         "camera behind a stone border below")
    ap.add_argument("--mask", action="append", default=[], metavar="X0,Y0,X1,Y1",
                    help="paint this full-frame rectangle grey before labelling: "
                         "the broadcast's scoreboard carries a yellow fuel icon "
                         "and yellow arrows that the detector boxes on every "
                         "frame. Einstein 2026: 440,0,1480,165 15,58,440,122 "
                         "1480,58,1905,122")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    masks = [tuple(int(v) for v in m.split(",")) for m in args.mask]
    rows = tuple(int(v) for v in args.rows.split(":")) if args.rows else None

    import cv2
    from ultralytics import YOLO

    fuel_model, robot_model = YOLO(args.fuel), YOLO(args.robots)
    if args.device:
        fuel_model.to(args.device)
        robot_model.to(args.device)
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    prev_dir = args.out / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)
    report = {"videos": {}, "settings": {k: v for k, v in vars(args).items()
                                         if k not in ("out", "video")}}
    previews = 0
    for spec in args.video:
        path, split, start = spec.rsplit(":", 2)
        if split not in ("train", "val"):
            raise SystemExit(f"{spec}: the split is train or val")
        stem = Path(path).stem
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"could not open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stats = {"frames": 0, "skipped_grey": 0, "tiles": 0, "fuel": 0, "motion": 0,
                 "model+colour": 0, "robots": 0, "greyed_balls": 0.0, "grey_per_frame": []}
        t = float(start)
        while t <= float(start) + args.length:
            fi = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi - 2))
            ok_p, prev = cap.read()
            cap.read()
            ok, frame = cap.read()
            t += args.every
            if not (ok and ok_p):
                break
            for x0, y0, x1, y1 in masks:
                frame[y0:y1, x0:x1] = 118
                prev[y0:y1, x0:x1] = 118
            if rows:
                frame, prev = frame[rows[0]:rows[1]], prev[rows[0]:rows[1]]
            r = label_frame(frame, prev, fuel_model, robot_model)
            stats["grey_per_frame"].append(round(r["grey_balls"], 1))
            if r["grey_balls"] > args.max_grey:
                stats["skipped_grey"] += 1
                continue
            h, w = frame.shape[:2]
            img = paint_grey(frame, r["grey"])
            name = f"{stem}_{fi:06d}"
            cv2.imwrite(str(args.out / "images" / split / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (args.out / "labels" / split / f"{name}.txt").write_text(
                "\n".join(to_yolo(c, b, w, h) for c, b, _ in r["labels"]) + "\n")
            tw, th = min(SAVE_TILE, w), min(SAVE_TILE, h)
            for x0, y0 in pick_tiles(r["labels"], w, h, args.tiles, tw, th):
                keep, cut = clip_labels(r["labels"], x0, y0, tw, th)
                tile = paint_grey(img[y0:y0 + th, x0:x0 + tw], boxes=cut)
                tname = f"{name}_t{x0}_{y0}"
                cv2.imwrite(str(args.out / "images" / split / f"{tname}.jpg"), tile,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (args.out / "labels" / split / f"{tname}.txt").write_text(
                    "\n".join(to_yolo(c, b, tw, th) for c, b, _ in keep) + "\n")
                stats["tiles"] += 1
            stats["frames"] += 1
            for c, _, src in r["labels"]:
                if c == 0:
                    stats["fuel"] += 1
                    if src in ("motion", "model+colour"):
                        stats[src] += 1
                else:
                    stats["robots"] += 1
            stats["greyed_balls"] += r["grey_balls"]
            if previews < args.preview and stats["frames"] % 25 == 1:
                cv2.imwrite(str(prev_dir / f"{name}.jpg"), draw(img, r["labels"]))
                previews += 1
        cap.release()
        stats["greyed_balls"] = round(stats["greyed_balls"], 1)
        report["videos"][stem] = dict(stats, split=split, start=float(start))
        print(f"{stem} ({split}): {stats['frames']} frames + {stats['tiles']} tiles, "
              f"{stats['fuel']} fuel ({stats['motion']} in flight added by motion, "
              f"{stats['model+colour']} low-confidence confirmed by colour), "
              f"{stats['robots']} robots; {stats['skipped_grey']} frames skipped "
              f"for too much unlabelled yellow", flush=True)
    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    (args.out / "relabel_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}/dataset.yaml; previews in {prev_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
