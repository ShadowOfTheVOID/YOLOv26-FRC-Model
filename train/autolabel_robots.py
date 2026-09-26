#!/usr/bin/env python3
"""Propose robot boxes with an open-vocabulary detector: no hand labels, no video.

Why not the motion heuristic in autolabel_objects.py: on the real 2026nhdur
preview it boxed three people in the stands and found one robot of six, and
after the field-line fix it still missed the robot in the trench, the one in
the corner and a clearly visible blue one, and boxed the red trench. It keys on
bumper colour AND movement, and a robot that sits still, or has a bumper the
broadcast has crushed to near-black, fails both.

YOLOE (`yoloe-26s-seg.pt`) takes the class as a text prompt, so it can be asked
for robots it has never been trained on. Measured on two real frames, each
robot hand-marked:

    2026nhdur (5 robots): 4 found; both hubs boxed as well, one box spanning
                          two robots
    second frame (5):     full frame found only 74. Three overlapping tiles at
                          2x added 7314 and the corner robot, at 0.09-0.34;
                          both red robots on the right still missed

So it is a recall problem, not a precision one. Once the tall boxes (the hubs)
and the boxes that swallow two others are removed, what is left are robots.
That decides how the output is used. A frame where it found most of the robots
is a good training frame; a frame where it found two of six would teach the
detector that the other four are floor. So frames are GATED, not labelled
partially: below --min-robots, the whole frame is moved out of the dataset to
dataset/skipped/robots/, image and label together, and `--restore` puts every
one back.

The first five real previews (nhdur, ~27 robots hand-counted) settled how
strict that can be. YOLOE found ~16 of 27 robots (~60%); no frame had all of
its robots boxed, and at --min-robots 4 the gate kept one frame of five -- the
one where the blue ladder had passed as a robot. Of the 10 robots it missed,
5 appear at conf 0.02-0.12 and 5 not at all, so no lower threshold closes the
gap. A gate that keeps nothing trains nothing, so the default is 2: frames
still carry unlabelled robots.

A robot whose alliance cannot be read is PAINTED OUT (flat grey, the colour
Ultralytics pads with) rather than dropping its frame. It was found; only its
class is unknown. Guessing a class is worse than no label, and leaving it
unboxed teaches that a robot is floor, but a grey patch is neither. On the
906 nhdur frames dropping them cost 176 frames (a fifth of the dataset) for
one robot each. Fuel boxes centred inside a painted robot are removed with
it. Painted frames' originals, image and label, are kept under
dataset/skipped/robots/original/ and `--restore` puts them back.

Lower-confidence boxes were tried as paint-out regions for the robots YOLOE
misses outright, and rejected: on the five previews they covered 3 of 10
missed robots and, on one frame, 30% of the picture.

Fuel is greyed out before YOLOE looks. A robot sitting in a pile of fuel, or
with a full hopper, stopped looking like a robot to it: in the user's preview
robot 1058 in the middle of the central pile had no box at all while fuel
boxes covered it. With every fuel-coloured pixel set to grey first, measured
at --conf 0.1 on four real frames with 16 robots marked:

    as it was:     10 of 16 found; 1058 in the pile not found at all
    fuel greyed:   14 of 16; 1058 at 0.40, a red robot 0.41 -> 0.91,
                   a robot at the bottom edge 0.15 -> 0.44
    extra prompts ("robot covered in yellow balls", "machine full of
                   balls"), no greying: no better than as it was

Two of those frames were previews with label outlines drawn in yellow, which
greying also removes, so their "as it was" is pessimistic; the two clean
frames alone went from 6 of 9 to 8 of 9. Only the proposal sees the greyed
copy: alliance is read from the real pixels, where the bumper still is.

Alliance comes from the bottom band of the box, where the bumper is: a hue
vote among pixels saturated enough to have a hue at all. Measured: 7314 blue
0.52 / red 0.00, 49/25 red 0.48 / 0.00. The fixed saturation >= 120 gate the
motion heuristic uses read robot 69's navy bumper as no colour at all; its
pixels have median saturation 8-93 and value 25-89, and no threshold separates
that from the grey floor. Such a robot has unknown alliance and is painted out.

    # look first: draws kept boxes in alliance colour, rejects in grey with why
    .venv-train/bin/python train/autolabel_robots.py --preview previews/robots
    # how many frames survive at each --min-robots, changing nothing
    .venv-train/bin/python train/autolabel_robots.py --dry-run
    # append robot boxes to the fuel labels; move rejected frames aside
    .venv-train/bin/python train/autolabel_robots.py

It appends to label files written by autolabel_fuel.py and skips any that
already have a robot box, so a second run adds nothing twice. Run
drop_offcamera.py and autolabel_fuel.py first.

What it does not fix: robots it misses in frames that pass the gate are still
unlabelled, ~40% of them on the previews. A detector trained on these labels
learns robots from the 60% and is taught, wrongly, that the rest are floor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATASET = ROOT / "dataset"
CLASSES = {"fuel": 0, "robot_blue": 1, "robot_red": 2, "hub_blue": 3, "hub_red": 4}

WEIGHTS = "yoloe-26s-seg.pt"
# Of ten prompts tried on the nhdur frame, "robotic vehicle" covered 4 of 5
# robots, "robot" 3, "wheeled robot" 3; "machine", "cart", "vehicle", "box",
# "bumper" and "metal cart with wheels" found none. Together they cover what
# each finds alone.
ROBOT_PROMPTS = ("robotic vehicle", "robot", "wheeled robot")
# Things in every frame that are not robots. Asked for by name, each takes
# its own class instead of competing for "robot"; ~50 people a frame were
# boxed as "person" on the nhdur frame. All measurements above include them.
DECOYS = ("person", "chair", "hub")

TALL = 1.3          # h/w above this is not a robot: the hubs came out 1.0-2.3
HOLDS = 0.8         # a box with two others this far inside it spans two robots
BIG = 3.0           # area over this many times the other robots' median
DUP = 0.8           # a box this far inside another robot box is the same robot
# autolabel_fuel.py's HSV_LO / HSV_HI, repeated so this imports without cv2
FUEL_LO, FUEL_HI = (18, 90, 90), (38, 255, 255)
HUE_SAT, HUE_VAL = 60, 30
BLUE_HUE = (95, 130)
RED_HUE = (10, 165)  # red is <= 10 or >= 165 on OpenCV's 0-180 hue
MIN_VOTE = 0.05      # of the band's pixels
MAX_PER_ALLIANCE = 3


# --- pure box logic (numpy only, tested in CI) ------------------------------

def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(a, b) -> float:
    """Fraction of box a that lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def merge(dets: list, thr: float = 0.5) -> list:
    """Greedy NMS over (x1, y1, x2, y2, conf) from every prompt and tile."""
    kept = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(iou(d, k) <= thr for k in kept):
            kept.append(d)
    return kept


def screen(dets: list, field_top: int = 0) -> list:
    """(det, reason) for each merged box; reason None means it is a robot.

    NMS alone kept a box around both red robots on the nhdur frame: its IoU
    with each of them was 0.40 and 0.29, under any sane threshold, because it
    is so much larger than either. What gives it away is that both sit inside
    it.

    Size is judged against the frame's other boxes, not in pixels, because a
    robot's size depends on the camera. The red alliance wall and ladder came
    out as a 0.14 "robot" 265 x 285 px, 9.7x the median of the four real
    robots in the same frame (~7,800 px^2 each).

    The median is over boxes that are not already rejected. It first included
    the two hubs (tall, ~56,000 and ~82,000 px^2), which lifted it far enough
    that the blue ladder -- 34,800 px^2 against real robots of 6,100-9,500 --
    came out 3.7x and passed as a robot, in the one frame of the first five
    real previews that the gate kept. Against the robots alone it is 4.4x.
    Real robots in those frames differ by at most ~1.7x near to far, so the
    limit is 3x, not 4x. One other robot is enough to compare with: a frame
    with one robot found let the top of the red hub (182 x 215 px, 3.4x the
    blue robot beside it) through as a robot of unknown alliance, and it was
    painted grey.

    With fuel greyed out YOLOE also returns parts of robots: a 75 x 48 box
    wholly inside robot 11136's own. Of two robot boxes where one lies inside
    the other, the less confident is a duplicate, not a second robot.
    """
    first = []
    for d in dets:
        w, h = d[2] - d[0], d[3] - d[1]
        if h > TALL * w:
            why = "tall"
        elif sum(inside(o, d) >= HOLDS for o in dets if o is not d) >= 2:
            why = "spans two"
        elif field_top and d[3] < field_top:
            why = "above field"
        else:
            why = None
        first.append((d, why))
    out = []
    for d, why in first:
        if why is None:
            others = [(o[2] - o[0]) * (o[3] - o[1]) for o, w in first
                      if w is None and o is not d]
            area = (d[2] - d[0]) * (d[3] - d[1])
            if others and area > BIG * float(np.median(others)):
                why = "too big"
        out.append((d, why))
    alive = [d for d, why in out if why is None]
    for i, (d, why) in enumerate(out):
        if why is None and any(
                o is not d and o[4] > d[4]
                and (inside(d, o) >= DUP or inside(o, d) >= DUP) for o in alive):
            out[i] = (d, "duplicate")
    return out


def fuel_mask(hsv: np.ndarray) -> np.ndarray:
    """True where a pixel is fuel-coloured, by autolabel_fuel.py's gate."""
    lo, hi = np.array(FUEL_LO), np.array(FUEL_HI)
    return ((hsv >= lo) & (hsv <= hi)).all(axis=-1)


def alliance(hsv_band: np.ndarray):
    """(class or None, blue share, red share) from the bumper band's pixels."""
    px = hsv_band.reshape(-1, 3).astype(int)
    if not len(px):
        return None, 0.0, 0.0
    hued = (px[:, 1] >= HUE_SAT) & (px[:, 2] >= HUE_VAL)
    blue = float((hued & (px[:, 0] >= BLUE_HUE[0]) & (px[:, 0] <= BLUE_HUE[1])).mean())
    red = float((hued & ((px[:, 0] <= RED_HUE[0]) | (px[:, 0] >= RED_HUE[1]))).mean())
    # A blue robot beside the red ramp picks up some red and the other way
    # round; the bumper has to win clearly, not by a few pixels.
    if blue >= MIN_VOTE and blue >= 2 * red:
        return "robot_blue", blue, red
    if red >= MIN_VOTE and red >= 2 * blue:
        return "robot_red", blue, red
    return None, blue, red


def verdict(labelled: list, unknown: int, min_robots: int):
    """Why this frame cannot be a training frame, or None if it can.

    Robots of unknown alliance count towards --min-robots: they are painted
    out, so they are not left in the frame as floor.
    """
    for cls in ("robot_blue", "robot_red"):
        n = sum(c == cls for c, _ in labelled)
        if n > MAX_PER_ALLIANCE:
            return f"{n} {cls} -- one of them is not a robot"
    if len(labelled) + unknown < min_robots:
        return (f"{len(labelled) + unknown} robot(s) found, under --min-robots "
                f"{min_robots}; the rest would be labelled floor")
    return None


def yolo_lines(labelled: list, W: int, H: int) -> list:
    return [f"{CLASSES[c]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
            f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"
            for c, (x1, y1, x2, y2, _) in labelled]


PAINT = 114   # Ultralytics' letterbox grey


def paint(img: np.ndarray, boxes: list, keep: list) -> np.ndarray:
    """Grey out `boxes`, then put back every pixel inside the `keep` boxes.

    A robot of unknown alliance often overlaps a labelled one; painting the
    overlap would leave a labelled robot with a grey hole in it.
    """
    out = img.copy()
    H, W = img.shape[:2]

    def px(d):
        return (max(int(d[0]), 0), max(int(d[1]), 0),
                min(int(round(d[2])), W), min(int(round(d[3])), H))
    for d in boxes:
        x1, y1, x2, y2 = px(d)
        out[y1:y2, x1:x2] = PAINT
    for d in keep:
        x1, y1, x2, y2 = px(d)
        out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return out


def drop_covered(label_text: str, boxes: list, W: int, H: int) -> str:
    """Label lines whose box centre is not inside any painted box."""
    kept = []
    for line in label_text.splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        cx, cy = float(f[1]) * W, float(f[2]) * H
        if not any(d[0] <= cx <= d[2] and d[1] <= cy <= d[3] for d in boxes):
            kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")


def strip_robots(label_text: str) -> str:
    kept = [line for line in label_text.splitlines()
            if line.split()[:1] not in (["1"], ["2"])]
    return "\n".join(kept) + ("\n" if kept else "")


def has_robots(label_text: str) -> bool:
    return any(line.split()[:1] in (["1"], ["2"]) for line in label_text.splitlines())


# --- model and images --------------------------------------------------------

def load(weights: str):
    from ultralytics import YOLOE
    model = YOLOE(weights)
    names = list(ROBOT_PROMPTS + DECOYS)
    model.set_classes(names, model.get_text_pe(names))
    return model


def propose(model, img, conf: float, imgsz: int = 1280, tiles: bool = True) -> list:
    """Robot boxes from the whole frame plus three half-width tiles at 2x.

    Robots in a wide broadcast are ~100 px across. On the second test frame
    the whole-frame pass found one of five; the tiles found three.
    """
    import cv2
    H, W = img.shape[:2]
    views = [(img, 0, 1.0)]
    if tiles:
        half = W // 2
        for x0 in (0, W // 4, W - half):
            views.append((cv2.resize(img[:, x0:x0 + half], None, fx=2, fy=2), x0, 2.0))
    dets = []
    for view, x0, scale in views:
        r = model.predict(view, conf=conf, imgsz=imgsz, verbose=False)[0]
        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                           r.boxes.cls.tolist()):
            if int(k) < len(ROBOT_PROMPTS):
                x1, y1, x2, y2 = (v / scale for v in b)
                dets.append((x1 + x0, y1, x2 + x0, y2, float(c)))
    return merge(dets)


def grey_fuel(img, hsv):
    """A copy with every fuel-coloured pixel (grown by one) set to grey."""
    import cv2
    mask = cv2.dilate(fuel_mask(hsv).astype(np.uint8), np.ones((3, 3), np.uint8))
    out = img.copy()
    out[mask > 0] = PAINT
    return out


def label_frame(model, img, args):
    """(labelled, rejected, unknown boxes, field_top, reason) for one image."""
    import cv2
    field_top = 0
    if args.field_line:
        from autolabel_objects import field_line
        field_top = field_line(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    seen = grey_fuel(img, hsv) if args.grey_fuel else img
    labelled, rejected, unknown = [], [], []
    for d, why in screen(propose(model, seen, args.conf, args.imgsz, args.tiles), field_top):
        if why:
            rejected.append((d, why))
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in d[:4])
        band = hsv[max(y1, int(y2 - 0.4 * (y2 - y1))):y2, max(x1, 0):x2]
        cls, _, _ = alliance(band)
        if cls is None:
            unknown.append(d)
        else:
            labelled.append((cls, d))
    return (labelled, rejected, unknown, field_top,
            verdict(labelled, len(unknown), args.min_robots))


def draw(img, labelled, rejected, unknown, field_top, reason):
    import cv2
    out = paint(img, unknown, [d for _, d in labelled])
    if field_top:
        cv2.line(out, (0, field_top), (out.shape[1], field_top), (255, 255, 255), 1)
    for (d, why) in rejected:
        p1, p2 = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
        cv2.rectangle(out, p1, p2, (140, 140, 140), 1)
        cv2.putText(out, f"{why} {d[4]:.2f}", (p1[0], max(p1[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
    for cls, d in labelled:
        col = (255, 0, 0) if cls == "robot_blue" else (0, 0, 255)
        p1, p2 = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
        cv2.rectangle(out, p1, p2, col, 2)
        cv2.putText(out, f"{d[4]:.2f}", (p1[0], max(p1[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.putText(out, reason or "KEEP", (10, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if reason else (0, 255, 0), 2)
    return out


def restore(skipped: Path) -> int:
    """Undo every run: frames moved aside, frames painted, robot lines added.

    Robot lines (classes 1 and 2) are stripped from every label file, so this
    also removes robot boxes written by autolabel_objects.py. Fuel and hub
    lines are untouched.
    """
    moved = painted = stripped = 0
    for split in ("train", "val"):
        for img in sorted((skipped / split).glob("*.jpg")):
            img.rename(DATASET / "images" / split / img.name)
            lab = skipped / split / f"{img.stem}.txt"
            if lab.exists():
                lab.rename(DATASET / "labels" / split / lab.name)
            moved += 1
        orig = skipped / "original" / split
        for img in sorted(orig.glob("*.jpg")):
            img.replace(DATASET / "images" / split / img.name)
            lab = orig / f"{img.stem}.txt"
            if lab.exists():
                lab.replace(DATASET / "labels" / split / lab.name)
            painted += 1
        for lab in (DATASET / "labels" / split).glob("*.txt"):
            text = lab.read_text()
            if has_robots(text):
                lab.write_text(strip_robots(text))
                stripped += 1
    print(f"restored {moved} moved frames and {painted} painted frames; "
          f"removed robot boxes from {stripped} label files")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=WEIGHTS,
                    help="YOLOE weights; downloaded from the Ultralytics GitHub "
                         "release on first use (~30 MB, plus ~240 MB for the "
                         "text encoder)")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="lower finds more robots and more of everything else. "
                         "Real robots scored 0.07-0.77 in the test frames")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--no-tiles", dest="tiles", action="store_false",
                    help="whole frame only: ~4x faster, finds fewer robots")
    ap.add_argument("--no-field-line", dest="field_line", action="store_false")
    ap.add_argument("--no-grey-fuel", dest="grey_fuel", action="store_false",
                    help="let YOLOE see the fuel. It then missed robots "
                         "sitting in piles of it (10 of 16 found, against 14)")
    ap.add_argument("--min-robots", type=int, default=2,
                    help="a frame with fewer is moved out. 4 kept 1 of the "
                         "first 5 real previews (and that one was wrong); "
                         "see --dry-run for how many survive each value")
    ap.add_argument("--preview", type=Path, metavar="DIR",
                    help="write annotated frames here and change nothing")
    ap.add_argument("--images", nargs="*", type=Path,
                    help="with --preview: these images instead of a dataset sample")
    ap.add_argument("--sample", type=int, default=12,
                    help="with --preview: how many dataset frames, spread evenly")
    ap.add_argument("--dry-run", action="store_true",
                    help="label nothing, move nothing; report what would happen")
    ap.add_argument("--skipped", type=Path, default=DATASET / "skipped" / "robots")
    ap.add_argument("--restore", action="store_true",
                    help="undo every run: frames moved aside or painted go back, "
                         "and robot boxes are removed from every label file")
    args = ap.parse_args()

    if args.restore:
        return restore(args.skipped)

    import cv2
    images = args.images or sorted(p for split in ("train", "val")
                                   for p in (DATASET / "images" / split).glob("*.jpg"))
    if not images:
        print("no images under dataset/images -- run prepare_dataset.py first")
        return 1
    if args.preview and not args.images:
        step = max(len(images) // args.sample, 1)
        images = images[::step][:args.sample]

    model = load(args.weights)
    counts, reasons = {}, {}
    kept = moved = boxes = already = painted = 0
    for i, src in enumerate(images):
        img = cv2.imread(str(src))
        if img is None:
            continue
        H, W = img.shape[:2]
        lab = DATASET / "labels" / src.parent.name / f"{src.stem}.txt"
        existing = lab.read_text() if lab.exists() else ""
        # A painted frame with no readable robot gets no robot line, so the
        # saved original is the other sign it was done. Without this check a
        # second run would save the painted copy over the real original.
        done_before = (args.skipped / "original" / src.parent.name / src.name).exists()
        if not args.preview and (has_robots(existing) or done_before):
            already += 1
            continue
        labelled, rejected, unknown, field_top, reason = label_frame(model, img, args)
        n = len(labelled) + len(unknown)
        counts[n] = counts.get(n, 0) + 1
        if reason:
            key = ("too many of one alliance" if "not a robot" in reason else
                   "too few robots")
            reasons[key] = reasons.get(key, 0) + 1
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.preview / f"{src.stem}.jpg"),
                        draw(img, labelled, rejected, unknown, field_top, reason))
            print(f"{src.name}: {len(labelled)} robots, {len(unknown)} painted out -- "
                  f"{reason or 'keep'}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} frames")
        if args.dry_run:
            kept += reason is None
            moved += reason is not None
            boxes += len(labelled) if reason is None else 0
            painted += bool(unknown) and reason is None
            continue
        if reason:
            dest = args.skipped / src.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            src.rename(dest / src.name)
            if lab.exists():
                lab.rename(dest / lab.name)
            moved += 1
            continue
        if unknown:
            # Keep the original, then write the painted copy as a NEW file.
            # Renaming first matters: prepare_dataset.py without --copy makes
            # these symlinks into data/frames/, and writing through one would
            # paint the harvested frame itself.
            orig = args.skipped / "original" / src.parent.name
            orig.mkdir(parents=True, exist_ok=True)
            src.rename(orig / src.name)
            if lab.exists():
                (orig / lab.name).write_text(existing)
            cv2.imwrite(str(src), paint(img, unknown, [d for _, d in labelled]),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            existing = drop_covered(existing, unknown, W, H)
            painted += 1
        if existing and not existing.endswith("\n"):
            existing += "\n"
        lines = yolo_lines(labelled, W, H)
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text(existing + "\n".join(lines) + "\n")
        kept += 1
        boxes += len(lines)

    if args.preview:
        print(f"\npreviews -> {args.preview}")
        return 0
    print("\nrobots per frame (including unknown alliance):")
    for n in sorted(counts):
        print(f"  {n:2d}: {counts[n]} frames")
    for m in range(2, 7):
        print(f"  --min-robots {m} would keep at most "
              f"{sum(v for k, v in counts.items() if k >= m)} frames")
    for why, n in sorted(reasons.items(), key=lambda r: -r[1]):
        print(f"  rejected, {why}: {n}")
    if already:
        print(f"{already} frames already had robot boxes and were left alone")
    verb = "would be" if args.dry_run else "were"
    print(f"\n{kept} frames {verb} labelled ({boxes} robot boxes; {painted} with "
          f"a robot of unknown alliance painted out), {moved} {verb} moved to "
          f"{args.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
