#!/usr/bin/env python3
"""Draw the label files onto their images, so you can see what the model sees.

Training images stay clean on purpose -- boxes live in dataset/labels/*.txt --
so there is otherwise no way to eyeball whether the annotations are any good.
This writes copies with the boxes drawn. Nothing in dataset/ is modified.

    python3 train/preview_labels.py             # 8 random frames
    python3 train/preview_labels.py -n 20 --out /tmp/check
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASET = ROOT / "dataset"

# BGR, matching the class order in dataset.yaml
COLOURS = {0: (0, 255, 255), 1: (255, 80, 0), 2: (0, 0, 255),
           3: (255, 180, 0), 4: (60, 60, 255)}
NAMES = {0: "fuel", 1: "robot_blue", 2: "robot_red", 3: "hub_blue", 4: "hub_red"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--count", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "review" / "label_preview")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    imgs = sorted((DATASET / "images" / args.split).glob("*.jpg"))
    if not imgs:
        print(f"no images in dataset/images/{args.split} -- run prepare_dataset.py")
        return 1

    random.Random(args.seed).shuffle(imgs)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.out.glob("*.jpg"):
        f.unlink()

    shown = 0
    empty = 0
    for src in imgs:
        if shown >= args.count:
            break
        lab = DATASET / "labels" / args.split / f"{src.stem}.txt"
        img = cv2.imread(str(src))
        if img is None:
            continue
        H, W = img.shape[:2]
        counts = {}
        if lab.exists() and lab.stat().st_size:
            for line in lab.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                c = int(p[0])
                cx, cy, bw, bh = (float(v) for v in p[1:])
                x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
                x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOURS.get(c, (255, 255, 255)), 1)
                counts[c] = counts.get(c, 0) + 1
        else:
            empty += 1
        summary = ", ".join(f"{NAMES.get(c, c)}={n}" for c, n in sorted(counts.items())) \
                  or "NO LABELS"
        cv2.rectangle(img, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(img, f"{src.name}   {summary}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(args.out / src.name), img)
        shown += 1

    print(f"wrote {shown} annotated copies to {args.out}")
    if empty:
        print(f"  ! {empty} of them had no labels -- run train/autolabel_fuel.py")
    print("\nThe dataset images themselves are untouched; these are copies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
