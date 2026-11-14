#!/usr/bin/env python3
"""Label fuel and robots with SAM 3 text prompts, into a NEW dataset beside the old one.

Why. autolabel_fuel.py is a colour heuristic scaled off a frame's isolated
balls. It fails on the frames that matter most: fuel piled in a corral (6-22
isolated balls against 85), balls high against the crowd (its field line
left them unlabelled, so the released models miss them), balls in hoppers.
Every one of those failures has been a threshold, and more thresholds is what
CLAUDE.md says not to add. SAM 3 (Meta, Nov 2025) finds every instance of a
text phrase, so "yellow ball" in a pile should come back as many balls rather
than one blob or none. Whether it does on broadcast frames is NOT measured
yet: run --compare first and look.

What it keeps from autolabel_robots.py, because those rules were measured on
real frames and the proposer is the only thing changing: screen() (tall
boxes are hubs, a box holding two others spans two robots, 3x the median is
not a robot, a box inside another is a duplicate), alliance() from the bumper
band's hue, painting out a robot whose alliance cannot be read, and the
--min-robots gate.

Tiles, not the whole frame. SAM 3 squashes every input to 1008x1008. Our
frames are 1920x504 broadcast strips: whole, they would be squashed to half
width and stretched to double height, and ~17x12 px fuel would reach the
model at ~9 px. Fuel runs on overlapping 480 px squares (each enlarged ~2x),
robots on squares the height of the frame. A box touching a tile edge that
is not the image edge is cut off by that tile; the overlap means a
neighbouring tile sees it whole, so the cut copy is dropped.

Frames are gated, not partly labelled, as everywhere else in train/:
  * a fuel box over --clump times the frame's median fuel box is a pile SAM 3
    returned as one object. Unboxed balls teach fuel = background, so the
    frame is left out (listed in skipped.txt);
  * robots under --min-robots, or more than three per alliance: left out.

Hubs are not prompted. Hub labels came from geometry recorded by hand
(`run.py db hub`, autolabel_objects.py), which beats any guess, so existing
hub lines are carried over from the source labels unchanged.

The source dataset is never modified. The output is a dataset of its own,
images symlinked (painted copies where a robot was painted out), ready for
train.py / train_rfdetr.py:

    # look first: old labels left, SAM 3 right, for 12 random frames
    python train/autolabel_sam3.py --data dataset-scout/dataset.yaml --compare 12
    # then label everything into dataset-sam3/
    python train/autolabel_sam3.py --data dataset-scout/dataset.yaml --out dataset-sam3

Needs transformers >= 5 and access to the gated facebook/sam3 weights
(request it on Hugging Face; set HF_TOKEN). RF-DETR needs transformers < 5:
run this in its own step and reinstall afterwards (the Colab notebook does).
SAM 3 is Meta's SAM License, not Apache: fine for producing training labels;
the models that ship are YOLO26 / RF-DETR, not SAM 3.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autolabel_robots import (MAX_PER_ALLIANCE, alliance, drop_covered, merge,  # noqa: E402
                              paint, screen)
from subset_classes import yaml_names  # noqa: E402

MODEL = "facebook/sam3"
FUEL_PROMPT = "yellow ball"
ROBOT_PROMPT = "robot"
FUEL_TILE = 480        # px of source per fuel tile; SAM 3 enlarges it to 1008
ROBOT_TILE = 1008      # robots: tiles as tall as the frame, up to SAM 3's input
OVERLAP = 0.5          # of a tile; must exceed the largest object it cuts
EDGE = 2               # px: a box this close to an inner tile edge is cut off
CLUMP = 4.0            # x median fuel box area = a pile returned as one object
MIN_FOR_MEDIAN = 4     # fewer fuel boxes than this: no clump test (no median)
HUB_CLASSES = ("hub_blue", "hub_red")
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

Box = Tuple[float, float, float, float, float]   # x1, y1, x2, y2, score


# --- pure logic (numpy only, tested in CI) ----------------------------------

def tiles(W: int, H: int, side: int, overlap: float = OVERLAP) -> List[Tuple[int, int, int, int]]:
    """Square tiles of `side` (at most the frame's short side) covering the
    frame, neighbours overlapping by `overlap` of a tile. Last row/column is
    pushed back to end exactly on the image edge, so nothing is left over."""
    side = max(1, min(side, W, H))
    step = max(1, int(side * (1 - overlap)))

    def starts(n):
        s = list(range(0, max(n - side, 0) + 1, step))
        if s[-1] + side < n:
            s.append(n - side)
        return s
    return [(x, y, x + side, y + side) for y in starts(H) for x in starts(W)]


def interior(box: Sequence[float], tile: Sequence[int], W: int, H: int, edge: int = EDGE) -> bool:
    """False when the box touches a tile edge that is not also the image edge:
    that tile saw only part of the object, and an overlapping tile saw it all."""
    x1, y1, x2, y2 = box[:4]
    tx1, ty1, tx2, ty2 = tile
    return not ((tx1 > 0 and x1 <= tx1 + edge) or (ty1 > 0 and y1 <= ty1 + edge)
                or (tx2 < W and x2 >= tx2 - edge) or (ty2 < H and y2 >= ty2 - edge))


def clumps(fuel: Sequence[Box], factor: float = CLUMP) -> List[int]:
    """Indices of fuel boxes over `factor` x the median fuel box area."""
    if len(fuel) < MIN_FOR_MEDIAN:
        return []
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in fuel]
    med = float(np.median(areas))
    return [i for i, a in enumerate(areas) if a > factor * med]


def yolo_line(cls_index: int, b: Sequence[float], W: int, H: int) -> str:
    x1, y1, x2, y2 = (min(max(v, 0.0), lim) for v, lim in zip(b[:4], (W, H, W, H)))
    return (f"{cls_index} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")


def hub_lines(label_text: str, names: Sequence[str]) -> List[str]:
    """The source label lines that are hubs, carried over as they are."""
    hub_ids = {str(i) for i, n in enumerate(names) if n in HUB_CLASSES}
    return [ln for ln in label_text.splitlines() if ln.split()[:1] and ln.split()[0] in hub_ids]


def assemble(other_lines: List[str], robot_lines: List[str], unknown: list,
             W: int, H: int) -> List[str]:
    """A frame's label lines once unknown-alliance robots are painted out.

    Fuel and hubs centred in a painted robot go with it. Labelled robots do
    NOT: paint() keeps their pixels, so dropping their lines would leave a
    visible robot unboxed, which teaches robot = background. (Found on a
    random-weight SAM 3 run: overlapping unknown boxes emptied a frame.)
    """
    if unknown:
        other_lines = drop_covered("\n".join(other_lines), unknown, W, H).splitlines()
    return list(other_lines) + list(robot_lines)


def frame_verdict(n_clumps: int, labelled: list, unknown: int, min_robots: int,
                  robots_wanted: bool) -> str:
    """Why this frame is left out, or '' if it is a training frame."""
    if n_clumps:
        return f"{n_clumps} fuel box(es) look like piles -- their balls would be unlabelled"
    if robots_wanted:
        for cls in ("robot_blue", "robot_red"):
            n = sum(c == cls for c, _ in labelled)
            if n > MAX_PER_ALLIANCE:
                return f"{n} {cls} -- one of them is not a robot"
        if len(labelled) + unknown < min_robots:
            return f"{len(labelled) + unknown} robot(s) found, under --min-robots {min_robots}"
    return ""


# --- SAM 3 -------------------------------------------------------------------

class Sam3:
    """One model, prompts encoded once, vision features once per tile."""

    def __init__(self, model_id: str = MODEL):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = (torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float32)
        token = os.environ.get("HF_TOKEN")
        self.processor = Sam3Processor.from_pretrained(model_id, token=token)
        self.model = Sam3Model.from_pretrained(model_id, token=token, dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self._text: Dict[str, dict] = {}

    def _prompt(self, text: str) -> dict:
        if text not in self._text:
            t = self.processor(text=text, return_tensors="pt")
            self._text[text] = {k: v.to(self.device) for k, v in t.items()}
        return self._text[text]

    def find(self, rgb: np.ndarray, prompts: Sequence[str], threshold: float) -> Dict[str, List[Box]]:
        """Boxes per prompt in this image's pixels."""
        from PIL import Image
        torch = self.torch
        img = Image.fromarray(rgb)
        with torch.inference_mode():
            px = self.processor(images=img, return_tensors="pt")["pixel_values"]
            vision = self.model.get_vision_features(pixel_values=px.to(self.device, self.dtype))
            out = {}
            for p in prompts:
                o = self.model(vision_embeds=vision, **self._prompt(p))
                r = self.processor.post_process_object_detection(
                    o, threshold=threshold, target_sizes=[(rgb.shape[0], rgb.shape[1])])[0]
                out[p] = [(*map(float, b), float(s))
                          for b, s in zip(r["boxes"].float().cpu().tolist(), r["scores"].float().cpu().tolist())]
        return out

    def tiled(self, rgb: np.ndarray, prompt: str, side: int, threshold: float) -> List[Box]:
        H, W = rgb.shape[:2]
        found = []
        for t in tiles(W, H, side):
            x1, y1, x2, y2 = t
            for b in self.find(rgb[y1:y2, x1:x2], [prompt], threshold)[prompt]:
                g = (b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1, b[4])
                if interior(g, t, W, H):
                    found.append(g)
        return merge(found)


def label_image(sam: Sam3, bgr: np.ndarray, src_label: str, names: Sequence[str], args):
    """(label lines, painted image or None, verdict, drawable boxes) for one frame."""
    import cv2
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    idx = {n: i for i, n in enumerate(names)}
    lines, draw = [], []
    fuel = sam.tiled(rgb, args.fuel_prompt, args.fuel_tile, args.conf) if "fuel" in idx else []
    bad = set(clumps(fuel, args.clump))
    for i, b in enumerate(fuel):
        if i not in bad:
            lines.append(yolo_line(idx["fuel"], b, W, H))
        draw.append(("pile" if i in bad else "fuel", b))
    robots_wanted = "robot_blue" in idx and "robot_red" in idx
    labelled, unknown, robot_lines = [], [], []
    if robots_wanted:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        for d, why in screen(sam.tiled(rgb, args.robot_prompt, args.robot_tile, args.conf)):
            if why:
                draw.append((f"x {why}", d))
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in d[:4])
            cls, _, _ = alliance(hsv[max(y1, int(y2 - 0.4 * (y2 - y1))):y2, max(x1, 0):x2])
            if cls is None:
                unknown.append(d)
                draw.append(("unknown", d))
            else:
                labelled.append((cls, d))
                robot_lines.append(yolo_line(idx[cls], d, W, H))
                draw.append((cls, d))
    lines += hub_lines(src_label, names)
    painted = paint(bgr, unknown, [d for _, d in labelled]) if unknown else None
    lines = assemble(lines, robot_lines, unknown, W, H)
    verdict = frame_verdict(len(bad), labelled, len(unknown), args.min_robots, robots_wanted)
    return lines, painted, verdict, draw


def make_model(model_id: str):
    """The labeller; a seam so tests can run everything below without SAM 3."""
    return Sam3(model_id)


# --- previews and dataset ----------------------------------------------------

COLOURS = {"fuel": (0, 255, 255), "robot_blue": (255, 80, 0), "robot_red": (0, 0, 255),
           "hub_blue": (255, 180, 0), "hub_red": (60, 60, 255), "pile": (255, 0, 255),
           "unknown": (128, 128, 128)}


def _draw(img, boxes, title):
    import cv2
    out = img.copy()
    for name, b in boxes:
        c = COLOURS.get(name, (160, 160, 160))
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 1 if name == "fuel" else 2)
        if name.startswith("x ") or name in ("pile", "unknown"):
            cv2.putText(out, name, (int(b[0]), max(10, int(b[1]) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def boxes_from_label(text: str, names: Sequence[str], W: int, H: int):
    out = []
    for ln in text.splitlines():
        f = ln.split()
        if len(f) >= 5:
            c, cx, cy, w, h = int(f[0]), *map(float, f[1:5])
            out.append((names[c] if c < len(names) else str(c),
                        ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)))
    return out


def count(boxes) -> str:
    n = {}
    for name, _ in boxes:
        n[name] = n.get(name, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(n.items())) or "nothing"


def images_of(root: Path, split: str) -> List[Path]:
    d = root / "images" / split
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT) if d.is_dir() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="source dataset.yaml (never modified)")
    ap.add_argument("--out", type=Path, help="new dataset directory to write")
    ap.add_argument("--compare", type=int, default=0,
                    help="instead: draw old vs SAM 3 labels for N random frames into --preview")
    ap.add_argument("--preview", type=Path, default=Path("sam3_compare"))
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fuel-prompt", default=FUEL_PROMPT)
    ap.add_argument("--robot-prompt", default=ROBOT_PROMPT)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--fuel-tile", type=int, default=FUEL_TILE)
    ap.add_argument("--robot-tile", type=int, default=ROBOT_TILE)
    ap.add_argument("--clump", type=float, default=CLUMP)
    ap.add_argument("--min-robots", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N images per split (testing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.out and not args.compare:
        ap.error("give --out DIR to build a dataset, or --compare N to look first")

    import cv2
    src = args.data.resolve().parent
    names = yaml_names(args.data.read_text())
    if "fuel" not in names and "robot_blue" not in names:
        raise SystemExit(f"{args.data}: no fuel or robot classes in {names}")
    sam = make_model(args.model)
    print(f"SAM 3 on {sam.device}; classes {names}; prompts {args.fuel_prompt!r} / {args.robot_prompt!r}")

    if args.compare:
        pool = [p for s in args.splits.split(",") for p in images_of(src, s)]
        random.Random(args.seed).shuffle(pool)
        args.preview.mkdir(parents=True, exist_ok=True)
        for p in pool[:args.compare]:
            bgr = cv2.imread(str(p))
            H, W = bgr.shape[:2]
            lab = src / "labels" / p.parent.name / (p.stem + ".txt")
            old_text = lab.read_text() if lab.exists() else ""
            lines, painted, verdict, drawn = label_image(sam, bgr, old_text, names, args)
            old = boxes_from_label(old_text, names, W, H)
            new_kept = boxes_from_label("\n".join(lines), names, W, H)
            left = _draw(bgr, old, f"current labels: {count(old)}")
            right = _draw(painted if painted is not None else bgr, drawn,
                          f"SAM 3: {count(new_kept)}" + (f"  LEFT OUT: {verdict}" if verdict else ""))
            cv2.imwrite(str(args.preview / f"{p.stem}.jpg"), np.vstack([left, right]))
            print(f"{p.name}: current {count(old)} | SAM 3 {count(new_kept)}"
                  + (f" | left out: {verdict}" if verdict else ""), flush=True)
        print(f"previews in {args.preview}/ -- top: current labels, bottom: SAM 3 "
              "(magenta = pile, grey = unknown alliance, painted out)")
        return 0

    out = args.out.resolve()
    if out == src:
        raise SystemExit("--out must be a new directory, not the source dataset")
    skipped, kept = [], 0
    for split in args.splits.split(","):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        imgs = images_of(src, split)
        if args.limit:
            imgs = imgs[:args.limit]
        for n, p in enumerate(imgs, 1):
            dst_img = out / "images" / split / p.name
            dst_lab = out / "labels" / split / (p.stem + ".txt")
            if dst_lab.exists() and (dst_img.exists() or dst_img.is_symlink()):
                kept += 1
                continue                       # resumable: a cut-off run carries on
            lab = src / "labels" / split / (p.stem + ".txt")
            bgr = cv2.imread(str(p))
            lines, painted, verdict, _ = label_image(
                sam, bgr, lab.read_text() if lab.exists() else "", names, args)
            if verdict:
                skipped.append(f"{split}/{p.name}\t{verdict}")
                continue
            if painted is not None:
                cv2.imwrite(str(dst_img), painted)
            else:
                if dst_img.is_symlink() or dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(p.resolve())
            dst_lab.write_text("\n".join(lines) + ("\n" if lines else ""))
            kept += 1
            if n % 50 == 0:
                print(f"  {split}: {n}/{len(imgs)}", flush=True)
    yaml = "path: .\ntrain: images/train\nval: images/val\nnames:\n" + "".join(
        f"  {i}: {n}\n" for i, n in enumerate(names))
    (out / "dataset.yaml").write_text(yaml)
    with open(out / "skipped.txt", "a") as fh:
        fh.write("".join(s + "\n" for s in skipped))
    print(f"{kept} frames labelled into {out}; {len(skipped)} left out (reasons in skipped.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
