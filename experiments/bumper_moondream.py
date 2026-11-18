#!/usr/bin/env python3
"""Read team numbers off robot bumpers with Moondream, snapped to the match's teams.

Experimental; not wired into run.py. Measured on public photos only -- no
broadcast frames yet (YouTube refuses downloads from the dev container).

Why. Robots hidden behind people or hubs come back under new track numbers
(on qm7, 1058 was #5, #8 and #11), and `--teams` folds them by hand. Merging
tracks by position was reasoned through and would have given 1058 the number
of 611. A bumper number is the robot's own ID; if it can be read, tracks join
by what the robot says it is, not by where it was.

How small can it be? Measured 2026-11-14 with moondream2 (Apache-2.0, revision
pinned below) on CPU, robots cropped from one full-res Commons photo of a 2024
Crescendo field, downscaled so the numerals are H px tall, then re-JPEG'd at
q70 like a stream frame (`--scale-test` reproduces it):

    numeral H px     35      20      14      10      7
    red 360          360     360     360     360     360
    blue 3268        32 68   32 68   32 69   32      "no robot"
    red _31 hidden   131     131     34      "none"  unreadable

  * Three digits on a red bumper survived to 7 px; four digits on blue broke
    at 14 px. FRC numerals are >= 4 in tall, which on a 1080p wide broadcast
    shot is roughly 10 px -- right at the edge. Close shots will read; wide
    shots mostly will not.
  * The failure that matters is not "unreadable", it is a plausible wrong
    number: "32 69", and "131" read off a bumper whose first digit is hidden
    behind a truss. So answers are never trusted as read: `snap()` accepts
    only one of the three teams on that bumper colour's alliance (from TBA's
    match record), exactly or one digit off, and `vote()` wants agreement
    across several frames of the same track.
  * One digit off still lets a hidden-digit misread through when the alliance
    happens to hold a near number ("131" vs 1318). The vote is the guard;
    one frame is not enough evidence.
  * ~2.5 s per question on a 4-core CPU: fine offline, one read per track per
    second or so, never per frame and never live.

The robot boxes come from any detector: RF-DETR weights from
train/train_rfdetr.py (.pth) or the released YOLO weights (.pt).

    python experiments/bumper_moondream.py --scale-test photo.jpg \
        --box 2130,716,2407,926=360 --box 3331,768,3601,933=
    python experiments/bumper_moondream.py --video qm7.mp4 --weights best.pt \
        --red 1058,611,8013 --blue 5687,7407,1153 --every 1.0 --out reads.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

MODEL = "vikhyatk/moondream2"
REVISION = "5d6c926f44e26b07957b0dd315bbedcb4c17a5fe"  # Apache-2.0; pinned: remote code runs
# Moondream 3.1 (9B MoE, 2B active; Moondream Model License 1.0 -- products
# built on it are allowed, offering general-purpose Moondream as a hosted
# service is not). It runs only through Moondream's Photon engine (`pip install
# moondream`): NVIDIA Ampere or newer, or Apple Silicon -- not a Colab T4, not
# AMD, not a CPU. Untested on bumpers until someone runs --scale-test with it.
MODEL_31 = "moondream3.1-9B-A2B"
QUESTION = ("What is the team number printed on this robot's bumper? Reply with only "
            "the digits. If you cannot read it, reply unreadable.")


def digits(answer: str) -> str:
    """The digits in an answer: "32 68" -> "3268", "unreadable" -> ""."""
    return re.sub(r"\D", "", answer or "")


def _distance(a: str, b: str) -> int:
    """Levenshtein distance; team numbers are at most 5 digits."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap(answer: str, teams: Sequence[str]) -> Optional[str]:
    """The alliance team an answer names, or None.

    Exact match wins. Otherwise one edit away ("32 69" -> 3268) is accepted
    for reads of three or more digits, and only when exactly one team is that
    close -- two candidates at distance 1 means the read cannot tell them
    apart. Short reads ("32", "34") are never snapped: too many teams are one
    edit from two digits.
    """
    got = digits(answer)
    teams = [str(t) for t in teams]
    if not got:
        return None
    if got in teams:
        return got
    if len(got) < 3:
        return None
    close = [t for t in teams if _distance(got, t) == 1]
    return close[0] if len(close) == 1 else None


def vote(snapped: Iterable[Optional[str]], min_votes: int = 2,
         min_share: float = 0.6) -> Optional[str]:
    """One team for a track from its per-frame snaps, or None.

    Needs at least `min_votes` reads naming the winner and the winner to hold
    `min_share` of the reads that named anyone; unreadable frames abstain.
    """
    named = [s for s in snapped if s]
    if not named:
        return None
    team, n = Counter(named).most_common(1)[0]
    return team if n >= min_votes and n / len(named) >= min_share else None


def load_model(version: str = "2"):
    """Moondream "2" (transformers, runs anywhere) or "3.1" (Photon)."""
    if version == "3.1":
        import moondream
        return moondream.vl(local=True, model=MODEL_31)
    import torch
    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    return AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True,
        dtype=torch.float16 if dev != "cpu" else torch.float32,
        device_map={"": dev})


def ask(model, crop) -> str:
    return model.query(crop, QUESTION)["answer"].strip()


def scale_test(image: Path, boxes: List[str], heights: Sequence[int], ref: int,
               version: str = "2") -> None:
    """Re-run the size table above on any photo: each --box x0,y0,x1,y1=TRUTH
    (empty truth = honestly unreadable) is shrunk so its numerals are H px."""
    from PIL import Image
    src = Image.open(image).convert("RGB")
    model = load_model(version)
    print(f"moondream {version}")
    print(f"{'truth':>8} {'H px':>4}  {'answer':30} verdict  seconds")
    for spec in boxes:
        coords, _, truth = spec.partition("=")
        box = tuple(int(v) for v in coords.split(","))
        for h in heights:
            c = src.crop(box)
            f = h / ref
            c = c.resize((max(8, round(c.width * f)), max(8, round(c.height * f))), Image.LANCZOS)
            buf = io.BytesIO()
            c.save(buf, "JPEG", quality=70)
            c = Image.open(buf).convert("RGB")
            t0 = time.time()
            a = ask(model, c)
            ok = digits(a) == truth if truth else not digits(a)
            print(f"{truth or '-':>8} {h:>4}  {a[:30]!r:30} {'OK   ' if ok else 'WRONG'}  "
                  f"{time.time() - t0:.1f}", flush=True)


def detector(weights: Path):
    """frame (RGB numpy) -> [(x0, y0, x1, y1, class_name)] for robots."""
    if weights.suffix == ".pth":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
        from train_rfdetr import detection_names, load_rfdetr   # see the class-id traps there
        m, names = load_rfdetr(weights)

        def run(frame):
            d = m.predict(frame, threshold=0.4)
            named = detection_names(d.class_id, names, (d.data or {}).get("class_name"))
            return [(int(x0), int(y0), int(x1), int(y1), n)
                    for (x0, y0, x1, y1), n in zip(d.xyxy, named) if n]
        return run
    from ultralytics import YOLO
    m = YOLO(str(weights))

    def run(frame):
        r = m.predict(frame[:, :, ::-1], conf=0.4, verbose=False)[0]
        return [(*map(int, b.xyxy[0].tolist()), r.names[int(b.cls)]) for b in r.boxes]
    return run


def video_reads(video: Path, weights: Path, red: List[str], blue: List[str],
                every: float, out: Path, version: str = "2") -> None:
    """One Moondream read per robot box every `every` seconds, snapped to its
    alliance by the detector's colour class. Writes a CSV to check by eye."""
    import cv2
    from PIL import Image
    find, model = detector(weights), load_model(version)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step, i = max(1, round(fps * every)), 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x0", "y0", "x1", "y1", "class", "answer", "team", "seconds"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                rgb = frame[:, :, ::-1].copy()
                for x0, y0, x1, y1, name in find(rgb):
                    teams = red if "red" in name else blue if "blue" in name else red + blue
                    t0 = time.time()
                    a = ask(model, Image.fromarray(rgb[y0:y1, x0:x1]))
                    w.writerow([f"{i / fps:.2f}", x0, y0, x1, y1, name, a,
                                snap(a, teams) or "", f"{time.time() - t0:.1f}"])
                    fh.flush()
            i += 1
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale-test", type=Path, help="photo for the size table")
    ap.add_argument("--box", action="append", default=[], help="x0,y0,x1,y1=TEAM (scale test)")
    ap.add_argument("--heights", default="35,20,14,10,7")
    ap.add_argument("--ref-height", type=int, default=35,
                    help="numeral height in px at full size in the photo")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--weights", type=Path, help="RF-DETR .pth or YOLO .pt with robot classes")
    ap.add_argument("--red", default="", help="red alliance team numbers, comma-separated")
    ap.add_argument("--blue", default="")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between reads")
    ap.add_argument("--out", type=Path, default=Path("bumper_reads.csv"))
    ap.add_argument("--moondream", choices=("2", "3.1"), default="2",
                    help="3.1 needs `pip install moondream` and an Ampere+ GPU or Apple Silicon")
    args = ap.parse_args()
    if args.scale_test:
        scale_test(args.scale_test, args.box, [int(h) for h in args.heights.split(",")],
                   args.ref_height, args.moondream)
    elif args.video and args.weights:
        video_reads(args.video, args.weights, [t for t in args.red.split(",") if t],
                    [t for t in args.blue.split(",") if t], args.every, args.out, args.moondream)
    else:
        ap.error("give --scale-test, or --video with --weights")


if __name__ == "__main__":
    main()
