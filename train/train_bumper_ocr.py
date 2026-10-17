#!/usr/bin/env python3
"""Train the bumper-number reader (tbavid/bumpernet.py).

    # 1. robot crops from a match video (needs the robot model)
    python train/train_bumper_ocr.py extract --video e8.mp4 --out crops/e8 --step 2.0
    # 2. read the numbers by eye into train/bumper/e8_labels.txt ("index team")
    # 3. synthetic pre-training, then fine-tuning on real strips
    python train/train_bumper_ocr.py synth --steps 4000 --out ocr_synth.pt
    python train/train_bumper_ocr.py finetune --init ocr_synth.pt --out bumper_ocr.pt \\
        --steps 2500 crops/e8:train/bumper/e8_labels.txt:red=8613,6800,254:blue=4065,4414,1323 ...
    # 4. score a held-out match
    python train/train_bumper_ocr.py eval --weights bumper_ocr.pt crops/e1:train/bumper/e1_labels.txt:...

What was measured (2026-10-17, CPU): synthetic only, Einstein 1 held out,
65% of crops right when forced to answer; fine-tuned on Einstein 4/5/8,
74% forced, 83% on the 56% it is sure of. Template matching was 56%.

Synthetic strips are white digits drawn in Hershey fonts, italic or not, then
sheared, rotated, warped in perspective, shrunk to 7-24 px, blurred and
thresholded like the bumper mask, with stray white bits for decals. They are
what teaches the digits a match never shows (Einstein 8 has no 7 or 9).
Real strips are picked from each crop's candidates by the current model --
the one most likely to read as the crop's label -- because the widest white
strip is often a decal, not the number.
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tbavid import bumperocr as B          # noqa: E402
from tbavid import bumpernet as N          # noqa: E402


def extract(args):
    import cv2
    from ultralytics import YOLO
    m = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta, k, t = [], 0, 10.0
    while t < n / fps - 5:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, img = cap.read()
        t += args.step
        if not ok:
            continue
        r = m.predict(img, imgsz=960, conf=0.35, verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            name = m.names[int(c)]
            if not name.startswith("robot"):
                continue
            alliance = name.split("_")[1]
            crop = B.crop(img, (x1, y1, x2 - x1, y2 - y1))
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            top = int(hsv.shape[0] * (1 - B.BUMPER_PART))
            band = B.bumper_band(hsv[top:], alliance)
            if not band:
                continue
            y0, y1b = top + band[0], top + band[1]
            if not B.numeral_groups(B.white_mask(hsv[y0:y1b]),
                                    max(B.MIN_DIGIT_PX, int(0.3 * (y1b - y0)))):
                continue
            cv2.imwrite(f"{args.out}/{k:04d}.png", crop)
            meta.append({"k": k, "t": round(t - args.step, 1), "al": alliance})
            k += 1
    json.dump(meta, open(f"{args.out}/meta.json", "w"))
    print(f"{k} crops in {args.out}")


FONTS = None


def synth(text=None):
    import cv2
    global FONTS
    FONTS = FONTS or [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
                      cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX]
    text = text or str(random.choice([random.randint(1, 99), random.randint(100, 999),
                                      random.randint(1000, 11999), random.randint(1000, 9999)]))
    f = random.choice(FONTS) | (cv2.FONT_ITALIC if random.random() < .5 else 0)
    th = random.randint(3, 9)
    (tw, tht), b = cv2.getTextSize(text, f, 2.0, th)
    img = np.zeros((tht + b + 40, tw + 60), np.uint8)
    cv2.putText(img, text, (30, tht + 20), f, 2.0, 255, th, cv2.LINE_AA)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-25, 25), 1)
    M[0, 1] += random.uniform(-.45, .25)
    img = cv2.warpAffine(img, M, (w, h))
    d = lambda: random.uniform(-.12, .12)                     # noqa: E731
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * d(), h * d()], [w * (1 + d()), h * d()],
                      [w * (1 + d()), h * (1 + d())], [w * d(), h * (1 + d())]])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
    ys, xs = np.nonzero(img > 127)
    if ys.size == 0:
        return synth()
    p = random.randint(1, 6)
    img = img[max(0, ys.min() - p):ys.max() + p, max(0, xs.min() - p):xs.max() + p]
    th_ = random.randint(7, 24)
    img = cv2.resize(img, (max(4, int(img.shape[1] * th_ / img.shape[0])), th_),
                     interpolation=cv2.INTER_AREA)
    if random.random() < .6:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(.3, 1.0))
    img = (img > random.randint(70, 170)).astype(np.float32)
    for _ in range(random.randint(0, 3)):
        y, x = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)
        cv2.rectangle(img, (x, y), (x + random.randint(1, 8), y + random.randint(1, 3)), 1.0, -1)
    return N.normalise(img), text


def fit(net, source, steps, lr):
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    net.train()
    for s in range(steps):
        items = [source() for _ in range(64)]
        x = torch.tensor(np.stack([i[0] for i in items]))[:, None]
        tg = torch.cat([torch.tensor(N.encode(i[1])) for i in items])
        tl = torch.tensor([len(i[1]) for i in items])
        lp = net(x).permute(1, 0, 2)
        loss = F.ctc_loss(lp, tg, torch.full((64,), lp.shape[0], dtype=torch.long), tl,
                          zero_infinity=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 500 == 0:
            print(s, round(loss.item(), 3), flush=True)
    net.eval()


def parse_set(spec):
    """'crops/e8:labels.txt:red=1,2,3:blue=4,5,6' -> (dir, labels, roster)."""
    d, labels, *sides = spec.split(":")
    roster = {s.split("=")[0]: s.split("=")[1].split(",") for s in sides}
    lab = dict(line.split() for line in open(labels) if line.strip())
    meta = {m["k"]: m for m in json.load(open(f"{d}/meta.json"))}
    return d, lab, roster, meta


def real_strips(reader, sets, max_nll):
    import cv2
    out = []
    for spec in sets:
        d, lab, roster, meta = parse_set(spec)
        for k, team in lab.items():
            crop = cv2.imread(f"{d}/{int(k):04d}.png")
            best = None
            for img in N.regions(crop, meta[int(k)]["al"]):
                with reader.torch.no_grad():
                    nll = float(N.seq_nll(reader.net(reader.torch.tensor(img)[None, None])[0], [team])[0])
                if best is None or nll < best[0]:
                    best = (nll, img)
            if best and best[0] < max_nll:
                out.append((best[1], team))
    return out


def evaluate(reader, sets):
    import cv2
    for spec in sets:
        d, lab, roster, meta = parse_set(spec)
        res = []
        for k, team in lab.items():
            m = meta[int(k)]
            cands = roster[m["al"]]
            nll = reader.nll(cv2.imread(f"{d}/{int(k):04d}.png"), m["al"], [int(c) for c in cands])
            if team not in cands or nll is None:
                res.append((None, 0.0, team))
                continue
            o = np.argsort(nll)
            res.append((cands[o[0]], float(nll[o[1]] - nll[o[0]]), team))
        for th in (0, 1, 2, 3, 5):
            named = [r for r in res if r[0] and r[1] >= th]
            right = sum(r[0] == r[2] for r in named)
            print(f"{d} margin>={th}: named {len(named)}/{len(res)}, right {right} "
                  f"({100 * right / max(1, len(named)):.0f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=float, default=2.5)
    p.add_argument("--weights", default="fuel_withBotbest.pt")
    p = sub.add_parser("synth")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", default="ocr_synth.pt")
    p = sub.add_parser("finetune")
    p.add_argument("--init", required=True)
    p.add_argument("--out", default="bumper_ocr.pt")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--max-nll", type=float, default=20.0,
                   help="drop crops none of whose strips plausibly reads as the label")
    p.add_argument("sets", nargs="+")
    p = sub.add_parser("eval")
    p.add_argument("--weights", required=True)
    p.add_argument("sets", nargs="+")
    args = ap.parse_args()
    random.seed(0)
    if args.cmd == "extract":
        return extract(args)
    import torch
    torch.manual_seed(0)
    if args.cmd == "synth":
        net = N.build()
        fit(net, synth, args.steps, 2e-3)
        torch.save(net.state_dict(), args.out)
    elif args.cmd == "finetune":
        import cv2
        reader = N.Reader(args.init)
        real = real_strips(reader, args.sets, args.max_nll)
        print(f"{len(real)} real strips")

        def mixed():
            if random.random() < .5:
                return synth()
            img, team = random.choice(real)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-6, 6), random.uniform(.9, 1.1))
            M[0, 2] += random.uniform(-4, 4)
            img = cv2.warpAffine(img, M, (w, h))
            if random.random() < .3:
                img = cv2.dilate(img, np.ones((2, 2)))
            if random.random() < .3:
                img = cv2.erode(img, np.ones((2, 2)))
            return (img > .5).astype(np.float32), team
        fit(reader.net, mixed, args.steps, 5e-4)
        torch.save(reader.net.state_dict(), args.out)
    else:
        evaluate(N.Reader(args.weights), args.sets)


if __name__ == "__main__":
    main()
