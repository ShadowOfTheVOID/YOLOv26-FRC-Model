#!/usr/bin/env python3
"""Export a trained fuel model to .onnx for the apps, and prove it is the same model.

The Hub Counter and Watchtower apps run models with ONNX Runtime
(tbavid/onnxdet.py) so they ship without torch. An export is only worth
shipping if it finds the same balls, so this runs the original (.pt with
Ultralytics, or the RF-DETR checkpoint with rfdetr) and the export on the
same images and refuses -- exit 1, no file kept -- unless every detection
matches: same count per image, each box within MATCH_PX and its score within
MATCH_SCORE. fuel_relabel.pt's export matched 437 of 437 boxes, worst
0.0002 px (2026-11-18).

The images are drawn here (field-grey floor, a hub-coloured block, yellow
balls of broadcast sizes at crop sizes the counter uses: 640, 900, 1280 px
squares), so the check needs no dataset; `--images DIR` adds real frames.

    python train/export_onnx.py models/fuel_relabel.pt
    python train/export_onnx.py runs/rfdetr_bake_sam3/checkpoint_best_total.pth --rfdetr
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MATCH_PX = 0.5
MATCH_SCORE = 1e-3
CONF = 0.05             # the hub counter's DET_CONF: compare every box it would see


def scenes(seed: int = 0) -> List[np.ndarray]:
    import cv2
    rng = np.random.default_rng(seed)
    out = []
    for side, n in ((640, 40), (640, 120), (900, 60), (1280, 200)):
        img = np.full((side, side, 3), (70, 80, 75), np.uint8)
        img = (rng.normal(0, 12, img.shape) + img).clip(0, 255).astype(np.uint8)
        cv2.rectangle(img, (side // 4, side // 2), (3 * side // 4, side), (90, 90, 160), -1)
        for _ in range(n):
            r = int(rng.uniform(6, 14))
            x, y = (int(v) for v in rng.integers(r, side - r, 2))
            cv2.circle(img, (x, y), r, (40, 210, 235), -1)
            cv2.circle(img, (int(x - r / 3), int(y - r / 3)), max(1, r // 3), (150, 245, 255), -1)
        out.append(img)
    return out


def compare(ref, got, label: str) -> List[str]:
    """ref/got: per image (xyxy, score). Problems found, empty if they agree."""
    bad = []
    worst = 0.0
    for i, ((rb, rs), (gb, gs)) in enumerate(zip(ref, got)):
        if len(rb) != len(gb):
            bad.append(f"image {i}: {len(rb)} boxes from the original, {len(gb)} from the export")
            continue
        used = set()
        for b, s in zip(rb, rs):
            d = np.abs(gb - b).max(1) if len(gb) else np.array([np.inf])
            for k in np.argsort(d):
                if k not in used and d[k] <= MATCH_PX and abs(gs[k] - s) <= MATCH_SCORE:
                    used.add(int(k))
                    worst = max(worst, float(d[k]))
                    break
            else:
                bad.append(f"image {i}: box {np.round(b, 1).tolist()} score {s:.3f} has no match")
                break
    n = sum(len(r[0]) for r in ref)
    print(f"{label}: {n} boxes compared, worst {worst:.4f} px" + ("" if bad else " -- identical"))
    return bad


def export_yolo(weights: Path, out: Path, imgsz: int, images) -> List[str]:
    from ultralytics import YOLO
    from tbavid.onnxdet import OnnxDetector
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / weights.name
        shutil.copy(weights, src)
        made = Path(YOLO(str(src)).export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True))
        shutil.move(str(made), out)
    pt, ox = YOLO(str(weights)), OnnxDetector(str(out))
    ref = [(r.boxes.xyxy.numpy(), r.boxes.conf.numpy()) for im in images
           for r in pt.predict([im], imgsz=imgsz, conf=CONF, max_det=1500, classes=[0],
                               verbose=False, device="cpu")]
    got = [(b, s) for b, s, _ in (ox.predict([im], conf=CONF, max_det=1500, classes=[0])[0]
                                  for im in images)]
    return compare(ref, got, f"{weights.name} vs {out.name}")


def export_rfdetr(weights: Path, out: Path, images) -> List[str]:
    """RF-DETR's own export (model.export writes inference_model.onnx), then the
    same comparison against its own predict at the counter's threshold."""
    sys.path.insert(0, str(ROOT / "train"))
    from train_rfdetr import load_rfdetr
    from tbavid.onnxdet import OnnxDetector
    model, names = load_rfdetr(weights)
    with tempfile.TemporaryDirectory() as tmp:
        model.export(output_dir=tmp)
        made = next(Path(tmp).rglob("*.onnx"))
        shutil.move(str(made), out)
    ox = OnnxDetector(str(out))
    if not ox.names:
        ox.names, ox.nc = list(names), len(names)
    ref, got = [], []
    for im in images:
        d = model.predict(im[:, :, ::-1].copy(), threshold=CONF)
        keep = np.asarray(d.class_id) == 0
        ref.append((np.asarray(d.xyxy)[keep], np.asarray(d.confidence)[keep]))
        b, s, _ = ox.predict([im], conf=CONF, max_det=10000, classes=[0])[0]
        got.append((b, s))
    return compare(ref, got, f"{weights.name} vs {out.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--rfdetr", action="store_true", help="an RF-DETR checkpoint, not a YOLO .pt")
    ap.add_argument("--out", type=Path, help="default: beside the weights, .onnx")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO input (the counter's crops: 640)")
    ap.add_argument("--images", type=Path, help="a folder of real frames to compare on as well")
    args = ap.parse_args(argv)
    import cv2
    out = args.out or args.weights.with_suffix(".onnx")
    images = scenes()
    if args.images:
        images += [cv2.imread(str(p)) for p in sorted(args.images.glob("*.jpg"))[:20]]
    bad = (export_rfdetr(args.weights, out, images) if args.rfdetr
           else export_yolo(args.weights, out, args.imgsz, images))
    if bad:
        out.unlink(missing_ok=True)
        print("REFUSED -- the export does not find the same balls:")
        for b in bad[:10]:
            print("  " + b)
        return 1
    print(f"wrote {out} ({out.stat().st_size >> 20} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
