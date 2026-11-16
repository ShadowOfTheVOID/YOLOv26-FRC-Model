#!/usr/bin/env python3
"""Score a YOLO26 model and an RF-DETR model with ONE metric on ONE val set.

Why not compare the numbers each trainer prints: Ultralytics and RF-DETR
compute mAP with different code (interpolation, area ranges, max detections),
so a 2-point gap between their logs can be the evaluators disagreeing rather
than the models. Here both models' predictions go through the same matching
and the same AP, written out below, against the same label files.

What it reports per class, and why each:

  * AP50 -- the usual summary, all-point interpolated, IoU >= 0.5.
  * AP50 on small boxes (shorter side < 24 px at source resolution) -- fuel
    is ~17x12 px on broadcast frames, and DETR-style models have historically
    been weaker on tiny objects. This is the column that decides fuel.
  * recall / precision at --conf -- what `run.py detect` actually sees at its
    working threshold, not an area under a curve.
  * median ms per image on this machine -- a model that wins by 1 AP and costs
    3x the time is not a win for anything that runs live.

The numbers are agreement with the labels. Labels came from the
auto-labellers, so a model that beats them looks WORSE here; check
disagreements by eye with train/preview_labels.py before believing a gap.

    python train/compare_detectors.py --data dataset-scout/dataset.yaml \\
        --yolo runs/yolo_bake/weights/best.pt \\
        --rfdetr runs/rfdetr/checkpoint_best_total.pth --rfdetr-size small \\
        --imgsz 960
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_classes import yaml_names  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SMALL_PX = 24
Box = Tuple[float, float, float, float]          # x0, y0, x1, y1 in pixels
Pred = Tuple[Box, float, str]                    # box, score, class name


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def is_small(b: Box) -> bool:
    return min(b[2] - b[0], b[3] - b[1]) < SMALL_PX


def match(preds: List[Tuple[Box, float]], truths: List[Box],
          thr: float = 0.5) -> List[Tuple[Box, float, bool, int]]:
    """Greedy by score, as VOC/COCO do: each prediction takes the best
    still-unmatched truth at IoU >= thr. Returns (box, score, true positive,
    index of the truth it took or -1)."""
    used = set()
    out = []
    for box, score in sorted(preds, key=lambda p: -p[1]):
        best, bi = thr, -1
        for i, t in enumerate(truths):
            if i in used:
                continue
            v = iou(box, t)
            if v >= best:
                best, bi = v, i
        if bi >= 0:
            used.add(bi)
        out.append((box, score, bi >= 0, bi))
    return out


def average_precision(scored: Sequence[Tuple[float, bool]], n_truth: int) -> float:
    """All-point interpolated AP from (score, is_tp) over every image."""
    if n_truth == 0:
        return float("nan")
    tp = fp = 0
    points = []
    for _, hit in sorted(scored, key=lambda s: -s[0]):
        tp += hit
        fp += not hit
        points.append((tp / n_truth, tp / (tp + fp)))
    ap, prev_r = 0.0, 0.0
    for i, (r, _) in enumerate(points):
        p_interp = max(p for _, p in points[i:])
        ap += (r - prev_r) * p_interp
        prev_r = r
    return ap


def evaluate(per_image: List[Tuple[List[Pred], List[Tuple[Box, str]]]],
             names: Sequence[str], conf: float) -> Dict[str, dict]:
    """per_image: (predictions, truths) for each val image -> per-class stats."""
    out = {}
    for name in names:
        scored, scored_small = [], []
        n_truth = n_small = tp_at = fp_at = 0
        for preds, truths in per_image:
            t = [b for b, c in truths if c == name]
            p = [(b, s) for b, s, c in preds if c == name]
            n_truth += len(t)
            small = {i for i, b in enumerate(t) if is_small(b)}
            n_small += len(small)
            for box, score, hit, ti in match(p, t):
                scored.append((score, hit))
                # small-box AP: hits on small truths, plus false boxes that are
                # small themselves (a large false box is not a small-object error)
                if hit and ti in small:
                    scored_small.append((score, True))
                elif not hit and is_small(box):
                    scored_small.append((score, False))
                if score >= conf:
                    tp_at += hit
                    fp_at += not hit
        out[name] = {
            "truth": n_truth,
            "AP50": average_precision(scored, n_truth),
            "AP50_small": average_precision(scored_small, n_small),
            "small": n_small,
            "recall": tp_at / n_truth if n_truth else float("nan"),
            "precision": tp_at / (tp_at + fp_at) if tp_at + fp_at else float("nan"),
        }
    return out


def val_set(data_yaml: Path) -> Tuple[List[Path], Path, List[str]]:
    root = Path(data_yaml).resolve().parent
    names = yaml_names(Path(data_yaml).read_text())
    images = sorted(p for p in (root / "images" / "val").iterdir() if p.suffix.lower() in IMAGE_EXT)
    return images, root / "labels" / "val", names


def read_truth(label: Path, w: int, h: int, names: Sequence[str]) -> List[Tuple[Box, str]]:
    out = []
    if label.exists():
        for line in label.read_text().split("\n"):
            f = line.split()
            if len(f) >= 5:
                c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
                out.append(((
                    (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h),
                    names[c]))
    return out


def yolo_predictor(weights: Path, imgsz: int):
    from ultralytics import YOLO
    m = YOLO(str(weights))

    def run(path: Path) -> List[Pred]:
        r = m.predict(str(path), imgsz=imgsz, conf=0.001, max_det=1000, verbose=False)[0]
        return [(tuple(b.xyxy[0].tolist()), float(b.conf), r.names[int(b.cls)]) for b in r.boxes]
    return run


def rfdetr_predictor(weights: Path, size: str = ""):
    """Size, resolution and class names come from the checkpoint
    (train_rfdetr.load_rfdetr) -- see there for the 1-based class_names trap."""
    from train_rfdetr import load_rfdetr
    m, names = load_rfdetr(weights, size)
    # Time the model as it would be deployed, or RF-DETR's ms/image is unfairly
    # high next to YOLO's: 1.12 has a separate optimised predictor (and warns
    # that plain predict() is not optimised); 1.5 optimises in place.
    if hasattr(m, "inference"):
        m = m.inference()
    elif hasattr(m, "optimize_for_inference"):
        m.optimize_for_inference()

    from train_rfdetr import detection_names

    def run(path: Path) -> List[Pred]:
        from PIL import Image
        d = m.predict(Image.open(path).convert("RGB"), threshold=0.001)
        named = detection_names(d.class_id, names, (d.data or {}).get("class_name"))
        return [(tuple(map(float, b)), float(s), n)
                for b, s, n in zip(d.xyxy, d.confidence, named) if n]
    return run


def run_model(predict, images: List[Path], label_dir: Path, names: Sequence[str]):
    from PIL import Image
    per_image, times = [], []
    for i, path in enumerate(images):
        w, h = Image.open(path).size
        t0 = time.perf_counter()
        preds = predict(path)
        if i >= 3:                       # first calls include warm-up / compilation
            times.append(time.perf_counter() - t0)
        per_image.append((preds, read_truth(label_dir / (path.stem + ".txt"), w, h, names)))
    return per_image, (statistics.median(times) * 1000 if times else float("nan"))


def yolo_spec(spec: str, imgsz: int) -> Tuple[str, str, int]:
    """'[NAME=]WEIGHTS[@IMGSZ]' -> (name, weights, imgsz). A bare path is
    named 'yolo26', as the single --yolo of the first comparison was."""
    name, sep, rest = spec.partition("=")
    if not sep:
        name, rest = "yolo26", spec
    path, at, size = rest.rpartition("@")
    if not at or not size.isdigit():
        path, size = rest, str(imgsz)
    return name, path, int(size)


def table(results: Dict[str, Tuple[Dict[str, dict], float]], names: Sequence[str]) -> str:
    w = max([10] + [len(m) for m in results])
    rows = [f"{'model':{w}} {'class':12} {'truth':>6} {'AP50':>6} {'AP50sm':>7} "
            f"{'recall':>7} {'prec':>6}"]
    for model, (stats, ms) in results.items():
        for n in names:
            s = stats[n]
            rows.append(f"{model:{w}} {n:12} {s['truth']:>6} {s['AP50']:>6.3f} "
                        f"{s['AP50_small']:>7.3f} {s['recall']:>7.3f} {s['precision']:>6.3f}")
        rows.append(f"{model:{w}} {'ms/image':12} {ms:>6.1f}")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--yolo", action="append", default=[], metavar="[NAME=]WEIGHTS[@IMGSZ]",
                    help="repeatable: several YOLO variants in one table, e.g. "
                         "p2=runs/yolo26s_p2/weights/best.pt  hi=runs/x/weights/best.pt@1280")
    ap.add_argument("--rfdetr", type=Path)
    ap.add_argument("--rfdetr-size", default="",
                    help="default: read from the checkpoint")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="YOLO inference size; RF-DETR uses the resolution it was trained at")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="working threshold for recall/precision")
    ap.add_argument("--limit", type=int, default=0, help="first N val images only")
    args = ap.parse_args()
    images, label_dir, names = val_set(args.data)
    if args.limit:
        images = images[:args.limit]
    results = {}
    for spec in args.yolo:
        name, path, imgsz = yolo_spec(spec, args.imgsz)
        if not Path(path).exists():
            print(f"  ! {name}: {path} not found -- skipped")
            continue
        per, ms = run_model(yolo_predictor(Path(path), imgsz), images, label_dir, names)
        results[name] = (evaluate(per, names, args.conf), ms)
    if args.rfdetr:
        per, ms = run_model(rfdetr_predictor(args.rfdetr, args.rfdetr_size),
                            images, label_dir, names)
        results["rfdetr"] = (evaluate(per, names, args.conf), ms)
    if not results:
        ap.error("give --yolo and/or --rfdetr")
    print(f"{len(images)} val images, conf {args.conf}, small = shorter side < {SMALL_PX} px")
    print(table(results, names))


if __name__ == "__main__":
    main()
