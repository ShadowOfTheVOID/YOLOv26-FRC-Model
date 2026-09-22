#!/usr/bin/env python3
"""How fast is this model on this machine, and is that fast enough.

For the scouting detector the answer does not matter: `run.py detect` runs
offline over harvested frames and can take all night.

For `run.py count` at a scrimmage it is a correctness property. A model that
cannot process frames as fast as the camera produces them misses balls between
the frames it does see. The score comes out low, with no gap and nothing odd
about it, and with no FMS there is no second number anywhere that would
disagree. `count.health()` reports that while it is happening; this is how to
find out beforehand, on the box you will actually use.

    python3 train/benchmark.py --weights runs/count26/weights/best.pt \\
        --imgsz 960 --fps 30

What it does NOT measure: the tracker, the counting, and whatever else the box
is doing during a match. Inference is the dominant cost and the rest is not
free, so treat the number here as a ceiling and leave headroom -- the default
target is 90% of the camera's rate for that reason.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Below this fraction of the camera's rate, balls are being missed.
TARGET_FRAC = 0.9


def summarise(times, fps_target: float = 0.0) -> dict:
    """Per-frame timings -> the numbers worth printing.

    The median and the 95th percentile, not the mean. A mean hides the stalls,
    and a stall is exactly when a ball goes past: a model averaging 30 fps that
    pauses for 200 ms every few seconds drops balls in the pauses while looking
    fine on the average.
    """
    if not times:
        return {}
    ordered = sorted(times)
    med = statistics.median(ordered)
    p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
    out = {"frames": len(times),
           "medianMs": round(med * 1000, 1),
           "p95Ms": round(p95 * 1000, 1),
           "fps": round(1.0 / med, 1) if med > 0 else 0.0,
           "worstFps": round(1.0 / p95, 1) if p95 > 0 else 0.0}
    if fps_target:
        out["target"] = fps_target
        out["keepingUp"] = out["fps"] >= fps_target * TARGET_FRAC
        out["missedFrac"] = round(max(0.0, 1.0 - out["fps"] / fps_target), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=0.0,
                    help="the camera's real frame rate, to judge against")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10,
                    help="frames run before timing starts. The first few are "
                         "always slow -- weights load lazily and CUDA compiles "
                         "kernels on first use -- and timing them would report "
                         "a speed the model never has again.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import numpy as np
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "this needs ultralytics and numpy:\n"
            "  .venv-detect/bin/pip install -r requirements-detect.txt")

    if not args.weights.exists():
        raise SystemExit(f"no weights at {args.weights}")

    model = YOLO(str(args.weights))
    names = getattr(model, "names", {}) or {}
    # Random pixels, not a real frame. Inference cost is set by the input size
    # and the network, not by what is in the picture, and requiring a sample
    # frame would make this impossible to run before an event.
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)

    times = []
    for _ in range(args.frames):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
        times.append(time.perf_counter() - t0)

    got = summarise(times, args.fps)
    print(f"{args.weights.name}  imgsz={args.imgsz}  "
          f"{len(names)} class(es): {', '.join(names.values()) if names else '?'}")
    print(f"  median {got['medianMs']} ms  ->  {got['fps']} fps")
    print(f"  p95    {got['p95Ms']} ms  ->  {got['worstFps']} fps in the slow moments")

    if not args.fps:
        print("\nPass --fps <camera rate> and this will say whether that is "
              "enough.\nNothing at a scrimmage will tell you afterwards.")
        return 0

    if got["keepingUp"]:
        print(f"\nOK for a {args.fps:.0f} fps camera. Run the counter with "
              f"--expect-fps {args.fps:.0f}\nand it will say if that stops "
              f"being true.")
        return 0

    print(f"\n!! NOT enough for a {args.fps:.0f} fps camera: roughly "
          f"{got['missedFrac'] * 100:.0f}% of frames would go past unseen,\n"
          f"   and the balls crossing the hub in them would not be counted.\n"
          f"   In rough order of what to try:\n"
          f"     - export to ONNX:   train/train.py --export onnx\n"
          f"     - a smaller model:  --model yolo26n.pt\n"
          f"     - a smaller imgsz:  --imgsz {max(320, args.imgsz // 2)}\n"
          f"     - fewer classes:    train/subset_classes.py\n"
          f"     - or run the camera slower, which costs balls honestly "
          f"rather than silently")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
