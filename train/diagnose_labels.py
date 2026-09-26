#!/usr/bin/env python3
"""Which pass proposed what, and does any knob move it.

Run from the repo root with the training venv:
    .venv-train/bin/python train/diagnose_labels.py
"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import autolabel_fuel as af

ROOT = Path(__file__).resolve().parent.parent

print("=== which code is actually running ===")
for cmd in (["git", "log", "-1", "--format=%h %s"], ["git", "status", "--short", "train/"]):
    try:
        print("  " + (subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                     check=False).stdout.strip() or "(clean)"))
    except Exception as exc:
        print(f"  ! {exc}")
print(f"  autolabel_fuel.py from {af.__file__}")
print(f"  has sat_ratio: {'sat_ratio' in af.detect.__code__.co_varnames}")
print(f"  detect returns {af.detect.__doc__.splitlines()[0]}")

images = sorted(p for split in ("train", "val")
                for p in (ROOT / "dataset" / "images" / split).glob("*.jpg"))
if not images:
    raise SystemExit("no dataset images")
src = images[len(images) // 2]
img = cv2.imread(str(src))
print(f"\n=== {src.name} ===")
print(f"{'sat-ratio':>10} {'gated':>7} {'split':>7} {'rescued':>8} {'dropped':>8} {'total':>7}")
for ratio in (0.0, 0.4, 0.7, 0.85, 1.0):
    g, r, res, h, d = af.detect(img, 60, 3000, 0.62, False, sat_ratio=ratio)
    print(f"{ratio:>10} {len(g):>7} {len(r):>7} {len(res):>8} {len(d):>8} "
          f"{len(g)+len(r)+len(res):>7}")
print("\nIf rescued/dropped never move, the code being imported is not this code.")
