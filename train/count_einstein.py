#!/usr/bin/env python3
"""Count Einstein 1 with the shipped hub counter, once per model, against the scoreboard.

Why this and not mAP. On the SAM 3 dataset YOLO26 and RF-DETR scored 0.940
and 0.935 fuel AP50 on the same 487 Einstein 1 frames (2026-11-16): the
detectors have reached the labels' ceiling, so the score table can no longer
tell models apart. What the team needs is balls counted into a hub, and
deploy/HUB_FEED.md measured exactly that on Einstein 1 -- a match none of
the 2026-11-16 models trained on (it is the rebuild's validation video).

What runs is tbavid.hubcount.count_recording: the counter `run.py hubcount`
and the apps ship -- colour area-crossing at every frame, plus the model on
full-resolution crops around each hub (hubmodel.ModelEye, fuel class only)
blended at the shipped weight 0.5. One run per model gives three answers per
hub: colour alone (the same for every model), the model alone, and the blend.

The setup is rebuilt from what the repo recorded, not the scratch file the
documented numbers came from (that was never saved):
  * outlines: Einstein 4's, from experiments/area_hub_count.py;
  * Einstein 1's hub shift, template-matched and recorded in HUB_FEED.md:
    blue +16,+13 px, red +28,+5 px;
  * one ball 229 px, measured on Einstein 1 (HUB_FEED.md).
So absolute errors may differ from the documented 9.1% (colour) and 7.8%
(colour + fuel_relabel crops). Every model gets the same setup, so which one
counts best is a fair answer.

Truth, from the scoreboard (HUB_FEED.md): at the buzzer, video t=172 s,
blue 621 and red 415; at the AUTO decision, t=29-30 s, blue 95 and red 96 --
red by ONE ball, so the AUTO column is reported but decides nothing.

    python train/count_einstein.py --video e1.mp4 \\
        --model current=models/fuel_relabel.pt --model sam3=runs/yolo_bake_sam3/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent

# experiments/area_hub_count.py's Einstein 4 outlines (full-frame px)
E4_BLUE = [(385, 262), (400, 215), (440, 190), (560, 195), (585, 232), (578, 258), (530, 268)]
E4_RED = [(1352, 262), (1362, 225), (1400, 205), (1505, 205), (1530, 245), (1540, 272), (1420, 282)]
E1_SHIFT = {"blue": (16, 13), "red": (28, 5)}
E1_BALL = 229.0
E1_TRUTH = {"buzzer": (172.0, {"blue": 621, "red": 415}),
            "auto": (30.0, {"blue": 95, "red": 96})}
CAMERA = "broadcast"


def shifted(poly: Sequence[Tuple[float, float]], dx: float, dy: float) -> List[List[float]]:
    return [[x + dx, y + dy] for x, y in poly]


def e1_setup(video: str, weights: str = "", weight: float = 0.5) -> Dict:
    """The setup dict for Einstein 1, with a model blended in when given."""
    from tbavid.hubcount import RULES
    cam = {"name": CAMERA, "source": video, "ball_area": E1_BALL,
           "zones": [{"hub": "blue", "outline": shifted(E4_BLUE, *E1_SHIFT["blue"])},
                     {"hub": "red", "outline": shifted(E4_RED, *E1_SHIFT["red"])}]}
    if weights:
        cam["model"] = {"weights": weights, "weight": weight}
    return {"rules": RULES, "cameras": [cam]}


def hub_values(per_zone: Dict[str, float]) -> Dict[str, float]:
    """Zone name ('broadcast/blue0') -> hub totals."""
    out = {"blue": 0.0, "red": 0.0}
    for name, v in per_zone.items():
        hub = "blue" if "/blue" in name else "red" if "/red" in name else None
        if hub:
            out[hub] += v
    return out


def error(counted: Dict[str, float], truth: Dict[str, int]) -> float:
    """Mean over the two hubs of |counted - truth| / truth, in percent."""
    return 100.0 * sum(abs(counted[h] - truth[h]) / truth[h] for h in truth) / len(truth)


def at(timeline: Sequence[Sequence[float]], t: float) -> Dict[str, float]:
    """The (red, blue) timeline's last row at or before t."""
    row = None
    for r in timeline:
        if r[0] <= t + 1e-6:
            row = r
    return {"red": row[1], "blue": row[2]} if row else {"red": 0, "blue": 0}


def table(rows: List[Dict]) -> str:
    t_end, truth = E1_TRUTH["buzzer"]
    t_auto, auto = E1_TRUTH["auto"]
    out = [f"Einstein 1, buzzer t={t_end:.0f}s truth blue {truth['blue']} / red {truth['red']}; "
           f"AUTO t={t_auto:.0f}s truth {auto['blue']} - {auto['red']} (red by 1: decides nothing)",
           f"{'counter':34} {'blue':>6} {'red':>6} {'error':>7}   AUTO blue-red"]
    for r in rows:
        a = r.get("auto")
        out.append(f"{r['label']:34} {r['blue']:>6.0f} {r['red']:>6.0f} {r['error']:>6.1f}%"
                   + (f"   {a['blue']:.0f}-{a['red']:.0f}" if a else ""))
    return "\n".join(out)


def run(video: str, models: List[Tuple[str, str]], end: float = E1_TRUTH["buzzer"][0]) -> List[Dict]:
    from tbavid.hubcount import count_recording, setup_from_dict
    _, truth = E1_TRUTH["buzzer"]
    t_auto, _ = E1_TRUTH["auto"]
    rows = []
    r = count_recording(setup_from_dict(e1_setup(video)), video, CAMERA, end=end, every=1.0)
    c = {"blue": r["blue"], "red": r["red"]}
    rows.append(dict(label="colour counter alone (shipped)", **c, error=error(c, truth),
                     auto=at(r["timeline"], t_auto)))
    print(f"colour: {c} in {r['took_s']}s ({r['speed']} fps)", flush=True)
    for name, weights in models:
        r = count_recording(setup_from_dict(e1_setup(video, weights)), video, CAMERA,
                            end=end, every=1.0)
        m = hub_values(r["model"])
        b = {"blue": r["blue"], "red": r["red"]}
        rows.append(dict(label=f"{name}: model alone", **m, error=error(m, truth)))
        rows.append(dict(label=f"{name}: colour + model (0.5)", **b, error=error(b, truth),
                         auto=at(r["timeline"], t_auto)))
        print(f"{name}: model {m}, blend {b} in {r['took_s']}s", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the Einstein 1 broadcast (1080p)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=WEIGHTS.pt")
    ap.add_argument("--end", type=float, default=E1_TRUTH["buzzer"][0],
                    help="video seconds to count (errors are only meaningful at the buzzer, 172)")
    ap.add_argument("--out", type=Path, help="also write the table (and a .json beside it)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    models = []
    for spec in args.model:
        name, _, w = spec.partition("=")
        if not w:
            name, w = Path(spec).stem, spec
        if not Path(w).exists():
            print(f"  ! {name}: {w} not found -- skipped", flush=True)
            continue
        models.append((name, w))
    rows = run(args.video, models, args.end)
    text = table(rows)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
