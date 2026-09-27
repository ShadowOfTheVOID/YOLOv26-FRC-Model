"""Count fuel into each hub from yellow AREA crossing the funnel outline.

Experimental; validated on one match only. Not wired into run.py.

Why this exists. On 2026cmptx Einstein Playoff Match 4 (Johnson 484 - Daly
842, 1080p60 broadcast) the model-based counters were nowhere near the
scoreboard: `run.py count` counted 0 balls for both alliances and `run.py
shots` 13/23 on the banner-cropped video, 44/123 on the full frame. The
ByteTrack tracker cannot hold ~20 px balls: 107,736 tracks were "too few
frames to be a ball". Drum shooters make it worse -- three to five balls
arrive side by side as one yellow blob, so any one-blob-one-ball rule
undercounts them.

What this does instead. No model, no long tracks. Each yellow blob near a hub
is matched to the previous frame by position only. When a blob's centre
crosses the hand-drawn outline of the funnel mouth it counts
+/- round(area / one ball's area): a clump of four counts 4, a ball that flies
across the outline or bounces off the hood goes in and out and nets 0, one
that drops down the funnel nets +1. A closed outline rather than a line,
because a ball launched from the near side crosses any single line upward
first and downward second, which nets to zero whether it scored or not.

Measured on Einstein 4 (outlines below, --start 6 --end 182):

    alliance  scoreboard            --ball-area 272   measured (280 px)
    blue      479 fuel              444   (93%)       422   (88%)
    red       807 at the buzzer,    769   (95%/91%)   744   (92%/88%)
              842 after draining

The count is only as good as the one-ball area it divides by: 272 against
280 px, a 3% difference from sampling frames by timestamp or by index, moves
both totals about 5%.

Checked every 5 s against the scoreboard: flat while a hub is inactive (hub
shifts alternate every ~25 s), blue within 5-10% throughout, and the count
leads the scoreboard by 1-3 s because the hub scores a ball as it passes
through, not as it enters.

What went wrong on the way, so nobody repeats it:
  * Dense optical flow (Farneback) across the outline: blue came out 76 and
    went negative. Balls move 20-30 px/frame and flow mis-measures them.
  * Strict colour gate HSV (18,90,90)-(38,255,255): blue 363. Balls behind
    the clear polycarbonate hood lose saturation. The loose gate
    (15,55,45)-(40,255,255) from autolabel_fuel.py fixed it -- but the ball
    area must be measured with the SAME gate (272 px loose vs 235 px strict),
    or the count inflates by the ratio (blue 532 with the mismatch).

Not fixed, not known:
  * The parameters were chosen on the same match they are scored on. Until a
    second match with a known score agrees, 93%/95% is a fit, not a result.
  * Red takes two false drops (-20, -25) while its hub is inactive:
    something yellow leaving the outline. Not traced.
  * Hub totals only. Per-robot credit from a broadcast is not attempted.
  * It counts physical balls; a ball into an inactive hub scores nothing but
    is counted here. Nobody did that in Einstein 4.
  * Outlines are in the video's own pixels and must be redrawn per camera.

Usage (outlines are x,y pairs in full-frame pixels):

    python experiments/area_hub_count.py match.mp4 --start 6 --end 182 \\
        --blue 385,262,400,215,440,190,560,195,585,232,578,258,530,268 \\
        --red 1352,262,1362,225,1400,205,1505,205,1530,245,1540,272,1420,282 \\
        --timeline timeline.csv

The outlines above are Einstein's 2026 Championship camera. Draw new ones on
a still (`ffmpeg -ss 60 -i match.mp4 -frames:v 1 still.png`) around the clear
hood and funnel mouth of each hub, bottom edge on the solid front rim.
"""
import argparse
import csv
import json

import cv2
import numpy as np

LOOSE_LO = (15, 55, 45)     # autolabel_fuel.LOOSE_LO: keeps balls seen through the hood
LOOSE_HI = (40, 255, 255)
PAD = 60                    # search margin around each outline, px
MIN_AREA = 40               # smaller yellow specks are noise


def parse_poly(text):
    v = [int(x) for x in text.split(",")]
    if len(v) < 6 or len(v) % 2:
        raise argparse.ArgumentTypeError("need x,y pairs, at least three points")
    return list(zip(v[0::2], v[1::2]))


def regions(polys):
    out = {}
    for name, p in polys.items():
        p = np.array(p)
        x0, y0 = p.min(0) - PAD
        x1, y1 = p.max(0) + PAD
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        out[name] = ((x0, y0, int(x1), int(y1)),
                     (p - (x0, y0)).reshape(-1, 1, 2).astype(np.float32))
    return out


def yellow(frame, box, lo, hi):
    x0, y0, x1, y1 = box
    m = cv2.inRange(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV),
                    np.array(lo, np.uint8), np.array(hi, np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def ball_area(cap, fps, t0, t1, regs, lo, hi):
    """Median area of isolated round blobs near the hubs, one frame a second.

    Measured with the counting gate on purpose: a looser gate draws every blob
    larger, and a divisor from a different gate scales the whole count."""
    areas = []
    for t in np.arange(t0 + 4, t1 - 10, 1.0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            break
        for box, _ in regs.values():
            n, _, st, _ = cv2.connectedComponentsWithStats(yellow(f, box, lo, hi))
            for i in range(1, n):
                a, w, h = st[i, 4], st[i, 2], st[i, 3]
                if a > 60 and 0.75 < w / h < 1.33 and a > 0.6 * w * h:
                    areas.append(a)
    return float(np.median(areas)) if areas else 0.0


def count(path, polys, t0, t1, ball=0.0, lo=LOOSE_LO, hi=LOOSE_HI):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    regs = regions(polys)
    if not ball:
        ball = ball_area(cap, fps, t0, t1, regs, lo, hi)
    if not ball:
        raise SystemExit("no isolated balls found near the hubs to size one by")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    prev = {k: [] for k in regs}
    events = {k: [] for k in regs}
    fi = int(t0 * fps)
    while fi < int(t1 * fps):
        ok, f = cap.read()
        if not ok:
            break
        t = fi / fps
        for k, (box, poly) in regs.items():
            n, _, st, cen = cv2.connectedComponentsWithStats(yellow(f, box, lo, hi))
            cur = [{"c": cen[i], "a": st[i, 4], "v": np.zeros(2),
                    "in": cv2.pointPolygonTest(poly, (float(cen[i][0]), float(cen[i][1])), False) >= 0}
                   for i in range(1, n) if st[i, 4] >= MIN_AREA]
            pairs = []
            for i, b in enumerate(prev[k]):
                pred = b["c"] + b["v"]
                reach = max(25.0, 1.5 * np.sqrt(b["a"]))
                for j, c in enumerate(cur):
                    d = float(np.linalg.norm(c["c"] - pred))
                    if d <= reach:
                        pairs.append((d, i, j))
            used_i, used_j = set(), set()
            for _, i, j in sorted(pairs):
                if i in used_i or j in used_j:
                    continue
                used_i.add(i)
                used_j.add(j)
                b, c = prev[k][i], cur[j]
                c["v"] = c["c"] - b["c"]
                if b["in"] != c["in"]:
                    nb = max(1, int(round(max(b["a"], c["a"]) / ball)))
                    events[k].append((round(t, 3), nb if c["in"] else -nb))
            prev[k] = cur
        fi += 1
    return {"fps": fps, "ball_area_px": ball, "events": events,
            "totals": {k: sum(n for _, n in ev) for k, ev in events.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--blue", type=parse_poly, required=True, help="funnel outline, x,y,x,y,...")
    ap.add_argument("--red", type=parse_poly, required=True)
    ap.add_argument("--start", type=float, default=0.0, help="seconds, match start")
    ap.add_argument("--end", type=float, default=1e9, help="seconds, a few past the buzzer")
    ap.add_argument("--ball-area", type=float, default=0.0,
                    help="px area of one ball under the colour gate; measured from the video if 0")
    ap.add_argument("--timeline", help="write cumulative counts per second to this CSV")
    ap.add_argument("--events", help="write every crossing to this JSON")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 60.0)
    cap.release()
    r = count(a.video, {"blue": a.blue, "red": a.red}, a.start, min(a.end, dur), a.ball_area)
    print(f"ball area {r['ball_area_px']:.0f} px   blue {r['totals']['blue']}   red {r['totals']['red']}")
    if a.timeline:
        with open(a.timeline, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "blue_cum", "red_cum"])
            for s in range(int(a.start), int(min(a.end, dur)) + 1):
                w.writerow([s] + [sum(n for t, n in r["events"][k] if t <= s) for k in ("blue", "red")])
    if a.events:
        with open(a.events, "w") as fh:
            json.dump(r, fh)


if __name__ == "__main__":
    main()
