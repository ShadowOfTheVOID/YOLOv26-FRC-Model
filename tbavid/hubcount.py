"""Count fuel into each hub from a fixed camera, fast enough to feed bioarena.

    python3 run.py hubfeed --source 0 --red 812,300,...  --blue 210,305,...

This is `experiments/area_hub_count.py` -- yellow area crossing a hand-drawn
outline of the funnel mouth -- turned from a batch script over a broadcast
into a live counter, because that is the only counter in this repository that
fits the bioarena feed's latency budget (spec 5.2: camera to count <= 80 ms
typical, 200 ms p99):

  * `count.BallCounter` holds every score 12 frames (400 ms at 30 fps) so a
    ball that passed over the hub can withdraw it. That hold alone is twice
    the p99 budget, and it needs the detector's tracks, which on a broadcast
    counted 0 balls (Einstein 4) and 19 of 74 (nhdur qm7).
  * The area counter needs no model and no long tracks. A blob is matched to
    the previous frame by position, and it counts on the frame its centre
    crosses the outline -- the "count at the plane" the spec asks for. The
    per-frame work is a colour gate on two small regions, a few milliseconds
    on a laptop CPU, so no GPU is needed and no frame queue builds up.

## What was measured, and what was not

On the 2026 Championship broadcast it counted 93% / 95% of the scoreboard on
Einstein 4, whose settings were fitted to it, and 120% / 137% blind on
Einstein 5 (see the experiment's docstring). It sees WHEN a hub scores --
flat in every inactive window -- and over-counts HOW MANY per burst.

A scrimmage camera is not a broadcast: closer, fixed, and set up for this. The
thresholds below are the experiment's, scaled by the one-ball area so they
keep their size relative to a ball; none has been checked at any other scale.
**Nothing here is validated on a practice-field camera.** The spec's
acceptance test -- twenty balls by hand-count against each tally, zero
disagreement -- is the first measurement, and until it passes the count must
not decide the AUTO winner (bioarena's "Counted" mode).

## Monotonic out, signed in

The experiment counts +n when a blob crosses into the outline and -n when one
crosses out: a ball that flies across the mouth or bounces off the hood nets
zero. The feed may never go down within a session (spec 4.2), so what is
reported is the high-water mark of that net: a ball that goes in and comes
back out is reported on entry, and its exit is absorbed by the next ball in,
which is then not reported again. The cumulative total is right as soon as
the hub scores again; in between it is one high, and bioarena credits that
one to the shift the false entry happened in. `owed` in the status line is
how many are being absorbed at the moment.
"""
from __future__ import annotations

import csv
import math
import statistics
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Blob = Tuple[float, float, float]            # cx, cy, area (px)

# autolabel_fuel.LOOSE_LO/HI. The strict gate lost balls seen through the
# clear hood (blue 363 against 444 on Einstein 4). The one-ball area must be
# measured under the SAME gate, or the count scales by the ratio of the two.
LOOSE_LO = (15, 55, 45)
LOOSE_HI = (40, 255, 255)

# The experiment's pixel constants at its measured one-ball area (272 px, a
# ~19 px ball on a 1080p broadcast), kept as fractions of a ball so a close
# camera with 2000 px balls gets the same geometry.
REF_BALL_AREA = 272.0
MIN_AREA_FRAC = 40.0 / REF_BALL_AREA      # smaller yellow specks are noise
PAD_BALLS = 60.0 / math.sqrt(REF_BALL_AREA)   # search margin, in ball widths
MIN_REACH_BALLS = 25.0 / math.sqrt(REF_BALL_AREA)


def parse_poly(text: str) -> List[Point]:
    """'x,y,x,y,...' -> [(x, y), ...], at least three points."""
    v = [float(x) for x in text.replace(" ", "").split(",") if x]
    if len(v) < 6 or len(v) % 2:
        raise ValueError("an outline is x,y pairs, at least three points")
    return list(zip(v[0::2], v[1::2]))


def point_in_poly(poly: Sequence[Point], p: Point) -> bool:
    """Even-odd ray cast. Points on an edge may land either side, which is
    fine: a crossing is a change of side between two frames, and a ball's
    centre is on the edge for at most one of them."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def region_of(poly: Sequence[Point], ball_area: float,
              width: int, height: int) -> Tuple[int, int, int, int]:
    """The pixel box searched for blobs: the outline plus a margin wide
    enough to see a ball the frame before it crosses."""
    pad = PAD_BALLS * math.sqrt(ball_area)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0 = max(0, int(min(xs) - pad))
    y0 = max(0, int(min(ys) - pad))
    x1 = min(width, int(math.ceil(max(xs) + pad)))
    y1 = min(height, int(math.ceil(max(ys) + pad)))
    return x0, y0, x1, y1


class CrossingCounter:
    """One hub: yellow blobs in, balls across the outline out.

    Pure state, no OpenCV, so every rule is testable without a camera.
    `update` takes one frame's blobs in full-frame pixels and returns how many
    balls to report NOW -- the rise in the high-water mark, never negative.
    """

    def __init__(self, poly: Sequence[Point], ball_area: float):
        if ball_area <= 0:
            raise ValueError("ball_area must be positive; measure it with "
                             "run.py hubfeed --measure")
        self.poly = list(poly)
        self.ball_area = float(ball_area)
        self.min_area = MIN_AREA_FRAC * self.ball_area
        self.min_reach = MIN_REACH_BALLS * math.sqrt(self.ball_area)
        self.prev: List[Dict] = []
        self.net = 0            # the experiment's signed count
        self.reported = 0       # what the feed has been told: max(net) so far
        self.entries = 0
        self.exits = 0

    @property
    def owed(self) -> int:
        """Balls reported that have since come back out, not yet absorbed."""
        return self.reported - self.net

    def balls_in(self, area: float) -> int:
        """A clump of four drum-fed balls is one blob of four balls' area."""
        return max(1, int(round(area / self.ball_area)))

    def update(self, blobs: Sequence[Blob]) -> int:
        cur = [{"c": (b[0], b[1]), "a": b[2], "v": (0.0, 0.0),
                "in": point_in_poly(self.poly, (b[0], b[1]))}
               for b in blobs if b[2] >= self.min_area]
        # Nearest pairs first, each blob used once, predicted by last motion.
        # Balls move 20-30 px a frame on a broadcast; dense optical flow
        # mis-measured exactly that and drove the count negative.
        pairs = []
        for i, b in enumerate(self.prev):
            px, py = b["c"][0] + b["v"][0], b["c"][1] + b["v"][1]
            reach = max(self.min_reach, 1.5 * math.sqrt(b["a"]))
            for j, c in enumerate(cur):
                d = math.hypot(c["c"][0] - px, c["c"][1] - py)
                if d <= reach:
                    pairs.append((d, i, j))
        used_i, used_j = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            b, c = self.prev[i], cur[j]
            c["v"] = (c["c"][0] - b["c"][0], c["c"][1] - b["c"][1])
            if b["in"] != c["in"]:
                n = self.balls_in(max(b["a"], c["a"]))
                if c["in"]:
                    self.net += n
                    self.entries += n
                else:
                    self.net -= n
                    self.exits += n
        self.prev = cur
        rise = max(0, self.net - self.reported)
        self.reported += rise
        return rise


# -- the camera side ------------------------------------------------------
#
# OpenCV is imported inside these functions, never at module level: the tests
# and CI have no cv2, and neither does a harvest-only install.

def frame_time(pos_msec: float, now: float) -> float:
    """When the frame was captured, on time.monotonic()'s clock.

    Linux V4L2 cameras report the driver's buffer timestamp as
    CAP_PROP_POS_MSEC, on CLOCK_MONOTONIC -- the same clock as
    time.monotonic() -- which is the capture instant the spec wants `age_ms`
    measured from. Other backends report a file position or nothing, so the
    stamp is used only when it is plausibly a capture time (up to 1 s old and
    not in the future); otherwise the moment the read returned is the best
    available, and `age_ms` then leaves out the time the frame sat in the
    driver's buffer.
    """
    t = pos_msec / 1000.0
    if 0.0 <= now - t <= 1.0:
        return t
    return now


def yellow_blobs(frame, box, lo=LOOSE_LO, hi=LOOSE_HI) -> List[Blob]:
    """Connected yellow components in `box`, in full-frame pixels."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    return [(float(cen[i][0]) + x0, float(cen[i][1]) + y0, float(st[i, 4]))
            for i in range(1, n)]


def isolated_ball_areas(frame, box, lo=LOOSE_LO, hi=LOOSE_HI) -> List[float]:
    """Areas of blobs that look like one ball: round and filled. The
    experiment's test, so the area it yields is the one it divided by."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        a, w, h = st[i, 4], st[i, 2], st[i, 3]
        if a > 60 and 0.75 < w / h < 1.33 and a > 0.6 * w * h:
            out.append(float(a))
    return out


def open_source(source: str, fps: float = 0.0, size: str = ""):
    """A cv2.VideoCapture with the driver's queue cut to one frame.

    A queue of frames is latency: at 30 fps each queued frame is 33 ms of the
    80 ms budget spent before the ball is even looked at (spec 5.2).
    """
    import cv2

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source {source!r}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if size:
        w, h = (int(v) for v in size.lower().split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def measure(source: str, polys: Dict[str, List[Point]], seconds: float,
            still_path: Optional[str], fps: float = 0.0, size: str = "",
            out=print) -> Optional[float]:
    """Commissioning: the one-ball area, and a still to draw outlines on.

    The count is only as good as the area it divides by -- 3% on the area
    moved both Einstein totals about 5% -- so it is measured on this camera,
    with balls sitting still near the hubs, not guessed.
    """
    import cv2

    cap = open_source(source, fps, size)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("the source gave no frames")
    h, w = frame.shape[:2]
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out(f"source: {w}x{h}, reports {real_fps:.1f} fps")
    if still_path:
        still = frame.copy()
        import numpy as np
        colours = {"red": (0, 0, 255), "blue": (255, 0, 0)}
        for hub, poly in polys.items():
            pts = np.array(poly, np.int32).reshape(-1, 1, 2)
            cv2.polylines(still, [pts], True, colours.get(hub, (0, 255, 0)), 2)
        cv2.imwrite(still_path, still)
        out(f"wrote {still_path}: draw each hub's funnel-mouth outline on it "
            f"(x,y pairs, bottom edge on the solid front rim)")
    if not polys:
        out("no outlines given, so no ball area measured")
        return None
    # A generous region while measuring: the area is not known yet, so size
    # the margin as if balls were large.
    boxes = [region_of(p, 4000.0, w, h) for p in polys.values()]
    areas: List[float] = []
    frames = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        for box in boxes:
            areas += isolated_ball_areas(frame, box)
    cap.release()
    if not areas:
        out(f"no isolated ball near either hub in {frames} frames. Put a few "
            f"balls, apart from each other, near each hub and run it again.")
        return None
    area = statistics.median(areas)
    out(f"one ball = {area:.0f} px under the counting gate "
        f"(median of {len(areas)} blobs over {frames} frames). "
        f"Pass --ball-area {area:.0f}")
    return area


class Health:
    """Frame rate and capture-to-processed lag over the last two seconds.

    A counter that falls behind the camera misses balls between the frames it
    sees and the count is simply low -- the same silent failure `count.py`
    times itself for. Here it also goes out to bioarena's operator as `info`.
    """

    def __init__(self, window: int = 120):
        self.window = window
        self.stamps: List[float] = []
        self.lags: List[float] = []
        self.last_frame = float("-inf")

    def frame(self, captured: float, done: float) -> None:
        self.stamps.append(captured)
        self.lags.append(done - captured)
        del self.stamps[:-self.window]
        del self.lags[:-self.window]
        self.last_frame = done

    def fps(self) -> float:
        if len(self.stamps) < 2 or self.stamps[-1] <= self.stamps[0]:
            return 0.0
        return (len(self.stamps) - 1) / (self.stamps[-1] - self.stamps[0])

    def lag_ms(self) -> float:
        return 1000 * max(self.lags) if self.lags else 0.0


STALE_S = 0.5    # a camera silent this long is blind; stop the heartbeat


def run(sender, hubs: Dict[str, Tuple[str, List[Point]]], ball_area: float,
        fps: float = 0.0, size: str = "", realtime: bool = False,
        log_path: Optional[str] = None, out=print,
        stop: Optional[threading.Event] = None) -> Dict[str, CrossingCounter]:
    """Count from the camera(s) into `sender` until stopped.

    `hubs` maps 'red'/'blue' to (source, outline). Two hubs on one source are
    one capture loop and one decode per frame; a hub with its own camera gets
    its own loop. A hub not given is never counted and reads 0, which is
    what the spec asks of a half field.

    The heartbeat runs on its own thread and only while every camera is
    delivering frames. A counter whose camera has died must not tell bioarena
    it is alive: frozen counts behind an ONLINE badge would let a match start
    in Counted mode with nobody watching the hubs.
    """
    stop = stop or threading.Event()
    counters = {h: CrossingCounter(p, ball_area) for h, (_, p) in hubs.items()}
    by_source: Dict[str, List[str]] = {}
    for hub, (src, _) in hubs.items():
        by_source.setdefault(src, []).append(hub)
    health = {src: Health() for src in by_source}
    log_lock = threading.Lock()
    log_file = open(log_path, "a", newline="") if log_path else None
    log = csv.writer(log_file) if log_file else None
    if log and log_file.tell() == 0:
        log.writerow(["wall", "session", "hub", "reported", "net", "balls",
                      "capture_lag_ms"])
    errors: List[str] = []

    def loop(src: str, names: List[str]) -> None:
        import cv2
        try:
            cap = open_source(src, fps, size)
        except SystemExit as e:
            errors.append(str(e))
            stop.set()
            return
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        boxes: Dict[str, Tuple[int, int, int, int]] = {}
        started = time.monotonic()
        n = 0
        hl = health[src]
        while not stop.is_set():
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok:
                errors.append(f"{src}: the source stopped giving frames")
                stop.set()
                break
            captured = frame_time(cap.get(cv2.CAP_PROP_POS_MSEC), now)
            if realtime:
                # A recording played at camera speed, so the heartbeat,
                # age_ms and bioarena's view match what a camera would do.
                due = started + n / file_fps
                if due > now:
                    time.sleep(due - now)
                captured = time.monotonic()
            n += 1
            if not boxes:
                h, w = frame.shape[:2]
                boxes = {hub: region_of(counters[hub].poly, ball_area, w, h)
                         for hub in names}
            for hub in names:
                rise = counters[hub].update(yellow_blobs(frame, boxes[hub]))
                if rise:
                    sender.score(hub, rise, captured)
                    if log:
                        with log_lock:
                            log.writerow([
                                time.strftime("%Y-%m-%dT%H:%M:%S"),
                                sender.session, hub, counters[hub].reported,
                                counters[hub].net, rise,
                                round((time.monotonic() - captured) * 1000)])
                            log_file.flush()
            hl.frame(captured, time.monotonic())
        cap.release()

    threads = [threading.Thread(target=loop, args=(src, names), daemon=True,
                                name=f"cam {src}")
               for src, names in by_source.items()]
    for t in threads:
        t.start()

    blind_said = False
    began = last_status = time.monotonic()
    try:
        while not stop.is_set():
            now = time.monotonic()
            stale = [s for s, hl in health.items()
                     if now - hl.last_frame > STALE_S]
            sender.info = info_line(health, counters)
            if stale:
                # A camera takes a second or two to open; only say so after.
                if not blind_said and now - began > 3.0:
                    out(f"no frames from {', '.join(stale)}: heartbeat held, "
                        f"bioarena will show OFFLINE")
                    blind_said = True
            else:
                blind_said = False
                sender.heartbeat()
            sender.poll_replies()
            if now - last_status >= 5.0:
                last_status = now
                out(status_line(sender, health, counters))
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
        if log_file:
            log_file.close()
    for e in errors:
        out(e)
    return counters


def info_line(health: Dict[str, Health],
              counters: Dict[str, CrossingCounter]) -> str:
    """<= 64 chars for bioarena's tooltip (spec 4.2)."""
    cams = " ".join(f"{hl.fps():.0f}fps/{hl.lag_ms():.0f}ms"
                    for hl in health.values())
    owed = sum(c.owed for c in counters.values())
    return f"cam {cams} owed {owed}"[:64]


def status_line(sender, health: Dict[str, Health],
                counters: Dict[str, CrossingCounter]) -> str:
    link = "no reply from bioarena"
    if sender.linked():
        r = sender.last_reply or {}
        link = (f"bioarena {r.get('match_state', '?')} {r.get('shift', '')} "
                f"rtt {sender.rtt_ms} ms")
    hubs = "  ".join(f"{h} {c.reported} (in {c.entries} out {c.exits})"
                     for h, c in sorted(counters.items()))
    errs = f"  send errors {sender.send_errors}: {sender.last_error}" \
        if sender.send_errors else ""
    return f"{hubs}  |  {info_line(health, counters)}  |  {link}{errs}"
