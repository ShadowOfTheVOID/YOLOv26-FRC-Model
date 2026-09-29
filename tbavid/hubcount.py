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
# cx, cy, area (px), then optionally the pixel covariance cxx, cyy, cxy --
# the blob's spread, which the blur correction reads its length from.
Blob = Tuple[float, ...]

# autolabel_fuel.LOOSE_LO/HI. The strict gate lost balls seen through the
# clear hood (blue 363 against 444 on Einstein 4). The one-ball area must be
# measured under the SAME gate, or the count scales by the ratio of the two.
LOOSE_LO = (15, 55, 45)
LOOSE_HI = (40, 255, 255)

# The experiment's pixel constants at its measured one-ball area (272 px, a
# ~19 px ball on a 1080p broadcast), kept as fractions of a ball so a close
# camera with 2000 px balls gets the same geometry.
REF_BALL_AREA = 272.0
# The floor and reach were doubled / x1.5 by the leave-one-out sweep over the
# three Einstein matches (2026-09-29): the experiment's 40 px floor let rim
# spray and crowd specks cross the mouth as balls, and the 25 px reach broke
# fast 60 fps balls into two tracks. Two of the three held-out folds chose
# exactly these; mouth error at the buzzer-side checkpoints went 30% -> 21%
# best fit. The third fold (Einstein 4 held out) chose reach x0.75.
MIN_AREA_FRAC = 80.0 / REF_BALL_AREA      # smaller yellow specks are noise
PAD_BALLS = 60.0 / math.sqrt(REF_BALL_AREA)   # search margin, in ball widths
MIN_REACH_BALLS = 37.5 / math.sqrt(REF_BALL_AREA)
REACH_PER_BLOB = 2.25   # reach also grows with the blob: x sqrt(its area)


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

    def __init__(self, poly: Sequence[Point], ball_area: float,
                 blur: float = 0.0):
        if ball_area <= 0:
            raise ValueError("ball_area must be positive; measure it with "
                             "run.py hubfeed --measure")
        if not 0.0 <= blur <= 1.0:
            raise ValueError("blur is a fraction, 0 to 1")
        self.poly = list(poly)
        self.ball_area = float(ball_area)
        self.blur = float(blur)
        self.diameter = math.sqrt(4.0 * self.ball_area / math.pi)
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

    def balls_in(self, area: float, cov=None, vel=(0.0, 0.0)) -> int:
        """A clump of four drum-fed balls is one blob of four balls' area.

        With `blur` > 0, one ball is taken to be smeared along its motion:
        a ball of diameter d whose blob is L long along its velocity covers
        A1 + d(L - d), and `blur` is the fraction of that smear credited.
        blur = 1 halved every Einstein total (the long blobs are mostly real
        trains of drum-fed balls); the right fraction differed per match
        (Einstein 4 best at 0, 5 at 0.2-0.3, 1 at 0.5-0.7), and a value
        picked on two matches did no better than 0 on the third. So it is a
        per-camera calibration, set against a hand-counted recording from
        that camera (`calibrate`), never a constant.
        """
        one = self.ball_area
        speed = math.hypot(vel[0], vel[1])
        if self.blur > 0 and cov is not None and speed > 0:
            ux, uy = vel[0] / speed, vel[1] / speed
            cxx, cyy, cxy = cov
            var = cxx * ux * ux + cyy * uy * uy + 2 * cxy * ux * uy
            length = 4.0 * math.sqrt(max(var, 0.0))
            one += self.blur * self.diameter * max(0.0, length - self.diameter)
        return max(1, int(round(area / one)))

    def update(self, blobs: Sequence[Blob]) -> int:
        cur = [{"c": (b[0], b[1]), "a": b[2], "v": (0.0, 0.0),
                "cov": tuple(b[3:6]) if len(b) >= 6 else None,
                "in": point_in_poly(self.poly, (b[0], b[1]))}
               for b in blobs if b[2] >= self.min_area]
        # Nearest pairs first, each blob used once, predicted by last motion.
        # Balls move 20-30 px a frame on a broadcast; dense optical flow
        # mis-measured exactly that and drove the count negative.
        pairs = []
        for i, b in enumerate(self.prev):
            px, py = b["c"][0] + b["v"][0], b["c"][1] + b["v"][1]
            reach = max(self.min_reach, REACH_PER_BLOB * math.sqrt(b["a"]))
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
            way = self.crossed(b, c)
            if way:
                big = b if b["a"] >= c["a"] else c
                n = self.balls_in(big["a"], big["cov"], c["v"])
                if way > 0:
                    self.net += n
                    self.entries += n
                else:
                    self.net -= n
                    self.exits += n
        self.prev = cur
        rise = max(0, self.net - self.reported)
        self.reported += rise
        return rise

    def crossed(self, b: Dict, c: Dict) -> int:
        """+1 if a blob moving from b to c crossed into the outline, -1 out."""
        if b["in"] == c["in"]:
            return 0
        return 1 if c["in"] else -1


def _side(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_cross(p: Point, q: Point, a: Point, b: Point) -> bool:
    """Does the move p->q cross the segment a-b?

    The move must END strictly on one side and not start on that same side,
    so a ball that stops exactly on the line counts on the frame it leaves
    it, and only once.
    """
    d1, d2 = _side(a, b, p), _side(a, b, q)
    if d2 == 0 or (d1 != 0 and (d1 > 0) == (d2 > 0)):
        return False
    d3, d4 = _side(p, q, a), _side(p, q, b)
    return d3 == 0 or d4 == 0 or (d3 > 0) != (d4 > 0)


class ExitLineCounter(CrossingCounter):
    """Balls leaving the hub through an exit, counted across a line.

    Every ball that scores comes back out of the hub, so the exit count IS
    the score -- and a ball that clips the rim and drops behind the hub, the
    error that sank the funnel-mouth counts (Einstein 1: 832 net entries over
    the red hood against 415 real), never reaches an exit.

    `line` is two points across the exit, `out` any point on the side balls
    go once they are out. A ball counts when its path from one frame to the
    next crosses the segment towards `out`; one that crosses back (a bounce
    off whatever is beyond the exit) is taken off again, so it nets zero. The
    path is tested, not which side each end is on, so a ball moving several
    of its own widths a frame is still caught -- and the segment's ends limit
    it, so a ball passing beyond them does not count.
    """

    def __init__(self, line: Sequence[Point], out: Point, ball_area: float,
                 blur: float = 0.0):
        a, b = (tuple(map(float, line[0])), tuple(map(float, line[1])))
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 1.0:
            raise ValueError("an exit line needs two different ends")
        out = tuple(map(float, out))
        if abs(_side(a, b, out)) < 1e-9:
            raise ValueError("the 'out' point must be off the line, on the "
                             "side balls go once they are out")
        # `poly` is what the search box and the drawings are built from.
        super().__init__([a, b, out], ball_area, blur)
        self.line = (a, b)
        self.out = out
        self.out_sign = 1 if _side(a, b, out) > 0 else -1

    def crossed(self, b: Dict, c: Dict) -> int:
        a0, a1 = self.line
        if not segments_cross(b["c"], c["c"], a0, a1):
            return 0
        after = _side(a0, a1, c["c"])
        if after == 0:
            return 0
        return 1 if (after > 0) == (self.out_sign > 0) else -1

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


def yellow_mask(frame, box, lo=LOOSE_LO, hi=LOOSE_HI):
    import cv2
    import numpy as np

    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def blobs_of(mask, box) -> List[Blob]:
    """Connected components of a mask, with centre, area and pixel
    covariance, in full-frame pixels."""
    import cv2
    import numpy as np

    x0, y0 = box[0], box[1]
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return []
    ys, xs = np.nonzero(lab)
    L = lab[ys, xs]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    cnt = np.maximum(np.bincount(L, minlength=n), 1)
    mx = np.bincount(L, xs, n) / cnt
    my = np.bincount(L, ys, n) / cnt
    cxx = np.bincount(L, xs * xs, n) / cnt - mx * mx
    cyy = np.bincount(L, ys * ys, n) / cnt - my * my
    cxy = np.bincount(L, xs * ys, n) / cnt - mx * my
    return [(float(cen[i][0]) + x0, float(cen[i][1]) + y0, float(st[i, 4]),
             float(cxx[i]), float(cyy[i]), float(cxy[i])) for i in range(1, n)]


def yellow_blobs(frame, box, lo=LOOSE_LO, hi=LOOSE_HI) -> List[Blob]:
    """Connected yellow components in `box`, in full-frame pixels."""
    return blobs_of(yellow_mask(frame, box, lo, hi), box)


class ZoneEye:
    """What one zone sees each frame: its search box, and -- with
    `remove_static` -- a running map of pixels that have been yellow most of
    the last ~3 s, removed before blobs are found.

    Removing them was measured to move the Einstein totals only a few
    percent (the crossings touching them were mostly balls passing in front
    of balls resting on the hood, not shirts), so it is off unless a
    camera's calibration says it helps.
    """

    STATIC_S = 3.0
    STATIC_FRAC = 0.5

    def __init__(self, box, remove_static: bool = False, fps: float = 30.0):
        self.box = box
        self.remove_static = remove_static
        self.alpha = 1.0 / (self.STATIC_S * (fps or 30.0))
        self.freq = None

    def blobs(self, frame) -> List[Blob]:
        m = yellow_mask(frame, self.box)
        if self.remove_static:
            y = (m > 0).astype("float32")
            if self.freq is None:
                self.freq = y
            else:
                self.freq *= (1.0 - self.alpha)
                self.freq += self.alpha * y
            m = m.copy()
            m[self.freq > self.STATIC_FRAC] = 0
        return blobs_of(m, self.box)


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


STREAM_SITES = ("twitch.tv", "youtube.com", "youtu.be", "kick.com")


def is_stream_page(source: str) -> bool:
    """A live-stream page (Twitch, YouTube, Kick) rather than a video URL.

    Such a page is HTML; the video behind it has to be looked up first.
    """
    s = str(source).strip().lower()
    if not s.startswith(("http://", "https://")):
        s = "https://" + s if any(s.startswith(h) or s.startswith("www." + h)
                                  for h in STREAM_SITES) else s
    if not s.startswith(("http://", "https://")):
        return False
    host = s.split("://", 1)[1].split("/", 1)[0]
    return any(host == h or host.endswith("." + h) for h in STREAM_SITES)


def stream_name(source: str) -> str:
    """'https://www.twitch.tv/firstinspires' -> 'firstinspires'."""
    path = str(source).split("://", 1)[-1].split("?", 1)[0].rstrip("/")
    parts = [p for p in path.split("/")[1:] if p]
    if len(parts) >= 2 and parts[-2] == "videos":      # a Twitch past broadcast
        return f"vod-{parts[-1]}"[:24]
    return (parts[-1] if parts else path.split("/")[0])[:24] or "stream"


def resolve_stream(page: str, timeout: float = 45.0) -> str:
    """The HLS address behind a stream page, via yt-dlp.

    Tried as the module first, from this Python's own environment: a
    double-clicked launcher runs .venv/bin/python without .venv/bin on PATH,
    so a `yt-dlp` installed there is not found by name. Same approach as
    `live.resolve`, which reads the scoreboard off Twitch for `run.py live`.
    """
    import importlib.util
    import shutil
    import subprocess
    import sys

    url = page if "://" in page else "https://" + page
    if importlib.util.find_spec("yt_dlp") is not None:
        cmd = [sys.executable, "-m", "yt_dlp"]
    elif shutil.which("yt-dlp"):
        cmd = [shutil.which("yt-dlp")]
    else:
        raise SystemExit("reading a Twitch or YouTube stream needs yt-dlp: "
                         ".venv/bin/pip install yt-dlp")
    try:
        proc = subprocess.run(cmd + ["-g", "--no-playlist", "--no-warnings",
                                     "-f", "best[height<=1080]/best", url],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"{url}: no answer from the stream site in {timeout:.0f} s")
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if proc.returncode != 0 or not lines:
        why = (proc.stderr.strip().splitlines() or ["no stream found"])[-1]
        if "not currently live" in why:
            raise SystemExit(f"{url}: the channel is not live right now")
        raise SystemExit(f"{url}: {why.replace('ERROR: ', '')}")
    return lines[0]


NETWORK_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://",
                   "udp://", "tcp://", "srt://")


def is_network_camera(source: str) -> bool:
    """A camera reached over the network -- a Wi-Fi IP camera, or a phone
    running an IP-camera app -- by its rtsp:// or http:// address. A
    Twitch / YouTube page is a stream, not a camera, and is looked up first.
    """
    s = str(source).strip().lower()
    return s.startswith(NETWORK_SCHEMES) and not is_stream_page(s)


# FFmpeg options for a network camera, set before it is opened. TCP rather
# than RTSP's default UDP: over Wi-Fi, UDP loses packets and the picture
# smears, which is worse for counting than a few ms. No input buffering and
# low-delay decoding, because every buffered frame is latency (spec 5.2), and
# a 5 s socket timeout so a camera that drops off Wi-Fi is noticed and
# reconnected rather than waited on.
NETWORK_OPTIONS = ("rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
                   "max_delay;0|stimeout;5000000|timeout;5000000")


def open_source(source: str, fps: float = 0.0, size: str = ""):
    """A cv2.VideoCapture with the driver's queue cut to one frame.

    A queue of frames is latency: at 30 fps each queued frame is 33 ms of the
    80 ms budget spent before the ball is even looked at (spec 5.2).

    A Twitch / YouTube page is looked up to its HLS stream first. That stream
    is seconds behind the field -- Twitch's delay, not ours -- so it suits
    practice and scouting, never bioarena's AUTO call.
    """
    import cv2

    if is_stream_page(source):
        cap = cv2.VideoCapture(resolve_stream(str(source)), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise SystemExit(f"could not open the stream behind {source!r}")
        return cap
    if is_network_camera(source):
        import os
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", NETWORK_OPTIONS)
        cap = cv2.VideoCapture(str(source).strip(), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise SystemExit(f"could not reach the network camera at "
                             f"{source!r} -- same Wi-Fi? address and port "
                             f"right? (open it in a browser or VLC to check)")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
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
            colour = next((c for k, c in colours.items() if k in hub),
                          (0, 255, 0))
            cv2.polylines(still, [pts], True, colour, 2)
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
STREAM_RETRIES = 5   # reconnects in a row before a stream counts as gone


# -- the camera setup -----------------------------------------------------
#
# One camera watching both hubs from the side of the field is what the Einstein
# broadcasts tested, and from in front a ball that clips the rim and drops
# behind the hub looks exactly like one that goes in (net entries over the red
# hood: 819 on Einstein 4, 832 on Einstein 1, against real totals of 804 and
# 415). The fix is more cameras, closer, each seeing part of a hub well --
# so a hub may be counted from any number of outlines on any number of
# cameras, each camera with its own ball size, since a camera a metre from a
# chute and one across the field see very different balls.

COMBINE = ("sum", "max", "median")
HUB_NAMES = ("red", "blue")


class Zone:
    """One outline -- or one exit line -- on one camera, counting into one hub.

    An outline counts balls crossing into the hub's mouth; an exit line
    (`line` two points, `out` a point beyond it) counts balls coming back
    out, which is the score and cannot be faked by a ball behind the hub.
    """

    def __init__(self, name: str, hub: str, outline: Sequence[Point],
                 ball_area: float, blur: float = 0.0,
                 line: Optional[Sequence[Point]] = None,
                 out: Optional[Point] = None):
        self.name = name
        self.hub = hub
        self.line = [tuple(map(float, p)) for p in line] if line else None
        self.out = tuple(map(float, out)) if out else None
        self.outline = [tuple(map(float, p)) for p in outline] if outline else None
        self.counter = self.new_counter(ball_area, blur)

    @property
    def kind(self) -> str:
        return "exit" if self.line else "outline"

    def new_counter(self, ball_area: float, blur: float = 0.0) -> CrossingCounter:
        if self.line:
            if len(self.line) != 2 or self.out is None:
                raise ValueError("an exit line is two points plus an 'out' point")
            return ExitLineCounter(self.line, self.out, ball_area, blur)
        return CrossingCounter(self.outline or [], ball_area, blur)


class Camera:
    def __init__(self, name: str, source: str, ball_area: float,
                 zones: List[Zone], fps: float = 0.0, size: str = "",
                 blur: float = 0.0, remove_static: bool = False):
        self.name = name
        self.source = str(source)
        self.ball_area = float(ball_area)
        self.zones = zones
        self.fps = fps
        self.size = size
        self.blur = float(blur)
        self.remove_static = bool(remove_static)


class Setup:
    """Every camera, and how each hub combines the zones that count it.

    `sum`: the zones see different balls (one camera per exit chute, or per
    side of a hub), so the hub's count is their total. `max`: they see the
    same balls from different angles and the one that missed fewest is
    taken. `median`: three or more see the same balls and the odd one out is
    outvoted. Each zone's own count never goes down, and a sum, a max and an
    order statistic of non-decreasing counts never go down either, so the
    combined count keeps the feed's rule. An even number of zones under
    `median` takes the lower middle one.
    """

    def __init__(self, cameras: List[Camera],
                 combine: Optional[Dict[str, str]] = None,
                 measuring: bool = False):
        self.cameras = cameras
        self.combine = {h: "sum" for h in HUB_NAMES}
        self.combine.update(combine or {})
        self.validate(measuring)

    def zones(self, hub: str) -> List[Zone]:
        return [z for c in self.cameras for z in c.zones if z.hub == hub]

    def hubs(self) -> List[str]:
        return [h for h in HUB_NAMES if self.zones(h)]

    def validate(self, measuring: bool = False) -> None:
        """`measuring`: the setup is being commissioned with --measure, which
        is how a camera's ball area is found, so it may not have one yet."""
        if not self.cameras:
            raise ValueError("no cameras")
        names, sources, zone_names = set(), set(), set()
        for c in self.cameras:
            if c.name in names:
                raise ValueError(f"two cameras are named {c.name!r}")
            names.add(c.name)
            if c.source in sources:
                # One device opened twice fails on most drivers, and when it
                # does not, both loops count the same balls.
                raise ValueError(f"source {c.source!r} is used by two cameras; "
                                 f"give that camera several zones instead")
            sources.add(c.source)
            if c.ball_area <= 0 and not measuring:
                raise ValueError(f"camera {c.name!r}: ball_area must be "
                                 f"measured (run.py hubfeed --measure)")
            if not c.zones:
                raise ValueError(f"camera {c.name!r} has no zones")
            if not 0.0 <= c.blur <= 1.0:
                raise ValueError(f"camera {c.name!r}: blur is a fraction, "
                                 f"0 to 1 (got {c.blur})")
            for z in c.zones:
                if z.hub not in HUB_NAMES:
                    raise ValueError(f"zone {z.name!r}: hub must be red or "
                                     f"blue, not {z.hub!r}")
                if z.name in zone_names:
                    raise ValueError(f"two zones are named {z.name!r}")
                zone_names.add(z.name)
        for hub, how in self.combine.items():
            if hub not in HUB_NAMES or how not in COMBINE:
                raise ValueError(f"combine: {hub!r}: {how!r} is not one of "
                                 f"{', '.join(COMBINE)}")
        if not self.hubs():
            raise ValueError("no zone counts into either hub")


def _outline(raw) -> List[Point]:
    if isinstance(raw, str):
        return parse_poly(raw)
    pts = [(float(p[0]), float(p[1])) for p in raw]
    if len(pts) < 3:
        raise ValueError("an outline needs at least three points")
    return pts


def setup_from_dict(cfg: Dict, measuring: bool = False) -> Setup:
    """The JSON setup file, parsed. Outlines are [[x, y], ...] or "x,y,...".

        {"combine": {"red": "sum", "blue": "max"},
         "cameras": [
           {"name": "red-exit", "source": "0", "ball_area": 1800, "fps": 60,
            "zones": [{"hub": "red", "outline": [[100,80],[520,80],[520,300]]}]},
           {"name": "blue-high", "source": "1", "ball_area": 950,
            "zones": [{"hub": "blue", "outline": "40,60,600,60,600,200"}]}]}
    """
    cams = []
    for i, c in enumerate(cfg.get("cameras") or []):
        name = str(c.get("name") or f"cam{i}")
        if "source" not in c:
            raise ValueError(f"camera {name!r} has no source")
        area = float(c.get("ball_area") or 0)
        blur = float(c.get("blur") or 0)
        zones = []
        for j, z in enumerate(c.get("zones") or []):
            zname = str(z.get("name") or f"{name}/{z.get('hub')}{j}")
            try:
                if z.get("line"):
                    line = _outline(list(z["line"]) + [z.get("out") or [0, 0]])
                    zones.append(Zone(zname, str(z.get("hub")), [],
                                      area if area > 0 else 1.0,
                                      min(max(blur, 0.0), 1.0),
                                      line=line[:2], out=line[2]))
                else:
                    zones.append(Zone(zname, str(z.get("hub")),
                                      _outline(z.get("outline") or []),
                                      area if area > 0 else 1.0,
                                      min(max(blur, 0.0), 1.0)))
            except ValueError as e:
                raise ValueError(f"zone {zname!r}: {e}")
        cams.append(Camera(name, c["source"], area, zones,
                           float(c.get("fps") or 0), str(c.get("size") or ""),
                           blur, bool(c.get("remove_static", False))))
    return Setup(cams, cfg.get("combine"), measuring)


def _zone_dict(z: Zone) -> Dict:
    pt = lambda p: [round(p[0], 1), round(p[1], 1)]
    if z.line:
        return {"name": z.name, "hub": z.hub, "line": [pt(p) for p in z.line],
                "out": pt(z.out)}
    return {"name": z.name, "hub": z.hub,
            "outline": [pt(p) for p in z.counter.poly]}


def setup_to_dict(setup: "Setup") -> Dict:
    """The inverse of setup_from_dict, for saving what the GUI built."""
    cams = []
    for c in setup.cameras:
        d = {"name": c.name, "source": c.source,
             "ball_area": round(c.ball_area, 1),
             "zones": [_zone_dict(z) for z in c.zones]}
        if c.fps:
            d["fps"] = c.fps
        if c.size:
            d["size"] = c.size
        if c.blur:
            d["blur"] = c.blur
        if c.remove_static:
            d["remove_static"] = True
        cams.append(d)
    return {"combine": dict(setup.combine), "cameras": cams}


def load_setup(path: str, measuring: bool = False) -> Setup:
    """A setup file. A recording named by a relative path is found next to
    the file, so a setup and its test videos can be moved together."""
    import json
    import os
    with open(path) as fh:
        cfg = json.load(fh)
    base = os.path.dirname(os.path.abspath(path))
    for c in cfg.get("cameras") or []:
        src = str(c.get("source", ""))
        if (src and not src.isdigit() and "://" not in src
                and not os.path.isabs(src)
                and os.path.exists(os.path.join(base, src))):
            c["source"] = os.path.join(base, src)
    return setup_from_dict(cfg, measuring)


def setup_from_flags(source: str, outlines: Dict[str, Sequence[Point]],
                     sources: Dict[str, Optional[str]], ball_area: float,
                     fps: float = 0.0, size: str = "") -> Setup:
    """The single-camera flags (--source --red --blue --red-source ...) as a
    Setup: hubs sharing a source become one camera with two zones."""
    by_source: Dict[str, List[Tuple[str, Sequence[Point]]]] = {}
    for hub, poly in outlines.items():
        by_source.setdefault(str(sources.get(hub) or source), []).append(
            (hub, poly))
    cams = []
    for src, hubs in by_source.items():
        name = "cam" if len(by_source) == 1 else f"cam-{'-'.join(h for h, _ in hubs)}"
        cams.append(Camera(name, src, ball_area,
                           [Zone(hub if len(by_source) == 1 else f"{name}/{hub}",
                                 hub, poly, ball_area if ball_area > 0 else 1.0)
                            for hub, poly in hubs], fps, size))
    return Setup(cams)


class HubTally:
    """Each hub's combined count, and how much of it bioarena has not heard.

    Camera threads call `rise` after updating their zones; it is locked so two
    cameras on one hub cannot both report the same increase.
    """

    def __init__(self, setup: Setup):
        self.setup = setup
        self.zones = {h: setup.zones(h) for h in HUB_NAMES}
        self.sent = {h: 0 for h in HUB_NAMES}
        self._lock = threading.Lock()

    def value(self, hub: str) -> int:
        counts = sorted(z.counter.reported for z in self.zones[hub])
        if not counts:
            return 0
        how = self.setup.combine[hub]
        if how == "max":
            return counts[-1]
        if how == "median":
            return counts[(len(counts) - 1) // 2]
        return sum(counts)

    def rise(self, hub: str) -> int:
        with self._lock:
            v = self.value(hub)
            r = v - self.sent[hub]
            if r > 0:
                self.sent[hub] = v
                return r
            return 0


def run(sender, setup: Setup, realtime: bool = False,
        log_path: Optional[str] = None, out=print,
        stop: Optional[threading.Event] = None,
        on_frame: Optional[Callable] = None,
        monitor: Optional[Dict] = None) -> HubTally:
    """Count from every camera into `sender` until stopped.

    One capture thread per camera; a camera with several zones decodes each
    frame once. A hub no zone counts reads 0, which is what the spec asks of
    a half field.

    The heartbeat runs on its own thread and only while EVERY camera is
    delivering frames -- including a redundant one under `max` or `median`.
    A counter with a dead camera must not tell bioarena it is alive: frozen
    or quietly degraded counts behind an ONLINE badge would let a match
    start in Counted mode with a hub half-watched. Fail closed; restart
    without that camera if the match must go on.

    `on_frame(camera_name, frame)` is called from each camera's thread with
    every frame it counted, for a live preview; it must return at once.
    `monitor`, if given, is filled with the live `health` and `tally` so a
    display can read them.
    """
    stop = stop or threading.Event()
    tally = HubTally(setup)
    health = {c.name: Health() for c in setup.cameras}
    if monitor is not None:
        monitor.update(health=health, tally=tally, errors=[])
    log_lock = threading.Lock()
    log_file = open(log_path, "a", newline="") if log_path else None
    log = csv.writer(log_file) if log_file else None
    if log and log_file.tell() == 0:
        log.writerow(["wall", "session", "camera", "zone", "hub",
                      "zone_reported", "zone_net", "hub_count", "balls_sent",
                      "capture_lag_ms"])
    errors: List[str] = []

    def loop(cam: Camera) -> None:
        import cv2
        try:
            cap = open_source(cam.source, cam.fps, cam.size)
        except SystemExit as e:
            errors.append(f"{cam.name}: {e}")
            stop.set()
            return
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        eyes: Dict[str, ZoneEye] = {}
        started = time.monotonic()
        n = 0
        hl = health[cam.name]
        stream = is_stream_page(cam.source)
        wireless = is_network_camera(cam.source)
        src = str(cam.source)
        pace = realtime and (stream or (not src.isdigit() and "://" not in src))
        drops = 0
        while not stop.is_set():
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok and (wireless or (stream and drops < STREAM_RETRIES)):
                # An HLS stream hiccups (a slow segment, a CDN switch) and a
                # Wi-Fi camera drops out, where a USB camera does not.
                # Reconnect rather than end the session; the heartbeat is
                # held meanwhile, so bioarena sees OFFLINE and nothing
                # pretends to be watching. A Wi-Fi camera is retried for as
                # long as the counter runs -- it is coming back when the
                # signal does -- a Twitch stream five times.
                drops += 1
                out(f"{cam.name}: {'connection' if wireless else 'stream'} "
                    f"dropped, reconnecting (try {drops}"
                    f"{'' if wireless else f'/{STREAM_RETRIES}'})")
                cap.release()
                stop.wait(min(2.0 * drops, 5.0))
                try:
                    cap = open_source(cam.source)
                except SystemExit as e:
                    out(f"{cam.name}: {e}")
                # The picture after a gap is not the one before it: a ball
                # remembered from before the drop, matched to a different one
                # after, would be a crossing that never happened.
                for z in cam.zones:
                    z.counter.prev = []
                continue
            if not ok:
                errors.append(f"{cam.name} ({cam.source}): the source "
                              f"stopped giving frames")
                stop.set()
                break
            drops = 0
            captured = frame_time(cap.get(cv2.CAP_PROP_POS_MSEC), now)
            if pace:
                # A recording played at camera speed, so the heartbeat,
                # age_ms and bioarena's view match what a camera would do.
                # A Twitch past broadcast is a recording too: unpaced it was
                # read at 315 fps. A live camera is never paced -- it
                # already delivers at its own rate, and a wrong reported
                # fps would slow it down.
                due = started + n / file_fps
                if due > now:
                    time.sleep(due - now)
                captured = time.monotonic()
            n += 1
            if not eyes:
                h, w = frame.shape[:2]
                eyes = {z.name: ZoneEye(region_of(z.counter.poly,
                                                  cam.ball_area, w, h),
                                        cam.remove_static, file_fps)
                        for z in cam.zones}
            touched = []
            for z in cam.zones:
                if z.counter.update(eyes[z.name].blobs(frame)):
                    touched.append(z)
            for z in touched:
                sent = tally.rise(z.hub)
                if sent:
                    sender.score(z.hub, sent, captured)
                if log:
                    with log_lock:
                        log.writerow([
                            time.strftime("%Y-%m-%dT%H:%M:%S"), sender.session,
                            cam.name, z.name, z.hub, z.counter.reported,
                            z.counter.net, tally.value(z.hub), sent,
                            round((time.monotonic() - captured) * 1000)])
                        log_file.flush()
            hl.frame(captured, time.monotonic())
            if on_frame:
                on_frame(cam.name, frame)
        cap.release()

    threads = [threading.Thread(target=loop, args=(c,), daemon=True,
                                name=f"cam {c.name}") for c in setup.cameras]
    for t in threads:
        t.start()

    blind_said = False
    began = last_status = time.monotonic()
    try:
        while not stop.is_set():
            now = time.monotonic()
            stale = [s for s, hl in health.items()
                     if now - hl.last_frame > STALE_S]
            sender.info = info_line(health, tally)
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
                out(status_line(sender, health, tally))
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
    if monitor is not None:
        monitor["errors"] = list(errors)
    return tally


def info_line(health: Dict[str, Health], tally: HubTally) -> str:
    """<= 64 chars for bioarena's tooltip (spec 4.2): the slowest camera and
    the worst lag, since with several cameras the weakest one is the news."""
    fps = [hl.fps() for hl in health.values()]
    lag = [hl.lag_ms() for hl in health.values()]
    owed = sum(z.counter.owed for zs in tally.zones.values() for z in zs)
    cams = f"{len(fps)} cams min " if len(fps) > 1 else "cam "
    return (f"{cams}{min(fps):.0f}fps lag {max(lag):.0f}ms "
            f"owed {owed}")[:64]


def status_line(sender, health: Dict[str, Health], tally: HubTally) -> str:
    link = "no reply from bioarena"
    if sender.linked():
        r = sender.last_reply or {}
        link = (f"bioarena {r.get('match_state', '?')} {r.get('shift', '')} "
                f"rtt {sender.rtt_ms} ms")
    parts = []
    for hub in HUB_NAMES:
        zs = tally.zones[hub]
        if not zs:
            continue
        detail = ", ".join(f"{z.name} {z.counter.reported}" for z in zs)
        how = f" {tally.setup.combine[hub]}" if len(zs) > 1 else ""
        parts.append(f"{hub} {tally.value(hub)}{how} [{detail}]")
    errs = f"  send errors {sender.send_errors}: {sender.last_error}" \
        if sender.send_errors else ""
    return f"{'  '.join(parts)}  |  {info_line(health, tally)}  |  {link}{errs}"


# -- calibration against a hand count --------------------------------------

BLUR_STEPS = tuple(round(0.1 * i, 1) for i in range(11))


def record_blobs(camera: Camera, video: str, remove_static: bool,
                 start: float = 0.0, end: float = 0.0,
                 progress: Optional[Callable[[float], None]] = None,
                 stop: Optional[threading.Event] = None) -> List[Dict]:
    """Every zone's blobs for every frame of a recording, decoded once, so
    the blur settings can be replayed over it in seconds."""
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"could not open {video!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    first = int(start * fps)
    last = int(end * fps) if end else total
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    eyes = None
    frames: List[Dict] = []
    fi = first
    while (not last or fi < last) and not (stop and stop.is_set()):
        ok, frame = cap.read()
        if not ok:
            break
        if eyes is None:
            h, w = frame.shape[:2]
            eyes = {z.name: ZoneEye(region_of(z.counter.poly,
                                              camera.ball_area, w, h),
                                    remove_static, fps)
                    for z in camera.zones}
        frames.append({name: eye.blobs(frame) for name, eye in eyes.items()})
        fi += 1
        if progress and fi % 60 == 0 and last > first:
            progress((fi - first) / (last - first))
    cap.release()
    return frames


def replay_counts(camera: Camera, frames: List[Dict], blur: float) -> Dict[str, int]:
    """Each hub's count from recorded blobs at one blur setting."""
    counters = {z.name: (z.hub, z.new_counter(camera.ball_area, blur))
                for z in camera.zones}
    for rec in frames:
        for name, (_, c) in counters.items():
            c.update(rec.get(name, []))
    out: Dict[str, int] = {}
    for hub, c in counters.values():
        out[hub] = out.get(hub, 0) + c.reported
    return out


def calibrate(camera: Camera, video: str, hand: Dict[str, int],
              start: float = 0.0, end: float = 0.0, out=print,
              progress=None, stop=None) -> Dict:
    """Pick this camera's `blur` and `remove_static` from a recording whose
    balls were counted by hand.

    Replays every setting over the recording and keeps the one whose counts
    are nearest the hand count (total absolute error over the hubs given;
    ties go to less correction). This is fitting, so it is only as good as
    the recording: use a few minutes of real shooting from the camera's
    real position, and check the choice on a second recording.
    """
    results = []
    for remove_static in (False, True):
        out(f"decoding {video} ({'static removed' if remove_static else 'as is'})")
        frames = record_blobs(camera, video, remove_static, start, end,
                              progress, stop)
        if stop and stop.is_set():
            break
        for blur in BLUR_STEPS:
            got = replay_counts(camera, frames, blur)
            err = sum(abs(got.get(h, 0) - n) for h, n in hand.items())
            results.append({"blur": blur, "remove_static": remove_static,
                            "counts": got, "error": err})
    if not results:
        return {}
    for r in results:
        out(f"  blur {r['blur']:.1f}  static {'on ' if r['remove_static'] else 'off'}"
            f"  " + "  ".join(f"{h} {r['counts'].get(h, 0)}/{n}"
                              for h, n in hand.items())
            + f"  error {r['error']}")
    best = min(results, key=lambda r: (r["error"], r["blur"], r["remove_static"]))
    base = next(r for r in results if r["blur"] == 0 and not r["remove_static"])
    out(f"best: blur {best['blur']}, remove_static {best['remove_static']} "
        f"(error {best['error']}; uncorrected {base['error']})")
    return {"best": best, "uncorrected": base, "all": results}
