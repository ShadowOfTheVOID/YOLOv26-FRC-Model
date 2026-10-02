"""The trained fuel model as a second hub counter, blended with the colour one.

The colour counter (hubcount.CrossingCounter) over-counts: on Einstein 1 it
read 99% / 117% of the official totals at the buzzer, because balls that clip
the rim and drop behind the hub look like scores from in front. The model
counter (detections on a crop around the hub -> trackvis.BallTracker ->
count.BallCounter) under-counts the other way, 81% / 103%, because the
tracker loses balls fired in streams. They fail on different balls, so a
weighted mean of the two is nearer than either: on Einstein 1, the one
Einstein match fuel_relabel.pt never trained on, colour alone was 9.1% mean
error against the official checkpoints, the model alone 12.1%, the mean of
the two 7.8% (CHANGELOG, 2026-10-02).

Both halves are cumulative and never go down, so a weighted mean of them,
rounded, never goes down either -- the feed's rule holds.

The model half is slower to report: BallCounter confirms a score only after
the ball has been gone `vanish` frames and nothing has reappeared near it for
`reacquire` frames. Half of every ball therefore arrives that much later than
the colour half; DEFAULT_MODEL holds the hold to what the tuning measured.

`ModelCounter` is pure state (no torch, no OpenCV) so it is tested without a
model; `ModelEye` runs the network and needs requirements-detect.txt.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .count import REACQUIRE_PX, BallCounter
from .trackvis import ASSIST_CONF, BallTracker

Point = Tuple[float, float]
XYXY = Tuple[float, float, float, float]

# Geometry of the crop the model sees, in widths of the hub's outline. Chosen
# on the broadcasts, where an outline was ~200 px wide: a 640 px square, 140 px
# above the outline's centre (shots arc in from above), detections kept within
# 250 px of it. On 640 px crops at full resolution moving-ball recall was 93%
# against 84% for the whole 1920 px frame at imgsz 960, at the same cost.
CROP_WIDTHS = 3.2
CROP_UP_WIDTHS = 0.7
NEAR_WIDTHS = 1.25
CROP_MIN_PX = 640
MODEL_IMGSZ = 640
MODEL_FPS = 30.0        # tuned at 30 fps: every other frame of a 60 fps source
DET_CONF = 0.05         # what the model reports; the tracker uses >= KEEP_CONF
KEEP_CONF = 0.1

# Tuned on Einstein 4/5/8/1 with fuel_relabel.pt (deploy/HUB_FEED.md, "The
# combo, tuned"): the BallCounter windows in 30 fps frames, the hub box's pad
# in outline widths (15 px on a 194 px broadcast outline), and whether
# colour_assist adds moving yellow blobs. The settings the three-match fits
# agreed on; they also hold a score for the shortest time tried -- gone 2
# frames, not back within 2 -- about 170 ms at 30 fps where reacquire 6 took
# 300 ms, which the AUTO call's ~200 ms budget needs.
DEFAULT_MODEL = {"min_track": 2, "vanish": 2, "reacquire": 2,
                 "require_entry": True, "pad": 0.08, "assist": False}
# The model's share. 0.7 fit Einstein 4/5/8 best, but the model trained on
# those three; on Einstein 1, which it never saw, 0.5-0.6 was better (4.3% /
# 4.2% against 4.9% at 0.7) and the fits were flat from 0.5 to 0.7 (6.9 /
# 6.6 / 6.7%), so the plain mean -- the least trust in the model.
DEFAULT_WEIGHT = 0.5
REF_OUTLINE_PX = 194.0  # the broadcast outline width REACQUIRE_PX was set on


def crop_box(poly: Sequence[Point], width: int, height: int) -> Tuple[int, int, int, int]:
    """The square the model is run on for one outline, inside the frame."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    w = max(max(xs) - min(xs), 1.0)
    side = int(min(max(CROP_MIN_PX, CROP_WIDTHS * w), width, height))
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2 - CROP_UP_WIDTHS * w
    x0 = int(min(max(cx - side / 2, 0), width - side))
    y0 = int(min(max(cy - side / 2, 0), height - side))
    return x0, y0, x0 + side, y0 + side


class ModelCounter:
    """One hub from model detections: BallTracker -> BallCounter on the
    outline's bounding box. Pure state; `update` takes one model frame's
    detections in full-frame pixels and returns the rise in `reported`."""

    def __init__(self, poly: Sequence[Point], fps: float = MODEL_FPS,
                 min_track: int = DEFAULT_MODEL["min_track"],
                 vanish: int = DEFAULT_MODEL["vanish"],
                 reacquire: int = DEFAULT_MODEL["reacquire"],
                 require_entry: bool = DEFAULT_MODEL["require_entry"],
                 pad: float = DEFAULT_MODEL["pad"],   # outline widths
                 assist: bool = DEFAULT_MODEL["assist"]):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        self.box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        width = max(self.box[2], 1.0)
        self.margin = NEAR_WIDTHS * width
        self.assist = assist
        self.tracker = BallTracker()
        self.counter = BallCounter({"hub": self.box}, min_track_frames=min_track,
                                   vanish_frames=vanish,
                                   reacquire_frames=reacquire,
                                   reacquire_px=REACQUIRE_PX * width / REF_OUTLINE_PX,
                                   require_entry=require_entry,
                                   pad=pad * width, fps=fps)
        self.frame = 0
        self.reported = 0

    def near(self, b: XYXY) -> bool:
        x, y, w, h = self.box
        m = self.margin
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return x - m <= cx <= x + w + m and y - m <= cy <= y + h + m

    def update(self, t: float, dets: Sequence[Tuple[XYXY, float]],
               assist: Sequence[XYXY] = ()) -> int:
        return self.count(t, self.track(dets, assist))

    def track(self, dets: Sequence[Tuple[XYXY, float]],
              assist: Sequence[XYXY] = ()) -> Dict[int, XYXY]:
        """The tracker's step. Split from `count` so the tuning could replay
        one tracking pass under every counter setting."""
        keep = [(b, c) for b, c in dets if c >= KEEP_CONF and self.near(b)]
        if self.assist:
            keep += [(tuple(b), ASSIST_CONF) for b in assist if self.near(b)]
        return self.tracker.update(keep)

    def count(self, t: float, got: Dict[int, XYXY]) -> int:
        self.counter.update(self.frame, t, {tid: (b[0], b[1], b[2] - b[0], b[3] - b[1])
                                            for tid, b in got.items()})
        self.frame += 1
        now = self.counter.totals.get("hub", 0)
        rise = max(0, now - self.reported)
        self.reported += rise
        return rise


def blend(colour: int, model: int, weight: float) -> int:
    """The zone's count: round(weight * model + (1 - weight) * colour).
    Monotone in both, so it never goes down while they do not."""
    return int(math.floor(weight * model + (1.0 - weight) * colour + 0.5))


class ModelEye:
    """Runs the model on every model zone of one camera, one batch a frame.

    Loads Ultralytics lazily: the API host and the colour-only counter never
    import torch. `device` is passed to Ultralytics ("" lets it choose: CUDA,
    then Apple MPS, then CPU). On a 4-core CPU two 640 px crops took 0.3 s,
    too slow to keep up live; a laptop GPU or Apple silicon is needed.
    """

    def __init__(self, weights: str, polys: Dict[str, Sequence[Point]],
                 device: str = ""):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device or None
        self.polys = dict(polys)
        self.crops: Dict[str, Tuple[int, int, int, int]] = {}
        self.prev = None
        self.ball_px = 18.0

    def detect(self, frame) -> Tuple[List[Tuple[XYXY, float]], List[XYXY]]:
        """Every crop's detections in full-frame pixels, and colour_assist's
        moving yellow blobs the model did not report."""
        from .trackvis import colour_assist
        if not self.crops:
            h, w = frame.shape[:2]
            self.crops = {n: crop_box(p, w, h) for n, p in self.polys.items()}
        names = list(self.crops)
        imgs = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in (self.crops[n] for n in names)]
        kw = {"device": self.device} if self.device else {}
        rs = self.model.predict(imgs, imgsz=MODEL_IMGSZ, conf=DET_CONF,
                                max_det=1500, classes=[0], verbose=False, **kw)
        dets: List[Tuple[XYXY, float]] = []
        for n, r in zip(names, rs):
            x0, y0 = self.crops[n][:2]
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                dets.append(((b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), float(c)))
        assist: List[XYXY] = []
        if self.prev is not None:
            strong = sorted(max(b[2] - b[0], b[3] - b[1]) for b, c in dets if c >= KEEP_CONF)
            if strong:
                self.ball_px = 0.8 * self.ball_px + 0.2 * strong[len(strong) // 2]
            assist = [e[0] for e in colour_assist(frame, self.prev,
                                                  [b for b, c in dets if c >= KEEP_CONF],
                                                  self.ball_px)]
        self.prev = frame
        return dets, assist


def model_stride(fps: float) -> int:
    """Run the model on every n-th frame so it sees ~MODEL_FPS: the counter's
    windows were tuned there, and every frame doubles the cost for nothing
    measured (the model counter at 60 fps was 25% vs 17% at 30)."""
    return max(1, int(round((fps or MODEL_FPS) / MODEL_FPS)))


def parse_model(raw: Optional[Dict]) -> Optional[Dict]:
    """A camera's "model" entry from the setup file, checked and filled in."""
    if not raw:
        return None
    if not raw.get("weights"):
        raise ValueError("model: needs \"weights\", the .pt file")
    w = float(raw.get("weight", DEFAULT_WEIGHT))
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"model: weight is the model's share, 0 to 1 (got {w})")
    cfg = dict(DEFAULT_MODEL)
    cfg.update({k: raw[k] for k in DEFAULT_MODEL if k in raw})
    return {"weights": str(raw["weights"]), "weight": w,
            "device": str(raw.get("device") or ""), "counter": cfg}
