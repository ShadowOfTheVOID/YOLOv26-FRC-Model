"""frc-fms counter plugin: the measured colour crossing counter, per hub camera.

frc-fms (github.com/arnan-bajaj/frc-fms) loads a counter per hub from its
config/vision.yaml and calls `process(frame, t) -> new fuel` on every frame.
Its built-in `zone` counter runs the fuel model on a crop around the hub;
that approach -- model + tracker -- measured 17% (30 fps) to 25% (60 fps)
held-out error on the four scored Einstein matches. This one is
`hubcount.CrossingCounter`, which measured 10.1% on the same four, every
frame at 60 fps, the AUTO winner right on all four (deploy/HUB_FEED.md). It
needs no model, no torch and no GPU, and keeps up with 60 fps on a laptop.

In frc-fms's config/vision.yaml, with this repository on PYTHONPATH:

    defaults:
      counter: "tbavid.fms_counter:ColourCounter"
    hubs:
      red:
        source: 0
        roi: [800, 300, 320, 180]      # pick_roi.py: the funnel mouth
        ball_area: 900                 # run.py hubfeed --measure 5 (see below)
      blue:
        source: 1
        outline: [[810, 300], [1110, 300], [1100, 470], [820, 470]]
        ball_area: 850

Zone keys, any of:
- `roi: [x, y, w, h]` -- frc-fms's own; used as the mouth outline.
- `outline: [[x, y], ...]` -- a funnel-mouth polygon (balls counted moving
  down into it).
- `line: [[x, y], [x, y]]` with `out: [x, y]` -- an exit line, balls counted
  crossing towards `out`.
- `setup: cams.json` with `camera: NAME` -- the outlines, ball size and blur
  of a camera drawn in `run.py hubgui`; that camera's zones for this hub.

`ball_area` -- one ball's pixel area on that camera -- is required unless
`setup` supplies it. Measure it with balls lying near the hub:

    python run.py hubfeed --source 0 --red 810,300,1110,300,1100,470,820,470 --measure 5

Measuring it automatically from the first frames was tried and is refused:
on the Einstein 4 recording those frames are the title card, it measured 98
px against a real 272, and blue counted 176 where the offline count was
about 100. A camera pointed at an empty hub before a match is no better.
Within 0.5x-2x of the true area the error stays within 5 points, because
the counter learns one ball from the crossings anyway.

Optional: `blur` (0.3), `remove_static` (false), `fps` (30).

The colour + model combo (tbavid/hubmodel.py: 6.8% mean error on the four
Einstein matches against 10.1% for colour alone; 7-9% expected on a match
nobody tuned on) is the same plugin with a model added:

    hubs:
      red:
        counter: "tbavid.fms_counter:ComboCounter"
        model: models/fuel_relabel.pt
        model_weight: 0.5          # the model's share; 0.5 is the tuned value
        device: mps                # optional: cuda / mps / cpu, else the best found
        outline: [[810, 300], [1110, 300], [1100, 470], [820, 470]]
        ball_area: 900

`ColourCounter` with `model:` is the same thing; `ComboCounter` refuses to
start without one. The model runs on its own thread, so frc-fms's camera
loop never waits for it, and a model that cannot keep up (more than 20% of
its frames skipped after 10 s) is dropped and the hub counts by colour --
the console says so. Under rescore.py, which reads a recording faster than
real time, the plugin notices and waits for the model on every frame
instead, so a re-count skips nothing. Exit lines stay colour only. Needs Ultralytics in the
environment frc-fms's vision runs in, and a GPU or Apple silicon.

Like every frc-fms counter it returns NEW fuel for the frame; frc-fms
timestamps and buckets them.
"""
from __future__ import annotations

from typing import List, Optional

from . import hubcount as HC


class ColourCounter:
    """frc-fms counter interface: __init__(cfg), process(frame, t), total, draw."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hub = cfg.get("hub", "?")
        self.roi = cfg.get("roi")
        self.total = 0
        self.ball_area: Optional[float] = (float(cfg["ball_area"])
                                           if cfg.get("ball_area") else None)
        self.blur = float(cfg.get("blur", HC.DEFAULT_BLUR))
        self.remove_static = bool(cfg.get("remove_static", False))
        self.fps = float(cfg.get("fps", 30) or 30)
        self.zones = self._zone_specs(cfg)
        if not self.zones:
            raise ValueError(f"hub {self.hub}: give roi, outline, line+out, or "
                             f"setup+camera in config/vision.yaml")
        if not self.ball_area:
            raise ValueError(f"hub {self.hub}: set ball_area (one ball's pixel area on "
                             f"this camera) -- measure it with run.py hubfeed --measure 5")
        self.counters: List = []
        self.eyes: List = []
        # The model half (optional): weights, share, and the worker once built.
        from .hubmodel import DEFAULT_WEIGHT, parse_model
        self.model = parse_model({"weights": cfg["model"],
                                  "weight": cfg.get("model_weight", DEFAULT_WEIGHT),
                                  "device": cfg.get("device", "")}) \
            if cfg.get("model") else None
        self.models: List = []          # a ModelCounter per zone, None for exits
        self.worker = None
        self.frames = 0
        self.offline = False
        self._clock0 = None          # (first t, wall clock then)
        self._said_off = False

    def _zone_specs(self, cfg: dict) -> list:
        """[(kind, points, out)] from whichever keys the hub gives."""
        if cfg.get("setup"):
            setup = HC.load_setup(cfg["setup"])
            cams = [c for c in setup.cameras if c.name == cfg.get("camera")] \
                if cfg.get("camera") else setup.cameras
            if len(cams) != 1:
                raise ValueError(f"hub {self.hub}: name the camera in {cfg['setup']}: "
                                 + ", ".join(c.name for c in setup.cameras))
            cam = cams[0]
            if not cfg.get("ball_area") and cam.ball_area > 0:
                self.ball_area = cam.ball_area
            if "blur" not in cfg:
                self.blur = cam.blur
            if "remove_static" not in cfg:
                self.remove_static = cam.remove_static
            return [("exit", list(z.line), z.out) if z.line else
                    ("outline", list(z.outline or []), None)
                    for z in cam.zones if z.hub == self.hub]
        if cfg.get("line"):
            return [("exit", [tuple(p) for p in cfg["line"]], tuple(cfg["out"]))]
        if cfg.get("outline"):
            return [("outline", [tuple(p) for p in cfg["outline"]], None)]
        if self.roi:
            x, y, w, h = self.roi
            return [("outline", [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], None)]
        return []

    def _build(self, frame) -> None:
        h, w = frame.shape[:2]
        for kind, pts, out in self.zones:
            if kind == "exit":
                c = HC.ExitLineCounter(pts, out, self.ball_area, self.blur)
            else:
                c = HC.CrossingCounter(pts, self.ball_area, self.blur)
            self.counters.append(c)
            self.eyes.append(HC.ZoneEye(HC.region_of(c.poly, self.ball_area, w, h),
                                        self.remove_static, self.fps))
        if self.model:
            self._build_model()

    def _build_model(self) -> None:
        from .hubmodel import ModelCounter, ModelEye, ModelWorker, model_stride
        self.stride = model_stride(self.fps)
        outlines = {i: c.poly for i, (c, (kind, _, _)) in
                    enumerate(zip(self.counters, self.zones)) if kind == "outline"}
        self.models = [ModelCounter(outlines[i], self.fps / self.stride,
                                    **self.model["counter"]) if i in outlines else None
                       for i in range(len(self.counters))]
        eye = ModelEye(self.model["weights"], outlines, self.model["device"])
        self.worker = ModelWorker(
            eye, {i: m for i, m in enumerate(self.models) if m is not None},
            on_rise=lambda i, tag: None,       # read back in process()
            on_off=lambda offered, skipped, tag: print(
                f"[tbavid] hub {self.hub}: the model skipped {skipped} of "
                f"{offered} frames -- turned off, counting by colour only. "
                f"Use Apple silicon or a GPU.", flush=True),
            on_error=lambda e: print(f"[tbavid] hub {self.hub}: model failed, "
                                     f"counting by colour only: {e}", flush=True),
            name=f"tbavid model {self.hub}")

    def _is_offline(self, t: float) -> bool:
        """frc-fms's rescore.py feeds a recording as fast as it decodes, with
        the recording's own timestamps. Live, frames come at camera speed and
        the model may skip some; offline it must not, or it would skip nearly
        every frame and drop itself, so once video time runs well ahead of
        the clock (1.5x over 2 s) the plugin waits for the model on each frame,
        as run.py hubcount --model does."""
        import time
        if self.offline:
            return True
        now = time.monotonic()
        if self._clock0 is None:
            self._clock0 = (t, now)
            return False
        dt, dw = t - self._clock0[0], now - self._clock0[1]
        if dt >= 2.0 and dt > 1.5 * dw:
            self.offline = True
        return self.offline

    def _zone_count(self, i: int) -> int:
        from .hubmodel import blend
        colour = self.counters[i].reported
        m = self.models[i] if self.models else None
        if m is None or self.worker is None or self.worker.off or self.worker.error:
            return colour
        return blend(colour, m.reported, self.model["weight"])

    def process(self, frame, t: float) -> int:
        if not self.counters:
            self._build(frame)
        for c, eye in zip(self.counters, self.eyes):
            c.update(eye.blobs(frame))
        if self.worker is not None and self.frames % self.stride == 0:
            self.worker.offer(frame, t)
            if self._is_offline(t):
                self.worker.wait_idle()
        self.frames += 1
        # The blend can sit below what was already reported (when a model
        # that fell behind is dropped and colour alone is lower); frc-fms adds
        # events, so nothing is taken back -- new fuel waits until the count
        # passes the total again.
        now = sum(self._zone_count(i) for i in range(len(self.counters)))
        new = max(0, now - self.total)
        self.total += new
        return new

    def draw(self, frame):
        import cv2
        import numpy as np
        for c in self.counters:
            pts = np.array(c.poly, np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        label = f"{self.hub}: {self.total}"
        if self.worker is not None:
            colour = sum(c.reported for c in self.counters)
            model = sum(m.reported for m in self.models if m is not None)
            label += (f"  (colour {colour}, model off)" if self.worker.off or self.worker.error
                      else f"  (colour {colour}, model {model})")
        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        return frame


class ComboCounter(ColourCounter):
    """ColourCounter with the fuel model blended in; `model:` is required."""

    def __init__(self, cfg: dict):
        if not cfg.get("model"):
            raise ValueError(f"hub {cfg.get('hub', '?')}: ComboCounter needs "
                             f"model: path/to/fuel_relabel.pt in config/vision.yaml")
        super().__init__(cfg)
