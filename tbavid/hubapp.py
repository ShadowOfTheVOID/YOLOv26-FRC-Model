"""Everything the hub counter's user interfaces do, with no user interface.

The web page (`hubweb.py`, standard library only, opened in any browser)
holds no logic of its own: adding cameras and streams, drawing outlines,
measuring, calibrating, starting the feed and reading its live state all
happen here, where they are tested without a browser. (A Qt window sat on
this too until the user chose the website.)

It edits the same setup dictionary `run.py hubfeed --setup cams.json` runs,
so a setup made in the page also runs headless. OpenCV is imported
only inside functions: the helpers are tested in CI without it.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .camprofile import STANDARD_COLOUR
from .hubcount import DEFAULT_BLUR
from .hubmodel import BUILTIN, DEFAULT_WEIGHT, bundled_model

HUBS = ("red", "blue")
COMBINE = ("sum", "max", "median")
DEFAULT_TARGET = "10.0.100.5:8411"
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")


# -- pure helpers ------------------------------------------------------------

def fit_scale(w: int, h: int, max_w: int = 960, max_h: int = 540) -> float:
    """The factor that fits a w x h frame in a view, never enlarging."""
    if w <= 0 or h <= 0:
        return 1.0
    return min(1.0, max_w / w, max_h / h)


def to_frame(x: float, y: float, scale: float) -> Tuple[float, float]:
    """View pixels -> frame pixels. Outlines are stored in the frame's own
    pixels, so a setup drawn on a scaled preview runs on the full frame."""
    return round(x / scale, 1), round(y / scale, 1)


def to_canvas(pts: Sequence[Sequence[float]], scale: float) -> List[float]:
    return [v * scale for p in pts for v in p]


def is_file_source(source: str) -> bool:
    from .hubcount import is_stream_page
    s = str(source)
    return not s.isdigit() and "://" not in s and not is_stream_page(s)


def source_kind(source: str) -> str:
    """'camera' (USB / built-in, or an iPhone over Continuity Camera),
    'wireless' (rtsp:// or http:// camera), 'stream' (Twitch / YouTube page)
    or 'file'."""
    from .hubcount import is_network_camera, is_stream_page
    s = str(source)
    if s.isdigit():
        return "camera"
    if is_stream_page(s):
        return "stream"
    if is_network_camera(s):
        return "wireless"
    return "file" if is_file_source(s) else "camera"


def redact(source: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://***@host/... for logs: camera
    passwords sit in the address and the activity log is on screen."""
    s = str(source)
    if "://" in s and "@" in s.split("://", 1)[1].split("/", 1)[0]:
        scheme, rest = s.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return s


WIRELESS_WARNING = ("Wi-Fi adds delay and drops out, so check the picture "
                    "keeps up before trusting it. Use your own router, not the "
                    "field's Wi-Fi, and check the event allows it: official "
                    "FRC events ban personal Wi-Fi networks.")


STREAM_WARNING = ("A stream runs several seconds behind the field (Twitch's "
                  "delay), far past the 200 ms bioarena's AUTO call allows. "
                  "Use it for practice and scouting, not to decide AUTO.")


def next_name(existing: Sequence[str], base: str) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}{i}" in existing:
        i += 1
    return f"{base}{i}"


def new_setup() -> Dict:
    from .hubcount import RULES
    return {"rules": RULES, "combine": {"red": "sum", "blue": "sum"}, "cameras": []}


def zone_points(z: Dict) -> List:
    """The points a zone is drawn and searched around, either kind."""
    if z.get("line"):
        return list(z["line"]) + [z["out"]]
    return list(z.get("outline") or [])


def problems(cfg: Dict) -> List[str]:
    """What stops this setup from running, in words a person can act on."""
    out = []
    cams = cfg.get("cameras") or []
    if not cams:
        return ["Add a camera (or a recording) first."]
    for c in cams:
        if not c.get("zones"):
            out.append(f"{c['name']}: draw at least one hub outline.")
        if not float(c.get("ball_area") or 0) > 0:
            out.append(f"{c['name']}: measure the ball size "
                       f"(put a few balls near the hub, then Measure).")
        m = c.get("model") or {}
        if m.get("weights") == BUILTIN and bundled_model() is None:
            out.append(f"{c['name']}: this copy has no built-in fuel model "
                       f"-- choose the .pt file, or turn the model off.")
        elif m.get("weights") and m["weights"] != BUILTIN and not os.path.exists(m["weights"]):
            out.append(f"{c['name']}: the fuel model {m['weights']} is not there "
                       f"-- pick it again, or turn the model off.")
    if any((c.get("model") or {}).get("weights") for c in cams) and not _has_ultralytics():
        # Found here, not at Start: the counter would open, run a moment
        # and die in its camera thread, which reads as a camera fault.
        out.append("The fuel model needs Ultralytics: start this page from "
                   ".venv-train (run.py hubgui), or turn the model off.")
    return out


def _has_ultralytics() -> bool:
    import importlib.util
    return importlib.util.find_spec("ultralytics") is not None


def list_dir(path: str) -> Dict:
    """Folders and videos in `path`, for picking a recording from a page
    that cannot see the disk itself."""
    path = os.path.abspath(os.path.expanduser(path or "."))
    dirs, videos, models = [], [], []
    try:
        for name in sorted(os.listdir(path), key=str.lower):
            if name.startswith("."):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                dirs.append(name)
            elif name.lower().endswith(VIDEO_EXT):
                videos.append(name)
            elif name.lower().endswith(".pt"):
                models.append(name)
    except OSError as e:
        return {"path": path, "parent": os.path.dirname(path), "dirs": [],
                "videos": [], "models": [], "error": str(e)}
    return {"path": path, "parent": os.path.dirname(path), "dirs": dirs,
            "videos": videos, "models": models}


# -- camera access (OpenCV, imported lazily) ----------------------------------

def probe_cameras(max_index: int = 6) -> List[Dict]:
    """Which camera numbers open, and at what size. Numbers can change when
    cameras are re-plugged, so the interfaces always show a picture."""
    import cv2
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                found.append({"index": i, "size": f"{w}x{h}"})
        cap.release()
    return found


def grab_frame(source: str, fps: float = 0.0, size: str = "",
               at_s: Optional[float] = None, image: Optional[Dict] = None,
               out: Callable[[str], None] = print):
    """One frame to draw on. A recording is read at `at_s`, or a third of the
    way in: a broadcast opens on a title card (Einstein 4 at 3 s showed only
    "EINSTEIN PLAYOFFS", and outlines were drawn on it). A camera is read a
    few times so exposure has settled."""
    import cv2
    from .hubcount import open_source

    if is_file_source(source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"could not open {source}")
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        f = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pos = int(n // 3) if at_s is None else int(at_s * f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos, max(0, int(n) - 1)))
        ok, frame = cap.read()
        cap.release()
    else:
        cap = open_source(source, fps, size, image, out)
        frame = None
        for _ in range(10):
            ok, fr = cap.read()
            if ok:
                frame = fr
        cap.release()
        ok = frame is not None
    if not ok:
        raise ValueError(f"no frame from {source}")
    return frame


def fuel_overlay(frame, colour: Optional[Dict] = None):
    """The picture with every pixel the colour gate takes as fuel painted
    magenta: what a colour setting does is judged on the camera's own
    picture, not on numbers. (Magenta: the one colour a field has none of.)"""
    import cv2
    import numpy as np
    from .camprofile import gate

    lo, hi = gate(colour)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
    out = frame.copy()
    out[m] = (0.35 * frame[m] + 0.65 * np.array((255, 0, 255))).astype(np.uint8)
    return out


def encode(frame, scale: float = 1.0, fmt: str = ".jpg") -> bytes:
    import cv2
    img = frame
    if scale != 1.0:
        img = cv2.resize(frame, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, 80] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, img, params)
    return buf.tobytes()


# -- the controller -------------------------------------------------------------

class Recorder:
    """One camera to an .mp4 while it counts. Frames are written on a thread
    of their own and dropped (and counted) if the disk falls behind, so a
    recording can never slow the counter, which must keep inside the feed's
    latency budget."""

    def __init__(self, path: str, fps: float, queue_frames: int = 120):
        import queue
        self.path, self.fps = path, max(1.0, float(fps))
        self.frames = self.dropped = 0
        self.q = queue.Queue(maxsize=queue_frames)
        self.writer = None
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"record {path}")
        self.thread.start()

    def offer(self, frame) -> None:
        try:
            self.q.put_nowait(frame)
        except Exception:
            self.dropped += 1

    def _run(self) -> None:
        import cv2
        while True:
            frame = self.q.get()
            if frame is None:
                break
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                              self.fps, (w, h))
            self.writer.write(frame)
            self.frames += 1
        if self.writer is not None:
            self.writer.release()

    def close(self) -> None:
        try:
            self.q.put(None, timeout=30)
        except Exception:
            pass
        self.thread.join(timeout=30)


class HubController:
    """The setup being edited, the jobs running on it, and the live feed.

    Thread-safe: front ends call it from their UI thread while cameras,
    measuring and calibration run on workers. Long jobs go through `job`,
    which runs one at a time and records its outcome in `job_state` so a
    page that polls and a window that gets a callback see the same thing.
    """

    def __init__(self, setup_path: Optional[str] = None):
        self.lock = threading.RLock()
        self.cfg = new_setup()
        self.path = setup_path
        self.frames: Dict[str, object] = {}         # setup-time pictures
        self.live_frames: Dict[str, object] = {}    # latest while running
        self.log: deque = deque(maxlen=400)
        self._log_id = 0
        self.job_state: Dict = {"name": None, "running": False, "error": None,
                                "result": None, "progress": None, "id": 0}
        self.running = False
        self.sender = None
        self.monitor: Dict = {}
        self.stop_evt: Optional[threading.Event] = None
        self.listen_stop: Optional[threading.Event] = None
        self.feed_target = ""
        # What the page's address box starts with. The Watchtower app sets
        # it to its own server (http://KEY@127.0.0.1:8000), so counts go
        # there without anyone typing the vision key.
        self.default_target = ""
        # The Hub Counter app's one-click updater (tbavid/appupdate.py), shown
        # on the page; None from a checkout and inside Watchtower, whose own
        # Home offers its updates.
        self.updater = None
        # Where saved camera presets live; None = beside the setup file. The
        # Watchtower app points it at the Hub Counter app's file, so a preset
        # saved in either app is offered in both.
        self.presets_file: Optional[str] = None
        # The spec's acceptance test, live: counts when it was started.
        self.ball_test: Optional[Dict] = None
        # Recording the cameras while counting (Recorder per camera).
        self.recorders: Dict[str, "Recorder"] = {}
        if setup_path and os.path.exists(setup_path):
            self.load(setup_path)

    # -- messages -----------------------------------------------------------
    def say(self, text: str) -> None:
        with self.lock:
            self._log_id += 1
            self.log.append((self._log_id, time.strftime("%H:%M:%S"), str(text)))

    def messages_since(self, last_id: int = 0) -> List[Tuple[int, str, str]]:
        with self.lock:
            return [m for m in self.log if m[0] > last_id]

    # -- long jobs -------------------------------------------------------------
    def job(self, name: str, fn: Callable, done: Optional[Callable] = None) -> bool:
        """Run `fn` on a worker. False if another job is still running."""
        with self.lock:
            if self.job_state["running"]:
                self.say(f"busy with {self.job_state['name']} -- wait for it")
                return False
            self.job_state = {"name": name, "running": True, "error": None,
                              "result": None, "progress": None,
                              "id": self.job_state["id"] + 1}

        def work():
            result, error = None, None
            try:
                result = fn()
            except Exception as e:                  # shown, never swallowed
                error = str(e)
                self.say(f"{name} failed: {e}")
            with self.lock:
                self.job_state.update(running=False, error=error,
                                      result=result if _jsonable(result) else None)
            if done:
                done(result, error)
        threading.Thread(target=work, daemon=True).start()
        return True

    def progress(self, frac: float) -> None:
        with self.lock:
            self.job_state["progress"] = round(frac, 3)

    # -- setup file ------------------------------------------------------------
    def load(self, path: str) -> None:
        from .hubcount import load_setup, setup_to_dict
        setup = load_setup(path, measuring=True)
        cfg = setup_to_dict(setup)
        with self.lock:
            self.cfg = cfg
            self.path = path
            self.frames.clear()
        for note in setup.notes:
            self.say(note)
        self.say(f"opened {path}")

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path or os.path.abspath("cams.json")
        with self.lock:
            data = json.dumps(self.cfg, indent=2)
            self.path = path
        with open(path, "w") as fh:
            fh.write(data)
        self.say(f"saved {path}")
        return path

    # -- cameras ----------------------------------------------------------------
    def camera(self, name: str) -> Dict:
        with self.lock:
            for c in self.cfg["cameras"]:
                if c["name"] == name:
                    return c
        raise KeyError(f"no camera {name!r}")

    def add_camera(self, source: str, name: str = "") -> Dict:
        source = str(source).strip()
        if not source:
            raise ValueError("no source given")
        if is_file_source(source) and not os.path.exists(source):
            raise ValueError(f"no such file: {source}")
        with self.lock:
            if any(str(c["source"]) == source for c in self.cfg["cameras"]):
                raise ValueError(f"{source} is already added")
            if not name:
                kind = source_kind(source)
                if kind == "wireless":
                    host = source.split("://", 1)[1].split("/", 1)[0]
                    host = host.rsplit("@", 1)[-1].split(":", 1)[0]
                    name = f"wifi-{host.split('.')[-1] if host.replace('.', '').isdigit() else host}"[:20]
                elif kind == "stream":
                    from .hubcount import stream_name
                    name = stream_name(source)
                elif source.isdigit():
                    name = f"cam{source}"
                else:
                    name = os.path.splitext(os.path.basename(source))[0][:20]
            cam = {"name": next_name([c["name"] for c in self.cfg["cameras"]],
                                     name),
                   "source": source, "ball_area": 0, "blur": DEFAULT_BLUR,
                   "zones": []}
            self.cfg["cameras"].append(cam)
        self.say(f"added {cam['name']} ({redact(source)})")
        if source_kind(source) == "stream":
            self.say(STREAM_WARNING)
        elif source_kind(source) == "wireless":
            self.say(WIRELESS_WARNING)
        return cam

    # -- camera presets (camprofile) -------------------------------------------
    @property
    def presets_path(self) -> str:
        """Saved presets live beside the setup file (~/Documents/Hub Counter
        in the app), so every setup made there shares them."""
        if self.presets_file:
            return self.presets_file
        base = os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()
        return os.path.join(base, "camera-presets.json")

    def presets(self) -> List[Dict]:
        from .camprofile import all_presets
        return all_presets(self.presets_path)

    def apply_preset(self, cam_name: str, pid: str) -> Dict:
        """Size, frame rate, picture and colour from a preset; the outlines
        and ball size stay (re-measure the ball if the size changed)."""
        from .camprofile import find
        p = find(self.presets(), pid)
        with self.lock:
            c = self.camera(cam_name)
            old_size = c.get("size", "")
            c.update(preset=p["id"], fps=p["fps"], size=p["size"],
                     image=dict(p["image"]), colour=dict(p["colour"]))
        self.say(f"{c['name']}: preset {p['label']}")
        if p["size"] and old_size != p["size"] and c.get("ball_area"):
            self.say(f"{c['name']}: the picture size changed, so measure the "
                     f"ball again (and check the outlines)")
        return c

    def save_preset(self, cam_name: str, label: str) -> Dict:
        from .camprofile import save_preset
        with self.lock:
            c = json.loads(json.dumps(self.camera(cam_name)))
        p = save_preset(self.presets_path, label, c)
        with self.lock:
            self.camera(cam_name)["preset"] = p["id"]
        self.say(f"saved preset {p['label']} ({self.presets_path})")
        return p

    def delete_preset(self, pid: str) -> None:
        from .camprofile import delete_preset
        delete_preset(self.presets_path, pid)
        with self.lock:
            for c in self.cfg["cameras"]:
                if c.get("preset") == pid:
                    c.pop("preset")

    def remove_camera(self, name: str) -> None:
        with self.lock:
            self.cfg["cameras"] = [c for c in self.cfg["cameras"]
                                   if c["name"] != name]
            self.frames.pop(name, None)

    def update_camera(self, name: str, fields: Dict) -> Dict:
        """Change a camera's settings; returns it (its name may change)."""
        with self.lock:
            c = self.camera(name)
            new = str(fields.get("name", name)).strip() or name
            if new != name:
                new = next_name([x["name"] for x in self.cfg["cameras"]
                                 if x is not c], new)
                if name in self.frames:
                    self.frames[new] = self.frames.pop(name)
                c["name"] = new
            if "source" in fields and str(fields["source"]).strip():
                c["source"] = str(fields["source"]).strip()
            before = (float(c.get("fps") or 0), str(c.get("size") or ""))
            for k in ("fps", "ball_area"):
                if k in fields:
                    try:
                        c[k] = float(fields[k] or 0)
                    except (TypeError, ValueError):
                        raise ValueError(f"{k} must be a number")
            if "size" in fields:
                c["size"] = str(fields["size"] or "").strip()
            if (float(c.get("fps") or 0), str(c.get("size") or "")) != before:
                c.pop("preset", None)
            if "blur" in fields:
                b = float(fields["blur"] or 0)
                if not 0 <= b <= 1:
                    raise ValueError("blur is 0 to 1")
                c["blur"] = round(b, 2)
            if "remove_static" in fields:
                c["remove_static"] = bool(fields["remove_static"])
            # Picture settings and colour gate (camprofile). Any change makes
            # the camera "custom": its preset no longer describes it.
            from .camprofile import check_colour, check_image
            if "image" in fields:
                c["image"] = check_image(dict(c.get("image") or {}, **(fields["image"] or {})))
                c.pop("preset", None)
            if "colour" in fields:
                c["colour"] = check_colour(dict(c.get("colour") or {}, **(fields["colour"] or {})))
                c.pop("preset", None)
            if fields.get("colour_reset"):
                c.pop("colour", None)
                c.pop("preset", None)
            if "model" in fields:
                # A path to turn the model blend on, "" to turn it off. The
                # counter settings tuned in hubmodel are kept unless a setup
                # file already overrides them.
                w = str(fields["model"] or "").strip()
                if w:
                    c["model"] = dict(c.get("model") or {}, weights=w)
                    c["model"].setdefault("weight", DEFAULT_WEIGHT)
                else:
                    c.pop("model", None)
            if "model_weight" in fields and c.get("model"):
                mw = float(fields["model_weight"])
                if not 0 <= mw <= 1:
                    raise ValueError("the model's share is 0 to 1")
                c["model"]["weight"] = round(mw, 2)
            return c

    def add_zone(self, cam_name: str, hub: str,
                 points: Sequence[Sequence[float]], kind: str = "outline") -> Dict:
        """An outline (3+ corners around the hub's mouth) or an exit line
        (`kind="exit"`: the two ends of the exit, then a point outside it)."""
        from .hubcount import ExitLineCounter
        if hub not in HUBS:
            raise ValueError("hub must be red or blue")
        pts = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in points]
        # a double-click lands as two clicks on the same spot
        while len(pts) >= 2 and pts[-1] == pts[-2]:
            pts.pop()
        if kind == "exit":
            if len(pts) != 3:
                raise ValueError("an exit line is three clicks: its two ends, "
                                 "then a point on the side balls go out to")
            ExitLineCounter(pts[:2], pts[2], 1.0)   # refuses a bad line
            body = {"line": pts[:2], "out": pts[2]}
        elif kind == "outline":
            if len(pts) < 3:
                raise ValueError("an outline needs at least three corners")
            body = {"outline": pts}
        else:
            raise ValueError(f"unknown zone kind {kind!r}")
        with self.lock:
            c = self.camera(cam_name)
            names = [z["name"] for x in self.cfg["cameras"] for z in x["zones"]]
            base = f"{c['name']}-{hub}" + ("-exit" if kind == "exit" else "")
            zone = {"name": next_name(names, base), "hub": hub, **body}
            c["zones"].append(zone)
        what = "exit line" if kind == "exit" else f"outline ({len(pts)} corners)"
        self.say(f"{zone['name']}: {what} saved")
        return zone

    def delete_zone(self, cam_name: str, zone_name: str) -> None:
        with self.lock:
            c = self.camera(cam_name)
            c["zones"] = [z for z in c["zones"] if z["name"] != zone_name]

    def set_confirm(self, hub: str, seconds) -> None:
        """Exits confirm a hub's outline entries after `seconds`; 0 = off
        (hubcount.HubTally). Off by default: only a hand-counted recording
        from the real camera can show it helps (run.py hubcount --hand)."""
        from .hubcount import MAX_CONFIRM_S
        try:
            v = float(seconds or 0)
        except (TypeError, ValueError):
            raise ValueError("the exit delay is a number of seconds") from None
        if hub not in HUBS or not 0 <= v <= MAX_CONFIRM_S:
            raise ValueError(f"the exit delay is 0 (off) to {MAX_CONFIRM_S:g} s")
        with self.lock:
            conf = self.cfg.setdefault("confirm", {})
            if v:
                conf[hub] = round(v, 2)
            else:
                conf.pop(hub, None)
            if not conf:
                self.cfg.pop("confirm", None)

    def set_combine(self, hub: str, how: str) -> None:
        if hub not in HUBS or how not in COMBINE:
            raise ValueError(f"combine is one of {', '.join(COMBINE)}")
        with self.lock:
            self.cfg.setdefault("combine", {})[hub] = how

    # -- pictures ---------------------------------------------------------------
    def grab(self, cam_name: str, at_s: Optional[float] = None):
        c = self.camera(cam_name)
        if self.running and not is_file_source(c["source"]):
            raise ValueError("the camera is in use by the counter; its live "
                             "picture is shown instead")
        frame = grab_frame(c["source"], float(c.get("fps") or 0),
                           c.get("size", ""), at_s, c.get("image"), self.say)
        with self.lock:
            self.frames[c["name"]] = frame
        return frame

    def picture(self, cam_name: str, fuel: bool = False):
        """What to show for a camera: live while running, else its still;
        with `fuel`, the colour gate's pixels painted over it."""
        with self.lock:
            if self.running and cam_name in self.live_frames:
                frame = self.live_frames[cam_name]
            else:
                frame = self.frames.get(cam_name)
            colour = next((c.get("colour") for c in self.cfg["cameras"]
                           if c["name"] == cam_name), None)
        if fuel and frame is not None:
            frame = fuel_overlay(frame, colour)
        return frame

    def measure(self, cam_name: str, seconds: float = 5.0) -> Optional[float]:
        from .hubcount import measure
        c = self.camera(cam_name)
        if not c.get("zones"):
            raise ValueError("draw the hub outline first: the ball is measured "
                             "near it")
        polys = {z["name"]: [tuple(p) for p in zone_points(z)] for z in c["zones"]}
        area = measure(c["source"], polys, seconds, None,
                       float(c.get("fps") or 0), c.get("size", ""), out=self.say,
                       image=c.get("image"), colour=c.get("colour"))
        if area:
            with self.lock:
                c["ball_area"] = round(area)
        return area

    def calibrate(self, cam_name: str, video: str, hand: Dict[str, int]) -> Dict:
        from .hubcount import calibrate, setup_from_dict
        c = self.camera(cam_name)
        issues = problems({"cameras": [c]})
        if issues:
            raise ValueError(" ".join(issues))
        cam = setup_from_dict({"cameras": [dict(c, blur=0, remove_static=False)]}
                              ).cameras[0]
        r = calibrate(cam, video, hand, out=self.say, progress=self.progress)
        best, base = r["best"], r["uncorrected"]
        return {"camera": c["name"], "blur": best["blur"],
                "remove_static": best["remove_static"],
                "counts": best["counts"], "error": best["error"],
                "uncorrected": base["counts"], "uncorrected_error": base["error"],
                "hand": hand}

    # -- the feed ------------------------------------------------------------------
    def start(self, target: str = DEFAULT_TARGET, practice: bool = False,
              realtime: bool = True, log_csv: bool = True) -> None:
        from . import hubcount, hubfeed
        if self.running:
            raise ValueError("already running")
        with self.lock:
            cfg = json.loads(json.dumps(self.cfg))
        issues = problems(cfg)
        if issues:
            raise ValueError("Not ready yet: " + " ".join(issues))
        setup = hubcount.setup_from_dict(cfg)
        from .fmslink import FmsSender, is_fms_target, split_target
        fms = is_fms_target(target)
        if fms and practice:
            self.say("practice mode is for bioarena's UDP feed; sending to frc-fms instead")
            practice = False
        host, port = ("", 0) if fms else hubfeed.parse_target(target.strip() or DEFAULT_TARGET)
        self.practice = practice
        if practice:
            host = "127.0.0.1"
            self.listen_stop = threading.Event()
            threading.Thread(target=hubfeed.listen, daemon=True,
                             kwargs=dict(port=port, bind="127.0.0.1",
                                         out=lambda t: self.say(f"[test receiver] {t}"),
                                         stop=self.listen_stop)).start()
        log_path = None
        if log_csv:
            base = os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()
            log_path = os.path.join(base, time.strftime("hubfeed_%Y%m%d_%H%M%S.csv"))
            self.say(f"logging counts to {log_path}")
        if fms:
            # frc-fms: timestamped events over HTTP (tbavid/fmslink.py). The
            # key is in the URL; it is not shown or logged.
            self.sender = FmsSender(target.strip())
            self.feed_target = split_target(target)[0]
        else:
            self.sender = hubfeed.FeedSender((host, port))
            self.feed_target = f"{host}:{port}"
        self.stop_evt = threading.Event()
        self.monitor = {}
        realtime = realtime and any(source_kind(c.source) in ("file", "stream")
                                    for c in setup.cameras)
        if any(source_kind(c.source) == "stream" for c in setup.cameras):
            self.say("counting from a stream: " + STREAM_WARNING)

        def on_frame(name, frame):
            with self.lock:
                self.live_frames[name] = frame
                rec = self.recorders.get(name)
            if rec is not None:
                rec.offer(frame)

        def work():
            try:
                hubcount.run(self.sender, setup, realtime=realtime,
                             log_path=log_path, out=self.say, stop=self.stop_evt,
                             on_frame=on_frame, monitor=self.monitor)
            except Exception as e:
                self.say(f"error: {e}")
            finally:
                self.stop_recording()
                with self.lock:
                    self.running = False
                    self.live_frames.clear()
                    self.ball_test = None
                if self.listen_stop:
                    self.listen_stop.set()
                    self.listen_stop = None
                s = self.sender
                if hasattr(s, "close"):
                    s.close()          # frc-fms: deliver what is still queued
                    if s.pending():
                        self.say(f"! {s.pending()} fuel events never reached frc-fms: "
                                 f"{s.last_error}")
                self.say(f"stopped: red {s.counts['red']}, blue {s.counts['blue']}")
        with self.lock:
            self.running = True
        self.say(f"session {self.sender.session}: sending to {self.feed_target}")
        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        if self.stop_evt:
            self.stop_evt.set()

    # -- tests the page can run (the practice field, 2026-10-09) ---------------
    def start_ball_test(self) -> Dict:
        """The spec's field acceptance (section 9): drop balls in by hand,
        then compare. Counts from now are what the test sees."""
        if not self.running or self.sender is None:
            raise ValueError("start counting first, then start the ball test")
        with self.lock:
            self.ball_test = {"base": dict(self.sender.counts), "at": time.time()}
        self.say("ball test started: drop the balls in, count them by hand, then Check")
        return self.ball_test

    def check_ball_test(self, hand: Dict[str, int]) -> Dict:
        with self.lock:
            bt = self.ball_test
        if bt is None or self.sender is None:
            raise ValueError("no ball test running")
        res = {}
        for hub, n in hand.items():
            if hub not in HUBS or n in (None, ""):
                continue
            seen = self.sender.counts[hub] - bt["base"][hub]
            res[hub] = {"hand": int(n), "counted": seen, "off": seen - int(n)}
            self.say(f"ball test {hub}: counted {seen}, by hand {int(n)} -> "
                     + ("match" if seen == int(n) else f"off by {seen - int(n):+d}"))
        return {"hubs": res, "pass": bool(res) and all(r["off"] == 0 for r in res.values())}

    def start_recording(self) -> List[str]:
        """Record every camera while it counts, to test with afterwards
        (Test a recording). Files go beside the setup, in recordings/."""
        if not self.running:
            raise ValueError("start counting first: the recording is what the counter sees")
        base = os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()
        folder = os.path.join(base, "recordings")
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        health = self.monitor.get("health") or {}
        made = []
        with self.lock:
            for c in self.cfg["cameras"]:
                if c["name"] in self.recorders:
                    continue
                fps = health[c["name"]].fps() if c["name"] in health else 0
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in c["name"])
                path = os.path.join(folder, f"{safe}_{stamp}.mp4")
                self.recorders[c["name"]] = Recorder(path, fps or float(c.get("fps") or 30))
                made.append(path)
        for m in made:
            self.say(f"recording {m}")
        return made

    def stop_recording(self) -> List[str]:
        with self.lock:
            recs, self.recorders = self.recorders, {}
        done = []
        for r in recs.values():
            r.close()
            if r.frames:
                done.append(r.path)
                self.say(f"recorded {r.frames} frames to {r.path}"
                         + (f" ({r.dropped} dropped: the disk could not keep up)" if r.dropped else ""))
        return done

    def test_recording(self, cam_name: str, video: str, hand: Dict[str, int]) -> Dict:
        """Every way of counting this camera's zones, on a recording, against
        a hand count (hubcount.hand_test)."""
        from .hubcount import hand_test, setup_from_dict
        with self.lock:
            cfg = json.loads(json.dumps(self.cfg))
        c = next((x for x in cfg["cameras"] if x["name"] == cam_name), None)
        if c is None:
            raise ValueError(f"no camera {cam_name!r}")
        issues = problems({"cameras": [c]})
        if issues:
            raise ValueError(" ".join(issues))
        setup = setup_from_dict({"combine": cfg.get("combine"), "confirm": cfg.get("confirm"),
                                 "cameras": [c]})
        r = hand_test(setup, cam_name, video, hand, progress=self.progress)
        for hub, h in r["hubs"].items():
            best = next((x for x in h["rows"] if x.get("best")), None)
            if best:
                self.say(f"test {hub}: closest is {best['label']} ({best['count']}, "
                         f"{best['error']:+d} against {h['hand']} by hand)")
        return r

    # Set by the web server: its Share (the page on Wi-Fi, behind a PIN).
    share = None

    # Set by the web server: shuts it down. A double-clicked app has no
    # terminal to press Ctrl-C in, so the page's Quit button is the way out.
    on_quit: Optional[Callable[[], None]] = None

    def quit(self) -> None:
        """Stop counting, then end the server (after this reply is sent)."""
        self.stop()
        if self.on_quit:
            self.on_quit()

    # -- what the front ends draw -------------------------------------------------
    def state(self, last_log: int = 0) -> Dict:
        """Everything a front end shows, as plain JSON-able data."""
        with self.lock:
            cfg = json.loads(json.dumps(self.cfg))
            job = dict(self.job_state)
            running = self.running
            pictures = {n: tuple(int(v) for v in f.shape[1::-1])
                        for n, f in {**self.frames, **self.live_frames}.items()}
        kinds = {c["name"]: source_kind(c["source"]) for c in cfg["cameras"]}
        out = {"cfg": cfg, "path": self.path, "running": running, "job": job,
               "kinds": kinds, "stream_warning": STREAM_WARNING,
               "wireless_warning": WIRELESS_WARNING,
               "problems": problems(cfg), "pictures": pictures,
               "builtin_model": bundled_model() is not None,
               "default_target": self.default_target,
               "update": self.updater.status() if self.updater else None,
               "presets": self.presets(),
               "standard_colour": dict(STANDARD_COLOUR),
               "log": self.messages_since(last_log)}
        s = self.sender
        live = {"counts": {"red": 0, "blue": 0}, "linked": False, "reply": None,
                "rtt_ms": None, "cameras": {}, "stale": [], "zones": {},
                "session": None, "target": self.feed_target,
                "errors": list(self.monitor.get("errors", []))}
        if s is not None:
            live.update(counts=dict(s.counts), linked=s.linked(),
                        reply=s.last_reply, rtt_ms=s.rtt_ms, session=s.session)
        health = self.monitor.get("health") or {}
        now = time.monotonic()
        for n, h in health.items():
            live["cameras"][n] = {"fps": round(h.fps(), 1),
                                  "lag_ms": round(h.lag_ms())}
            if running and now - h.last_frame > 0.5:
                live["stale"].append(n)
        live["slow"] = slow_cameras(health) if running else {}
        tally = self.monitor.get("tally")
        if tally is not None:
            live["zones"] = {z.name: z.reported
                             for zs in tally.zones.values() for z in zs}
            # Both halves of a blended zone, so a page can show which one
            # moved: a model half stuck at 0 is a model that is not running.
            live["parts"] = {z.name: {"colour": z.counter.reported,
                                      "model": z.model.reported}
                             for zs in tally.zones.values() for z in zs
                             if z.model is not None}
        if tally is not None:
            # Hubs with an outline and an exit line: the exits against the
            # outline a moment ago (hubcount.HubTally.check).
            live["check"] = {h: c for h in HUBS for c in [tally.check(h)] if c}
        from .hubcount import latency_summary
        live["latency"] = latency_summary(list(self.monitor.get("latency") or []))
        with self.lock:
            bt = self.ball_test
            live["recording"] = [r.path for r in self.recorders.values()]
        if bt is not None and s is not None:
            live["ball_test"] = {h: s.counts[h] - bt["base"][h] for h in HUBS}
        live["model_behind"] = dict(self.monitor.get("model_dropped") or {})
        live["model_off"] = list(self.monitor.get("model_off") or [])
        out["live"] = live
        return out

    def board(self) -> Dict:
        """The scoreboard's data: small, since the page asks ten times a second."""
        with self.lock:
            running = self.running
        s = self.sender
        health = self.monitor.get("health") or {}
        now = time.monotonic()
        stale = [n for n, h in health.items()
                 if running and now - h.last_frame > 0.5]
        lags = [h.lag_ms() for h in health.values()]
        from .fmslink import FmsSender
        if isinstance(s, FmsSender):
            v = board_view(dict(s.counts), running, False, None, stale,
                           round(max(lags)) if lags else None,
                           list(self.monitor.get("errors", [])), False,
                           slow_cameras(health) if running else {})
            v["source"], v["linked"] = "fms", s.linked()
            if running and not s.linked() and "alert" not in v:
                v["alert"] = f"frc-fms not answering: {s.last_error or 'no reply yet'}"
            return v
        return board_view(dict(s.counts) if s else {}, running,
                          bool(s and s.linked()), s.last_reply if s else None,
                          stale, round(max(lags)) if lags else None,
                          list(self.monitor.get("errors", [])),
                          bool(getattr(self, "practice", False)),
                          slow_cameras(health) if running else {})


BOARD_REPLY_KEYS = ("match_state", "match_time_s", "shift", "hub_active",
                    "match_count", "credited", "auto_count")


def slow_cameras(health: Dict) -> Dict[str, float]:
    """Cameras delivering under hubcount.MIN_FPS, once a second of frames is in."""
    from .hubcount import MIN_FPS
    out = {}
    for n, h in health.items():
        fps = h.fps()
        if len(h.stamps) >= 30 and 0 < fps < MIN_FPS:
            out[n] = round(fps, 1)
    return out


def board_view(counts: Dict[str, int], running: bool, linked: bool,
               reply: Optional[Dict], stale: List[str], lag_ms: Optional[float],
               errors: List[str], practice: bool = False,
               slow: Optional[Dict[str, float]] = None) -> Dict:
    """What the scoreboard shows, decided here so the page only draws it.

    Linked to bioarena, the big numbers are its `credited` -- the score, this
    match, only fuel scored while that hub was active (spec 4.4) -- with its
    match clock and shift. Otherwise they are the counter's own totals since
    it started, which are not a match score: nothing here knows when a match
    starts, and a dark hub's fuel is in them. The page says which it is.
    """
    out = {"running": running, "linked": linked, "stale": list(stale),
           "lag_ms": lag_ms, "errors": errors[-3:],
           "raw": {h: int(counts.get(h, 0)) for h in HUBS}}
    r = reply if (linked and isinstance(reply, dict)) else None
    if r is not None and isinstance(r.get("credited"), dict):
        # Practice replies come from the built-in stand-in, not the field.
        out["source"] = "practice" if practice else "bioarena"
        out["score"] = {h: int(r["credited"].get(h, 0) or 0) for h in HUBS}
        for k in BOARD_REPLY_KEYS:
            if k in r:
                out[k] = r[k]
    else:
        out["source"] = "counter"
        out["score"] = dict(out["raw"])
    if not running:
        out["alert"] = "counter stopped"
    elif stale:
        out["alert"] = "no picture from " + ", ".join(stale)
    elif slow:
        out["alert"] = ("slow camera: " + ", ".join(f"{n} {f:.0f} fps" for n, f in slow.items())
                        + " -- counts run low under 30 fps; add light or lower the resolution")
    elif not linked and r is None and errors:
        out["alert"] = errors[-1]
    return out


def _jsonable(x) -> bool:
    try:
        json.dumps(x)
        return True
    except (TypeError, ValueError):
        return False
