"""A window for the hub counter: cameras, hub outlines, calibration, the feed.

    python3 run.py hubgui                 # or double-click hubfeed.command

Everything `run.py hubfeed` does from flags and a JSON file, done by pointing
and clicking, because the people setting this up at a scrimmage are standing
next to a field with a laptop, not reading a runbook:

  1. Add each camera (or a recording, to practise on) and grab a frame.
  2. Draw each hub's outline by clicking its corners on that frame.
  3. Measure one ball with a few balls sitting near the hub.
  4. Optionally calibrate against a recording you counted by hand.
  5. Start. The counts, the bioarena link and each camera's frame rate are
     shown live over the camera picture.

The window edits the same setup file `run.py hubfeed --setup` reads, so a
setup made here runs headless too. Tk ships with Python; frames are shown by
encoding them as PNG, so there is no Pillow dependency. tkinter and OpenCV
are imported only when the window opens, so the pure helpers below can be
tested without either.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

HUB_COLOURS = {"red": "#e53935", "blue": "#1e88e5"}
CANVAS_W, CANVAS_H = 960, 540
DEFAULT_TARGET = "10.0.100.5:8411"


# -- pure helpers (tested without a display) -----------------------------

def fit_scale(w: int, h: int, max_w: int = CANVAS_W,
              max_h: int = CANVAS_H) -> float:
    """The factor that fits a w x h frame in the canvas, never enlarging."""
    if w <= 0 or h <= 0:
        return 1.0
    return min(1.0, max_w / w, max_h / h)


def to_frame(x: float, y: float, scale: float) -> Tuple[float, float]:
    """Canvas pixels -> frame pixels. Outlines are stored in the frame's own
    pixels, so a setup made on a scaled preview runs on the full frame."""
    return round(x / scale, 1), round(y / scale, 1)


def to_canvas(pts: Sequence[Sequence[float]], scale: float) -> List[float]:
    return [v * scale for p in pts for v in p]


def is_file_source(source: str) -> bool:
    s = str(source)
    return not s.isdigit() and "://" not in s


def next_name(existing: Sequence[str], base: str) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}{i}" in existing:
        i += 1
    return f"{base}{i}"


def new_setup() -> Dict:
    return {"combine": {"red": "sum", "blue": "sum"}, "cameras": []}


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
    return out


def probe_cameras(max_index: int = 6) -> List[Tuple[int, str]]:
    """Which camera numbers open, and at what size. Numbers can change when
    cameras are re-plugged, so the window always shows a picture to check."""
    import cv2
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                found.append((i, f"{w}x{h}"))
        cap.release()
    return found


def grab_frame(source: str, fps: float = 0.0, size: str = "",
               at_s: Optional[float] = None):
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
        cap = open_source(source, fps, size)
        frame, ok = None, False
        for _ in range(10):
            ok, fr = cap.read()
            if ok:
                frame = fr
        cap.release()
        ok = frame is not None
    if not ok:
        raise ValueError(f"no frame from {source}")
    return frame


def png_data(frame, scale: float) -> bytes:
    """A frame scaled for the canvas, as base64 PNG for tk.PhotoImage."""
    import base64

    import cv2
    img = frame
    if scale != 1.0:
        img = cv2.resize(frame, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes())


# -- the window ------------------------------------------------------------

class HubApp:
    """Composition, not inheritance, so importing this module needs no Tk."""

    def __init__(self, setup_path: Optional[str] = None):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("Hub FUEL counter")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.q: "queue.Queue" = queue.Queue()
        self.cfg = new_setup()
        self.path = setup_path
        self.frames: Dict[str, object] = {}      # camera name -> last frame
        self.scale = 1.0
        self.photo = None
        self.drawing: Optional[Dict] = None      # {"hub":..., "pts":[...]}
        self.running = False
        self.stop_evt: Optional[threading.Event] = None
        self.listen_stop: Optional[threading.Event] = None
        self.sender = None
        self.monitor: Dict = {}
        self.live_frames: Dict[str, object] = {}
        self.live_lock = threading.Lock()
        self.busy = False
        self.current: Optional[int] = None       # camera the form is showing

        self._build()
        if setup_path and os.path.exists(setup_path):
            self._load(setup_path)
        self.root.after(100, self._poll)

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        root = self.root
        top = ttk.Frame(root, padding=4)
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Open setup…", command=self.open_setup).pack(side="left")
        ttk.Button(top, text="Save", command=self.save).pack(side="left")
        ttk.Button(top, text="Save as…", command=self.save_as).pack(side="left")
        self.path_lbl = ttk.Label(top, text="(unsaved setup)")
        self.path_lbl.pack(side="left", padx=8)

        main = ttk.Frame(root)
        main.pack(side="top", fill="both", expand=True)
        left = ttk.Frame(main, padding=4)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(main, padding=4)
        right.pack(side="right", fill="y")

        # scoreboard over the picture
        score = ttk.Frame(left)
        score.pack(side="top", fill="x")
        self.red_lbl = tk.Label(score, text="RED 0", fg="white",
                                bg=HUB_COLOURS["red"], font=("Helvetica", 28, "bold"),
                                width=10)
        self.red_lbl.pack(side="left", padx=2)
        self.blue_lbl = tk.Label(score, text="BLUE 0", fg="white",
                                 bg=HUB_COLOURS["blue"], font=("Helvetica", 28, "bold"),
                                 width=10)
        self.blue_lbl.pack(side="left", padx=2)
        self.link_lbl = tk.Label(score, text="stopped", font=("Helvetica", 14),
                                 anchor="w", justify="left")
        self.link_lbl.pack(side="left", padx=10, fill="x", expand=True)

        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                bg="#222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="top")
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Double-Button-1>", lambda e: self.finish_outline())
        self.canvas.bind("<Button-3>", lambda e: self.finish_outline())
        self.canvas.bind("<Button-2>", lambda e: self.finish_outline())
        root.bind("<Return>", lambda e: self.finish_outline())
        root.bind("<Escape>", lambda e: self.cancel_outline())
        self.hint = ttk.Label(left, text="Add a camera or a recording to begin.",
                              font=("Helvetica", 13))
        self.hint.pack(side="top", fill="x", pady=2)
        self.log = tk.Text(left, height=8, width=110, state="disabled",
                           font=("Courier", 11))
        self.log.pack(side="top", fill="both", expand=True)

        # cameras
        box = ttk.LabelFrame(right, text="1. Cameras", padding=4)
        box.pack(fill="x")
        self.cam_list = tk.Listbox(box, height=4, exportselection=False, width=34)
        self.cam_list.pack(fill="x")
        self.cam_list.bind("<<ListboxSelect>>", lambda e: self._select_camera())
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Button(row, text="Find cameras", command=self.find_cameras).pack(side="left")
        ttk.Button(row, text="Add recording…", command=self.add_recording).pack(side="left")
        ttk.Button(row, text="Remove", command=self.remove_camera).pack(side="left")

        form = ttk.Frame(box)
        form.pack(fill="x", pady=2)
        self.v = {k: tk.StringVar() for k in ("name", "source", "fps", "size",
                                               "ball_area")}
        for r, (k, label) in enumerate((("name", "Name"), ("source", "Source"),
                                        ("fps", "Frame rate"),
                                        ("size", "Size (WxH)"),
                                        ("ball_area", "One ball (px)"))):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(form, textvariable=self.v[k], width=22)
            e.grid(row=r, column=1, sticky="we")
            e.bind("<FocusOut>", lambda ev: self._store_camera())
            e.bind("<Return>", lambda ev: self._store_camera())
        self.blur = tk.DoubleVar(value=0.0)
        ttk.Label(form, text="Blur correction").grid(row=5, column=0, sticky="w")
        tk.Scale(form, variable=self.blur, from_=0.0, to=1.0, resolution=0.1,
                 orient="horizontal", length=150,
                 command=lambda _: self._store_camera()).grid(row=5, column=1)
        self.static = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Ignore yellow that stays still",
                        variable=self.static,
                        command=self._store_camera).grid(row=6, column=0,
                                                         columnspan=2, sticky="w")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Button(row, text="Show picture", command=self.grab).pack(side="left")
        ttk.Button(row, text="Measure ball", command=self.measure).pack(side="left")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="Recording: picture at").pack(side="left")
        self.seek = tk.StringVar()
        e = ttk.Entry(row, textvariable=self.seek, width=6)
        e.pack(side="left")
        e.bind("<Return>", lambda ev: self.grab())
        ttk.Label(row, text="s (blank = 1/3 in)").pack(side="left")

        # zones
        box = ttk.LabelFrame(right, text="2. Hub outlines", padding=4)
        box.pack(fill="x", pady=4)
        self.zone_list = tk.Listbox(box, height=4, exportselection=False, width=34)
        self.zone_list.pack(fill="x")
        row = ttk.Frame(box)
        row.pack(fill="x")
        tk.Button(row, text="Draw RED", fg=HUB_COLOURS["red"],
                  command=lambda: self.start_outline("red")).pack(side="left")
        tk.Button(row, text="Draw BLUE", fg=HUB_COLOURS["blue"],
                  command=lambda: self.start_outline("blue")).pack(side="left")
        ttk.Button(row, text="Delete", command=self.delete_zone).pack(side="left")
        comb = ttk.Frame(box)
        comb.pack(fill="x", pady=2)
        self.combine = {}
        for i, hub in enumerate(("red", "blue")):
            ttk.Label(comb, text=f"{hub} zones:").grid(row=i, column=0, sticky="w")
            var = tk.StringVar(value="sum")
            ttk.OptionMenu(comb, var, "sum", "sum", "max", "median",
                           command=lambda _v, h=hub: self._store_combine()
                           ).grid(row=i, column=1, sticky="w")
            self.combine[hub] = var
        ttk.Label(box, wraplength=260, foreground="#555",
                  text="sum: cameras see different balls.  max / median: "
                       "several cameras watch the same balls.").pack(fill="x")

        # calibrate
        box = ttk.LabelFrame(right, text="3. Calibrate (optional)", padding=4)
        box.pack(fill="x")
        ttk.Label(box, wraplength=260, foreground="#555",
                  text="Record this camera while balls go in, count them by "
                       "hand, then pick the recording.").pack(fill="x")
        ttk.Button(box, text="Calibrate from recording…",
                   command=self.calibrate).pack(anchor="w")

        # run
        box = ttk.LabelFrame(right, text="4. Run", padding=4)
        box.pack(fill="x", pady=4)
        self.target = tk.StringVar(value=DEFAULT_TARGET)
        ttk.Label(box, text="bioarena address").pack(anchor="w")
        ttk.Entry(box, textvariable=self.target, width=24).pack(anchor="w")
        self.local = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Practice: send to a test receiver here",
                        variable=self.local).pack(anchor="w")
        self.realtime = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Play recordings at real speed",
                        variable=self.realtime).pack(anchor="w")
        self.log_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Save a CSV log of every count",
                        variable=self.log_on).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        self.start_btn = tk.Button(row, text="START", bg="#43a047", fg="white",
                                   font=("Helvetica", 16, "bold"), width=8,
                                   command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(row, text="STOP", font=("Helvetica", 16, "bold"),
                                  width=8, command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

    # -- messages from worker threads ---------------------------------------
    def say(self, text: str) -> None:
        self.q.put(("log", str(text)))

    def _write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._write_log(payload)
                elif kind == "call":
                    payload()
        except queue.Empty:
            pass
        if self.running:
            self._refresh_live()
        self.root.after(100 if self.running else 200, self._poll)

    def _in_thread(self, fn, done=None) -> None:
        if self.busy:
            self.say("busy -- wait for the current job to finish")
            return
        self.busy = True

        def work():
            try:
                result = fn()
                if done:
                    self.q.put(("call", lambda: done(result)))
            except Exception as e:           # shown, never swallowed
                self.say(f"error: {e}")
            finally:
                self.busy = False
        threading.Thread(target=work, daemon=True).start()

    # -- setup file ---------------------------------------------------------
    def open_setup(self) -> None:
        from tkinter import filedialog
        p = filedialog.askopenfilename(filetypes=[("Setup", "*.json")])
        if p:
            self._load(p)

    def _load(self, path: str) -> None:
        from .hubcount import load_setup, setup_to_dict
        try:
            self.cfg = setup_to_dict(load_setup(path, measuring=True))
        except (OSError, ValueError) as e:
            self._error(f"Could not open {path}: {e}")
            return
        self.path = path
        self.path_lbl.configure(text=path)
        for hub, var in self.combine.items():
            var.set(self.cfg["combine"].get(hub, "sum"))
        self._refresh_cameras(select=0)
        self.say(f"opened {path}")

    def save(self) -> None:
        if not self.path:
            return self.save_as()
        self._store_camera()
        with open(self.path, "w") as fh:
            json.dump(self.cfg, fh, indent=2)
        self.path_lbl.configure(text=self.path)
        self.say(f"saved {self.path}")

    def save_as(self) -> None:
        from tkinter import filedialog
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         initialfile="cams.json",
                                         filetypes=[("Setup", "*.json")])
        if p:
            self.path = p
            self.save()

    # -- cameras -----------------------------------------------------------
    def _cam(self) -> Optional[Dict]:
        # The camera the form shows, not whatever the list has selected: a
        # click on another camera fires the form's FocusOut first, and that
        # save must land on the camera being edited.
        if self.current is None or self.current >= len(self.cfg["cameras"]):
            return None
        return self.cfg["cameras"][self.current]

    def _refresh_cameras(self, select: Optional[int] = None) -> None:
        self.cam_list.delete(0, "end")
        for c in self.cfg["cameras"]:
            area = c.get("ball_area") or 0
            self.cam_list.insert("end", f"{c['name']}  ({c['source']})"
                                        f"{'' if area else '  -- measure ball'}")
        if select is not None and self.cfg["cameras"]:
            select = min(select, len(self.cfg["cameras"]) - 1)
            self.cam_list.selection_clear(0, "end")
            self.cam_list.selection_set(select)
            self._select_camera()
        elif not self.cfg["cameras"]:
            self.current = None
            self.zone_list.delete(0, "end")
            self.canvas.delete("all")

    def _select_camera(self) -> None:
        sel = self.cam_list.curselection()
        if not sel:
            return
        self.current = sel[0]
        c = self._cam()
        for k in ("name", "source", "fps", "size", "ball_area"):
            val = c.get(k, "")
            self.v[k].set("" if val in (None, 0, 0.0) else str(val))
        self.blur.set(float(c.get("blur") or 0))
        self.static.set(bool(c.get("remove_static")))
        self._refresh_zones()
        if c["name"] in self.frames:
            self._draw()
        else:
            self.grab()

    def _store_camera(self) -> None:
        c = self._cam()
        if not c:
            return
        name = self.v["name"].get().strip() or c["name"]
        others = [x["name"] for x in self.cfg["cameras"] if x is not c]
        if name != c["name"]:
            name = next_name(others, name)
            if c["name"] in self.frames:
                self.frames[name] = self.frames.pop(c["name"])
        c["name"] = name
        c["source"] = self.v["source"].get().strip() or c["source"]
        for k, cast in (("fps", float), ("ball_area", float)):
            try:
                c[k] = cast(self.v[k].get()) if self.v[k].get().strip() else 0
            except ValueError:
                pass
        c["size"] = self.v["size"].get().strip()
        c["blur"] = round(float(self.blur.get()), 1)
        c["remove_static"] = bool(self.static.get())
        self._refresh_cameras()
        if self.current is not None:
            self.cam_list.selection_set(self.current)

    def _add_camera(self, source: str, name: str) -> None:
        names = [c["name"] for c in self.cfg["cameras"]]
        if any(str(c["source"]) == str(source) for c in self.cfg["cameras"]):
            self.say(f"{source} is already in the list")
            return
        self.cfg["cameras"].append({"name": next_name(names, name),
                                    "source": str(source), "ball_area": 0,
                                    "zones": []})
        self._refresh_cameras(select=len(self.cfg["cameras"]) - 1)
        self.hint.configure(text="Now draw each hub's outline: press Draw RED "
                                 "or Draw BLUE and click the corners.")

    def find_cameras(self) -> None:
        self.say("looking for cameras (a few seconds)…")

        def done(found):
            if not found:
                self._error("No camera answered. Check it is plugged in, and "
                            "on macOS that this terminal may use the camera "
                            "(System Settings > Privacy > Camera).")
                return
            self._choose_camera(found)
        self._in_thread(probe_cameras, done)

    def _choose_camera(self, found) -> None:
        tk, ttk = self.tk, self.ttk
        win = tk.Toplevel(self.root)
        win.title("Add a camera")
        ttk.Label(win, text="Cameras found. The numbers can change when a "
                            "camera is re-plugged,\nso check the picture after "
                            "adding.", padding=6).pack()
        for idx, size in found:
            ttk.Button(win, text=f"Camera {idx}  ({size})",
                       command=lambda i=idx: (win.destroy(),
                                              self._add_camera(str(i), f"cam{i}"))
                       ).pack(fill="x", padx=6, pady=2)

    def add_recording(self) -> None:
        from tkinter import filedialog
        p = filedialog.askopenfilename(
            title="A recording to set up or practise on",
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi *.m4v"), ("All", "*")])
        if p:
            self._add_camera(p, os.path.splitext(os.path.basename(p))[0][:20])

    def remove_camera(self) -> None:
        c = self._cam()
        if c:
            self.cfg["cameras"].remove(c)
            self.frames.pop(c["name"], None)
            self.current = None
            self._refresh_cameras(select=0)

    def grab(self) -> None:
        c = self._cam()
        if not c:
            return
        if self.running and not is_file_source(c["source"]):
            self.say("the camera is in use by the counter; its live picture "
                     "is shown instead")
            return
        name = c["name"]

        def done(frame):
            self.frames[name] = frame
            self._draw()
        try:
            at = float(self.seek.get()) if self.seek.get().strip() else None
        except ValueError:
            at = None
        self._in_thread(lambda: grab_frame(c["source"], float(c.get("fps") or 0),
                                           c.get("size", ""), at), done)

    def measure(self) -> None:
        from .hubcount import measure
        c = self._cam()
        if not c:
            return
        if not c.get("zones"):
            self._error("Draw the hub outline first: the ball is measured "
                        "near it.")
            return
        self.say(f"measuring {c['name']} for 5 s -- balls should be sitting "
                 f"apart near the hub…")

        def done(area):
            if area:
                c["ball_area"] = round(area)
                self.v["ball_area"].set(str(round(area)))
                self._refresh_cameras(select=self.cfg["cameras"].index(c))
        polys = {z["name"]: [tuple(p) for p in z["outline"]] for z in c["zones"]}
        self._in_thread(lambda: measure(c["source"], polys, 5.0, None,
                                        float(c.get("fps") or 0),
                                        c.get("size", ""), out=self.say), done)

    # -- outlines -------------------------------------------------------------
    def _refresh_zones(self) -> None:
        self.zone_list.delete(0, "end")
        c = self._cam()
        for z in (c or {}).get("zones", []):
            self.zone_list.insert("end", f"{z['hub'].upper():5} {z['name']}  "
                                         f"({len(z['outline'])} points)")

    def start_outline(self, hub: str) -> None:
        c = self._cam()
        if not c or c["name"] not in self.frames:
            self._error("Pick a camera and show its picture first.")
            return
        self.drawing = {"hub": hub, "pts": []}
        self.hint.configure(text=f"Click the corners of the {hub.upper()} hub's "
                                 f"opening. Double-click, right-click or Enter "
                                 f"to finish; Esc to cancel.")
        self._draw()

    def _click(self, e) -> None:
        if not self.drawing:
            return
        self.drawing["pts"].append(to_frame(e.x, e.y, self.scale))
        self._draw()

    def finish_outline(self) -> None:
        if not self.drawing:
            return
        pts = self.drawing["pts"]
        # a double-click also lands as a single click: drop the duplicate
        if len(pts) >= 2 and pts[-1] == pts[-2]:
            pts.pop()
        if len(pts) < 3:
            self.hint.configure(text="An outline needs at least three corners.")
            return
        c = self._cam()
        names = [z["name"] for cam in self.cfg["cameras"] for z in cam["zones"]]
        c["zones"].append({"name": next_name(names, f"{c['name']}-{self.drawing['hub']}"),
                           "hub": self.drawing["hub"],
                           "outline": [list(p) for p in pts]})
        self.drawing = None
        self._refresh_zones()
        self._draw()
        self.hint.configure(text="Outline saved. Put a few balls near the hub "
                                 "and press Measure ball, then START.")

    def cancel_outline(self) -> None:
        self.drawing = None
        self._draw()

    def delete_zone(self) -> None:
        c = self._cam()
        sel = self.zone_list.curselection()
        if c and sel:
            del c["zones"][sel[0]]
            self._refresh_zones()
            self._draw()

    def _store_combine(self) -> None:
        self.cfg["combine"] = {h: v.get() for h, v in self.combine.items()}

    # -- drawing --------------------------------------------------------------
    def _draw(self, frame=None, counts: Optional[Dict[str, int]] = None) -> None:
        c = self._cam()
        if not c:
            return
        frame = frame if frame is not None else self.frames.get(c["name"])
        self.canvas.delete("all")
        if frame is None:
            return
        h, w = frame.shape[:2]
        self.scale = fit_scale(w, h)
        self.photo = self.tk.PhotoImage(data=png_data(frame, self.scale))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        for z in c.get("zones", []):
            pts = to_canvas(z["outline"], self.scale)
            col = HUB_COLOURS.get(z["hub"], "#0f0")
            self.canvas.create_polygon(*pts, outline=col, fill="", width=3)
            label = z["name"]
            if counts and z["name"] in counts:
                label = f"{z['name']}: {counts[z['name']]}"
            self.canvas.create_text(pts[0], pts[1] - 12, text=label, fill=col,
                                    anchor="w", font=("Helvetica", 13, "bold"))
        if self.drawing:
            pts = to_canvas(self.drawing["pts"], self.scale)
            col = HUB_COLOURS[self.drawing["hub"]]
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill=col, width=2, dash=(4, 2))
            for i in range(0, len(pts), 2):
                self.canvas.create_oval(pts[i] - 4, pts[i + 1] - 4, pts[i] + 4,
                                        pts[i + 1] + 4, fill=col, outline="white")

    # -- calibration ------------------------------------------------------------
    def calibrate(self) -> None:
        from tkinter import filedialog, simpledialog

        from .hubcount import calibrate, setup_from_dict
        c = self._cam()
        if not c:
            return
        problems_ = [p for p in problems({"cameras": [c]})]
        if problems_:
            self._error("\n".join(problems_))
            return
        video = filedialog.askopenfilename(
            title=f"A recording from {c['name']}, same position and settings",
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi *.m4v"), ("All", "*")])
        if not video:
            return
        hand = {}
        for hub in sorted({z["hub"] for z in c["zones"]}):
            n = simpledialog.askinteger("Hand count", f"Balls that went into the "
                                        f"{hub.upper()} hub in this recording:",
                                        parent=self.root, minvalue=0)
            if n is None:
                return
            hand[hub] = n
        cam = setup_from_dict({"cameras": [dict(c, blur=0, remove_static=False)]}
                              ).cameras[0]
        self.say(f"calibrating {c['name']} on {os.path.basename(video)} "
                 f"(decodes it twice)…")

        def done(r):
            if not r:
                return
            best, base = r["best"], r["uncorrected"]
            msg = (f"Best: blur {best['blur']}, ignore still yellow "
                   f"{'on' if best['remove_static'] else 'off'}\n"
                   + "\n".join(f"{h}: {best['counts'].get(h, 0)} counted, "
                               f"{n} by hand" for h, n in hand.items())
                   + f"\n\nWithout correction: "
                   + ", ".join(f"{h} {base['counts'].get(h, 0)}" for h in hand)
                   + "\n\nThis is fitted to one recording. Check it on a second "
                     "one before trusting it.\n\nUse these settings?")
            from tkinter import messagebox
            if messagebox.askyesno("Calibration", msg, parent=self.root):
                c["blur"] = best["blur"]
                c["remove_static"] = best["remove_static"]
                self._select_camera()

        def progress(frac):
            self.q.put(("call", lambda: self.hint.configure(
                text=f"calibrating… {frac:.0%} of this pass")))
        self._in_thread(lambda: calibrate(cam, video, hand, out=self.say,
                                          progress=progress), done)

    # -- running ----------------------------------------------------------------
    def start(self) -> None:
        from . import hubcount, hubfeed
        self._store_camera()
        self._store_combine()
        issues = problems(self.cfg)
        if issues:
            self._error("Not ready yet:\n\n" + "\n".join(issues))
            return
        try:
            setup = hubcount.setup_from_dict(self.cfg)
        except ValueError as e:
            self._error(str(e))
            return
        target_text = self.target.get().strip() or DEFAULT_TARGET
        try:
            target = hubfeed.parse_target(target_text)
        except ValueError:
            self._error(f"bioarena address should look like {DEFAULT_TARGET}")
            return
        if self.local.get():
            target = ("127.0.0.1", target[1])
            self.listen_stop = threading.Event()
            threading.Thread(target=hubfeed.listen, daemon=True,
                             kwargs=dict(port=target[1], bind="127.0.0.1",
                                         out=lambda t: self.say(f"[test receiver] {t}"),
                                         stop=self.listen_stop)).start()
        log_path = None
        if self.log_on.get():
            base = os.path.dirname(self.path) if self.path else os.getcwd()
            log_path = os.path.join(base, time.strftime("hubfeed_%Y%m%d_%H%M%S.csv"))
            self.say(f"logging counts to {log_path}")
        self.sender = hubfeed.FeedSender(target)
        self.stop_evt = threading.Event()
        self.monitor = {}
        realtime = self.realtime.get() and any(is_file_source(c.source)
                                               for c in setup.cameras)

        def on_frame(name, frame):
            with self.live_lock:
                self.live_frames[name] = frame

        def work():
            try:
                hubcount.run(self.sender, setup, realtime=realtime,
                             log_path=log_path, out=self.say, stop=self.stop_evt,
                             on_frame=on_frame, monitor=self.monitor)
            except Exception as e:
                self.say(f"error: {e}")
            finally:
                self.q.put(("call", self._stopped))
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal", bg="#e53935", fg="white")
        self.say(f"session {self.sender.session}: sending to "
                 f"{target[0]}:{target[1]}")
        self.hint.configure(text="Running. Leave it on across matches; bioarena "
                                 "takes each match's start itself.")
        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        if self.stop_evt:
            self.stop_evt.set()

    def _stopped(self) -> None:
        self.running = False
        if self.listen_stop:
            self.listen_stop.set()
            self.listen_stop = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled", bg=self.root.cget("bg"),
                                fg="black")
        self.link_lbl.configure(text="stopped", fg="black")
        for e in self.monitor.get("errors", []):
            self._error(e)
        if self.sender:
            self.say(f"stopped: red {self.sender.counts['red']}, "
                     f"blue {self.sender.counts['blue']}")

    def _refresh_live(self) -> None:
        s = self.sender
        if not s:
            return
        self.red_lbl.configure(text=f"RED {s.counts['red']}")
        self.blue_lbl.configure(text=f"BLUE {s.counts['blue']}")
        health = self.monitor.get("health", {})
        cams = "   ".join(f"{n} {h.fps():.0f} fps" for n, h in health.items())
        now = time.monotonic()
        stale = [n for n, h in health.items() if now - h.last_frame > 0.5]
        if s.linked():
            r = s.last_reply or {}
            link = (f"bioarena OK  {r.get('match_state', '')} "
                    f"{r.get('shift', '')}  {s.rtt_ms} ms")
            colour = "#2e7d32"
        else:
            link, colour = "no reply from bioarena", "#c62828"
        if stale:
            link, colour = f"NO PICTURE from {', '.join(stale)} -- OFFLINE", "#c62828"
        self.link_lbl.configure(text=f"{link}\n{cams}", fg=colour)
        c = self._cam()
        tally = self.monitor.get("tally")
        if c and tally is not None:
            with self.live_lock:
                frame = self.live_frames.get(c["name"])
            if frame is not None:
                counts = {z.name: z.counter.reported
                          for zs in tally.zones.values() for z in zs}
                self._draw(frame, counts)

    # -- misc --------------------------------------------------------------------
    def _error(self, text: str) -> None:
        from tkinter import messagebox
        self._write_log(f"! {text}")
        messagebox.showwarning("Hub counter", text, parent=self.root)

    def quit(self) -> None:
        self.stop()
        if self.listen_stop:
            self.listen_stop.set()
        self.root.after(300, self.root.destroy)

    def mainloop(self) -> None:
        self.root.mainloop()


def main(setup_path: Optional[str] = None) -> int:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        raise SystemExit(
            "this Python has no tkinter. On macOS with Homebrew: "
            "brew install python-tk (matching your Python version); the "
            "python.org installer includes it. On Debian/Ubuntu: "
            "sudo apt install python3-tk")
    HubApp(setup_path).mainloop()
    return 0
