"""Watchtower.app: the Watchtower FMS (watchtower-fms v2) as one desktop app.

Watchtower v2 runs from its website (watchtower.systemoverload.org): sign up,
create an event, paste a launch command that installs Python and packages and
runs launch.py. Its own desktop app was dropped (its maintainer's decision,
2026-10-30). This is our copy for our own events: the same Watchtower on a
double-click, nothing installed, no terminal, no website account.

What a double-click does, as launch.py does it:

- first launch: ~/Documents/Watchtower/config/event.yaml with random PINs and
  a vision key (Watchtower's own fms.init), kept from then on; a v0.1 app's
  event.yaml there is read as it is (fms.config merges the new defaults in);
- the FMS on port 8000, open to phones on the same Wi-Fi (Control, refs,
  scorers, emcee, field display, stream overlay);
- Watchtower's own hub-camera runner (vision/run_vision.py, the colour
  counter copied from our tbavid) in a second process, restarted if it
  stops, with cameras, outlines, ball size (drop test), picture and fuel
  colour all set in Control -> Setup -> Cameras. Not for a REEFSCAPE event,
  which is scored by hand;
- one window on /control: everything else -- event, teams, schedule, PINs,
  TBA, field link to bioarena, cameras, Match13 -- is Control -> Setup, the
  same page the website's users get. The menu opens the field display, the
  overlay, the match queue and the data folder, checks for updates and quits.

Until v0.6 this app wrapped watchtower-fms v0.1 with our own Home window and
our hub counter page beside it; v2 moved both into Control -> Setup, so the
app now shows Watchtower's own pages and nothing in front of them.

Pages open in their own window (pywebview: WebKit on macOS, Edge WebView2 on
Windows), never framed: Watchtower keeps logins in localStorage, which WebKit
partitions for cross-origin frames. Where there is no window toolkit (Linux,
the Pi, --no-window) /control opens in the browser.

Watchtower's code is arnan-bajaj/watchtower-fms at the commit in
.github/watchtower-release, bundled by the spec; from a checkout,
WATCHTOWER_SRC points at a clone of it.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
VISION_RESTART_S = 3.0


def data_dir() -> Path:
    d = Path.home() / "Documents" / "Watchtower"
    d.mkdir(parents=True, exist_ok=True)
    return d


def watchtower_src() -> Path:
    """Where the fms and vision packages are: inside the app, or a clone
    named by WATCHTOWER_SRC (default: watchtower-src at the repo root)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(os.environ.get("WATCHTOWER_SRC") or HERE.parent.parent / "watchtower-src")


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def first_setup(d: Path, src: Path) -> dict:
    """config/event.yaml with PINs and a vision key, made once by
    Watchtower's own fms.init from its examples (copied fresh each launch, so
    an app update brings new example settings for a new event.yaml)."""
    cfg = d / "config"
    cfg.mkdir(exist_ok=True)
    for name in ("event.example.yaml", "vision.example.yaml"):
        shutil.copyfile(src / "config" / name, cfg / name)
    from fms import init
    return init.init(str(cfg))


# launch.py's camera config: the colour counter, hubs empty until Setup ->
# Cameras picks them, test recordings in the data folder's recordings/.
CAMERAS = """# Written by the Watchtower app on every start, as launch.py writes it.
# Each hub's camera, outline and ball size are set in Control -> Setup -> Cameras.
fms_url: http://127.0.0.1:{port}
record_dir: ""
test_dir: '{tests}'
defaults: {{counter: colour, width: 1920, height: 1080, fps: 60}}
hubs: {{red: {{}}, blue: {{}}}}
"""


def vision_config(d: Path, port: int) -> Path:
    p = d / "config" / "vision.cameras.yaml"
    p.write_text(CAMERAS.format(port=port, tests=(d / "recordings").as_posix()))
    return p


def run_vision(config: str) -> int:
    """`Watchtower --vision CONFIG`: Watchtower's vision runner, in this
    process. The app starts itself this way, since a frozen app has no
    python to run vision/run_vision.py with.

    OpenCV first, through tbavid.cvload: in the frozen Mac app OpenCV's own
    loader refused ("recursion is detected", v0.5.7) and the cameras never
    opened; this process is where OpenCV is used now."""
    from tbavid import cvload
    cvload.load()
    src = watchtower_src()
    vision = src / "vision"
    sys.path[:0] = [str(vision), str(src)]
    os.chdir(vision)
    threading.Thread(target=_exit_with_fms, args=(config,), daemon=True, name="fms watch").start()
    import run_vision as rv
    sys.argv = ["run_vision.py", "--config", config]
    rv.main()
    return 0


ORPHAN_S = 20.0


def _exit_with_fms(config: str) -> None:
    """End this vision process once the FMS has been gone ORPHAN_S: when the
    app is killed (Windows' Task Manager, a crash) nothing else would stop
    it, and it would keep a camera open with nothing to count for."""
    import re
    try:
        port = int(re.search(r"fms_url: http://[^:]+:(\d+)", Path(config).read_text()).group(1))
    except (OSError, AttributeError, ValueError):
        return
    gone = None
    while True:
        time.sleep(2)
        if port_open(port):
            gone = None
        elif gone is None:
            gone = time.monotonic()
        elif time.monotonic() - gone > ORPHAN_S:
            print("the FMS is gone: stopping the hub cameras", flush=True)
            os._exit(0)


class Vision:
    """The hub-camera runner as a child process, restarted if it stops by
    itself: a camera unplugged mid-event must not end counting for the day."""

    def __init__(self, d: Path, config: Path):
        self.d, self.config = d, config
        self.proc = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="vision")

    def command(self) -> list:
        me = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, str(Path(__file__).resolve())]
        return me + ["--vision", str(self.config)]

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        log = open(self.d / "vision.log", "a", buffering=1)
        while not self.stopping.is_set():
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(self.command(), cwd=str(self.d), env=env,
                                         stdout=log, stderr=log, stdin=subprocess.DEVNULL)
            code = self.proc.wait()
            if self.stopping.is_set():
                return
            print(f"hub cameras stopped (exit {code}); restarting in {VISION_RESTART_S:.0f} s",
                  file=log)
            self.stopping.wait(VISION_RESTART_S)

    def stop(self) -> None:
        self.stopping.set()
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


TBA_TTL = 30.0          # s a TBA schedule is reused; TBA itself caches about a minute


def add_pages(app, cfg: dict = None) -> None:
    """Our match queue on Watchtower's own server, so phones and the public
    address reach it like /ref or /display. /queue: the matches still to play,
    for teams in the pits, who otherwise had to ask Control what was next. It
    reads the same /ws state as every other page, so it needs no PIN.

    /queue/tba.json is TBA's schedule for the same page (queue_tba.py). The
    page is public, so TBA is asked at most once per TBA_TTL per event and
    once every 2 s in all, with 16 events remembered: a stranger looping over
    event keys cannot turn this into a TBA flood."""
    import re
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    import queue_tba
    cfg = cfg or {}
    default_event = str((cfg.get("event") or {}).get("tba_event_key") or "").strip().lower()
    read_key = str((cfg.get("tba") or {}).get("read_key") or "")
    queue = Path(getattr(sys, "_MEIPASS", HERE)) / "queue.html"
    app.add_api_route("/queue", lambda: FileResponse(queue, headers={"Cache-Control": "no-store"}),
                      methods=["GET"])
    cache: dict = {}            # event -> (fetched, result)
    lock = threading.Lock()
    last = [0.0]

    def tba_json(event: str = ""):
        key = (event or default_event).strip().lower()
        if not re.fullmatch(r"\d{4}[a-z0-9]{1,16}", key):
            raise HTTPException(400, "No TBA event key: open /queue?event=2026catstd, or set it in Control -> Setup.")
        with lock:
            now = time.time()
            hit = cache.get(key)
            if hit and now - hit[0] < TBA_TTL:
                return {**hit[1], "event": key, "age": round(now - hit[0])}
            if now - last[0] < 2.0:
                if hit:
                    return {**hit[1], "event": key, "age": round(now - hit[0])}
                raise HTTPException(429, "Busy; trying again shortly.")
            last[0] = now
            r = queue_tba.tba_matches(key, read_key)
            if "error" in r and hit and "error" not in hit[1]:
                # Offline or TBA down: keep showing the last schedule, and say so.
                return {**hit[1], "event": key, "age": round(now - hit[0]), "error": r["error"]}
            cache[key] = (now, r)
            while len(cache) > 16:
                cache.pop(min(cache, key=lambda k: cache[k][0]))
            return {**r, "event": key, "age": 0}
    app.add_api_route("/queue/tba.json", tba_json, methods=["GET"])


class FmsServer:
    """Watchtower's FastAPI app under uvicorn, in a thread, as `python -m
    fms.server` runs it."""

    def __init__(self, host: str, port: int):
        import uvicorn
        from fms import server             # reads FMS_CONFIG at import
        self.module = server
        add_pages(server.app, server.CFG)
        self.server = uvicorn.Server(uvicorn.Config(server.app, host=host, port=port,
                                                    log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="watchtower fms")

    def start(self, port: int, wait: float = 30.0) -> bool:
        self.thread.start()
        t = time.time()
        while time.time() - t < wait:
            if port_open(port):
                return True
            if not self.thread.is_alive():
                return False
            time.sleep(0.2)
        return False

    def match_running(self) -> bool:
        try:
            return self.module.current() is not None
        except Exception:
            return False

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


class App:
    """Everything the window's menu does; the lifecycle in one place."""

    def __init__(self, d: Path, port: int, fms: FmsServer, vision, browser: bool = False):
        self.d, self.port, self.fms, self.vision, self.browser = d, port, fms, vision, browser
        self.base = f"http://localhost:{port}"
        self.window = None
        self.extra = []
        self.done = threading.Event()
        from tbavid.appupdate import Updater, build_version
        self.updater = Updater("Watchtower", build_version(), on_exit=self.quit)
        self.updater.check_async()

    def open_page(self, path: str, title: str, size=(1280, 760)) -> None:
        if self.browser or self.window is None:
            webbrowser.open(self.base + path)
            return
        import webview
        self.extra.append(webview.create_window(title, self.base + path, width=size[0], height=size[1]))

    def open_folder(self) -> None:
        p = str(self.d)
        if sys.platform == "darwin":
            subprocess.Popen(["open", p])
        elif sys.platform == "win32":
            os.startfile(p)                      # noqa: S606 -- the user's own folder
        else:
            subprocess.Popen(["xdg-open", p])

    def check_update(self) -> None:
        """Menu -> Check for updates. Never automatic, refused mid-match
        (installing closes the app for a minute)."""
        self.updater.check()
        st = self.updater.status()
        if not st["available"]:
            self.tell("Watchtower is up to date", f"This is version {st['current']}.")
            return
        if not st["can_install"]:
            self.tell("Update available", f"Version {st['latest']} is out. Download it from "
                      f"{st['release_url'] or 'the Releases page'}.")
            webbrowser.open(st["release_url"])
            return
        if self.ask("Update available", f"Install version {st['latest']} now? Watchtower closes, "
                    f"updates and reopens (about a minute). Your event and matches are kept."):
            r = self.updater.install(self.fms.match_running())
            if r.get("error"):
                self.tell("Not now", r["error"])

    def tell(self, title: str, msg: str) -> None:
        if self.window is not None:
            try:
                self.window.create_confirmation_dialog(title, msg)
                return
            except Exception:
                pass
        print(f"{title}: {msg}")

    def ask(self, title: str, msg: str) -> bool:
        if self.window is not None:
            try:
                return bool(self.window.create_confirmation_dialog(title, msg))
            except Exception:
                pass
        return False

    def quit(self) -> None:
        for w in [*self.extra, self.window]:
            try:
                if w is not None:
                    w.destroy()
            except Exception:
                pass
        self.done.set()

    def stop_all(self) -> None:
        if self.vision:
            self.vision.stop()
        self.fms.stop()


def menu(app: App):
    from webview.menu import Menu, MenuAction, MenuSeparator
    return [
        Menu("Watchtower", [
            MenuAction("Field display", lambda: app.open_page("/display", "Watchtower: Field display")),
            MenuAction("Stream overlay", lambda: app.open_page("/overlay", "Watchtower: Overlay", (1920 // 2, 1080 // 2))),
            MenuAction("Match queue", lambda: app.open_page("/queue", "Watchtower: Match queue", (1000, 800))),
            MenuSeparator(),
            MenuAction("Open event folder", app.open_folder),
            MenuAction("Check for updates...", lambda: threading.Thread(target=app.check_update, daemon=True).start()),
            MenuSeparator(),
            MenuAction("Quit Watchtower", app.quit),
        ]),
    ]


def selftest_window(app: App, out: str) -> None:
    """--selftest-window (CI on macOS and Windows): /control loads in the
    window and the field display opens in a second one."""
    import json
    steps, ok, err = [], False, ""
    try:
        for _ in range(150):
            title = app.window.evaluate_js("document.title") or ""
            if "Control" in title or "Watchtower" in title:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError(f"/control never loaded (title {title!r})")
        steps.append(f"Control loaded ({title})")
        app.open_page("/display", "Watchtower: Field display")
        time.sleep(3)
        t2 = app.extra[-1].evaluate_js("document.title") or ""
        steps.append(f"field display opened ({t2})")
        ok = True
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    Path(out).write_text(json.dumps({"ok": ok, "steps": steps, "error": err}))
    app.quit()


def run_window(app: App, selftest: str = "") -> bool:
    """The Control window until it closes. False if this computer has no
    window toolkit (then the caller falls back to the browser)."""
    try:
        import webview
    except ImportError:
        return False
    app.window = webview.create_window("Watchtower", app.base + "/control",
                                       width=1440, height=900, min_size=(390, 600))
    app.window.events.closed += app.quit      # closing Control quits the app
    try:
        private = os.environ.get("WATCHTOWER_WEBVIEW_PRIVATE") == "1"
        (app.d / "webview").mkdir(exist_ok=True)
        webview.start(func=(lambda: selftest_window(app, selftest)) if selftest else None,
                      private_mode=private, menu=menu(app),
                      storage_path=None if private else str(app.d / "webview"))
    except Exception as e:                        # no GTK/WebKit, no display
        print(f"no app window here ({type(e).__name__}: {e}); using the browser")
        app.window = None
        return False
    return True


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--vision":
        return run_vision(sys.argv[2])
    import argparse
    ap = argparse.ArgumentParser(prog="Watchtower", description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-browser", action="store_true",
                    help="no window and no browser (a computer with no screen, a test)")
    ap.add_argument("--no-window", action="store_true", help="use the browser instead of the app's window")
    ap.add_argument("--no-vision", action="store_true", help="no hub cameras (score by hand)")
    ap.add_argument("--restarted", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--selftest-window", metavar="RESULT", default="",
                    help="open the window, check it, write RESULT (JSON) and quit (CI)")
    args, _ = ap.parse_known_args()
    if args.no_browser:
        args.no_window = True
    d = data_dir()
    src = watchtower_src()
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(HERE))    # queue_tba.py beside this file (in the app: bundled)
    os.chdir(d)                      # Watchtower's config/ and data/ are relative
    if getattr(sys, "frozen", False):
        log = open(d / "watchtower.log", "a", buffering=1)
        sys.stdout = sys.stderr = log
    made = first_setup(d, src)
    if made["pins"]:
        print(f"first launch: made {d / 'config' / 'event.yaml'} with new PINs")
    os.environ["FMS_CONFIG"] = str(d / "config" / "event.yaml")
    from fms import config
    cfg = config.load(os.environ["FMS_CONFIG"])
    port = int(cfg["server"]["port"])
    if args.restarted:               # an update: the old process may still hold the port
        for _ in range(50):
            if not port_open(port):
                break
            time.sleep(0.2)
    if port_open(port):
        # Opened again while running (or something else has the port): show
        # Control in the browser rather than start a second Watchtower.
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{port}/control")
        return 0

    # Always on the Wi-Fi, whatever event.yaml's server.host says: refs,
    # scorers and the field display are phones and other screens.
    fms = FmsServer("0.0.0.0", port)
    if not fms.start(port):
        print(f"! Watchtower did not start on port {port}; see the messages above")
        return 1
    print(f"watchtower at http://localhost:{port}/control")
    vision = None
    if cfg["event"]["game"] == "rebuilt" and not args.no_vision:
        vision = Vision(d, vision_config(d, port))
        vision.start()
    from tbavid import keepawake
    release, how = keepawake.hold("running a match")   # no sleep mid-match
    print(f"keep awake: {how}")
    app = App(d, port, fms, vision, browser=args.no_window)
    try:
        if not args.no_window and run_window(app, args.selftest_window):
            return 0
        if args.selftest_window:     # asked to test the window, and there is none
            import json
            Path(args.selftest_window).write_text(json.dumps(
                {"ok": False, "steps": [], "error": "no window toolkit here (see watchtower.log)"}))
            return 1
        # Browser mode: Control in the browser; runs until stopped (Ctrl+C,
        # or the system's quit, which sends SIGTERM).
        app.browser = True
        signal.signal(signal.SIGTERM, lambda *_: app.done.set())
        if not args.no_browser:
            threading.Timer(0.5, lambda: webbrowser.open(app.base + "/control")).start()
        try:
            while not app.done.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        return 0
    finally:
        app.stop_all()
        release()


if __name__ == "__main__":
    sys.exit(main())
