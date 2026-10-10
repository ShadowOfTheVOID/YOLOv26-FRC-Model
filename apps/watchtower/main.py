"""Watchtower.app: the Watchtower FMS and the hub counter as one desktop app.

Without it a scrimmage needs a terminal: clone watchtower-fms, make a venv,
`python -m fms.init`, `./run.sh`, then start the hub counter and type
`http://<vision key>@<host>:8000` into it. This does all of that on a
double-click, in its own windows (no browser, no address bar):

- first launch: ~/Documents/Watchtower/config/event.yaml is made with random
  PINs and a vision key (Watchtower's own fms.init), and kept from then on;
- the FMS server starts on port 8000, open to phones on the same Wi-Fi
  (refs, emcee, field display), as `python -m fms.server` would;
- the hub counter starts on 127.0.0.1:8790 with its address box already set
  to this FMS and its vision key, so counts arrive with nothing typed;
- the Home window (home.html) shows the event, the PINs and the phone
  address, edits the event name, date and teams, and opens Scorekeeper, Hub
  cameras and Field display each in its own app window. Closing Home quits.

Each page gets its own top-level window, not a frame inside Home: Watchtower
keeps logins in localStorage, which WebKit partitions for framed pages from
another origin, so a framed Scorekeeper would forget its login.

The windows are pywebview: WebKit on macOS, Edge WebView2 on Windows. Where
neither is there (Linux and the Pi: no window toolkit is bundled, or
--no-window) the same Home page opens in the browser, served on
127.0.0.1:8789 behind a per-launch token, with every setting still editable:
nobody edits event.yaml by hand on any OS.

Watchtower's code is arnan-bajaj/watchtower-fms at the tag in
.github/watchtower-release, bundled by the spec; from a checkout,
WATCHTOWER_SRC points at a clone of it.
"""
from __future__ import annotations

import html
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

FMS_PORT = 8000
HUB_PORT = 8790
HOME_PORT = 8789          # the Home page in browser mode (127.0.0.1 only)
HERE = Path(__file__).resolve().parent


def data_dir() -> Path:
    d = Path.home() / "Documents" / "Watchtower"
    d.mkdir(parents=True, exist_ok=True)
    return d


def watchtower_src() -> Path:
    """Where the fms package and its example configs are: inside the app, or
    a clone named by WATCHTOWER_SRC (default: watchtower-src at the repo root)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(os.environ.get("WATCHTOWER_SRC") or HERE.parent.parent / "watchtower-src")


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def lan_ip() -> str:
    """This computer's address on the Wi-Fi, for the phone links."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))        # nothing is sent
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


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


def vision_feed(d: Path) -> str:
    """The first `feeds:` entry of config/vision.yaml as host:port: the
    field system Watchtower's own vision would send to (and, since
    watchtower-fms PR #3, its mock runs too). "" if none or unreadable."""
    import yaml
    try:
        feeds = (yaml.safe_load((d / "config" / "vision.yaml").read_text()) or {}).get("feeds")
    except (OSError, yaml.YAMLError, AttributeError):
        return ""
    if isinstance(feeds, dict):
        feeds = [feeds]
    for f in feeds or []:
        if isinstance(f, dict) and f.get("host"):
            return f"{f['host']}:{int(f.get('port') or 8411)}"
        if isinstance(f, str) and f.strip():
            return f.strip() if ":" in f else f"{f.strip()}:8411"
    return ""


def counter_target(ev: dict) -> str:
    """What the hub counter sends to: Watchtower on this computer, and
    bioarena too when its address is set (tbavid.fmslink.FanOut)."""
    t = f"http://{ev['key']}@127.0.0.1:{ev['port']}"
    return f"{t}, {ev['field_feed']}" if ev.get("field_feed") else t


def read_event(d: Path) -> dict:
    import yaml
    ev = yaml.safe_load((d / "config" / "event.yaml").read_text()) or {}
    srv, e = ev.get("server") or {}, ev.get("event") or {}
    return {"name": e.get("name", ""), "date": str(e.get("date") or ""),
            "teams": list(e.get("teams") or []),
            "pins": srv.get("pins") or {}, "key": srv.get("vision_key", ""),
            "public_url": str(srv.get("public_url") or "").rstrip("/"),
            "field_feed": str(srv.get("field_feed") or "").strip() or vision_feed(d),
            "port": int(srv.get("port") or FMS_PORT)}


TBA_TTL = 30.0          # s a TBA schedule is reused; TBA itself caches about a minute


def add_pages(app, cfg: dict = None) -> None:
    """Our pages on Watchtower's own server, so phones and the public address
    reach them like /ref or /display. /queue: the matches still to play, for
    teams in the pits, who otherwise had to ask the scorekeeper what was next
    (Watchtower's pages show only the current match). It reads the same /ws
    state as every other page, so it needs no PIN.

    /queue/tba.json is TBA's schedule for the same page, for an event
    Watchtower is not running (?event=) or before the scorekeeper has
    entered one. The Read API key stays here, never sent to the page. The
    page is public (the proxy), so TBA is asked at most once per TBA_TTL per
    event and once every 2 s in all, with 16 events remembered: a stranger
    looping over event keys cannot turn this into a TBA flood."""
    import re
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    import settings
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
            raise HTTPException(400, "No TBA event key: open /queue?event=2026catstd, or set it on The Blue Alliance tab.")
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
            r = settings.tba_matches(key, read_key)
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
    fms.server` runs it (same host and port from event.yaml)."""

    def __init__(self, host: str, port: int):
        import uvicorn
        from fms import server             # reads config/event.yaml at import
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

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


VIEWS = {   # name: (window title, path on the FMS, or None for the hub counter, size)
    "control": ("Watchtower: Scorekeeper", "/control", (1320, 860)),
    "hub": ("Watchtower: Hub cameras", None, (1400, 900)),
    "display": ("Watchtower: Field display", "/display", (1280, 720)),
    "queue": ("Watchtower: Match queue", "/queue", (1000, 800)),
}


class App:
    """The Home window's Python side (window.pywebview.api in home.html)
    and the other windows it opens."""

    def __init__(self, d: Path, ev: dict, ctl, stop_all, browser: bool = False):
        self.d, self.ev, self.ctl, self.stop_all = d, ev, ctl, stop_all
        self.browser = browser          # Home in the browser: views open as tabs
        self.on_close = None            # browser mode: how to end the Home server
        self.windows: dict = {}
        self.home = None
        self.ip = lan_ip()
        # One-click updates, shown on Overview. Updating quits like Quit does
        # (closing the windows lets the installer replace the app).
        from tbavid.appupdate import Updater, build_version
        self.updater = Updater("Watchtower", build_version(), on_exit=self.quit)
        self.updater.check_async()

    # -- called from home.html ----------------------------------------------
    def state(self) -> dict:
        st = self.ctl.state()
        live = st.get("live") or {}
        ip = lan_ip()
        from tbavid.hubweb import host_name
        name = host_name()
        return {"name": self.ev["name"], "date": self.ev["date"], "teams": self.ev["teams"],
                "pins": {k: str(v) for k, v in self.ev["pins"].items()},
                "phone_url": f"http://{ip}:{self.ev['port']}",
                # The friendly .local name shown to refs, or "" -- the QR still
                # carries the IP, so a scan connects even where .local does not.
                "phone_name": f"http://{name}:{self.ev['port']}" if name else "",
                # Set in Phones & PINs when a proxy or tunnel serves this FMS
                # under a public name: then that leads, and the QR carries it.
                "public_url": self.ev.get("public_url", ""),
                "on_network": ip != "127.0.0.1",
                "fms_ok": port_open(self.ev["port"]), "data_dir": str(self.d),
                "hub": {"running": bool(st.get("running")), "linked": bool(live.get("linked")),
                        "counts": live.get("counts") or {"red": 0, "blue": 0},
                        "cameras": len(st["cfg"].get("cameras") or [])}}

    def qr(self, text: str) -> str:
        """An SVG QR code, so a phone opens the page by pointing its camera."""
        import io
        import qrcode
        import qrcode.image.svg
        buf = io.BytesIO()
        qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, border=2).save(buf)
        return buf.getvalue().decode("utf-8")

    def get_settings(self) -> dict:
        import settings
        return {"values": settings.load(self.d / "config" / "event.yaml"),
                "fields": [{"name": ".".join(p), "kind": k, "label": l} for p, k, l in settings.FIELDS]}

    def local_offset(self, date: str) -> float:
        import settings
        return settings.local_utc_offset(date or "")

    def new_pins(self) -> dict:
        import settings
        return settings.new_pins()

    def parse_team_text(self, text: str) -> dict:
        import settings
        return settings.parse_team_text(text)

    def tba_teams(self, event_key: str, read_key: str) -> dict:
        import settings
        return settings.tba_teams(event_key, read_key)

    def save_settings(self, values: dict) -> dict:
        import settings
        try:
            r = settings.save(self.d / "config" / "event.yaml", values or {})
        except (ValueError, OSError) as e:
            return {"error": str(e)}
        if r["changed"]:
            # Watchtower reads event.yaml once, at import, and starts a TBA
            # thread that cannot be stopped: a clean restart is a new process.
            threading.Timer(0.8, self.restart).start()
        return r

    def open_view(self, name: str) -> None:
        """Scorekeeper, Hub cameras or Field display in its own window
        (brought to the front if it is already open)."""
        if name not in VIEWS:
            return
        title, path, _ = VIEWS[name]
        if self.browser:
            webbrowser.open(f"http://127.0.0.1:{HUB_PORT}/" if path is None
                            else f"http://127.0.0.1:{self.ev['port']}{path}")
            return
        import webview
        w = self.windows.get(name)
        if w is not None:
            try:
                w.restore()
                w.show()
                return
            except Exception:
                self.windows.pop(name, None)
        title, path, (wd, ht) = VIEWS[name]
        url = (f"http://127.0.0.1:{HUB_PORT}/" if path is None
               else f"http://127.0.0.1:{self.ev['port']}{path}")
        w = webview.create_window(title, url, width=wd, height=ht, min_size=(900, 600))
        w.events.closed += lambda: self.windows.pop(name, None)
        self.windows[name] = w

    def fullscreen(self, name: str) -> None:
        if self.browser:                # the browser's own full screen (F11)
            self.open_view(name)
            return
        if name not in self.windows:
            self.open_view(name)
            time.sleep(1.0)
        w = self.windows.get(name)
        if w is not None:
            w.toggle_fullscreen()

    def open_folder(self) -> None:
        opener = {"darwin": ["open"], "win32": ["explorer"]}.get(sys.platform, ["xdg-open"])
        try:
            subprocess.Popen(opener + [str(self.d)])
        except OSError:
            pass

    def quit(self) -> None:
        threading.Thread(target=self._close_all, daemon=True).start()

    def update_status(self) -> dict:
        return self.updater.status()

    def update_install(self) -> dict:
        return self.updater.install(bool(self.ctl.state().get("running")))

    def open_release(self) -> None:
        """'What's new' in the real browser: a link followed inside the app
        window would take Home away."""
        if self.updater.release_url:
            webbrowser.open(self.updater.release_url)

    # -- lifecycle ------------------------------------------------------------
    def restart(self) -> None:
        self.stop_all()
        keep = [a for a in sys.argv[1:] if a in ("--no-window", "--no-browser")]
        cmd = [sys.executable] + ([] if getattr(sys, "frozen", False)
                                  else [str(Path(__file__).resolve())]) + keep + ["--restarted"]
        log = open(self.d / "watchtower.log", "a", buffering=1)
        print(f"restarting to apply the settings: {cmd}", file=log)
        # Its own session, so it outlives this process and whatever started it.
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
              if sys.platform == "win32" else {"start_new_session": True})
        subprocess.Popen(cmd, cwd=str(self.d), stdout=log, stderr=log, stdin=subprocess.DEVNULL, **kw)
        self._close_all()

    def _close_all(self) -> None:
        if self.on_close:
            self.on_close()
        for w in list(self.windows.values()) + ([self.home] if self.home else []):
            try:
                w.destroy()
            except Exception:
                pass


class HomeApi:
    """What home.html may call (window.pywebview.api), and nothing else.

    pywebview builds the page's API by walking every public attribute of
    this object, recursively. Handed App itself, it walked into App.home --
    the window -- and on Windows read WebView2's controller from a
    background thread: "CoreWebView2Controller members can only be accessed
    from the UI thread", and the window never finished loading (macOS's
    WebKit let it pass). So only methods here, and the app behind an
    underscore name, which pywebview skips."""

    def __init__(self, app: "App"):
        self._app = app

    def state(self):
        return self._app.state()

    def qr(self, text):
        return self._app.qr(text)

    def get_settings(self):
        return self._app.get_settings()

    def local_offset(self, date):
        return self._app.local_offset(date)

    def new_pins(self):
        return self._app.new_pins()

    def save_settings(self, values):
        return self._app.save_settings(values)

    def tba_teams(self, event_key, read_key):
        return self._app.tba_teams(event_key, read_key)

    def parse_team_text(self, text):
        return self._app.parse_team_text(text)

    def open_view(self, name):
        return self._app.open_view(name)

    def fullscreen(self, name):
        return self._app.fullscreen(name)

    def open_folder(self):
        return self._app.open_folder()

    def quit(self):
        return self._app.quit()

    def update_status(self):
        return self._app.update_status()

    def update_install(self):
        return self._app.update_install()

    def open_release(self):
        return self._app.open_release()


def selftest_windows(app: "App", out: str) -> None:
    """--selftest-window: what CI can check of the real windows on each OS.
    Home loads and fills itself through the Python API; Scorekeeper, Hub
    cameras and Field display each open in their own window on their page.
    Writes a JSON result, then quits."""
    import json
    res = {"ok": False, "steps": []}

    def log_tail() -> str:
        try:
            tail = (app.d / "watchtower.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return " / ".join(l.strip() for l in tail.splitlines()[-25:] if l.strip())[-1500:]

    def watchdog():
        # A window engine that never answers (evaluate_js blocks for good)
        # must still produce a result: on CI it is the only report there is.
        time.sleep(180)
        res["error"] = "stuck after: " + " | ".join(res["steps"] or ["start"]) + " | log: " + log_tail()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        os._exit(3)
    threading.Thread(target=watchdog, daemon=True, name="selftest watchdog").start()

    last_err = []

    def until(fn, t=30.0):
        end = time.time() + t
        while time.time() < end:
            try:
                v = fn()
                if v:
                    return v
            except Exception as e:                  # kept for the report
                last_err[:] = [f"{type(e).__name__}: {e}"]
            time.sleep(0.3)
        return None

    try:
        # A cold first start (WebView2 setting up its profile on a fresh
        # machine) can take longer than pywebview's own 20 s wait.
        res["steps"].append(f"home loaded: {app.home.events.loaded.wait(90)}")
        js_pin = "document.getElementById('pin_control_t').textContent"
        pin = until(lambda: (lambda v: v if v and v != "——" else None)(app.home.evaluate_js(js_pin)))
        res["steps"].append(f"home filled by the API: control PIN shown = {pin == str(app.ev['pins'].get('control'))}")
        assert pin == str(app.ev["pins"].get("control")), f"Home did not show the control PIN {last_err}"
        # Our camera setup page, inside Home's Hub cameras tab.
        app.home.evaluate_js("show('hub')")
        tab = until(lambda: app.home.evaluate_js("window.__hubLoaded === true"))
        res["steps"].append(f"hub cameras tab: setup page loaded inside Home = {bool(tab)}")
        assert tab, f"the Hub cameras tab never loaded {last_err}"
        app.home.evaluate_js("show('home')")
        for name, marker in (("control", "FMS Control"), ("display", "")):
            app.open_view(name)
            w = until(lambda: app.windows.get(name), 10)
            assert w is not None, f"{name}: no window"
            loaded = w.events.loaded.wait(30)
            title = until(lambda: w.evaluate_js("document.title") or "(untitled)")
            body = until(lambda: w.evaluate_js("document.body ? document.body.innerText.slice(0, 3000) : ''") or " ")
            res["steps"].append(f"{name}: window opened, loaded={loaded}, title {title!r}")
            assert loaded and title, f"{name}: page never loaded {last_err}"
            if marker:
                assert marker.lower() in f"{title} {body}".lower(), f"{name}: not the expected page"
        res["ok"] = True
    except Exception as e:                          # report anything, never hang
        res["error"] = f"{type(e).__name__}: {e}"
        # What pywebview and the window engine said: on a CI machine the log
        # is the only place the real reason shows.
        res["error"] += " | log: " + log_tail()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    app.quit()


HOME_API = ("state", "qr", "get_settings", "local_offset", "new_pins", "save_settings", "tba_teams",
            "parse_team_text",
            "open_view", "fullscreen", "open_folder", "quit", "update_status", "update_install",
            "open_release")


def serve_home(app: "App", token: str):
    """home.html for a browser, its window.pywebview.api calls sent as POST
    /api/<name> with the launch token. Bound to 127.0.0.1, and every API
    call needs the token in a header: a page from elsewhere cannot send that
    header without a CORS preflight, which this never answers, so it cannot
    read the PINs or change the settings."""
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    page = (Path(getattr(sys, "_MEIPASS", HERE)) / "home.html").read_text(encoding="utf-8")
    shim = ("<script>(()=>{const t=new URLSearchParams(location.search).get('t')||'';"
            "window.pywebview={api:new Proxy({},{get:(_,m)=>(...a)=>fetch('/api/'+m,"
            "{method:'POST',headers:{'Content-Type':'application/json','X-Home-Token':t},"
            "body:JSON.stringify(a)}).then(r=>r.json()).then(j=>{if(j&&j.__error)throw new Error(j.__error);return j.result;})})};"
            "addEventListener('DOMContentLoaded',()=>setTimeout(()=>dispatchEvent(new Event('pywebviewready')),0));})();</script>")
    body = page.replace("<script>", shim + "\n<script>", 1).encode("utf-8")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, data, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] != "/":
                return self._send(404, b"{}")
            self._send(200, body, "text/html; charset=utf-8")

        def do_POST(self):
            name = self.path[len("/api/"):] if self.path.startswith("/api/") else ""
            if self.headers.get("X-Home-Token") != token or name not in HOME_API:
                return self._send(403, b'{"__error": "not allowed"}')
            n = int(self.headers.get("Content-Length") or 0)
            try:
                args = json.loads(self.rfile.read(n) or b"[]")
                res = getattr(app, name)(*args)
                out = {"result": res}
            except Exception as e:                  # the page shows it
                out = {"__error": f"{type(e).__name__}: {e}"}
            self._send(200, json.dumps(out).encode("utf-8"))

    httpd = ThreadingHTTPServer(("127.0.0.1", HOME_PORT), H)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="home page").start()
    return httpd


def run_windows(d: Path, ev: dict, ctl, stop_all, hub: threading.Thread,
                selftest: str = "") -> bool:
    """Home + on-demand windows until Home closes. False if this computer has
    no window toolkit (then the caller falls back to the browser)."""
    try:
        import webview
    except ImportError:
        return False
    app = App(d, ev, ctl, stop_all)
    home_html = (Path(getattr(sys, "_MEIPASS", HERE)) / "home.html").read_text(encoding="utf-8")
    app.home = webview.create_window("Watchtower", html=home_html, js_api=HomeApi(app),
                                     # Room for the camera setup page in the Hub cameras tab.
                                     width=1440, height=900, min_size=(980, 640))
    app.home.events.closed += app._close_all     # closing Home quits the app
    # Quit on the Hub cameras page ends the hub server: close the windows too.
    threading.Thread(target=lambda: (hub.join(), app._close_all()), daemon=True,
                     name="quit from the hub page").start()
    func = None
    if selftest:
        func = lambda: selftest_windows(app, selftest)
    try:
        # Logins (localStorage) kept between launches, in the data folder.
        # WATCHTOWER_WEBVIEW_PRIVATE=1 is for testing on Linux with Qt's engine,
        # whose persistent profile hung with a second window; WebKit (macOS)
        # and WebView2 (Windows) keep the logins.
        private = os.environ.get("WATCHTOWER_WEBVIEW_PRIVATE") == "1"
        (d / "webview").mkdir(exist_ok=True)
        webview.start(func=func, private_mode=private,
                      storage_path=None if private else str(d / "webview"))
    except Exception as e:                        # no GTK/WebKit, no display
        print(f"no app window here ({type(e).__name__}: {e}); using the browser")
        return False
    return True


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="Watchtower", description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-browser", action="store_true",
                    help="no window and no browser (a computer with no screen, a test)")
    ap.add_argument("--no-window", action="store_true",
                    help="use the browser and a start page instead of the app's windows")
    ap.add_argument("--restarted", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--selftest-window", metavar="RESULT", default="",
                    help="open every window, check it, write RESULT (JSON) and quit (CI)")
    args, _ = ap.parse_known_args()
    if args.no_browser:
        args.no_window = True
    d = data_dir()
    if args.restarted:               # the old process may still hold the ports
        for _ in range(50):
            if not (port_open(HUB_PORT) or port_open(FMS_PORT)):
                break
            time.sleep(0.2)
    if port_open(HUB_PORT) or port_open(FMS_PORT):
        # Opened again while running: the windows are already up. In browser
        # mode, show Home again (its address, with this launch's token, was
        # saved for this). A port taken by something else shows up here too.
        link = d / "webview" / "home-url.txt"
        if not args.no_browser and link.exists() and port_open(HOME_PORT):
            webbrowser.open(link.read_text().strip())
        return 0
    os.chdir(d)                      # Watchtower's config/ and data/ are relative
    if getattr(sys, "frozen", False):
        log = open(d / "watchtower.log", "a", buffering=1)
        sys.stdout = sys.stderr = log
    os.environ.setdefault("YOLO_CONFIG_DIR", str(d / "ultralytics"))
    src = watchtower_src()
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(HERE))    # settings.py beside this file (in the app: bundled)
    made = first_setup(d, src)
    if made["pins"]:
        print(f"first launch: made {d / 'config' / 'event.yaml'} with new PINs")
    os.environ["FMS_CONFIG"] = str(d / "config" / "event.yaml")
    ev = read_event(d)

    # Always on the Wi-Fi, whatever event.yaml's server.host says: refs,
    # emcee and the field display are phones and other screens.
    fms = FmsServer("0.0.0.0", ev["port"])
    if not fms.start(ev["port"]):
        print(f"! Watchtower did not start on port {ev['port']}; see the messages above")
        return 1
    print(f"watchtower at http://localhost:{ev['port']}/")
    if sys.platform == "darwin":       # run.sh's caffeinate: no sleep mid-match
        try:
            subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        except OSError:
            pass

    from tbavid import hubweb
    from tbavid.hubapp import HubController
    ctl = HubController(str(d / "cams.json"))
    ctl.default_target = counter_target(ev)
    if ev.get("field_feed"):
        print(f"hub counter: to Watchtower and bioarena at {ev['field_feed']}")
    # Camera presets saved here or in the Hub Counter app are the same file.
    ctl.presets_file = str(Path.home() / "Documents" / "Hub Counter" / "camera-presets.json")
    hub = threading.Thread(target=hubweb.serve, args=(ctl, HUB_PORT, "127.0.0.1"),
                           kwargs={"open_browser": False}, daemon=True, name="hub counter")
    hub.start()
    stopped = threading.Event()

    def stop_all() -> None:
        if stopped.is_set():
            return
        stopped.set()
        if ctl.on_quit:              # stops counting and the hub page's server
            ctl.quit()
        hub.join(timeout=5)
        fms.stop()

    if not args.no_window and run_windows(d, ev, ctl, stop_all, hub, args.selftest_window):
        stop_all()
        return 0
    if args.selftest_window:         # asked to test windows, and there are none
        import json
        Path(args.selftest_window).write_text(json.dumps(
            {"ok": False, "steps": [], "error": "no window toolkit here (see watchtower.log)"}))
        stop_all()
        return 1
    # Browser mode: the same Home page, served here. Quit (Home or the hub
    # page) ends it.
    import secrets
    app = App(d, ev, ctl, stop_all, browser=True)
    token = secrets.token_urlsafe(16)
    home = serve_home(app, token)
    done = threading.Event()
    app.on_close = lambda: (stop_all(), done.set())
    url = f"http://127.0.0.1:{HOME_PORT}/?t={token}"
    (d / "webview").mkdir(exist_ok=True)
    (d / "webview" / "home-url.txt").write_text(url)
    print(f"watchtower home at {url}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        while hub.is_alive() and not done.is_set():
            hub.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    stop_all()
    home.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
