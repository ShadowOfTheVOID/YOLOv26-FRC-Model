"""The hub counter as a web page: standard library only, any browser.

    python3 run.py hubgui                   # opens http://127.0.0.1:8790

No toolkit to install -- the Tk window needed `brew install python-tk` on a
Homebrew Mac -- and it looks the same on every machine. All the logic is in
`hubapp.HubController`; this file only turns HTTP into calls on it.

It listens on 127.0.0.1 and controls cameras and reads the disk, so it also
refuses requests whose Host header is not local (DNS rebinding) and POSTs
that are not JSON (a cross-site form cannot send application/json without a
preflight this server never answers). `--bind` can widen it for a trusted
field network; the page is then an open control panel to anyone on it.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .hubapp import HubController, encode, fit_scale, list_dir, probe_cameras

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def make_handler(ctl: HubController, allow_remote: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):     # quiet; the page has its log
            pass

        # -- plumbing ---------------------------------------------------------
        def _host_ok(self) -> bool:
            if allow_remote:
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in LOCAL_HOSTS

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        # -- GET ----------------------------------------------------------------
        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"local only", "text/plain")
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._json(ctl.state(int(q.get("log", 0) or 0)))
            if u.path == "/api/ls":
                return self._json(list_dir(q.get("path", "")))
            if u.path == "/frame.jpg":
                frame = ctl.picture(q.get("cam", ""))
                if frame is None:
                    return self._send(404, b"no picture", "text/plain")
                h, w = frame.shape[:2]
                mw = int(q.get("w", 960) or 960)
                return self._send(200, encode(frame, fit_scale(w, h, mw, mw)),
                                  "image/jpeg")
            self._send(404, b"not found", "text/plain")

        # -- POST ---------------------------------------------------------------
        def do_POST(self):
            if not self._host_ok():
                return self._send(403, b"local only", "text/plain")
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._send(415, b"json only", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            action = urlparse(self.path).path.rsplit("/", 1)[-1]
            try:
                result = ACTIONS[action](ctl, body)
            except KeyError as e:
                return self._json({"error": f"unknown: {e}"}, 404)
            except (ValueError, OSError) as e:
                return self._json({"error": str(e)}, 400)
            self._json({"ok": True, "result": result})

    return Handler


def _at(body) -> Optional[float]:
    v = body.get("at")
    return float(v) if v not in (None, "") else None


ACTIONS = {
    "add_camera": lambda c, b: c.add_camera(b.get("source", ""), b.get("name", "")),
    "remove_camera": lambda c, b: c.remove_camera(b["name"]),
    "update_camera": lambda c, b: c.update_camera(b["name"], b.get("fields", {})),
    "add_zone": lambda c, b: c.add_zone(b["camera"], b["hub"], b["points"]),
    "delete_zone": lambda c, b: c.delete_zone(b["camera"], b["zone"]),
    "combine": lambda c, b: c.set_combine(b["hub"], b["how"]),
    "grab": lambda c, b: c.job("picture", lambda: (c.grab(b["camera"], _at(b)), None)[1]),
    "measure": lambda c, b: c.job("measure", lambda: c.measure(b["camera"])),
    "find_cameras": lambda c, b: c.job("find cameras", probe_cameras),
    "calibrate": lambda c, b: c.job("calibrate", lambda: c.calibrate(
        b["camera"], b["video"], {k: int(v) for k, v in b["hand"].items()})),
    "apply_calibration": lambda c, b: c.update_camera(
        b["camera"], {"blur": b["blur"], "remove_static": b["remove_static"]}),
    "start": lambda c, b: c.start(b.get("target", ""), bool(b.get("practice")),
                                  bool(b.get("realtime", True)),
                                  bool(b.get("log", True))),
    "stop": lambda c, b: c.stop(),
    "save": lambda c, b: c.save(b.get("path") or None),
    "load": lambda c, b: c.load(b["path"]),
}


def serve(ctl: HubController, port: int = 8790, bind: str = "127.0.0.1",
          open_browser: bool = True) -> None:
    remote = bind not in ("127.0.0.1", "localhost", "::1")
    httpd = ThreadingHTTPServer((bind, port), make_handler(ctl, remote))
    url = f"http://{'127.0.0.1' if bind in ('0.0.0.0', '') else bind}:{port}/"
    print(f"hub counter at {url}  (Ctrl-C to quit)")
    if remote:
        print("! listening beyond this machine: anyone who can reach it can "
              "start, stop and reconfigure the counter")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop()
        httpd.server_close()


def main(setup_path: Optional[str] = None, port: int = 8790,
         bind: str = "127.0.0.1", open_browser: bool = True) -> int:
    serve(HubController(setup_path), port, bind, open_browser)
    return 0


# The page lives beside this file so it can be edited as HTML. Read at import:
# it is ~30 kB and never changes while the server runs.
PAGE = (Path(__file__).with_name("hubweb.html")).read_text(encoding="utf-8")
