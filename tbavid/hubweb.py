"""The hub counter as a web page: standard library only, any browser.

    python3 run.py hubgui                   # opens http://127.0.0.1:8790
                                            # scoreboard at /board

No toolkit to install -- the Tk window needed `brew install python-tk` on a
Homebrew Mac -- and it looks the same on every machine. All the logic is in
`hubapp.HubController`; this file only turns HTTP into calls on it.

It listens on 127.0.0.1 and controls cameras and reads the disk, so it also
refuses requests whose Host header is not local (DNS rebinding) and POSTs
that are not JSON (a cross-site form cannot send application/json without a
preflight this server never answers). `--bind` can widen it for a trusted
field network; the page is then an open control panel to anyone on it.

Sharing (`Share`, the page's Share button, `--share`) is the safer way to
open it to other devices: a second listener on every interface, port 8791,
where everything but the read-only scoreboard needs a PIN shown only on the
host. A phone at the scoring table, or the page of a Pi with no screen,
then works over Wi-Fi without the venue's whole network getting the
controls. Only the host can quit or stop sharing.
"""
from __future__ import annotations

import hmac
import json
import secrets
import socket
import threading
import time
from pathlib import Path
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .hubapp import HubController, encode, fit_scale, list_dir, probe_cameras

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
SHARE_PORT = 8791
OPEN_PATHS = {"/board", "/api/board", "/login"}   # no PIN on the shared port
HOST_ONLY = {"quit", "share_on", "share_off", "update_install"}  # never from a shared device
MAX_TRIES = 5                                     # wrong PINs per address ...
LOCKOUT_S = 60.0                                  # ... before a minute's wait


def lan_ip() -> str:
    """This machine's address on the network it would use to reach others
    (no packet is sent: connecting a UDP socket only picks the route)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class Share:
    """The page on the network, behind a PIN."""

    def __init__(self, ctl: HubController):
        self.ctl = ctl
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.pin = ""
        self.port = SHARE_PORT
        self.tokens: set = set()
        self.fails: dict = {}           # client address -> [times of wrong PINs]
        self.lock = threading.Lock()

    @property
    def on(self) -> bool:
        return self.httpd is not None

    def url(self) -> str:
        return f"http://{lan_ip()}:{self.port}/"

    def start(self, pin: str = "", port: int = SHARE_PORT) -> dict:
        with self.lock:
            if self.httpd is None:
                self.pin = str(pin) if pin else f"{secrets.randbelow(10**6):06d}"
                self.port = port
                self.tokens = set()
                self.httpd = ThreadingHTTPServer(("0.0.0.0", port),
                                                 make_handler(self.ctl, True, self))
                threading.Thread(target=self.httpd.serve_forever, daemon=True,
                                 name="hub page, shared").start()
                self.ctl.say(f"sharing on Wi-Fi at {self.url()} (PIN on this computer's page)")
        return self.status(True)

    def stop(self) -> dict:
        with self.lock:
            if self.httpd is not None:
                httpd, self.httpd = self.httpd, None
                threading.Thread(target=lambda: (httpd.shutdown(), httpd.server_close()),
                                 daemon=True).start()
                self.tokens = set()
                self.ctl.say("stopped sharing on Wi-Fi")
        return self.status(True)

    def status(self, host: bool) -> dict:
        if not self.on:
            return {"on": False}
        out = {"on": True, "url": self.url()}
        if host:
            out["pin"] = self.pin
        return out

    def login(self, addr: str, pin: str) -> Optional[str]:
        """A session token for the right PIN; None, and a strike, otherwise.
        Five wrong in a minute from one address locks it out for a minute:
        a 6-digit PIN would otherwise fall to a script in minutes."""
        now = time.monotonic()
        with self.lock:
            tries = [t for t in self.fails.get(addr, []) if now - t < LOCKOUT_S]
            if len(tries) >= MAX_TRIES:
                self.fails[addr] = tries
                raise PermissionError("too many wrong PINs; wait a minute")
            if self.pin and hmac.compare_digest(str(pin).strip(), self.pin):
                self.fails.pop(addr, None)
                token = secrets.token_urlsafe(24)
                self.tokens.add(token)
                return token
            self.fails[addr] = tries + [now]
            return None

    def allowed(self, cookie_header: str) -> bool:
        for part in (cookie_header or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "hubpin" and v in self.tokens:
                return True
        return False


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Hub Counter</title>
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0b0e14;color:#e7ebf3;font:16px system-ui,sans-serif}
form{background:#141925;padding:28px;border-radius:14px;width:min(320px,90vw)}
input{width:100%;box-sizing:border-box;font-size:28px;letter-spacing:6px;text-align:center;
padding:10px;border-radius:10px;border:1px solid #2a3142;background:#0b0e14;color:#fff}
button{width:100%;margin-top:14px;padding:12px;border:0;border-radius:10px;background:#5b7cfa;
color:#fff;font-size:16px;font-weight:600}p{color:#9aa3b5;font-size:14px}#e{color:#f87171}</style>
</head><body><form id="f"><b>Hub Counter</b><p>Enter the PIN shown on the counting computer.</p>
<input id="p" inputmode="numeric" autocomplete="one-time-code" autofocus>
<button>Open</button><p id="e"></p></form><script>
document.getElementById("f").onsubmit=async e=>{e.preventDefault();
const r=await fetch("/login",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({pin:document.getElementById("p").value})});
if(r.ok)location.href="/";else document.getElementById("e").textContent=(await r.json()).error;};
</script></body></html>"""


def make_handler(ctl: HubController, allow_remote: bool = False,
                 share: Optional[Share] = None):
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

        def _locked(self, path: str) -> bool:
            """On the shared port, true when this request needs the PIN first."""
            return (share is not None and path not in OPEN_PATHS
                    and not share.allowed(self.headers.get("Cookie", "")))

        # -- GET ----------------------------------------------------------------
        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"local only", "text/plain")
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            if self._locked(u.path):
                if u.path == "/":
                    return self._send(200, LOGIN_PAGE.encode(), "text/html; charset=utf-8")
                return self._json({"error": "PIN needed"}, 401)
            if u.path == "/login":
                return self._send(200, LOGIN_PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/board":
                return self._send(200, BOARD.encode(), "text/html; charset=utf-8")
            if u.path == "/api/board":
                return self._json(ctl.board())
            if u.path == "/api/state":
                st = ctl.state(int(q.get("log", 0) or 0))
                st["remote"] = share is not None
                sh = getattr(ctl, "share", None)
                st["share"] = sh.status(host=share is None) if sh else {"on": False}
                return self._json(st)
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
            path = urlparse(self.path).path
            if share is not None and path == "/login":
                try:
                    token = share.login(self.client_address[0], body.get("pin", ""))
                except PermissionError as e:
                    return self._json({"error": str(e)}, 429)
                if token is None:
                    return self._json({"error": "wrong PIN"}, 403)
                self.send_response(200)
                self.send_header("Set-Cookie", f"hubpin={token}; HttpOnly; SameSite=Strict; Path=/")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "11")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
                return
            if self._locked(path):
                return self._json({"error": "PIN needed"}, 401)
            action = path.rsplit("/", 1)[-1]
            if share is not None and action in HOST_ONLY:
                return self._json({"error": "only the counting computer can do that"}, 403)
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
    "add_zone": lambda c, b: c.add_zone(b["camera"], b["hub"], b["points"],
                                        b.get("kind", "outline")),
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
    "quit": lambda c, b: c.quit(),
    # One-click update (the Hub Counter app sets c.updater; tbavid/appupdate.py).
    "update_install": lambda c, b: (c.updater.install(c.running) if c.updater
                                    else {"error": "updates come with the app, not a checkout"}),
    "share_on": lambda c, b: c.share.start(b.get("pin", "")),
    "share_off": lambda c, b: c.share.stop(),
    "load": lambda c, b: c.load(b["path"]),
}


def serve(ctl: HubController, port: int = 8790, bind: str = "127.0.0.1",
          open_browser: bool = True, share: bool = False, pin: str = "") -> None:
    remote = bind not in ("127.0.0.1", "localhost", "::1")
    httpd = ThreadingHTTPServer((bind, port), make_handler(ctl, remote))
    ctl.share = Share(ctl)
    if share:
        st = ctl.share.start(pin)
        print(f"shared on Wi-Fi: {st['url']}  PIN {st['pin']}")
    # Later than the reply to the Quit click, and from another thread:
    # shutdown() waits for serve_forever, which this request is inside.
    ctl.on_quit = lambda: threading.Timer(0.3, httpd.shutdown).start()
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
        ctl.share.stop()
        httpd.server_close()


def main(setup_path: Optional[str] = None, port: int = 8790,
         bind: str = "127.0.0.1", open_browser: bool = True,
         share: bool = False, pin: str = "") -> int:
    serve(HubController(setup_path), port, bind, open_browser, share, pin)
    return 0


# The page lives beside this file so it can be edited as HTML. Read at import:
# it is ~30 kB and never changes while the server runs.
PAGE = (Path(__file__).with_name("hubweb.html")).read_text(encoding="utf-8")
# The scoreboard, for a screen beside the field: /board.
BOARD = (Path(__file__).with_name("hubboard.html")).read_text(encoding="utf-8")
