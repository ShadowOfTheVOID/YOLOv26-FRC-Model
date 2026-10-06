"""Watchtower.app: the Watchtower FMS and the hub counter, set up and started together.

Without it a scrimmage needs a terminal: clone watchtower-fms, make a venv,
`python -m fms.init`, `./run.sh`, then start the hub counter and type
`http://<vision key>@<host>:8000` into it. This does all of that on a
double-click:

- first launch: ~/Documents/Watchtower/config/event.yaml is made with random
  PINs and a vision key (Watchtower's own fms.init), and kept from then on;
- the FMS server starts on port 8000, open to phones on the same Wi-Fi
  (refs, emcee, field display), as `python -m fms.server` would;
- the hub counter page starts on 127.0.0.1:8790 with its address box already
  set to this FMS and its vision key, so counts arrive with nothing typed;
- a start page (Watchtower - start here.html, in that folder) opens with
  every link and PIN, and the phone addresses.

Quit on the hub counter page stops both. The event name, date and teams are
in event.yaml (Watchtower reads them at start): edit, then Quit and reopen.

Watchtower's code is arnan-bajaj/watchtower-fms at the tag in
.github/watchtower-release, bundled by Watchtower.spec; from a checkout,
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


def read_event(d: Path) -> dict:
    import yaml
    ev = yaml.safe_load((d / "config" / "event.yaml").read_text()) or {}
    srv = ev.get("server") or {}
    return {"name": (ev.get("event") or {}).get("name", ""),
            "teams": (ev.get("event") or {}).get("teams") or [],
            "pins": srv.get("pins") or {}, "key": srv.get("vision_key", ""),
            "port": int(srv.get("port") or FMS_PORT)}


def start_page(d: Path, ev: dict, ip: str) -> Path:
    """One local page with every link and PIN. Rewritten each launch: the
    Wi-Fi address changes between venues, and PINs can be edited."""
    p, port, e = ev["pins"], ev["port"], html.escape
    phone = f"http://{ip}:{port}"
    teams = (f"{len(ev['teams'])} teams" if ev["teams"]
             else "<b>no teams yet</b>: add them to event.yaml (below)")
    page = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Watchtower</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#14171c;--mute:#5d6673;--line:#e1e5ea;--acc:#2563eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1115;--card:#171a21;--fg:#e8ebf0;--mute:#98a1ae;--line:#272c35;--acc:#6ea0ff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:760px;margin:0 auto;padding:24px 16px}}
h1{{margin:0 0 4px;font-size:28px}} h2{{font-size:17px;margin:0 0 8px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}}
a{{color:var(--acc)}} .mute{{color:var(--mute)}} code{{font-size:14px;word-break:break-all}}
.pin{{font:600 20px ui-monospace,Menlo,Consolas,monospace;letter-spacing:2px}}
table{{border-collapse:collapse;width:100%}} td{{padding:6px 8px 6px 0;border-top:1px solid var(--line);vertical-align:top}}
</style>
<main>
<h1>Watchtower</h1>
<p class="mute">{e(ev['name'] or 'Event')} · {teams} · running on this computer</p>

<div class="card"><h2>On this computer</h2>
<table>
<tr><td><a href="http://localhost:{port}/">Scorekeeper / field</a></td><td>Control PIN <span class="pin">{e(str(p.get('control', '')))}</span></td></tr>
<tr><td><a href="http://127.0.0.1:{HUB_PORT}/">Hub cameras</a></td><td>Draw the hub outlines, measure a ball, Start. Counts go to Watchtower already: the address box is set.</td></tr>
</table></div>

<div class="card"><h2>On phones and other laptops (same Wi-Fi)</h2>
<p>Open <a href="{phone}/"><code>{phone}</code></a></p>
<table>
<tr><td>Ref</td><td class="pin">{e(str(p.get('ref', '')))}</td></tr>
<tr><td>Emcee</td><td class="pin">{e(str(p.get('emcee', '')))}</td></tr>
</table>
<p class="mute">If phones cannot connect: allow Watchtower when the firewall asks, and use your own router (venue and school Wi-Fi often keep devices apart).</p></div>

<div class="card"><h2>Event settings</h2>
<p>Name, date, teams and PINs are in <code>{e(str(d / 'config' / 'event.yaml'))}</code>.
Edit it in any text editor, then Quit (on the hub camera page) and open Watchtower again.</p></div>

<div class="card"><h2>Stop</h2>
<p><b>Quit</b> on the <a href="http://127.0.0.1:{HUB_PORT}/">hub camera page</a> stops Watchtower and the counter. Closing tabs leaves both running; open Watchtower again to get this page back.</p></div>
</main>"""
    out = d / "Watchtower - start here.html"
    out.write_text(page, encoding="utf-8")
    return out


class FmsServer:
    """Watchtower's FastAPI app under uvicorn, in a thread, as `python -m
    fms.server` runs it (same host and port from event.yaml)."""

    def __init__(self, host: str, port: int):
        import uvicorn
        from fms import server             # reads config/event.yaml at import
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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="Watchtower", description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open the start page (a computer with no screen, a test)")
    args, _ = ap.parse_known_args()
    d = data_dir()
    if port_open(HUB_PORT) or port_open(FMS_PORT):
        # Opened again while running: just show the page again. (A port taken
        # by something else shows up here too; the log says so.)
        page = d / "Watchtower - start here.html"
        if not args.no_browser and page.exists():
            webbrowser.open(page.as_uri())
        return 0
    os.chdir(d)                      # Watchtower's config/ and data/ are relative
    if getattr(sys, "frozen", False):
        log = open(d / "watchtower.log", "a", buffering=1)
        sys.stdout = sys.stderr = log
    os.environ.setdefault("YOLO_CONFIG_DIR", str(d / "ultralytics"))
    src = watchtower_src()
    sys.path.insert(0, str(src))
    made = first_setup(d, src)
    if made["pins"]:
        print(f"first launch: made {d / 'config' / 'event.yaml'} with new PINs")
    os.environ["FMS_CONFIG"] = str(d / "config" / "event.yaml")
    ev = read_event(d)

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
    ctl.default_target = f"http://{ev['key']}@127.0.0.1:{ev['port']}"
    page = start_page(d, ev, lan_ip())
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(page.as_uri())).start()
    try:
        hubweb.serve(ctl, HUB_PORT, "127.0.0.1", open_browser=False)
    finally:
        fms.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
