"""Smoke test a built Watchtower app on any OS: python smoke_test.py PATH_TO_PROGRAM

Starts it with a scratch home folder and checks what a double-click must
give: the first-launch event.yaml with PINs; the FMS on :8000 serving
Control and the match queue; Watchtower's own hub-camera runner reporting to
it (a second copy of the program, `--vision`); a count accepted with
event.yaml's vision key and refused with a wrong one; and stopping the app
stopping both. A bundle missing a uvicorn protocol module, the FMS pages,
a vision module or the example configs fails here, not at a venue.
"""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

FMS = "http://127.0.0.1:8000"


def fail(msg: str) -> None:
    print("FAIL: " + msg)
    if os.environ.get("GITHUB_ACTIONS"):
        print("::error title=Watchtower smoke test::" + msg.replace("\n", " "))


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode("utf-8", "replace")


def post(url: str, body: dict, headers: dict = None) -> int:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def up(url: str) -> bool:
    try:
        get(url)
        return True
    except OSError:
        return False


def log_tail(home: str, name: str) -> str:
    p = os.path.join(home, "Documents", "Watchtower", name)
    lines = open(p, errors="replace").read().splitlines()[-15:] if os.path.exists(p) else []
    return " / ".join(l.strip() for l in lines if l.strip())[-1500:]


def stop(proc) -> None:
    """What the system's quit does: SIGTERM (Windows has only terminate)."""
    if sys.platform == "win32":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGTERM)


def main(program: str) -> int:
    home = tempfile.mkdtemp(prefix="watchtower-home-")
    env = dict(os.environ, HOME=home, USERPROFILE=home)
    proc = subprocess.Popen([program, "--no-browser"], env=env)
    try:
        for _ in range(120):
            if up(FMS + "/control"):
                break
            if proc.poll() is not None:
                fail(f"exited with code {proc.returncode} before Control came up | {log_tail(home, 'watchtower.log')}")
                return 1
            time.sleep(1)
        else:
            fail("the FMS (:8000) never came up | " + log_tail(home, "watchtower.log"))
            return 1
        d = os.path.join(home, "Documents", "Watchtower")
        ev = open(os.path.join(d, "config", "event.yaml")).read()
        assert re.search(r'control: "\d{6}"', ev), "event.yaml has no Control PIN"
        assert "Match Queue" in get(FMS + "/queue"), "the match queue page is not served"
        # Watchtower's own camera runner, a second copy of the program, reports
        # each hub ("no camera" until Setup picks one).
        for _ in range(60):
            vision = json.loads(get(FMS + "/api/state")).get("vision") or {}
            if {"red", "blue"} <= set(vision):
                break
            time.sleep(1)
        else:
            fail("the hub-camera runner never reported | " + log_tail(home, "vision.log"))
            return 1
        key = re.search(r'vision_key: "([^"]+)"', ev).group(1)
        code = post(FMS + "/api/vision/events", {"events": {"red": [[1.0, 1]]}}, {"X-Vision-Key": key})
        assert code == 200, f"Watchtower refused event.yaml's vision key ({code})"
        code = post(FMS + "/api/vision/events", {}, {"X-Vision-Key": "wrong"})
        assert code == 401, f"a wrong vision key got {code}, not 401"
        stop(proc)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            fail("still running 30 s after being told to quit")
            return 1
        time.sleep(2)
        if up(FMS + "/control"):
            fail("quitting left the FMS running")
            return 1
        print(f"OK: first-launch setup, Control, match queue, hub cameras reporting "
              f"({', '.join(sorted(vision))}), vision key, quit (exit code {proc.returncode})")
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()


def window_test(program: str) -> int:
    """--window: the real app windows (WebKit on macOS, WebView2 on Windows):
    Control loads in the main window and the field display opens in another.
    Needs a desktop session, which GitHub's macOS and Windows runners have."""
    home = tempfile.mkdtemp(prefix="watchtower-home-")
    env = dict(os.environ, HOME=home, USERPROFILE=home)
    out = os.path.join(home, "window.json")
    try:
        code = subprocess.run([program, "--selftest-window", out], env=env, timeout=300).returncode
    except subprocess.TimeoutExpired:
        log = os.path.join(home, "Documents", "Watchtower", "watchtower.log")
        tail = open(log, errors="replace").read().splitlines()[-25:] if os.path.exists(log) else []
        fail("app windows: no answer in 300 s | log: " + " / ".join(l.strip() for l in tail if l.strip())[-1500:])
        return 1
    res = json.load(open(out)) if os.path.exists(out) else {"ok": False, "error": f"no result (exit {code})"}
    for s in res.get("steps", []):
        print("  " + s)
    if not res.get("ok"):
        fail("app windows: " + res.get("error", "failed"))
        return 1
    print("OK: app windows (" + "; ".join(res.get("steps", [])) + ")")
    if os.environ.get("GITHUB_ACTIONS"):
        print("::notice title=Watchtower windows::" + "; ".join(res.get("steps", [])))
    return 0


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--window"]:
            sys.exit(window_test(sys.argv[2]))
        sys.exit(main(sys.argv[1]))
    except (AssertionError, OSError, subprocess.TimeoutExpired) as e:
        fail(f"{type(e).__name__}: {e}")
        sys.exit(1)
