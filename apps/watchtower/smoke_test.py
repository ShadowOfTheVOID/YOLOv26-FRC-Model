"""Smoke test a built Watchtower app on any OS: python smoke_test.py PATH_TO_PROGRAM

Starts it with a scratch home folder and checks what a double-click must
give: the first-launch event.yaml with PINs and the start page; the FMS on
:8000; the hub counter on :8790 with its address box already set to that FMS;
a count accepted with that address's vision key (and refused with a wrong
one); and Quit stopping both. A bundle missing a uvicorn protocol module, the
FMS pages or the example configs fails here, not at a venue.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

FMS = "http://127.0.0.1:8000"
HUB = "http://127.0.0.1:8790"


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


def main(program: str) -> int:
    home = tempfile.mkdtemp(prefix="watchtower-home-")
    env = dict(os.environ, HOME=home, USERPROFILE=home)
    proc = subprocess.Popen([program, "--no-browser"], env=env)
    try:
        for _ in range(120):                # torch-sized bundles start slowly
            if up(FMS + "/") and up(HUB + "/"):
                break
            if proc.poll() is not None:
                fail(f"exited with code {proc.returncode} before both pages came up")
                return 1
            time.sleep(1)
        else:
            fail("the FMS (:8000) and the hub counter (:8790) never both came up")
            return 1
        d = os.path.join(home, "Documents", "Watchtower")
        assert os.path.isfile(os.path.join(d, "config", "event.yaml")), "no event.yaml made"
        assert os.path.isfile(os.path.join(d, "Watchtower - start here.html")), "no start page"
        target = json.loads(get(HUB + "/api/state"))["default_target"]
        assert target.startswith("http://") and target.endswith("@127.0.0.1:8000"), \
            "the hub counter is not set to this Watchtower"
        key = target[len("http://"):target.index("@")]
        code = post(FMS + "/api/vision/events", {"events": {"red": [[1.0, 1]]}},
                    {"X-Vision-Key": key})
        assert code == 200, f"Watchtower refused the preset vision key ({code})"
        code = post(FMS + "/api/vision/events", {}, {"X-Vision-Key": "wrong"})
        assert code == 401, f"a wrong vision key got {code}, not 401"
        assert post(HUB + "/api/quit", {}) == 200
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            fail("still running 30 s after Quit")
            return 1
        if up(FMS + "/"):
            fail("Quit left the FMS running")
            return 1
        print(f"OK: first-launch setup, FMS, hub counter preset to it, vision key, Quit "
              f"(exit code {proc.returncode})")
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except (AssertionError, OSError, subprocess.TimeoutExpired) as e:
        fail(f"{type(e).__name__}: {e}")
        sys.exit(1)
