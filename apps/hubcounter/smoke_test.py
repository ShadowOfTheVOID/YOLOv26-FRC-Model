"""Smoke test a built Hub Counter on any OS: python smoke_test.py PATH_TO_PROGRAM

Starts the program with a scratch home folder, then checks that the page
and the scoreboard load, that ~/Documents/Hub Counter was made, and that
the page's Quit ends the program; with HUBCOUNTER_REQUIRE_MODEL=1, also that
the built-in fuel model loads and runs inside it. A frozen build that cannot find its HTML,
or never exits, fails here instead of on someone's laptop.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

URL = "http://127.0.0.1:8790"


def get(path: str) -> str:
    with urllib.request.urlopen(URL + path, timeout=5) as r:
        return r.read().decode("utf-8", "replace")


def fail(msg: str) -> None:
    """Print, and annotate under GitHub Actions: the annotation shows on the
    PR's checks, where a failure is read without opening the raw log."""
    print("FAIL: " + msg)
    if os.environ.get("GITHUB_ACTIONS"):
        print("::error title=Hub Counter smoke test::" + msg.replace("\n", " "))


def main(program: str) -> int:
    home = tempfile.mkdtemp(prefix="hubcounter-home-")
    env = dict(os.environ, HOME=home, USERPROFILE=home)
    proc = subprocess.Popen([program], env=env)
    try:
        for _ in range(60):                 # Windows and first launches are slow
            try:
                page = get("/")
                break
            except OSError:
                time.sleep(1)
        else:
            fail("the page never came up")
            return 1
        assert "Hub Counter" in page, "the page is not the hub counter's"
        get("/board")
        docs = os.path.join(home, "Documents", "Hub Counter")
        assert os.path.isdir(docs), f"{docs} was not made"
        req = urllib.request.Request(URL + "/api/quit", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"]
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            fail("still running 30 s after Quit")
            return 1
        print(f"OK: page, board, data folder, Quit (exit code {proc.returncode})")
    finally:
        if proc.poll() is None:
            proc.kill()
    # The built-in model, run once inside the build: catches a missing
    # Ultralytics yaml, a torch library the bundler left out, or no model.
    if os.environ.get("HUBCOUNTER_REQUIRE_MODEL") == "1":
        result = os.path.join(home, "model.txt")
        code = subprocess.run([program, "--selftest-model", result], env=env,
                              timeout=600).returncode
        msg = open(result).read().strip() if os.path.exists(result) else "no result"
        if code == 0:
            print("OK: " + msg)
            # Shown on the PR's checks: which device the model ran on.
            print(f"::notice title=Built-in model::{msg}")
        else:
            fail(f"the built-in model did not run (exit {code}): {msg}")
        return code
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except (AssertionError, OSError, subprocess.TimeoutExpired) as e:
        fail(f"{type(e).__name__}: {e}")
        sys.exit(1)
