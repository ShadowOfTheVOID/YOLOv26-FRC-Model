"""Hub Counter.app: the hub fuel counter's web page, as a double-click app.

The same page as `run.py hubgui` (tbavid/hubweb.py), without a terminal: it
opens in the default browser and stops from its Quit button. Only the hub
counter is bundled -- not the TBA scraper or the training tools.

Where things go: ~/Documents/Hub Counter/ holds cams.json (the setup, opened
again next launch) and the hubfeed_*.csv count logs, because an app bundle is
read-only and a user looks in Documents.

Opening the app while it already runs (the browser tab was closed, say) opens
the page again instead of failing on the taken port.
"""
from __future__ import annotations

import os
import socket
import sys
import webbrowser
from pathlib import Path

PORT = 8790
URL = f"http://127.0.0.1:{PORT}/"


def data_dir() -> Path:
    d = Path.home() / "Documents" / "Hub Counter"
    d.mkdir(parents=True, exist_ok=True)
    return d


def already_running(port: int = PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def ultralytics_home() -> None:
    """Ultralytics writes a settings file at import, by default under
    ~/.config, which a sandboxed or managed account may refuse (the build
    test's scratch home did, with a warning per launch)."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(data_dir() / "ultralytics"))


def selftest_model(out: str) -> int:
    """`Hub Counter --selftest-model RESULT_FILE`, for the build's smoke test:
    load the built-in model and run it once. Exit 0 if it ran. The result
    goes to a file because a windowed Windows build has no stdout.

    What this proves is that the model and its libraries are inside the
    build. If the GPU device fails -- GitHub's macOS runners are VMs, where
    Apple's GPU (MPS) may not work as on a real Mac -- it says so and tries
    the CPU, which exercises the same bundled code."""
    import time
    msg, code = "", 1
    try:
        import numpy as np
        from tbavid.hubmodel import BUILTIN, ModelEye, pick_device
        frame = np.zeros((720, 1280, 3), np.uint8)
        poly = {"red": [(100, 100), (300, 100), (300, 200), (100, 200)]}
        notes = []
        for device in dict.fromkeys([pick_device(), "cpu"]):
            t = time.time()
            try:
                ModelEye(BUILTIN, poly, device).detect(frame)
                msg, code = "; ".join(notes + [f"model ran on {device} in {time.time() - t:.1f} s"]), 0
                break
            except Exception as e:                  # report it, then the CPU
                notes.append(f"{device} failed: {type(e).__name__}: {e}")
        else:
            msg = "; ".join(notes)
    except Exception as e:                          # report anything, never hang
        msg = f"{type(e).__name__}: {e}"
    with open(out, "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    return code


def main() -> int:
    ultralytics_home()
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest-model":
        return selftest_model(sys.argv[2])
    if already_running():
        webbrowser.open(URL)
        return 0
    d = data_dir()
    os.chdir(d)                     # file pickers and logs start here
    # Frozen apps have no console; keep a log a user can send when asking why.
    if getattr(sys, "frozen", False):
        log = open(d / "hubcounter.log", "a", buffering=1)
        sys.stdout = sys.stderr = log
    from tbavid import hubweb
    return hubweb.main(str(d / "cams.json"), PORT, "127.0.0.1", True)


if __name__ == "__main__":
    sys.exit(main())
