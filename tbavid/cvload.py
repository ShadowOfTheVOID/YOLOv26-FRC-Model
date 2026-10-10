"""Load OpenCV in the bundled apps even when its own loader refuses.

The Watchtower app on a Mac (v0.5.7, 2026-10-10) could not open a camera:
every `import cv2` raised OpenCV's 'recursion is detected during loading of
"cv2" binary extensions', on a fresh start with no camera set up. The same
build loads OpenCV on Linux. opencv-python's __init__ is a loader: it sets
sys.OpenCV_LOADER, puts the extension's folder on sys.path and imports
"cv2" again to reach the compiled module. In a frozen app that second import
can find the package again instead of the binary, and a failed attempt
leaves the flag set, so every later import fails the same way.

load() imports cv2 normally; if that fails, it clears the flag and loads the
compiled module (cv2*.so / cv2*.pyd beside the package) directly, which is
the whole OpenCV API. The apps call it once, on the main thread, at start,
so every later `import cv2` gets the loaded module. Standard library only.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from typing import List, Optional

error: str = ""          # why the normal import failed, for the log


def _candidates() -> List[str]:
    roots = list(sys.path)
    base = getattr(sys, "_MEIPASS", "")
    if base:
        roots += [base, os.path.join(os.path.dirname(base), "Frameworks"),
                  os.path.join(os.path.dirname(base), "Resources")]
    roots.append(os.path.dirname(os.path.abspath(sys.executable)))
    found, seen = [], set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for pat in ("cv2/cv2*.so", "cv2/cv2*.pyd", "cv2/python-3*/cv2*.so",
                    "cv2/python-3*/cv2*.pyd"):
            for p in glob.glob(os.path.join(root, pat)):
                real = os.path.realpath(p)
                if real not in seen:
                    seen.add(real)
                    found.append(p)
    return found


def load():
    """The cv2 module, or None (with `error` saying why)."""
    global error
    try:
        import cv2
        return cv2
    except ImportError as e:
        error = str(e)
    if hasattr(sys, "OpenCV_LOADER"):
        try:
            del sys.OpenCV_LOADER
        except AttributeError:
            pass
    sys.modules.pop("cv2", None)
    last: Optional[BaseException] = None
    for path in _candidates():
        try:
            spec = importlib.util.spec_from_file_location("cv2", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["cv2"] = mod
            spec.loader.exec_module(mod)
            print(f"OpenCV: its loader failed ({error}); loaded {path} directly", flush=True)
            return mod
        except Exception as e:                     # noqa: BLE001 -- try the next one
            sys.modules.pop("cv2", None)
            last = e
    if last is not None:
        error += f"; loading the binary directly failed too: {last}"
    print(f"OpenCV could not be loaded: {error}", flush=True)
    return None
