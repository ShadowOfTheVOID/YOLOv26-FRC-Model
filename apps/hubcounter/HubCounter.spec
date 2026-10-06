# PyInstaller spec for Hub Counter: an .app on macOS, a folder with
# "Hub Counter.exe" on Windows, a folder with "Hub Counter" on Linux. Each is
# built on its own OS by .github/workflows/hub-app.yml; by hand, on that OS:
#   pip install pyinstaller opencv-python-headless numpy yt-dlp torch torchvision
#   pip install --no-deps ultralytics ultralytics-thop
#   pip install matplotlib pillow pyyaml requests psutil polars
#   pyinstaller apps/hubcounter/HubCounter.spec      (from the repo root)
# (hub-app.yml does exactly this; on Linux x64 it takes torch from PyTorch's
# CPU index, since PyPI's Linux torch carries CUDA and would be ~3 GB.)
#
# What goes in: tbavid's hub modules and the two pages they serve, OpenCV,
# PyTorch + Ultralytics for the colour+model blend, and models/fuel_relabel.pt
# (downloaded by the workflow; `model: built-in`). The scraper (tba,
# download, pipeline, ...) is never imported by them and is excluded.
# HUBCOUNTER_REQUIRE_MODEL=1 makes a build without the model fail instead of
# shipping one whose "Use built-in model" button is missing.
import os
import sys

import glob
import importlib.util

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
TB = os.path.join(ROOT, "tbavid")

SCRAPER = ["tbavid.tba", "tbavid.download", "tbavid.pipeline", "tbavid.stream",
           "tbavid.shots", "tbavid.crop", "tbavid.formats", "tbavid.render",
           "tbavid.scoreboard", "tbavid.audio", "tbavid.db", "tbavid.api",
           "tbavid.detect", "tbavid.shooting"]
HEAVY = ["tkinter", "PyQt5", "PyQt6", "PySide6", "IPython", "jupyter",
         "tensorboard", "onnx", "onnxruntime", "openvino", "tensorrt"]

MODEL = os.path.join(ROOT, "models", "fuel_relabel.pt")
if not os.path.isfile(MODEL) and os.environ.get("HUBCOUNTER_REQUIRE_MODEL") == "1":
    raise SystemExit(f"{MODEL} is missing (hub-app.yml downloads it from the release)")
MODELS = [(MODEL, "models")] if os.path.isfile(MODEL) else []


def torchvision_ops():
    """torchvision's compiled ops (_C_stable: nms, roi_align), which it loads
    by path with torch.ops.load_library, so the bundler never sees them;
    collect_dynamic_libs skips them too (no lib* name). Without them the
    first prediction failed: "operator torchvision::nms does not exist"."""
    tv = os.path.dirname(importlib.util.find_spec("torchvision").origin)
    found = [f for ext in ("*.so", "*.pyd", "*.dylib", "*.dll")
             for f in glob.glob(os.path.join(tv, ext))]
    return [(f, "torchvision") for f in found]

a = Analysis(
    [os.path.join(SPECPATH, "main.py")],
    pathex=[ROOT],
    binaries=torchvision_ops(),
    datas=[(os.path.join(TB, "hubweb.html"), "tbavid"),
           (os.path.join(TB, "hubboard.html"), "tbavid")] + MODELS
          # Ultralytics reads its default.yaml and tracker yamls at import
          # and builds layers by name, so all of it goes in.
          + collect_data_files("ultralytics"),
    hiddenimports=["tbavid.hubweb", "tbavid.hubapp", "tbavid.hubcount",
                   "tbavid.hubfeed", "tbavid.fmslink", "tbavid.hubmodel",
                   "tbavid.count", "tbavid.trackvis", "cv2", "numpy", "yt_dlp",
                   "torch", "torchvision"] + collect_submodules("ultralytics"),
    excludes=SCRAPER + HEAVY,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Hub Counter",
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Hub Counter", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Hub Counter.app",
        bundle_identifier="org.tbavid.hubcounter",
        info_plist={
            "CFBundleShortVersionString": os.environ.get("HUBCOUNTER_VERSION", "0.0.0"),
            # Without these macOS refuses the camera, and since macOS 15 the
            # local network (UDP to bioarena, HTTP to frc-fms), without asking.
            # The combo runs the model on Apple silicon's GPU (MPS).
            "NSCameraUsageDescription":
                "Hub Counter counts fuel from the hub cameras.",
            "NSLocalNetworkUsageDescription":
                "Hub Counter sends counts to the field system (bioarena or frc-fms) "
                "and reads Wi-Fi cameras on this network.",
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
        },
    )
