# PyInstaller spec for Hub Counter: an .app on macOS, a folder with
# "Hub Counter.exe" on Windows, a folder with "Hub Counter" on Linux. Each is
# built on its own OS by .github/workflows/hub-app.yml; by hand, on that OS:
#   pip install pyinstaller opencv-python-headless numpy yt-dlp torch torchvision
#   pip install --no-deps ultralytics ultralytics-thop
#   pip install matplotlib pillow pyyaml requests psutil polars
#   pyinstaller apps/hubcounter/HubCounter.spec      (from the repo root)
# (hub-app.yml does exactly this; on Linux, x64 and arm64, it takes torch from
# PyTorch's CPU index: PyPI's Linux torch carries CUDA, and the arm64 build
# came out at 3049 MB, over GitHub's 2 GB asset limit.)
#
# What goes in: tbavid's hub modules and the two pages they serve, OpenCV,
# PyTorch + Ultralytics for the colour+model blend, and models/fuel_relabel.pt
# (downloaded by the workflow; `model: built-in`). The scraper (tba,
# download, pipeline, ...) is never imported by them and is excluded.
# HUBCOUNTER_REQUIRE_MODEL=1 makes a build without the model fail instead of
# shipping one whose "Use built-in model" button is missing.
#
# The same spec builds the Watchtower app (apps/watchtower/) with
# HUBAPP=watchtower: the hub counter plus the Watchtower FMS from a clone of
# arnan-bajaj/watchtower-fms at WATCHTOWER_SRC (hub-app.yml clones the tag
# in .github/watchtower-release). One spec, so the model, torchvision and
# page fixes above apply to both.
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

APP = os.environ.get("HUBAPP", "hubcounter")
# Both apps open in their own windows (pywebview): only this OS's backend,
# or the bundler drags in Qt or GTK modules that are not installed and warns.
WEBVIEW_OS = {"darwin": ("cocoa",), "win32": ("winforms", "edgechromium", "mshtml")}.get(
    sys.platform, ("gtk", "qt"))
WEBVIEW = collect_submodules(
    "webview", filter=lambda m: not m.startswith("webview.platforms.")
    or m.split(".")[2] in WEBVIEW_OS)
if APP == "watchtower":
    WSRC = os.path.abspath(os.environ.get("WATCHTOWER_SRC") or os.path.join(ROOT, "watchtower-src"))
    if not os.path.isfile(os.path.join(WSRC, "fms", "server.py")):
        raise SystemExit(f"no Watchtower at {WSRC}: clone arnan-bajaj/watchtower-fms there")
    sys.path.insert(0, WSRC)
    NAME, ENTRY, BUNDLE_ID = "Watchtower", os.path.join(ROOT, "apps", "watchtower", "main.py"), "org.tbavid.watchtower"
    WAPP = os.path.join(ROOT, "apps", "watchtower")
    EXTRA_PATH = [WSRC, WAPP]
    EXTRA_DATAS = [(os.path.join(WAPP, "home.html"), "."),
                   (os.path.join(WSRC, "fms", "static"), "fms/static"),
                   (os.path.join(WSRC, "config", "event.example.yaml"), "config"),
                   (os.path.join(WSRC, "config", "vision.example.yaml"), "config")]
    # uvicorn picks its loop, protocols and lifespan by name at start.
    EXTRA_IMPORTS = (collect_submodules("fms") + collect_submodules("uvicorn")
                     + ["fastapi", "starlette", "websockets", "yaml", "requests", "settings"]
                     + collect_submodules("qrcode") + WEBVIEW)
    EXTRA_DATAS += collect_data_files("webview")
else:
    NAME, ENTRY, BUNDLE_ID = "Hub Counter", os.path.join(SPECPATH, "main.py"), "org.tbavid.hubcounter"
    EXTRA_PATH = []
    EXTRA_DATAS = collect_data_files("webview")
    EXTRA_IMPORTS = WEBVIEW

# The build's version, read at run time by tbavid/appupdate.py to offer
# one-click updates (0.0.0 for pull-request builds, which offer none).
import tempfile
_VDIR = tempfile.mkdtemp(prefix="appversion-")
with open(os.path.join(_VDIR, "app_version.txt"), "w") as _f:
    _f.write(os.environ.get("HUBCOUNTER_VERSION", "0.0.0").lstrip("v"))
EXTRA_DATAS = EXTRA_DATAS + [(os.path.join(_VDIR, "app_version.txt"), ".")]
EXTRA_IMPORTS = EXTRA_IMPORTS + ["tbavid.appupdate"]

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
    [ENTRY],
    pathex=[ROOT] + EXTRA_PATH,
    binaries=torchvision_ops(),
    datas=[(os.path.join(TB, "hubweb.html"), "tbavid"),
           (os.path.join(TB, "hubboard.html"), "tbavid")] + MODELS
          # Ultralytics reads its default.yaml and tracker yamls at import
          # and builds layers by name, so all of it goes in.
          + collect_data_files("ultralytics") + EXTRA_DATAS,
    hiddenimports=["tbavid.hubweb", "tbavid.hubapp", "tbavid.hubcount",
                   "tbavid.hubfeed", "tbavid.fmslink", "tbavid.hubmodel",
                   "tbavid.count", "tbavid.trackvis", "cv2", "numpy", "yt_dlp",
                   "torch", "torchvision"] + collect_submodules("ultralytics") + EXTRA_IMPORTS,
    excludes=SCRAPER + HEAVY,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME,
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        bundle_identifier=BUNDLE_ID,
        info_plist={
            "CFBundleShortVersionString": os.environ.get("HUBCOUNTER_VERSION", "0.0.0"),
            # Without these macOS refuses the camera, and since macOS 15 the
            # local network (UDP to bioarena, HTTP to frc-fms), without asking.
            # The combo runs the model on Apple silicon's GPU (MPS).
            "NSCameraUsageDescription":
                f"{NAME} counts fuel from the hub cameras.",
            "NSLocalNetworkUsageDescription":
                ("Watchtower serves the ref, emcee and field pages to phones on this "
                 "network, and reads Wi-Fi cameras." if APP == "watchtower" else
                 "Hub Counter sends counts to the field system (bioarena or frc-fms) "
                 "and reads Wi-Fi cameras on this network."),
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
        },
    )
