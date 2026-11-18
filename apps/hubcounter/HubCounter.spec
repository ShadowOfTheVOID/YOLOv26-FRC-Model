# PyInstaller spec for Hub Counter: an .app on macOS, a folder with
# "Hub Counter.exe" on Windows, a folder with "Hub Counter" on Linux. Each is
# built on its own OS by .github/workflows/hub-app.yml; by hand, on that OS:
#   pip install pyinstaller opencv-python-headless numpy yt-dlp onnxruntime
#   (Windows: onnxruntime-directml instead, for the GPU-or-NPU fallback)
#   python train/export_onnx.py models/fuel_relabel.pt   (needs torch + ultralytics,
#       in another venv: they must NOT be in the one that builds)
#   pyinstaller apps/hubcounter/HubCounter.spec      (from the repo root)
# (hub-app.yml does exactly this.) Until 2026-11-18 the apps carried torch +
# Ultralytics: the Linux arm64 build reached 3049 MB with PyPI's CUDA torch,
# and the CPU builds were still hundreds of MB. ONNX Runtime runs the same
# weights (export_onnx.py refuses an export whose boxes differ).
#
# What goes in: tbavid's hub modules and the two pages they serve, OpenCV,
# ONNX Runtime for the colour+model blend, and models/fuel_relabel.onnx
# (exported by the workflow from the release's .pt; `model: built-in`). The scraper (tba,
# download, pipeline, ...) is never imported by them and is excluded.
# HUBCOUNTER_REQUIRE_MODEL=1 makes a build without the model fail instead of
# shipping one whose "Use built-in model" button is missing.
#
# The same spec builds the Watchtower app (apps/watchtower/) with
# HUBAPP=watchtower: the Watchtower FMS and its own hub-camera runner from a
# clone of arnan-bajaj/watchtower-fms at WATCHTOWER_SRC (hub-app.yml checks
# out the commit in .github/watchtower-release). Not our hub page (Watchtower
# v2's cameras are in its own Control -> Setup), but the fuel model and ONNX
# Runtime: the model is every hub's primary counter there too.
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
         "tensorboard", "onnx", "openvino", "tensorrt",
         # the model runs on onnxruntime; a .pt chosen by hand needs a checkout
         "torch", "torchvision", "ultralytics", "rfdetr", "matplotlib", "scipy"]

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
    VSRC = os.path.join(WSRC, "vision")
    sys.path[:0] = [WSRC, VSRC]
    NAME, ENTRY, BUNDLE_ID = "Watchtower", os.path.join(ROOT, "apps", "watchtower", "main.py"), "org.tbavid.watchtower"
    WAPP = os.path.join(ROOT, "apps", "watchtower")
    EXTRA_PATH = [WSRC, VSRC, WAPP]
    EXTRA_DATAS = [(os.path.join(WAPP, "queue.html"), "."),
                   (os.path.join(WAPP, "theme.css"), "."),
                   (os.path.join(WSRC, "fms", "static"), "fms/static"),
                   (os.path.join(WSRC, "config", "event.example.yaml"), "config"),
                   (os.path.join(WSRC, "config", "vision.example.yaml"), "config"),
                   # run_vision.py is run from its folder (Watchtower --vision);
                   # the counter plugins folder is read at start.
                   (VSRC, "vision")]
    # uvicorn picks its loop, protocols and lifespan by name at start; vision's
    # modules import each other by bare name from vision/.
    VISION = ["run_vision", "dropfit", "hubsetup", "vconfig", "count_feed", "scaler",
              "pick_roi", "rescore"]
    EXTRA_IMPORTS = (collect_submodules("fms") + collect_submodules("uvicorn")
                     + collect_submodules("counters") + collect_submodules("hubcounter")
                     + ["fastapi", "starlette", "websockets", "yaml", "requests", "queue_tba",
                        "tbavid.keepawake", "tbavid.cvload",
                        # the fuel model, the primary counter (main.use_our_model_runtime)
                        "tbavid.hubmodel", "tbavid.onnxdet", "tbavid.count", "tbavid.trackvis",
                        "tbavid.fms_counter", "tbavid.hubcount", "tbavid.camprofile",
                        "onnxruntime"] + VISION + WEBVIEW)
    EXTRA_DATAS += collect_data_files("webview")
    HUB_IMPORTS = ["cv2", "numpy"]
else:
    NAME, ENTRY, BUNDLE_ID = "Hub Counter", os.path.join(SPECPATH, "main.py"), "org.tbavid.hubcounter"
    EXTRA_PATH = []
    EXTRA_DATAS = collect_data_files("webview")
    EXTRA_IMPORTS = WEBVIEW
    HUB_IMPORTS = ["tbavid.hubweb", "tbavid.hubapp", "tbavid.hubcount",
                   "tbavid.hubfeed", "tbavid.fmslink", "tbavid.hubmodel",
                   "tbavid.count", "tbavid.trackvis", "tbavid.camcheck",
                   "tbavid.camprofile", "tbavid.onnxdet", "cv2", "numpy", "yt_dlp",
                   "onnxruntime"]

# The build's version, read at run time by tbavid/appupdate.py to offer
# one-click updates (0.0.0 for pull-request builds, which offer none).
import tempfile
_VDIR = tempfile.mkdtemp(prefix="appversion-")
with open(os.path.join(_VDIR, "app_version.txt"), "w") as _f:
    _f.write(os.environ.get("HUBCOUNTER_VERSION", "0.0.0").lstrip("v"))
EXTRA_DATAS = EXTRA_DATAS + [(os.path.join(_VDIR, "app_version.txt"), ".")]
EXTRA_IMPORTS = EXTRA_IMPORTS + ["tbavid.appupdate"]

MODEL = os.path.join(ROOT, "models", "fuel_relabel.onnx")
if not os.path.isfile(MODEL) and os.environ.get("HUBCOUNTER_REQUIRE_MODEL") == "1":
    raise SystemExit(f"{MODEL} is missing (hub-app.yml downloads it from the release)")
MODELS = [(MODEL, "models")] if os.path.isfile(MODEL) else []


a = Analysis(
    [ENTRY],
    pathex=[ROOT] + EXTRA_PATH,
    binaries=[],
    datas=([(os.path.join(TB, "hubweb.html"), "tbavid"),
            (os.path.join(TB, "hubboard.html"), "tbavid")] if APP != "watchtower" else [])
          + MODELS + EXTRA_DATAS,
    hiddenimports=HUB_IMPORTS + EXTRA_IMPORTS,
    excludes=SCRAPER + HEAVY,
    noarchive=False,
)
pyz = PYZ(a.pure)
# The app's own icon, not PyInstaller's default (the Python snake on
# Windows). Drawn by apps/icons/make_icons.py; the files are committed.
ICONS = os.path.join(ROOT, "apps", "icons", "Watchtower" if APP == "watchtower" else "HubCounter")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME,
          console=False, upx=False,
          icon=ICONS + (".icns" if sys.platform == "darwin" else ".ico"))
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=ICONS + ".icns",
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
