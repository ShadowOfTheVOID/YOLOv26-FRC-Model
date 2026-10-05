# PyInstaller spec for Hub Counter.app (macOS). Built by
# .github/workflows/mac-app.yml on a macOS runner; to build by hand on a Mac:
#   pip install pyinstaller opencv-python-headless numpy yt-dlp
#   pyinstaller apps/hubcounter/HubCounter.spec      (from the repo root)
#
# Only the hub counter goes in: tbavid's hub modules and the two pages they
# serve. The scraper (tba, download, pipeline, ...) is never imported by them
# and is excluded outright, as are torch / ultralytics: the colour counter
# needs neither, and with them the app is over 1 GB. The fuel-model blend
# (--model) therefore stays a terminal-version feature; the page says so.
import os

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
TB = os.path.join(ROOT, "tbavid")

SCRAPER = ["tbavid.tba", "tbavid.download", "tbavid.pipeline", "tbavid.stream",
           "tbavid.shots", "tbavid.crop", "tbavid.formats", "tbavid.render",
           "tbavid.scoreboard", "tbavid.audio", "tbavid.db", "tbavid.api",
           "tbavid.detect", "tbavid.shooting"]
HEAVY = ["torch", "torchvision", "ultralytics", "matplotlib", "pandas",
         "polars", "scipy", "tkinter", "PyQt5", "PyQt6", "PySide6"]

a = Analysis(
    [os.path.join(SPECPATH, "main.py")],
    pathex=[ROOT],
    datas=[(os.path.join(TB, "hubweb.html"), "tbavid"),
           (os.path.join(TB, "hubboard.html"), "tbavid")],
    hiddenimports=["tbavid.hubweb", "tbavid.hubapp", "tbavid.hubcount",
                   "tbavid.hubfeed", "tbavid.fmslink", "tbavid.hubmodel",
                   "tbavid.count", "tbavid.trackvis", "cv2", "numpy", "yt_dlp"],
    excludes=SCRAPER + HEAVY,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Hub Counter",
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Hub Counter", upx=False)
app = BUNDLE(
    coll,
    name="Hub Counter.app",
    bundle_identifier="org.tbavid.hubcounter",
    info_plist={
        "CFBundleShortVersionString": os.environ.get("HUBCOUNTER_VERSION", "0.0.0"),
        # Without these macOS refuses the camera, and since macOS 15 the
        # local network (UDP to bioarena, HTTP to frc-fms), without asking.
        "NSCameraUsageDescription":
            "Hub Counter counts fuel from the hub cameras.",
        "NSLocalNetworkUsageDescription":
            "Hub Counter sends counts to the field system (bioarena or frc-fms) "
            "and reads Wi-Fi cameras on this network.",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    },
)
