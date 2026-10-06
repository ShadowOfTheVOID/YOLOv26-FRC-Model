"""One-click updates for the Hub Counter and Watchtower apps.

Before this, a new release meant: find the release page, download the
right file, run the installer or drag the app again. Each app now asks
GitHub for the latest release when it starts (public API, no key), and if
it is newer than the build's own version, its page shows "Update
available" with an Update button. Nothing installs by itself -- the user
chose a button over automatic updates -- and the button is refused while
the cameras are counting, so a match is never interrupted.

What Update does:
  Windows  download *-Setup-windows.exe; a small .cmd waits for this app to
           exit, runs the installer silently into the same folder (the
           installer's fixed AppId replaces the old version) and starts the
           app again.
  macOS    download *-mac.dmg; a small shell script waits for this app to
           exit, mounts the .dmg, replaces the .app in place, unmounts and
           opens it again. Downloaded by Python, the file carries no
           quarantine flag, so Gatekeeper's first-open prompt does not repeat.
  Linux/Pi only a link to the release page (a tar.gz has no installer).

Settings and logs live in ~/Documents/<App>, outside the app, so they
survive. Builds whose version is 0.0.0 (pull-request and local builds) and
runs from a checkout never offer updates.

Standard library only: the apps start this before anything heavy.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path, PureWindowsPath
from typing import Callable, Optional

REPO = "ShadowOfTheVOID/YOLOv26-FRC-Model"
# APP_UPDATE_API points the tests at a stand-in for GitHub.
API = os.environ.get("APP_UPDATE_API", f"https://api.github.com/repos/{REPO}/releases/latest")


def version_tuple(v: str) -> tuple:
    """'v0.4.10' -> (0, 4, 10); anything unparsable -> () (never "newer")."""
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", (v or "").strip())
    return tuple(int(x) for x in m.groups()) if m else ()


def build_version() -> str:
    """The version baked into this build (app_version.txt, written by the
    spec from the release tag), or "" when run from a checkout."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return ""
    try:
        return (Path(base) / "app_version.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def asset_name(app: str, platform: str = sys.platform, machine: str = "") -> str:
    """The release file for this app on this computer. app is the asset
    prefix: "Watchtower" or "HubCounter"."""
    if platform == "darwin":
        return f"{app}-mac.dmg"
    if platform == "win32":
        return f"{app}-Setup-windows.exe"
    import platform as _p
    arch = machine or _p.machine()
    return f"{app}-linux-{'arm64' if arch.lower() in ('aarch64', 'arm64') else 'x64'}.tar.gz"


class Updater:
    """Checks once at start; installs on request. The page reads status()."""

    def __init__(self, app: str, version: str, on_exit: Callable[[], None],
                 platform: str = sys.platform):
        self.app, self.version, self.on_exit, self.platform = app, version, on_exit, platform
        self.state = "idle"          # idle | checking | ready | downloading | installing | error
        self.latest = ""
        self.release_url = ""
        self.asset_url = ""
        self.progress = 0.0
        self.error = ""
        self.lock = threading.Lock()

    # -- checking ----------------------------------------------------------
    def enabled(self) -> bool:
        v = version_tuple(self.version)
        return bool(v) and v != (0, 0, 0)

    def check(self, timeout: float = 10.0) -> None:
        if not self.enabled():
            return
        with self.lock:
            self.state = "checking"
        try:
            req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json",
                                                       "User-Agent": f"{self.app} {self.version}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rel = json.loads(r.read().decode("utf-8"))
            want = asset_name(self.app, self.platform)
            url = next((a.get("browser_download_url", "") for a in rel.get("assets") or []
                        if a.get("name") == want), "")
            with self.lock:
                self.latest = rel.get("tag_name", "")
                self.release_url = rel.get("html_url", "")
                self.asset_url = url
                self.state = "idle"
        except (OSError, ValueError) as e:       # offline at the venue: say nothing
            with self.lock:
                self.state, self.error = "idle", ""
            print(f"update check failed: {e}")

    def check_async(self) -> None:
        threading.Thread(target=self.check, daemon=True, name="update check").start()

    def available(self) -> bool:
        a, b = version_tuple(self.latest), version_tuple(self.version)
        return bool(a and b) and a > b

    def can_install(self) -> bool:
        return self.available() and bool(self.asset_url) and self.platform in ("darwin", "win32")

    def status(self) -> dict:
        with self.lock:
            return {"current": self.version, "latest": self.latest,
                    "available": self.available(), "can_install": self.can_install(),
                    "state": self.state, "progress": round(self.progress, 3),
                    "error": self.error, "release_url": self.release_url}

    # -- installing ----------------------------------------------------------
    def install(self, counting: bool) -> dict:
        """Start the update. Refused while counting: an update closes the app."""
        if counting:
            return {"error": "Stop the cameras first: updating closes the app for a minute."}
        if not self.can_install():
            return {"error": "No update to install here."}
        with self.lock:
            if self.state in ("downloading", "installing"):
                return {"ok": True}
            self.state, self.progress, self.error = "downloading", 0.0, ""
        threading.Thread(target=self._install, daemon=True, name="update install").start()
        return {"ok": True}

    def _install(self) -> None:
        try:
            work = Path(tempfile.mkdtemp(prefix=f"{self.app}-update-"))
            path = work / asset_name(self.app, self.platform)
            self._download(self.asset_url, path)
            with self.lock:
                self.state = "installing"
            script = self.write_helper(path, work)
            launch_helper(script, self.platform)
            time.sleep(0.5)
            self.on_exit()                       # the helper waits for this exit
        except Exception as e:                   # the page shows it; the app keeps running
            with self.lock:
                self.state, self.error = "error", f"Update failed: {e}"

    def _download(self, url: str, dest: Path) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": f"{self.app} {self.version}"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    with self.lock:
                        self.progress = done / total
        if total and done != total:
            raise OSError(f"download stopped at {done >> 20} of {total >> 20} MB")

    def write_helper(self, package: Path, work: Path, pid: Optional[int] = None,
                     exe: Optional[str] = None) -> Path:
        """The script that runs after this app exits: install, then reopen."""
        pid = pid or os.getpid()
        exe = exe or sys.executable
        if self.platform == "win32":
            folder = str(PureWindowsPath(exe).parent)
            text = "\r\n".join([
                "@echo off",
                ":wait",
                f'tasklist /FI "PID eq {pid}" 2>NUL | find "{pid}" >NUL && (timeout /t 1 /nobreak >NUL & goto wait)',
                f'"{package}" /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /NORESTART /DIR="{folder}"',
                f'start "" "{exe}"',
                ""])
            script = work / "update.cmd"
            script.write_text(text, encoding="utf-8")
            return script
        # macOS: .../Name.app/Contents/MacOS/Name -> .../Name.app
        app_bundle = Path(exe).parents[2]
        mnt = work / "mnt"
        text = "\n".join([
            "#!/bin/sh",
            f"while kill -0 {pid} 2>/dev/null; do sleep 1; done",
            f'mkdir -p "{mnt}"',
            f'hdiutil attach "{package}" -nobrowse -readonly -mountpoint "{mnt}" || exit 1',
            f'new=$(ls -d "{mnt}"/*.app | head -1)',
            f'[ -d "$new" ] || {{ hdiutil detach "{mnt}"; exit 1; }}',
            f'rm -rf "{app_bundle}.old" && mv "{app_bundle}" "{app_bundle}.old"',
            f'if ditto "$new" "{app_bundle}"; then rm -rf "{app_bundle}.old"; '
            f'else rm -rf "{app_bundle}"; mv "{app_bundle}.old" "{app_bundle}"; fi',
            f'hdiutil detach "{mnt}"',
            f'open "{app_bundle}"',
            ""])
        script = work / "update.sh"
        script.write_text(text, encoding="utf-8")
        script.chmod(0o755)
        return script


def launch_helper(script: Path, platform: str = sys.platform) -> None:
    """Start the helper so that it outlives this app."""
    if platform == "win32":
        subprocess.Popen(["cmd", "/c", str(script)], close_fds=True,
                         creationflags=0x00000008 | 0x00000200)   # DETACHED | NEW_GROUP
    else:
        subprocess.Popen(["/bin/sh", str(script)], start_new_session=True, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
