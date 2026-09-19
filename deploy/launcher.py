#!/usr/bin/env python3
"""Everything the one-click launchers do before run.py starts.

    start.command    macOS -- double-click it
    start.bat        Windows -- double-click it
    start.sh         Linux

Those three files are deliberately thin: they hold the settings you are meant
to edit and then hand over to this one, which is the only copy of the
bootstrap logic. Nothing here is required to use the project -- it is all
things you would otherwise do by hand once (make a venv, install the Python
deps, write .env, check ffmpeg is on PATH) and then forget about.

Settings arrive as TBAVID_* environment variables so the launchers never have
to build an argument list -- quoting a list in cmd.exe is its own small
tragedy. Any command-line arguments given here are passed straight to run.py
instead, which is what makes `./start.sh status` work from a terminal.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = os.name == "nt"
VENV = ROOT / ".venv"

# One place to change where the setup advice points, per platform.
INSTALL_HELP = {
    "darwin": "  brew install ffmpeg yt-dlp tesseract\n"
              "  (no brew? https://brew.sh)",
    "win32": "  winget install Gyan.FFmpeg\n"
             "  winget install UB-Mannheim.TesseractOCR\n"
             "  then reopen this window so PATH is picked up",
    "linux": "  sudo apt install ffmpeg tesseract-ocr\n"
             "  or run deploy/debian_setup.sh, which needs no root for yt-dlp",
}


def say(text=""):
    print(text, flush=True)


def rule(title):
    say(f"\n=== {title} ===")


def env(name, default=""):
    return os.environ.get(f"TBAVID_{name}", default).strip()


def flag(name, default=False):
    val = env(name, "1" if default else "0").lower()
    return val in ("1", "true", "yes", "on")


def quiet(cmd) -> bool:
    """Run a command, discard its output, report only whether it worked."""
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


# --------------------------------------------------------------- interpreter

def venv_python(venv: Path = VENV):
    p = venv / ("Scripts" if WINDOWS else "bin") / ("python.exe" if WINDOWS
                                                    else "python")
    return p if p.exists() else None


def deps_ok(py) -> bool:
    return quiet([str(py), "-c", "import requests, numpy"])


def bin_dir(py: Path) -> Path:
    return Path(py).parent


def make_venv() -> Path | None:
    """Create .venv and install requirements.txt into it. None if it can't."""
    say(f"  creating {VENV.name}/ (one time, ~10s)")
    if not quiet([sys.executable, "-m", "venv", str(VENV)]):
        say("  ! could not create a virtualenv.")
        if sys.platform.startswith("linux"):
            say("    Debian splits it out:  sudo apt install python3-venv")
            say("    or install the deps system-wide: sudo apt install "
                "python3-requests python3-numpy")
        return None
    py = venv_python()
    if py is None:
        say("  ! virtualenv created but has no python in it")
        return None
    say("  installing requests and numpy")
    if not quiet([str(py), "-m", "pip", "install", "--quiet", "--upgrade",
                  "pip"]):
        say("  (pip self-upgrade skipped)")
    if not quiet([str(py), "-m", "pip", "install", "--quiet", "-r",
                  str(ROOT / "requirements.txt")]):
        say("  ! pip install failed -- are you online?")
        return None
    return py


def pick_python():
    """The interpreter run.py gets. Prefers .venv, makes one if allowed."""
    py = venv_python()
    if py and deps_ok(py):
        return py
    if py:
        say("  .venv exists but is missing requests/numpy")
        if flag("AUTO_INSTALL", True) and quiet(
                [str(py), "-m", "pip", "install", "--quiet", "-r",
                 str(ROOT / "requirements.txt")]):
            return py
    if deps_ok(sys.executable):
        # Debian with python3-requests installed, or an already-active venv.
        return Path(sys.executable)
    if not flag("AUTO_INSTALL", True):
        say("  ! requests/numpy are missing and AUTO_INSTALL is off.")
        say(f"    {sys.executable} -m pip install -r requirements.txt")
        return None
    return make_venv()


# -------------------------------------------------------------- outside tools

def ensure_ytdlp(py: Path) -> bool:
    """yt-dlp on PATH, or pip-installed next to the chosen interpreter.

    It is pinned to nothing on purpose: YouTube changes, and a yt-dlp that is
    a few months old simply stops downloading. That failure looks like a
    broken project, so updating it is a supported knob rather than folklore.
    """
    if flag("UPDATE_YTDLP") and py:
        say("  updating yt-dlp")
        quiet([str(py), "-m", "pip", "install", "--quiet", "--upgrade",
               "yt-dlp"])
    if shutil.which("yt-dlp"):
        return True
    if py and flag("AUTO_INSTALL", True):
        say("  installing yt-dlp")
        if quiet([str(py), "-m", "pip", "install", "--quiet", "yt-dlp"]):
            return shutil.which("yt-dlp") is not None
    return False


def check_tools(py) -> bool:
    """ffmpeg/ffprobe/yt-dlp are hard requirements; tesseract is not."""
    ok = True
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        say(f"  {tool:9} {'ok' if path else 'MISSING'}")
        ok = ok and bool(path)
    say(f"  {'yt-dlp':9} {'ok' if ensure_ytdlp(py) else 'MISSING'}")
    ok = ok and shutil.which("yt-dlp") is not None
    if not shutil.which("tesseract"):
        say("  tesseract missing -- frames still work, scoreboard labels do "
            "not.\n             Set \"score_labels\": false in config.json to "
            "silence it.")
    if not ok:
        say("\n  Install what is missing, then run this again:")
        say(INSTALL_HELP.get(sys.platform, INSTALL_HELP["linux"]))
    return ok


# ------------------------------------------------------------------- the key

def have_key() -> bool:
    if os.environ.get("TBA_AUTH_KEY", "").strip():
        return True
    envfile = ROOT / ".env"
    if not envfile.exists():
        return False
    for line in envfile.read_text(errors="replace").splitlines():
        name, _, val = line.strip().partition("=")
        if name.strip() == "TBA_AUTH_KEY" and val.strip().strip("'\""):
            return True
    return False


def ensure_key() -> bool:
    """Ask once, write .env, and never ask again."""
    if have_key():
        say("  TBA key ok")
        return True
    say("  No TBA API key yet. Get one (free, instant) at")
    say("    https://www.thebluealliance.com/account")
    if not sys.stdin or not sys.stdin.isatty():
        say("  Then write it to .env as:  TBA_AUTH_KEY=...")
        return False
    try:
        key = input("\n  Paste your key here (or Enter to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if not key:
        return False
    envfile = ROOT / ".env"
    existing = envfile.read_text() if envfile.exists() else ""
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    envfile.write_text(f"{existing}{sep}TBA_AUTH_KEY={key}\n")
    if not WINDOWS:
        os.chmod(envfile, 0o600)          # no equivalent worth doing on NT
    say("  saved to .env (gitignored)")
    return True


# ------------------------------------------------------------- what to run

MENU = [
    ("pull 5 matches", ["pull", "-n", "5"]),
    ("pull 20 matches, max 2 per event", ["pull", "-n", "20",
                                          "--per-event-cap", "2"]),
    ("dry run -- show picks, download nothing", ["pull", "-n", "5",
                                                 "--dry-run"]),
    ("status -- ledger and dataset summary", ["status"]),
    ("verify -- check OCR totals against TBA", ["verify"]),
    ("serve -- scouting API on :8781", ["serve"]),
]


def menu():
    say("\n  What do you want to do?\n")
    for i, (label, _) in enumerate(MENU, 1):
        say(f"    {i}. {label}")
    say("    q. quit")
    try:
        choice = input("\n  Number: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if choice in ("q", "quit", ""):
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(MENU):
        return list(MENU[int(choice) - 1][1])
    say("  not a choice")
    return None


def build_args():
    """run.py's argv, from the TBAVID_* settings the launcher exported."""
    mode = env("MODE", "pull") or "pull"
    if mode == "menu":
        return menu()
    args = [mode]
    if mode in ("pull", "fetch"):
        count = env("COUNT", "5")
        if count:
            args += ["-n", count]
        cap = env("PER_EVENT_CAP", "0")
        if cap and cap != "0":
            args += ["--per-event-cap", cap]
        shard = env("SHARD")
        if shard:
            args += ["--shard", shard]
        if flag("DRY_RUN"):
            args.append("--dry-run")
        if mode == "pull" and flag("REVIEW"):
            args.append("--review")
    extra = env("EXTRA")
    if extra:
        args += shlex.split(extra)
    return args


def doctor(py):
    """Same idea as deploy/doctor.sh, minus bash -- for the .bat's sake."""
    rule("versions")
    for cmd in (["ffmpeg", "-version"], ["yt-dlp", "--version"],
                ["tesseract", "--version"]):
        if shutil.which(cmd[0]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True)
                say(f"  {(out.stdout or out.stderr).splitlines()[0][:70]}")
            except OSError:
                say(f"  {cmd[0]}: unreadable")
        else:
            say(f"  {cmd[0]}: missing")
    say(f"  python: {py}")
    rule("folders")
    for name in ("videos", "frames", "labels", "raw"):
        d = ROOT / "data" / name
        n = len(list(d.iterdir())) if d.is_dir() else 0
        say(f"  data/{name:8} {n} files")
    return 0


# ------------------------------------------------------------------- driver

def main(argv):
    os.chdir(ROOT)
    say("TBACroppedOutVid")
    say(f"  folder: {ROOT}")

    rule("python")
    py = pick_python()
    if py is None:
        return 1
    say(f"  {py}")
    # yt-dlp installed into the venv lands beside that python, and the
    # pipeline finds its tools with shutil.which -- so the venv has to be on
    # PATH here, not just for the subprocess.
    os.environ["PATH"] = str(bin_dir(py)) + os.pathsep + os.environ["PATH"]

    rule("tools")
    if not check_tools(py):
        return 1

    mode = env("MODE", "pull")
    if mode == "doctor" and not argv:
        return doctor(py)

    rule("key")
    if not ensure_key():
        return 1

    args = argv or build_args()
    if not args:
        say("\nnothing to do")
        return 0

    rule("running")
    say(f"  python run.py {' '.join(args)}\n")
    try:
        code = subprocess.run([str(py), str(ROOT / "run.py"), *args]).returncode
    except KeyboardInterrupt:
        say("\n  stopped")
        return 130
    if code != 0:
        say("\n  That run failed. The usual causes, in order:")
        say("    - download errors    -> set UPDATE_YTDLP=1 in the launcher")
        say("    - no matches left    -> raise COUNT, or EXTRA=\"--retry-failed\"")
        say("    - disk full          -> python run.py prune --all")
        say("    - something else     -> python run.py status, and keep the "
            "traceback")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
