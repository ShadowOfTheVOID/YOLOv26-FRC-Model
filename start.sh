#!/usr/bin/env bash
#
#  TBACroppedOutVid -- one-click start on Linux.
#
#    ./start.sh                   run with the settings below
#    ./start.sh status            run one command, ignoring the settings
#    ./start.sh --install-desktop add it to your applications menu, so it
#                                 really is one click from then on
#
#  Everything you would normally want to change is in the block below. Edit,
#  save, run again. Lines starting with # are notes and are ignored.
#
#  First run only: it makes a .venv, installs the Python packages, and asks
#  for your TBA key once. After that it just runs.
#
# ---------------------------------------------------------------- settings --

# What to do. "pull" is the full run: pick matches, download, crop, export
# frames. Other useful values:
#   menu    ask each time instead of deciding here
#   fetch   download and crop, but do not export frames
#   status  ledger and dataset summary, downloads nothing
#   verify  check the OCR'd fuel totals against TBA's official scores
#   serve   the read-only scouting API on http://127.0.0.1:8781
#   doctor  check the install and print what is where, run nothing
export TBAVID_MODE="pull"

# How many never-pulled matches to fetch. Each one costs roughly 330 MB of
# disk and a few minutes. 5 is a comfortable first run; 20 is an evening.
export TBAVID_COUNT="5"

# Most videos from any one event. 0 means no limit. A cap of 2 spreads the
# harvest across events, which is better training data than 20 matches from
# one field.
export TBAVID_PER_EVENT_CAP="0"

# Splitting the work with teammates: you take slice N of M, e.g. "2/4". No two
# people with different N can ever pull the same match. Empty = take anything.
export TBAVID_SHARD=""

# 1 = show what would be pulled and download nothing. Good for a sanity check.
export TBAVID_DRY_RUN="0"

# 1 = open the shot review UI in a browser before frames are exported.
export TBAVID_REVIEW="0"

# ------------------------------------------------------------ if it breaks --

# 1 = update yt-dlp before running. Try this first when downloads start
# failing for no apparent reason -- YouTube changes and a stale yt-dlp stops
# working. It is the single most common cause of "it used to work".
export TBAVID_UPDATE_YTDLP="0"

# 1 = create .venv and install missing Python packages automatically.
# 0 = never touch the disk; complain instead.
export TBAVID_AUTO_INSTALL="1"

# Anything else, appended to the command verbatim. Examples:
#   "--retry-failed"           reconsider matches that failed before
#   "--seed 7"                 reproducible picks
#   "--include-noncompetitive" also offseason events and practice matches
export TBAVID_EXTRA=""

# 1 = ask the desktop not to suspend the machine mid-harvest (needs
# systemd-inhibit, which is there on any systemd distro).
KEEP_AWAKE="1"

# 1 = always wait for Return at the end. "auto" waits only when the
# applications-menu entry started this, which is the case where the terminal
# would otherwise close before you could read anything.
KEEP_OPEN="auto"

# ------------------------------------------------------- nothing below here --

# readlink -f resolves a symlink in ~/bin back to the real project folder;
# macOS has no such readlink, hence the fallback.
SELF="$(readlink -f "$0" 2>/dev/null || echo "$0")"
cd "$(dirname "$SELF")" || exit 1

# deploy/debian_setup.sh puts yt-dlp and a static ffmpeg here, and a
# non-interactive shell does not read the .bashrc line that adds it.
export PATH="$HOME/.local/bin:$PATH"

# Register with the desktop so it appears in the applications menu. The Exec
# path has to be absolute -- a .desktop file has no working directory.
if [ "${1:-}" = "--install-desktop" ]; then
  dir="$(pwd)"
  apps="$HOME/.local/share/applications"
  mkdir -p "$apps"
  cat > "$apps/tbavid.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=TBACroppedOutVid
Comment=Harvest FRC match video into training frames
Exec=env TBAVID_FROM_DESKTOP=1 $dir/start.sh
Path=$dir
Terminal=true
Categories=Utility;
DESKTOP
  chmod +x "$apps/tbavid.desktop"
  command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$apps" >/dev/null 2>&1
  echo "Added to your applications menu: $apps/tbavid.desktop"
  echo "Edit $dir/start.sh to change what it does."
  exit 0
fi

PY=""
for cand in python3 python; do
  command -v "$cand" >/dev/null 2>&1 && { PY="$cand"; break; }
done
if [ -z "$PY" ]; then
  echo "No Python found."
  echo "  Debian/Ubuntu:  sudo apt install python3 python3-venv"
  echo "  Fedora:         sudo dnf install python3"
  exit 1
fi

RUN=("$PY" deploy/launcher.py "$@")
# --what=idle:sleep leaves a deliberate `systemctl suspend` alone; only the
# automatic idle suspend is blocked.
if [ "$KEEP_AWAKE" = "1" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  RUN=(systemd-inhibit --what=idle:sleep --why="harvesting match video" \
       "${RUN[@]}")
fi
"${RUN[@]}"
STATUS=$?

if [ "$KEEP_OPEN" = "1" ] || { [ "$KEEP_OPEN" = "auto" ] \
   && [ -n "${TBAVID_FROM_DESKTOP:-}" ]; }; then
  echo
  read -r -p "Done (exit $STATUS). Press Return to close. "
fi
exit $STATUS
