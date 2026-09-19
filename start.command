#!/usr/bin/env bash
#
#  TBACroppedOutVid -- double-click this file in Finder to harvest matches.
#
#  Everything you would normally want to change is in the block below. Edit it
#  with TextEdit (or anything), save, double-click again. Lines starting with #
#  are notes and are ignored.
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

# 1 = stop the Mac sleeping while a long harvest runs.
KEEP_AWAKE="1"

# 1 = leave this window open when it finishes, so you can read the output.
KEEP_OPEN="1"

# ------------------------------------------------------- nothing below here --

cd "$(dirname "$0")" || exit 1

# Finder does not give a double-clicked script your shell's PATH, so Homebrew
# is invisible to it unless we say where it is. Both prefixes: /opt/homebrew
# on Apple Silicon, /usr/local on Intel.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

PY=""
for cand in python3 python; do
  command -v "$cand" >/dev/null 2>&1 && { PY="$cand"; break; }
done
if [ -z "$PY" ]; then
  echo "No Python found."
  echo "  Install it with:  brew install python"
  echo "  or download it from https://www.python.org/downloads/"
  [ "$KEEP_OPEN" = "1" ] && read -r -p "
Press Return to close. "
  exit 1
fi

RUN=("$PY" deploy/launcher.py "$@")
# caffeinate -i only blocks idle sleep; closing the lid still sleeps, which is
# the behaviour you want from a laptop.
if [ "$KEEP_AWAKE" = "1" ] && command -v caffeinate >/dev/null 2>&1; then
  RUN=(caffeinate -i "${RUN[@]}")
fi
"${RUN[@]}"
STATUS=$?

if [ "$KEEP_OPEN" = "1" ]; then
  echo
  read -r -p "Done (exit $STATUS). Press Return to close this window. "
fi
exit $STATUS
