#!/bin/bash
# Download match videos on a Mac, for the Colab notebook to label and train on.
#
# Why: YouTube refuses Colab's machines ("Sign in to confirm you're not a
# bot", 2026-11-18) but not a home connection. This downloads the matches
# here and copies them into Google Drive (frc-harvest/), where
# train/colab_harvest_1280.ipynb finds them and goes straight to labelling.
#
#   bash deploy/harvest_on_mac.sh          # 40 matches
#   bash deploy/harvest_on_mac.sh 20       # or another number
#
# Needs: Homebrew (brew.sh), Google Drive for desktop signed in to the
# Drive that holds frc-harvest/, and a TBA Read API key (it asks; or set
# TBA_AUTH_KEY). Safe to run again: matches already downloaded are kept and
# never downloaded twice.
set -euo pipefail
N="${1:-40}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="$HOME/frc-harvest"                  # downloads land here first, then go to Drive

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
DRIVE="$(ls -d "$HOME"/Library/CloudStorage/GoogleDrive-*/"My Drive" 2>/dev/null | head -1 || true)"
[ -n "$DRIVE" ] || { echo "Google Drive for desktop is not set up on this Mac (no ~/Library/CloudStorage/GoogleDrive-*). Install it from google.com/drive/download and sign in."; exit 1; }
DEST="$DRIVE/frc-harvest"
mkdir -p "$DEST/hdata" "$DEST/state" "$LOCAL/hdata"

say "Tools (ffmpeg, and a Python environment with yt-dlp)"
command -v ffmpeg >/dev/null || brew install ffmpeg
[ -x "$REPO/.venv-harvest/bin/python" ] || python3 -m venv "$REPO/.venv-harvest"
"$REPO/.venv-harvest/bin/pip" -q install --upgrade -r "$REPO/requirements.txt" yt-dlp
export PATH="$REPO/.venv-harvest/bin:$PATH"            # the yt-dlp just installed

if [ -z "${TBA_AUTH_KEY:-}" ]; then
  read -rsp "TBA Read API key (thebluealliance.com -> Account): " TBA_AUTH_KEY; echo
fi
export TBA_AUTH_KEY

say "Picking up where Colab or an earlier run left off"
rsync -a "$DEST/hdata/" "$LOCAL/hdata/"
rsync -a "$DEST/state/" "$REPO/state/"                 # the never-twice ledger and saved TBA lists
[ -f "$DEST/tba_cache.zip" ] && unzip -q -n "$DEST/tba_cache.zip" -d "$REPO/state/tba_cache" || true

# The notebook's settings: clean main-camera clips only, no scoreboard OCR
CFG="$REPO/harvest-config.json"
"$REPO/.venv-harvest/bin/python" - "$REPO/config.json" "$CFG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg.update({"keep_raw": False, "score_labels": False})
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY

say "Downloading $N matches (about 1-2 minutes each)"
export TBAVID_DATA="$LOCAL/hdata"
cd "$REPO"
set +e
"$REPO/.venv-harvest/bin/python" run.py --config "$CFG" fetch -n "$N" --per-event-cap 2 --seed 2026 --retry-failed
code=$?
set -e

say "Copying to Google Drive ($DEST)"
rsync -a "$LOCAL/hdata/" "$DEST/hdata/"
rsync -a "$REPO/state/" "$DEST/state/"
ok=$("$REPO/.venv-harvest/bin/python" -c "import json,sys; m=json.load(open(sys.argv[1])); print(sum(e.get('status')=='ok' for e in m.get('videos',{}).values()))" "$LOCAL/hdata/review/manifest.json" 2>/dev/null || echo 0)
say "$ok matches ready in Drive."
echo "Wait for Google Drive to finish uploading (its menu-bar icon), then in Colab: L4 runtime, Run all."
exit $code
