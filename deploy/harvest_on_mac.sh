#!/bin/bash
# Download match videos on a Mac, for the Colab notebook to label and train on.
#
# Why: YouTube refuses Colab's machines ("Sign in to confirm you're not a
# bot", 2026-11-18) but not a home connection. This downloads the matches
# here and copies them into Google Drive (frc-harvest/), where
# train/colab_harvest_1280.ipynb finds them and goes straight to labelling.
#
#   bash deploy/harvest_on_mac.sh          # 40 matches, from a checkout
#   bash deploy/harvest_on_mac.sh 20       # or another number
#
# or with nothing checked out (it fetches the code into ~/frc-harvest/repo):
#   curl -fsSL https://raw.githubusercontent.com/ShadowOfTheVOID/YOLOv26-FRC-Model/main/deploy/harvest_on_mac.sh | bash
#   ... | bash -s -- 20                    # another number
#
# Needs: a TBA Read API key (it asks; or set TBA_AUTH_KEY). With Google Drive
# for desktop signed in, the results go straight into frc-harvest/ in Drive;
# without it they are packed into ~/frc-harvest/harvest.zip, which you drag
# into the frc-harvest folder at drive.google.com (the notebook unpacks it). No
# Homebrew: ffmpeg and ffprobe are static builds fetched into
# ~/frc-harvest/bin when they are not already installed (osxexperts.net for
# Apple silicon, evermeet.cx for Intel); yt-dlp is its standalone Mac program
# and deno (which yt-dlp needs for YouTube) its release build, both into the
# same folder; Python 3 is the one macOS's developer tools have. Safe to run again: matches already downloaded are kept and
# never downloaded twice.
set -euo pipefail
N="${1:-40}"
LOCAL="$HOME/frc-harvest"                  # downloads land here first, then go to Drive

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# Inside a checkout, use it. Piped from curl there is none (and $0 is just
# "bash"): fetch the code into ~/frc-harvest/repo, keeping its state/ (the
# never-twice ledger) across runs.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$HERE" ] && [ -f "$HERE/../run.py" ]; then
  REPO="$(cd "$HERE/.." && pwd)"
else
  REPO="$LOCAL/repo"
  say "Getting the code (${FRC_REF:-main}) into $REPO"
  mkdir -p "$REPO"
  curl -fsSL "https://codeload.github.com/ShadowOfTheVOID/YOLOv26-FRC-Model/tar.gz/refs/heads/${FRC_REF:-main}" \
    | tar -xz -C "$REPO" --strip-components 1
fi

command -v python3 >/dev/null && python3 -c "import venv" 2>/dev/null \
  || { echo "Python 3 is needed: run  xcode-select --install  (Apple's developer tools), then this again."; exit 1; }
DRIVE="$(ls -d "$HOME"/Library/CloudStorage/GoogleDrive-*/"My Drive" 2>/dev/null | head -1 || true)"
DEST=""
if [ -n "$DRIVE" ]; then
  DEST="$DRIVE/frc-harvest"
  mkdir -p "$DEST/hdata" "$DEST/state"
fi
mkdir -p "$LOCAL/hdata" "$LOCAL/state"

say "Tools (ffmpeg, and a Python environment with yt-dlp)"
BIN="$LOCAL/bin"
export PATH="$BIN:$PATH"
latest() {   # the newest link on a download page matching a pattern
  curl -fsSL "$1" | grep -oE "$2" | head -1
}
fetch_bin() {   # $1 = ffmpeg or ffprobe
  local url
  if [ "$(uname -m)" = "arm64" ]; then
    url="$(latest https://www.osxexperts.net/ "https://www\.osxexperts\.net/$1[0-9]+arm\.zip")"
  else
    url="https://evermeet.cx/ffmpeg/$(latest https://evermeet.cx/ffmpeg/ "$1-[0-9.]+\.zip")"
  fi
  [ -n "$url" ] && [ "${url%/}" != "https://evermeet.cx/ffmpeg" ] \
    || { echo "Could not find a $1 download; install ffmpeg yourself and run this again."; exit 1; }
  echo "  $1 <- $url"
  curl -fsSL "$url" -o "$BIN/$1.zip" && unzip -o -q "$BIN/$1.zip" -x "__MACOSX/*" -d "$BIN" && rm "$BIN/$1.zip"
  chmod +x "$BIN/$1"
}
mkdir -p "$BIN"
for t in ffmpeg ffprobe; do command -v "$t" >/dev/null || fetch_bin "$t"; done
ffmpeg -version | head -1
# yt-dlp from pip needs Python 3.11+ (2026.10.26); macOS's python3 is 3.9, so
# pip quietly installed an old yt-dlp that YouTube no longer answers, and
# every download failed (2026-11-18). The standalone program carries its own
# Python. Current yt-dlp also needs a JavaScript runtime for YouTube: deno.
GH=https://github.com
if [ ! -x "$BIN/yt-dlp" ]; then
  echo "  yt-dlp <- $GH/yt-dlp/yt-dlp (yt-dlp_macos)"
  curl -fsSL "$GH/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos" -o "$BIN/yt-dlp"
  chmod +x "$BIN/yt-dlp"
else
  "$BIN/yt-dlp" -U 2>&1 | tail -1 || true             # YouTube changes; keep it current
fi
if ! command -v deno >/dev/null; then
  arch="$( [ "$(uname -m)" = arm64 ] && echo aarch64 || echo x86_64 )"
  echo "  deno <- $GH/denoland/deno ($arch)"
  curl -fsSL "$GH/denoland/deno/releases/latest/download/deno-$arch-apple-darwin.zip" -o "$BIN/deno.zip" \
    && unzip -o -q "$BIN/deno.zip" -d "$BIN" && rm "$BIN/deno.zip" && chmod +x "$BIN/deno"
fi
xattr -d com.apple.quarantine "$BIN"/* 2>/dev/null || true
echo "yt-dlp $("$BIN/yt-dlp" --version)"
[ -x "$REPO/.venv-harvest/bin/python" ] || python3 -m venv "$REPO/.venv-harvest"
"$REPO/.venv-harvest/bin/pip" -q install --upgrade -r "$REPO/requirements.txt"
"$REPO/.venv-harvest/bin/pip" -q uninstall -y yt-dlp 2>/dev/null || true   # the old one an earlier run put here

if [ -z "${TBA_AUTH_KEY:-}" ]; then
  # from the keyboard: when piped from curl, stdin is this script itself
  read -rsp "TBA Read API key (thebluealliance.com -> Account): " TBA_AUTH_KEY < /dev/tty; echo
fi
export TBA_AUTH_KEY

mkdir -p "$REPO/state"
rsync -a "$LOCAL/state/" "$REPO/state/"                # the never-twice ledger from earlier runs here
if [ -n "$DEST" ]; then
  say "Picking up where Colab or an earlier run left off"
  rsync -a "$DEST/hdata/" "$LOCAL/hdata/"
  rsync -a "$DEST/state/" "$REPO/state/"               # the never-twice ledger and saved TBA lists
  [ -f "$DEST/tba_cache.zip" ] && unzip -q -n "$DEST/tba_cache.zip" -d "$REPO/state/tba_cache" || true
fi

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

rsync -a "$REPO/state/" "$LOCAL/state/"                # kept here too, so a re-run never repeats a match
ok=$("$REPO/.venv-harvest/bin/python" -c "import json,sys; m=json.load(open(sys.argv[1])); print(sum(e.get('status')=='ok' for e in m.get('videos',{}).values()))" "$LOCAL/hdata/review/manifest.json" 2>/dev/null || echo 0)
if [ -n "$DEST" ]; then
  say "Copying to Google Drive ($DEST)"
  rsync -a "$LOCAL/hdata/" "$DEST/hdata/"
  rsync -a "$REPO/state/" "$DEST/state/"
  say "$ok matches ready in Drive."
  echo "Wait for Google Drive to finish uploading (its menu-bar icon), then in Colab: L4 runtime, Run all."
else
  # No Drive app: one file to drag into Drive in the browser. Stored, not
  # compressed (the clips are already compressed video).
  say "Packing $ok matches into $LOCAL/harvest.zip"
  rm -f "$LOCAL/harvest.zip"
  (cd "$LOCAL" && zip -r -0 -q harvest.zip hdata state -x "*/.DS_Store")
  ls -lh "$LOCAL/harvest.zip" | awk '{print "  " $5}'
  echo "Upload it: open drive.google.com, go into the frc-harvest folder, and drag harvest.zip in"
  echo "(replace the old one if Drive asks). Then in Colab: L4 runtime, Run all."
  open "$LOCAL" 2>/dev/null || true
  open "https://drive.google.com/drive/search?q=frc-harvest" 2>/dev/null || true
fi
exit $code
