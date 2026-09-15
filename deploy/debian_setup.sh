#!/usr/bin/env bash
# Set up the harvester on Debian. Uses apt packages where they are good enough
# and standalone binaries where they are not, so there is no pip, no venv and
# no PEP 668 "externally-managed-environment" argument to have.
#
#   ./deploy/debian_setup.sh          # apt where possible (needs sudo)
#   ./deploy/debian_setup.sh --nosudo # static binaries into ~/.local/bin only
set -u
cd "$(dirname "$0")/.."
NOSUDO=0
[ "${1:-}" = "--nosudo" ] && NOSUDO=1
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

say() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "python"
if have python3; then
  python3 -c 'import sys; print("  python3", ".".join(map(str, sys.version_info[:3])));
import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' || {
    echo "  ! need python 3.9+"; exit 1; }
else
  echo "  ! no python3. apt install python3"; exit 1
fi

say "python modules (requests, numpy)"
# Debian packages these, so pip never has to enter the picture.
for mod in requests numpy; do
  if python3 -c "import $mod" 2>/dev/null; then
    echo "  $mod: present"
  elif [ "$NOSUDO" = "0" ]; then
    echo "  installing python3-$mod"
    sudo apt-get install -y "python3-$mod" >/dev/null 2>&1 \
      && echo "  $mod: installed" || echo "  ! python3-$mod failed"
  else
    echo "  ! $mod missing; apt install python3-$mod (or pip install --user $mod)"
  fi
done

say "ffmpeg"
if have ffmpeg && have ffprobe; then
  echo "  $(ffmpeg -version | head -1 | cut -c1-50)"
elif [ "$NOSUDO" = "0" ]; then
  sudo apt-get install -y ffmpeg >/dev/null 2>&1 && echo "  installed from apt" \
    || echo "  ! apt install ffmpeg failed"
else
  # Static build: one tarball, two binaries, no root.
  echo "  fetching static ffmpeg into ~/.local/bin"
  curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    -o /tmp/ff.tar.xz && tar -xf /tmp/ff.tar.xz -C /tmp \
    && cp /tmp/ffmpeg-*-static/ffmpeg /tmp/ffmpeg-*-static/ffprobe "$HOME/.local/bin/" \
    && echo "  ok" || echo "  ! static ffmpeg fetch failed"
fi

say "yt-dlp"
# Deliberately NOT apt: the packaged version lags, and a stale yt-dlp simply
# stops working against YouTube. The official standalone binary is one file.
curl -fsSL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
  -o "$HOME/.local/bin/yt-dlp" && chmod +x "$HOME/.local/bin/yt-dlp" \
  && echo "  $("$HOME/.local/bin/yt-dlp" --version) in ~/.local/bin" \
  || echo "  ! download failed"

say "tesseract (only needed for scoreboard labels)"
if have tesseract; then
  echo "  $(tesseract --version 2>&1 | head -1)"
elif [ "$NOSUDO" = "0" ]; then
  sudo apt-get install -y tesseract-ocr >/dev/null 2>&1 && echo "  installed" \
    || echo "  ! apt install tesseract-ocr failed"
else
  echo "  ! not available without root. Set \"score_labels\": false in config.json"
  echo "    to harvest frames without the scoreboard reader."
fi

say "tuning config.json for slower hardware"
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("config.json")
cfg = json.loads(p.read_text())
# prefer_h264: AV1 has no hardware decode before ~2020 and is punishing in
# software. keep_clean: skips an x264 encode of a file that gets deleted.
# keep_raw: sources are re-downloadable and cost ~110 MB each.
before = {k: cfg.get(k) for k in ("prefer_h264", "keep_raw", "keep_clean")}
cfg["prefer_h264"] = True
cfg["keep_raw"] = False
cfg["keep_clean"] = False
p.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"  was {before}")
print(f"  now {{'prefer_h264': True, 'keep_raw': False, 'keep_clean': False}}")
print("  -> ~100 MB per match instead of ~333 MB, and several minutes faster")
PY

say "result"
ALLOK=1
for t in python3 ffmpeg ffprobe yt-dlp; do
  if have "$t"; then echo "  $t  ok"; else echo "  $t  MISSING"; ALLOK=0; fi
done
have tesseract && echo "  tesseract  ok" || echo "  tesseract  missing (set score_labels=false)"
python3 -c "import requests, numpy" 2>/dev/null && echo "  requests+numpy  ok" || { echo "  requests/numpy MISSING"; ALLOK=0; }

if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
  echo
  echo "  add this to ~/.bashrc:"
  echo '    export PATH="$HOME/.local/bin:$PATH"'
fi
[ "$ALLOK" = "1" ] && echo "
Ready. Put your key in .env then:  python3 run.py pull -n 3 --dry-run"
