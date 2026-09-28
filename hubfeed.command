#!/usr/bin/env bash
#
#  Hub FUEL counter -- double-click this file in Finder. It opens the counter
#  in your web browser (nothing else to install).
#
#  First run only: makes .venv and installs numpy + OpenCV into it. The setup
#  you save is cams.json in this folder, and it is opened again next time.
#
cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/python ]; then
    echo "first run: setting up .venv (a minute or two)..."
    python3 -m venv .venv || { echo "python3 not found -- install it from python.org"; read -r; exit 1; }
fi
if ! .venv/bin/python -c "import cv2, numpy, yt_dlp" 2>/dev/null; then
    # yt-dlp finds the video behind a Twitch / YouTube stream address
    .venv/bin/pip install -q -r requirements.txt opencv-python-headless yt-dlp || { read -r; exit 1; }
fi
exec .venv/bin/python run.py hubgui --setup cams.json
