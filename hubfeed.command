#!/usr/bin/env bash
#
#  Hub FUEL counter -- double-click this file in Finder to open the window.
#
#  First run only: makes .venv and installs numpy + OpenCV into it. After that
#  it just opens. The setup you save is cams.json in this folder, and it is
#  opened again next time.
#
cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/python ]; then
    echo "first run: setting up .venv (a minute or two)..."
    python3 -m venv .venv || { echo "python3 not found -- install it from python.org"; read -r; exit 1; }
fi
if ! .venv/bin/python -c "import cv2, numpy" 2>/dev/null; then
    .venv/bin/pip install -q -r requirements.txt opencv-python-headless || { read -r; exit 1; }
fi
if ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
    echo "This Python has no tkinter (the window toolkit)."
    echo "Homebrew: brew install python-tk   then delete .venv and run this again."
    echo "Or install Python from python.org, which includes it."
    read -r; exit 1
fi
exec .venv/bin/python run.py hubgui --setup cams.json
