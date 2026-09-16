#!/usr/bin/env bash
# Package the code for another machine.
#
#   ./deploy/make_code_archive.sh                  -> tbavid_code.tgz     (no key)
#   ./deploy/make_code_archive.sh --with-key       -> tbavid_code_WITH_KEY.tgz
#
# The default excludes .env on purpose: this archive is what gets uploaded to
# Kaggle and Colab, where notebooks and their versions are shareable and
# permanent. --with-key is for copying to a machine you control, over scp, and
# the filename is loud so the two never get mixed up.
#
# The work happens in deploy/package.py so that macOS, Windows and Linux all
# build the same archive from the same code. On Windows, skip this wrapper:
#
#   python deploy\package.py code
#   python deploy\package.py code --with-key
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
exec "$PY" deploy/package.py code "$@"
