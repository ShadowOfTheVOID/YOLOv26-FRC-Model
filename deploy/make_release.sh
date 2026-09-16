#!/usr/bin/env bash
# Build distributable archives for a GitHub release.
#
#   ./deploy/make_release.sh            -> dist/tbavid-<date>.{zip,tar.gz,tar.xz}
#   ./deploy/make_release.sh v0.2.0     -> dist/tbavid-v0.2.0.{...}
#
# Every artifact is keyless and each one is checked for the key before it is
# kept. Source only: no videos, frames, database or model weights.
#
# The work happens in deploy/package.py so that macOS, Windows and Linux all
# build the same archive from the same code. This wrapper exists because
# DEBIAN.md, SETUP.md and the Colab notebook call it by name. On Windows, skip
# it and run the Python directly:
#
#   python deploy\package.py release v0.2.0
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
exec "$PY" deploy/package.py release "$@"
