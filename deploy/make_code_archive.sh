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
set -e
cd "$(dirname "$0")/.."

WITH_KEY=0
[ "${1:-}" = "--with-key" ] && WITH_KEY=1

FILES="run.py serve.py config.json requirements.txt tbavid/ train/ tests/ deploy/ README.md SCOUTING.md QUICKSTART_DEBIAN.txt"

if [ "$WITH_KEY" = "1" ]; then
  if [ ! -f .env ]; then echo "no .env to include" >&2; exit 1; fi
  OUT=${2:-tbavid_code_WITH_KEY.tgz}
  tar -czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' $FILES .env
  chmod 600 "$OUT"
  echo "wrote $OUT ($(du -h "$OUT" | cut -f1))  mode 600"
  echo
  echo "  !! This archive CONTAINS your TBA key."
  echo "  !! Copy it only to machines you control (scp). Never upload it to"
  echo "  !! Kaggle, Colab, GitHub or anywhere shareable."
  echo "  !! For those, run this script with no arguments instead."
else
  OUT=${1:-tbavid_code.tgz}
  # *WITH_KEY* files live in deploy/ alongside everything else, and deploy/ is
  # bundled wholesale -- without this exclude the "safe to upload" archive
  # quietly carries the key. That is not hypothetical; it happened.
  tar -czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='*WITH_KEY*' $FILES

  # Belt and braces: if a key is configured, prove it is not in the output.
  if [ -f .env ]; then
    KEY=$(grep '^TBA_AUTH_KEY=' .env | cut -d= -f2- | tr -d '\r\n')
    if [ -n "$KEY" ] && tar -xzOf "$OUT" 2>/dev/null | grep -qF "$KEY"; then
      echo "REFUSING: $OUT contains your TBA key." >&2
      rm -f "$OUT"
      exit 1
    fi
  fi
  echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
  echo "Contains no .env, no key, no data/, no state/ -- safe to upload."
fi
