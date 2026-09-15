#!/usr/bin/env bash
# Build distributable archives for a GitHub release.
#
#   ./deploy/make_release.sh            -> dist/tbavid-<date>.{zip,tar.gz,tar.xz}
#   ./deploy/make_release.sh v0.2.0     -> dist/tbavid-v0.2.0.{...}
#
# Every artifact is keyless and each one is checked for the key before it is
# kept. Source only: no videos, frames, database or model weights.
set -e
cd "$(dirname "$0")/.."

VER=${1:-$(date +%Y.%m.%d)}
NAME="tbavid-$VER"
DIST="dist"
STAGE="$DIST/$NAME"

rm -rf "$STAGE"; mkdir -p "$STAGE"

# Explicit allowlist. A denylist would eventually miss something, and the thing
# it misses is the file with the key in it.
for item in run.py serve.py config.json requirements.txt README.md SCOUTING.md \
            QUICKSTART_DEBIAN.txt tbavid train tests deploy; do
  [ -e "$item" ] && cp -R "$item" "$STAGE/"
done
find "$STAGE" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true
find "$STAGE" -name '*WITH_KEY*' -delete 2>/dev/null || true

# Refuse to ship anything containing the key, whatever the filename.
if [ -f .env ]; then
  KEY=$(grep '^TBA_AUTH_KEY=' .env | cut -d= -f2- | tr -d '\r\n')
  if [ -n "$KEY" ] && grep -rqF "$KEY" "$STAGE" 2>/dev/null; then
    echo "ABORT: the staged tree contains your TBA key:" >&2
    grep -rlF "$KEY" "$STAGE" >&2
    rm -rf "$STAGE"; exit 1
  fi
fi

( cd "$DIST" && zip -qr "$NAME.zip" "$NAME" )
tar -czf "$DIST/$NAME.tar.gz" -C "$DIST" "$NAME"
tar -cJf "$DIST/$NAME.tar.xz" -C "$DIST" "$NAME"

# Same check again, on the finished artifacts rather than the staging tree.
if [ -f .env ] && [ -n "${KEY:-}" ]; then
  for f in "$DIST/$NAME".zip "$DIST/$NAME".tar.gz "$DIST/$NAME".tar.xz; do
    case "$f" in
      *.zip)    body=$(unzip -p "$f" 2>/dev/null) ;;
      *)        body=$(tar -xzOf "$f" 2>/dev/null || tar -xJOf "$f" 2>/dev/null) ;;
    esac
    if printf '%s' "$body" | grep -qF "$KEY"; then
      echo "ABORT: $f contains the key" >&2; rm -f "$f"; exit 1
    fi
  done
fi

rm -rf "$STAGE"
echo "dist/"
for f in "$DIST/$NAME".zip "$DIST/$NAME".tar.gz "$DIST/$NAME".tar.xz; do
  printf "  %-34s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done
echo
N=$(tar -tzf "$DIST/$NAME.tar.gz" | grep -vc '/$')
echo "All keyless and verified. $N files per archive."
