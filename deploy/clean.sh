#!/usr/bin/env bash
# Get the tree ready to commit. Removes build junk, then refuses to pass if
# anything carrying the TBA key would be tracked.
#
#   ./deploy/clean.sh          # clean + check
#   ./deploy/clean.sh --check  # check only, change nothing
set -u
cd "$(dirname "$0")/.."
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

if [ "$CHECK_ONLY" = "0" ]; then
  echo "=== removing build artifacts ==="
  find . -path ./.venv -prune -o -path ./.venv-train -prune -o \
         -name '__pycache__' -type d -print -exec rm -rf {} + 2>/dev/null | head -5
  find . -path ./.venv -prune -o -path ./.venv-train -prune -o \
         -name '*.pyc' -delete 2>/dev/null
  find . -name '.DS_Store' -delete 2>/dev/null
  rm -rf dist
  rm -f tbavid_code*.tgz
  rm -rf data/.ocr_work/* 2>/dev/null
  echo "  removed: dist/, tbavid_code*.tgz, __pycache__, *.pyc, .DS_Store"
  echo
  echo "  kept (gitignored, not uploaded): data/ state/ dataset/ runs/ .venv*/"
fi

echo
echo "=== what git would track ==="
if command -v git >/dev/null && [ -d .git ]; then
  TRACKED=$(git ls-files --cached --others --exclude-standard)
else
  # No repo yet: approximate by applying .gitignore with rsync.
  TMP=$(mktemp -d)
  rsync -a --filter=':- .gitignore' --exclude='.git' ./ "$TMP/" 2>/dev/null
  TRACKED=$(cd "$TMP" && find . -type f | sed 's|^\./||' | sort)
fi
echo "$TRACKED" | wc -l | xargs echo "  files:"
echo "$TRACKED" | sed 's/^/    /' | head -12
echo "    ..."

echo
echo "=== secret check ==="
FAIL=0
if [ -f .env ]; then
  KEY=$(grep '^TBA_AUTH_KEY=' .env | cut -d= -f2- | tr -d '\r\n')
  if [ -n "$KEY" ]; then
    HITS=$(echo "$TRACKED" | while read -r f; do
             [ -f "$f" ] && grep -lF "$KEY" "$f" 2>/dev/null; done)
    if [ -n "$HITS" ]; then
      echo "  !! THE KEY WOULD BE COMMITTED, in:"; echo "$HITS" | sed 's/^/     /'
      FAIL=1
    else
      echo "  key not present in any tracked file"
    fi
  fi
fi
echo "$TRACKED" | grep -i 'with_key' && { echo "  !! a WITH_KEY file is tracked"; FAIL=1; } \
  || echo "  no WITH_KEY files tracked"
echo "$TRACKED" | grep -qx '.env' && { echo "  !! .env is tracked"; FAIL=1; } \
  || echo "  .env not tracked"
[ -n "$TMP" ] && rm -rf "$TMP" 2>/dev/null

echo
if [ "$FAIL" = "1" ]; then
  echo "NOT SAFE TO PUSH -- fix the above first."; exit 1
fi
echo "Safe to push:"
echo "  git init && git add -A && git commit -m 'FRC scouting video pipeline'"
echo "  gh repo create TBACroppedOutVid --private --source=. --push"
