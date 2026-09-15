#!/usr/bin/env bash
# Unattended overnight harvest on macOS.
#
#   ./deploy/overnight.sh 40
#
# Detaches, keeps the Mac awake, and survives the terminal closing. Safe to
# interrupt: the ledger records every video as it completes, so a re-run picks
# up where this left off instead of re-downloading.
set -u
cd "$(dirname "$0")/.."

COUNT=${1:-20}
BATCH=${2:-5}
LOG="logs/overnight_$(date +%Y%m%d_%H%M).log"
mkdir -p logs

if [ ! -f .env ] && [ -z "${TBA_AUTH_KEY:-}" ]; then
  echo "No TBA key (.env or \$TBA_AUTH_KEY). Aborting." >&2; exit 1
fi

FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
# ~100 MB/match with keep_raw and keep_clean off, ~333 MB with them on.
NEED_GB=$(python3 -c "import json;c=json.load(open('config.json'));\
per=0.34 if (c.get('keep_raw') or c.get('keep_clean')) else 0.11;\
print(round(per*$COUNT,1))")
echo "disk free: ${FREE_GB} GB | estimated need: ${NEED_GB} GB"
if [ "$(python3 -c "print(1 if $NEED_GB > $FREE_GB - 5 else 0)")" = "1" ]; then
  echo "Not enough headroom. Set keep_raw/keep_clean false, or lower the count." >&2
  exit 1
fi

run_batches() {
  local done_n=0
  while [ "$done_n" -lt "$COUNT" ]; do
    local n=$(( COUNT - done_n )); [ "$n" -gt "$BATCH" ] && n=$BATCH
    echo "=== $(date '+%H:%M:%S') batch of $n ($done_n/$COUNT done) ==="
    # Batches rather than one big pull: each one commits to the ledger and the
    # database, so a crash at hour six costs one batch, not the night.
    .venv/bin/python -u run.py pull -n "$n" --per-event-cap 2 || echo "batch failed, continuing"
    .venv/bin/python -u run.py db sync || true
    done_n=$(( done_n + n ))
    df -g . | awk 'NR==2 {print "  disk free: " $4 " GB"}'
  done
  echo "=== $(date '+%H:%M:%S') finished ==="
  .venv/bin/python -u run.py status
  .venv/bin/python -u run.py verify || true
}

# caffeinate -is blocks idle sleep for as long as the command runs; without it
# the Mac sleeps mid-download and you wake to a partial night's work.
export -f run_batches 2>/dev/null || true
nohup caffeinate -is bash -c "
  cd '$PWD'
  COUNT=$COUNT BATCH=$BATCH
  $(declare -f run_batches)
  run_batches
" > "$LOG" 2>&1 &

echo "started (pid $!) -> $LOG"
echo "  watch:  tail -f $LOG"
echo "  stop:   pkill -f 'run.py pull'"
