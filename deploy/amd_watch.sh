#!/usr/bin/env bash
# Watch a training run in the droplet's `rocm` container until it finishes,
# keeping the weights safe on the HOST the whole time.
#
# Run on the host (root@<droplet-name>), not inside the container:
#
#   nohup bash deploy/amd_watch.sh fuel26_mi300x > /root/frc/watch.log 2>&1 &
#   tail -f /root/frc/watch.log
#
# Why it exists: files written inside the container die with the container,
# and the droplet itself is destroyed when the credit runs out. Ultralytics
# rewrites best.pt after every epoch, so copying it out on a timer means the
# worst case is losing a few epochs, not the run. When training ends it
# gathers what is needed to reproduce or improve the model -- weights, the
# per-epoch results and the exact arguments -- into one tarball, and prints
# the scp line to bring it home.
#
# It never deletes anything and never touches the droplet itself: destroying
# the droplet stays a decision for a person.
#
# UNTESTED: written against the 1-Click PyTorch image's layout (a container
# named `rocm`, work under /workspace/frc) but not yet run on a droplet or
# against a simulated one. Treat its first run as the test.
set -u

RUN=${1:-fuel26_mi300x}
CONTAINER=${CONTAINER:-rocm}
WORK=${WORK:-/workspace/frc}
OUT=${OUT:-/root/frc/out}
EVERY=${EVERY:-300}          # seconds between checks

SRC="$WORK/runs/$RUN"
mkdir -p "$OUT"

say() { printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$*"; }
in_box() { docker exec "$CONTAINER" "$@"; }

running() {
  # Matched on the run name, so a second run with another name is ignored.
  in_box ps -eo args 2>/dev/null | grep -q "[t]rain/train.py.*--name $RUN"
}

grab() {
  # Copy one file out if it exists; quiet when it does not yet.
  in_box test -f "$SRC/$1" && docker cp "$CONTAINER:$SRC/$1" "$OUT/$(basename "$1")" >/dev/null
}

epochs_done() {
  # results.csv has a header row, then one row per finished epoch.
  local n
  n=$(in_box sh -c "wc -l < '$SRC/results.csv'" 2>/dev/null) || { echo 0; return; }
  echo $(( n > 0 ? n - 1 : 0 ))
}

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  say "no container named '$CONTAINER' -- run this on the droplet host, not inside it"
  exit 1
fi

if ! running; then
  say "no training run named '$RUN' is running in $CONTAINER"
  say "(start it first, or pass the right name: bash $0 <run-name>)"
  exit 1
fi

say "watching $RUN; copying best.pt to $OUT every ${EVERY}s"
last=-1
while running; do
  done_now=$(epochs_done)
  if [ "$done_now" != "$last" ]; then
    grab weights/best.pt && say "epoch $done_now done -- best.pt saved to $OUT"
    grab results.csv
    last=$done_now
  fi
  sleep "$EVERY"
done

say "training process has exited after $(epochs_done) epochs; collecting results"
for f in weights/best.pt weights/last.pt results.csv args.yaml results.png; do
  grab "$f" || say "  (no $f)"
done

# Did it finish, or die? The log is the only place that says.
if in_box test -f "$WORK/train.log"; then
  docker cp "$CONTAINER:$WORK/train.log" "$OUT/train.log" >/dev/null
  if grep -q "Traceback" "$OUT/train.log"; then
    say "the log contains a Traceback -- it crashed rather than finished:"
    grep -v NNPACK "$OUT/train.log" | tail -15
  fi
fi

( cd "$OUT" && tar -czf weights.tgz ./*.pt results.csv args.yaml 2>/dev/null )
say "packed $OUT/weights.tgz"
say "from your Mac:  scp mi300x:$OUT/weights.tgz ~/Desktop/"
say "then destroy the droplet in the console -- it bills while it idles"
