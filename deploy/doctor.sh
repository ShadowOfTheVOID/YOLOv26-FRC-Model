#!/usr/bin/env bash
# Explain what is in each folder and flag mismatches that actually matter.
cd "$(dirname "$0")/.."
PY=${PY:-python3}
count() { ls "$1" 2>/dev/null | wc -l | tr -d ' '; }

echo "=== data/  (pipeline output) ==="
echo "  data/frames/   $(count data/frames) files   <- one JPG per sampled frame"
echo "  data/labels/   $(count data/labels) files   <- one CSV per MATCH (scoring timeline)"
echo "  data/videos/   $(count data/videos) files"
echo "  data/raw/      $(count data/raw) files"
echo
echo "  These two are different things. frames >> labels is normal:"
echo "  ~120 frames per match, but only 1 scoring CSV per match."
echo
$PY - <<'PY'
import json, pathlib, collections
mp = pathlib.Path("data/review/manifest.json")
if not mp.exists():
    print("  no manifest yet"); raise SystemExit
vids = json.loads(mp.read_text()).get("videos", {})
ok   = [v for v in vids.values() if v.get("status") == "ok"]
sb   = [v for v in vids.values() if (v.get("score") or {}).get("series")]
bad  = [v for v in vids.values() if (v.get("score") or {}).get("error")]
csvs = len(list(pathlib.Path("data/labels").glob("*.csv"))) if pathlib.Path("data/labels").exists() else 0
print(f"  matches in manifest        {len(vids)}")
print(f"  status ok                  {len(ok)}")
print(f"  scoreboard read cleanly    {len(sb)}   <- should equal the CSV count ({csvs})")
print(f"  scoreboard FAILED          {len(bad)}")
if bad:
    print("\n  Failed scoreboard reads (these produce no CSV, frames are still fine):")
    for v in list(bad)[:6]:
        print(f"    {v.get('event_key','?'):10} {(v.get('score') or {}).get('error','')[:60]}")
    reasons = collections.Counter((v.get("score") or {}).get("error","")[:40] for v in bad)
    print("\n  by reason:")
    for r, n in reasons.most_common():
        print(f"    {n:3}x {r}")
PY
echo
echo "=== dataset/  (built by prepare_dataset.py) ==="
for s in train val; do
  i=$(count "dataset/images/$s"); l=$(count "dataset/labels/$s")
  printf "  %-5s images %-6s labels %-6s" "$s" "$i" "$l"
  if [ "$i" = "0" ]; then echo "  (not built yet)"
  elif [ "$i" != "$l" ]; then echo "  <-- MISMATCH: run train/autolabel_fuel.py"
  else echo "  ok"; fi
done
echo
echo "  If labels here is 0 or far below images, training will silently learn"
echo "  nothing. dataset/labels must have one .txt per image."
