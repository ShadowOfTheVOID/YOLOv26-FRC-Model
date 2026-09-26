# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that pulls FRC match videos from The Blue Alliance, cuts them to
main-camera footage, exports frames with scoreboard-derived labels, and trains
a YOLO26 detector for fuel (the 2026 game piece), robots and hubs. The trained
model feeds `run.py detect` / `run.py count`, which write into a SQLite
scouting database served read-only to a separate scouting app
(`ShadowOfTheVOID/frc-scouting`). The local checkout on the maintainer's Mac is
`~/dev/TBACroppedOutVid`.

## Commands

```bash
# harvest / database / API environment (no torch)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# training / detection environment -- kept separate so torch never reaches the API host
python3 -m venv .venv-train && .venv-train/bin/pip install "ultralytics>=8.4" opencv-python-headless

# tests: one script, no network, no ffmpeg, no video files; CI runs exactly this
.venv/bin/python tests/test_pipeline.py
# a single test (pytest is not a dependency, but the functions are pytest-shaped)
python3 -m pytest tests/test_pipeline.py -q -k test_db
```

There is no linter or formatter configured. `tests/test_pipeline.py` lists its
tests explicitly in `main()` — a new test function must be added there or CI
will not run it.

`run.py` is the CLI for everything on the harvest side (`pull`, `stream`,
`live`, `export`, `db`, `serve`, `detect`, `count`, `audit`, `verify`, ...);
the README has the full table. The TBA key comes from `$TBA_AUTH_KEY` or a
gitignored `.env`. `data/`, `state/` and `.env` are gitignored; `TBAVID_DATA`
moves bulk data elsewhere (`tbavid/config.py`).

## Architecture

**Three roles, three dependency sets** (`DEPLOY.md`). Harvest box: ffmpeg,
yt-dlp, tesseract, `requirements.txt`. API host: python3 and nothing else.
Detect box: harvest set plus `requirements-detect.txt`. The API path —
`serve.py`, `tbavid/api.py`, `tbavid/db.py` — must import only the standard
library; `test_api_stays_stdlib` enforces it, because the API runs under
systemd on hosts with no pip.

**Harvest flow** (`tbavid/pipeline.py` orchestrates): `tba.py` picks
competition matches (the ledger in `state/seen.json` guarantees a video is
never pulled twice; `--shard N/M` splits work across people) → `download.py`
→ `shots.py` keeps main-camera shots → `crop.py` / `formats.py` remove the
broadcast banner using per-district layout profiles → `render.py` → frames
into `data/frames/<event>_<match>_<ytkey>_NNNNNN.jpg` → `scoreboard.py` OCRs
fuel counts into `data/labels/<match>.csv`. `stream.py` does the same from a
whole event-day stream by finding matches with `audio.py` cues. Per-video
state lives in `data/review/manifest.json`; `db.py` builds `data/scouting.db`
from it.

**Training flow** (`train/`, see `train/README.md`):
`prepare_dataset.py` (splits by match, never by frame, hash-stable) →
`drop_offcamera.py` → `autolabel_fuel.py` → optional `autolabel_robots.py`
(YOLOE text prompts; drops frames it cannot label fully) and
`autolabel_objects.py` (hubs replay geometry recorded with `run.py db hub`) → `subset_classes.py` → `train.py` → `benchmark.py`. Label
once, derive twice: the five-class scouting detector and the three-class
counting model (`fuel`, `hub_blue`, `hub_red`) come from one labelled set.
Training writes weights only; `run.py detect --weights .../best.pt` is what
fills the database's `detections` table afterwards.

**Scoring and scouting logic** is pure state, no model or video, so it is
tested without a GPU: `count.BallCounter` counts balls into each hub (a fuel
track that enters a hub region and vanishes, with guards against blobs,
occlusion and pass-overs); `shooting.ShotCounter` adds whose they were and the
misses — a shot is a ball that starts at a robot and gets clear of it, and
broken flights are stitched back together. Its per-hub totals must equal
`BallCounter`'s for the same balls; a test holds them to that.

## Labelling: what was learned the hard way

`autolabel_fuel.py` is a colour heuristic with several passes — colour gate,
cluster split (watershed for small groups, Hough for dense ones), shade rescue
for balls in hoppers, reflection filter, a learned field line, and a stripe
test for painted lines. Every one of them is scaled off the frame's
*isolated* balls. Consequences:

- It works on close broadcast shots with scattered fuel (the `2026nhdur`
  matches) and fails on wide shots whose fuel sits in one corral (gal, kylou,
  mimas, mimil: 6–22 isolated balls against 85). Frames it cannot handle are
  quarantined to `dataset/skipped/` rather than labelled — an unboxed pile
  teaches the detector that fuel is background. `prepare_dataset.py --matches`
  restricts a dataset to broadcasts it handles.
- Crowd shots pass every per-frame check (the yellow is inside boxes; it is
  just T-shirts). `drop_offcamera.py` catches them because the camera is fixed
  within a match: distance from the match's median frame, as a MAD z-score.
  Run it before labelling.
- Shading scales value and leaves saturation alone. Relative-saturation tests
  separate shaded fuel from walls and reflections; brightness tests do not.
- Tune against real frames, not synthetic ones. Twice a synthetic test passed
  while real frames failed (a "reflection" made by dimming keeps its
  saturation; no floor does that). `--preview --show-dropped` colours boxes by
  pass; `train/diagnose_labels.py` proves which code is running;
  `train/preview_labels.py` draws what is actually in the label files.

The path to the corral frames and the other broadcasts is the bootstrap in
`train/README.md` — train, relabel with the detector, correct, retrain — not
more thresholds.

## Training on the AMD Developer Cloud MI300X

Full walkthrough: `deploy/AMD_DEVCLOUD.md`. What bit on the first real run:

- On the PyTorch 1-Click image torch lives in a Docker container named `rocm`.
  `docker`, `scp`, `rocm-smi`, `tmux` run on the host; `python3`, `pip`,
  `train.py` run in the container (`docker exec -it rocm /bin/bash`, work in
  `/workspace/frc`). "command not found" means the wrong side.
- Install Ultralytics with `--no-deps` or pip replaces ROCm torch with PyPI's
  CUDA build silently. The hand-carried list must include `polars`.
- `train.py` repoints a moved dataset's `path:` automatically.
- `NNPACK ... Unsupported hardware` spam is harmless (one per dataloader worker).
- `yolo26s --p2 --imgsz 1280` crashes in the loss's TaskAlignedAssigner: one
  16.6 GiB allocation refused with 177–186 GiB free. It asks for the same
  16.6 GiB at batch 16 and batch 4, so batch is not what sizes it. Every
  crash had P2 at 1280; the warmup at 960 without P2 ran clean at batch 16.
  Which of the two is responsible has not been separated — `--p2 --imgsz
  960` is the next experiment. Known good: `--imgsz 960`, no `--p2`.
- `received 0 items of ancdata` / `Pin memory thread exited unexpectedly` is
  the dataloader exceeding the container's open-file limit on a dense batch.
  `train.py` shares tensors via `/dev/shm` to avoid it; `ulimit -n 65536`
  works on older code. On the torch 2.12+rocm7.14 image the helper that
  sharing starts (`torch_shm_manager`) could not find librocm-openblas.so.0
  nor libamdhip64 / librocprofiler-sdk (all in the venv's `_rocm_sdk_*`
  folders); `train.py` puts every such folder on `LD_LIBRARY_PATH`. Resume with `--model runs/<name>/weights/last.pt
  --resume`.
- Run training under `nohup ... > train.log 2>&1 &`; a foreground run dies
  when the terminal is needed for anything else.

## Conventions

- `CHANGELOG.md` has an `## Unreleased` section and every user-visible change
  gets an entry there. A release is a `## vX.Y.Z` section plus a pushed tag;
  `.github/workflows/release.yml` builds from the changelog and fails on a tag
  with no section.
- Comments and docstrings explain the failure that motivated the code, with
  the measured numbers — match that when changing anything. Commit messages
  follow the same style: why, what was measured, what it does not fix.
- CI also fails if a TBA key or a `*WITH_KEY*` file is tracked.

## Current work (as of 2026-09-26 — delete this section once stale)

**Goal**: per-robot scouting (who shot, made, missed) and replacing FMS
scoring at the 10-st-throwdown scrimmage, **Saturday 2026-10-10**.

**v0.3.0 is prepared but not yet tagged** (this session cannot push tags; the
user tags `main` after merging PR #8). Two models become release assets (not in git; checksums in
CHANGELOG.md): `fuel_best.pt` (fuel) and `fuel_withBotbest.pt` (fuel,
robot_blue, robot_red), both yolo26s at imgsz 960, trained only on the
2026nhdur broadcast. QUICKSTART.md is how to run them. Validation numbers are
agreement with the auto-labellers, not with reality.

What the qm7 video runs established (see CHANGELOG v0.3.0 for the detail):
- Robot detection is good: every box checked by eye was a real robot in the
  right alliance colour. Robots hidden behind people or hubs come back under
  new numbers (1058 was #5/#8/#11); `--teams` folds them. Merging by position
  was tried in reasoning and would have given 1058 the number of 611 -- do
  not merge by guess.
- **Broadcast-angle counting is not FMS-grade**: the scoreboard showed 74
  balls in qm7's first 30 s; ShotCounter's hub totals saw 19 and `run.py
  count` 0. The tracker cannot hold ~15 px balls fired in streams (thousands
  of broken tracks per 30 s). Threshold tuning will not close that gap.
- Shot rules that were each measured wrong on real video and fixed: a ball a
  robot drives away from is not a shot (must travel itself); a ball pushed
  ahead is not a shot (must also move relative to the robot); a hopper ball
  shot later still counts (sliding launch window); flights are capped at
  2.5 s (stitch chains walked through piles for 4-8 s); frame windows scale
  with fps (sources are 60 fps). Per-robot **misses** remain the least
  reliable output -- do not present them as scouting data.
- Hub active/inactive is not modelled; the rule is unconfirmed.

**Recommended path for the scrimmage** (put to the user, awaiting answers):
a close camera per hub (entry or exit chute) with a line-crossing counter,
built and validated on a recording of a practice hub; a human scorekeeper
regardless. Open questions: can they record a hub, is the scoreboard OCR
right (red 63 at 30 s), live or recorded counting (the Mac processed 18 fps
against 60 fps video).

Training/droplet notes: the MI300X guide is `deploy/AMD_DEVCLOUD.md`; the
scouting dataset recipe is `autolabel_fuel` -> `autolabel_robots` (fuel
greyed before YOLOE; unknown-alliance robots painted out; `--min-robots` 2)
-> `subset_classes --classes fuel,robot_blue,robot_red --copy`.
