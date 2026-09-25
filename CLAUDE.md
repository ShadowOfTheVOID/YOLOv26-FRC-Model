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
`drop_offcamera.py` → `autolabel_fuel.py` → optional `autolabel_objects.py`
(robots need the cleaned videos; hubs replay geometry recorded with
`run.py db hub`) → `subset_classes.py` → `train.py` → `benchmark.py`. Label
once, derive twice: the five-class scouting detector and the three-class
counting model (`fuel`, `hub_blue`, `hub_red`) come from one labelled set.
Training writes weights only; `run.py detect --weights .../best.pt` is what
fills the database's `detections` table afterwards.

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
- `yolo26s --p2 --imgsz 1280 --batch 16` crashes in the loss's
  TaskAlignedAssigner: one 16.6 GiB allocation refused with 177 GiB free (the
  VF will not grant a single block that large). The tensor scales with
  batch × objects (mosaic packs ~500) × anchors (P2 at 1280 ≈ 136k), so
  `--batch 4` is the current fix; `--imgsz 960 --batch 8` is the fallback.
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

## Current work (as of 2026-09-25 — delete this section once stale)

- Branch `claude/exciting-gauss-cyri8u`, pushed, not merged.
- Dataset: `dataset-fuel/` built with `--matches 2026nhdur`, off-camera frames
  dropped, fuel-only — 721 train / 185 val, 200,252 boxes. Val is a single
  match, so its mAP says "converged", not "works at a scrimmage".
- Droplet: 1x MI300X, reachable from the Mac as `ssh mi300x` (alias in
  `~/.ssh/config`). The first full run crashed on the assigner OOM above; the
  next step was relaunching with `--batch 4` and confirming epoch 1 completes.
  `deploy/amd_watch.sh` (untested) copies `best.pt` to the host as it trains.
- After training: bring home `best.pt`, `results.csv`, `args.yaml`; destroy the
  droplet (it bills while idle, and is destroyed without warning when credit
  runs out); run `run.py detect`; then bootstrap the quarantined frames and the
  six matches `--matches` left out.
