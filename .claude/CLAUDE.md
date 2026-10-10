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

**v0.3.0 is tagged; v0.4.0's notes are in CHANGELOG** (sessions cannot push
tags; the user tags `main` and attaches `fuel_relabel.pt` / `scout_relabel.pt`
by hand, checksums in its section). v0.3.0 history: Two models become release assets (not in git; checksums in
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

**bioarena `main` (f4987b0, checked 2026-10-02) has NO counter feed
receiver**: no listener on 8411 and no Counted mode. `assignAutoWinner` picks
random or forced red/blue at AUTO start, so the UDP feed decides nothing
there until bioarena ships the spec. frc-fms (20524ce) has its own feed to
the same spec, and our receiver stand-in accepts it.
**Scrimmage scoring runs through bioarena** (Team 841's cheesy-arena fork).
Its "Hub FUEL Counter Feed" spec is the contract: UDP JSON to
10.0.100.5:8411, cumulative red/blue per session, never decreasing, sent on
change plus a 100 ms heartbeat, `age_ms` from capture; the AUTO winner is
decided at T+23.000 s so camera-to-count must stay under ~200 ms p99.
`run.py hubfeed` (`tbavid/hubfeed.py` stdlib sender + `Receiver` stand-in,
`tbavid/hubcount.py` the area-crossing counter from
`experiments/area_hub_count.py`) implements it; `deploy/HUB_FEED.md` is the
runbook. Several cameras: `--setup cams.json`
(`Setup`/`HubTally` in hubcount.py; zones per hub combined by sum/max/median,
any camera stale holds the heartbeat). `run.py hubgui` edits the same cams.json: a web page
(`hubweb.py` + `hubweb.html`, stdlib), a thin view over
`hubapp.HubController` -- put logic there, not in the page. The user chose
the website; the Qt window was removed. It also takes a Twitch/YouTube
stream as a source (`hubcount.is_stream_page`, yt-dlp), which the user
wants kept; streams are seconds late, so never for the AUTO call.
`run.py track` (`trackvis.py`): renderer with `BallTracker`, distance-based
linking -- ByteTrack loses balls at the apex because a turning 15 px ball
stops overlapping its prediction (1 of 25 flights vs 106 with BallTracker on
Einstein 4). shots/count still use ByteTrack; switching them is untested.
The released models MISS balls high against the crowd (autolabel_fuel's
field line left them unlabelled in training); `trackvis.colour_assist`
adds moving yellow blobs at a continue-only confidence. Real fix: relabel
with balls in flight and retrain.
Wireless cameras: `hubcount.is_network_camera` (rtsp/http, not a stream
page), FFmpeg over TCP unbuffered, reconnect forever, pre-drop blobs cleared.
Zones are an outline (funnel mouth) or an exit line (`ExitLineCounter`,
`"line"`+`"out"`); the user wanted exits counted (exits equal entries, rim
bounces never reach an exit), but on Central Valley (2026-10-05, exits in
view) exit lines were 82-85% wrong -- balls pile at the exit -- and the
page now recommends a raised outline (top edge 3-4 balls above the hood;
38.9% -> 6.4% there). Exits are not visible enough on broadcasts
to validate (9-34%); only a practice hub can. Per-camera `blur`/`remove_static` exist only as calibration
against a hand-counted recording -- the best value differed on every Einstein
match. Plumbing is tested; counting is unvalidated on a practice-field
camera -- the spec's 20-ball acceptance test comes before bioarena's
`counted` mode. Do not swap in `BallCounter` at its default
windows: their 400 ms hold breaks the budget (the --model blend uses it
at vanish 2 / reacquire 2, ~170 ms, and only for half of each ball). Full sweep (HUB_FEED.md "Every fix together"), held-out mean error
over Einstein 4/5/1: mouth tuned 30%, model+BallTracker+assist 16%,
mean of both 13%, exits 76% (hidden on broadcasts); AUTO winner right in
every fold. The model path is offline only. Sweep scripts were scratch,
not in the repo. Then (2026-09-29) outlines changed to downward entries
only, exits ignored, one ball learned from crossings (30th pct of last 80),
round up at 0.65, default blur 0.3: held-out 13%, shipped defaults 16% /
6% / 9%. More knobs (outline size/offset, clump caps) overfit -- 27-28%
held out -- so do not add knobs without a held-out check.
End goal is a live scoreboard: bioarena's display is the official one; the
counter also serves `/board` (`hubboard.html`, `hubapp.board_view`) --
bioarena's credited score + clock/shift/hub_active from its status reply
when linked, camera counts (labelled, not a match score) otherwise.
Measured 59.9 fps, 5 ms median / 14 ms max capture-to-count on 60 fps video.
Tuning has plateaued on E4/E5/E1 (another sweep: 14% held out vs 13%);
don't retune on these three -- more scored matches are needed, and YouTube
refuses downloads here (bot check), so they must come via Drive. Setup
tolerance: outlines +-10 px / +-15% and ball 0.5-2x cost <=5 points; frame
rate is what matters (30 fps 12.6%, 20 fps 25.1%) -> MIN_FPS warning.
Einstein 8 (from Drive, 2026-09-29) was the first match held out from all
tuning: 9.1% with the shipped defaults via count_recording, AUTO right.
Four-way leave-one-out re-tuning is WORSE (21%) than keeping the settings
(10.1% mean); best four-match fit gains <1 point. Settings unchanged.
Every-frame (60 fps) run of the model counter on all four (HUB_FEED.md
"Every frame at 60 fps"): model+assist 25% held out at 60 fps vs 17% at 30;
combos with the colour counter 13-14% vs colour alone 10.1% (60 fps) /
11.1% (30 fps). The colour counter at 60 fps is the best; the model adds
nothing.
Model on full-res 640 px hub crops (imgsz 640, same cost as 960 full
frame): moving-ball recall 84% -> 93%; counting held out, no assist 36% ->
21%, with assist 17% -> 17%; mean(colour, crop model) 9.0% vs colour 10.1%
(first combo to beat colour, 1 point on 4 matches). Not built into run.py
yet. Retrained fuel_relabel on E1 crops (2026-10-02; E1 is the only
Einstein match it never trained on): alone 18.0% -> 12.1%, mean(colour,
model) 7.8% vs colour 9.1% -- same as old model's 7.7%; retraining helped
the model, not the combo. Then built and tuned (2026-10-02):
`--model fuel_relabel.pt` on hubcount/hubfeed (tbavid/hubmodel.py; cams.json
"model"), w 0.5, BallCounter 2/2/2 + require_entry + pad 0.08 widths:
E4/E5/E8/E1 8.7/5.6/8.4/4.5% (mean 6.8% vs colour 10.1%), AUTO right on all
four. E4/5/8 are in the model's training set; fair E1 estimate 7-9%. Do
not raise w on the E4/5/8 fit (0.7) -- that is the training set talking. User chose: if crops don't clearly work, retrain (option 2).
frc-fms (github.com/arnan-bajaj/frc-fms, the team's own scrimmage FMS;
2026-10-01) is supported alongside bioarena (deploy/FRC_FMS.md): plugin
tbavid.fms_counter:ColourCounter for its run_vision.py, and fmslink.FmsSender
when --target is http://KEY@host:8000 (wall-clock event times, never-dropped
queue). Both verified against a running frc-fms. Its `zone` counter takes all
model classes -- a 3-class model would count robots as fuel.
Drive videos download with curl from
drive.usercontent.google.com/download?id=ID&export=download&confirm=t.

**Recommended path for the scrimmage** (put to the user, awaiting answers):
a close camera per hub (entry or exit chute) with a line-crossing counter,
built and validated on a recording of a practice hub; a human scorekeeper
regardless. Open questions: can they record a hub, is the scoreboard OCR
right (red 63 at 30 s), live or recorded counting (the Mac processed 18 fps
against 60 fps video).

Hub Counter app (`apps/hubcounter/`, hub-app.yml, all four OSes): bundles
torch + Ultralytics + fuel_relabel.pt (`model: built-in`, hubmodel.bundled_model;
the source release is `.github/models-release`, and release.yml copies both .pt
files onto every release). Both Linux builds take CPU torch (PyPI's carries CUDA: arm64 3 GB >
2 GB asset limit). torchvision's `_C_stable` ops are loaded by path, so the
spec lists them by hand; the smoke test's `--selftest-model` is what catches
that, since every page check passed without them.
Watchtower app (`apps/watchtower/main.py`, built by the same spec with
HUBAPP=watchtower from a clone of watchtower-fms at `.github/watchtower-release`):
runs fms.init into ~/Documents/Watchtower/config, uvicorn in a thread on :8000,
and the hub counter with `HubController.default_target` set to
http://KEY@127.0.0.1:8000, plus bioarena's host:port (`fmslink.FanOut`, one
comma-separated target box): vision.yaml's first `feeds:` or the spec's
10.0.100.5:8411, read-only on Phones & PINs (the user wants nothing typed) --
Watchtower forwards to bioarena only from its own vision runner, which the
app does not run. FanOut.linked is any target answering. The public site's
offline page is deploy/watchtower-offline.html (Watchtower's look, inlined CSS)
via deploy/Caddyfile.watchtower handle_errors. fms.server reads event.yaml at import, so chdir and
FMS_CONFIG come first. It is a pywebview desktop app (home.html + settings.py,
which edits event.yaml line by line; the user wants no hand-editing of YAML).
Watchtower's pages are top-level windows, never frames: Watchtower keeps logins
in localStorage, which WebKit partitions for cross-origin frames. Our hub page
has no login, so it IS framed, as Home's "Hub cameras" tab (the user wants
Watchtower = our setup UI + its add-ons). The Hub Counter app also opens in its
own pywebview window (apps/hubcounter/main.py run_window). pywebview walks a
js_api object's public attributes: give it methods only (HomeApi), or on
Windows it touches WebView2 off the UI thread and the window never loads. Saving
settings relaunches the process (fms.server reads config at import and starts
an unstoppable TBA thread). Linux/Pi fall back to the browser. Qt's engine
(local testing only) hangs with a persistent profile and a second window:
WATCHTOWER_WEBVIEW_PRIVATE=1.
One-click updates (`tbavid/appupdate.py`, stdlib): both apps check GitHub's
latest release at start against `app_version.txt` (the spec writes it from
HUBCOUNTER_VERSION; 0.0.0 = never check). Update button only, refused while
counting (the user chose that over automatic). Mac/Windows install via a
helper script that waits for the app to exit; Linux/Pi get a link. v0.5.0 is the
first build with it (0.4.4 -> 0.5.0 is a manual download); the first real
one-click update is 0.5.0 -> 0.5.1. v0.5.2: TBA tab split into event / Write key
(auth_id + auth_secret, both password fields) / Read key; settings pages centred.
v0.5.3: pass guard (a counted ball that came in AND left the outline sideways
and flew on is taken back as owed; measured HUB_FEED.md "Passes beside the hub")
and outline + exit line on one hub take the larger kind, not the sum.
Also in v0.5.3: HubTally cross-check (exits vs outline `lag` s ago, warns
outside 0.8-1.25, score untouched) and per-hub `"confirm": {hub: s}` (count =
high-water of exits + outline entries in the last s seconds), OFF by default;
`run.py hubcount --hand red=N` is the practice-field test (2026-10-09 plan in
HUB_FEED.md "Practice field"). On broadcasts the exits are blind, so confirm
is useless there (CV 37/106 of 159/810).
Match-day safety (Unreleased, after v0.5.4): HubController.start's worker
restarts hubcount.run when it stops by itself (same sender, so session and
counts carry on; recordings are not restarted; user_stop vs stop_evt), the
setup is locked while counting (hubweb.EDITS refused via ctl.editable()),
and tbavid/keepawake.py holds the machine awake from the counting thread.
Page step 5 "Test" (v0.5.4; v0.5.3 was tagged before it merged): ball test (HubController.start/check_ball_test,
baseline of sender.counts), speed (hubcount.latency_summary over run()'s
monitor["latency"], spec 80/200 ms), Record (hubapp.Recorder, mp4v on its own
thread, drops rather than blocks), Test a recording (hubcount.hand_test: every
counting method vs hand count, closest marked, Use applies confirm).
Streamlined setup (2026-10-09, v0.5.5): step 5's "Check a recording"
(hubapp.check_recording) = hand_test + calibrate on one recording; a change is
offered only when worth_changing (beats current by >2 balls or 5%). The step-3
Calibrate button is gone. Step 2 hides combine/confirm unless a hub has >1
outline / outline+exit. tbavid/camcheck.py (numpy only): camera's "reference"
grey thumbnail in cams.json, phase correlation -> ok/moved/different/size;
checked on grab, grab_all (page runs it on opening a setup) and every 10 s
while counting; "Move the outlines" = shift_zones. Untested on a real venue
camera. Bugs fixed then: SystemExit from open_source hung the job queue;
empty hand-count hub sent as 0; test_recording dropped "rules" (blur 0 -> 0.3).
v0.5.4 two boxes, one feed: hubfeed.RelayIn on the main box (UDP 8412, the
partner sends the unchanged spec feed there; Receiver rules, forwards rises
with the datagram's own age_ms, replies with bioarena's last reply; silent
partner = PARTNER in run()'s stale list -> heartbeat held). Page: step 4
"Main box" switch; CLI hubfeed --partner-port.
Watchtower's optional `server.public_url` (Phones & PINs; e.g. our Caddy
proxy https://watchtower.systemoverload.org, nothing built in): when set,
phones get that link and its QR instead of the .local name; the raw IP stays
as the fallback. The app's own hub counter still posts to 127.0.0.1.
Camera presets (`tbavid/camprofile.py`, stdlib; cams.json "preset"/"image"/
"colour"): built-in ELP OV4689 60 fps (720p MJPG, 8 ms, 4600 K -- unchecked
on the real camera) and USB webcam; saved ones in camera-presets.json beside
the setup. `image` applies to numeric sources only, read back, ignored ones
logged. `colour` is the per-camera HSV gate (standard = LOOSE_LO/HI); the
page's magenta overlay is /frame.jpg?fuel=1.
Training/droplet notes: the MI300X guide is `deploy/AMD_DEVCLOUD.md`; the
scouting dataset recipe is `autolabel_fuel` -> `autolabel_robots` (fuel
greyed before YOLOE; unknown-alliance robots painted out; `--min-robots` 2)
-> `subset_classes --classes fuel,robot_blue,robot_red --copy`.
