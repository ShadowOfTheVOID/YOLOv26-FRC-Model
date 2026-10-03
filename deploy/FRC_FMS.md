# Hub counting into frc-fms

[frc-fms](https://github.com/arnan-bajaj/frc-fms) is a scrimmage FMS. Its
vision side only reports timestamped fuel events, `(t, hub, n)`, to
`POST /api/vision/events`. The FMS decides from its own match timeline which
period each ball belongs to, calls the AUTO result and drives the hub
lights. This repository connects to it in two ways. Both were run against a
real frc-fms server (2026-10-01, notes at the end).

| | what runs | use it when |
|---|---|---|
| **A. Plugin** | frc-fms's own `vision/run_vision.py`, with this repo's colour counter as its counter | you run frc-fms's stack as its README says |
| **B. Sender** | this repo's `run.py hubgui` / `run.py hubfeed`, posting to frc-fms | you want this repo's web page: outlines drawn on the picture, ball measuring, calibration, several cameras per hub, `/board` |

Either way the counter is `hubcount.CrossingCounter`. On the four scored
Einstein matches (deploy/HUB_FEED.md) it measured 10.1% held-out error at
60 fps, with the AUTO winner right on all four. The same four matches
measured 17% (30 fps) to 25% (60 fps) for model + tracker counting, which is
what frc-fms's built-in `zone` counter does. It needs no model and no GPU.

Either way it can also blend in the retrained fuel model (the combo:
6.8% on the same four, 7-9% expected on a new match; needs Apple silicon or
a GPU): `ComboCounter` in A, `--model` or the page's *Fuel model* in B.
Either way frc-fms's control page shows it under *Setup -> Vision* as
"<fps> fps <counter>", with its "vision ok" pill.

## A. Plugin: frc-fms runs our counter

In frc-fms's `config/vision.yaml`:

```yaml
defaults:
  counter: "tbavid.fms_counter:ColourCounter"
  fps: 60
hubs:
  red:
    source: 0
    roi: [800, 300, 320, 180]        # from pick_roi.py: box the funnel mouth
    ball_area: 900                   # one ball, in pixels, on this camera
  blue:
    source: 1
    outline: [[810, 300], [1110, 300], [1100, 470], [820, 470]]   # or a polygon
    ball_area: 850
```

Then run vision with this repository on the Python path:

```bash
cd frc-fms/vision
PYTHONPATH=~/dev/TBACroppedOutVid python run_vision.py --config ../config/vision.yaml --preview
```

Each hub needs one of these zone settings:
- `roi` (frc-fms's own box, used as the mouth outline);
- `outline` (a polygon);
- `line` plus `out` (an exit line, counting balls that cross towards `out`);
- `setup: cams.json` plus `camera: NAME`, to reuse a camera drawn in
  `run.py hubgui` (outlines, ball size and blur together).

**`ball_area` is required.** Measure it with a few balls lying near the hub:

```bash
python run.py hubfeed --source 0 --red 810,300,1110,300,1100,470,820,470 --measure 5
```

Within 0.5x-2x of the true area the error stays within 5 points, so it does
not have to be exact. Measuring it automatically from the first frames was
tried and refused: on the Einstein 4 recording those frames are the title
card, and it measured 98 px against a real 272.

frc-fms's recordings and `rescore.py` work unchanged with the plugin. They
only call `process(frame, t)`.

### The combo in the plugin

```yaml
defaults:
  counter: "tbavid.fms_counter:ComboCounter"
  model: /path/to/fuel_relabel.pt
  model_weight: 0.5       # optional; the tuned value
  device: mps             # optional; cuda, then mps, then cpu otherwise
  fps: 60
```

The hub keys are ColourCounter's. Ultralytics must be installed where
frc-fms's vision runs (its `vision/requirements.txt` already has it). The
model runs on its own thread, so frc-fms's camera loop is never held up:
- **Live:** a model that skips more than 20% of its frames after 10 s is
  turned off and the hub counts by colour. The console says
  `[tbavid] hub red: the model skipped 311 of 320 frames -- turned off`,
  which is what happened running frc-fms's `run_vision.py` on this
  container's 4 CPU cores. The posted totals were then exactly the colour
  plugin's (red 108, blue 152 on Einstein 1).
- **`rescore.py`:** it passes the timestamps saved with the recording. The
  plugin sees frames stamped more than 30 s in the past (or video time
  running ahead of the clock) and waits for the model on every frame, so a
  re-count skips nothing. Speed alone was tried first and failed: on a busy
  CPU rescore.py read slower than real time, looked live, and the model was
  dropped after skipping 321 of 326 frames.
- frc-fms's page shows the counter as `tbavid.fms_counter:ComboCounter`
  (its runner names the plugin). The `--preview` window shows both halves.

Checked against our own counter on Einstein 1, 6-36 s, with the model
allowed to finish every frame (a GPU stand-in on this CPU):
- **The plugin and `run.py hubcount --model` agree.** Fed the same frames,
  the red model half matched frame for frame (94 / 94 over the whole
  30 s), with both hub plugins loaded as frc-fms loads them. Blue was
  88 / 87.
- **`rescore.py` keeps the model.** On a re-encoded clip of the same 30 s
  it counted red 103 (our counter: 101 on the original video; the
  re-encode alone moved colour 107 -> 112).
- **The model half depends on which frames it samples.** It runs on every
  other frame of a 60 fps camera. Started one frame later, so on the odd
  frames, red's model half read 106 instead of 94 (+13%) and blue's 84
  instead of 87. The blended counts moved less: red 107 against 101, blue
  94 against 95. Colour, which sees every frame, did not move. This is
  sampling noise in the model half over 30 s, not a plugin fault; the
  tuning (deploy/HUB_FEED.md) measured whole matches on one phase only.

## B. Sender: this repo's page posts to frc-fms

Give a URL as the target, with frc-fms's `vision_key` (`server.vision_key`
in its `config/event.yaml`) as the user part:

```bash
python run.py hubgui --setup cams.json
#   then in the page's target box: http://VISIONKEY@192.168.1.10:8000
python run.py hubfeed --setup cams.json --target http://VISIONKEY@192.168.1.10:8000
```

A `host:port` target still means bioarena's UDP feed. What changes for
frc-fms (`tbavid/fmslink.py`):
- **Wall-clock timestamps.** Each ball is sent with `time.time()` of the
  frame it was seen in, moved from the capture clock. frc-fms buckets balls
  into periods by that time, so a ball posted late still lands in the
  period it was scored in.
- **Nothing is dropped.** Events stay queued until a POST that carried them
  succeeds. frc-fms adds events rather than taking a running total, so a
  lost event would be a lost ball.
- **Its own thread.** A POST runs every 0.25 s on a separate thread, so a
  slow FMS never stalls a camera loop.
- **Status.** The page and `/board` show "sent to frc-fms" or "frc-fms not
  answering". The match score is on frc-fms's own `/display`, which knows
  the periods and active hubs.
- **frc-fms's control page** (*Setup -> Vision*, and the "vision ok" pill)
  gets each hub's frame rate and counter, e.g. "59.0 fps tbavid colour +
  model", in the `status` frc-fms reads. A dead camera or one under 28 fps
  is sent as an error, which turns the pill red. A model dropped for being
  too slow is named there ("model off ... too slow") but is not an error,
  as counting goes on by colour. Before 2026-10-02 only the counter's name
  was sent, and that panel read "undefined fps".

Practice mode (the built-in bioarena stand-in) does not apply. Run frc-fms
itself to rehearse, or its mock vision.

## Using a retrained model with frc-fms's `zone` counter

`zone` takes every box the model returns. A three-class model
(`fuel_withBotbest.pt`, or one trained from `train/relabel_video.py`'s
`fuel, robot_blue, robot_red` set) would count robots as fuel there.

Either give `zone` a fuel-only model
(`train/subset_classes.py --src dataset_relabel --classes fuel`, then
train), or add `classes=[0]` to the `self.model.predict(...)` call in
frc-fms's `vision/counters/zone.py`.

frc-fms's match recordings (`vision/recordings/<match>_<hub>.mp4`) come
from the real camera mount. They are the best training data there is: feed
them to `train/relabel_video.py --video`.

## What was checked (2026-10-01, frc-fms at 5fc1549)

A scratch copy of frc-fms (`python -m fms.server`, test vision key) was fed
the Einstein 4 recording, played at camera speed.

- **B, `run.py hubfeed --target http://KEY@127.0.0.1:8000`:** frc-fms stored
  red 172 and blue 93 fuel, timestamped in wall-clock time over the run.
  `run.py hubcount` on the same video reaches exactly 172 / 93 at the
  moment the run was stopped (37.9 s in). Nothing was lost or duplicated.
- **A, frc-fms's `run_vision.py` with the plugin:** both hubs ran at
  59.9 fps, and frc-fms's status showed
  `tbavid.fms_counter:ColourCounter`. It stored red 217 and blue 159 with
  no duplicated events. Offline, at the point each hub had reached, the
  counts are 219-222 and 157-160. Run offline on the same frames, the
  plugin and `count_recording` give the same counts.

## Rechecked 2026-10-02, frc-fms at 20524ce

frc-fms gained 20 commits: first-run setup (`python -m fms.init` writes
`config/event.yaml` and `config/vision.yaml`), `vision_key` read from
`event.yaml`, and its own UDP count feed to bioarena (`feeds:`, `--feed`).
The counter interface and `/api/vision/events` did not change.

- **A, plugin:** Einstein 1 at 59.9 fps in its `run_vision.py`. frc-fms
  stored red 108 / blue 152, equal to the counter's totals and to an offline
  run over the same frames, with no duplicates.
- **B, `run.py hubfeed --target http://KEY@...`:** stored the same 108 / 152.
- **frc-fms's count feed** was accepted by `run.py hubfeed-listen`, so the
  two follow the same protocol. bioarena itself has no receiver yet
  (deploy/HUB_FEED.md).

frc-fms's counters, Einstein 1 (held out of the retraining dataset's
training split), 6-172 s, 30 fps, mean error over the official checkpoints:

| counter | error | at 30 s (truth 95 / 96) | final (truth 621 / 415) | AUTO |
|---|---|---|---|---|
| `tbavid.fms_counter:ColourCounter` | 13.9% | 85 / 102 | 509 / 465 | right |
| `zone`, `fuel_relabel.pt` (retrained) | 37.8% | 52 / 58 | 295 / 270 | right |
| `zone`, `fuel_best.pt` (v0.3.0) | 51.5% | 48 / 39 | 228 / 169 | wrong |

Counts are blue / red. The retrained model cuts `zone`'s error by 14 points
and gets AUTO right where the old one did not. `zone` still undercounts by
about half: it counts a ball only if a box confirms it inside the hub
rectangle in two frames, and fast 15 px balls often fail that. The colour
plugin remains the counter to use in frc-fms. Use `fuel_relabel.pt` there
only when a model is wanted. Never use a 3-class model with `zone`, because
it counts robots as fuel.

