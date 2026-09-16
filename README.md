# TBACroppedOutVid

Turn FRC match broadcasts into a YOLO training set and a scouting database.

Given a TBA API key, it picks match videos you have never pulled before,
throws away everything that is not the main field camera, crops off the
burned-in scoreboard, exports sampled frames, and reads the fuel counters off
the scoreboard before discarding it so every frame is labelled with what
scored and when.

```bash
python3 run.py pull -n 20        # harvest
python3 run.py verify            # check the counters against TBA
python3 run.py serve             # read-only JSON API for a scouting app
```

**Why the cropping matters.** A broadcast frame is not training data. The 2026
Championship feed runs a permanent split screen with a second camera in the
bottom third, every frame carries a score banner, and both would teach a
detector the wrong thing. Both are removed automatically, per video, with no
hardcoded coordinates — see [How it decides what to keep](#how-it-decides-what-to-keep).

**Status.** Working end to end on 30+ matches across ~15 events. Fuel
detection is auto-labelled; robot and hub labels are proposals that need
review. Per-robot attribution is designed but not live — see
[SCOUTING.md](SCOUTING.md).

| | |
| --- | --- |
| harvesting | this README |
| the dataset itself + provenance | [DATA.md](DATA.md) |
| scouting database + API | [SCOUTING.md](SCOUTING.md) |
| training a detector | [train/README.md](train/README.md) |
| running it on Colab / Kaggle / Debian | [deploy/SETUP.md](deploy/SETUP.md) |
| running it on Windows | [deploy/WINDOWS.md](deploy/WINDOWS.md) |

## Why

Two things make a raw broadcast bad training data:

- **The side camera.** The 2026 Championship feed runs a permanent split
  screen: the elevated field camera on top, a low side camera on the bottom
  third, with a static divider between them. It is a *region of every frame*,
  not a shot the director cuts to. Other feeds do also cut away to replays and
  crowd shots. Both get removed.
- **The score banner.** Static clutter a detector will happily learn to key
  off of.

The banner does get read before it is thrown away, though, because it is the
only free ground truth for whether fuel actually went in - see
[Scoreboard labels](#scoreboard-labels).

## Setup

```bash
brew install ffmpeg yt-dlp tesseract
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'TBA_AUTH_KEY=your-key-here' > .env      # from thebluealliance.com/account
```

`.env`, `data/` and `state/` are gitignored.

## Use

```bash
.venv/bin/python run.py pull -n 10          # full unattended run
.venv/bin/python run.py pull -n 10 --review # ...with a human pass before frames
.venv/bin/python run.py pull -n 2 --dry-run # show picks, download nothing
.venv/bin/python run.py status              # ledger + dataset summary
```

Output lands in these folders:

- `data/videos/` — cleaned, cropped, main-camera-only `.mp4` per match
- `data/frames/` — sampled JPGs, named `<event>_<match>_<ytkey>_NNNNNN.jpg`
- `data/labels/` — one CSV per match, a row per exported frame (see below)

### Other commands

| command | what it does |
| --- | --- |
| `fetch` | pick + download + clean, no frame export |
| `review` | shot review UI, then export frames |
| `export --force` | re-export frames after changing sampling config |
| `cropcheck [video_id]` | proof frames with the crop box drawn on |
| `scoreboard` | re-read the scoreboard counters via OCR |
| `verify` | check OCR'd fuel totals against TBA's official score breakdown |
| `reprocess` | re-run crop/render/scoreboard on already-downloaded sources |
| `db build` / `db sync` | build the scouting database (`sync` also fetches rosters) |
| `db hub` | record an event's hub geometry |
| `serve` | read-only JSON API for the scouting app |
| `prune` | delete downloaded sources to reclaim disk |
| `audit` | report which harvested videos are not competition footage |

## How it decides what to keep

1. **Cuts** — one low-res decode pass, then frame-to-frame mean absolute
   difference with an adaptive threshold (median + `cut_sigma`×MAD, floored at
   `cut_min_delta`). ffmpeg's own `scene` filter is *not* used: it is
   histogram-based and scored a real camera cut at 0.037 in testing, because
   two shots of the same venue share a histogram.
2. **Clustering** — each shot gets a signature (mean of its middle 50% of
   frames at 64×36 RGB). Greedy duration-weighted clustering; the fixed main
   camera collapses into one cluster that owns most of the runtime, so the
   longest-total-duration cluster wins. Near-duplicate clusters are absorbed
   so exposure drift doesn't split the main camera in two.
3. **Quarantine** — if the winning cluster covers less than
   `reject_threshold` of the runtime, the classification isn't trustworthy.
   That video is logged loudly, kept out of `data/frames/`, and parked in
   `data/review/quarantine/` for `run.py review` to rescue.
4. **Crop** — burned-in graphics are pixel-static, so a per-row temporal
   variance map over main-camera frames finds them. Median across columns, not
   mean, so the ticking match clock doesn't hide the band. Two bands come out:
   the score banner at the top, and the split-screen divider in the lower half,
   below which everything is the side camera. A strict threshold (1% of field
   variance) anchors the divider — only a real graphic is that still — and a
   loose one (8%) then widens to its true edges; the loose threshold alone
   false-positives on the dark, barely-moving bleacher rows *inside* the main
   camera's own panel. Sides are left alone: the hubs sit close to the left and
   right edges.

   On the calibration match this turned a 1920x1080 broadcast into a
   1920x504 main-camera strip.

## Scoreboard labels

Each alliance has a cumulative fuel counter in the banner, rendered `N / M`.
`N` steps up every time fuel scores, so the difference between two reads is
the number of balls scored in between. (The big centre numbers are *points*,
not balls — they include the autonomous bonus, which is why they sit a constant
offset above the counter. `M` is a target threshold and does step mid-match, so
nothing may assume it is constant.)

Finding the counters is automatic. Inside a band that is otherwise pixel-static
the only things that vary are the digits, so connected components of the
variance map give candidate boxes; a candidate is a fuel counter if it OCRs as
`N / M` (the point total is a bare integer — the slash is what separates them),
never goes backwards, and actually moves. Which alliance it belongs to comes
from the box's own background colour.

Reads are cleaned into a monotonic step function: a value can never decrease,
large jumps need two consecutive reads to agree (killing a stray digit turning
129 into 1129), and small increments are taken on a single read — demanding
confirmation for those loses the counter exactly when an alliance is scoring
hardest.

`data/labels/<video_id>.csv` then has a row per exported frame:

| column | meaning |
| --- | --- |
| `frame` | filename in `data/frames/` |
| `t_clean` | timestamp in the cleaned video |
| `t_source` | timestamp in the original broadcast |
| `<alliance>_total` | cumulative fuel scored at that moment |
| `<alliance>_scored_next` | fuel scored in the next `score_window_s` |
| `<alliance>_scored_prev` | fuel scored in the previous `score_window_s` |

**The counter lags the ball.** It updates a beat after fuel actually lands, so
`_scored_next` is the column you want for "this frame shows a shot that went
in". Calibrate the lag with `score_latency_s` if you need it tighter. And note
the counter tells you *an alliance* scored, not *which robot* shot it —
attributing a score to a specific bot is the model's job.

`run.py verify` cross-checks the OCR'd totals against TBA's official
`score_breakdown` for that match.

## Splitting the work across several people

Hand the same code to N people, give each a different slice, and no two of
them will ever download the same match:

```bash
python3 run.py pull -n 20 --shard 1/4      # person 1
python3 run.py pull -n 20 --shard 2/4      # person 2
python3 run.py pull -n 20 --shard 3/4      # person 3
python3 run.py pull -n 20 --shard 4/4      # person 4
```

No server, no shared file, no messaging. Each worker walks the same TBA
catalogue and keeps only the videos where
`crc32(youtube_id) % 4 == their_slice`. The partition is a property of the
video id, so it holds without anyone talking to anyone.

It is `crc32` rather than the builtin `hash()` on purpose: Python randomises
string hashing per process, so `hash()` would reassign videos to different
people on every run and the guarantee would evaporate.

Everyone must use the **same M** and a **different N**. Get that wrong and you
get overlap — `merge` counts and reports it rather than hiding it.

### Collecting the results

On each person's machine:

```bash
python3 run.py export-share ~/my_harvest
```

That bundles `frames/`, `labels/`, `manifest.json` and `seen.json`. Send the
folder to whoever is collecting, who runs, once per person:

```bash
python3 run.py merge ~/from_alice
python3 run.py merge ~/from_bob
python3 run.py db sync
```

Merging is a union keyed on video id, so it is safe to run twice and safe to
run out of order. Frames that already exist are left alone.

## What counts as a match

Only official competition footage is ever picked. Two filters do it:

- **The event.** TBA's `event_type` has to be one of regional, district,
  district championship, championship division, championship final or Festival
  of Champions. Offseason (99), preseason (100) and unlabeled (-1) events are
  never walked — they run mixed or stand-in rosters, non-standard fields and
  demo rules, which is not what a competition detector should learn.
- **The match.** Inside a kept event, only real match play counts:
  `qm`, `ef`, `qf`, `sf`, `f`. Practice matches are skipped even when TBA has
  a video for them.

`--include-noncompetitive` (or `"competitive_only": false` in `config.json`)
widens the catalogue to the whole season if you want it.

## Never pulling the same video twice

`state/seen.json` records every YouTube video ID ever considered, keyed on the
video ID rather than the match key (one stream can be linked to many matches).
Statuses: `ok`, `too_long`, `too_short`, `unavailable`, `failed`, `rejected`.
All of them are skipped on later runs; `--retry-failed` reconsiders only
`failed` and `unavailable`.

TBA responses are cached in `state/tba_cache/` and revalidated with
`If-None-Match`, so repeat runs cost almost nothing. Event walking is lazy —
a season has ~1000 events and we stop as soon as enough unseen videos are
found.

## Config

All knobs live in `config.json`; see `tbavid/config.py` for defaults.

| key | meaning |
| --- | --- |
| `season` | year to pull from |
| `competitive_only` | official competition events and match play only (default true) |
| `max_duration_s` | reject longer videos (TBA sometimes links full-day streams) |
| `min_shot_s` | drop shots shorter than this even if main-camera |
| `cut_sigma`, `cut_min_delta` | cut-detection sensitivity |
| `shot_cluster_dist` | how similar two shots must be to be the same camera |
| `shot_absorb_dist` | looser distance for folding clusters into the winner |
| `reject_threshold` | minimum main-camera coverage before quarantine |
| `crop.*` | banner auto-detect toggle and manual fractions |
| `sample_fps` | frames per second of cleaned video to export |
| `dedupe_hamming` | perceptual-hash distance below which a frame is a duplicate |
| `score_labels` | read the scoreboard at all |
| `score_sample_fps` | how often to OCR the counters |
| `score_window_s` | lookahead/lookbehind for the `_scored_next/prev` columns |
| `score_latency_s` | shift, if you calibrate the counter's lag |
| `max_frames_per_video` | cap after dedupe (0 = unlimited) |
| `keep_raw` | keep downloads so a later `review` can re-render without re-downloading |

Frames are exported at the cleaned video's **native resolution**. Fuel balls
are only ~25px across at 1080p; pre-shrinking here would cost detail that
training at a larger `imgsz` can otherwise exploit.

A note on `dedupe_hamming`: the dHash grid is 17x16, not the classic 9x8. At
9x8 each cell of a 1920x504 strip covers 213x63 px, so a moving ball is
invisible to the hash and genuinely different frames score a median distance of
only 3 — the default threshold of 6 was discarding 380 of 539 real frames as
"duplicates". At 17x16 the median is 18 and dedupe does its actual job, which
is dropping dead-time stills.

## Scouting app

`data/scouting.db` plus a read-only JSON API — see **[SCOUTING.md](SCOUTING.md)**.

```bash
.venv/bin/python run.py db sync
.venv/bin/python run.py serve
```

## Training

See **[train/README.md](train/README.md)**. Fuel boxes are auto-labelled by
colour (56k in ~3s); robots and hubs are proposed by
`train/autolabel_objects.py` and need review.

## Where to run it

Harvest locally (YouTube blocks datacenter IPs), train on Colab or a rented
GPU (73 h on an 8 GB M2), host the API anywhere. See
**[deploy/SETUP.md](deploy/SETUP.md)** for click-by-click Colab and Replit
setup, [deploy/README.md](deploy/README.md) for the reasoning, and
[`deploy/colab_train.ipynb`](deploy/colab_train.ipynb) for the notebook.

## Known limitations

Measured on 30 matches across ~15 events, so these are observed rather than
theoretical:

- **Scoreboard reading fails on roughly 40% of events.** Offseason and some
  district broadcasts use layouts the detector has not met — a scoreboard at
  the *bottom* of the frame, or fuel shown as a bare number in a gauge rather
  than `N / M`. A failed read is flagged (`scoreboard_ok=0`) and left NULL
  rather than guessed at.
- **Those same videos can keep scoreboard in the frames.** The layouts that
  defeat the reader also defeat the crop. Use
  `prepare_dataset.py --scoreboard-ok-only` to train on verified matches only,
  or set a per-event `crop.overrides` entry after checking with
  `run.py cropcheck`.
- **Robot and hub labels are proposals**, not ground truth. Colour-plus-motion
  still boxes people wearing alliance colours near the field.
- **Per-robot attribution is not live.** `alliance_fuel` is alliance-level; the
  scoreboard never says which of three robots scored. The identity layer
  (`tbavid/identify.py`) needs a trained detector and a tracker first.
- **Downloading YouTube video is against YouTube's terms.** That is true
  wherever you run it. See [deploy/README.md](deploy/README.md).

## Not included

Labeling, the dataset YAML, and training. YOLO26 needs a current `ultralytics`
on Python ≥3.10 — a separate environment from this harvester.
