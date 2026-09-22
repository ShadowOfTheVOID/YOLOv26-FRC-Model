# TBACroppedOutVid

Turn FRC match broadcasts into a YOLO training set and a scouting database.

Given a TBA API key, it picks match videos you have never pulled before,
throws away everything that is not the main field camera, crops off the
burned-in scoreboard, exports sampled frames, and reads the fuel counters off
the scoreboard before discarding it so every frame is labelled with what
scored and when.

```bash
python3 run.py pull -n 20        # harvest per-match videos TBA links
python3 run.py stream --url <s> --event 2026casnf   # or a whole event day
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
[SCOUTING.md](SCOUTING.md). The crop is checked against a per-feed layout
profile, so a divider a single-camera broadcast does not have can no longer
crop away half its field — see
[Which broadcast is this](#which-broadcast-is-this).

| | |
| --- | --- |
| harvesting | this README |
| the dataset itself + provenance | [DATA.md](DATA.md) |
| scouting database + API | [SCOUTING.md](SCOUTING.md) |
| training a detector | [train/README.md](train/README.md) |
| running it on Colab / Kaggle / Debian | [deploy/SETUP.md](deploy/SETUP.md) |
| running it on Windows | [deploy/WINDOWS.md](deploy/WINDOWS.md) |
| hosting the API so nobody starts it by hand | [deploy/HOSTING.md](deploy/HOSTING.md) |
| deploying the whole thing, in order | [DEPLOY.md](DEPLOY.md) |
| counting balls with no scoreboard | [DEPLOY.md](DEPLOY.md#5-counting-balls-with-no-scoreboard) |
| running a trained `.pt` over the frames | [DEPLOY.md](DEPLOY.md#4-run-it) |

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
| `formats` | broadcast layout profiles: list, explain, calibrate |
| `stream` | read a whole event-day stream, cutting each match out by its audio cues |
| `live` | scout a live feed: read the scoreboard as it happens, keep no video |
| `detect` | run a trained `.pt` over harvested frames and record what it found |
| `count` | count scored fuel from the detector alone — no scoreboard, no OCR |
| `scoreboard` | re-read the scoreboard counters via OCR |
| `verify` | check OCR'd fuel totals against TBA's official score breakdown |
| `reprocess` | re-run crop/render/scoreboard on already-downloaded sources |
| `db build` / `db sync` | build the scouting database (`sync` also fetches rosters) |
| `db export` | a copy of the database that can be served from a read-only disk |
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

## Scouting a live feed, keeping nothing

Everything above builds a **training set**: it downloads a broadcast, crops it,
samples frames and fills a disk, all for a detector that does not exist yet.
That is no use to somebody who wants to know how an alliance is scoring this
afternoon.

`live` is the other job. It reads the one thing in a broadcast that is already
ground truth — the burned-in fuel counter — off a live feed, and throws every
frame away the moment it has been read.

```bash
./run.py live --url <twitch or youtube live> --event 2026caclv --match qm14
./run.py live --url <stream> --event 2026caclv --hub http://localhost:6059
```

**It keeps nothing.** Each pass records a few seconds to a temp file, OCRs the
two counter boxes, and deletes it. Peak disk is one chunk. No frames, no
cleaned video, nothing enters the manifest or the dataset — the only thing
written is the scoring rows, which is what a scouting app reads. Verified: the
clip is unlinked on the failure paths too, including when the probe raises
part-way through.

What you get is the **scoring timeline per alliance** — when fuel went in, to
the second, for the match on the field. Nothing else in either repo produces
that live, and it answers *is this alliance front-loading or finishing strong*
rather than just *how much did they get*.

What you do not get is **which robot**. The counter says an alliance scored and
never which of its three did — the same ceiling [SCOUTING.md](SCOUTING.md)
describes, and being live does not move it. Per-robot needs a trained detector
and a tracker, and neither exists yet.

### It will not guess which match it is watching

`--match qm14` names it, or `--hub` asks a running scouting hub what is on the
field — the hub knows, because that is what arms the six scouting phones. One
of the two is required. A timeline filed against the wrong match key is worse
than no timeline: `db.py` joins the roster onto that key, so it would credit an
alliance's scoring to six robots that were not on the field.

### Two things worth knowing before an event

- **The counters are found by watching the digits move**, so the warm-up has to
  run *while a match is being played*. Pointed at an idle field it will
  correctly find nothing and say so rather than locking onto the match clock.
- **Rows are marked `status='live'`.** A counter read off a broadcast as it
  happened and one read off a downloaded video are the same measurement, but
  not the same evidence — the live one cannot be re-read, because the video is
  gone.

Live rows land in `matches`, `match_teams` and `score_events`, which are what
`team_match_fuel` and `team_summary` are built from — so they appear in a
scouting app through the existing API with nothing to change on that side.

## One stream instead of one upload per match

`pull` walks TBA looking for `match.videos[]` — a per-match link somebody
uploaded. Plenty of events do not have those. A district weekend publishes one
continuous multi-hour broadcast per day and nothing per match, so the picker
finds nothing and `download.py` refuses the stream with `TOO_LONG`.

`stream` is the other way in. Point it at the broadcast once and it finds the
matches inside it, cuts each one out, and hands the clips to the same pipeline
everything else goes through.

```bash
./run.py stream --url <stream> --event 2026casnf   # a day, in one command
./run.py stream --file day1.mp4 --event 2026casnf  # one already on disk
./run.py stream --url <stream> --listen-only       # what can you hear? no writes
```

### How it finds the matches

Not from the video. Every stage in `shots.py` works on what the camera is
looking at, and between matches the camera is looking at the same field from
the same place — there is no cut to find.

From the **audio**. The field plays a sound to start a match and a sound at the
buzzer, and in a multi-hour stream those are the only events that are all of:
loud, *tonal* (a horn puts its energy in a few narrow bands; applause and
commentary do not), repeated dozens of times, and **separated by a fixed
interval**, because a match is a fixed length.

That last property is the one that does the work, and it is why there is no
frequency anywhere in [`tbavid/audio.py`](tbavid/audio.py). A band-pass filter
at the horn's pitch would need that pitch from somewhere, and a wrong guess
finds nothing while looking exactly like a stream with no matches in it.
Instead every loud tonal burst is collected and the *spacing* that repeats most
consistently is the match. A crowd can be loud twice; it cannot be loud twice
at the same spacing forty times running.

So it does not care whether the audio is the FIRST Webcast Unit's YouTube feed,
a Twitch re-encode, or a phone held up in the stands — which matters for the
California district, where Central Valley is on Twitch and the rest are not.

`--listen-only` prints the measured interval whether or not it agrees with
`stream.match_s`, so a season with a different match length is one config edit
rather than a detector that quietly finds nothing.

### What it refuses to do

Audio says *where* the matches are. It cannot say *which* they are, and a match
key is the most damaging thing here to get wrong: `db.py` joins a roster onto
it, so one mislabelled clip credits an alliance's fuel to six robots that were
not on the field.

Identity comes from TBA's schedule, and only when the evidence supports it:

| situation | what happens |
| --- | --- |
| confirmed cue pairs == TBA's match count | aligned in order, one to one |
| short, and cues recovered from a single heard horn close the gap **exactly** | aligned in order — TBA's count is the corroboration |
| `--from-match qm14` | aligned from there; the operator knows something the audio cannot |
| more intervals than TBA has matches | **no keys assigned** — at least one is not a match and nothing says which |
| anything else | **no keys assigned** |

The last two are not failures. **Training frames do not need a match key;
scouting rows do.** An unidentified clip is still a cropped, main-camera view
of a real field with real robots on it, so it keeps its frames and enters the
database under its own video id — never a real match key, so no roster joins
onto it and no team is credited with anything. What it loses is attribution,
which is exactly what was not established.

A match whose buzzer is not in the recording is dropped rather than shipped as
a fragment: the scoreboard reader takes its final count off the end of the
video, so a 20-second stub under a real match key would write a confident,
wrong final fuel.

Every clip records what established its identity — a count that agreed, an
operator's say-so, or nothing — so a row in the scouting database can always be
traced back to it.

## Which broadcast is this

Step 4 above measures the overlay from the pixels, which needs no list of
events kept up to date and was already right on feeds nobody had looked at.
What a measurement cannot do is know when it is wrong, and it fails silently
both ways:

- **A band that isn't there.** The divider hunt looks for the strongest static
  horizontal edge in the lower half of the frame. On a single-camera feed that
  edge is the guardrail, the scoring table or the front row of the bleachers,
  and if it clears the threshold the bottom of the field is cropped away — and
  the frames still look perfectly plausible.
- **A band that is and reads shallow.** The banner walk stops at the first row
  that moves, so a layout with a gap between the banner and the field stops it
  early and leaves a strip of scoreboard in the training set, which is the one
  thing a detector must never get to see.

A **format profile** is the missing opinion: what this feed's layout is
supposed to look like, so the measurement can be checked against something.
Profiles live in [`tbavid/formats.py`](tbavid/formats.py).

```bash
./run.py formats                        # the profiles, and what selects each
./run.py formats --district ca          # which one an event would get, and why
./run.py formats --calibrate --video <id>   # what a real download measures
```

**The measurement still wins.** A profile is not a set of coordinates — that is
`crop.overrides`, and it needs a human to read numbers off a frame for every
event. A profile does three things and no more:

1. tunes the detector's thresholds for this layout before it runs;
2. supplies the fallback when detection is inconclusive, per layout, instead of
   the single global `crop.top` that otherwise has to serve every feed at once;
3. vetoes a band its layout does not have — the one case where the pixels
   genuinely mislead.

| profile | layout | provenance |
| --- | --- | --- |
| `ca_district` | FIRST California **weekend district events**: one field camera, banner on top, no permanent side panel | declared |
| `champs_split` | 2026 Championship: field camera over a low side camera, static divider between | measured |
| `generic` | unknown feed — detect and take the answer, as before profiles existed | measured |

### Two things about the California district in particular

**It is not one production.** The 2026 FIRST California district runs its
weekend events as `2026caclv` (Central Valley), `2026casnf` (San Francisco),
`2026calas` (Los Angeles), `2026caven` (Ventura County), `2026caoec` (Orange
County) and Aerospace Valley. Most of those are YouTube webcasts carried by the
FIRST Webcast Unit; Central Valley goes out on Twitch instead. Different rigs,
so `ca_district`'s banner range is wide and per-event calibration is worth
doing — `crop.formats` takes one event key at a time for exactly that.

**The state championships are excluded.** `2026cancmp` (FIRST California
Northern State Championship) and its southern counterpart carry district `ca`
and are a different, larger production. `ca_district` is gated to TBA
`event_type` 1 — a weekend district event — so a state championship falls
through to `generic` instead. That matters in the direction people forget: the
profile's job is to *veto* a split-screen divider, and vetoing one that is
genuinely there keeps a whole side-camera panel in the training set. A veto is
only safe on a layout somebody has established. An event whose type could not
be read fails the gate for the same reason, rather than having it waived.

```bash
./run.py formats --district ca --event-type 1    # -> ca_district
./run.py formats --district ca --event-type 2    # -> generic, deliberately
```

**`provenance` is load-bearing, not a comment.** `measured` means the numbers
came out of this pipeline reading that feed's footage. `declared` means they
describe the layout as specified and no footage has been through
`formats --calibrate` yet — so the profile's *veto and tuning* are in force,
but its fallback fractions are a starting point rather than a reading.
`ca_district` is `declared`. Calibrating it is one command against any
California district match already in the manifest, and it prints the profile
its measurements imply rather than writing anything:

```bash
./run.py pull -n 4                            # any weekend CA district event
./run.py formats --calibrate --event 2026casnf
./run.py cropcheck --video <id>               # then look at the frames
```

Worth calibrating the Twitch-carried event separately from a Webcast Unit one:
`2026caclv` against `2026casnf` is the comparison that says whether one profile
can cover both.

Nothing promotes itself from `declared` to `measured`; that edit is a person's,
after they have looked at the frames.

### What picks the profile

The event's district, from TBA. `/events/{year}/simple` — the one request per
season the video picker already makes — carries each event's `district` object,
so the district is known before the video is downloaded and costs nothing
extra. A regional has `district: null`, which is also how a regional is
recognised.

Order of precedence, first match winning:

1. `crop.formats[<event key>]` in config.json — one event
2. `crop.format` — the whole run
3. the event's TBA district abbreviation (`ca` → `ca_district`), subject to
   the profile's `event_type` gate
4. the event key, then the video title

```json
"crop": {
  "format": "auto",
  "formats": {"2026casj": "generic"}
}
```

Both config routes mean the same thing: a person has looked at the footage. An
unknown profile name there stops the run rather than harvesting a batch against
the wrong layout.

Every entry in the manifest records which profile it got and why, and the crop
notes carry the reconciliation — including a refused divider, so a veto is
visible afterwards instead of being a number that quietly differs.

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

## Cutting a release

Add a `## vX.Y.Z` section to [CHANGELOG.md](CHANGELOG.md), then push the tag:

```bash
git tag -a v0.3.0 -m "v0.3.0" && git push origin v0.3.0
```

CI does the rest — runs the suite, refuses to proceed if a key is committed,
builds the `.zip`/`.tar.gz`/`.tar.xz`, extracts the notes from the changelog,
checks the built archive actually runs, and publishes the release with
`SHA256SUMS` attached. A tag with no changelog section fails rather than
publishing empty notes.

To rehearse without publishing, run the `release` workflow manually from the
Actions tab with a version — it builds and verifies, and uploads the artifacts
for inspection without creating a release.

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
| `crop.format` | force one broadcast layout profile for the whole run (`auto` picks per event) |
| `crop.formats` | `{event_key: profile}`, for one event at a time |
| `stream.match_s` | how long a match lasts, for the cue-interval search |
| `stream.match_window_s` | how far the measured interval may sit from it |
| `stream.pre_roll_s`, `stream.post_roll_s` | padding either side of the cues |
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
- **`ca_district` is a `declared` profile, not a measured one.** Its veto and
  its thresholds are in force, and those are what it is mainly for — but its
  banner range is the layout as specified rather than a reading off California
  district footage, because none has been through this pipeline yet. One
  `formats --calibrate` run against a weekend district event replaces it with
  measurements, and it is worth doing twice: the district's events do not all
  come off the same rig. See
  [Which broadcast is this](#which-broadcast-is-this).
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
