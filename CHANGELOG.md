# Changelog

Notable changes per release. Dates are the release date, not the merge date.

Each `## vX.Y.Z` section below is the release body: `.github/workflows/release.yml`
reads it with `python3 deploy/package.py notes vX.Y.Z` when the tag is pushed,
so the notes and this file cannot drift apart. A tag with no section here fails
the build rather than publishing an empty release.

## Unreleased

Nothing yet.

## v0.3.0 — 2026-09-26

The first release with trained models. It adds counting fuel into hubs,
per-robot shot scouting (`run.py shots`), robot labelling without hand
labels, and training on an AMD MI300X. It covers everything since v0.2.0;
v0.2.1 was tagged without notes.

### Models (attached to this release, not in the source archives)

| file | classes | trained on | validation (one held-out nhdur match) |
| --- | --- | --- | --- |
| `fuel_best.pt` | fuel | 721 frames, 2026nhdur, 100 epochs | mAP50 0.613, mAP50-95 0.321, P 0.666, R 0.623 |
| `fuel_withBotbest.pt` | fuel, robot_blue, robot_red | 659 frames, 2026nhdur, early-stopped at 62 epochs | fuel mAP50 0.581; robot_blue 0.699; robot_red ~0.71 |

Both are YOLO26s at `--imgsz 960`. Check the downloads with `sha256sum`:

```
202429d3e8a7640e444e8053b229998b363f698469274829a069dd015994b5f3  fuel_best.pt
0c5267368787e054f9fbd57c49078726306ca3370b7185752c3c2229dfffd14e  fuel_withBotbest.pt
```

What the numbers do and do not mean:

- Validation labels come from the same automatic labellers the models were
  trained on, so the scores measure agreement with the labellers, not with
  reality. Both models have only seen one event's broadcast (2026nhdur).
- On a real match video (nhdur qm7, first 30 s) the broadcast scoreboard
  showed 74 balls scored; the shot counter's hub totals saw 19. Counting from
  a broadcast camera angle is **not** a replacement for FMS scoring. Use a
  close camera per hub, and a human scorekeeper.
- Robot detection is good: every robot box checked by eye on qm7 was a real
  robot in the right alliance colour. Per-robot **misses** are the least
  reliable output and should not be used as scouting data yet.

### Added

- **QUICKSTART.md**: from a fresh clone to running the released models --
  install, download and checksum the weights, preview detections, hub boxes,
  `run.py count`, `run.py shots` with `--teams`, the speed check -- and what
  the measured results say to trust. Linked from the top of the README.

- **`run.py detect` — a trained `.pt` finally has somewhere to go.** Until now
  `ultralytics` appeared in exactly one file, on the training side: a finished
  model loaded nowhere, the `detections` table was written by nothing, and
  `identify.py` waited on "a tracker upstream" that did not exist. Everything
  downstream of the detector was designed and unreachable. This runs the model
  over the exported frames in play order with tracking persisted between them,
  and records boxes, classes, confidences and tracks — each tagged with the
  weights that produced it, so two models' opinions can never be read as one.
  Idempotent per match, so better weights replace a match rather than
  accumulating beside it.
  It does **not** close the per-robot gap and does not pretend to:
  `assign_tracks` still needs a scorer, `identify.py` still measures a bumper
  number at ~5 px tall, and with no scorer it records nothing rather than
  naming whichever team sorts first. `--assign` reports that as the answer.
- **`run.py count` — scored fuel from the detector alone, no scoreboard and no
  OCR** (`tbavid/count.py`, plus `deploy/frc-counter.service`). Everything else
  that produces a fuel number reads the broadcast's burned-in counter, which is
  right when there is one and no answer at all on a field that renders none — a
  practice field, an offseason event, a demo, somebody's own game system. The
  same `.pt` already has a `fuel` class and a hub per alliance, so this counts
  the event itself: a fuel track vanishing inside a hub region.
  Almost all of it is refusing the three things that look identical to that — a
  one-frame false detection, a ball a robot drove in front of (it never crossed
  *into* the hub), and a ball that passed *over* it and comes back the other
  side. The last cannot be settled in the moment, so a score is held for a few
  frames and a reappearance withdraws it, which is the same shape as
  `clean_series` wanting two reads before believing a large jump. Every refusal
  is counted and reported, because a counter that rejects silently is one
  nobody can debug.
  Note the deliberate opposite of `detect`: that one discards fuel track ids
  because it runs at 3 fps where a ball moves further between samples than its
  own width; this runs at a camera's native rate where tracking the ball is the
  whole method.
- **`run.py count --scoreboard` — the scoreboard at a scrimmage**
  (`tbavid/field.py`). With no FMS, the count is not a cross-check against a
  real score, it *is* the score, and a number on a laptop nobody can see, with
  no clock and no way to correct it, is not a scoreboard. So: a match clock, so
  fuel thrown about between matches does not score and the detector stops at
  the buzzer; the score handed out as JSON over stdlib HTTP and as one line
  per change on stdout, with no display of its own because whatever shows the
  score at a field already exists; and a referee's correction, which works
  after the buzzer because that is when corrections happen. `detected`
  and `adjusted` are kept apart in the record and on screen, because "the
  camera missed two" and "the camera saw two that never happened" are different
  facts about a setup. Points per ball are configurable and default to 1, since
  `db.py` already refuses to convert fuel to points and a scrimmage runs
  whatever rules its organiser chose.
- **The training code produces a model for counting, not just for scouting.**
  `train/subset_classes.py` derives a `fuel`/`hub_blue`/`hub_red` dataset from
  the labelled five-class one, remapping the label indices — the step that
  fails silently if it is wrong, since a model trains perfectly happily on fuel
  labelled as hubs. `train.py` gained `--data`, so a derived set can actually
  be trained (it previously hardcoded the five-class path, and counted that
  one's labels while training against another), and `--export`, because on a
  CPU box ONNX is often the difference between keeping up with the camera and
  not. `train/benchmark.py` measures achievable frame rate on the machine you
  will use, reporting the p95 as well as the median: a model averaging 30 fps
  that stalls for 200 ms every few seconds drops balls in the stalls while
  looking fine on the average.
  With two class orders now in play, a model that does not carry its own class
  names is refused by both `detect.load` and `count.run_source` rather than
  assumed to be the five-class one — that assumption would relabel every
  detection without failing anything.
- **The counter checks itself, because nothing else can.** With no FMS there
  is no second number anywhere that would disagree with a wrong count, and the
  failure that matters is silent: a box too slow for the camera misses balls
  between the frames it does see, and the score comes out low with no gap and
  nothing odd about it. `--expect-fps` gives it the camera's rate and it
  reports whether it is keeping up, what fraction of frames went past unseen,
  and every reason it refused a ball — in the same payload as the score.
  Without that rate it reports `keepingUp: null` rather than guessing, since it
  cannot tell a slow processor from a slow camera. And `POST /clock` follows an
  outside clock, which is the only quantity at a scrimmage that can be compared
  against anything — moving it never moves the score.
- **Deployment is documented end to end** in [DEPLOY.md](DEPLOY.md): three
  roles with three different dependency sets, in the order to do them, with
  the `.pt` as the only step left. `requirements-detect.txt` keeps torch and
  ultralytics out of `requirements.txt` on purpose — the API host runs
  `serve.py` with python3 and nothing else, and a test now asserts that the
  whole serving path imports no third-party package, because breaking that
  promise would only show up on a machine nobody is sitting at.
- **`run.py live` — scout a live feed and keep no video.** Everything else
  here builds a training set, which is no use to somebody who wants to know how
  an alliance is scoring this afternoon. This reads the burned-in fuel counter
  off a live stream and deletes every frame the moment it has been read: peak
  disk is one chunk, nothing enters the manifest or the dataset, and the clip
  is unlinked on the failure paths too. What it gives is the per-alliance
  scoring timeline for the match on the field — when fuel went in, to the
  second — which nothing else in either repo produces live. What it cannot give
  is which robot: the counter says an alliance scored and never which of its
  three did, and being live does not move that ceiling.
  It refuses to guess the match: `--match` names it or `--hub` asks a running
  scouting hub what is on the field, and one is required, because a timeline
  filed against the wrong key credits an alliance's scoring to six robots that
  were not on it. Rows are marked `status='live'`, since a reading that cannot
  be re-read is not the same evidence as one that can.
- **`run.py stream` — one broadcast in, one clip per match out.** Events that
  publish a single multi-hour stream per day rather than per-match uploads were
  simply unreachable: the picker looks for `match.videos[]` and finds nothing,
  and `download.py` refuses the stream it is handed. Now the matches are found
  inside it and cut out, and the clips go through the same pipeline as anything
  else. `--file` reads one already on disk; `--listen-only` reports what the
  audio contains without writing anything.
- **Matches are found by listening, not watching** (`tbavid/audio.py`). Between
  matches the camera is looking at the same field from the same place, so there
  is no cut for `shots.py` to find. The field's start sound and buzzer are the
  only events in a broadcast that are loud, tonal, repeated dozens of times and
  separated by a fixed interval — and the fixed interval is what identifies
  them. There is no frequency in the file: a band-pass at the horn's pitch
  would need that pitch from somewhere, and a wrong guess finds nothing while
  looking exactly like a stream with no matches in it. Every loud tonal burst
  is collected and the spacing that repeats most consistently is the match, so
  it works on the Webcast Unit's YouTube feed, a Twitch re-encode or a phone in
  the stands alike.
- **`deploy/frc-harvest.service` and `run.py db export`, so the API is not
  something somebody has to start.** `python3 serve.py` on a laptop dies with
  the lid, and the scouting app then shows an empty column that looks exactly
  like "no footage of these robots" rather than "nothing is listening" -- two
  states it goes out of its way to distinguish. Under systemd it survives a
  crash, a reboot and a host rebuild, and it carries no secrets because a
  GET-only API over a rebuildable database has nothing to authenticate to.
  `db export` exists because `cp` is not good enough and fails confusingly: the
  working database is WAL, a WAL database must create its `-shm` companion
  before even a read-only connection can read it, and the unit mounts its data
  directory read-only on purpose -- so a copied database starts fine and then
  answers every request with "attempt to write a readonly database". Confirmed
  as an unprivileged user against a 0555 directory. The export goes through
  SQLite's backup API, which also avoids catching the file mid-write, and
  leaves one file with no sidecars. See [deploy/HOSTING.md](deploy/HOSTING.md).
- **Identity comes from TBA or not at all.** Audio says where a match is, never
  which one it is, and `db.py` joins a roster onto the match key — so one
  mislabelled clip credits an alliance's fuel to six robots that were not on
  the field. Keys are assigned when the confirmed cue count equals TBA's match
  count, when recovered cues close the gap to it exactly, or when
  `--from-match` says where the day starts. Otherwise none are, and that is not
  a failure: training frames do not need a match key, scouting rows do, so an
  unidentified clip keeps its frames and enters the database under its own
  video id where no roster can join onto it.

- **Broadcast layout profiles** (`tbavid/formats.py`). The crop has always
  measured the overlay from the pixels, which needs no list of events kept up
  to date. What it could not do is know when it was wrong, and it failed
  silently both ways: a divider found on a single-camera feed crops away the
  bottom of the field, and frames that lost half a field still look plausible.
  A profile says what a layout is supposed to be, so a measurement can be
  checked against it. The measurement still wins — a profile only tunes the
  thresholds, supplies the fallback when detection is inconclusive, and vetoes
  a band its layout does not have.
- **`ca_district`, for FIRST California's weekend district events** —
  `2026caclv`, `2026casnf`, `2026calas`, `2026caven`, `2026caoec` and
  Aerospace Valley. One field camera, a banner across the top, no permanent
  side panel, so its main job is that veto. Marked `declared` rather than
  `measured`: the veto and thresholds are in force, but the banner range
  describes the layout as specified, because no California district footage has
  been through this pipeline yet — and those events are not one production
  (most are FIRST Webcast Unit on YouTube; Central Valley is on Twitch), so
  the range is wide and per-event calibration is worth doing.
- **A profile can be gated to a TBA `event_type`,** and `ca_district` is gated
  to a weekend district event. FIRST California's state championships
  (`2026cancmp` and its southern counterpart) carry district `ca` and are a
  different, larger production: selecting them into a profile whose whole
  contribution is a no-split-screen veto would keep an entire side-camera
  panel in the training set if they do run one, which is the failure the
  profiles exist to prevent with the sign flipped. They fall through to
  `generic`, which has no opinion. An event whose type could not be read fails
  the gate rather than having it waived, and config still outranks it — that
  gate stops the pipeline guessing, not a person who has measured one.
- **`run.py formats`** — list the profiles, ask which one an event would get
  and why, or measure a real download with `--calibrate`. Calibration prints
  the profile its measurements imply and writes nothing; promoting a profile
  from `declared` to `measured` stays a person's edit, after they have looked
  at the frames.
- **The district comes from TBA.** `/events/{year}/simple` — the one request
  per season the video picker already makes — carries each event's `district`,
  so the layout is known before the download and costs no extra request. A
  regional reads as no district, which is also how a regional is recognised.
  `crop.format` and `crop.formats` override it per run or per event.
- Every manifest entry records which profile it got and why, and the crop notes
  carry the reconciliation, so a refused divider is visible afterwards rather
  than being a number that quietly differs.
- **`deploy/AMD_DEVCLOUD.md` — training on an AMD Instinct MI300X.** The
  Colab and Kaggle pages assume a small GPU you lose in twelve hours; 192 GB
  that stays up wants different settings, and the obvious ones are wrong. With
  ~1,100 training frames, a batch big enough to fill the card leaves 11
  optimizer steps per epoch and converges worse than the 8 GB M2 did, so the
  page spends the memory on resolution, the P2 head and a bigger checkpoint
  instead, and on eight independent runs rather than one DDP job. It also
  documents the failure that eats an afternoon: `pip install ultralytics`
  inside a ROCm image resolves torch from PyPI, PyPI's torch is the CUDA
  build, it silently replaces the ROCm one, and the run falls back to the CPU
  with nothing in the log to say so.
- **`train.py` prints which chip it got.** ROCm reports AMD hardware through
  `torch.cuda`, so `device: 0` said nothing about whether that was an MI300X
  or a mistake, and `device: cpu` scrolled past as one line. It now names the
  GPU and its memory, reports the HIP version, and on a CPU fallback says
  which of the two causes it is looking at.
- **`train.py --batch` takes a fraction or `-1`, plus `--workers`, `--cache`
  and `--no-amp`.** `--batch 0.70` fills 70% of a card whose memory you have
  not measured; `--workers 32 --cache ram` is what stops a fast GPU idling
  while eight cores decode JPEGs; `--no-amp` is the fix for the NaN-loss and
  zero-mAP symptom AMP produces on some ROCm builds.

### Fixed

- Release archives now include `QUICKSTART.md` and `requirements-detect.txt`
  (the packager ships an explicit list, and neither was on it -- the quick
  start's own install step would have failed from an archive).
- `requirements-detect.txt` allowed `ultralytics>=8.3`, which cannot load
  YOLO26 weights -- every model this project has released. It needs 8.4.

- **`autolabel_objects.py` deleted every fuel box it found.** Without
  `--append` it rewrote each label file from scratch, and the documented
  workflow runs it *after* `autolabel_fuel.py` — so one run threw away ~200
  fuel proposals per frame across the whole dataset, with no error and no
  symptom until a model trained on it stopped seeing fuel. Appending is now
  the default; `--overwrite` is the explicit way to start over, and `--append`
  is accepted and ignored so existing commands keep working.
- **Hubs no longer need the videos.** The script bailed on any match whose
  cleaned video was missing, but the video is only needed for the robots'
  temporal-median background — hub boxes are replayed from geometry in the
  database and need nothing. Since a frame bundle ships without videos (~330 MB
  a match), that meant every machine except the one that harvested got no hub
  labels either. It now writes what it can and says which part it skipped.
- **`autolabel_fuel.py` proposed a fraction of the fuel on a busy frame.**
  Fuel rests in loose groups, a group is one connected component, and the fill
  gate that correctly rejects a heap rejected every pair and triple with it —
  92 proposals on a frame holding several hundred balls. Components that fail
  the gate are now split with a distance transform and a watershed, one seed
  per ball centre, and each piece is sized against the median ball in its own
  band of the frame before being kept. On a synthetic frame of 18 balls in
  groups plus a 20-ball heap, proposals went from 12 (three of them merged
  pairs) to 18.
  Cutting only fixed the pairs, though, and the clusters hold more fuel than
  the open floor does. A ball inside a cluster is surrounded by yellow, so the
  mask has no seam to cut there at all; what is still visible is its shading.
  Those are now found with a Hough circle search over the gradient, radius
  range pinned to the band's measured ball — 30 of 30 on a hex-packed cluster
  where the watershed found none, and no box pair overlapping by more than 0.5
  IoU. `--no-hough` turns it off.
- **Fuel a robot was carrying was never labelled, and the floor's reflections
  were.** Two failures of the same colour gate, in opposite directions.
  A ball in a hopper reads V=103 at its highlight and V=58 at its rim against
  a floor of 90, so it survived as a 7x7 dot below `--min-area` and a robot
  holding four contributed nothing — the balls that decide whether a score is
  attributed at all. The highlight is now used as a seed and grown to the
  band's ball size, kept only where a much looser gate agrees it is yellow;
  Hough is no use there because a hopper bar cuts the gradient (one of four on
  a synthetic hopper, against four of four this way). Meanwhile the glossy
  floor gave most balls a mirrored copy below, which the gate happily labelled
  as fuel, roughly doubling the count where counting matters. A proposal with
  a brighter one above it, within `--reach` ball-heights and aligned to half a
  width, is now dropped as its reflection.
  On a synthetic frame of 13 balls (9 on the floor, 4 behind hopper bars), 9
  reflections and a yellow banner: 13 kept, 9 dropped, nothing on the banner.
  `--no-rescue` and `--no-reflections` turn each off, `--show-dropped` draws
  the removals in green, and `--close`, `--min-cover`, `--v-ratio`, `--reach`
  and `--loose-lo/--loose-hi` tune them. ~70 ms a frame all told.
- **The rescue pass boxed the arena wall.** Its saturation floor was absolute,
  and any floor low enough to admit a ball in shade also admits the tan wall,
  the rail and every washed-out surface behind them. Shading turns out to
  scale a pixel's value and leave its saturation alone — fuel reads S=191 in
  arena light and S=191 in a hopper, while the wall reads S=68 at any
  brightness — so the test is now relative to the balls that frame has already
  found (`--sat-ratio`, default 0.7). On a synthetic frame carrying a tan
  wall, a blown-out highlight and a yellow banner, the wall's box is gone and
  all 13 real balls stay.
  Rescued boxes are also re-centred on the yellow around the highlight rather
  than on the highlight itself, which sits off-centre toward the light: box
  centres landed within 1–3 px of the true ball centres instead of half off
  the ball.
- **The splitter carved the arena into balls.** The back of the pit, the far
  wall, the sponsor boards and the painted field border are all long yellow
  shapes, and a cluster splitter turned loose on one proposes a ball every few
  pixels across it. Saturation cannot help — a sponsor board measures S=177
  against fuel's S=191 — so two geometric guards were added instead.
  A **field line**, learned per frame: fuel is on the floor, so nothing above
  the floor is fuel. A percentile of the accepted boxes was tried first and
  fails for the obvious reason, that the false positives are themselves above
  the field and drag the line up with them — on a test frame with 13 round
  yellow objects in the crowd it landed at row 36 of 440 and excluded nothing.
  Density works where position does not: bin the balls by row, keep bins
  holding a quarter of the busiest bin, take the longest unbroken run. That
  put the line at row 166 and excluded all 13. `--roi-top` overrides it.
  A **stripe test** for long yellow shapes about a ball thick, since the
  painted border sits below the line and inside the field. A row of balls is
  scalloped where paint is flat, but balls overlapping by a third are nearly
  as smooth (0.119 against paint's 0.100), so the pixels vote too: a sphere
  casts a seam against its neighbour and paint has no brightness structure at
  all (0.132 against 0.000, or 0.012 for paint with texture on it). Flat by
  both measures, and only then, means paint. Tunable with `--flat-v`.
  On an arena scene carrying a pit band, a painted border, 13 crowd objects
  and a merged row of 27 real balls along a wall: 0, 0, 0 and the row intact.
  Split pieces now face the same saturation test as rescued ones.
- **Three faults the guards above introduced or left behind.** Reflections
  were filtered pass by pass, each list against itself, so a ball that came
  through the colour gate and a reflection that came out of the splitter were
  never compared and the reflection survived; they are now filtered across
  every pass at once. The field line clipped the balls in flight — the fuel
  actually being shot at a hub — because its margin was two diameters of the
  far ball; at four it clears an arc and still excludes the crowd (measured
  both with the crowd well above the flight zone and almost level with it:
  4 of 4 balls in the air kept, 0 of 13 crowd objects, either way), and
  `--roi-margin` tunes it. And the stripe test called anything longer than
  three ball-diameters a candidate, which a clump of four in a row is: raised
  to six, since the pit band and the painted border run tens of diameters and
  nothing shorter is worth the risk.
- **The reflection filter was eating fuel out of the piles.** Measured on a
  real broadcast frame rather than a synthetic one: of 23 boxes it dropped, 14
  kept 85% or more of their source's saturation. Those were not reflections —
  they were balls lower in a pile, shaded by the ones above, and shading
  scales value while leaving saturation alone (the same fact that finds a ball
  inside a hopper). A reflection does lose saturation, being the ball's colour
  mixed with the grey it reflects in, so both conditions are now required
  before anything is dropped. On that frame: 23 drops down to 9 and the total
  up from 207 to 221. On a synthetic with reflections modelled physically —
  blended toward the floor colour rather than merely dimmed — all 11 are still
  caught, and a ball stacked under another ball survives. `--sat-keep` tunes it.
- **A frame that cannot be labelled is now removed instead of mislabelled.**
  Checked against four other broadcasts, the labeller fell apart on all of
  them, and the cause is one number: every measurement here is scaled off the
  ISOLATED balls, and those matches keep their fuel in one corral. The frame
  this was tuned on shows 85 isolated balls; the four that failed show 6, 16,
  20 and 22, so ball size, ball saturation and the field line are all being
  read off noise. The decisive measure is what it leaves behind — of the
  yellow pixels on the frame, how many ended up inside a proposal: 59% on the
  working frame against 13%, 38%, 35% and 50%, which is a corral of several
  hundred balls in plain sight with nothing drawn on it.
  Writing those labels is worse than writing none, because Ultralytics reads
  a missing box as "nothing here" — so the frame would teach the detector
  that a mass of fuel is background, which is the thing it most needs to find.
  Frames below `--min-singles` or `--min-coverage` are now moved to
  `dataset/skipped/` and reported per match, with `--no-quarantine` to label
  them anyway. Re-running `prepare_dataset.py` restores anything moved.
- **`CLAUDE.md`** gives a Claude Code session on another machine the context
  this work was done in: the commands, the harvest and training flows, the
  stdlib-only rule for the API path, what the fuel labeller can and cannot
  label and why, and the pitfalls of the MI300X droplet — with a dated
  "current work" section meant to be deleted once it goes stale.
- **A robot driving through fuel is no longer a 0-for-41 shooter.** The first
  scouting model's run on 2026nhdur qm7 credited the robot plowing the centre
  pile with 42 shots and 41 misses in 15 s of auto. The balls it passed sat
  still while it drove away -- which "gets clear of the robot" as surely as a
  launch -- and the tracker gives pile balls new ids constantly, so each one
  "started at the robot". A shot must now move itself: a robot-width of its
  own travel inside 0.3 s (continuations of a broken flight measured from
  where it broke). Rejected balls are counted under `not_launched`.
- **Robots are numbered #1, #2, #3 in `run.py shots`, not by tracker id.**
  The tracker numbers every object it follows, fuel included, so robot 1307
  was labelled "R1283" in the first annotated video and read as a misread
  team number. Nothing reads bumpers; `--teams 1=1307` maps the new numbers.
  A robot the tracker loses and finds again under a new id keeps its number
  when it reappears within 1.5 robot-widths of where it was, same alliance,
  within 3 s: qm7 grew robots 16099, 15856, 16580 and 21187 mid-match, each
  splitting a real robot's tally.
- **Shots: flights capped at 2.5 s, hopper shots kept, launches seen from
  above.** From the annotated qm7 video (frames checked by eye):
  - Stitching broken flights walked from ball to ball through the fuel
    piles for 4-8 s -- a seven-piece "flight" from 22.2 s to 30.0 s -- ending
    as misses or as makes credited to the wrong robot. A shot not in a hub
    2.5 s after launch is now a miss, final, and no later piece joins it.
  - The not-launched rule timed its 0.3 s from a ball's first sighting, so a
    ball tracked in a hopper and shot seconds later was thrown away. It is a
    sliding window now.
  - 13 of 19 makes were unattributed. A launch zone reaching 0.8 of a
    robot's height above it was tried and attributed none of them (still 3
    blue, 10 red); from a broadcast camera "above a robot" is the floor
    behind it. It is off (`launch_up` 0).
  - A launch must be fast both on its own and relative to its robot. The
    run after the sliding window credited 1058 with 11 shots and 11 misses
    while it plowed the pile (a robot at speed pushes balls a robot-width in
    0.3 s). Relative travel alone then made it far worse -- 76 shots, 76
    misses -- because a still ball a fast robot drives away from moves fast
    relative to it. Both tests are required now.
  Robots split across several numbers (1058 as #5, #8, #11) are left split
  on purpose: `--teams 5=1058,8=1058,11=1058` folds them; merging by guess
  could mix two robots' shots.
- **Counting windows keep their length in time on 60 fps video.** Both
  counters measure "missing for N frames", "held for N frames" in frames,
  chosen on 30 fps footage; the harvested broadcasts are 60 fps, so every
  window was half as long -- flights ended and robots were forgotten twice as
  fast as intended. `run.py shots` and `run.py count` now scale them by the
  source's frame rate (read from the file, or `--expect-fps` for count).
- **`run.py shots` crashed at the very end of a run** printing its report,
  after the not-launched rule added an ignore reason the report had no line
  for -- losing every result. It names that reason now and cannot crash on
  an unnamed one.
- **`run.py shots` keeps one box per robot.** The same model sometimes boxes a
  robot twice (0.58 and 0.37 on one robot of the held-out match); every box
  became its own track, splitting that robot's shots. The less confident of
  two boxes that overlap that much is dropped each frame.
- **`train.py` no longer dies before epoch 1 on the newer MI300X image.**
  Every dataloader worker failed with "no response from torch_shm_manager":
  the helper program that file_system sharing starts could not load
  librocm-openblas.so.0, which the torch 2.12+rocm7.14 image keeps in the
  venv at `_rocm_sdk_core/lib/host-math/lib` where only Python's torch finds
  it; with that found it then missed libamdhip64.so.7 and
  librocprofiler-sdk.so.1. `train.py` now puts every library folder of the
  ROCm package and torch/lib on `LD_LIBRARY_PATH` for the helper.
- **Robots in piles of fuel are found: fuel is greyed out before YOLOE
  looks.** In the first label preview robot 1058, in the middle of the
  central pile, had no box and fuel boxes all over it -- the model would
  have learned that a robot in fuel is background, exactly when `run.py
  shots` needs it. With fuel-coloured pixels (autolabel_fuel.py's gate) set
  to grey first, four real frames with 16 robots marked went from 10 found
  to 14 at `--conf 0.1`; 1058 from nothing to 0.40, a red robot from 0.41
  to 0.91. The two clean frames alone: 6 of 9 to 8 of 9. Alliance is still
  read from the real pixels. Extra prompts ("robot covered in yellow
  balls") were tried and added nothing. `--no-grey-fuel` turns it off.
- **The red hub is no longer painted out as a robot.** "Too big" needed two
  other robots to compare with; a frame with one let the hub's top (3.4x the
  robot beside it) through as a robot of unknown alliance. One is enough
  now. A box inside a more confident robot box (fuel greying made YOLOE box
  parts of robots too) is dropped as a duplicate.
- **A robot of unknown alliance is painted out instead of costing its
  frame.** On the 906 nhdur frames, 176 were dropped for one robot whose
  bumper colour could not be read -- a fifth of the dataset, for one box
  each. That robot is now filled with flat grey (Ultralytics' pad colour),
  fuel boxes centred in it are removed, and the frame is kept: neither a
  guessed class nor a robot left as "floor". Originals go to
  `dataset/skipped/robots/original/`; the painted copy is written as a new
  file, so a symlinked dataset never paints the harvested frame. `--restore`
  now undoes everything: moved frames, painted frames, and robot lines in
  every label file. Checked end to end on real frames, symlink included:
  after `--restore` every image hashes as before. Painting the
  lower-confidence boxes as well, for robots YOLOE misses, was measured and
  not adopted: it covered 3 of 10 misses and 30% of one frame.
- **`autolabel_robots.py --dry-run` reported "(0 robot boxes)"** and "would
  labelled": it never counted boxes in a dry run. It does now. Labels written
  by a real run were never affected. First full dry run on the 906 nhdur
  frames: 591 kept at `--min-robots 2`; 176 dropped for a robot of unknown
  alliance, 120 for too few robots, 19 for more than three of one alliance.
- **Robot labelling: the blue ladder no longer passes as a robot, and the
  gate keeps frames.** On the first five real previews the "too big" rule
  compared each box with a median that included the two hubs (~56,000 and
  ~82,000 px^2), so the ladder (34,800 px^2, beside robots of 6,100-9,500)
  came out 3.7x and passed -- in the only frame the gate kept. The median is
  now over boxes not already rejected, and the limit is 3x (real robots
  differed by at most ~1.7x near to far). The same previews put YOLOE's
  recall at ~16 of 27 robots; no frame had every robot boxed, so
  `--min-robots` 4 kept nothing usable. The default is now 2, which keeps 3 of
  those 5 frames; the missed robots in them are left for the bootstrap.
- **Robot labels without hand-labelling: `train/autolabel_robots.py`.** Asks
  an open-vocabulary detector (YOLOE, `yoloe-26s-seg.pt`, prompted "robotic
  vehicle" / "robot" / "wheeled robot") for robots it was never trained on,
  over the whole frame and three 2x tiles. On two real frames with every robot
  hand-marked it found 4 of 5 and 2 of 5 at the default `--conf 0.1` (the
  corner robot scored 0.09), and nothing else once three rules
  are applied: too tall (the hubs), a box holding two others (one box around
  both red robots), and ~10x the area of the frame's other boxes (the red
  alliance wall). Alliance is a hue vote in the box's bumper band; the navy
  bumper of robot 69 has median saturation 8-93 and is left unknown rather
  than guessed. Recall is the weakness, so a frame is labelled only if every
  robot found has a readable alliance and at least `--min-robots` were
  found; otherwise it is moved to `dataset/skipped/robots/` with its label,
  and `--restore` brings them back. Both test frames were rejected, so how
  many of a real dataset survive is unmeasured: `--dry-run` reports it per
  `--min-robots` value before anything is written. Missed robots in frames
  that pass are still unlabelled; the bootstrap in `train/README.md` is the
  next round.
- **Robot proposals stop boxing the crowd and start finding the near robots.**
  The first real preview boxed three people in the stands and one robot of
  six. People in alliance colours who move are exactly what the robot test
  looks for, and nothing told it where the field was; meanwhile `--roi-bottom`
  defaulted to 0.80 and threw away every robot in the near fifth of the frame,
  which on that camera was most of them. Robots now have to sit below the
  field line learned from the fuel on the same frame, the line autolabel_fuel
  already uses, so it follows whatever camera is in use instead of assuming
  one. `--roi-bottom` is now 0.98, since static things by the near rail
  already fail the motion test. On a synthetic frame built to match that
  preview, the old settings gave 2 boxes in the stands and 0 near robots; the
  new ones give 0 and 2, plus the far robot. `--no-field-line` turns it off.
- **Robots can be auto-labelled without the match videos, and their boxes
  cover the robot, not just its bumper.** `autolabel_objects.py` finds robots
  as bumper colour that has moved against the match's empty-field background,
  and took that background from the cleaned match video — which a frame bundle
  does not include, so robots could not be proposed off the harvesting
  machine at all. The camera is fixed for a match, so the median of the
  match's own exported frames is that background: a robot is in any one place
  for a small share of the match and drops out. Those frames are also the
  ones the dataset images were copied from, so the background lines up pixel
  for pixel. Separately, the box it drew was the bumper band only, and a shot
  leaves from the top of the robot — so no launch would have started inside
  any robot's box and `run.py shots` could never have attributed one. Boxes
  now grow from the bumper up through the moving pixels connected to it,
  using a more sensitive motion mask than the bumper gate, since a dark frame
  over grey carpet differs from the empty field by only ~25 levels, under the
  bumper gate's 40. On a synthetic match: background clean, each box from
  bumper to shooter, two adjacent robots kept apart, a static blue ramp
  ignored. Not yet checked on real frames — `--preview` is the check.
- **`run.py shots` — per-robot scouting from a model over video.** Runs the
  detector with tracking over a match video or a camera, feeds robots and
  balls into `ShotCounter`, and prints each shot as it is decided and a
  per-robot summary: shots, made, missed, accuracy. `--teams 3=254,7=254`
  names robot tracks (several ids per team is normal — a tracker that loses a
  robot gives it a new id), `--annotate` writes a video with robot ids, balls,
  hubs and each outcome drawn on, which is how ids become team numbers and how
  shots get checked by eye, and `--out` writes it all as JSON. Shot times are
  match time when the frame rate is known, not processing time, and it warns
  below 15 fps, where a ball crosses the field between frames — exported 3 fps
  frames are no use for this. It refuses a model with no robot classes rather
  than producing a scouting sheet with no misses and nobody's makes. Tested end
  to end against a stand-in model, so everything but the network runs in CI.
- **`count.py` counts with a fuel-only model when you give it the hubs.** It
  refused any model without hub classes, even with both hub boxes passed in —
  and the first trained model is fuel-only. Hub classes are now needed only
  when the hubs must be learned. On a fixed camera, drawing two boxes during
  setup is easier and more exact than detecting them every frame anyway. The
  refusal still names every missing class at once, and now also says how to
  pass the boxes.
- **`tbavid/shooting.py` — who shot, and the misses.** `count.py` knows how
  many balls went into each hub; scouting needs whose they were, and the
  misses, which `count.py` cannot see at all since a ball that never reaches a
  hub is not an event there. `ShotCounter` follows each ball from the robot
  that launched it to wherever it ended and files **made**, **missed** or
  **wrong hub** against that robot track. A shot is a ball that starts at a
  robot *and gets clear of it* — measured from the robot's box as it is now,
  so a ball riding in a hopper is not a shot every time its robot drives. A
  make with no visible shooter is kept as *unattributed*, so per-hub totals
  always equal `BallCounter`'s for the same balls; a test holds them to that.
  Flights the tracker breaks are stitched back together by where the ball was
  heading. The first detector's per-frame recall of 0.62 makes those routine,
  and each would otherwise read as a miss by the shooter plus a make by
  nobody. That includes a flight broken before the ball cleared its robot,
  which the first version called "carried". `by_team()` folds robot tracks
  into teams once something assigns them. Logic only: nothing trains robot
  detection yet, and it needs native-frame-rate video, not 3 fps frames.
- **Training no longer dies on a dense batch after a few good epochs.** On
  the MI300X droplet a clean run (mAP50 0.49 → 0.54 over epochs 2–4) died in
  epoch 5 with `received 0 items of ancdata` and `Pin memory thread exited
  unexpectedly` — nothing about files, though that is the cause. PyTorch's
  dataloader hands tensors between processes as open file descriptors; a
  batch of 16 frames carrying 6,454 fuel labels, times 16 workers, ran the
  container past its open-file limit. `train.py` now uses the `file_system`
  sharing strategy on Linux, which goes through `/dev/shm` instead.
- **The MI300X recipe drops `--p2` and trains at 960.** `yolo26s --p2
  --imgsz 1280` died in epoch 1 inside the loss's `TaskAlignedAssigner`: one
  16.6 GiB allocation refused with 177 GiB free. Cutting the batch from 16 to
  4 was the first fix and did nothing — the request was the same 16.6 GiB —
  so batch is not what sizes it. Every crash had P2 at 1280, and the warmup
  at 960 without P2 ran clean, so the recipe uses that until the two are
  separated.
- **A dataset built on one machine now trains on another.** `prepare_dataset.py`
  and `subset_classes.py` write an absolute `path:` into the yaml — they have
  to, since Ultralytics resolves a relative one against its own datasets
  directory rather than the yaml's — so a set built on a Mac said
  `/Users/.../dataset-fuel`, and the first run on an AMD droplet stopped with
  "images not found" after the GPU was already up. The Colab and Kaggle
  notebooks each rewrote the line by hand; nothing else did. `train.py` now
  repoints `path:` at the yaml's own directory whenever the recorded one is
  missing and the images are beside the yaml, and prints that it did.
- **`deploy/AMD_DEVCLOUD.md` describes the machine you actually get.** On the
  PyTorch 1-Click image torch lives in a Docker container named `rocm`, not on
  the host, so the guide's `python3`/`pip` steps failed as written on the
  host. It now separates host commands (`docker`, `scp`, `rocm-smi`, `tmux`)
  from container ones (`python3`, `pip`, `train.py`), copies the data in with
  `docker cp`, and warns that files written in the container die with it.
  Batch and workers drop to 16 each: the 1x plan has 20 vCPU, and a first
  harvest of ~700 training frames wants optimizer steps more than memory.
- **`train/drop_offcamera.py` removes frames that are not the main camera.**
  A crowd shot that survived the harvester's shot cut reached the labeller,
  which proposed 49 boxes on spectators' yellow shirts — and every quality
  check passed it, because they ask whether the visible yellow ended up inside
  a box and on that frame it did. Nothing in a colour gate can know the yellow
  is a T-shirt, and nothing in the per-frame checks separated it: the dark
  arena's real field frame scores lower on floor area and on ball-size spread
  than the crowd shot does.
  What separates them is that the camera does not move within a match, so the
  match's own median frame is what the field looks like and a crowd shot is a
  different picture rather than a bad one. Each frame is scored by its mean
  absolute difference from that median on a 64×24 thumbnail, as a modified
  z-score against the match's own spread (median and MAD, so a handful of
  crowd shots cannot raise their own bar). On a synthetic match of 40 field
  frames and 3 crowd shots: field frames peak at z=1.1, crowd shots score 238,
  and the default threshold of 8 sits between them with a 216× margin.
  Run it before `autolabel_fuel.py`; `--dry-run` lists without moving.
- **`prepare_dataset.py --matches`** builds a dataset from named matches or a
  whole event. The fuel labeller is a colour heuristic and does not survive
  every broadcast: a wide shot whose fuel sits in one corral gives it 9–22
  isolated balls to measure from against 85 on a close one, and ball size,
  ball saturation and the field region are all derived from those. It is
  better to train on the broadcasts it handles than on labels it got wrong
  everywhere else.
- **The preview names which pass proposed each box** — red through the gate,
  orange split out of a cluster, cyan rescued from shade, green dropped as a
  reflection, with the learned field line drawn in white. `train/diagnose_labels.py`
  reports the same counts across a sweep of a setting, which is how you tell a
  fix that did not work from a fix that never ran.
  The band-local size also catches the merge the fill gate never could: two
  balls side by side fill their bounding box to 0.82 and passed as a single
  ball of twice the local size. New knobs: `--no-split`, `--merge-factor`,
  `--max-split`, `--hsv-lo/--hsv-hi` for broadcasts whose yellow sits outside
  the default gate, and `--preview-frame` to preview a specific frame instead
  of the middle one. The preview now separates gated boxes (red) from
  recovered ones (orange) and counts the heaps it skipped.

### Changed

- `CLAUDE.md` moved to `.claude/CLAUDE.md`, out of the repo's front page;
  Claude Code reads project instructions from either place.

### Removed

- `deploy/make_release.sh`: its header said three documents call it; none
  do. Releases come from pushing a tag (`.github/workflows/release.yml`), and
  `python deploy/package.py release <version>` still builds one locally.
- `deploy/amd_watch.sh`: marked untested and never run; the MI300X guide
  copies weights out with `docker cp`, which is what was actually used.

## v0.2.0 — 2026-09-16

Harvesting is now restricted to official competition footage, and an existing
harvest can be checked against that same standard.

### Added

- **Competitive-only harvesting.** Only official competition events are walked:
  regional, district, district championship, championship division,
  championship final and Festival of Champions. Offseason, preseason and
  unlabeled events are never picked. Within an event only real match play
  counts (`qm`, `ef`, `qf`, `sf`, `f`), so practice matches are skipped even
  when TBA has a video for them.
- **`run.py audit`** reports which videos in an existing harvest are not
  competition footage, and how many frames each one contributes. It reports
  only and deletes nothing. A season TBA does not answer for is reported as
  unknown rather than condemned.
- **`DATA.md` and `docs/provenance/`** record what is in the current dataset,
  where the frame bundle lives, and the data-quality problems found in it.

### Changed

- Event listing uses `/events/{year}/simple` instead of `/events/{year}/keys`,
  because a bare event key does not carry `event_type`. Still one request per
  season, still cached and revalidated with `If-None-Match`.

### Added (release process)

- **Releases are cut by pushing a tag.** `.github/workflows/release.yml` runs
  the suite, scans the tree for a committed key, builds all three archives with
  `deploy/package.py`, extracts the notes from this file, proves the built
  archive runs by extracting it and running the suite from inside it, then
  creates the GitHub release with the archives and `SHA256SUMS` attached.
  Nothing is built or uploaded by hand, and the checksums in the notes are
  always the ones from the artifacts actually attached.
- `workflow_dispatch` builds and verifies a version without publishing, so a
  release can be rehearsed before its tag exists.
- `python3 deploy/package.py notes vX.Y.Z` prints a release's section from this
  file.

### Fixed

- **`run.py pull` crashed on every video it successfully downloaded.**
  `_process()` read `shard` and `shards`, which are parameters of `fetch()`,
  not of it — `NameError: name 'shards' is not defined`. The conditional
  tested `shards > 1` first, so it raised on single-worker runs too, and it
  raised at the very end: after the download, shot analysis, crop, render and
  scoreboard OCR had all completed for that video. Present since the initial
  commit and in the `Beta` release. The shard is now passed in, and
  `reprocess` preserves the one already recorded instead of erasing it.
- A test walks the symbol table of every function in `tbavid/`, `run.py` and
  `serve.py` and fails on any global that no module global defines. The crash
  above needed a real video to reach, which the suite deliberately has none of.
- `deploy/make_code_archive.sh` built its file list from a hardcoded string and
  aborted once `QUICKSTART_DEBIAN.txt` was deleted from the repo — leaving a
  half-written archive behind that had never reached the keyless check. The
  list is now built from what is on disk, and a failed run removes its own
  partial output.
- `deploy/make_release.sh` no longer stages the deleted `QUICKSTART_DEBIAN.txt`,
  and now includes `DATA.md`, `CHANGELOG.md`, `LICENSE` and `docs/`.
- Packaging works on Windows and macOS, not just Linux. Both `make_*.sh`
  scripts are now thin wrappers around `deploy/package.py`, which uses
  `zipfile`, `tarfile`, `lzma` and `hashlib` from the standard library instead
  of `zip`, `tar`, `xz` and `sha256sum` — macOS has no `sha256sum` and Git Bash
  usually has no `zip`. `release` now also writes `dist/SHA256SUMS`.

### Escape hatch

- `--include-noncompetitive` on `pull`/`fetch`, or `"competitive_only": false`
  in `config.json`, restores the previous behaviour. The flag wins over config.

### Known issues

- The frame bundle harvested before this release contains 926 frames (12.0%)
  from `2026kylou`, an offseason event. `run.py audit` finds them; nothing
  deletes them for you.
- `2026gal_qm62` carries a bad scoreboard read (blue final 726 against a 19–285
  range elsewhere) and is not flagged as an error.
- A one-alliance scoreboard read is still recorded as a good read: the
  `scoreboard_ok` test in `tbavid/db.py` is satisfied by a series containing
  only one alliance. `2026mnmi2_qm11` and `2026mnmi_qm39` are affected.

## v0.1.0-beta — 2026-09-15

Released as tag `Beta`. First working end-to-end pipeline: pick unseen match
videos from TBA, strip them to main-camera footage, crop the scoreboard off,
export sampled frames, and read the fuel counters before discarding the banner.
Includes the scouting database, the read-only JSON API, and deployment notes
for Debian, Windows, Colab and Kaggle.
