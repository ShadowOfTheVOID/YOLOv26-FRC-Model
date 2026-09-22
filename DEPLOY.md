# Deploying all of this

Everything here is built and tested except one thing: **the `.pt`**. This page
is the order to do it in, what runs where, and what each box actually needs
installed — because the three roles have very different requirements and
installing everything everywhere is how a scouting laptop ends up with four
gigabytes of CUDA on it.

Nothing has been removed to make room for any of this. `pull`, `stream`,
`export`, `review`, `prune` and the whole dataset path are intact and are still
the point — the detector is trained on what they produce.

## Three roles, three dependency sets

| role | what it does | needs |
| --- | --- | --- |
| **harvest box** | `pull`, `stream`, `live`, `export`, `db` | python3, ffmpeg, yt-dlp, tesseract, `requirements.txt` |
| **API host** | `serve.py` under systemd | python3 and nothing else |
| **detect box** | `detect` | the harvest set plus `requirements-detect.txt` (torch) |

They can all be one machine. They are listed apart because the API host must
stay trivial: `serve.py` and everything under `tbavid/api.py` import only the
standard library, which is what lets it run on the cheapest box you have and
survive a `pip` that would break anything else.

The scouting app is its own repository and its own laptop; see its `setup.md`.

## 1. Harvest (the scraper — still here, still the foundation)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'TBA_AUTH_KEY=...' > .env

.venv/bin/python run.py pull -n 40                       # per-match uploads
.venv/bin/python run.py stream --url <s> --event 2026casnf   # a whole event day
.venv/bin/python run.py status
```

`./run.py formats --calibrate --event <key>` before a long run on a feed nobody
has measured — see the README. `audit` says whether what you pulled is
competition footage.

## 2. Build and serve the database

```bash
.venv/bin/python run.py db sync        # from the manifest, plus TBA rosters
.venv/bin/python run.py db export      # a copy a read-only host can serve
```

Then the API, once, on the host:

```bash
sudo cp deploy/frc-harvest.service /etc/systemd/system/
sudo systemctl enable --now frc-harvest
```

Full steps, and why `cp` is not good enough for that database, in
[deploy/HOSTING.md](deploy/HOSTING.md). Point the scouting app's **VIDEO
HARVEST** address at it and the numbers appear in its dashboard.

At this point everything works and nothing has seen a neural network.

## 3. Label, then train — the one part left to you

```bash
.venv-train/bin/python train/autolabel_fuel.py --append       # proposals
.venv-train/bin/python train/autolabel_objects.py --append    # proposals
# correct a subset by hand -- see train/README.md, "What still needs a human"
.venv-train/bin/python train/prepare_dataset.py
.venv-train/bin/python train/train.py --p2
```

Out comes `runs/<name>/weights/best.pt`.

## 4. Run it

```bash
python3 -m venv .venv-detect
.venv-detect/bin/pip install -r requirements-detect.txt
.venv-detect/bin/python run.py detect --weights runs/<name>/weights/best.pt
```

That fills the `detections` table — boxes, classes, confidences and tracks, one
row per detected object per frame, each tagged with the weights that produced
it so two models' opinions can never be confused for one.

### What the `.pt` does not finish

**It does not give you team numbers**, and this is worth being straight about
before anybody promises the strategy team per-robot data.

`detect --assign` runs the identity layer, and the identity layer needs a
`scorer` — a function that looks at a robot's bumper crop and scores it against
the six teams TBA says are on the field. That function does not exist, and
`identify.py` explains why it is hard rather than merely unwritten: measured on
a real broadcast frame, a bumper number is about **5 px tall** and
motion-blurred, and tesseract returns the empty string at every
page-segmentation mode tried. That is missing detail, not a tuning problem.

With no scorer, `assign_tracks` records nothing and says so. It does not pick
whichever team sorts first, because a wrong team number in a scouting database
is worse than a missing one — the same rule everything else here follows.

So after step 4 you have: per-alliance scoring (already had it, from the
scoreboard), and now robot and hub positions and tracks. You do not have "team
254 scored 40 of that". Closing that needs a scorer, and the honest routes are
a higher-resolution source than a broadcast, or start-of-match station
positions from TBA's roster — neither of which is written.

## 5. Being the scoreboard at a scrimmage

Everything above reads fuel off the broadcast's burned-in counter, which is the
field's own arithmetic and beats any vision system. **At a scrimmage there is
no FMS**, so nothing else is counting. This is not a cross-check against a real
score — it *is* the score, and it has to behave like one.

```bash
.venv-detect/bin/python run.py count --weights runs/<name>/weights/best.pt \
    --source 0 --hub-blue 300,100,100,100 --hub-red 700,100,100,100 \
    --scoreboard --auto-points 4 --teleop-points 2
```

**There is no display here**, deliberately. Whatever is showing the score at
the field already exists; this is the thing that knows what the score *is*, and
it hands it over two ways:

```bash
curl -s localhost:8780/state          # the whole match, as JSON
run.py count ... --scoreboard --feed  # one JSON line per change, on stdout
```

`--port 0` turns the HTTP side off entirely if the pipe is all you want.

The clock and a referee's correction are the other half of the interface:

```bash
curl -X POST localhost:8780/start     # /stop  /reset
curl -X POST localhost:8780/adjust -d '{"alliance":"red","delta":1}'
```

**Two things a counter needs before it is the scoring authority**, each because
of something counting alone cannot do:

- **A match clock.** Fuel gets thrown around between matches and robots get
  tested on the field. Nothing scores until the match is started and the
  detector stops at the buzzer; a counter running continuously would add all of
  it to the score.
- **A referee's correction.** The one that matters. The counter is careful and
  still fallible — it cannot see a ball occluded for its whole flight.
  Everywhere else this repo answers uncertainty by recording nothing, which is
  right for a scouting number that can simply be absent and useless when a
  match needs a final score in thirty seconds. So the person at the table has
  the last word, ±1 at a time, and it keeps working after the buzzer because
  that is when most corrections happen. The record keeps **both** halves:
  `detected` and `adjusted` are never folded together, because "the camera
  missed two" and "the camera saw two that never happened" are different facts
  about your setup and both are worth knowing afterwards. `/state` carries
  `detected` and `adjusted` beside the total so whatever draws the score can
  show that distinction if it wants to.

**Points are yours to set.** `--auto-points` / `--teleop-points` default to 1,
so with nothing configured the score IS the ball count. `db.py` already
refuses to convert fuel to points — "the 2026 fuel-to-points rule is not pinned
down here" — and a scrimmage runs whatever rules you chose. Nothing here
invents them.

There is no password on the scoreboard. It is a closed field network for an
afternoon, and a password on the scoring table is one somebody has to type
while a match waits. Do not put it on the open internet.

and as a service, on the box with the camera:

```bash
sudo cp deploy/frc-counter.service /etc/systemd/system/
sudo systemctl edit frc-counter        # weights, camera, hub boxes
sudo systemctl enable --now frc-counter
```

**Give it the hub boxes.** Without them it spends the first 90 frames learning
where the hub is and counts nothing while it does — correct, because with no
hub there is no *in* for a ball to go, but 90 frames of a match you do not get
back. The camera does not move, so reading them off one frame once is right
every time after. `run.py db hub --event <key> --alliance blue --box x,y,w,h`
records them per event and `--event` then picks them up.

### How it decides a ball scored

A ball that goes in stops being visible, so the event is a fuel track
**vanishing inside a hub region**. Three other things look exactly like that,
and refusing them is most of what [`tbavid/count.py`](tbavid/count.py) does:

| looks like a score | why it isn't | rule |
| --- | --- | --- |
| a one-frame false detection | a real ball was there for longer | `--min-frames` |
| a ball a robot drove in front of | it never crossed *into* the hub | entry required (`--allow-inside` to disable) |
| a ball that passed *over* the hub | it comes back the other side | a score is held `--reacquire` frames and a reappearance withdraws it |

That last one is the same shape as `scoreboard.clean_series` wanting two reads
before believing a large jump: commit late, let the next few frames take it
back. It costs a fraction of a second of latency against being confidently
wrong about a score.

Every refusal is counted and reported, because a counter that rejects silently
is one nobody can debug:

```
counted 2 ball(s): blue=2, red=0
  not counted: 1 vanished away from any hub
```

### Honest limits

- **It needs the frame rate.** `detect` deliberately throws fuel track ids away
  because it runs at 3 fps, where a ball moves further between samples than its
  own width. This runs on a live camera at native rate, where tracking a ball
  is the whole method. On a CPU it will fall behind and drop balls, which is
  worse than no count because it looks like one. Check with `--frames` on a
  recording before trusting it at an event.
- **It cannot see a ball it never detects.** One occluded for its whole flight
  is simply missing.
- **Where a real scoreboard exists, read that instead.** At a scrimmage there
  isn't one, which is the whole point — but at a real event the burned-in
  counter is the field's own arithmetic, and `run.py verify` already checks
  this repo's readings against TBA's official totals.

`--match` files the result exactly like any other reading, so it reaches a
scouting app through the same API.

## Checklist

```bash
python3 tests/test_pipeline.py                    # 225 checks, no network
python3 run.py status                             # the harvest
python3 run.py formats                            # layout profiles
curl -s localhost:8781/health                     # the API host
python3 run.py detect --weights <pt> --limit 1    # the model, on one match
```

In the scouting app: `python3 server/tests_api.py`, then **TEST KEYS** on the
admin panel, which is the only thing that can tell a dead harvest address from
one with no footage behind it.
