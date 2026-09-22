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

## Checklist

```bash
python3 tests/test_pipeline.py                    # 187 checks, no network
python3 run.py status                             # the harvest
python3 run.py formats                            # layout profiles
curl -s localhost:8781/health                     # the API host
python3 run.py detect --weights <pt> --limit 1    # the model, on one match
```

In the scouting app: `python3 server/tests_api.py`, then **TEST KEYS** on the
admin panel, which is the only thing that can tell a dead harvest address from
one with no footage behind it.
