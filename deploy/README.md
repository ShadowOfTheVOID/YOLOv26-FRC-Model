# Where to run each piece

Three workloads with different needs. Splitting them costs nothing because
they already talk through files and SQLite.

| workload | bound by | run it on | why not elsewhere |
| --- | --- | --- | --- |
| harvest (`run.py pull`) | network + CPU + **disk** | your Mac mini | YouTube blocks datacenter IPs |
| training | **GPU** | Colab or a rented GPU | 73 h on an 8 GB M2 |
| scouting API (`run.py serve`) | nothing | Replit / Render / Fly, or local | — |

## Harvesting: keep it local

Three separate reasons, and only the last is technical.

**Terms.** YouTube's ToS prohibits downloading video except through features
YouTube provides, and yt-dlp is not one. That applies wherever you run it,
including your own machine — it is not something a platform choice fixes.
Pulling TBA-linked match video for vision work is common in FRC and the videos
are posted by FIRST and event organizers, but the call is yours to make
knowingly.

**Whose infrastructure.** Kaggle is Google, and so is YouTube. Running a
downloader on Google infrastructure against Google's own service is a good way
to get an account flagged, independently of any terms question.

**Bot detection.** The one that does **not** move to the cloud. yt-dlp from Colab, Replit or
any datacenter IP hits *"Sign in to confirm you're not a bot"* on a large share
of videos, because that is exactly what YouTube's bot detection is looking for.
Your home connection is the asset here.

It is also the cheap half: ~4 minutes of CPU per match, no GPU.

Disk is the real constraint on a Mac mini. Measured at **333 MB/match** with
everything kept; **100 MB/match** with `keep_raw` and `keep_clean` off, which
puts 100 matches in ~10 GB. Or send the bulk elsewhere:

```bash
export TBAVID_DATA=/Volumes/YourSSD/tbavid
```

## Running it on a spare Linux box

An old laptop on Debian makes a decent dedicated harvester — home IP, runs
24/7, frees your main machine. See **[DEBIAN.md](DEBIAN.md)**. No GUI or IDE
required; it is all command line.

## Training: free, and local is viable

**Cut the dataset before renting anything.** 3 fps sampling gives ~480 frames
per match that are mostly near-duplicates of their neighbours; a detector wants
variety, not density. `--per-match 80 --prefer-scoring` keeps 80 frames spread
across each match, two thirds of them drawn from moments where the scoreboard
says fuel landed just after:

```bash
.venv-train/bin/python train/prepare_dataset.py --clean --per-match 80 --prefer-scoring
.venv-train/bin/python train/autolabel_fuel.py
```

6 matches: 2886 frames -> 480. Measured on the 8 GB M2 at that size:

| config | per epoch | 60 epochs | 100 epochs |
| --- | --- | --- | --- |
| `yolo26n` 960 / batch 4 | ~3.3 min | ~3.3 h | ~5.5 h |
| `yolo26s --p2` 960 / batch 2 | ~10 min | ~10 h | ~17 h |

That is an overnight run on hardware you already own, for nothing. Start with
`yolo26n` — if it cannot learn fuel, a bigger model will not rescue bad labels,
and you will find that out in one night instead of three.

Memory is *not* the blocker on 8 GB: `yolo26s-p2` peaks at 4.4 GB. Wall-clock
is, and shrinking the dataset is the lever that costs no money.

### Kaggle needs a phone-verified account

Unverified Kaggle accounts have Internet disabled and cannot enable it, which
breaks `pip install` and every network call. Verification is free. Without it,
use Colab for training and your own machine for harvesting.

### Free GPU, when you outgrow that

- **Colab free** — T4, ~12 h sessions, GPU not guaranteed and throttled after
  heavy use. Use [`colab_train.ipynb`](colab_train.ipynb).
- **Kaggle Notebooks** — a weekly GPU quota (generous, check the current
  figure) and persistent Datasets to store `dataset.tgz` between sessions,
  which Colab free does not give you without Drive.

Both want a **portable archive** — `prepare_dataset.py` symlinks by default and
a plain `tar` of symlinks arrives empty:

```bash
.venv-train/bin/python train/prepare_dataset.py --copy --clean --per-match 80 --prefer-scoring
tar -czf dataset.tgz dataset/          # ~100 MB at 80 frames/match
```

Checkpoint to Drive during the run: free sessions disconnect and wipe local
disk. `patience=30` usually stops well before 100 epochs anyway.

## Scouting API: Replit works, training there does not

Replit has no GPU worth training on, but it is a reasonable host for the API,
which is why that was written against the standard library only — no build
step, no wheels to compile.

```bash
python3 run.py serve            # honours $PORT and $HOST
```

Set `HOST=0.0.0.0`; most hosts inject `$PORT` themselves. Upload
`data/scouting.db` alongside the code — the API opens it **read-only**, so the
deployed copy is a snapshot. Refresh it by rebuilding locally
(`run.py db sync`) and re-uploading; nothing on the host writes to it.

Same applies to Render, Fly.io, or a Raspberry Pi on your bench. The database
is a single file and the server is one stdlib process.

### If the scouting app needs frame images too
The API serves JSON only. Frames are ~100 MB/match, so host them as static
files next to the API or on object storage and reference them by the `file`
column in `frames`.
