# Step-by-step: cloud setup

Everything here is free. Nothing needs a paid plan.

## First: are you phone-verified on Kaggle?

Kaggle keeps **Internet off** for unverified accounts, and it cannot be turned
on. That breaks every notebook here — `pip install` fails, the TBA API is
unreachable, yt-dlp cannot run. There is no way around it inside the notebook.

Verification is **free** (Settings -> Phone Verification, takes a minute) and
also unlocks GPU and TPU quota. If you have a phone number, do that first and
the rest of this section applies.

**If you cannot or would rather not verify:**

| | do this |
| --- | --- |
| harvesting | your Mac mini — `./deploy/overnight.sh` |
| training | **Colab** — no verification, Internet works, free T4 |
| the API | anywhere; it needs no internet access of its own |

Colab needs only a Google account. It cannot run with the tab closed, which is
the reason to prefer Kaggle, but a working notebook you have to babysit beats a
faster one you cannot start.

## Kaggle or Colab?

*(Assuming you are verified. If not, see above — use Colab.)*

**Kaggle, for training.** The deciding feature is background execution:
Kaggle's *Save & Run All* runs a notebook headless with the tab closed, which
free Colab cannot do. Kaggle also commits to a weekly GPU quota rather than
Colab free's best-effort allocation, and its Datasets give you somewhere to
keep the archive between sessions without Drive.

| | Kaggle | Colab free |
| --- | --- | --- |
| runs with the tab closed | **yes** | no |
| GPU allocation | committed weekly quota | best effort, throttled after use |
| session cap | ~12 h | ~12 h, plus idle disconnects |
| dataset between sessions | Datasets | Google Drive |
| storage that persists | `/kaggle/working`, ~20 GB | Drive, 15 GB free |

Use Colab if you are already living in Drive and want to poke at things
interactively. Use Kaggle for anything you want to start and walk away from.
Check the current quota figures yourself — both platforms change them.

**Neither for harvesting.** See "Running overnight" below: your Mac mini is the
right machine for that.

## The notebooks

| file | what it does | where |
| --- | --- | --- |
| [`kaggle_train.ipynb`](kaggle_train.ipynb) | train the detector | Kaggle (recommended) |
| [`colab_train.ipynb`](colab_train.ipynb) | train the detector | Colab |
| [`kaggle_harvest.ipynb`](kaggle_harvest.ipynb) | run the video pipeline | Kaggle |
| [`colab_harvest.ipynb`](colab_harvest.ipynb) | run the video pipeline | Colab |

### For the pipeline specifically

Training and harvesting want different things, and the ranking flips.

Harvesting is **incremental**: `state/seen.json` records what has already been
pulled, and it has to survive between sessions or each run re-downloads work
you already have. That is a read-write persistence problem, and it is where
both platforms are awkward.

| | Kaggle | Colab free | your Mac mini |
| --- | --- | --- | --- |
| runs with the tab closed | yes | no | yes (`overnight.sh`) |
| CPU (ffmpeg is the bottleneck) | ~4 vCPU | ~2 vCPU | 8-core M2 |
| ledger persistence | output -> re-attach as input each run | read-write Drive mount | just a file on disk |
| YouTube bot detection | datacenter IP, may block | datacenter IP, may block | **home IP** |

**Kaggle if you must pick a cloud** — background execution and double the CPU
outweigh the clunkier state loop. Colab's Drive mount is nicer for the ledger
but you have to leave the tab open, which defeats the point.

**But the Mac mini wins this one outright.** It is faster than either, its
state is just files, and it is the only option YouTube does not treat as a bot.
Disk was the reason to consider cloud, and `keep_raw: false` plus
`keep_clean: false` cuts a match from 333 MB to ~100 MB — 40 matches in 4.4 GB.
Cloud harvesting is the fallback for when that is genuinely not enough.

---

# Part 1 — Colab: train the model

Colab gives you a GPU for free. You only send it the dataset, never the videos.

## Step 1. Build a portable dataset (on your Mac)

```bash
cd ~/dev/TBACroppedOutVid
.venv-train/bin/python train/prepare_dataset.py --clean --copy --per-match 80 --prefer-scoring
.venv-train/bin/python train/autolabel_fuel.py
```

`--copy` is **not optional here.** Without it the script symlinks the images,
and a `tar` of symlinks arrives on Colab as an empty folder.

Check it worked — you want roughly equal image and label counts:

```bash
ls dataset/images/train | wc -l
ls dataset/labels/train | wc -l
```

## Step 2. Make the archive

```bash
tar -czf dataset.tgz dataset/
ls -lh dataset.tgz          # ~100 MB at 80 frames/match
```

## Step 3. Put it on Google Drive

Upload `dataset.tgz` to the **top level** of My Drive (drive.google.com, drag
and drop). Top level matters because the notebook looks for
`/content/drive/MyDrive/dataset.tgz`.

Use Drive rather than Colab's file picker: free sessions disconnect after a few
hours and wipe local disk. Anything not on Drive is gone.

## Step 4. Open the notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook**
3. Choose `deploy/colab_train.ipynb` from this repo

## Step 5. Turn the GPU on — do this before running anything

**Runtime → Change runtime type → Hardware accelerator → GPU → Save.**

Skipping this is the single most common mistake. It will train on CPU without
complaining, roughly 20x slower than your Mac mini.

## Step 6. Run the cells top to bottom

Shift+Enter through them. Two need your attention:

- **The Drive cell** pops up an authorisation window. Allow it.
- **The training cell** is the long one. Watch the first epoch: if `box_loss`
  and `cls_loss` are falling, it is learning.

## Step 7. Save the weights before the session dies

The notebook copies `best.pt` to `MyDrive/frc_weights/`. **Run that cell as
soon as training finishes** — do not leave it for later, the session can drop
at any time.

## Step 8. Bring it home

Download `best.pt` from Drive, then:

```bash
mkdir -p weights
mv ~/Downloads/best.pt weights/
.venv-train/bin/yolo predict model=weights/best.pt source=data/frames/ imgsz=1280
```

## If something goes wrong

| symptom | cause |
| --- | --- |
| `dataset/images/train` is empty after extracting | built without `--copy` — symlinks |
| `No labels found` | ran `prepare_dataset.py` but not `autolabel_fuel.py` |
| Training is glacial | GPU not enabled (Step 5) |
| `CUDA out of memory` | lower `batch` to 4, then 2 |
| "Cannot connect to GPU backend" | free quota exhausted; wait, or train locally |
| Everything vanished | session disconnected — that is why Step 7 exists |

---

# Part 1b — Colab: harvest the videos

Use [`colab_harvest.ipynb`](colab_harvest.ipynb). This is the option when your
Mac is short on disk: Colab processes on its own ~100 GB ephemeral disk and
only the small output goes to Drive.

**Runtime → CPU.** Harvesting needs no GPU, and requesting one spends quota you
want for training.

## The honest caveat, first

YouTube applies bot detection to datacenter IPs, and Colab is one. Downloads
may work fine, may work for some videos, or may fail with *"Sign in to confirm
you're not a bot"* — and which you get changes over time. **Cell 1 of the
notebook answers this in about 30 seconds.** Run it before anything else. If it
fails there is a cookies workaround at the bottom of the notebook, and if that
also fails, harvest on your Mac; a home IP is the one thing Colab cannot
provide.

## Steps

1. **Build the code archive** on your Mac and upload it to the top level of
   My Drive:
   ```bash
   ./deploy/make_code_archive.sh          # 76 KB, no .env, no data
   ```
2. **Add your TBA key to Colab secrets** — key icon in the left sidebar, name
   it `TBA_AUTH_KEY`, enable for the notebook. Do not paste it into a cell.
3. **Upload and open** `colab_harvest.ipynb`, set the runtime to CPU.
4. **Run cell 1** and stop if it fails.
5. **Run the rest**, starting with `pull -n 3` to prove the chain before
   committing an hour.
6. **Run the Drive-sync cell after every batch.** Sessions disconnect without
   warning and take local disk with them.

## What lands on Drive

Only what is expensive to regenerate — about **100 MB per match**:

```
MyDrive/tbavid/frames/      exported JPGs
MyDrive/tbavid/labels/      per-frame scoring CSVs
MyDrive/tbavid/scouting.db  the database
MyDrive/tbavid/seen.json    the ledger
```

Raw and cleaned videos stay on the ephemeral disk and are discarded
(`keep_raw` and `keep_clean` off). Free Drive is 15 GB, so that is roughly
150 matches.

**`seen.json` is the important one.** It is the record of what has already been
pulled; restore it at the start of a session and save it at the end, or the
next session will cheerfully re-download everything you already have.

---

# Running overnight, unattended

## Free Colab: no

Closing the tab ends the runtime. Free Colab also disconnects idle sessions and
caps them around 12 hours even when you are watching. Background execution is a
Colab Pro feature. Keep-alive scripts that fake interaction violate Colab's
terms and break without warning — do not build a pipeline on one.

## Your Mac mini: yes, and it is the better machine for this

It is a desktop with no lid to close, on a home IP that YouTube does not treat
as a bot. The only obstacle is macOS sleeping mid-run, which `caffeinate`
handles.

```bash
./deploy/overnight.sh 40          # 40 matches, in batches of 5
```

It detaches with `nohup`, so closing the terminal (or ssh-ing out) is fine.
Logs land in `logs/`:

```bash
tail -f logs/overnight_*.log
pkill -f 'run.py pull'            # to stop early
```

Before starting it checks free disk against what the run will need and refuses
rather than filling the volume — roughly 0.11 GB/match with `keep_raw` and
`keep_clean` off, 0.34 GB/match with them on. **Set them off first** if you are
pulling more than about 20 matches:

```json
"keep_raw": false,
"keep_clean": false
```

Work is committed per batch, so an interruption at hour six costs one batch,
not the night. The ledger means a re-run continues instead of starting over.

Roughly 4-6 minutes a match, so 40 matches is about 3-4 hours.

## Kaggle: yes, for training

**Save Version → Save & Run All (Commit)**, then close the tab. Kaggle runs the
notebook to completion in the background and attaches the output to that
version. This is the reason to prefer Kaggle over Colab; see
[`kaggle_train.ipynb`](kaggle_train.ipynb).

Two settings catch people out, both in the right-hand panel: **Accelerator**
defaults to None (it will train on CPU without complaining), and **Internet**
is off by default and needs a phone-verified account to enable — pip will fail
without it.

The same datacenter-IP caveat applies to *harvesting* there, so it is training
that belongs on Kaggle, not downloading.

---

# "Is Kaggle actually doing anything?"

Kaggle hides its status in places you would not guess, and one of its defaults
makes a working run look like a dead one.

## The log stops after a few seconds of debugger warnings

```
8.5s  Debugger warning: It seems that frozen modules are being used ...
9.2s  Note: Debugging will proceed. Set PYDEVD_DISABLE_FILE_VALIDATION=1 ...
```

That is **normal Kaggle boilerplate**, not an error, and it is not your code.
A log that stops there usually means the run is still in an early cell that
happens to be quiet — `pip install` produces no output for 30-60 s.

Give it two minutes and reload the version's Logs page. If it has not moved
after five, something is wrong: open the Logs and read the bottom.

Cell 0 of every notebook here prints a banner immediately for exactly this
reason, including whether Internet is on:

```
==========================================================
  notebook started  2026-09-13 21:04:11
  python 3.11.x on Linux-...
  gpu: none (fine for harvesting)
  internet: ON
==========================================================
```

If you do not see that banner, the run has not reached your first cell yet.

## While the editor is open

- **Bottom-right corner**: session status plus live CPU / RAM / GPU meters. If
  CPU is pinned, something is happening.
- **Left of each cell**: `[*]` means running, a number means finished. A cell
  that has been `[*]` for twenty minutes is normal for a pull; it is not
  normal for a pip install.
- The **timer** at the top right counts session runtime.

## The trap: `!command` shows nothing until it finishes

`!python3 run.py pull -n 20` buffers its output. You get an empty cell for half
an hour and then everything at once, which is indistinguishable from a hang.

The harvest notebook streams instead, printing elapsed time per line:

```
[00:04] [1/4] picking 5 unseen 2026 videos from TBA
[00:31]   downloaded 3NTYaOQBYHg.mp4 (194s)
[01:12]   6 shots / 6 clusters, main camera covers 84% of runtime
[02:26]   scoreboard: 395 scoring events (blue=726, red=61)
```

If those `[MM:SS]` lines keep appearing, it is working. Roughly 4-6 minutes a
match.

## After Save & Run All (the headless run)

This is the part people lose. The committed run is **separate from the editor
session** — closing the editor does not show it to you, and there is no
progress bar on the notebook page.

1. Open your notebook's page and click the **Version** count (top right, next
   to Save Version).
2. The version list shows each run with a status dot: **Running**,
   **Complete**, or **Failed** (red).
3. Click a version -> **Logs** to see the console output as it accumulates. A
   growing log is the definitive "it is alive" signal.
4. **Output** on the same version lists the files produced. Empty until it
   finishes.
5. The **bell icon** top-right notifies you when a commit completes.

## It says Failed — now what

Open **Logs** and read from the bottom. The usual causes, in order:

| log says | cause |
| --- | --- |
| `Sign in to confirm you're not a bot` | YouTube blocked the datacenter IP — cell 1 was warning you |
| `ModuleNotFoundError` / pip timeouts | **Internet is off** (Settings panel; needs a phone-verified account) |
| `No TBA API key found` | secret not added, or not enabled for this notebook |
| `CUDA out of memory` (training) | lower `batch` |
| stops dead around 12 h | session limit; pull in smaller batches |
| `No space left` | `/kaggle/working` is capped ~20 GB — keep `keep_raw`/`keep_clean` false |

## Fastest way to know it works at all

Run `-n 1` interactively first. One match takes ~5 minutes and exercises the
entire chain — TBA, yt-dlp, ffmpeg, tesseract, the database. If that works, a
batch of 20 is the same thing twenty times.

---

# Part 2 — Replit: host the scouting API

The API is standard-library only, so there is **nothing to install**. Six
files, 744 KB.

## Step 1. Build a current database (on your Mac)

```bash
cd ~/dev/TBACroppedOutVid
.venv/bin/python run.py db sync
```

## Step 2. Create the Repl

1. [replit.com](https://replit.com) → **Create Repl**
2. Template **Python**, give it a name, Create

## Step 3. Upload exactly these six files

Keep the folder structure — `tbavid/` and `data/` are folders, not prefixes.

```
serve.py
tbavid/__init__.py
tbavid/api.py
tbavid/config.py
tbavid/db.py
data/scouting.db
```

Use the Files pane's **⋮ → Upload file / Upload folder**. Create the `tbavid`
and `data` folders first, then upload into them.

**Do not upload** `.env` (your TBA key), `data/frames/`, `data/videos/`,
`data/raw/`, or `.venv/`. The API needs none of them, and the key is a secret
that has no business on a web host. Serving is read-only and never calls TBA.

## Step 4. Tell Replit how to start it

Create a file named `.replit` at the top level:

```toml
run = "python3 serve.py"

[env]
HOST = "0.0.0.0"

[[ports]]
localPort = 8781
externalPort = 80
```

`HOST=0.0.0.0` is required. The default binds to localhost, which on a host
means nothing outside the container can reach it — the server looks perfectly
healthy in the logs and is unreachable from the internet.

## Step 5. Run it

Press **Run**. The console should print:

```
scouting API on http://0.0.0.0:8781
  GET ^/health$
  ...
```

A webview opens. Add `/health` to the URL and you should see row counts.

## Step 6. Check it from outside

```bash
curl https://YOUR-REPL-NAME.YOUR-USERNAME.repl.co/health
curl https://YOUR-REPL-NAME.YOUR-USERNAME.repl.co/teams
```

## Step 7. Point the scouting app at it

CORS is already open (`Access-Control-Allow-Origin: *`), so a browser app can
call it directly:

```js
const API = 'https://YOUR-REPL-NAME.YOUR-USERNAME.repl.co';
const teams = await (await fetch(`${API}/teams`)).json();
const match = await (await fetch(`${API}/matches/2026gal_qm62`)).json();
```

`GET /schema` returns every table, column and route, so the app can discover
the shape without reading any of this.

## Keeping it up

A free Repl sleeps when idle and wakes on the next request — the first call
after a nap is slow, then it is fine. For a URL that never sleeps you need
Replit's paid deployments; a sleeping free Repl is usually fine for scouting,
where traffic comes in bursts around matches.

## Refreshing the data after a new pull

The deployed database is a **snapshot**. Nothing on Replit writes to it.

```bash
.venv/bin/python run.py pull -n 5
.venv/bin/python run.py db sync
```

Then re-upload `data/scouting.db`, replacing the old one, and hit Run again.

## If something goes wrong

| symptom | cause |
| --- | --- |
| `No database at ...` | `data/scouting.db` missing or in the wrong folder |
| `ModuleNotFoundError: tbavid` | `tbavid/` uploaded as loose files, not a folder |
| Works in the webview, not from outside | `HOST=0.0.0.0` missing (Step 4) |
| `ModuleNotFoundError: numpy` | you ran `run.py serve`; use `serve.py` |
| CORS error in the browser | you are hitting `http://` — use the `https://` URL |

---

# What runs where, in one table

| | Colab | Replit | your Mac |
| --- | --- | --- | --- |
| harvest video | no — YouTube blocks datacenter IPs | no, same | **yes** |
| train | **yes** | no GPU | yes, overnight |
| serve the API | no | **yes** | yes |
