"""Writes train/colab_harvest_1280.ipynb: python train/make_harvest_notebook.py (from the repo root)."""
import json
import sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(r'''
# Harvest match videos and train YOLO26s at 1280

The 2026-11 bake-off picked **YOLO26s at 1280** as the best fuel detector (fuel AP50 0.963), but it was
trained on four Einstein matches only, one broadcast look. This notebook gives it what it lacks: many more
matches, from many events, cameras and lights.

1. **Harvest**: `run.py fetch` picks 2026 competition matches from The Blue Alliance (a few per event, for
   variety), downloads them and keeps the main camera with the banner cropped off.
2. **Label**: the same recipe as the bake-off: `relabel_video.py` (robots, and balls in flight) then SAM 3 for
   the fuel, in batches of a few matches, each saved to Drive as it finishes.
3. **Train** YOLO26s at 1280 on the harvest plus the Einstein 4/5/8 set.
4. **Judge it on Einstein 1**, never trained on: the score table and the counting test, against the
   bake-off's 1280 model and the current `fuel_relabel.pt`.
5. **Export** the winner to `.onnx` for the apps.

Every step saves to `MyDrive/frc-harvest/` and skips what is already done, so after a disconnect you just press
**Runtime → Run all** again.

**Before you start**

* **Colab Secrets** (key icon on the left): `TBA_AUTH_KEY` (thebluealliance.com → Account → Read API key) and
  `HF_TOKEN` (the same Hugging Face token as the bake-off; needs access to `facebook/sam3`). On a phone, where
  Secrets is hidden, the notebook asks for each one instead.
* `MyDrive/frc-bakeoff/` from the bake-off stays where it is: Einstein 1 and the Einstein training set come from it.
* **Runtime**: L4. Harvesting is mostly downloading; labelling and training need the GPU.

**Budget** (L4, ~4–5 units an hour): 40 matches ≈ 1 h harvest + 3–5 h labelling + 6–10 h training.
**Do the dry run first** (`DRY_RUN = True`: 2 matches, a few frames, 1 epoch; ~20 min).

**If downloads fail** with "Sign in to confirm you're not a bot": YouTube refuses some cloud computers,
Colab's included. The harvest then stops at once and marks nothing, so the matches are tried again later.
Three ways round it:

* **Retry later**: it is sometimes temporary.
* **A cookies file**: export `cookies.txt` from a browser signed in to a *spare* Google account (the "Get
  cookies.txt LOCALLY" extension), put it in `MyDrive/frc-harvest/` and set `YT_COOKIES = "cookies.txt"`.
  Your choice: cookies expire, and they let the downloader act as that account.
* **Harvest on your own computer** (YouTube does not block home connections), then let Colab label and train.
  In a checkout of this repo with the requirements installed (`pip install -r requirements.txt yt-dlp`, plus
  ffmpeg), and `TBA_AUTH_KEY` set:

  ```
  export TBAVID_DATA=~/frc-harvest/hdata
  python3 run.py fetch -n 40 --per-event-cap 2 --seed 2026
  ```

  then put that `hdata` folder (and the repo's `state/` folder, as `state`) into `MyDrive/frc-harvest/`
  and Run all here: the harvest step finds the matches already there and goes straight to labelling.
''')

md("## 1. Settings")
code(r'''
DRY_RUN = True            # True: 2 matches, a few frames, 1 epoch -- checks everything end to end. Then False.

DRIVE_DIR   = "/content/drive/MyDrive/frc-harvest"    # everything this notebook makes
BAKEOFF_DIR = "/content/drive/MyDrive/frc-bakeoff"    # the bake-off: Einstein 1 to judge on, Einstein 4/5/8 to train on
BRANCH      = "claude/practical-einstein-bgrb3r"      # switch to "main" once this work is merged

# Harvest
MATCHES       = 40        # how many matches to label and train on (more is better; each is ~150 MB in Drive)
PER_EVENT_CAP = 2         # at most this many from one event: variety beats volume
SEED          = 2026      # fixes which matches are picked, so a re-run picks the same ones
EXCLUDE       = ("2026cmptx",)   # Einstein Field: Einstein 1 is the test, never trained on
YT_COOKIES    = ""        # "" or a cookies.txt in DRIVE_DIR (see the top of the notebook)

# Label
EVERY        = 1.0        # seconds between labelled frames (0.5 doubles the frames and the SAM 3 time)
LABEL_BATCH  = 5          # matches labelled at a time; each batch is saved to Drive when done
FUEL_PROMPT  = "yellow ball"   # next season: describe the new game piece
WITH_EINSTEIN = True      # also train on the bake-off's Einstein 4/5/8 set

# Train
IMGSZ      = 1280
EPOCHS     = 80
YOLO_MODEL = "yolo26s.pt"
BATCH      = 0.8          # a fraction = Ultralytics picks the batch that fills 80% of the GPU (what worked at 1280)
RUN_NAME   = "yolo26s_1280_harvest"
''')

md("## 2. GPU")
code(r'''
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU: Runtime -> Change runtime type -> L4"
''')

md("## 3. Drive, code, packages, keys")
code(r'''
from google.colab import drive
drive.mount("/content/drive")
import glob, hashlib, json, os, pathlib, shutil, subprocess, sys
os.makedirs(f"{DRIVE_DIR}/runs", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/labelled", exist_ok=True)

%cd /content
!rm -rf frc && git clone -q --depth 1 -b {BRANCH} https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model frc
%cd /content/frc
!rm -rf runs && ln -s "{DRIVE_DIR}/runs" runs      # weights and checkpoints live in Drive
!pip -q install -r requirements.txt "ultralytics>=8.4" "transformers>=5.1" yt-dlp onnx onnxslim onnxruntime
REPO = "/content/frc"

def secret(names, prompt):
    """A Colab Secret, else ask (on a phone the Secrets panel is hidden). Never stored in the notebook."""
    from google.colab import userdata
    for n in names:
        try:
            v = userdata.get(n)
            if v:
                return v
        except Exception:
            pass
    from getpass import getpass
    return getpass(prompt).strip()

os.environ["TBA_AUTH_KEY"] = secret(("TBA_AUTH_KEY", "TBA_KEY", "tba_auth_key"),
                                    "TBA Read API key (thebluealliance.com -> Account): ")
assert os.environ["TBA_AUTH_KEY"], "No TBA key: add TBA_AUTH_KEY in Colab Secrets or paste it when asked."

def sh(args, cwd=REPO):
    """Run a step and show its output here as it comes. Colab does not show what a program
    started with subprocess.run prints, so the first dry run said only "nothing downloaded"
    and hid why (2026-11-18)."""
    p = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()

# The key, checked now: a wrong one used to surface only as "nothing downloaded". With the
# User-Agent the harvester itself sends: without one TBA's firewall answers 403 to ANY key
# (measured 2026-11-18: no User-Agent -> 403, this one -> 401 for a fake key), which this check
# first misreported as a refused key.
import urllib.request, urllib.error
problem = ""
try:
    req = urllib.request.Request("https://www.thebluealliance.com/api/v3/events/2026/simple",
                                 headers={"X-TBA-Auth-Key": os.environ["TBA_AUTH_KEY"],
                                          "User-Agent": "TBACroppedOutVid/1.0"})
    print(f"TBA key works: {len(json.load(urllib.request.urlopen(req, timeout=30)))} events in 2026")
except urllib.error.HTTPError as e:
    if e.code == 401:
        problem = ("TBA refused the key (401): use the Read API key from thebluealliance.com -> Account "
                   "-> Read API Keys (not a Write key).")
    elif e.code == 403 and os.path.exists(f"{DRIVE_DIR}/tba_cache.zip"):
        # TBA's firewall refuses Colab's machines whatever the key (2026-11-18: 403 here, the same
        # key answered elsewhere). The harvest then uses the event and match lists saved in Drive.
        print("TBA blocks Colab's machines (403): using the saved event and match lists (tba_cache.zip)")
    else:
        problem = (f"TBA answered {e.code}" + (" and blocks Colab's machines; there is no tba_cache.zip in "
                   f"{DRIVE_DIR} to use instead" if e.code == 403 else ": try again in a few minutes"))
if problem:                     # raised outside the except, which Colab's error display trips over
    raise SystemExit(problem)
''')

md(r'''
## 4. Harvest

Matches download to this computer's disk and are copied to Drive after every few, with the record of what
was already pulled (so a re-run never downloads the same match twice). Only the main-camera clips are kept,
not the raw downloads.
''')
code(r'''
H = "/content/hdata"                               # harvest data on the fast local disk
os.environ["TBAVID_DATA"] = H
DH = f"{DRIVE_DIR}/hdata"                          # ... copied to Drive after every round
os.makedirs(H, exist_ok=True); os.makedirs(DH, exist_ok=True); os.makedirs(f"{DRIVE_DIR}/state", exist_ok=True)

def sync(src, dst):
    subprocess.run(["rsync", "-a", f"{src}/", f"{dst}/"], check=False)

# bring back what earlier sessions harvested, and the never-twice ledger
sync(DH, H)
sync(f"{DRIVE_DIR}/state", f"{REPO}/state")
# TBA's event and match lists saved on a machine TBA answers: used when it refuses Colab (403)
if os.path.exists(f"{DRIVE_DIR}/tba_cache.zip"):
    os.makedirs(f"{REPO}/state/tba_cache", exist_ok=True)
    subprocess.run(["unzip", "-q", "-n", f"{DRIVE_DIR}/tba_cache.zip", "-d", f"{REPO}/state/tba_cache"])
    print(f"saved TBA lists: {len(os.listdir(f'{REPO}/state/tba_cache'))} files")

cfg = json.load(open(f"{REPO}/config.json"))
cfg.update({"keep_raw": False,          # the clean main-camera clip is what gets labelled
            "score_labels": False,      # scoreboard OCR is not needed to train a detector
            "ytdlp_cookies": f"{DRIVE_DIR}/{YT_COOKIES}" if YT_COOKIES else ""})
CFG = f"{REPO}/harvest-config.json"
json.dump(cfg, open(CFG, "w"), indent=1)

def manifest():
    p = f"{H}/review/manifest.json"
    return json.load(open(p)) if os.path.exists(p) else {"videos": {}}

def usable():
    """Harvested matches ready to label: kept, and not from the Einstein events."""
    return sorted(v for v, e in manifest()["videos"].items()
                  if e.get("status") == "ok" and not any(x in v for x in EXCLUDE))

target = 2 if DRY_RUN else MATCHES
rounds, code = 0, 0
while len(usable()) < target and rounds < 12:
    n = min(5, target - len(usable()))
    print(f"--- fetching {n} more ({len(usable())} of {target} so far)", flush=True)
    # --retry-failed: matches an older version marked "unavailable" when YouTube's bot check
    # (not the video) refused them are tried again; a really removed video costs a second.
    code = sh([sys.executable, "run.py", "--config", CFG, "fetch", "-n", str(n), "--retry-failed",
               "--per-event-cap", str(PER_EVENT_CAP), "--seed", str(SEED)])
    sync(H, DH); sync(f"{REPO}/state", f"{DRIVE_DIR}/state")      # saved after every round
    rounds += 1
    if code != 0:      # YouTube refusing this computer: every further try would get the same answer
        break

have = usable()
print(f"\n{len(have)} matches ready: {', '.join(have[:8])}{' ...' if len(have) > 8 else ''}")
if code != 0 and len(have) < target:
    raise SystemExit("The harvest stopped: the reason is just above. If YouTube refused this computer "
                     "('not a bot'), see 'If downloads fail' at the top: retry later, use a cookies.txt, "
                     "or harvest on your own computer (also there). The matches already here are kept.")
if not have:
    why = {}
    for v, e in manifest()["videos"].items():
        why[e.get("status")] = why.get(e.get("status"), 0) + 1
    raise SystemExit(f"Nothing downloaded ({why or 'no video was even tried'}). The fetch messages "
                     f"above say why: 'confirm you're not a bot' -> see 'If downloads fail' at the top; "
                     f"'nothing new to pull' -> every match is already in the ledger (state/ in Drive).")
''')

md(r'''
## 5. Label

The bake-off's recipe, batch by batch: `relabel_video.py` with the v0.3.0 models (robots, and balls in flight),
then SAM 3 for the fuel. Each batch is zipped to `frc-harvest/labelled/` the moment it is done, so a disconnect
costs one batch at most.
''')
code(r'''
os.environ["HF_TOKEN"] = secret(("HF_TOKEN", "HF_Token", "hf_token"),
                                "Hugging Face read token (huggingface.co -> Settings -> Access Tokens): ")

# the labelling models that built the bake-off's set, checked against their recorded checksums
SHA = {"fuel_best.pt": "202429d3e8a7640e444e8053b229998b363f698469274829a069dd015994b5f3",
       "fuel_withBotbest.pt": "0c5267368787e054f9fbd57c49078726306ca3370b7185752c3c2229dfffd14e"}
os.makedirs("/content/models", exist_ok=True)
for m, want in SHA.items():
    dst = f"/content/models/{m}"
    if not os.path.exists(dst):
        !curl -sL -o {dst} https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model/releases/download/v0.3.0/{m}
    assert hashlib.sha256(open(dst, "rb").read()).hexdigest() == want, f"{m} is not the original model"

DONE = f"{DRIVE_DIR}/labelled/labelled.json"       # zip name -> the matches in it
done = json.load(open(DONE)) if os.path.exists(DONE) else {}
labelled = {v for vs in done.values() for v in vs}
todo = [v for v in usable() if v not in labelled][: target]
every = 5.0 if DRY_RUN else EVERY
for i in range(0, len(todo), LABEL_BATCH):
    batch = todo[i:i + LABEL_BATCH]
    name = "batch-" + hashlib.md5(",".join(batch).encode()).hexdigest()[:10] + ".zip"
    print(f"===== labelling {len(batch)} matches -> {name}", flush=True)
    rl, sam = "/content/rl", "/content/sam"
    !rm -rf {rl} {sam}
    code = sh([sys.executable, "train/relabel_video.py", "--out", rl, "--from-scraper", "--data", H,
               "--matches", *batch, "--every", str(every), "--device", "0",
               "--fuel", "/content/models/fuel_best.pt", "--robots", "/content/models/fuel_withBotbest.pt"])
    if code != 0 or not glob.glob(f"{rl}/images/*/*"):
        print(f"!! relabel_video found nothing to label in {batch} -- skipped (its messages are above)"); continue
    limit = ["--limit", "8"] if DRY_RUN else []
    code = sh([sys.executable, "train/autolabel_sam3.py", "--data", f"{rl}/dataset.yaml", "--out", sam,
               "--fuel-prompt", FUEL_PROMPT, *limit])
    if code != 0:
        raise SystemExit("SAM 3 failed: its error is above (no access to facebook/sam3? a bad token?)")
    !cd {sam} && zip -qr "{DRIVE_DIR}/labelled/{name}" .
    done[name] = batch
    json.dump(done, open(DONE, "w"), indent=1)
    print(f"saved {name} ({len(glob.glob(f'{sam}/images/*/*'))} images)")
print(f"\n{len(done)} labelled batches, {sum(len(v) for v in done.values())} matches")
''')

md(r'''
## 6. The training set

Every labelled batch, plus the Einstein 4/5/8 set from the bake-off. Validation during training uses the
harvest's own held-out matches (split by match, never by frame); Einstein 1 is kept out entirely, for section 8.
''')
code(r'''
train, val, names = [], [], None
!rm -rf /content/lab && mkdir -p /content/lab
for name in sorted(done):
    d = f"/content/lab/{name[:-4]}"
    !mkdir -p {d} && unzip -q -o "{DRIVE_DIR}/labelled/{name}" -d {d}
    y = open(f"{d}/dataset.yaml").read()
    names = names or y[y.index("names:"):]
    for split, into in (("train", train), ("val", val)):
        if glob.glob(f"{d}/images/{split}/*"):
            into.append(f"{d}/images/{split}")
EIN = "/content/ein"
if WITH_EINSTEIN:
    z = f"{BAKEOFF_DIR}/dataset-sam3.zip"
    assert os.path.exists(z), f"{z} not found: set WITH_EINSTEIN = False, or put the bake-off's dataset back"
    if not os.path.exists(f"{EIN}/dataset.yaml"):
        !rm -rf {EIN} && mkdir -p {EIN} && unzip -q "{z}" -d {EIN}
    train.append(f"{EIN}/images/train")             # Einstein 4/5/8; its val (Einstein 1) is never trained on
assert train, "No labelled images: run section 5 first"
if not val:
    # Splits are by match (about 1 in 5 held out), so 2 dry-run matches can both land in train.
    if not DRY_RUN:
        raise SystemExit("No held-out match among the labelled ones: raise MATCHES and run sections 4-5 again")
    val = train[:1]
    print("dry run: no held-out match among 2, validating on a training batch (the real run holds out ~1 in 5)")
DATA = "/content/train1280.yaml"
open(DATA, "w").write("path: /content\n"
                      f"train: [{', '.join(train)}]\n"
                      f"val: [{', '.join(val)}]\n" + names)
count = lambda dirs: sum(len(glob.glob(f"{d}/*")) for d in dirs)
print(open(DATA).read()); print(f"{count(train)} training images, {count(val)} validation images")
''')

md(r'''
## 7. Train YOLO26s at 1280

Weights and checkpoints go to `frc-harvest/runs/`. After a disconnect, Run all again: it resumes from the last
epoch. 80 epochs on 40 matches is about 6–10 hours on an L4.
''')
code(r'''
EP = 1 if DRY_RUN else EPOCHS
name = RUN_NAME + ("_dry" if DRY_RUN else "")
run = pathlib.Path(f"runs/{name}")
last = run / "weights" / "last.pt"
if (run / "DONE").exists():
    print(f"{name}: already trained (delete runs/{name}/DONE in Drive to train again)")
elif last.exists():
    !python train/train.py --model {last} --resume --data {DATA} --name {name} && touch {run}/DONE
else:
    !python train/train.py --data {DATA} --model {YOLO_MODEL} --imgsz {IMGSZ} --epochs {EP} --batch {BATCH} --workers 8 --name {name} && touch {run}/DONE
assert (run / "DONE").exists(), "Training did not finish: its error is above. Run all again to resume."
BEST = str(run / "weights" / "best.pt")
''')

md(r'''
## 8. Judge it on Einstein 1

The two tests from the bake-off, on a match none of these models trained on: the score table on Einstein 1's
487 frames, and counting Einstein 1 against its scoreboard. Compared with the bake-off's 1280 model and the
current `fuel_relabel.pt`.
''')
code(r'''
cur = "/content/models/fuel_relabel.pt"
if not os.path.exists(cur):
    !curl -sL -o {cur} https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model/releases/download/v0.5.8/fuel_relabel.pt
assert hashlib.sha256(open(cur, "rb").read()).hexdigest() == \
    "bf9eb2c3cb25b119b54bd9a09d35227ad30a99a371a5dc532828cd9c0d7f4977", "fuel_relabel.pt is not the released one"
bake = f"{BAKEOFF_DIR}/runs/yolo26s_1280_bake_sam3/weights/best.pt"
tag = "dry" if DRY_RUN else "final"

if os.path.exists(f"{EIN}/dataset.yaml"):
    yolos = [f"--yolo harvest1280={BEST}@1280"] + ([f"--yolo bake1280={bake}@1280"] if os.path.exists(bake) else [])
    limit = "--limit 20" if DRY_RUN else ""
    !python train/compare_detectors.py --data {EIN}/dataset.yaml --imgsz 960 {" ".join(yolos)} {limit} --cache runs/compare_cache_{tag} | tee /content/compare.txt
    !cp /content/compare.txt runs/compare_einstein1_{tag}.txt
else:
    print("No Einstein dataset (WITH_EINSTEIN = False): score table skipped")

e1 = f"{BAKEOFF_DIR}/videos/e1.mp4"
if os.path.exists(e1):
    if not os.path.exists("/content/e1.mp4"):
        !cp "{e1}" /content/e1.mp4
    models = [f"--model current={cur}", f"--model harvest1280={BEST}"] + ([f"--model bake1280={bake}"] if os.path.exists(bake) else [])
    end = "--end 40" if DRY_RUN else ""
    !python train/count_einstein.py --video /content/e1.mp4 {" ".join(models)} {end} --out runs/count_e1_harvest_{tag}.txt
else:
    print(f"No {e1}: counting test skipped")
''')

md(r'''
## 9. Export for the apps

The trained model as `.onnx`, checked box for box against the `.pt` (`train/export_onnx.py` refuses an export
that differs). It is saved next to the weights in Drive. Whether it replaces the built-in model is decided from
section 8's counting test, not the score table: in the bake-off the best detector did not count best.
''')
code(r'''
!python train/export_onnx.py {BEST} --imgsz 640
print("saved:", BEST.replace(".pt", ".onnx"))
''')

md("## 10. Stop paying")
code(r'''
from google.colab import runtime
runtime.unassign()      # ends the session so it stops using compute units; everything is in Drive
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = sys.argv[1] if len(sys.argv) > 1 else "train/colab_harvest_1280.ipynb"
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
open(out, "a").write("\n")
print("wrote", out, len(cells), "cells")
