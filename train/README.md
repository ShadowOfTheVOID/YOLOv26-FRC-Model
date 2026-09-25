# Training

Separate venv from the harvester, because the harvester only needs
`requests` + `numpy` and shouldn't drag in torch:

```bash
/opt/homebrew/bin/python3 -m venv .venv-train
.venv-train/bin/pip install "ultralytics>=8.4" opencv-python-headless
```

Verified on this machine: Python 3.14.7, ultralytics 8.4.150, torch 2.14.0,
MPS available.

## Two models from one labelling effort

Label once; derive twice. The five classes exist for the scouting detector,
and the counter reads only three of them.

| | classes | runs | what matters |
| --- | --- | --- | --- |
| **scouting detector** | all five | `run.py detect`, offline over harvested frames | accuracy; it can take all night |
| **counting model** | `fuel`, `hub_blue`, `hub_red` | `run.py count`, live at a scrimmage | **speed is correctness** |

That second row is the one to think about. A counting model too slow for the
camera misses balls between the frames it sees; the score comes out low, with
no gap and nothing odd about it, and at a scrimmage there is no FMS to
disagree. Two unused classes are work done on every frame for an output
nothing reads.

```bash
python3 train/subset_classes.py                       # -> dataset-fuel/
python3 train/train.py --data dataset-fuel/dataset.yaml \
    --model yolo26n.pt --name count26 --export onnx
python3 train/benchmark.py --weights runs/count26/weights/best.pt \
    --imgsz 960 --fps 30                              # before the event
```

`benchmark.py` reports the median and the p95. The p95 is the one that matters:
a model averaging 30 fps that stalls for 200 ms every few seconds drops balls
in the stalls while looking fine on the average.

## The three steps

```bash
.venv-train/bin/python train/prepare_dataset.py --clean --per-match 80 --prefer-scoring
.venv-train/bin/python train/autolabel_fuel.py
.venv-train/bin/python train/train.py            # yolo26n first; add --p2 later
```

Not on this machine: [deploy/SETUP.md](../deploy/SETUP.md) for Colab and
Kaggle, [deploy/AMD_DEVCLOUD.md](../deploy/AMD_DEVCLOUD.md) for an AMD
Instinct MI300X, where the useful settings are different enough to be worth
their own page.

`--per-match 80` matters more than it looks. At 3 fps a match yields ~480
frames that are largely near-duplicates, and trimming to 80 spread-out ones
took a 100-epoch run on an 8 GB M2 from ~27 h to ~5.5 h with no real loss of
variety. `--prefer-scoring` biases the selection toward frames where the
scoreboard recorded fuel landing just after.

1. **`prepare_dataset.py`** symlinks `data/frames/` into the Ultralytics
   layout and writes `dataset/dataset.yaml`. It splits **by match, never by
   frame** — frames 1/3 s apart from one match are near-identical, so a
   random frame split leaks near-copies into val and the resulting metric is
   meaningless. The assignment is hash-based, so a match keeps its side of the
   split as the dataset grows.

2. **`autolabel_fuel.py`** proposes boxes for class 0 by colour. A frame holds
   ~240 fuel balls, so a single match is ~127k boxes — hand-drawing them isn't
   an option. Isolated balls measure a median area/bbox fill of 0.79 (a circle
   is 0.785), so thresholding separates them cleanly; heaps fall below the fill
   gate and are skipped rather than boxed as one amorphous blob. It produced
   56,692 proposals across 532 frames in 3.2 s. Re-running never overwrites a
   label file unless you pass `--overwrite`, so hand corrections survive.
   Preview before committing:

   ```bash
   .venv-train/bin/python train/autolabel_fuel.py --preview /tmp/check.jpg
   ```

3. **`train.py`** picks MPS/CUDA/CPU automatically and refuses to start if
   every label file is empty — Ultralytics reads an empty label as "this image
   contains nothing", so training on unlabelled frames silently teaches the
   model to detect nothing.

## Why `--p2`

Fuel is ~17x12 px. Stock `yolo26` detects at strides 8/16/32; `yolo26-p2` adds
a **stride-4** head, so a 17px ball lands on a 4px-resolution feature map
instead of a 2px one. `--p2` builds the P2 architecture and seeds it from the
pretrained checkpoint. Costs speed and memory; worth it here.

Other defaults set for small objects: `scale=0.25` (the stock 0.5 can shrink a
17px ball to 8px), `close_mosaic=15` so the run finishes on real full-frame
layouts.

## What still needs a human

`fuel` is auto-labelled by colour. Robots have an automatic path too, with
limits: `autolabel_robots.py` asks an open-vocabulary model for them and keeps
only frames where it found most of them (see its docstring for the measured
recall). Run it after `autolabel_fuel.py`, then derive the scouting set:

```bash
.venv-train/bin/python train/autolabel_robots.py --preview previews/robots   # look
.venv-train/bin/python train/autolabel_robots.py --dry-run                   # count
.venv-train/bin/python train/autolabel_robots.py                             # write
python3 train/subset_classes.py --classes fuel,robot_blue,robot_red --out dataset-scout
```

Everything else needs annotation — but far less than it looks:

- **The hubs are static.** The camera is fixed within a match, so the two hub
  boxes are identical across all 532 frames. Annotate once, replicate.
- **Don't annotate every frame.** 50–100 well-spread frames per match beats
  532 consecutive near-duplicates, and costs a tenth as much.
- Load `dataset/` into Label Studio, CVAT or Roboflow; the YOLO layout imports
  directly, fuel boxes and all.

## Detection is not the answer to your actual question

"How many balls went in from a specific bot" is three problems, and YOLO is
only the first:

1. **Detect** fuel, robots, hubs per frame — this pipeline.
2. **Track** fuel across frames (`model.track(..., tracker="bytetrack.yaml")`,
   built into Ultralytics) to get trajectories rather than isolated boxes.
3. **Associate** each trajectory's origin with a robot and its terminus with a
   hub, then reconcile the count against `data/labels/<match>.csv`.

The scoreboard CSV is ground truth for step 3, not a training target for
step 1 — it says an *alliance* scored N fuel around time t. For per-robot
attribution you need to tell the three robots on a side apart, which means
reading bumper numbers; TBA's match record already names the exact six teams
on the field, so that OCR is a 6-way choice, not open-ended.

## Before you trust any number

You have **one match**. There is no honest validation split from one match.
Pull 8–10 first:

```bash
.venv/bin/python run.py pull -n 10    # harvester venv, not .venv-train
```
