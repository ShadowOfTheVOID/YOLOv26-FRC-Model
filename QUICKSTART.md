# Quick start: run the trained models

From nothing to boxes on a video, a fuel count, and a per-robot shot sheet.
Harvesting and training are elsewhere (README.md, train/README.md); this page
is only about *using* the models.

## 1. Install (once)

```bash
git clone https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model.git
cd YOLOv26-FRC-Model
python3 -m venv .venv-train
.venv-train/bin/pip install -r requirements.txt -r requirements-detect.txt
```

Windows: use `.venv-train\Scripts\pip` and `.venv-train\Scripts\python`.

## 2. Get the models (once)

Download both from the [v0.3.0 release](https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model/releases/tag/v0.3.0)
into a `models/` folder (`*.pt` is gitignored, so they never get committed):

| file | detects | use it for |
| --- | --- | --- |
| `fuel_withBotbest.pt` | fuel, blue robots, red robots | per-robot shots (`run.py shots`) |
| `fuel_best.pt` | fuel | fuel counting (`run.py count`) |

Check them against the checksums in the release notes:

```bash
sha256sum models/*.pt        # macOS: shasum -a 256 models/*.pt
```

The shortcuts below save typing:

```bash
PY=.venv-train/bin/python
M=models/fuel_withBotbest.pt
```

## 3. See what the model sees

```bash
$PY -c "from ultralytics import YOLO; YOLO('$M').predict('match.mp4', imgsz=960, conf=0.25, save=True)"
```

The annotated copy lands under `runs/detect/` (the path is printed). Swap
`match.mp4` for a folder of `.jpg` frames or a single image.

## 4. Hub boxes (once per camera position)

Neither model detects hubs; you tell it where they are, in the video's own
pixels, as `x,y,width,height` around the top opening of each hub:

```bash
ffmpeg -loglevel error -ss 20 -i match.mp4 -frames:v 1 hubs.png
```

Open `hubs.png` in Preview, **Tools -> Show Inspector**, drag a rectangle over
each hub opening and read off its position and size. For the harvested
2026nhdur broadcasts they are `--hub-blue 500,120,165,95 --hub-red 1285,110,160,105`.

## 5. Count fuel into the hubs

```bash
$PY run.py count --weights models/fuel_best.pt --source match.mp4 \
    --hub-blue X,Y,W,H --hub-red X,Y,W,H
```

`--source 0` uses a live camera. `--scoreboard` turns it into the match's
scorekeeper (clock, JSON score feed, referee corrections); see
`run.py count --help`.

## 6. Per-robot shots

```bash
$PY run.py shots --weights $M --source match.mp4 \
    --hub-blue X,Y,W,H --hub-red X,Y,W,H \
    --annotate shots.mp4 --out shots.json
```

Watch `shots.mp4`: every robot carries a number (`#1`, `#2`, ...). Nothing
reads bumpers, so write down which number is which team -- a robot hidden
behind a referee can come back under a new number, which is fine -- then run
again with them folded into teams:

```bash
    --teams 2=6895,5=1058,8=1058,6=611
```

## 7. Will this laptop keep up with a live camera?

```bash
$PY train/benchmark.py --weights $M --imgsz 960 --fps 30
```

If it says it cannot keep up, a live count comes out low without warning.

## What to trust (measured, v0.3.0)

- **Robot boxes: good.** On a held-out match, every robot box checked by eye
  was a real robot in the right alliance colour.
- **Hub counts from a broadcast camera: not good enough to replace FMS.** On
  2026nhdur qm7 the scoreboard showed 74 balls in the first 30 s; the shot
  counter saw 19, and `run.py count` fewer. Balls ~15 px wide, fired in
  streams, cannot be tracked from that distance. For scoring, point a camera
  close at each hub, and keep a human scorekeeper.
- **Per-robot makes: rough. Per-robot misses: do not use yet.** Every error
  found so far has shown up as a false miss.
- Both models have only been trained on one event's broadcast (2026nhdur).
  Expect a new venue or camera to do worse until it is checked.
