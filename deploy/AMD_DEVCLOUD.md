# Training on an AMD MI300X (AMD Developer Cloud)

Colab gives you one T4 and takes it away after a few hours. The AMD Developer
Cloud gives you an MI300X — 192 GB of HBM3, or eight of them on the 8x droplet
— and it stays up until you destroy it. For a 73-hour run on an 8 GB M2 that is
the difference between a weekend and a lunch break.

**Read this first, because it is the thing that goes wrong:** nothing in this
repository has been run on an MI300X. The numbers below are the arithmetic from
the M2 measurements in [train/README.md](../train/README.md), not observations,
and they are labelled as estimates where they are estimates. Measure epoch 1
and believe that instead.

Two other things hold regardless of hardware:

- **Do not harvest on the droplet.** YouTube blocks datacenter IPs, so
  `run.py pull` fails there in a way that looks like a bug. Harvest at home,
  ship the dataset up.
- **The droplet is not where your weights live.** It is billed by the hour and
  destroyed when you are done. Step 8 is not optional.

## 0. What you need before you start

| | |
| --- | --- |
| an AMD Developer Cloud account | [amd.com](https://www.amd.com/en/developer/resources/rocm-hub/ai-cloud-development.html) — there is a complimentary GPU-hour allowance; the size of it has changed more than once, so read the terms rather than this table |
| a labelled dataset, built locally | `dataset/` or `dataset-fuel/`, from the three steps in [train/README.md](../train/README.md) |
| an SSH key | added during droplet creation; there is a web console too, but `scp` at the end wants the key |

## 1. Build the dataset locally, with `--copy`

```bash
.venv-train/bin/python train/prepare_dataset.py --clean --copy --per-match 80 --prefer-scoring
.venv-train/bin/python train/autolabel_fuel.py
tar -czf dataset.tgz dataset/
ls -lh dataset.tgz                 # ~100 MB at 80 frames/match
```

`--copy` is load-bearing, for two reasons that both end in an empty dataset.
`prepare_dataset.py` otherwise symlinks to `data/frames/` at an **absolute,
resolved** path: `tar` stores the links rather than the images, and even if you
rsync the frames too, the container mounts the repo somewhere else and every
link dangles. Ultralytics reads a missing image the same way it reads an empty
label file — as a frame containing nothing — so the run starts, reports a
healthy image count, and trains on nothing.

Check before you ship it:

```bash
ls dataset/images/train | wc -l
ls dataset/labels/train | wc -l     # want roughly the same number
```

## 2. Create the droplet

In the AMD Developer Cloud console: **Create → GPU Droplet**, then

- **GPU:** 1x MI300X to start. Take 8x only if you have read step 7 and still
  want it — on this dataset the eighth GPU is usually worth less than the
  first one was.
- **Image:** the ROCm/PyTorch quick-start image. It ships Ubuntu 24.04 with
  the ROCm stack and a working PyTorch, which is the whole reason to use it.
- **SSH key:** add yours.

Then:

```bash
ssh root@<droplet-ip>
rocm-smi                           # 1 or 8 MI300X rows, ~192 GB each
```

If `rocm-smi` prints nothing useful, stop here — everything below assumes the
driver sees the card.

## 3. Get PyTorch that actually talks to the GPU

The quick-start image already has one. Confirm before installing anything:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want a `+rocm` or `+gitXXXXX` version, a non-`None` HIP version, `True`,
and `AMD Instinct MI300X`. `torch.cuda` is not a typo: PyTorch exposes ROCm
devices through the same `torch.cuda` API, which is why `train.py`'s existing
device detection needs no AMD-specific branch — and also why the log cannot
tell you which vendor's chip it found, so `train.py` now prints the name.

Starting from a bare Ubuntu droplet instead, run the container rather than
installing ROCm by hand:

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G --security-opt seccomp=unconfined \
  -v /root/frc:/workspace/frc -w /workspace/frc \
  rocm/pytorch:latest
```

`--shm-size` is the one people leave off. The default 64 MB is enough for
`--workers 2` and dies as `DataLoader worker (pid N) is killed by signal: Bus
error` at the worker count step 6 actually asks for.

### The failure this page exists for

```bash
pip install ultralytics          # <-- do not do this here
```

That resolves `torch` from PyPI, PyPI's `torch` is the CUDA build, and it
**overwrites the ROCm one**. Nothing errors. `torch.cuda.is_available()` turns
False, `train.py` falls back to `cpu`, and your 192 GB accelerator watches a
CPU do ninety epochs. Install Ultralytics without letting it touch torch:

```bash
pip install --no-deps ultralytics ultralytics-thop
pip install opencv-python-headless pyyaml tqdm matplotlib pandas psutil \
            py-cpuinfo scipy requests pillow
python3 -c "import torch; assert torch.cuda.is_available(), 'torch got clobbered'; print('ok')"
```

Run that assert again after *any* later `pip install`. `opencv-python-headless`
rather than `opencv-python` for the usual reason: the non-headless build wants
`libGL`, which a container has no reason to carry.

## 4. Upload the dataset and the code

```bash
# from your laptop
scp dataset.tgz root@<droplet-ip>:/root/frc/
./deploy/make_code_archive.sh && scp tbavid_code.tgz root@<droplet-ip>:/root/frc/
# on the droplet
cd /root/frc && tar -xzf tbavid_code.tgz && tar -xzf dataset.tgz
```

The code archive excludes `.env` by design, and training does not need a TBA
key — it reads `dataset/`, not the API.

## 5. Warm MIOpen once, deliberately

MIOpen picks convolution kernels by benchmarking them the first time it sees a
shape, so the first iterations are slow in a way that looks like the whole run
will be. Give it a cache that survives a container restart, and let a 2-epoch
run pay the cost while you are watching:

```bash
export MIOPEN_USER_DB_PATH=/root/frc/.miopen
export MIOPEN_CUSTOM_CACHE_DIR=/root/frc/.miopen
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
mkdir -p "$MIOPEN_USER_DB_PATH"

python3 train/train.py --data dataset/dataset.yaml --model yolo26n.pt \
    --epochs 2 --batch 16 --workers 16 --name warmup
```

Two epochs also tells you the per-epoch wall time, which is the only number
worth planning the real run around. `expandable_segments` is fragmentation
insurance; it matters at the batch sizes below.

## 6. Train

Single GPU, the scouting detector, with the settings this hardware unlocks:

```bash
python3 train/train.py \
    --data dataset/dataset.yaml \
    --model yolo26s.pt --p2 \
    --imgsz 1280 \
    --epochs 100 \
    --batch 32 \
    --workers 32 \
    --cache ram \
    --name fuel26_mi300x
```

What changed from the laptop defaults, and why each one:

- **`--batch 32`** (was 4). The M2 measurement is ~1.1 GB per image for
  `s`+P2 at 960, so 32 is an *estimated* ~50 GB of 192 — deliberately short of
  the ceiling. `--batch 0.70` lets Ultralytics fill 70% of memory instead, and
  `--batch -1` lets it guess; both are now accepted. Read step 7 before
  raising it.
- **`--imgsz 1280`** (was 960). Fuel is ~17 px, and the native frame is
  1920x504. Resolution is what a small-object problem actually wants, and on
  this card it is affordable in a way batch size is not useful.
- **`--workers 32`, `--cache ram`**. With this much compute, JPEG decoding
  becomes the bottleneck and the GPU idles at 30% while eight cores decode.
  `ram` caches the decoded set — about 1.4 GB per 1000 frames at this size, so
  under 10 GB of system RAM for the whole harvest.
- **`--p2`** stops being a considered tradeoff. It costs speed and memory, and
  here you have both.

Eight GPUs, if you took the 8x droplet:

```bash
python3 train/train.py --data dataset/dataset.yaml --model yolo26l.pt --p2 \
    --imgsz 1280 --batch 96 --workers 64 --cache ram \
    --device 0,1,2,3,4,5,6,7 --name fuel26_ddp
```

`--batch` is the **total** across GPUs in Ultralytics, not per-GPU: 96 is 12
each. DDP spawns one process per GPU, so `--workers` is per process.

## 7. Where the 192 GB is a trap

This dataset is ~1,100 training frames at 80 frames/match. At `--batch 96`
that is **11 optimizer steps per epoch**, and a run with 1,100 steps in it
converges worse than the laptop's 27,000 did, on hardware that cost more.
Ultralytics' default `lr0=0.01` is set for a nominal batch of 64 and is not
rescaled for you.

So spend the card on the things that do help:

1. **Resolution** — `--imgsz 1280` or 1920. A 17 px ball is the actual
   problem; more pixels on it is the actual fix.
2. **A bigger model with the P2 head** — `yolo26l.pt --p2`, which an 8 GB
   laptop cannot hold at all.
3. **More data**, which is upstream of here: `--per-match 80` exists because
   the M2 took 27 h otherwise. On this box you can afford 200, and 8-10 more
   matches pulled at home will move mAP further than any setting on this page.
4. **Eight experiments, not one job.** On the 8x droplet, DDP on 11 steps per
   epoch is close to pointless; eight independent runs sweeping imgsz, model
   scale and `lr0` is eight answers in the time of one:

   ```bash
   for i in 0 1 2 3 4 5 6 7; do
     HIP_VISIBLE_DEVICES=$i python3 train/train.py --device 0 \
         --data dataset/dataset.yaml --name sweep$i ... &
   done; wait
   ```

   `HIP_VISIBLE_DEVICES=$i` is what makes each process see one card as its
   device 0 — the ROCm spelling of `CUDA_VISIBLE_DEVICES`.

## 8. Get the weights off the box before you destroy it

```bash
# on the droplet
tar -czf weights.tgz runs/fuel26_mi300x/weights/best.pt \
    runs/fuel26_mi300x/results.csv runs/fuel26_mi300x/args.yaml
# from your laptop
scp root@<droplet-ip>:/root/frc/weights.tgz .
```

Take `results.csv` and `args.yaml` as well as the weights. Six weeks on, a
`best.pt` whose settings you cannot reconstruct is a model you cannot improve.

Then destroy the droplet in the console. It bills while it idles.

## 9. Benchmark on the field box, not on this one

```bash
python3 train/train.py ... --export onnx     # on the droplet, or after
python3 train/benchmark.py --weights runs/count26/weights/best.pt \
    --imgsz 960 --fps 30                     # ON THE SCRIMMAGE LAPTOP
```

The counting model's speed requirement is about the machine at the event, and
an MI300X number tells you nothing about a laptop in a gym — it will make a
model that cannot keep up look comfortable. ONNX export runs through the CPU
and works fine here, but exporting is all it is: Ultralytics ships no
MIGraphX runtime, so the ONNX you get is for the field box, and TensorRT is
not an option on AMD at all. Benchmark where it will run, as
`train/benchmark.py` has said all along.

## Troubleshooting

| symptom | cause |
| --- | --- |
| `device: cpu` on a GPU droplet | a PyPI wheel replaced ROCm torch. Re-read step 3; `train.py` now prints this loudly |
| `DataLoader worker killed by signal: Bus error` | container `--shm-size` too small for `--workers`. 16 GB, or drop the workers |
| first 50 iterations crawl, then speed up | MIOpen kernel search. Expected once; step 5 makes it once *ever* |
| loss goes NaN / mAP stuck at 0 | AMP. `--no-amp` — costs speed, not accuracy |
| "No non-empty label files" | dangling symlinks: the dataset was built without `--copy`. Step 1 |
| `HIP out of memory` | lower `--batch`, or `--batch 0.70` and let Ultralytics size it |
| mAP worse than the laptop run | almost certainly step 7: too few steps per epoch at a large batch |
| `run.py pull` gets nothing here | YouTube blocks datacenter IPs. Harvest at home |
