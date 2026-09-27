# Hub FUEL counter feed to bioarena (scrimmage, 2026-10-10)

`run.py hubfeed` watches the hubs through a camera, counts fuel going into
each one, and sends both running totals to bioarena over UDP, following
Team 841's "Spec: Hub FUEL Counter Feed" (sections 4 and 5). bioarena does
the match baseline, shifts, AUTO winner and screens; this side only counts
and sends.

**Status: plumbing tested, counting not validated.** The protocol side is
covered by `tests/test_pipeline.py` against bioarena's acceptance rules, and
a synthetic video has been run end to end over loopback. The counting method
(`tbavid/hubcount.py`) is the area-crossing counter from
`experiments/area_hub_count.py`: 93-95% on the broadcast it was tuned on,
120-137% blind on the next match. It has never seen a practice-field camera.
Until the 20-ball acceptance test below passes, leave bioarena's AUTO winner
mode on `random`/`red`/`blue`, not `counted`, and keep a human scorekeeper.

## Measured on the Einstein broadcasts (2026-09-27)

`tbavid/hubcount.py` was run over the same Einstein 4 and 5 broadcasts as the
experiment (6-182 s, the experiment's outlines, one-ball area 272 / 300 px)
and compared against the broadcast's fuel counters in the corner boxes.
Those are fuel; the centre numbers are points and differ on Einstein 5.

The port is the experiment: on Einstein 4 its signed totals are identical
(blue 444, red 769). On Einstein 5 they are 721 / 937 against 712 / 932,
because the size floor, search margin and match reach scale with the 300 px
ball instead of staying at the 272 px values. The high-water mark the feed
reports was never more than 25 above the signed count.

| | AUTO decision (t=29): truth | counted | buzzer (t=172): truth | counted |
| --- | --- | --- | --- | --- |
| Einstein 4 blue | 91 | 81 | 479 | 445 (93%) |
| Einstein 4 red | 193 | 135 | 804 | 750 (93%) |
| Einstein 5 blue | 90 | 94 | 585 | 717 (123%) |
| Einstein 5 red | 161 | 228 | 669 | 918 (137%) |

Both AUTO winners came out right, by margins far wider than the error; a
close AUTO would not. The Einstein 5 excess is mostly in crossings of four or
more ball-areas, which net +402 red and +254 blue there against +114 and +54
on Einstein 4, and include single blobs of 20-28 balls; the crowd behind
Einstein 5's hoods has yellow and khaki shirts that Einstein 4's does not.
Capping the balls per crossing was tried and does not fix it: a cap of 3
brings Einstein 5 to 102% / 109% and drops Einstein 4 to 88% / 88%. Nothing
was changed. What that points at for the scrimmage is camera placement: an
outline whose search region sees no crowd, no yellow shirts, and no bumper.
Processing cost was 2.2 ms per 1080p frame on this VM's CPU.

## Why this counter and not `run.py count`

The spec's budget is camera-to-count <= 80 ms typical, 200 ms p99, because
bioarena calls the AUTO winner at T+23.000 s from whatever counts it holds
then. `run.py count` holds every score 12 frames (400 ms at 30 fps) to see
whether the ball comes back out, and needs the detector's tracks, which lost
the balls on broadcast footage. The area counter needs no model and no GPU,
counts on the frame a ball crosses the funnel-mouth outline, and spends a few
ms per frame on a laptop CPU.

## What you need

- A laptop wired into the field switch on the management VLAN, static
  `10.0.100.21/24` (the spec's suggestion, beside the e-stop panels at
  `.11`/`.12`). Not the field Wi-Fi: bioarena drops datagrams from team VLANs.
  bioarena's settings page needs the same address as `HubCounterAddress`.
- A webcam fixed where it sees each hub's funnel mouth. One camera for both
  hubs is fine; a camera per hub (closer, bigger balls) is better and is
  supported with `--red-source` / `--blue-source`. 60 fps if the camera does
  it (`--cam-fps 60`): it halves the wait for the next frame.
- Python with numpy and OpenCV:

  ```bash
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt opencv-python-headless
  ```

  macOS asks for camera permission for the terminal the first time.

## 1. Draw the outlines (once per camera position)

```bash
.venv/bin/python run.py hubfeed --source 0 --measure 1 --still still.png
```

Open `still.png` and read off a polygon around each hub's funnel mouth --
the clear hood and the opening, **bottom edge on the solid front rim** -- as
`x,y` pairs in the image's own pixels. A ball counts when its centre crosses
into that outline and is taken back when it crosses out, so the outline must
be somewhere a scored ball goes in and disappears, and a ball flying past
does not stop.

## 2. Measure one ball (once per camera position)

Put several balls near each hub, apart from each other, and:

```bash
.venv/bin/python run.py hubfeed --source 0 --measure 5 --still check.png \
    --red X,Y,X,Y,... --blue X,Y,X,Y,...
```

It prints `one ball = N px ... Pass --ball-area N`. Every crossing is divided
by this to turn a clump into a number of balls; on the broadcast, 3% on this
number moved the totals 5%. `check.png` shows the outlines drawn on the frame.

## 3. Check the link without bioarena

On any second machine (or the same one, with `--target 127.0.0.1:8411`):

```bash
python3 run.py hubfeed-listen --counter 10.0.100.21
```

It applies bioarena's acceptance rules, prints every count as bioarena would
receive it, and replies like bioarena, so the counter shows a link.

## 4. Run it

```bash
.venv/bin/python run.py hubfeed --source 0 --cam-fps 60 --ball-area N \
    --red X,Y,... --blue X,Y,... --log hubfeed.csv
```

Every 5 s it prints each hub's total with entries and exits, the camera's
frame rate and capture-to-count lag, and what bioarena last said (match
state, shift, round trip). `owed` is balls reported that have since come
back out of the outline; the next ball in is absorbed against them, because
the feed may never go down. A half field: give only the one hub; the other
reads 0.

Leave it running across matches; it never needs resetting. If it restarts,
bioarena keeps the match's score and loses only the balls scored while it was
down. If the camera stops delivering frames it stops the heartbeat, so
bioarena shows OFFLINE instead of frozen counts behind an ONLINE badge.

## 5. Validate before trusting it

In this order, and write the numbers down:

1. **Record a practice hub** from the scrimmage camera position with balls
   going in the way robots shoot them (singles, bursts, a few off the hood):
   `ffmpeg -f avfoundation -framerate 60 -i 0 hub.mp4` on a Mac.
   Hand-count the balls.
2. **Replay it** as a camera would deliver it and compare:
   `run.py hubfeed --source hub.mp4 --realtime --target 127.0.0.1:8411 ...`
   with `hubfeed-listen` running. Adjust the outline, not new thresholds.
3. **Spec acceptance on the field**: twenty balls by hand-count into each
   hub against the bioarena tally. Zero disagreement, including which hub.
4. Check the latency figure on bioarena's match panel. `age_ms` is measured
   from the capture timestamp on Linux (V4L2 driver stamp). On macOS the
   driver stamp is not usable, so it runs from when the frame was read and
   leaves out any time spent in the camera's buffer.

Only after 3 passes is `counted` mode worth turning on.

## Known limits

- A ball that flies across the outline, or bounces off the hood and back out,
  is reported for a moment until the next ball in absorbs it. bioarena credits
  it to the shift it happened in.
- Crossings are matched frame to frame by position; a ball moving more than
  about 1.5 of its own widths per frame beyond its predicted path is lost.
  A higher frame rate helps.
- It counts balls physically entering the hub. Active/inactive shifts are
  bioarena's; a ball into a dark hub is counted here and not credited there.
- Yellow is the only test. A yellow shirt or robot passing through the
  outline's search region will count; place the camera so none can.
