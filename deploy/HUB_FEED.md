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

Both AUTO winners came out right, by margins far wider than the error.

**A close AUTO: Einstein 1** (Newton vs Archimedes, from the same Drive
folder). Outlines moved by the hub shift template-matched from Einstein 4 on
a mid-match median (blue +16,+13, red +28,+5; the same method gives
Einstein 5's recorded blue shift exactly). One ball measured 229 px.

| t (s) | fuel counters: blue - red | counted: blue - red |
| --- | --- | --- |
| 26 (AUTO ends) | 85 - 80 | 155 - 177 |
| 29-30 (decision) | 95 - 96, red by 1 | 155 - 179, red by 24 |
| 172 (buzzer) | 621 - 415 | 754 - 766 |

The winner came out right by coincidence: it counted red ahead all through
AUTO while blue led at its end, and was 63% / 86% over at the decision. At
272 px (Einstein 4's ball) the decision reads 128 - 149, still red. After
AUTO it had red ahead from t=100 to the buzzer, when blue scored half again
as much. It still stayed flat through blue's inactive shift (t=60-95). **On
a close AUTO this counter's call is noise**, so a broadcast-angle camera
cannot be used for `counted` mode. The Einstein 5 excess is mostly in crossings of four or
more ball-areas, which net +402 red and +254 blue there against +114 and +54
on Einstein 4, and include single blobs of 20-28 balls; the crowd behind
Einstein 5's hoods has yellow and khaki shirts that Einstein 4's does not.
Capping the balls per crossing was tried and does not fix it: a cap of 3
brings Einstein 5 to 102% / 109% and drops Einstein 4 to 88% / 88%. Nothing
was changed. What that points at for the scrimmage is camera placement: an
outline whose search region sees no crowd, no yellow shirts, and no bumper.
Processing cost was 2.2 ms per 1080p frame on this VM's CPU.

### Two fixes tried on all three, and why neither helped (2026-09-27)

Both were chosen from a diagnosis before looking at the scores, with their
settings fixed in advance, and replayed over cached blobs from all three
matches (the replay reproduces the counts above exactly).

| variant | Einstein 4 buzzer | Einstein 5 | Einstein 1 | E1 decision (real 95 - 96) |
| --- | --- | --- | --- | --- |
| as shipped | 93% / 93% | 123% / 137% | 121% / 185% | 155 - 179 |
| blur-aware ball count | 56% / 47% | 62% / 55% | 39% / 65% | 69 - 79 |
| static yellow removed | 94% / 94% | 129% / 140% | 110% / 188% | 155 - 178 |
| both | 58% / 47% | 68% / 57% | 38% / 65% | 69 - 79 |

- **Blur.** A crossing blob's area grows with its speed (1.2x a resting ball
  when slow, 2x at 1.5 ball-widths per frame) while its width barely does,
  which looked like one ball smeared into a streak and counted as two.
  Dividing each blob by the area one ball smears to at its measured length
  along the motion halved every count, the well-counted Einstein 4
  included: the long blobs are mostly real trains of balls from drum
  shooters, not smears.
- **Static yellow** (pixels yellow more than half of a ~3 s average,
  removed before finding blobs). Crossings touching such pixels were 1% on
  Einstein 4 and 11-12% on 5 and 1, netting +224 and +467 -- but removing
  those pixels changed the totals by a few percent. They are mostly balls
  passing in front of balls resting on the hood, not shirts.
- **What the error actually is.** Net entries through the top of the red
  outline were 819 on Einstein 4 and 832 on Einstein 1, whose real red
  totals were 804 and 415. The same flow of balls down over the hood
  scored twice as often in one match as the other: from in front, a ball
  that clips the rim and drops behind the hub is indistinguishable from
  one that goes in. That is a camera-angle limit, and no change to the
  blob logic reaches it. The counter was left as shipped.

## Why this counter and not `run.py count`

The spec's budget is camera-to-count <= 80 ms typical, 200 ms p99, because
bioarena calls the AUTO winner at T+23.000 s from whatever counts it holds
then. `run.py count` holds every score 12 frames (400 ms at 30 fps) to see
whether the ball comes back out, and needs the detector's tracks, which lost
the balls on broadcast footage. The area counter needs no model and no GPU,
counts on the frame a ball crosses the funnel-mouth outline, and spends a few
ms per frame on a laptop CPU.

## The easy way: the web page

```bash
.venv/bin/python run.py hubgui --setup cams.json            # opens in your browser
```

On a Mac, double-clicking `hubfeed.command` does the same, installing what
it needs the first time. The page needs nothing but Python, OpenCV and (for
streams) yt-dlp; its logic lives in `tbavid/hubapp.py`. Work down the
right-hand side:

1. **Cameras.** *Find cameras* lists every camera that answers, with its
   size; pick one. *Add recording…* adds a video file instead, to set up or
   practise on (its picture is taken a third of the way in, past title
   cards, or at the second you type). Camera numbers can change when cameras
   are re-plugged, so always check the picture.
2. **Hub outlines.** *Draw RED* / *Draw BLUE*, then click the corners of the
   hub's opening on the picture; double-click, right-click or Enter to
   finish, Esc to cancel. A camera can have several outlines; choose how each
   hub combines them (sum / max / median, see below).
3. **Measure ball** with a few balls sitting apart near the hub (5 s).
4. **Calibrate (optional)**: pick a recording from this camera, type how
   many balls you counted going into each hub, and it tries every blur
   correction with and without ignoring still yellow, and offers the setting
   nearest your count.
5. **START.** The picture turns live, with each outline's count on it; the
   big boxes are what bioarena is being sent; the line beside them says
   whether bioarena is answering and each camera's frame rate, and turns red
   with *NO PICTURE* if a camera stops. Tick *Practice: send to a test
   receiver here* to try it all without bioarena.

**Wireless cameras.** Three ways:

- *iPhone + Mac*: Continuity Camera makes the iPhone a camera with no cable;
  it shows up under *Find cameras* like a USB one. Nothing else to set up.
- *Android phone*: an IP-camera app (e.g. IP Webcam) shows an address like
  `http://192.168.1.60:8080/video`.
- *Wi-Fi IP camera*: its RTSP address, e.g.
  `rtsp://user:pass@192.168.1.50:554/stream1`.

*Wireless camera* in the website takes the address (in `cams.json` it is
just the `"source"`). It is opened over TCP with FFmpeg's input buffering
off, so frames are not queued. If the Wi-Fi drops, the heartbeat stops
(bioarena shows OFFLINE) and the counter keeps reconnecting until the camera
is back, then forgets where the balls were before the gap so it cannot
invent a crossing. Passwords in an address are kept out of the log.

Wi-Fi costs delay and can drop out: check the frame rate in the status card
holds before trusting it for AUTO, and prefer a cable when you can. Use your
own router, never the field's Wi-Fi, and check the event's rules first:
FRC events prohibit teams running their own Wi-Fi networks.

Tested with a stand-in phone camera (MJPEG over HTTP, 30 fps): picture in
1.3-1.8 s, counting at 30 fps, and a simulated drop -- the camera killed for
4 s -- showed NO PICTURE, held the heartbeat, reconnected and resumed at
30 fps without stopping the session. Not yet tried with a real phone or IP
camera, or an RTSP one.

**Twitch / YouTube streams.** *Add stream* takes a live address
(`https://www.twitch.tv/<channel>`, a YouTube live URL, or a Twitch past
broadcast `https://www.twitch.tv/videos/<id>`), looks up the video behind it
with yt-dlp, and counts from it like a camera; a dropped stream reconnects up
to five times. A stream runs several seconds behind the field -- Twitch's
delay, nothing here can remove it -- so it is for practice and scouting, never
for bioarena's AUTO call; the status turns amber while one is counting. A past
broadcast plays at its own speed. Needs `pip install yt-dlp` (the Mac launcher
installs it). In a setup file a stream is just `"source":
"https://www.twitch.tv/<channel>"`.

*Save* writes the same `cams.json` that `run.py hubfeed --setup` reads, so a
setup made here also runs without it.

The web page listens on 127.0.0.1 only and refuses other host names and
non-JSON posts, because it can start cameras and list the disk. `--bind
0.0.0.0` opens it to the network -- then anyone who can reach the laptop can
reconfigure the counter.

## Count at the exits (recommended)

Every ball that scores comes back out of the hub, so the number leaving
through the exits is the score. Counting there removes the error that sank
the funnel-mouth counts: from in front, a ball that clips the rim and drops
behind the hub looks exactly like one that went in (Einstein 1: 832 net
entries over the red hood against 415 real), but it never reaches an exit.

In the website, **Red exit** / **Blue exit**, then three clicks: the two ends
of a line across the exit, then a point on the side the balls go out to. An
arrow shows the counting direction. A ball counts when its path between two
frames crosses the line outward; one that comes back across is taken off. A
clump crossing together counts by its area, as elsewhere. In `cams.json`:

```json
{"hub": "red", "line": [[820, 410], [820, 520]], "out": [900, 465]}
```

Put the camera where it sees each exit squarely, close enough that balls are
large, with nothing yellow beyond the line. Several exits per hub: one line
each, combined by `sum`.

**Not measured yet.** The broadcast cannot test it: the Einstein exits are
mostly behind the hubs, and a line across the visible part of the red
out-flow caught 9% (Einstein 4), 15% (5) and 34% (1) of the scoreboard. The
test is the spec's 20 balls through a practice hub's exits.

## Blur correction and still-yellow removal (per camera)

Two settings per camera, both off by default:

- `"blur"` (0-1): how much of a fast ball's smear to discount. Tested on
  three Einstein matches, the best value was different on each (0 on
  Einstein 4, 0.2-0.3 on 5, 0.5-0.7 on 1), and a value chosen on two matches
  did no better than 0 on the third. So it is not a fix, it is a
  calibration: set it from a hand-counted recording made by that camera in
  its real position, and check it on a second recording.
- `"remove_static"`: ignore pixels that have been yellow for most of the
  last ~3 s (resting balls, shirts). It moved the Einstein totals a few
  percent.

Calibrate in the window, or:

```bash
run.py hubfeed --setup cams.json --calibrate practice.mp4 --camera red-exit --count red=23
```

Neither setting can fix what the Einstein tests found the real error to be:
from in front, a ball that clips the rim and drops behind the hub looks like
one that went in. That needs the camera somewhere it can see the difference.

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

## Several cameras (`--setup`)

The Einstein tests say one camera in front of a hub cannot tell a ball that
clips the rim and drops behind it from one that goes in. More cameras,
closer, is the way to accuracy, so `--setup cams.json` takes any number of
cameras, each with its own source, frame rate, size and **ball area** (a
camera a metre from a chute and one across the field see very different
balls), and each with one or more outlines ("zones") counting into a hub.
`deploy/hubfeed.example.json` is a complete example; a relative video path
is found next to the setup file.

How a hub combines its zones is chosen per hub under `"combine"`:

| rule | when | example |
| --- | --- | --- |
| `sum` (default) | zones see **different** balls | one camera per exit chute, or per side of the hub |
| `max` | zones see the **same** balls; the one that missed fewest wins | two angles on one mouth |
| `median` | three or more see the same balls; the odd one out is outvoted | three angles, one of which glare fools |

`max` assumes cameras miss balls rather than invent them; if one over-counts
(the Einstein failure), `max` follows it and `median` does not. Every rule
keeps the count from ever going down, which the feed requires.

Commissioning is per camera:

```bash
run.py hubfeed --setup cams.json --measure 5 --still still.png
```

writes `still_<camera>.png` for each camera and prints each camera's
`"ball_area"` to paste into the file. Then run it with `--setup cams.json`
and the usual `--target` / `--log`. The status line shows each hub's
combined count and every zone's own count; the CSV log has one row per zone
change with the camera, zone, zone count and hub count, which is what to
read when two cameras disagree.

If **any** camera stops delivering frames the heartbeat stops and bioarena
shows OFFLINE, even under `max`/`median` where the others could carry on: a
half-watched hub must not look fully watched. Remove that camera from the
file and restart if the match has to go on. Each camera is its own thread
decoding its own frames, so check the frame rate in the status line with
every camera running -- a laptop that keeps up with one 60 fps camera may
not keep up with four. USB cameras on one hub or port can also run out of
bandwidth; lower `size` or `fps` if a camera's rate drops.

The single-camera flags (`--source --red --blue --red-source
--blue-source`) still work and are the same as a setup file with one or two
cameras.

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
