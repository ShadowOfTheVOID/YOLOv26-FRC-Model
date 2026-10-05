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

**bioarena cannot receive this feed yet (checked 2026-10-02).** bioarena
`main` (f4987b0, 2026-09-26) and its other branches have no receiver on 8411
and no Counted mode. It assigns the AUTO winner when AUTO starts, either at
random or forced red/blue (`assignAutoWinner`, `field/arena.go`), and cannot
change it later. The feed implements the draft spec and is tested against
`run.py hubfeed-listen`; it starts mattering once bioarena ships the
receiver. Until then, use the counts through frc-fms (deploy/FRC_FMS.md) or
for scouting, and set bioarena's AUTO winner by hand. frc-fms reached the
same conclusion (its commit 20524ce) and has its own feed to the same spec.
Run one feed per field, not both: each is its own session, and the spec
receiver accepts one source address.

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

**On a Mac, use Hub Counter.app** (`apps/hubcounter/README.md`). Download
`HubCounter-mac.zip` from the release, unzip it, and double-click. It opens
this page in your browser with nothing to install and no terminal. The
setup and count logs go to `Documents/Hub Counter/`. Stop it with the
page's **Quit** button. It counts by colour; the fuel-model blend still
needs the terminal version below.

```bash
.venv/bin/python run.py hubgui --setup cams.json            # opens in your browser
```

Double-clicking `hubfeed.command` does the same from a checkout, installing
what it needs the first time. The page needs nothing but Python, OpenCV and (for
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

## The live scoreboard (`/board`)

**At the scrimmage, bioarena's own match panel and audience display are the
scoreboard.** The counter feeds them, and bioarena applies the match clock,
the shifts and the active and inactive hubs.

`http://127.0.0.1:8790/board`, the **Scoreboard ↗** button on the page, is a
full-screen red and blue scoreboard. Use it for practice, for commissioning,
or as a second screen. It refreshes ten times a second. What it shows
depends on the link:

- **Linked to bioarena:** bioarena's `credited` count, the score. It adds
  the match phase and clock, AUTO counts, and a dimmed "HUB INACTIVE" half
  while a hub is dark (all from bioarena's status reply, spec 4.4).
- **Practice mode:** the same, from the built-in test receiver, labelled
  "Test receiver (practice)".
- **Not linked:** the camera counts since the counter started, labelled as
  such. These are not a match score: nothing here knows when a match
  starts, and fuel scored into a dark hub is included. **Zero** starts them
  from 0 on that screen only; the feed is untouched.

A red strip appears when:
- the counter is stopped;
- a camera has given no picture for 0.5 s;
- the page cannot reach the counter (after 1 s; the numbers are then
  frozen).

Press **F** for full screen.

**Measured 2026-09-29:** the Einstein 4 recording was fed through the whole
path in practice mode, paced like a 60 fps camera. The counter kept 59.9
fps, processing a frame in 5 ms median and 14 ms worst (capture-to-counted,
so camera and USB delay are not included). The spec budget is 200 ms.

To put it on a TV driven by another machine, run with `--bind 0.0.0.0` and
open `http://<laptop>:8790/board`. The warning above applies: the set-up
page is then open too.

## Counting recordings for scouting (`run.py hubcount`)

The same counter runs in two ways:
- **Live on the M4 Max at the scrimmage:** `hubfeed.command` or `run.py
  hubgui` feeds bioarena.
- **On recordings afterwards, for scouting:**

```bash
run.py hubcount match_q12.mp4 match_q13.mp4 --setup cams.json [--camera NAME]
# match_q12: red 650, blue 515 over 214.5 s (12856 frames at 261 fps, 4.4x real time)
#   -> match_q12_hubcount.csv   (video_s, red, blue every 0.5 s)
```

How it counts:
- Every frame is decoded; skipping frames would miss balls crossing
  between them.
- It runs as fast as the machine decodes, with no pacing and nothing sent.
- The zones and rules are those of the live counter.

Draw the outlines on the recording itself in `run.py hubgui` (Add
recording) and save them. A recording from another camera position needs
its own outlines and ball size.

The CSV is against **video time**, so the score at any moment of the match
can be read off it. The totals include whatever the video shows before the
start and after the buzzer, replays included. Read the timeline at the
match's start and end, or pass `--start` / `--end`.

**Measured 2026-09-29** on the Einstein 4 broadcast:
- **Speed:** 4.4x real time on a 4-core container, so the M4 Max should be
  faster.
- **Counts:** 650 / 515 red / blue at the buzzer, the same as the
  evaluation above (scoreboard 804 / 479).

**Older setup files:** before these rules the page saved every camera's blur
slider, 0 unless moved, which pins the setting that measured 20-34%. A
setup file without `"rules": 2` has its blur 0 read as 0.3, with a note
printed. Save it once from the page and it keeps what it has from then on.

## Count at the exits (untested; failed on both broadcasts)

**Use a raised outline instead** (section 1, "Draw the outlines"). On the
2026 Central Valley broadcast (2026-10-05), whose exits are in view, exit
lines got 25-32 of blue's 159 and 52-100 of red's 810: 82-85% error. The
raised outline got 166 / 808 (6.4%) on the same match. The balls do not
leave one at a time: they pour into a pile against the hub, so new ones push
into the heap instead of crossing the line as separate blobs. Referees and
robots also stand at the exit. Red's exit count was still under 20 at
141 s, against 634. The idea below still holds for a camera aimed right
at the chute, where balls cross before the pile. Check that with 20 balls
first.

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

## Every fix together, tuned on all three Einstein matches (2026-09-29)

About 850 configurations were swept on Einstein 4, 5 and 1:
- The mouth counter over ball size x0.8-1.4, blur 0-0.5, still-yellow
  removal, speck floor x0.5-2 and matching reach x0.75-1.5.
- The model counter (`fuel_best.pt` -> `trackvis.BallTracker` ->
  `count.BallCounter`) with and without colour assist, over the track,
  vanish, reacquire, entry and padding settings.
- The exit lines.
- Their combinations: mean, min, max, and the median of mouth, model and
  exit.

Error is the mean |count / scoreboard - 1| over every checkpoint from the
end of AUTO to the buzzer, both hubs. "Held out" means every setting was
chosen on the other two matches and then scored on this one. This is the
only honest number with three matches: the best fit on all three is
tuned to the answer.

| method | best fit (E4 / E5 / E1) | held out E4 / E5 / E1 | held-out mean |
|---|---|---|---|
| mouth, settings before this sweep | 12% / 31% / 69% | -- | -- |
| mouth, tuned | 11% / 7% / 33% | 43% / 7% / 39% | 30% |
| exit lines | -- | -- | 76% |
| model + BallTracker + BallCounter, no assist | 53% / 23% / 14% | -- | 36% |
| model + BallTracker + colour assist | 19% / 8% / 4% | 25% / 8% / 13% | 16% |
| **mean(mouth, model + assist)** | -- | 17% / 6% / 16% | **13%** |
| min(mouth, model + assist) | 8% / 5% / 5% | 41% / 5% / 5% | 17% |
| max(mouth, model + assist) | -- | 23% / 7% / 39% | 23% |
| median(mouth, model, exit) | -- | 41% / 5% / 6% | 17% |

Held-out buzzer counts, blue / red against the scoreboard
(E4 479 / 804, E5 585 / 669, E1 621 / 415):

| method | E4 | E5 | E1 |
|---|---|---|---|
| model + assist | 617 / 686 | 655 / 693 | 485 / 346 |
| mean(mouth, model) | 457 / 630 | 567 / 586 | 506 / 430 |

**The AUTO winner at T+23 s was right in every held-out run of every
method.** That includes Einstein 1's 95-96 AUTO.

What changed in the code:
- `hubcount.MIN_AREA_FRAC` doubled (the speck floor, 40 -> 80 px at the
  272 px reference ball).
- `MIN_REACH_BALLS` and the per-blob reach grew 1.5x (`REACH_PER_BLOB`
  2.25).

Two of the three held-out folds chose exactly these values.

The shipped `CrossingCounter` rerun on the recorded blobs gives:

| settings | E4 | E5 | E1 |
|---|---|---|---|
| camera defaults | 7% | 34% | 71% |
| ball area x0.9, `"blur": 0.3`, `"remove_static": true` | 8% | 7% | 39% |

Those are the settings to start a broadcast-like camera on before
calibrating.

What this does not fix:

- **Three matches tune too easily.** A spread of 5-43% between folds
  means the next match can land anywhere in that range. The "held out"
  column is the one to quote.
- **The model counter is offline only.** `run.py hubfeed` still runs the
  mouth counter alone. The model counter needs `fuel_best.pt` at ~960 px,
  about 18 fps on the M4 Max against 60 fps video. `BallCounter` also
  holds a ball ~400 ms before counting it, which breaks the AUTO budget.
  The combination is the better scorekeeper after the fact, not the live
  feed.
- **Exit lines stay unmeasurable on broadcasts.** The exits are behind the
  hubs, 9-34% visible. Their 76% error is the camera angle, not the
  counter. (Later, Central Valley's exits were in view and still gave
  82-85%: the balls pile up at the exit. See "Count at the exits".)

## Downward entries and a learned ball (2026-09-29, the current rules)

The sweep above left the mouth counter at 30% held out. Two findings from
the recorded Einstein blobs led to a better counter:

- **Balls crossing the mouth are bigger than the measured ball.** They were
  1.5-2.3x the still ball, and by a different ratio on each match. In
  Einstein 1's AUTO the blue outline saw 85 crossings for 85 real balls,
  but 56 of them were counted as two balls.
- **Outward crossings are not scores coming back out.** On Einstein 1 blue,
  60-100 s, there were 161 entries and 171 exits against 108 real balls.

So an outline now:
- counts only blobs moving **down** into it;
- ignores outward crossings;
- learns one ball as the 30th percentile of its last 80 crossing blobs
  (after 10);
- rounds a blob up to the next ball only at 0.65 of a ball;
- uses blur 0.3 when a camera doesn't set one.

About 1,900 more configurations were swept, scored the same way as above:

| rules | held out E4 / E5 / E1 | held-out mean |
|---|---|---|
| mouth, previous rules, tuned | 43% / 7% / 39% | 30% |
| + gates (direction, exits, clump cap, speed) | 19-20% / 10-13% / 39% | 23-24% |
| + outline size and position, rounding | 16% / 18-20% / 48% | 27-28% (overfit) |
| **learned ball, downward entries, no exits** | 18% / 7% / 14% | **13%** |

The shipped defaults, with no per-match setting except the measured ball,
on the recorded blobs:

| match | error | buzzer, blue / red | AUTO at the decision (real) |
|---|---|---|---|
| E4 | 16% | 518 / 636 (108% / 79%) | 86-133 (91-193), right |
| E5 | 6% | 603 / 630 (103% / 94%) | 78-135 (90-161), right |
| E1 | 9% | 614 / 487 (99% / 117%) | 101-104 (95-96), right |

The model counter's 16% was held out the same way, so the live mouth
counter now matches it with no model and no GPU.

Neighbouring settings all land within a few points of each other:
- percentile 20-35;
- memory 30-400;
- warm-up 5-10;
- still-yellow removal on or off.

The exceptions:
- **blur 0** measured 20% / 30% / 34%, hence the 0.3 default;
- **warm-up 50** left E1's AUTO on the measured ball and cost 14 points.

What it does not fix:
- **Einstein 4's red AUTO** reads 133 against 193. From 26 to 29 s the
  scoreboard rose 36 with not one crossing at the mouth, so the scoreboard
  lags the balls. No lag fitted all three matches (E4 was best at 0 s, E1
  at 6 s), so none is applied.
- **"Down" means down the picture.** A camera mounted upside down or on
  its side needs its picture turned first.
- **Exit lines are unchanged.** They are signed, use the measured ball and
  are unvalidated.
- **The learned ball assumes most crossings are single balls.** On the
  broadcasts the 30th percentile was one ball. A camera where nearly every
  crossing is a train of drum-fed balls would learn the train as one ball
  and count low. The hand-count calibration catches that. Build a counter
  with `learn=False` if it happens.
- **It is still three broadcasts.** The 20-ball acceptance test on a
  practice hub comes first.

## How forgiving the setup is (2026-09-29)

These tests used the current rules on the recorded Einstein blobs: mean
error over E4 / E5 / E1, against 10.5% as tuned.

| setup error | mean error |
|---|---|
| outline drawn 5-10 px high or low | 8.7-11.6% |
| outline drawn 7-15% too small or too big | 10.4-11.9% |
| one ball measured at 0.5x / 0.7x / 0.85x | 15.4% / 12.6% / 11.8% |
| one ball measured at 1.15x / 1.3x / 1.6x / 2x | 10.3% / 9.7% / 9.5% / 9.7% |
| **camera at 30 fps** (every 2nd frame) | **12.6%** |
| **camera at 20 fps** (every 3rd frame) | **25.1%** |

What this means for setting up:
- **Outlines** can be drawn by hand.
- **Ball size:** a measurement that is off costs little because the counter
  learns one ball from the crossings. If in doubt, round it up.
- **Frame rate** is the one that matters. Under 30 fps, balls jump too far
  between frames to be followed across the mouth. A webcam drops to 15-24
  fps by itself in dim light, so the page and `/board` now warn when a live
  camera runs under 28 fps (`hubcount.MIN_FPS`). Light the hub, or lower
  the resolution, before the match.

Einstein 1's AUTO was 95-96, so a single ball decides it. Its AUTO winner
flipped under a 5-10 px outline shift. No counter can call a one-ball AUTO
reliably, and bioarena's Counted mode should not be trusted for one.

**Tuning has stopped improving.** One more held-out sweep tried:
- counting a track only once;
- a minimum downward speed;
- learning the ball only from entries;
- percentile 20-40, rounding 0.25-0.5 and blur 0.2-0.4.

It came to 14% held out against 13% for the current rules, and the best
fit improved by only 1.5 points. The settings are left as they are: on
three matches, more tuning is fitting noise. Better accuracy now needs more
matches with known scores. YouTube refuses downloads from the build
server, so further Einstein matches have to come through Drive.

## A truly held-out match: Einstein 8 (2026-09-29)

Einstein 8 (Daly vs Curie) arrived after every setting above was chosen.
Setup:
- **Outlines:** Einstein 4's, moved +1 px. Template-matching the hubs
  against Einstein 4 gave a match score of 0.87; the same method reproduces
  Einstein 5's and 1's recorded shifts to within a few pixels.
- **Ball:** measured by the app's own method (328 px).
- **Run:** `count_recording`, the shipped `run.py hubcount` path, with
  nothing tuned.

| video s | 30 (AUTO call) | 61 | 101 | 141 | 173 (buzzer) |
|---|---|---|---|---|---|
| blue counted / real | 139 / 181 | 280 / 318 | 346 / 382 | 438 / 482 | 591 / 653 (91%) |
| red counted / real | 183 / 194 | 218 / 228 | 342 / 397 | 399 / 438 | 549 / 581 (94%) |

**Result: 9.1% error, and the AUTO winner right.**
- The ball on the tuning scale (266 px) gives 8.8%.
- It ran at 241 fps.
- While the blue hub was inactive (121-141 s), the counter held flat, as
  the scoreboard did.
- Blue ran about 10% low throughout, mostly in AUTO: 22-27 s gained 57
  real balls against 25 counted.

With four scored matches the tuning was re-run, each match held out once:

| | E4 | E5 | E1 | E8 | mean |
|---|---|---|---|---|---|
| current settings (chosen on E4, E5, E1) | 16.2% | 6.2% | 9.1% | **8.8%** | 10.1% |
| best fit on all four (percentile 15, blur 0.5, ...) | 15.7% | 7.3% | 7.0% | 6.8% | 9.2% |
| re-tuned on three, scored on the fourth | 16% | 7% | 27% | 12% | 21% |

Re-tuning is worse held out than leaving the settings alone. Among 452
combinations, the best on three matches is usually one that breaks on the
fourth. The best fit on all four gains less than a point. **The settings
are unchanged.** Einstein 8 shows they carry over to a match they were
not tuned on.

## Every frame at 60 fps, both counters (2026-09-29)

The model counter was earlier run on every 2nd frame to save time
(`fuel_best.pt` -> `trackvis.BallTracker` -> `count.BallCounter`). It was
rerun on every frame of all four matches: 40,000 frames, 2 hours of CPU.
Each match was held out once. The 30 fps rows use every 2nd frame of the
same run.

| counter | held-out E4 / E5 / E1 / E8 | mean | AUTO winner wrong |
|---|---|---|---|
| **colour counter, 60 fps (shipped, not re-tuned)** | 16.2 / 6.2 / 9.1 / 8.8% | **10.1%** | none |
| colour counter, 30 fps | 16.9 / 11.8 / 8.9 / 6.6% | 11.1% | none |
| model + colour assist, 30 fps | 22 / 8 / 8 / 14% | 17% | E8 |
| model + colour assist, 60 fps | 24 / 14 / 16 / 21% | 25% | E1 |
| model, no assist, 30 fps | 55 / 23 / 13 / 17% | 36% | E4, E1, E8 |
| model, no assist, 60 fps | 66 / 25 / 22 / 18% | 44% | E4, E1, E8 |
| mean(colour 60 fps, model 60 fps) | 20.5 / 9.1 / 11.2 / 16.4% | 14.3% | none |
| lower of the two, 60 fps | 16.2 / 6.2 / 9.1 / 20.7% | 13.1% | none |

The 60 fps model rows include the tracker's hold at both 30 and 60 frames,
since it is counted in frames and 60 fps halves it in time; the counter's
windows already scale with fps.

What this settles:
- **The colour counter at 60 fps is the best counter here**, and it is the
  one that runs live. It gains a point over 30 fps.
- **The model counter gets worse at 60 fps**, 17% to 25%. Nothing was
  tried to find out why. A likely cause is that the colour assist compares
  each frame with the one 2 frames earlier, only 33 ms at 60 fps, so
  flying balls barely move and fewer are picked up.
- **Adding the model makes the result worse:** 13-14% combined, against
  10.1% for the colour counter alone. It stays offline; there is no reason
  to add it to the live path.

## The model on full-resolution hub crops (2026-09-29)

At `imgsz 960` a 1080p frame is halved before the model sees it, so an
18 px ball reaches it as about 9 px. Instead, a 640x640 window around each
hub was cut at full resolution and passed at `imgsz 640`: two crops, about
the same cost as one 960 frame (227 ms against 205 ms on 4 CPU cores). On
60 Einstein 4 frames the model found 93% of the moving ball-sized yellow
blobs near the hubs, against 84% on the halved frame.

For counting, all four matches were run at 30 fps and each held out once:

| model counter | E4 / E5 / E1 / E8 | mean |
|---|---|---|
| full frame at 960, no colour assist | 55 / 23 / 13 / 17% | 36% |
| **hub crops, no colour assist** | 17 / 15 / 18 / 14% | **21%** |
| full frame at 960, colour assist | 22 / 8 / 8 / 14% | 17% |
| hub crops, colour assist | 9 / 17 / 15 / 10% | 17% |
| colour counter alone (shipped) | 16.2 / 6.2 / 9.1 / 8.8% | 10.1% |
| **mean(colour counter, hub crops + assist)** | 12.6 / 5.7 / 9.3 / 8.3% | **9.0%** |

What this shows:
- **The crops make the model a better detector.** Without colour assist
  its error fell from 36% to 21%; the assist had been hiding how many
  balls the halved frame missed.
- **It is still not a better counter on its own:** 17% either way with the
  assist, and the AUTO winner was wrong on Einstein 8 in both.
- **The mean of the colour counter and the crop model is the first
  combination to beat the colour counter:** 9.0% against 10.1%, with the
  AUTO winner right on all four. That is one point on four matches, too
  small to call settled.

The rest of the gap is in training: the labeller's field line left balls
high against the crowd unlabelled, and the model has never seen an Einstein
frame. Retraining with balls in flight labelled (`train/README.md`, the
bootstrap) is the next step for the model.

## The combo, tuned: colour + the retrained model (2026-10-02)

`fuel_relabel.pt`, retrained on the MI300X with balls in flight labelled, is
blended into the colour count: each outline's count is
`round(w * model + (1 - w) * colour)`, both halves never going down, so
neither does the blend. The colour counter over-counts (balls that clip the
rim and drop behind the hub look like scores) and the model under-counts
(the tracker drops balls fired in streams), so the two errors partly cancel.

```bash
# recordings (scouting / validation); needs the detect environment
.venv-train/bin/python run.py hubcount match.mp4 --setup cams.json --model fuel_relabel.pt
#   match.mp4: red 461, blue 573 ...
#     cam/blue0: 573 = colour 614 blended with model 532
# live, same flag
.venv-train/bin/python run.py hubfeed --setup cams.json --model fuel_relabel.pt --device mps
```

or per camera in cams.json: `"model": {"weights": "fuel_relabel.pt",
"weight": 0.5}` (also `device`, and the counter settings below by name).
Exit lines stay colour only.

**On the web page** (`run.py hubgui`, started from `.venv-train` so
Ultralytics is there): step 3, *Fuel model* -> *Choose model (.pt)*. The
slider is the model's share. While counting, each zone shows both halves
(`103 in (colour 103 · model 98)`); a model half stuck near 0 is a model
that is not keeping up. The page refuses to start, and says why, when the
model file has gone or Ultralytics is missing.

**A model that cannot keep up turns itself off.** Its half stops rising, so
the zone reads about half the colour count: on 4 CPU cores the page read
52 against colour's 103 after 40 s of Einstein 1. After 10 s, a model that
has skipped more than 20% of its frames is dropped for the session, the
count becomes the colour count, and the page says *Fuel model turned off*.
The feed still never goes down (only rises are sent). Checked on this
container: dropped at ~10 s, then 103 / 108, equal to colour alone.

The model sees a 640 px square cut around each outline at full resolution
(3.2 outline widths, shifted 0.7 widths up), at ~30 fps -- every other
frame of a 60 fps camera. Live, it runs on its own thread and always takes
the newest frame: the colour half is never delayed, and a model that falls
behind skips frames and says so every 10 s. On this container's 4 CPU cores
it ran at 2.6 fps, far too slow; it needs a GPU or Apple silicon.

Shipped settings, scored on all four Einstein matches at 30 fps against the
official checkpoints:

| | E4 | E5 | E8 | E1 | mean | AUTO winner |
|---|---|---|---|---|---|---|
| **combo, w = 0.5** | 8.7% | 5.6% | 8.4% | 4.5% | **6.8%** | right on all four |
| model alone | 7.2% | 5.1% | 11.1% | 5.8% | 7.3% | wrong on E8 and E1 |
| colour counter alone | 16.2% | 6.2% | 8.8% | 9.1% | 10.1% | right on all four |

Read it carefully:
- **E4, E5 and E8 were in fuel_relabel's training set**, so the model is
  flattered there (alone on E4: 7.2%, against 17% for the old model).
  **E1 is the only fair match.**
- **The honest E1 number is 7-9%, not 4.5%.** E1 helped pick the
  settings above. Picked on E4/E5/E8 alone, the best setting gave E1 9.0%
  (colour: 9.1%). The top 50 settings there were tied within ~1 point,
  but on E1 they ranged from 4.1% to 10.1%, median 7.0%. Leave-one-out over
  all four: combo 8.2%, model 9.7%, colour 10.1%.
- **The model alone got the AUTO winner wrong twice; the combo never did.**
  The colour half keeps that call right.
- **The weight is the plain mean on purpose.** 0.7 fit E4/E5/E8 best, which
  is the training set talking; E1 preferred 0.5-0.6, and the fits were flat
  from 0.5 to 0.7 (6.9 / 6.6 / 6.7%).
- **Latency:** the model half confirms a score once the ball has been gone
  2 frames and has not reappeared within 2 more -- about 170 ms at 30 fps,
  plus inference. Half of each ball arrives that late; the colour half is
  unchanged. A longer hold (reacquire 6, 300 ms) scored no better.
- `run.py hubcount --model` on 30 s of E1 gave 84 - 102 at 30 s (official
  95 - 96), in line with the replay that produced the table.
- **The model half is noisy by frame phase.** It sees every other frame at
  60 fps; started one frame later (odd frames instead of even) on E1 6-36 s,
  red's model half read 106 instead of 94 (+13%), blue's 84 instead of 87.
  The blend moved about half that (red 107 against 101), and colour did
  not move. The table above was measured on one phase, so treat
  differences of a few points between settings as noise.

What would settle it is the same thing as for colour alone: a hand-counted
recording from the real camera position. Run it with and without `--model`
and keep whichever is closer.

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

**Take the top edge 3-4 ball widths above the hood, not on its rim.** On
the 2026 Central Valley broadcast (2026-10-05) balls drop into the hub
behind the hood's mesh, over a hub top lit bright blue or red. There each
ball shows only as yellow fragments, mostly below the minimum blob, so a
crossing at the rim is mostly missed. An outline whose top edge sat on
the hood rim counted blue 57 of 159 and red 665 of 810: 38.9% mean error.
With the top edge raised 30-45 px (about 3-4 ball widths there), the balls
cross it as clean blobs against the wall behind. Raised about 45 px it
counted 166 / 808: 6.4%, AUTO right, 93-122% of the official count at
every checkpoint after the first. Raised about 60 px it over-counted blue
225 / 159 (21.4%), catching misses that arc down past the hood. So there is
a right height. Check it the same way here: drop 20 balls in and see that
about 20 count.

**The rule is "where a falling ball is still in clear view", not a fixed
raise.** Raising every outline automatically by k ball widths was tried on
all five scored matches:

| k | E4 | E5 | E1 | E8 | Central Valley |
|---|---|---|---|---|---|
| 0 (as drawn) | 16.2% | 6.2% | 9.1% | 8.8% | 38.9% |
| 1 | 14.8% | 20.0% | 28.4% | 10.1% | 24.2% |
| 2 | 30.7% | 13.7% | 18.7% | 30.1% | 9.6% |
| 3 | 21.7% | 16.5% | 24.7% | 28.9% | 5.7% |
| 4 | 31.3% | 26.4% | 22.9% | 44.4% | 7.0% |

The Einstein outlines were already drawn where balls are visible, so
raising them only adds flyovers. Central Valley's sat where they are
hidden. No single k suits both, so the counter does not raise outlines
itself; the person drawing them does, by looking.

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
