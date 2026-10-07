# Changelog

Notable changes per release. Dates are the release date, not the merge date.

Each `## vX.Y.Z` section below is the release body: `.github/workflows/release.yml`
reads it with `python3 deploy/package.py notes vX.Y.Z` when the tag is pushed,
so the notes and this file cannot drift apart. A tag with no section here fails
the build rather than publishing an empty release.

## Unreleased

- **Passes beside the hub are no longer scores.** A ball passed back to the
  alliance zone while the hub is active could fly through the corner of a
  hub's outline and count. Now, when that same ball leaves the outline
  moving sideways and is still seen flying on outside, its count is taken
  back. The feed never goes down, so it is still reported for a moment and
  the next ball into that hub absorbs it (shown as `owed`). Balls bouncing
  up off the hood and balls dropping out the bottom into the hub count as
  before, and counting latency is unchanged. Measured on five broadcast
  matches: mean error 14.0% -> 12.6% (Central Valley 26.6% -> 16.4%, where
  balls flying past were counted; the four Einstein matches 10.9% -> 11.7%).
  See deploy/HUB_FEED.md "Passes beside the hub taken back".
- **A hub can use an outline and an exit line together.** The hub now
  takes the larger of the two (each kind combined across cameras by the
  hub's sum / max / median setting) instead of adding them. A scored ball
  crosses the outline going in and the exit coming out, so adding counted
  it twice: on Central Valley, outline + exit added up was 30.5% off,
  against 16.4% for the larger of the two, the same as the outline alone.
  The exit acts as a floor: a chute camera that counts more than the
  outline wins, and a blocked or piled-up exit costs nothing.

## v0.5.2 — 2026-10-07

- **Watchtower: a Write key section for The Blue Alliance.** The TBA tab is
  now three parts: the event key; **Write key** (Auth ID and Auth Secret,
  the on/off switch for sending results, and how to get a key: request
  write access for the event at thebluealliance.com/request/apiwrite, which
  TBA approves by hand); and the optional **Read key** for importing teams.
  Both halves of the write key are masked like a password (the Auth ID was
  plain text), since together they can change the event's results on TBA.
  Same settings in event.yaml; nothing to redo.
- **Watchtower's settings pages are centred** in a wide window instead of
  hugging the left edge, with Save / Undo lined up under them.

## v0.5.1 — 2026-10-07

The first release the v0.5.0 apps can install themselves: on Mac and
Windows, open v0.5.0 and press **Update** on the Hub Counter page or
Watchtower's Overview (with the cameras stopped).

- **Watchtower through a public address.** Phones & PINs has a new optional
  *Public address* (e.g. `https://watchtower.systemoverload.org`, or any
  other proxy or tunnel that forwards to port 8000 on the Watchtower
  computer). When set, phones are given that address instead of the
  `.local` name and the QR code carries it, so refs and the emcee connect
  through the website from any network; only the raw Wi-Fi IP stays listed,
  as the fallback for when the proxy or internet is down. Empty by
  default: nothing changes until it is set. Stored as `server.public_url`
  in event.yaml. The Watchtower app's own hub counter keeps sending to
  127.0.0.1, so counting never depends on the internet. A Hub Counter on
  another computer can send through the proxy with
  `https://KEY@watchtower.systemoverload.org` (that already worked; the
  page and `--target` help now say so).
- **One-click updates start from v0.5.0.** v0.4.4 and earlier have no
  updater, so they need a manual download from the release page; from
  v0.5.0 on, the Update button installs each new release. v0.5.0 -> v0.5.1
  is the first real one-click update.

## v0.5.0 — 2026-10-07

- **Phones get a name, not a raw IP.** When you share the page on Wi-Fi (Hub
  Counter) or open Phones & PINs (Watchtower), the address now leads with
  `<computer>.local` -- which Macs and Windows announce on the network by
  themselves and phones resolve -- instead of `192.168.x.x`. The IP is still
  shown as a backup and the QR code still carries it, because not every
  network passes `.local` through and every device on the Wi-Fi must still
  get in. The page stays gated by its PIN; the name is a convenience, not a
  security measure.
- **The team's red and black**: app icons (Hub Counter a yellow fuel ball
  over a white hub on black, Watchtower a black lighthouse on red) and the
  pages' accent colour and logos (buttons, sliders, selection were blue).
  Fuel stays yellow and the blue alliance stays blue: those are the game. They replace
  PyInstaller's default (the Python icon on Windows) on the Mac app, the
  Windows program and its installer. Drawn by `apps/icons/make_icons.py`;
  the finished `.icns` / `.ico` / `.png` files are committed, so builds need
  nothing new.
- **Camera presets, picture settings and fuel colour** (Hub Counter page,
  step 1). Built in: **ELP OV4689 2.8-12 mm varifocal, 60 fps** (1280x720
  MJPG at 60 fps, exposure fixed at 8 ms, white balance at 4600 K) and **USB
  webcam (standard)** (1280x720 at 30 fps, automatic picture). Exposure,
  white balance, brightness, contrast, saturation, gain and the video format
  can be set per local camera, and settings a driver ignores are named in
  the log. The fuel colour gate (hue range, minimum colour and brightness)
  is adjustable per camera, with a magenta overlay of what counts as fuel.
  **Save as preset** keeps all of it in `camera-presets.json` for reuse.
  Both apps have it: Watchtower's Hub cameras tab is the same page, and the
  two apps share one presets file (`~/Documents/Hub Counter/`), so a preset
  saved in either is offered in both.
  Why: a webcam on auto exposure drops to 15-24 fps in a dim gym (20 fps
  measured 25.1% error on the Einstein replays, 60 fps 10.5%), 60 fps over
  USB 2 needs MJPG, and auto white balance moves the fuel's hue. The ELP
  values are a starting point, not yet checked on the camera; the standard
  colour gate is unchanged, so existing setups count exactly as before.
- **`attach-models` workflow**: copies `fuel_relabel.pt` / `scout_relabel.pt`
  (checksum-checked against this file) onto any existing release from the
  Actions tab, no computer needed. v0.4.4 went out without them because
  release.yml's Publish step failed before its models step.
- **One-click updates in both apps.** Each app asks GitHub for the latest
  release when it opens (public API, no key; silent when offline). If it is
  newer, the Hub Counter page and Watchtower's Overview show "Update
  available" with an **Update** button -- never automatic, and refused while
  the cameras count, so a match cannot be interrupted. Mac and Windows:
  download, close, install in place (Windows: the installer silently into
  the same folder; Mac: the .app swapped, the old one put back if the copy
  fails), reopen. Linux/Pi: a link to the release page. Builds now carry
  their release tag (`app_version.txt`); pull-request builds (0.0.0) never
  check. Only v0.4.4 -> next can be the first real test: tested here with a
  stand-in for GitHub and the helper scripts' text, not yet on a real Mac or
  Windows install.

## v0.4.4 — 2026-10-06

Watchtower becomes a desktop app: our camera setup page plus Watchtower's
own parts in one window, with every setting edited inside it (event,
teams, PINs, schedule, TBA, game rules) and the team list imported from
The Blue Alliance in one click, no key. The Hub Counter opens in its own
window too. Both apps install like other apps: a .dmg on Mac and an
installer on Windows.

| app | Mac | Windows | Linux PC | Raspberry Pi |
|---|---|---|---|---|
| Watchtower (FMS + hub counter) | `Watchtower-mac.dmg` | `Watchtower-Setup-windows.exe` | `Watchtower-linux-x64.tar.gz` | `Watchtower-linux-arm64.tar.gz` |
| Hub Counter | `HubCounter-mac.dmg` | `HubCounter-Setup-windows.exe` | `HubCounter-linux-x64.tar.gz` | `HubCounter-linux-arm64.tar.gz` |

None is signed: the first open asks once (Mac: System Settings → Privacy
& Security → Open Anyway; Windows: More info → Run anyway).

### Changed

- **The Hub Counter app opens in its own window**, like Watchtower (Mac:
  WebKit, Windows: WebView2), not a browser tab. Closing it quits. Linux
  and the Pi keep the browser, and `--no-browser` stays headless for a Pi
  with no screen. CI runs a window test on macOS and Windows
  (`apps/hubcounter/smoke_test.py --window`).
- **Watchtower's Home now contains our camera setup page** as its **Hub
  cameras** tab, edge to edge, rather than a separate window. It loads once
  and keeps counting behind the other tabs. Watchtower's own pages
  (Scorekeeper, Field display) stay in their own windows, because a framed
  page loses its login. Home opens at 1440x900 to fit it. Checked: the
  window self-test loads the tab inside Home, and a screenshot shows the
  page beside Watchtower's sidebar.

- **A release can be made from a phone.** Publishing it on GitHub
  (Releases → Draft a new release → new tag on main → Publish) creates the
  tag and the release together. `release.yml` now takes such a release over
  instead of failing on "already exists": the CHANGELOG notes replace what
  was typed, and the archives and models are attached. The app builds were
  already attaching to an existing release.

- **Proper installers instead of zips** for both apps:
  - **Mac:** `Watchtower-mac.dmg` / `HubCounter-mac.dmg`. Open it and drag
    the app onto Applications.
  - **Windows:** `Watchtower-Setup-windows.exe` /
    `HubCounter-Setup-windows.exe`, an Inno Setup installer
    (`apps/installer/windows.iss`). It installs per user, so no admin
    rights are needed, adds a Start-menu shortcut (desktop optional) and an
    uninstaller, and a newer one upgrades in place.
  - **Why:** the zip had to be extracted whole and run from wherever it
    landed. Not one self-extracting exe: with PyTorch inside, it would
    unpack ~1 GB on every start.
  - **CI tests what people download:** on Mac, the app is run out of the
    mounted .dmg; on Windows, the installer runs silently and the
    installed apps are smoke-tested.
  - Linux and the Pi keep the `.tar.gz`.

- **The Watchtower app is a desktop app with its own windows**, not pages in
  the browser. It is built on pywebview: WebKit on macOS, WebView2 on
  Windows.
  - **The Home window** has every setting a scrimmage needs (event, teams,
    schedule, time zone, PINs, The Blue Alliance, game rules). Nobody edits
    `event.yaml` by hand any more.
  - **Saving** checks the settings with Watchtower's own rules, changes only
    those lines (the file's comments stay), and restarts the app.
  - **Phones:** a QR code and address for refs and emcee, plus a warning
    when the computer is not on a network.
  - **Windows:** Scorekeeper, Hub cameras and Field display each open in
    their own window. A framed page would lose its login: WebKit partitions
    storage for pages framed from another origin.
  - **Closing Home quits.**
  - **Checked here** with Qt's engine, offscreen: Home filled itself through
    the app's API; the three windows loaded "FMS Control", "Hub Counter"
    and "FMS Display"; a save restarted the app serving the new name and
    teams; and Quit on the hub page closed it. CI runs the same window test
    on macOS and Windows (`smoke_test.py --window`).
  - **Import teams from TBA, no key needed.** A button on Event &
    schedule reads the team list from the event page's public **Scouting**
    tab on The Blue Alliance (`team_number,team_name,city,...` CSV; 11
    teams for 2026catstd today, with names) and fills the Teams box, with a
    names preview to check before Save. The Read API is only a fallback,
    for an event page with no list yet; its key goes on The Blue Alliance
    tab, or comes from `$TBA_AUTH_KEY`, and is stored as `tba.read_key`.
    Watchtower ignores that key; it is added as one line at the end of its
    section. Pasting the Scouting CSV (Copy to Clipboard) or TBA's JSON
    into the Teams box also turns it into a clean list. Only the
    team_number column counts, so the "75" in a photo URL is not a team.
  - **Linux and the Pi** open the same Home page in the browser instead (no
    window toolkit is bundled there), so settings are never hand-edited
    there either. It is served on 127.0.0.1:8789, and every call needs a
    per-launch token in a header, which another web page cannot send.
    Checked in Chromium: saving restarted the app, which then served the new
    name, date and teams.

### Added

- **Watchtower patch 0007** (`deploy/frc-fms-patches/`): Watchtower's own
  releases get the same Mac `.dmg`, Windows installer and desktop windows
  as ours, built from v0.4.4 and checked the same way in its CI.

## v0.4.3 — 2026-10-06

A Watchtower app: the Watchtower FMS and the hub counter in one double-click
program, already set up and pointed at each other, for Mac, Windows, Linux
and Raspberry Pi. Watchtower's own releases can build the same app
(`deploy/frc-fms-patches/0006`).

| app | download |
|---|---|
| Watchtower: FMS + hub counter (Mac / Windows / Linux / Pi) | `Watchtower-mac.zip`, `Watchtower-windows.zip`, `Watchtower-linux-x64.tar.gz`, `Watchtower-linux-arm64.tar.gz` |
| Hub counter only | `HubCounter-mac.zip`, `HubCounter-windows.zip`, `HubCounter-linux-x64.tar.gz`, `HubCounter-linux-arm64.tar.gz` |

### Added

- **A Watchtower app: the FMS and the hub counter set up and started with a
  double-click** (`apps/watchtower/`, `Watchtower-*.zip` / `.tar.gz` on every
  release, all four OSes). Running Watchtower used to mean cloning it, a
  venv, `python -m fms.init`, `./run.sh`, then typing
  `http://<vision key>@host:8000` into the counter.
  - **First launch:** the app makes `event.yaml` with PINs and a vision key.
  - **Every launch:** it starts the FMS on :8000 for phones and the hub
    counter already pointed at it, and opens a start page with every link
    and PIN.
  - **Inside:** the bundled Watchtower is its v0.1.0 tag
    (`.github/watchtower-release`).
  - **Checked here:** a packaged linux-x64 build counted Einstein 1 into its
    own Watchtower live (59.9 fps, both hubs), the live page updates
    (WebSocket) worked, a wrong vision key was refused (401), and Quit
    stopped both. CI's smoke test repeats this on every OS.
- **Watchtower can build the app in its own releases** (patch 0006:
  `.github/workflows/app.yml` there). It checks out this repository at the tag
  in its `.github/hubcounter-release` and bundles its own code as the FMS.
- **The hub counter's address box can be preset by the app that starts it**,
  and its label now names both targets: bioarena `host:port`, or Watchtower
  `http://KEY@host:8000`.

## v0.4.2 — 2026-10-06

Every Hub Counter app can now run the colour + fuel-model blend, with
`fuel_relabel.pt` built in, and every release carries both models. On a Mac
(Apple silicon) the model runs on the GPU; on Windows, Linux and the Pi it
runs on the CPU, where it usually cannot keep up and turns itself off. A Pi
can host the page on Wi-Fi with no screen (`--share --pin N --no-browser`,
start at boot).

| app | download |
|---|---|
| Mac (Apple silicon) | `HubCounter-mac.zip` |
| Windows 10/11 | `HubCounter-windows.zip` |
| Linux PC | `HubCounter-linux-x64.tar.gz` |
| Raspberry Pi 4/5 (64-bit Pi OS) | `HubCounter-linux-arm64.tar.gz` |

### Added

- **Every Hub Counter app can run the colour + model blend.** The builds
  carry PyTorch, Ultralytics and `fuel_relabel.pt`, and the page has a **Use
  built-in model** button. Setups save `model: built-in` rather than the file's
  path inside the app, which changes when the app moves or updates;
  `run.py hubfeed --model built-in` and frc-fms's `model: built-in` work too,
  if a checkout has `models/fuel_relabel.pt`.
  - **Where it keeps up:** on Apple silicon the model runs on the GPU (MPS).
    Windows and Linux carry CPU PyTorch, because a CUDA build is ~3 GB, over
    GitHub's 2 GB asset limit. On a CPU the model turns itself off after
    ~10 s and the hub counts by colour. A linux-x64 build tried here, on 4
    cores, did exactly that on Einstein 1.
  - **Size:** each download is 300-400 MB (Windows 288 MB, linux-x64 398 MB)
    instead of 60-100 MB. Both Linux builds take PyTorch's CPU wheels: PyPI's
    arm64 torch carries CUDA, and the Pi build came out at 3049 MB, which a
    release would refuse (2 GB per file). A build over 1.9 GB now fails CI.
  - **The smoke test now runs the model inside each build.** The first build
    passed every page check, yet failed on the first prediction:
    torchvision's `_C_stable` ops (`nms`) are loaded by path, so neither the
    bundler nor `collect_dynamic_libs` saw them. The spec now lists them.
- **Every release carries the models.** `release.yml` copies
  `fuel_relabel.pt` and `scout_relabel.pt` from the release named in
  `.github/models-release` (v0.4.0), after checking them against the
  checksums recorded here. `hub-app.yml` puts the same `fuel_relabel.pt` in
  the apps. Training new models means attaching them to that release by hand
  once and changing that file.

- **The app runs headless on a Pi:** `Hub Counter --share --pin N --no-browser`
  shares the page on Wi-Fi from the start. Before, a Pi with no screen could
  share only via someone pressing Share on the Pi's own page. The app README
  has a systemd service to start it at boot.

### Fixed

- **The Fuel model label on the page stayed "(off)" after turning the model
  on** until another camera was picked. It now updates on every refresh.

## v0.4.1 — 2026-10-06

The hub counter as a double-click app for Mac, Windows, Linux and Raspberry
Pi, shareable on Wi-Fi with a PIN; a counter plugin system for frc-fms with
our colour and combo counters as plugins; and outline guidance from the
Central Valley broadcast (top edge 3-4 balls above the hood: 38.9% -> 6.4%).
The fuel models are unchanged; use v0.4.0's `fuel_relabel.pt`. The app
builds are attached by the hub-app workflow.

### Added

- **A counter plugin system for frc-fms (watchtower-fms), and our counters as
  plugins of it** (`deploy/frc-fms-patches/0003-0005`, `pyproject.toml`).
  - **His side:** `counter:` finds plugins by short name, either installed
    packages registering `watchtower.counters` entry points or `plugin_paths:`
    folders. `--list-counters` lists them. A misspelt option is flagged
    (`ball_aera` → "did you mean `ball_area`?"). A plugin's `status()` shows
    under its hub on `/control`, and a plugin that fails to load says why
    instead of the hub reading "never connected".
  - **Ours:** `tbavid-colour` and `tbavid-combo` declare their name, needs
    and options, report `colour N · model M` or "model off" as their status,
    and release the model thread in `close()`. They are found by
    `pip install -e` this repository or by `plugin_paths:`, with no
    `PYTHONPATH`.
  - **Checked** against a running watchtower-fms, both ways. Its tests pass
    with the patches (37).
  - **Adding a plugin without editing config** (patch 0005). Drop a `.py`
    into his `vision/plugins/` (copy the `_example.py` template), or run
    `python run_vision.py --add-plugin ~/dev/TBACroppedOutVid` once. It
    records the folder as a one-line `.path` file (no symlink, so it works on
    Windows) and prints `tbavid-colour / tbavid-combo`. Files there are
    parsed, not imported, so a broken one cannot stop vision from starting.
    His tests: 42 pass.
- **A slow model is now dropped after 10 s of wall time, not only after 300
  frames.** Inside frc-fms's runner on 4 CPU cores, the combo dragged the
  camera loop from 60 to 13 fps, so 300 frames took about 45 s to arrive.
  Counting was wrong for all of AUTO before the model was dropped. Now it
  goes at about 10 s and the hub is back at 60 fps on colour.

- **The Hub Counter app for Mac, Windows, Linux and Raspberry Pi**
  (`apps/hubcounter/`, `.github/workflows/hub-app.yml`). The hub counter's
  web page as a double-click program: nothing to install and no terminal.
  It keeps the setup and logs in `Documents/Hub Counter/`, and opening it
  again only reopens the page. It holds the hub counter only: none of the
  scraper, PyTorch or Ultralytics, so no `--model` blend. Each OS's build
  runs on its own GitHub runner (macos-14, windows-latest, ubuntu-22.04,
  ubuntu-22.04-arm) for every pull request touching it (as artifacts), and
  is attached to each release. `apps/hubcounter/smoke_test.py` checks every
  build: the page and the board load, the data folder is made, and Quit ends
  the program. A Linux build of the same spec held exactly the nine hub
  modules and the two pages, and passed it.
- **Twitch / YouTube streams work inside the app.** Stream lookup ran
  `sys.executable -m yt_dlp`. In a packaged app `sys.executable` is the
  app itself, so adding a stream would have started a second copy of the
  counter. A packaged app now calls yt-dlp as a library, with the same
  format choice and messages.
- **Share on Wi-Fi.** A Share button on the page (or `run.py hubgui
  --share [--pin N]`) opens it to other devices on the same network, at
  port 8791, behind a 6-digit PIN shown only on the counting computer.
  That covers a phone at the table, or a Pi with no screen set up from a
  laptop. `/board` stays open for a TV. Quit and stopping the share are
  host-only. Five wrong PINs from one address lock it out for a minute.
  The page on the counting computer itself is unchanged (127.0.0.1, no
  PIN). Checked in Chromium: the PIN screen on a phone, the full page
  after it with Quit and Share hidden, and the address and PIN shown on
  the host.
- **A Quit button on the hub page**, since an app has no terminal to press
  Ctrl-C in. It asks first, and says so if counting is running or the
  setup is unsaved.

### Changed

- **The setup page recommends a raised outline, not exit lines.** It
  listed exit lines first as "recommended" and its hints said to draw one.
  On the 2026 Central Valley broadcast, the first with its exits in view,
  exit lines measured 82-85% error (blue 25-32 of 159, red 52-100 of 810):
  balls pour out into a pile against the hub instead of crossing the line
  one at a time. The raised outline got 6.4% on the same match. Exit lines
  stay, listed second and marked untested, for a camera aimed right at the
  chute.
- `run.py count`, `shots` and `detect` take `--device` (mps, 0, cpu). Not
  given, it is CUDA, then MPS, then CPU, and the choice is printed at
  startup. Before this Ultralytics chose, and it never picks MPS, so on a Mac
  these ran the model on the CPU.

### Measured

- **The colour + model combo was worse than colour alone on Central
  Valley**: 34.5% against 6.4% with the same raised outlines (final
  265 / 823 against 159 / 810). The model half counted blue 363. Use colour
  only on cameras like this.

- **Outline height decides the count on a wide broadcast** (2026 Central
  Valley, upper-bracket match 1, scored against the broadcast's fuel
  counts; deploy/HUB_FEED.md). With the top edge on the hood rim the
  colour counter got blue 57 / 159 and red 665 / 810 (38.9% error): balls
  dropped behind the hood's mesh over a lit hub top showed only as small
  fragments. Raised about 45 px it got 166 / 808, 6.4% error, AUTO right.
  Raised about 60 px it over-counted (blue 225 / 159, 21.4%). The setup
  page and runbook now say to put the top edge 3-4 ball widths above the
  hood. Doing that in code instead (every outline raised by k ball widths)
  was rejected. At k = 3, Central Valley went to 5.7% but the four
  Einsteins went from 6-16% to 17-29%: their outlines were already where
  balls are visible.

## v0.4.0 — 2026-10-03

Hub fuel counting for the scrimmage: a live counter that feeds bioarena's
UDP spec or frc-fms, a web page to set it up, and a blend of the colour
counter with a retrained fuel model. Also retrained models with balls in
flight labelled, and the steps to run all of it with frc-fms.

### Models (attached to this release, not in the source archives)

| file | classes | trained on | validation (Einstein 1, held out) |
| --- | --- | --- | --- |
| `fuel_relabel.pt` | fuel | `relabel_video.py` set, 12332 labelled frames (2026nhdur + Einstein 4/5/8), 40 epochs | mAP50 0.757, mAP50-95 0.506, R 0.72 |
| `scout_relabel.pt` | fuel, robot_blue, robot_red | the same set, fine-tuned from `fuel_withBotbest.pt`, 60 epochs | all mAP50 0.877, mAP50-95 0.671; fuel 0.764 / 0.508; robots 0.93 |

Both are YOLO26s at `--imgsz 960`, trained on an AMD MI300X. Check the
downloads with `sha256sum`:

```
bf9eb2c3cb25b119b54bd9a09d35227ad30a99a371a5dc532828cd9c0d7f4977  fuel_relabel.pt
0112eba7783c980f88a18626ad8500a82593e2f216a61f56173c57e8cf1dd2e6  scout_relabel.pt
```

What the numbers do and do not mean:

- The validation labels are the relabeller's, so these scores measure
  agreement with it, not with reality.
- What was measured against reality is hub counts against official scores:
  - `fuel_relabel.pt` counting alone on Einstein 1: 12.1% mean error,
    against 18.0% for `fuel_best.pt`.
  - Blended with the colour counter (`--model`): 6.8% over the four
    Einstein matches, against 10.1% for colour alone.
  - Einstein 4, 5 and 8 were in the training set. On a match nobody tuned
    on, expect 7-9%.
- Nothing has been checked against a hand-counted practice hub. Do the
  20-ball test, and keep a human scorekeeper.

### Added

- **Patches for frc-fms** (`deploy/frc-fms-patches/`, not yet in frc-fms):
  - its `zone` counter counts only the class named `fuel`. With a
    three-class model it had counted robot boxes as fuel: 6 in one frame of
    Einstein 1.
  - its README documents this repo's `ColourCounter` / `ComboCounter`
    plugin.
  They apply cleanly to frc-fms 20524ce, and its tests pass afterwards
  (26).
- **Step-by-step setup with frc-fms** (`deploy/FRC_FMS.md`, "Set up on the
  day", and `deploy/frc-fms.vision.yaml`). Covers installing both
  repositories, drawing the hubs and measuring a ball in `run.py hubgui`,
  then counting either inside frc-fms (way A, the plugin) or from our page
  posting to it (way B). Never both: each would post every ball. It ends
  with the checks before the first match (its control page's Vision panel,
  a 20-ball drop per hub), re-counting a recording with `rescore.py`, and
  a table of what goes wrong. The example config, loaded through frc-fms
  20524ce's own `vconfig` and `load_counter`, built `ComboCounter` for both
  hubs from a `cams.json`: outlines and ball size from the file, 60 fps,
  the model on every other frame.

- **`--model` for `run.py hubcount` / `hubfeed` — blend the fuel model into
  the colour count** (`tbavid/hubmodel.py`, `deploy/HUB_FEED.md` "The combo,
  tuned"). Each outline counts `round(0.5 * model + 0.5 * colour)`. The model
  half runs `fuel_relabel.pt` on a full-resolution crop around the hub ->
  BallTracker -> BallCounter, at ~30 fps on its own thread, so the colour
  half is never delayed. Also a `"model"` entry per camera in cams.json.
  Tuned on all four Einstein matches: mean error 6.8% against 10.1% for
  colour alone, with the AUTO winner right on all four (the model alone got
  it wrong on two). Three of the four were in the model's training set; on
  Einstein 1, the one it never saw, the honest estimate is 7-9% against
  9.1%. Needs a GPU or Apple silicon (picked automatically: CUDA, then MPS);
  on 4 CPU cores it ran at 2.6 fps.
  Not yet checked against a hand-counted practice-hub recording.
- **The colour + model combo inside frc-fms**: `tbavid.fms_counter:ComboCounter`
  (or `ColourCounter` with `model:`) in frc-fms's `config/vision.yaml`. The
  model runs on its own thread so frc-fms's camera loop never waits.
  Live, a model that cannot keep up is dropped and the hub counts by colour.
  Under `rescore.py` it waits for the model on every frame instead. Run
  through frc-fms 20524ce's own `run_vision.py` on 4 CPU cores, the model
  was dropped after 10 s and the totals equalled the colour plugin's.
  `hubcount.run` and the plugin share one model thread
  (`hubmodel.ModelWorker`). Fed the same frames of Einstein 1, the plugin
  and `run.py hubcount --model` agreed frame for frame (red model half
  94 / 94). Shifting the model by one frame (odd frames instead of even)
  moved red's model half from 94 to 106 and the blend from 101 to 107, so
  part of the model half's error is sampling noise.
- **The web page (`run.py hubgui`) can turn the fuel model on**: step 3,
  *Fuel model*, a .pt picker and the model's share. Each zone shows its
  colour and model halves while counting, and the page says when the model
  is falling behind. A model that skips more than 20% of its frames after a
  10 s warm-up is dropped for the session and counting goes on by colour:
  on 4 CPU cores a lagging model had held the page at 52 against colour's
  103.

- **`run.py hubfeed` — feed hub fuel counts to bioarena** (`tbavid/hubfeed.py`,
  `tbavid/hubcount.py`, `deploy/HUB_FEED.md`), for the 2026-10-10 scrimmage.
  Implements the counter's side of Team 841's "Hub FUEL Counter Feed" spec:
  UDP to 10.0.100.5:8411, both hubs' cumulative counts in one JSON datagram,
  a fresh session per start, sent the moment a count rises and every 100 ms
  otherwise, `age_ms` from the frame's capture time (the V4L2 driver stamp on
  Linux). The heartbeat stops when a camera stops delivering frames, so
  bioarena shows OFFLINE rather than frozen counts. `run.py hubfeed-listen`
  stands in for bioarena with its acceptance rules, to check the link before
  the field computer exists; the tests hold the sender to the same rules.
  The counting is `experiments/area_hub_count.py` made live, because
  `run.py count` holds each score 400 ms (twice the spec's p99 budget) and
  needs detector tracks that lost the balls on broadcasts. Its signed
  in/out crossings are reported as a high-water mark, since the feed may
  never go down: a ball in and back out is reported until the next ball in
  absorbs it. Its thresholds are scaled by the measured one-ball area and
  **none is validated on a practice-field camera** -- on broadcasts it was
  93-95% on the match it was fitted to and 120-137% blind. Only a synthetic
  video has been run end to end. The spec's 20-ball field acceptance test is
  the first real measurement; leave bioarena's AUTO winner off `counted`
  until it passes. Checked against the Einstein 4 and 5 broadcasts: the
  port's counts equal the experiment's on Einstein 4 and are within 1% on
  Einstein 5, both AUTO winners come out right, and the totals are 93% / 93%
  and 123% / 137% of the broadcast fuel counters (`deploy/HUB_FEED.md`).
  On Einstein 1, a one-ball AUTO (95 - 96 at the decision), it counted
  155 - 179: the right winner by coincidence, red ahead while blue led at
  AUTO's end, and red ahead at the buzzer of a match blue won 621 - 415.
  Two fixes were tried on all three and not kept: a blur-aware ball count
  (halved every total, Einstein 4 to 47-58%) and removing static yellow
  (a few percent either way). The error is balls that clip the rim and drop
  behind the hub, which look like scores from in front -- a camera-angle
  limit (`deploy/HUB_FEED.md`).

- **`run.py hubfeed --setup cams.json` — several cameras.** Any number of
  cameras, each with its own source, frame rate, size and one-ball area, each
  with outlines counting into a hub; per hub the zones are combined by `sum`
  (different balls, e.g. one camera per exit chute), `max` (same balls, take
  the one that missed fewest) or `median` (three or more, outvote the odd
  one). All three keep the feed's never-decreasing rule. Any camera going
  quiet stops the heartbeat. `--measure` works per camera and prints each
  camera's ball area for the file; `deploy/hubfeed.example.json` is a
  template. Motivated by the Einstein tests: from in front, a ball that clips
  the rim and drops behind the hub looks like a score, so accuracy needs
  cameras close to each hub. Checked end to end with two recordings as two
  cameras on one hub under `max`; not yet run on real multi-camera hardware.

- **`run.py hubgui` — the hub counter with a user interface**, as a web page
  (default; `tbavid/hubweb.py`, standard library, any browser) or a Qt
  window (`--ui qt`, `tbavid/hubqt.py`, needs PySide6). Both are views over
  one controller (`tbavid/hubapp.py`), so a button does the same thing in
  each. Find cameras or add a recording, click each hub's outline on the
  picture, measure the ball, calibrate, and start the feed; the live picture
  shows each outline's count, the score sent to bioarena, the bioarena link
  and every camera's frame rate, and says *NO PICTURE* when a camera stops.
  A practice switch runs a test receiver inside. It edits the same
  `cams.json` that `hubfeed --setup` runs headless; `hubfeed.command` opens
  the web page from Finder. The web page listens on 127.0.0.1 and refuses
  other host names (DNS rebinding) and non-JSON posts (cross-site forms).
  Both were driven end to end on Einstein 4 as the camera -- the page in
  Chromium with real mouse clicks, the window on Qt's offscreen platform --
  and counted the same (red 3, blue 1 at 0:16) into the test receiver. The
  first browser run found the click layer sized 0x0 before the picture was
  laid out, so no outline could be drawn; it is now sized on the picture's
  load. Neither has been opened on a Mac yet. (A Tk window came first and
  was replaced: Homebrew's Python needs `brew install python-tk` for it.)
- **`run.py track` — a readable tracked video, and balls followed through
  the top of their arc** (`tbavid/trackvis.py`, `tbavid/fuel_track.yaml`).
  Asked for: smaller labels, better following, and no losing the ball when
  it "plateaus". Measured on 4 s of Einstein 4 shooting (239 frames,
  `fuel_best.pt`, imgsz 960):
  - Labels: `predict(save=True)` put "fuel 0.85" on ~300 balls a frame and
    buried the picture; this draws thin boxes and a small id, scaled to the
    frame, and trails only for balls really travelling (piles and hoppers
    jitter in place and drew zigzags).
  - Following: ByteTrack at its defaults with conf 0.25 split the balls into
    2357 tracks, median 5 frames. conf 0.25 threw away the weak detections
    before ByteTrack's low-score association saw them, and the default
    match threshold needs 20% box overlap, which a 15-20 px ball moving its
    own width a frame barely has. `fuel_track.yaml` (conf 0.1, weak
    detections may only continue a track, match 0.95, 60-frame memory):
    1145 tracks, median 25 frames, short (<5 f) 1178 -> 167.
  - The top of the arc: still only 1 of 25 flights survived its apex. In 19
    of 20 flights lost while rising or at the top, the detector still had
    the ball at confidence 0.1-0.8 in the next frames -- the linking, not the
    detection, dropped it: a ball turning over stops overlapping a
    straight-line prediction. `BallTracker` links by distance to the
    predicted position, in a gate scaled by size, speed and time missing,
    and rejects a size far off (a robot). Result: 818 tracks, median 172
    frames, 762 lasting 30+ frames, 11 under 5, and 106 flights followed
    through their apex (ByteTrack: 1). Jumps over 2.5 ball widths in a
    frame: 35 of 129,884 links (0.03%), some real fast balls. It is the
    default; `--tracker bytetrack` keeps the tuned ByteTrack.
  - A ball the detector misses for a few frames is drawn where it should
    be, as a hollow circle, for up to `--coast` frames (display only).
  - `run.py shots --annotate` labels are smaller too.
  - **Balls high against the crowd, without retraining** (`colour_assist`,
    on by default, `--no-assist` to turn off). The released models miss
    them outright: `autolabel_fuel.py`'s field line drops every ball more
    than ~4 ball-widths above the far edge of the field (to keep crowd
    shirts out), so the top of every shot was in the training frames
    unlabelled and the model learned it as background. Of 1071 moving
    ball-shaped yellow blobs without a model box on the Einstein 4 clip,
    516 had no score at all and 555 only 0.01-0.1, so no threshold brings
    them back. The assist adds round, ball-sized yellow blobs that are
    moving (a shot moves; a shirt mostly does not) and not in a model box,
    at a confidence below the tracker's start threshold: colour can carry
    on a ball the model found at launch, never invent one. 1021 of 1308
    offered were used to continue a ball; flights followed through the
    apex 106 -> 114 on this broadcast; drawn magenta. Retraining on labels
    that keep balls in flight remains the real fix.
  Not done: `run.py shots` and `run.py count` still track with ByteTrack;
  moving them to `BallTracker` needs its own measurement against a
  scoreboard.
- **Wireless cameras** (*Wireless camera* in the website; an `rtsp://` or
  `http://` `"source"` in `cams.json`): Wi-Fi IP cameras and phone
  IP-camera apps. Opened through FFmpeg over TCP with input buffering off
  (UDP smears the picture over Wi-Fi; buffered frames are latency). A drop
  holds the heartbeat and reconnects for as long as the counter runs, and
  forgets the pre-drop blobs, since matching a ball from before the gap to
  one after it would invent a crossing. Camera passwords are kept out of
  the log. An iPhone on a Mac needs none of this: Continuity Camera lists
  it under *Find cameras*. Tested against a stand-in MJPEG phone camera:
  picture in 1.3-1.8 s, 30 fps, and a 4 s drop showed NO PICTURE, held the
  heartbeat and resumed at 30 fps. Not tried with a real phone, an IP
  camera, or RTSP.
- **Exit-line counting** (`hubcount.ExitLineCounter`; *Red exit* / *Blue
  exit* in the website; `"line"` + `"out"` zones in `cams.json`). Every
  scored ball comes back out of the hub, so balls crossing a line across an
  exit, outward, are the score -- and a ball that clips the rim and drops
  behind the hub, which a funnel-mouth outline cannot tell from a score
  (Einstein 1: 832 entries over the red hood, 415 real), never gets there.
  The ball's path between frames is tested against the segment, so a fast
  ball is still caught and one passing beyond the line's ends is not; a ball
  that crosses back is taken off. Unit-tested and driven in the browser.
  Not measured on real exits: on the Einstein broadcasts the exits are
  mostly hidden behind the hubs, and a line across the visible red out-flow
  caught 9-34% of the scoreboard. A practice hub's exits are the test.
- **The hub counter's interface is the web page.** The Qt window
  (`tbavid/hubqt.py`, `run.py hubgui --ui qt`) was removed once the website
  was chosen; `run.py hubgui` opens the page and no longer takes `--ui`.
- **Twitch and YouTube streams as a counter source** (`hubcount.open_source`,
  and *Add stream* in both front ends). A page address is looked up to its
  HLS stream with yt-dlp -- run from the app's own environment, since a
  double-clicked launcher does not have `.venv/bin` on PATH -- and opened
  through OpenCV's FFmpeg backend; a dropped stream reconnects up to five
  times with the heartbeat held meanwhile. A stream is several seconds
  behind the field, so the app says so when one is added and turns the
  status amber while one counts: practice and scouting, not bioarena's AUTO
  call. Tested end to end in the web page on a real Twitch past broadcast
  (2019 Championship, Newton): picture in 4.3 s, counting into the test
  receiver. That run found past broadcasts read unpaced at 315 fps; streams
  are now paced like recordings, and pacing no longer touches live cameras
  (it had applied to every camera once any source was a file). Not tested
  on a channel that was live at the time -- none of FIRST's were.
- **The counter's interfaces were redesigned** after the first version was
  called "a 1980 application": a dark control-room theme shared by the web
  page and the Qt window, score tiles that pulse on a new ball, a status
  card (connected / sending with no reply / no picture) with each camera's
  frame rate, a LIVE badge and per-hub count labels on the picture, and
  numbered steps that turn into check marks as the setup completes. The web
  page replaced every browser alert, confirm and prompt with its own dialogs
  and toasts, closes an outline by clicking its first corner, undoes a
  corner with Backspace, and keeps the scoreboard and picture on screen
  while the steps scroll. It moved into `tbavid/hubweb.html`; no web fonts
  or CDNs, since a field often has no internet. Driven again in Chromium
  (1440x900) and on Qt's offscreen platform, into the test receiver, with
  no page errors.
- **Per-camera `blur` and `remove_static`, and calibration** (`run.py
  hubfeed --calibrate VIDEO --camera NAME --count red=N,blue=M`, or the
  window). Both off by default. The best blur fraction was 0 / 0.2-0.3 /
  0.5-0.7 on Einstein 4 / 5 / 1, and one picked on two matches did no better
  than none on the third, so it is set per camera from a hand-counted
  recording, not fixed. The live code reproduces the offline experiment on
  Einstein 1: red identical at all 22 settings, blue within 2 (the offline
  cache stored positions as float32).

- **Hub counter tuned with every fix together** (`deploy/HUB_FEED.md`,
  "Every fix together"). About 850 configurations were swept on Einstein 4,
  5 and 1 and scored leave-one-out:
  - the mouth counter, the model counter and the exit lines;
  - their mean, min, max and median.

  Held-out mean error per method:

  | method | held-out mean error |
  |---|---|
  | mouth, tuned | 30% |
  | model + BallTracker + colour assist | 16% |
  | mean of the two | 13% |
  | exit lines on broadcasts | 76% (exits hidden) |

  The AUTO winner was right in every held-out run.

  In `hubcount`, the speck floor doubled and the matching reach grew 1.5x.
  Two of three folds chose exactly these. With ball area x0.9, blur 0.3
  and still-yellow removal, the live counter measures 8% / 7% / 39%
  (defaults: 7% / 34% / 71%).

  Not fixed: the model counter is offline only (18 fps; `BallCounter`'s
  400 ms hold), and three matches are too few to trust the best fit.

- **frc-fms support** (`deploy/FRC_FMS.md`). frc-fms is a scrimmage FMS
  whose vision side posts timestamped fuel events. Two ways in:
  - **Plugin:** `tbavid.fms_counter:ColourCounter` is a counter for
    frc-fms's own `run_vision.py`. It is the measured colour counter (10.1%
    held out on four Einsteins, against 17-25% for the model + tracker
    approach of frc-fms's `zone` counter), configured from frc-fms's `roi`,
    a polygon, an exit line, or a `cams.json` camera.
  - **Sender:** a `http://KEY@host:8000` target makes `run.py hubfeed`
    and the web page post to frc-fms (`tbavid/fmslink.py`). Events carry
    the wall-clock time they were seen and stay queued until a POST that
    carried them succeeds.

  Both were run against a real frc-fms server on the Einstein 4 recording:
  - the sender's stored totals matched the offline count exactly;
  - the plugin ran at 59.9 fps and matched the offline count at the point
    each hub reached.

  `ball_area` is required by the plugin: measuring it from the first
  frames read the title card (98 px against 272).
- **Measured on a held-out match, Einstein 8** (`deploy/HUB_FEED.md`).
  The counter had never been tuned on it. The shipped defaults through
  `run.py hubcount` measured 9.1% error, buzzer 91% / 94%, AUTO winner
  right. Re-tuning with four matches, holding each out once, measured 21%
  against 10.1% for the current settings, so they are unchanged.
- **`train/relabel_video.py`: a retraining set with the balls in flight
  labelled**, the bootstrap step of `train/README.md`. It fixes the two gaps
  the released models were measured to have: balls high against the crowd
  (left unlabelled by autolabel_fuel's field line) and only ever having seen
  the 2026nhdur broadcast. How it labels:
  - fuel from the detector on full-resolution tiles;
  - balls in flight it misses added by motion;
  - other ball-like yellow painted grey rather than left unlabelled;
  - robots from the scouting model;
  - whole frames plus 960 px full-resolution flight crops.

  `--from-scraper` labels every video `run.py pull` kept:
  - it reads the manifest and uses each clean render (main camera, banner
    cropped), or the raw download with the same shot ranges and crop;
  - it never samples just after a cut;
  - each match keeps its train/val side from `dataset/`.

  `train/README.md` trains on this set instead of the old one: they label
  the same matches, and the old labels are the ones that taught balls in
  flight as background. `subset_classes.py` now reads the class order from
  the source dataset's yaml instead of assuming the five-class one.
  `--rows` / `--mask` handle the Einstein split-screen and the scoreboard's
  yellow fuel icon; both were being labelled as fuel. `train.py` now counts
  the labels of every set a mixed `train: [...]` yaml names. Its old check
  looked only beside the yaml and would have refused the Einstein + scouting
  mix.
- **The model measured on full-resolution hub crops** (`deploy/HUB_FEED.md`).
  A 640 px crop around each hub, instead of the frame halved to 960, found
  93% of balls in flight against 84%. Counting error without colour assist
  fell from 36% to 21%; with the assist it stayed at 17%. Averaged with
  the colour counter it measured 9.0% against 10.1% for the colour counter
  alone. Not yet built into `run.py`.
- **Both counters measured on every frame at 60 fps**, four matches, each
  held out once (`deploy/HUB_FEED.md`). The shipped colour counter measured
  10.1% at 60 fps and 11.1% at 30 fps. The model counter with colour
  assist measured 25% at 60 fps and 17% at 30 fps. Combining the two
  measured 13-14%, so the colour counter alone stays the live and scouting
  counter.
- **Slow-camera warning** on the page's status and on `/board` when a live
  camera delivers under 28 fps (`hubcount.MIN_FPS`, `hubapp.slow_cameras`).
  The Einstein blobs replayed at 60 / 30 / 20 fps measured 10.5% / 12.6% /
  25.1% error, and webcams drop to 15-24 fps on their own in dim light.
  The other setup errors tested were outlines 10 px off or 15% mis-sized,
  and one ball measured anywhere from 0.5x to 2x. Each cost at most 5
  points (`deploy/HUB_FEED.md`, "How forgiving the setup is").
- **`run.py hubcount VIDEO... --setup cams.json` counts recordings for
  scouting** (`hubcount.count_recording`, `write_timeline`). Previously a
  file went through `hubfeed`, paced like a live camera, sent over UDP and
  logged against wall time.
  - It decodes every frame as fast as it can and sends nothing.
  - It writes each hub's count against video time (`video_s,red,blue`
    every 0.5 s) and prints the totals.

  Einstein 4: 4.4x real time on 4 cores; 650 / 515 at the buzzer, the same
  as the evaluation.
- **Live scoreboard at `/board`** (`tbavid/hubboard.html`,
  `hubapp.board_view`; the **Scoreboard ↗** button on `run.py hubgui`). A
  full-screen red and blue board that refreshes ten times a second.
  - Linked to bioarena, it shows bioarena's credited score, match phase,
    clock, AUTO counts and inactive hubs from the status reply.
  - Otherwise it shows the camera counts, labelled as not a match score,
    with a screen-only Zero.
  - It warns when the counter is stopped, a camera has gone blind, or the
    page lost the counter.

  The Einstein 4 recording, paced at 60 fps through the whole path in
  practice mode, ran at 59.9 fps with 5 ms median and 14 ms worst
  capture-to-count time. Checked at 1920x1080 and at phone width. At the
  scrimmage, bioarena's own display remains the official scoreboard.

### Changed

- **The hub counter's outlines count downward entries, with one ball
  learned from the crossings** (`hubcount.CrossingCounter`,
  `deploy/HUB_FEED.md` "Downward entries and a learned ball"). Held-out
  error over Einstein 4 / 5 / 1 went from 30% to 13%. The shipped
  defaults measure 16% / 6% / 9%, with the AUTO winner right on all three.
  - **Direction:** an outline counts only blobs moving down into it, and
    outward crossings no longer subtract. On Einstein 1 blue, 161 entries
    against 171 exits had wiped out 108 real balls.
  - **Ball size:** one ball is the 30th percentile of the zone's last 80
    crossing blobs. Balls at the mouth were 1.5-2.3x the measured still
    ball, and most of Einstein 1's AUTO crossings were being counted as two.
  - **Rounding:** a blob rounds up to the next ball at 0.65.
  - **Blur:** cameras default to blur 0.3 (it was 0; blur 0 measured
    20% / 30% / 34%). Saved setups now always write their blur, so a chosen
    0 stays 0.
  - **Old setup files:** the page used to save every camera's blur
    slider, 0 unless moved. A setup without `"rules": 2`, which saves now
    write, has its blur 0 read as 0.3, with a note printed. A blur 0
    saved from now on is kept.
  - **Exit lines keep the signed rule and the measured ball**
    (`signed=True, learn=False`). The broadcasts cannot test them.

### Measured

- **frc-fms's `zone` counter on Einstein 1 with the retrained fuel model**
  (`fuel_relabel.pt`, MI300X, 2026-10-02): 37.8% mean error against the
  official checkpoints, down from 51.5% with v0.3.0's `fuel_best.pt`, and
  AUTO right where the old model got it wrong. The colour plugin in the same
  frc-fms measured 13.9%. Details in deploy/FRC_FMS.md.
- **Colour counter + retrained model on hub crops, averaged, on Einstein 1**
  (the one Einstein match `fuel_relabel.pt` never trained on; counter
  settings picked on Einstein 4/5/8 with the old model's runs, then not
  touched): 7.8% mean error vs 9.1% for the colour counter alone. The model
  alone improved from 18.0% (`fuel_best.pt`) to 12.1%, but the average did
  not move (old model 7.7%), so the gain comes from averaging, not from
  retraining. The 30 s counts were 86 - 101 for the average and 101 - 104 for
  colour alone (official 95 - 96), and AUTO was right in both. The buzzer
  was 90% / 110% for the average and 99% / 117% for colour alone. One match,
  1.3 points: not enough to change the shipped counter. Not built into
  `run.py`; it needs the model at 30 fps on two 640 px crops, which takes
  0.3 s per frame on a 4-core CPU, so live use needs a GPU or Apple MPS.

### Fixed

- **frc-fms's control page showed our counter as "undefined fps"** and
  never flagged a dead camera. The sender now posts each hub's `fps`,
  `counter` ("tbavid colour + model") and an `error` for a stale camera or
  one under 28 fps, so its "vision ok" pill tracks it. Checked on frc-fms
  20524ce's `/control`: "53.1 fps tbavid colour" on both hubs, vision ok.
- **The page's *Scoreboard* button was dark blue on black.** Buttons now
  take the page's text colour; a link styled as one had kept the browser's.
- **`run.py hubfeed` printed `bioarena ?` when posting to frc-fms.** It now
  prints `frc-fms took it, rtt N ms`, or `no reply from frc-fms`. Seen on
  the 2026-10-02 run against frc-fms 20524ce, where both connection methods
  stored exactly the counter's totals on Einstein 1 (red 108, blue 152).
- **deploy/HUB_FEED.md now says bioarena cannot receive the feed yet.**
  bioarena `main` (f4987b0) has no receiver on 8411 and picks the AUTO winner
  at random or forced when AUTO starts.
- **A fuel-only set made with a relative `--src` could not be trained.**
  `subset_classes.py --src dataset_relabel` wrote `path:
  dataset_relabel-fuel`. `train.py` read that against the yaml's own
  directory, found no labels, and stopped. On the MI300X this happened after
  the 2.6 h scout run, so the fuel run never started. `subset_classes.py`
  now writes an absolute path. `train.py` rewrites any relative `path:` to
  the yaml's directory, and its "no labels" message names the directories
  it actually searched.
- **QUICKSTART's "see what the model sees" was killed part-way through a
  match video.** It called `predict(..., save=True)` without `stream=True`,
  so Ultralytics held every frame's result, decoded image included, until
  the end: ~6 MB a frame at 1080p against ~12,900 frames. On a 36 GB M4 Max
  macOS killed it at frame ~8,000 on one run and 10,008 on another. It now
  streams, and passes `max_det=1000`, since the default 300-box cap was
  reached on every frame of an Einstein video.

### Experimental

- `experiments/area_hub_count.py`: counts fuel into each hub from the yellow
  area crossing a hand-drawn funnel outline, without the model or long
  tracks, so a drum shooter's clump of balls counts as several. Tested on
  one match only, 2026 Einstein Playoff Match 4, and its settings were chosen
  on that same match. Against that match's scoreboard it counted blue 444
  of 479 (93%) and red 769 of 807 at the buzzer (95%). On the same video
  `run.py count` counted 0 and `run.py shots` 13/23 (cropped) and 44/123
  (full frame). Hub totals only, not per robot. Blind-tested on Einstein
  Match 5 with the same settings, it did not transfer: blue 713 against 594
  (120%) and red 931 against 682 (137%). It still stayed flat whenever a hub
  was inactive, but it counted too many balls per burst. Treat it as a
  record of what was tried, not a scorer. The script's docstring has the
  details.

## v0.3.0 — 2026-09-26

The first release with trained models. It adds counting fuel into hubs,
per-robot shot scouting (`run.py shots`), robot labelling without hand
labels, and training on an AMD MI300X. It covers everything since v0.2.0;
v0.2.1 was tagged without notes.

### Models (attached to this release, not in the source archives)

| file | classes | trained on | validation (one held-out nhdur match) |
| --- | --- | --- | --- |
| `fuel_best.pt` | fuel | 721 frames, 2026nhdur, 100 epochs | mAP50 0.613, mAP50-95 0.321, P 0.666, R 0.623 |
| `fuel_withBotbest.pt` | fuel, robot_blue, robot_red | 659 frames, 2026nhdur, early-stopped at 62 epochs | fuel mAP50 0.581; robot_blue 0.699; robot_red ~0.71 |

Both are YOLO26s at `--imgsz 960`. Check the downloads with `sha256sum`:

```
202429d3e8a7640e444e8053b229998b363f698469274829a069dd015994b5f3  fuel_best.pt
0c5267368787e054f9fbd57c49078726306ca3370b7185752c3c2229dfffd14e  fuel_withBotbest.pt
```

What the numbers do and do not mean:

- Validation labels come from the same automatic labellers the models were
  trained on, so the scores measure agreement with the labellers, not with
  reality. Both models have only seen one event's broadcast (2026nhdur).
- On a real match video (nhdur qm7, first 30 s) the broadcast scoreboard
  showed 74 balls scored; the shot counter's hub totals saw 19. Counting from
  a broadcast camera angle is **not** a replacement for FMS scoring. Use a
  close camera per hub, and a human scorekeeper.
- Robot detection is good: every robot box checked by eye on qm7 was a real
  robot in the right alliance colour. Per-robot **misses** are the least
  reliable output and should not be used as scouting data yet.

### Added

- **QUICKSTART.md**: from a fresh clone to running the released models --
  install, download and checksum the weights, preview detections, hub boxes,
  `run.py count`, `run.py shots` with `--teams`, the speed check -- and what
  the measured results say to trust. Linked from the top of the README.

- **`run.py detect` — a trained `.pt` finally has somewhere to go.** Until now
  `ultralytics` appeared in exactly one file, on the training side: a finished
  model loaded nowhere, the `detections` table was written by nothing, and
  `identify.py` waited on "a tracker upstream" that did not exist. Everything
  downstream of the detector was designed and unreachable. This runs the model
  over the exported frames in play order with tracking persisted between them,
  and records boxes, classes, confidences and tracks — each tagged with the
  weights that produced it, so two models' opinions can never be read as one.
  Idempotent per match, so better weights replace a match rather than
  accumulating beside it.
  It does **not** close the per-robot gap and does not pretend to:
  `assign_tracks` still needs a scorer, `identify.py` still measures a bumper
  number at ~5 px tall, and with no scorer it records nothing rather than
  naming whichever team sorts first. `--assign` reports that as the answer.
- **`run.py count` — scored fuel from the detector alone, no scoreboard and no
  OCR** (`tbavid/count.py`, plus `deploy/frc-counter.service`). Everything else
  that produces a fuel number reads the broadcast's burned-in counter, which is
  right when there is one and no answer at all on a field that renders none — a
  practice field, an offseason event, a demo, somebody's own game system. The
  same `.pt` already has a `fuel` class and a hub per alliance, so this counts
  the event itself: a fuel track vanishing inside a hub region.
  Almost all of it is refusing the three things that look identical to that — a
  one-frame false detection, a ball a robot drove in front of (it never crossed
  *into* the hub), and a ball that passed *over* it and comes back the other
  side. The last cannot be settled in the moment, so a score is held for a few
  frames and a reappearance withdraws it, which is the same shape as
  `clean_series` wanting two reads before believing a large jump. Every refusal
  is counted and reported, because a counter that rejects silently is one
  nobody can debug.
  Note the deliberate opposite of `detect`: that one discards fuel track ids
  because it runs at 3 fps where a ball moves further between samples than its
  own width; this runs at a camera's native rate where tracking the ball is the
  whole method.
- **`run.py count --scoreboard` — the scoreboard at a scrimmage**
  (`tbavid/field.py`). With no FMS, the count is not a cross-check against a
  real score, it *is* the score, and a number on a laptop nobody can see, with
  no clock and no way to correct it, is not a scoreboard. So: a match clock, so
  fuel thrown about between matches does not score and the detector stops at
  the buzzer; the score handed out as JSON over stdlib HTTP and as one line
  per change on stdout, with no display of its own because whatever shows the
  score at a field already exists; and a referee's correction, which works
  after the buzzer because that is when corrections happen. `detected`
  and `adjusted` are kept apart in the record and on screen, because "the
  camera missed two" and "the camera saw two that never happened" are different
  facts about a setup. Points per ball are configurable and default to 1, since
  `db.py` already refuses to convert fuel to points and a scrimmage runs
  whatever rules its organiser chose.
- **The training code produces a model for counting, not just for scouting.**
  `train/subset_classes.py` derives a `fuel`/`hub_blue`/`hub_red` dataset from
  the labelled five-class one, remapping the label indices — the step that
  fails silently if it is wrong, since a model trains perfectly happily on fuel
  labelled as hubs. `train.py` gained `--data`, so a derived set can actually
  be trained (it previously hardcoded the five-class path, and counted that
  one's labels while training against another), and `--export`, because on a
  CPU box ONNX is often the difference between keeping up with the camera and
  not. `train/benchmark.py` measures achievable frame rate on the machine you
  will use, reporting the p95 as well as the median: a model averaging 30 fps
  that stalls for 200 ms every few seconds drops balls in the stalls while
  looking fine on the average.
  With two class orders now in play, a model that does not carry its own class
  names is refused by both `detect.load` and `count.run_source` rather than
  assumed to be the five-class one — that assumption would relabel every
  detection without failing anything.
- **The counter checks itself, because nothing else can.** With no FMS there
  is no second number anywhere that would disagree with a wrong count, and the
  failure that matters is silent: a box too slow for the camera misses balls
  between the frames it does see, and the score comes out low with no gap and
  nothing odd about it. `--expect-fps` gives it the camera's rate and it
  reports whether it is keeping up, what fraction of frames went past unseen,
  and every reason it refused a ball — in the same payload as the score.
  Without that rate it reports `keepingUp: null` rather than guessing, since it
  cannot tell a slow processor from a slow camera. And `POST /clock` follows an
  outside clock, which is the only quantity at a scrimmage that can be compared
  against anything — moving it never moves the score.
- **Deployment is documented end to end** in [DEPLOY.md](DEPLOY.md): three
  roles with three different dependency sets, in the order to do them, with
  the `.pt` as the only step left. `requirements-detect.txt` keeps torch and
  ultralytics out of `requirements.txt` on purpose — the API host runs
  `serve.py` with python3 and nothing else, and a test now asserts that the
  whole serving path imports no third-party package, because breaking that
  promise would only show up on a machine nobody is sitting at.
- **`run.py live` — scout a live feed and keep no video.** Everything else
  here builds a training set, which is no use to somebody who wants to know how
  an alliance is scoring this afternoon. This reads the burned-in fuel counter
  off a live stream and deletes every frame the moment it has been read: peak
  disk is one chunk, nothing enters the manifest or the dataset, and the clip
  is unlinked on the failure paths too. What it gives is the per-alliance
  scoring timeline for the match on the field — when fuel went in, to the
  second — which nothing else in either repo produces live. What it cannot give
  is which robot: the counter says an alliance scored and never which of its
  three did, and being live does not move that ceiling.
  It refuses to guess the match: `--match` names it or `--hub` asks a running
  scouting hub what is on the field, and one is required, because a timeline
  filed against the wrong key credits an alliance's scoring to six robots that
  were not on it. Rows are marked `status='live'`, since a reading that cannot
  be re-read is not the same evidence as one that can.
- **`run.py stream` — one broadcast in, one clip per match out.** Events that
  publish a single multi-hour stream per day rather than per-match uploads were
  simply unreachable: the picker looks for `match.videos[]` and finds nothing,
  and `download.py` refuses the stream it is handed. Now the matches are found
  inside it and cut out, and the clips go through the same pipeline as anything
  else. `--file` reads one already on disk; `--listen-only` reports what the
  audio contains without writing anything.
- **Matches are found by listening, not watching** (`tbavid/audio.py`). Between
  matches the camera is looking at the same field from the same place, so there
  is no cut for `shots.py` to find. The field's start sound and buzzer are the
  only events in a broadcast that are loud, tonal, repeated dozens of times and
  separated by a fixed interval — and the fixed interval is what identifies
  them. There is no frequency in the file: a band-pass at the horn's pitch
  would need that pitch from somewhere, and a wrong guess finds nothing while
  looking exactly like a stream with no matches in it. Every loud tonal burst
  is collected and the spacing that repeats most consistently is the match, so
  it works on the Webcast Unit's YouTube feed, a Twitch re-encode or a phone in
  the stands alike.
- **`deploy/frc-harvest.service` and `run.py db export`, so the API is not
  something somebody has to start.** `python3 serve.py` on a laptop dies with
  the lid, and the scouting app then shows an empty column that looks exactly
  like "no footage of these robots" rather than "nothing is listening" -- two
  states it goes out of its way to distinguish. Under systemd it survives a
  crash, a reboot and a host rebuild, and it carries no secrets because a
  GET-only API over a rebuildable database has nothing to authenticate to.
  `db export` exists because `cp` is not good enough and fails confusingly: the
  working database is WAL, a WAL database must create its `-shm` companion
  before even a read-only connection can read it, and the unit mounts its data
  directory read-only on purpose -- so a copied database starts fine and then
  answers every request with "attempt to write a readonly database". Confirmed
  as an unprivileged user against a 0555 directory. The export goes through
  SQLite's backup API, which also avoids catching the file mid-write, and
  leaves one file with no sidecars. See [deploy/HOSTING.md](deploy/HOSTING.md).
- **Identity comes from TBA or not at all.** Audio says where a match is, never
  which one it is, and `db.py` joins a roster onto the match key — so one
  mislabelled clip credits an alliance's fuel to six robots that were not on
  the field. Keys are assigned when the confirmed cue count equals TBA's match
  count, when recovered cues close the gap to it exactly, or when
  `--from-match` says where the day starts. Otherwise none are, and that is not
  a failure: training frames do not need a match key, scouting rows do, so an
  unidentified clip keeps its frames and enters the database under its own
  video id where no roster can join onto it.

- **Broadcast layout profiles** (`tbavid/formats.py`). The crop has always
  measured the overlay from the pixels, which needs no list of events kept up
  to date. What it could not do is know when it was wrong, and it failed
  silently both ways: a divider found on a single-camera feed crops away the
  bottom of the field, and frames that lost half a field still look plausible.
  A profile says what a layout is supposed to be, so a measurement can be
  checked against it. The measurement still wins — a profile only tunes the
  thresholds, supplies the fallback when detection is inconclusive, and vetoes
  a band its layout does not have.
- **`ca_district`, for FIRST California's weekend district events** —
  `2026caclv`, `2026casnf`, `2026calas`, `2026caven`, `2026caoec` and
  Aerospace Valley. One field camera, a banner across the top, no permanent
  side panel, so its main job is that veto. Marked `declared` rather than
  `measured`: the veto and thresholds are in force, but the banner range
  describes the layout as specified, because no California district footage has
  been through this pipeline yet — and those events are not one production
  (most are FIRST Webcast Unit on YouTube; Central Valley is on Twitch), so
  the range is wide and per-event calibration is worth doing.
- **A profile can be gated to a TBA `event_type`,** and `ca_district` is gated
  to a weekend district event. FIRST California's state championships
  (`2026cancmp` and its southern counterpart) carry district `ca` and are a
  different, larger production: selecting them into a profile whose whole
  contribution is a no-split-screen veto would keep an entire side-camera
  panel in the training set if they do run one, which is the failure the
  profiles exist to prevent with the sign flipped. They fall through to
  `generic`, which has no opinion. An event whose type could not be read fails
  the gate rather than having it waived, and config still outranks it — that
  gate stops the pipeline guessing, not a person who has measured one.
- **`run.py formats`** — list the profiles, ask which one an event would get
  and why, or measure a real download with `--calibrate`. Calibration prints
  the profile its measurements imply and writes nothing; promoting a profile
  from `declared` to `measured` stays a person's edit, after they have looked
  at the frames.
- **The district comes from TBA.** `/events/{year}/simple` — the one request
  per season the video picker already makes — carries each event's `district`,
  so the layout is known before the download and costs no extra request. A
  regional reads as no district, which is also how a regional is recognised.
  `crop.format` and `crop.formats` override it per run or per event.
- Every manifest entry records which profile it got and why, and the crop notes
  carry the reconciliation, so a refused divider is visible afterwards rather
  than being a number that quietly differs.
- **`deploy/AMD_DEVCLOUD.md` — training on an AMD Instinct MI300X.** The
  Colab and Kaggle pages assume a small GPU you lose in twelve hours; 192 GB
  that stays up wants different settings, and the obvious ones are wrong. With
  ~1,100 training frames, a batch big enough to fill the card leaves 11
  optimizer steps per epoch and converges worse than the 8 GB M2 did, so the
  page spends the memory on resolution, the P2 head and a bigger checkpoint
  instead, and on eight independent runs rather than one DDP job. It also
  documents the failure that eats an afternoon: `pip install ultralytics`
  inside a ROCm image resolves torch from PyPI, PyPI's torch is the CUDA
  build, it silently replaces the ROCm one, and the run falls back to the CPU
  with nothing in the log to say so.
- **`train.py` prints which chip it got.** ROCm reports AMD hardware through
  `torch.cuda`, so `device: 0` said nothing about whether that was an MI300X
  or a mistake, and `device: cpu` scrolled past as one line. It now names the
  GPU and its memory, reports the HIP version, and on a CPU fallback says
  which of the two causes it is looking at.
- **`train.py --batch` takes a fraction or `-1`, plus `--workers`, `--cache`
  and `--no-amp`.** `--batch 0.70` fills 70% of a card whose memory you have
  not measured; `--workers 32 --cache ram` is what stops a fast GPU idling
  while eight cores decode JPEGs; `--no-amp` is the fix for the NaN-loss and
  zero-mAP symptom AMP produces on some ROCm builds.

### Fixed

- Release archives now include `QUICKSTART.md` and `requirements-detect.txt`
  (the packager ships an explicit list, and neither was on it -- the quick
  start's own install step would have failed from an archive).
- `requirements-detect.txt` allowed `ultralytics>=8.3`, which cannot load
  YOLO26 weights -- every model this project has released. It needs 8.4.

- **`autolabel_objects.py` deleted every fuel box it found.** Without
  `--append` it rewrote each label file from scratch, and the documented
  workflow runs it *after* `autolabel_fuel.py` — so one run threw away ~200
  fuel proposals per frame across the whole dataset, with no error and no
  symptom until a model trained on it stopped seeing fuel. Appending is now
  the default; `--overwrite` is the explicit way to start over, and `--append`
  is accepted and ignored so existing commands keep working.
- **Hubs no longer need the videos.** The script bailed on any match whose
  cleaned video was missing, but the video is only needed for the robots'
  temporal-median background — hub boxes are replayed from geometry in the
  database and need nothing. Since a frame bundle ships without videos (~330 MB
  a match), that meant every machine except the one that harvested got no hub
  labels either. It now writes what it can and says which part it skipped.
- **`autolabel_fuel.py` proposed a fraction of the fuel on a busy frame.**
  Fuel rests in loose groups, a group is one connected component, and the fill
  gate that correctly rejects a heap rejected every pair and triple with it —
  92 proposals on a frame holding several hundred balls. Components that fail
  the gate are now split with a distance transform and a watershed, one seed
  per ball centre, and each piece is sized against the median ball in its own
  band of the frame before being kept. On a synthetic frame of 18 balls in
  groups plus a 20-ball heap, proposals went from 12 (three of them merged
  pairs) to 18.
  Cutting only fixed the pairs, though, and the clusters hold more fuel than
  the open floor does. A ball inside a cluster is surrounded by yellow, so the
  mask has no seam to cut there at all; what is still visible is its shading.
  Those are now found with a Hough circle search over the gradient, radius
  range pinned to the band's measured ball — 30 of 30 on a hex-packed cluster
  where the watershed found none, and no box pair overlapping by more than 0.5
  IoU. `--no-hough` turns it off.
- **Fuel a robot was carrying was never labelled, and the floor's reflections
  were.** Two failures of the same colour gate, in opposite directions.
  A ball in a hopper reads V=103 at its highlight and V=58 at its rim against
  a floor of 90, so it survived as a 7x7 dot below `--min-area` and a robot
  holding four contributed nothing — the balls that decide whether a score is
  attributed at all. The highlight is now used as a seed and grown to the
  band's ball size, kept only where a much looser gate agrees it is yellow;
  Hough is no use there because a hopper bar cuts the gradient (one of four on
  a synthetic hopper, against four of four this way). Meanwhile the glossy
  floor gave most balls a mirrored copy below, which the gate happily labelled
  as fuel, roughly doubling the count where counting matters. A proposal with
  a brighter one above it, within `--reach` ball-heights and aligned to half a
  width, is now dropped as its reflection.
  On a synthetic frame of 13 balls (9 on the floor, 4 behind hopper bars), 9
  reflections and a yellow banner: 13 kept, 9 dropped, nothing on the banner.
  `--no-rescue` and `--no-reflections` turn each off, `--show-dropped` draws
  the removals in green, and `--close`, `--min-cover`, `--v-ratio`, `--reach`
  and `--loose-lo/--loose-hi` tune them. ~70 ms a frame all told.
- **The rescue pass boxed the arena wall.** Its saturation floor was absolute,
  and any floor low enough to admit a ball in shade also admits the tan wall,
  the rail and every washed-out surface behind them. Shading turns out to
  scale a pixel's value and leave its saturation alone — fuel reads S=191 in
  arena light and S=191 in a hopper, while the wall reads S=68 at any
  brightness — so the test is now relative to the balls that frame has already
  found (`--sat-ratio`, default 0.7). On a synthetic frame carrying a tan
  wall, a blown-out highlight and a yellow banner, the wall's box is gone and
  all 13 real balls stay.
  Rescued boxes are also re-centred on the yellow around the highlight rather
  than on the highlight itself, which sits off-centre toward the light: box
  centres landed within 1–3 px of the true ball centres instead of half off
  the ball.
- **The splitter carved the arena into balls.** The back of the pit, the far
  wall, the sponsor boards and the painted field border are all long yellow
  shapes, and a cluster splitter turned loose on one proposes a ball every few
  pixels across it. Saturation cannot help — a sponsor board measures S=177
  against fuel's S=191 — so two geometric guards were added instead.
  A **field line**, learned per frame: fuel is on the floor, so nothing above
  the floor is fuel. A percentile of the accepted boxes was tried first and
  fails for the obvious reason, that the false positives are themselves above
  the field and drag the line up with them — on a test frame with 13 round
  yellow objects in the crowd it landed at row 36 of 440 and excluded nothing.
  Density works where position does not: bin the balls by row, keep bins
  holding a quarter of the busiest bin, take the longest unbroken run. That
  put the line at row 166 and excluded all 13. `--roi-top` overrides it.
  A **stripe test** for long yellow shapes about a ball thick, since the
  painted border sits below the line and inside the field. A row of balls is
  scalloped where paint is flat, but balls overlapping by a third are nearly
  as smooth (0.119 against paint's 0.100), so the pixels vote too: a sphere
  casts a seam against its neighbour and paint has no brightness structure at
  all (0.132 against 0.000, or 0.012 for paint with texture on it). Flat by
  both measures, and only then, means paint. Tunable with `--flat-v`.
  On an arena scene carrying a pit band, a painted border, 13 crowd objects
  and a merged row of 27 real balls along a wall: 0, 0, 0 and the row intact.
  Split pieces now face the same saturation test as rescued ones.
- **Three faults the guards above introduced or left behind.** Reflections
  were filtered pass by pass, each list against itself, so a ball that came
  through the colour gate and a reflection that came out of the splitter were
  never compared and the reflection survived; they are now filtered across
  every pass at once. The field line clipped the balls in flight — the fuel
  actually being shot at a hub — because its margin was two diameters of the
  far ball; at four it clears an arc and still excludes the crowd (measured
  both with the crowd well above the flight zone and almost level with it:
  4 of 4 balls in the air kept, 0 of 13 crowd objects, either way), and
  `--roi-margin` tunes it. And the stripe test called anything longer than
  three ball-diameters a candidate, which a clump of four in a row is: raised
  to six, since the pit band and the painted border run tens of diameters and
  nothing shorter is worth the risk.
- **The reflection filter was eating fuel out of the piles.** Measured on a
  real broadcast frame rather than a synthetic one: of 23 boxes it dropped, 14
  kept 85% or more of their source's saturation. Those were not reflections —
  they were balls lower in a pile, shaded by the ones above, and shading
  scales value while leaving saturation alone (the same fact that finds a ball
  inside a hopper). A reflection does lose saturation, being the ball's colour
  mixed with the grey it reflects in, so both conditions are now required
  before anything is dropped. On that frame: 23 drops down to 9 and the total
  up from 207 to 221. On a synthetic with reflections modelled physically —
  blended toward the floor colour rather than merely dimmed — all 11 are still
  caught, and a ball stacked under another ball survives. `--sat-keep` tunes it.
- **A frame that cannot be labelled is now removed instead of mislabelled.**
  Checked against four other broadcasts, the labeller fell apart on all of
  them, and the cause is one number: every measurement here is scaled off the
  ISOLATED balls, and those matches keep their fuel in one corral. The frame
  this was tuned on shows 85 isolated balls; the four that failed show 6, 16,
  20 and 22, so ball size, ball saturation and the field line are all being
  read off noise. The decisive measure is what it leaves behind — of the
  yellow pixels on the frame, how many ended up inside a proposal: 59% on the
  working frame against 13%, 38%, 35% and 50%, which is a corral of several
  hundred balls in plain sight with nothing drawn on it.
  Writing those labels is worse than writing none, because Ultralytics reads
  a missing box as "nothing here" — so the frame would teach the detector
  that a mass of fuel is background, which is the thing it most needs to find.
  Frames below `--min-singles` or `--min-coverage` are now moved to
  `dataset/skipped/` and reported per match, with `--no-quarantine` to label
  them anyway. Re-running `prepare_dataset.py` restores anything moved.
- **`CLAUDE.md`** gives a Claude Code session on another machine the context
  this work was done in: the commands, the harvest and training flows, the
  stdlib-only rule for the API path, what the fuel labeller can and cannot
  label and why, and the pitfalls of the MI300X droplet — with a dated
  "current work" section meant to be deleted once it goes stale.
- **A robot driving through fuel is no longer a 0-for-41 shooter.** The first
  scouting model's run on 2026nhdur qm7 credited the robot plowing the centre
  pile with 42 shots and 41 misses in 15 s of auto. The balls it passed sat
  still while it drove away -- which "gets clear of the robot" as surely as a
  launch -- and the tracker gives pile balls new ids constantly, so each one
  "started at the robot". A shot must now move itself: a robot-width of its
  own travel inside 0.3 s (continuations of a broken flight measured from
  where it broke). Rejected balls are counted under `not_launched`.
- **Robots are numbered #1, #2, #3 in `run.py shots`, not by tracker id.**
  The tracker numbers every object it follows, fuel included, so robot 1307
  was labelled "R1283" in the first annotated video and read as a misread
  team number. Nothing reads bumpers; `--teams 1=1307` maps the new numbers.
  A robot the tracker loses and finds again under a new id keeps its number
  when it reappears within 1.5 robot-widths of where it was, same alliance,
  within 3 s: qm7 grew robots 16099, 15856, 16580 and 21187 mid-match, each
  splitting a real robot's tally.
- **Shots: flights capped at 2.5 s, hopper shots kept, launches seen from
  above.** From the annotated qm7 video (frames checked by eye):
  - Stitching broken flights walked from ball to ball through the fuel
    piles for 4-8 s -- a seven-piece "flight" from 22.2 s to 30.0 s -- ending
    as misses or as makes credited to the wrong robot. A shot not in a hub
    2.5 s after launch is now a miss, final, and no later piece joins it.
  - The not-launched rule timed its 0.3 s from a ball's first sighting, so a
    ball tracked in a hopper and shot seconds later was thrown away. It is a
    sliding window now.
  - 13 of 19 makes were unattributed. A launch zone reaching 0.8 of a
    robot's height above it was tried and attributed none of them (still 3
    blue, 10 red); from a broadcast camera "above a robot" is the floor
    behind it. It is off (`launch_up` 0).
  - A launch must be fast both on its own and relative to its robot. The
    run after the sliding window credited 1058 with 11 shots and 11 misses
    while it plowed the pile (a robot at speed pushes balls a robot-width in
    0.3 s). Relative travel alone then made it far worse -- 76 shots, 76
    misses -- because a still ball a fast robot drives away from moves fast
    relative to it. Both tests are required now.
  Robots split across several numbers (1058 as #5, #8, #11) are left split
  on purpose: `--teams 5=1058,8=1058,11=1058` folds them; merging by guess
  could mix two robots' shots.
- **Counting windows keep their length in time on 60 fps video.** Both
  counters measure "missing for N frames", "held for N frames" in frames,
  chosen on 30 fps footage; the harvested broadcasts are 60 fps, so every
  window was half as long -- flights ended and robots were forgotten twice as
  fast as intended. `run.py shots` and `run.py count` now scale them by the
  source's frame rate (read from the file, or `--expect-fps` for count).
- **`run.py shots` crashed at the very end of a run** printing its report,
  after the not-launched rule added an ignore reason the report had no line
  for -- losing every result. It names that reason now and cannot crash on
  an unnamed one.
- **`run.py shots` keeps one box per robot.** The same model sometimes boxes a
  robot twice (0.58 and 0.37 on one robot of the held-out match); every box
  became its own track, splitting that robot's shots. The less confident of
  two boxes that overlap that much is dropped each frame.
- **`train.py` no longer dies before epoch 1 on the newer MI300X image.**
  Every dataloader worker failed with "no response from torch_shm_manager":
  the helper program that file_system sharing starts could not load
  librocm-openblas.so.0, which the torch 2.12+rocm7.14 image keeps in the
  venv at `_rocm_sdk_core/lib/host-math/lib` where only Python's torch finds
  it; with that found it then missed libamdhip64.so.7 and
  librocprofiler-sdk.so.1. `train.py` now puts every library folder of the
  ROCm package and torch/lib on `LD_LIBRARY_PATH` for the helper.
- **Robots in piles of fuel are found: fuel is greyed out before YOLOE
  looks.** In the first label preview robot 1058, in the middle of the
  central pile, had no box and fuel boxes all over it -- the model would
  have learned that a robot in fuel is background, exactly when `run.py
  shots` needs it. With fuel-coloured pixels (autolabel_fuel.py's gate) set
  to grey first, four real frames with 16 robots marked went from 10 found
  to 14 at `--conf 0.1`; 1058 from nothing to 0.40, a red robot from 0.41
  to 0.91. The two clean frames alone: 6 of 9 to 8 of 9. Alliance is still
  read from the real pixels. Extra prompts ("robot covered in yellow
  balls") were tried and added nothing. `--no-grey-fuel` turns it off.
- **The red hub is no longer painted out as a robot.** "Too big" needed two
  other robots to compare with; a frame with one let the hub's top (3.4x the
  robot beside it) through as a robot of unknown alliance. One is enough
  now. A box inside a more confident robot box (fuel greying made YOLOE box
  parts of robots too) is dropped as a duplicate.
- **A robot of unknown alliance is painted out instead of costing its
  frame.** On the 906 nhdur frames, 176 were dropped for one robot whose
  bumper colour could not be read -- a fifth of the dataset, for one box
  each. That robot is now filled with flat grey (Ultralytics' pad colour),
  fuel boxes centred in it are removed, and the frame is kept: neither a
  guessed class nor a robot left as "floor". Originals go to
  `dataset/skipped/robots/original/`; the painted copy is written as a new
  file, so a symlinked dataset never paints the harvested frame. `--restore`
  now undoes everything: moved frames, painted frames, and robot lines in
  every label file. Checked end to end on real frames, symlink included:
  after `--restore` every image hashes as before. Painting the
  lower-confidence boxes as well, for robots YOLOE misses, was measured and
  not adopted: it covered 3 of 10 misses and 30% of one frame.
- **`autolabel_robots.py --dry-run` reported "(0 robot boxes)"** and "would
  labelled": it never counted boxes in a dry run. It does now. Labels written
  by a real run were never affected. First full dry run on the 906 nhdur
  frames: 591 kept at `--min-robots 2`; 176 dropped for a robot of unknown
  alliance, 120 for too few robots, 19 for more than three of one alliance.
- **Robot labelling: the blue ladder no longer passes as a robot, and the
  gate keeps frames.** On the first five real previews the "too big" rule
  compared each box with a median that included the two hubs (~56,000 and
  ~82,000 px^2), so the ladder (34,800 px^2, beside robots of 6,100-9,500)
  came out 3.7x and passed -- in the only frame the gate kept. The median is
  now over boxes not already rejected, and the limit is 3x (real robots
  differed by at most ~1.7x near to far). The same previews put YOLOE's
  recall at ~16 of 27 robots; no frame had every robot boxed, so
  `--min-robots` 4 kept nothing usable. The default is now 2, which keeps 3 of
  those 5 frames; the missed robots in them are left for the bootstrap.
- **Robot labels without hand-labelling: `train/autolabel_robots.py`.** Asks
  an open-vocabulary detector (YOLOE, `yoloe-26s-seg.pt`, prompted "robotic
  vehicle" / "robot" / "wheeled robot") for robots it was never trained on,
  over the whole frame and three 2x tiles. On two real frames with every robot
  hand-marked it found 4 of 5 and 2 of 5 at the default `--conf 0.1` (the
  corner robot scored 0.09), and nothing else once three rules
  are applied: too tall (the hubs), a box holding two others (one box around
  both red robots), and ~10x the area of the frame's other boxes (the red
  alliance wall). Alliance is a hue vote in the box's bumper band; the navy
  bumper of robot 69 has median saturation 8-93 and is left unknown rather
  than guessed. Recall is the weakness, so a frame is labelled only if every
  robot found has a readable alliance and at least `--min-robots` were
  found; otherwise it is moved to `dataset/skipped/robots/` with its label,
  and `--restore` brings them back. Both test frames were rejected, so how
  many of a real dataset survive is unmeasured: `--dry-run` reports it per
  `--min-robots` value before anything is written. Missed robots in frames
  that pass are still unlabelled; the bootstrap in `train/README.md` is the
  next round.
- **Robot proposals stop boxing the crowd and start finding the near robots.**
  The first real preview boxed three people in the stands and one robot of
  six. People in alliance colours who move are exactly what the robot test
  looks for, and nothing told it where the field was; meanwhile `--roi-bottom`
  defaulted to 0.80 and threw away every robot in the near fifth of the frame,
  which on that camera was most of them. Robots now have to sit below the
  field line learned from the fuel on the same frame, the line autolabel_fuel
  already uses, so it follows whatever camera is in use instead of assuming
  one. `--roi-bottom` is now 0.98, since static things by the near rail
  already fail the motion test. On a synthetic frame built to match that
  preview, the old settings gave 2 boxes in the stands and 0 near robots; the
  new ones give 0 and 2, plus the far robot. `--no-field-line` turns it off.
- **Robots can be auto-labelled without the match videos, and their boxes
  cover the robot, not just its bumper.** `autolabel_objects.py` finds robots
  as bumper colour that has moved against the match's empty-field background,
  and took that background from the cleaned match video — which a frame bundle
  does not include, so robots could not be proposed off the harvesting
  machine at all. The camera is fixed for a match, so the median of the
  match's own exported frames is that background: a robot is in any one place
  for a small share of the match and drops out. Those frames are also the
  ones the dataset images were copied from, so the background lines up pixel
  for pixel. Separately, the box it drew was the bumper band only, and a shot
  leaves from the top of the robot — so no launch would have started inside
  any robot's box and `run.py shots` could never have attributed one. Boxes
  now grow from the bumper up through the moving pixels connected to it,
  using a more sensitive motion mask than the bumper gate, since a dark frame
  over grey carpet differs from the empty field by only ~25 levels, under the
  bumper gate's 40. On a synthetic match: background clean, each box from
  bumper to shooter, two adjacent robots kept apart, a static blue ramp
  ignored. Not yet checked on real frames — `--preview` is the check.
- **`run.py shots` — per-robot scouting from a model over video.** Runs the
  detector with tracking over a match video or a camera, feeds robots and
  balls into `ShotCounter`, and prints each shot as it is decided and a
  per-robot summary: shots, made, missed, accuracy. `--teams 3=254,7=254`
  names robot tracks (several ids per team is normal — a tracker that loses a
  robot gives it a new id), `--annotate` writes a video with robot ids, balls,
  hubs and each outcome drawn on, which is how ids become team numbers and how
  shots get checked by eye, and `--out` writes it all as JSON. Shot times are
  match time when the frame rate is known, not processing time, and it warns
  below 15 fps, where a ball crosses the field between frames — exported 3 fps
  frames are no use for this. It refuses a model with no robot classes rather
  than producing a scouting sheet with no misses and nobody's makes. Tested end
  to end against a stand-in model, so everything but the network runs in CI.
- **`count.py` counts with a fuel-only model when you give it the hubs.** It
  refused any model without hub classes, even with both hub boxes passed in —
  and the first trained model is fuel-only. Hub classes are now needed only
  when the hubs must be learned. On a fixed camera, drawing two boxes during
  setup is easier and more exact than detecting them every frame anyway. The
  refusal still names every missing class at once, and now also says how to
  pass the boxes.
- **`tbavid/shooting.py` — who shot, and the misses.** `count.py` knows how
  many balls went into each hub; scouting needs whose they were, and the
  misses, which `count.py` cannot see at all since a ball that never reaches a
  hub is not an event there. `ShotCounter` follows each ball from the robot
  that launched it to wherever it ended and files **made**, **missed** or
  **wrong hub** against that robot track. A shot is a ball that starts at a
  robot *and gets clear of it* — measured from the robot's box as it is now,
  so a ball riding in a hopper is not a shot every time its robot drives. A
  make with no visible shooter is kept as *unattributed*, so per-hub totals
  always equal `BallCounter`'s for the same balls; a test holds them to that.
  Flights the tracker breaks are stitched back together by where the ball was
  heading. The first detector's per-frame recall of 0.62 makes those routine,
  and each would otherwise read as a miss by the shooter plus a make by
  nobody. That includes a flight broken before the ball cleared its robot,
  which the first version called "carried". `by_team()` folds robot tracks
  into teams once something assigns them. Logic only: nothing trains robot
  detection yet, and it needs native-frame-rate video, not 3 fps frames.
- **Training no longer dies on a dense batch after a few good epochs.** On
  the MI300X droplet a clean run (mAP50 0.49 → 0.54 over epochs 2–4) died in
  epoch 5 with `received 0 items of ancdata` and `Pin memory thread exited
  unexpectedly` — nothing about files, though that is the cause. PyTorch's
  dataloader hands tensors between processes as open file descriptors; a
  batch of 16 frames carrying 6,454 fuel labels, times 16 workers, ran the
  container past its open-file limit. `train.py` now uses the `file_system`
  sharing strategy on Linux, which goes through `/dev/shm` instead.
- **The MI300X recipe drops `--p2` and trains at 960.** `yolo26s --p2
  --imgsz 1280` died in epoch 1 inside the loss's `TaskAlignedAssigner`: one
  16.6 GiB allocation refused with 177 GiB free. Cutting the batch from 16 to
  4 was the first fix and did nothing — the request was the same 16.6 GiB —
  so batch is not what sizes it. Every crash had P2 at 1280, and the warmup
  at 960 without P2 ran clean, so the recipe uses that until the two are
  separated.
- **A dataset built on one machine now trains on another.** `prepare_dataset.py`
  and `subset_classes.py` write an absolute `path:` into the yaml — they have
  to, since Ultralytics resolves a relative one against its own datasets
  directory rather than the yaml's — so a set built on a Mac said
  `/Users/.../dataset-fuel`, and the first run on an AMD droplet stopped with
  "images not found" after the GPU was already up. The Colab and Kaggle
  notebooks each rewrote the line by hand; nothing else did. `train.py` now
  repoints `path:` at the yaml's own directory whenever the recorded one is
  missing and the images are beside the yaml, and prints that it did.
- **`deploy/AMD_DEVCLOUD.md` describes the machine you actually get.** On the
  PyTorch 1-Click image torch lives in a Docker container named `rocm`, not on
  the host, so the guide's `python3`/`pip` steps failed as written on the
  host. It now separates host commands (`docker`, `scp`, `rocm-smi`, `tmux`)
  from container ones (`python3`, `pip`, `train.py`), copies the data in with
  `docker cp`, and warns that files written in the container die with it.
  Batch and workers drop to 16 each: the 1x plan has 20 vCPU, and a first
  harvest of ~700 training frames wants optimizer steps more than memory.
- **`train/drop_offcamera.py` removes frames that are not the main camera.**
  A crowd shot that survived the harvester's shot cut reached the labeller,
  which proposed 49 boxes on spectators' yellow shirts — and every quality
  check passed it, because they ask whether the visible yellow ended up inside
  a box and on that frame it did. Nothing in a colour gate can know the yellow
  is a T-shirt, and nothing in the per-frame checks separated it: the dark
  arena's real field frame scores lower on floor area and on ball-size spread
  than the crowd shot does.
  What separates them is that the camera does not move within a match, so the
  match's own median frame is what the field looks like and a crowd shot is a
  different picture rather than a bad one. Each frame is scored by its mean
  absolute difference from that median on a 64×24 thumbnail, as a modified
  z-score against the match's own spread (median and MAD, so a handful of
  crowd shots cannot raise their own bar). On a synthetic match of 40 field
  frames and 3 crowd shots: field frames peak at z=1.1, crowd shots score 238,
  and the default threshold of 8 sits between them with a 216× margin.
  Run it before `autolabel_fuel.py`; `--dry-run` lists without moving.
- **`prepare_dataset.py --matches`** builds a dataset from named matches or a
  whole event. The fuel labeller is a colour heuristic and does not survive
  every broadcast: a wide shot whose fuel sits in one corral gives it 9–22
  isolated balls to measure from against 85 on a close one, and ball size,
  ball saturation and the field region are all derived from those. It is
  better to train on the broadcasts it handles than on labels it got wrong
  everywhere else.
- **The preview names which pass proposed each box** — red through the gate,
  orange split out of a cluster, cyan rescued from shade, green dropped as a
  reflection, with the learned field line drawn in white. `train/diagnose_labels.py`
  reports the same counts across a sweep of a setting, which is how you tell a
  fix that did not work from a fix that never ran.
  The band-local size also catches the merge the fill gate never could: two
  balls side by side fill their bounding box to 0.82 and passed as a single
  ball of twice the local size. New knobs: `--no-split`, `--merge-factor`,
  `--max-split`, `--hsv-lo/--hsv-hi` for broadcasts whose yellow sits outside
  the default gate, and `--preview-frame` to preview a specific frame instead
  of the middle one. The preview now separates gated boxes (red) from
  recovered ones (orange) and counts the heaps it skipped.

### Changed

- `CLAUDE.md` moved to `.claude/CLAUDE.md`, out of the repo's front page;
  Claude Code reads project instructions from either place.

### Removed

- `deploy/make_release.sh`: its header said three documents call it; none
  do. Releases come from pushing a tag (`.github/workflows/release.yml`), and
  `python deploy/package.py release <version>` still builds one locally.
- `deploy/amd_watch.sh`: marked untested and never run; the MI300X guide
  copies weights out with `docker cp`, which is what was actually used.

## v0.2.0 — 2026-09-16

Harvesting is now restricted to official competition footage, and an existing
harvest can be checked against that same standard.

### Added

- **Competitive-only harvesting.** Only official competition events are walked:
  regional, district, district championship, championship division,
  championship final and Festival of Champions. Offseason, preseason and
  unlabeled events are never picked. Within an event only real match play
  counts (`qm`, `ef`, `qf`, `sf`, `f`), so practice matches are skipped even
  when TBA has a video for them.
- **`run.py audit`** reports which videos in an existing harvest are not
  competition footage, and how many frames each one contributes. It reports
  only and deletes nothing. A season TBA does not answer for is reported as
  unknown rather than condemned.
- **`DATA.md` and `docs/provenance/`** record what is in the current dataset,
  where the frame bundle lives, and the data-quality problems found in it.

### Changed

- Event listing uses `/events/{year}/simple` instead of `/events/{year}/keys`,
  because a bare event key does not carry `event_type`. Still one request per
  season, still cached and revalidated with `If-None-Match`.

### Added (release process)

- **Releases are cut by pushing a tag.** `.github/workflows/release.yml` runs
  the suite, scans the tree for a committed key, builds all three archives with
  `deploy/package.py`, extracts the notes from this file, proves the built
  archive runs by extracting it and running the suite from inside it, then
  creates the GitHub release with the archives and `SHA256SUMS` attached.
  Nothing is built or uploaded by hand, and the checksums in the notes are
  always the ones from the artifacts actually attached.
- `workflow_dispatch` builds and verifies a version without publishing, so a
  release can be rehearsed before its tag exists.
- `python3 deploy/package.py notes vX.Y.Z` prints a release's section from this
  file.

### Fixed

- **`run.py pull` crashed on every video it successfully downloaded.**
  `_process()` read `shard` and `shards`, which are parameters of `fetch()`,
  not of it — `NameError: name 'shards' is not defined`. The conditional
  tested `shards > 1` first, so it raised on single-worker runs too, and it
  raised at the very end: after the download, shot analysis, crop, render and
  scoreboard OCR had all completed for that video. Present since the initial
  commit and in the `Beta` release. The shard is now passed in, and
  `reprocess` preserves the one already recorded instead of erasing it.
- A test walks the symbol table of every function in `tbavid/`, `run.py` and
  `serve.py` and fails on any global that no module global defines. The crash
  above needed a real video to reach, which the suite deliberately has none of.
- `deploy/make_code_archive.sh` built its file list from a hardcoded string and
  aborted once `QUICKSTART_DEBIAN.txt` was deleted from the repo — leaving a
  half-written archive behind that had never reached the keyless check. The
  list is now built from what is on disk, and a failed run removes its own
  partial output.
- `deploy/make_release.sh` no longer stages the deleted `QUICKSTART_DEBIAN.txt`,
  and now includes `DATA.md`, `CHANGELOG.md`, `LICENSE` and `docs/`.
- Packaging works on Windows and macOS, not just Linux. Both `make_*.sh`
  scripts are now thin wrappers around `deploy/package.py`, which uses
  `zipfile`, `tarfile`, `lzma` and `hashlib` from the standard library instead
  of `zip`, `tar`, `xz` and `sha256sum` — macOS has no `sha256sum` and Git Bash
  usually has no `zip`. `release` now also writes `dist/SHA256SUMS`.

### Escape hatch

- `--include-noncompetitive` on `pull`/`fetch`, or `"competitive_only": false`
  in `config.json`, restores the previous behaviour. The flag wins over config.

### Known issues

- The frame bundle harvested before this release contains 926 frames (12.0%)
  from `2026kylou`, an offseason event. `run.py audit` finds them; nothing
  deletes them for you.
- `2026gal_qm62` carries a bad scoreboard read (blue final 726 against a 19–285
  range elsewhere) and is not flagged as an error.
- A one-alliance scoreboard read is still recorded as a good read: the
  `scoreboard_ok` test in `tbavid/db.py` is satisfied by a series containing
  only one alliance. `2026mnmi2_qm11` and `2026mnmi_qm39` are affected.

## v0.1.0-beta — 2026-09-15

Released as tag `Beta`. First working end-to-end pipeline: pick unseen match
videos from TBA, strip them to main-camera footage, crop the scoreboard off,
export sampled frames, and read the fuel counters before discarding the banner.
Includes the scouting database, the read-only JSON API, and deployment notes
for Debian, Windows, Colab and Kaggle.
