# Watchtower app

The Watchtower FMS and the hub counter in one double-click program. Nothing
to install and no terminal: no clone, no venv, no `fms.init`, no `run.sh`,
no vision key to type.

## Get it

From the repository's **Releases**, download the one for your computer:

| computer | download | install |
|---|---|---|
| Mac (Apple silicon) | `Watchtower-mac.dmg` | open it, drag **Watchtower** onto **Applications** |
| Windows 10/11 | `Watchtower-Setup-windows.exe` | run it: Start-menu shortcut (desktop optional) and an uninstaller, no admin needed |
| Linux PC | `Watchtower-linux-x64.tar.gz` | extract, run **Watchtower** |
| Raspberry Pi 4/5 (64-bit Pi OS) | `Watchtower-linux-arm64.tar.gz` | same as Linux |

The first open shows the same one-time "unverified app" warning as the Hub
Counter app (`apps/hubcounter/README.md`): on Mac, System Settings → Privacy &
Security → **Open Anyway**; on Windows, **More info → Run anyway**.

**Updates** (from the release after v0.4.4): Overview shows "Update
available" with an **Update** button when a newer release is out; never
automatic, and greyed out while the hub cameras count. On Mac and Windows it
closes Watchtower, installs the new version in place and reopens it (about a
minute); event.yaml, matches and logs live in `~/Documents/Watchtower` and
are kept. Linux and the Pi get a link to the release page.

## What it is

A desktop app with its own windows, like any other program: no browser, no
address bar, no terminal, and no files to edit.

Watchtower combines our camera setup with Watchtower's own parts:

- **Home window** (opens on launch):
  - **Overview:** whether the field system, the network and the cameras are
    working, and what is still missing before the first match (teams, date,
    name).
  - **Hub cameras:** our camera setup page itself, inside Home: cameras,
    camera presets (ELP OV4689 60 fps, USB webcam, your saved ones),
    picture and fuel colour, hub outlines, ball size, fuel model, Start. It
    is the same page as the Hub Counter app, already sending its counts to
    this Watchtower, and it keeps counting while you use the other tabs.
    Saved camera presets are shared with the Hub Counter app
    (`~/Documents/Hub Counter/camera-presets.json`).
  - **Phones & PINs:** the address phones open, as text and as a QR code, and
    the scorekeeper, ref and emcee PINs, with a button for new PINs.
    **Field: bioarena** shows, read-only, where the hub counter feeds
    bioarena. That is the first `feeds:` entry in Watchtower's
    `config/vision.yaml`, or else the feed spec's `10.0.100.5:8411`.
    Nothing is typed. The counter sends every ball to Watchtower on this
    computer (port 8000) and to bioarena's UDP feed, which decides the AUTO
    winner and lights the hubs. Watchtower does not pass counts on to
    bioarena itself: only its own vision runner does, and the app does not
    run it. The camera page's *Send counts to* reads
    `http://KEY@127.0.0.1:8000, 10.0.100.5:8411`. With no bioarena at the
    venue the counter still reads as connected through Watchtower, and the
    page names bioarena as the target not answering.
    While Watchtower is off, watchtower.systemoverload.org shows
    `deploy/watchtower-offline.html`, an offline page in Watchtower's own
    look that reloads by itself once Watchtower is back
    (`deploy/Caddyfile.watchtower` sets it up).
  - **Event & schedule:** name, date, time zone, teams, qualification start,
    match cycle and lunch.
  - **The Blue Alliance:** event key; a **Write key** section (Auth ID and Auth Secret, both masked, plus how to request one) that sends results; an optional **Read key** for importing teams.
  - **Game rules:** period and shift lengths, points, fouls and
    ranking-point thresholds.
- **Scorekeeper and Field display**, Watchtower's own pages, each open in
  their own window from Overview (Watchtower keeps their logins, which a
  page framed inside Home would lose). Drag Field display to the projector and press Full
  screen.
- **Match queue** at `/queue` on the phone address (and the public
  address): the match on the field and every match still to play, with
  teams and planned times, updating live. No PIN, so share it with teams.
  A team types its number to see its next match and how many come before
  it; `/queue?team=841` links straight to that, `/queue?tv=1` fills a pit TV.
  `/queue?event=2026casj` shows that event's schedule from The Blue
  Alliance instead, and until a schedule is entered here the page shows
  TBA's for this event's key; both need the Read API key on The Blue
  Alliance tab.
- **Saving settings** checks them first (the scorekeeper PIN must differ from
  the others, times must be HH:MM, ...), writes them, and restarts the app in
  a few seconds. Matches are kept.
- **Phones and tablets** on the same Wi-Fi use the address on Phones & PINs
  (Watchtower always serves the network). Allow Watchtower when the firewall
  asks; on Windows, tick Private networks.
- **Quit:** close the Home window, or press Quit, to stop everything. On a Mac
  it keeps the computer awake while running.

On first launch it makes `Documents/Watchtower/config/event.yaml` with new
PINs. The app edits that file for you. On Linux and the Raspberry Pi the same
app opens in the browser instead of its own windows (no window toolkit is
bundled there), with the same pages.

The Watchtower inside is arnan-bajaj/watchtower-fms at the tag in
`.github/watchtower-release` (v0.1.0). The app holds no copy of its code in
this repository; the build clones it.

## Build it yourself

As for the Hub Counter app, plus:

```bash
pip install fastapi uvicorn websockets
git clone --branch v0.1.0 https://github.com/arnan-bajaj/watchtower-fms.git watchtower-src
HUBAPP=watchtower pyinstaller --noconfirm --workpath build/watchtower apps/hubcounter/HubCounter.spec
python apps/watchtower/smoke_test.py "dist/Watchtower/Watchtower"
```
