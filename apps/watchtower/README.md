# Watchtower app

The Watchtower FMS v2, with its hub cameras, in one double-click program.
Nothing to install and no terminal: no launch command, no Python, no
website account, no vision key to type.

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

**Updates**: Menu -> **Check for updates**; never automatic, and refused
while a match is running. On Mac and Windows it
closes Watchtower, installs the new version in place and reopens it (about a
minute); event.yaml, matches and logs live in `~/Documents/Watchtower` and
are kept. Linux and the Pi get a link to the release page.

## What it is

The Watchtower FMS, version 2 (arnan-bajaj/watchtower-fms, the same
Watchtower the website's users run), as one desktop app. A double-click
starts it; nothing is installed, no terminal, no website account.

- **One window: Control.** Watchtower's own page, exactly as on the
  website's events: the match on the field, scoring review, rankings,
  alliance selection, playoffs, and **Setup**, which holds everything else:
  event name, date, teams, schedule, PINs, TBA keys, the field link to
  bioarena (and Field control, which starts the field from Control), Match13
  predictions, and the **hub cameras**: each hub's camera, picture
  (exposure, white balance, ...), fuel colour with a magenta preview,
  Measure, the outline, and the **drop test** that sets the ball size.
  Each hub's card walks you through those steps in order. Setup -> Run and
  Test have the live counts, the ball test, speed, recording and checking a
  recording.
- **The menu** (Watchtower): Field display, Stream overlay (for OBS), Match
  queue, Open event folder, Check for updates, Quit.
- **Phones and tablets** on the same Wi-Fi: refs, scorers, emcee and the field
  display open the addresses Control -> Setup lists (PINs there too). Allow
  Watchtower when the firewall asks; on Windows, tick Private networks.
- **Hub cameras** run beside it as Watchtower's own camera runner (the colour
  counter, copied from this repository's `tbavid/`), restarted if it stops.
  None for a REEFSCAPE event, which scorers count on `/score`.
- **Match queue** at `/queue` on the phone address: the match on the field and
  every match still to play, live, no PIN. `/queue?team=841` jumps to a team,
  `/queue?tv=1` fills a pit TV, `/queue?event=2026casj` shows TBA's schedule
  (needs `tba.read_key` in event.yaml or `TBA_AUTH_KEY`).
- **Quit:** close the window or Menu -> Quit. While it runs the computer is
  kept awake.

Everything lives in `Documents/Watchtower`: `config/event.yaml` (made on first
launch with new PINs; Setup edits it for you), `data/` (matches, fuel,
backups), `recordings/`, and the logs `watchtower.log` and `vision.log`. An
event.yaml from the v0.5 app is read as it is.

On Linux and the Raspberry Pi Control opens in the browser instead of its own
window (no window toolkit is bundled there); stop it with Ctrl+C or by ending
the program.

Until v0.6 this app wrapped Watchtower v0.1 with a Home window of our own and
our Hub Counter page beside it. Watchtower v2 moved all of that into Control ->
Setup, so the app now shows Watchtower's own pages. The Hub Counter app is
unchanged and still feeds any Watchtower or bioarena over the network.

The Watchtower inside is arnan-bajaj/watchtower-fms at the commit in
`.github/watchtower-release`. This repository holds no copy of its code; the
build fetches it.

## Build it yourself

As for the Hub Counter app (no model needed), plus:

```bash
pip install fastapi uvicorn websockets pyyaml requests pywebview
git clone https://github.com/arnan-bajaj/watchtower-fms.git watchtower-src
git -C watchtower-src checkout "$(cat .github/watchtower-release)"
HUBAPP=watchtower pyinstaller --noconfirm --workpath build/watchtower apps/hubcounter/HubCounter.spec
python apps/watchtower/smoke_test.py "dist/Watchtower/Watchtower"
```
