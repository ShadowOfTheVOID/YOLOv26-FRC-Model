# Watchtower app

The Watchtower FMS and the hub counter in one double-click program. Nothing
to install and no terminal: no clone, no venv, no `fms.init`, no `run.sh`,
no vision key to type.

## Get it

From the repository's **Releases**, download the one for your computer:

| computer | download |
|---|---|
| Mac (Apple silicon) | `Watchtower-mac.zip` |
| Windows 10/11 | `Watchtower-windows.zip` |
| Linux PC | `Watchtower-linux-x64.tar.gz` |
| Raspberry Pi 4/5 (64-bit Pi OS) | `Watchtower-linux-arm64.tar.gz` |

Unzip and open it as for the Hub Counter app (`apps/hubcounter/README.md`,
including the one-time "unverified app" warning on Mac and Windows).

## What it is

A desktop app with its own windows, like any other program: no browser, no
address bar, no terminal, and no files to edit.

- **Home window** (opens on launch):
  - **Overview:** whether the field system, the network and the cameras are
    working, and what is still missing before the first match (teams, date,
    name).
  - **Phones & PINs:** the address phones open, as text and as a QR code, and
    the scorekeeper, ref and emcee PINs, with a button for new PINs.
  - **Event & schedule:** name, date, time zone, teams, qualification start,
    match cycle and lunch.
  - **The Blue Alliance:** event key, auth ID and secret, sending on/off.
  - **Game rules:** period and shift lengths, points, fouls and
    ranking-point thresholds.
- **Scorekeeper, Hub cameras and Field display** each open in their own
  window from Overview. Drag Field display to the projector and press Full
  screen.
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
