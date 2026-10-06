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

## What a double-click does

1. **First launch only:** makes `Documents/Watchtower/config/event.yaml` with
   new random PINs (scorekeeper, ref, emcee) and a vision key. It is kept
   from then on.
2. **Starts Watchtower** on port 8000, open to phones on the same Wi-Fi.
3. **Starts the hub counter** with its address box already set to this
   Watchtower and its key, so counts arrive with nothing typed.
4. **Opens a start page** with every link and PIN, and the address for
   phones.

Then:

- **Event name, date and teams:** edit them in
  `Documents/Watchtower/config/event.yaml`, Quit, and open the app again.
- **Hub cameras:** draw the outlines, measure a ball, and press Start.
- **Quit:** the Quit button on the hub camera page stops both. Opening the
  app while it runs brings the start page back.
- **Mac:** it keeps the Mac awake while running, as `run.sh` does.

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
