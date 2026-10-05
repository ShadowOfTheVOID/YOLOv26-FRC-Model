# Hub Counter app

The hub fuel counter (`run.py hubgui`) as a program you double-click: the
setup page opens in your browser. Nothing to install, no terminal. Only the
hub counter is in it; the TBA scraper and training tools are not.

## Get it

From the repository's **Releases** (v0.4.1 on), download the one for your
computer. Between releases: **Actions** tab -> the latest `hub-app` run ->
**Artifacts**.

| computer | download | open |
|---|---|---|
| Mac (Apple silicon, M1-M4) | `HubCounter-mac.zip` | unzip, drag **Hub Counter** to Applications, double-click |
| Windows 10/11 | `HubCounter-windows.zip` | unzip (Extract All), open the **Hub Counter** folder, double-click **Hub Counter.exe** |
| Linux PC | `HubCounter-linux-x64.tar.gz` | extract, open **Hub Counter**, run **Hub Counter** |
| Raspberry Pi 4/5 (64-bit Pi OS) | `HubCounter-linux-arm64.tar.gz` | same as Linux, on the Pi's desktop |

Keep the Windows and Linux folders whole: the program needs the
`_internal` folder beside it.

## First launch

None of the builds is signed, so each system warns once:

- **Mac:** "cannot verify". Click **Done**, then open **System Settings ->
  Privacy & Security** and click **Open Anyway**. Allow **camera** and
  **local network** access when asked. Without them the cameras show
  nothing and counts never reach bioarena / frc-fms.
- **Windows:** "Windows protected your PC". Click **More info -> Run
  anyway**. Allow it on **Private networks** when the firewall asks.
  Camera access: **Settings -> Privacy & security -> Camera -> Let desktop
  apps access your camera** must be on.
- **Linux / Pi:** if it will not start, right-click it -> Properties ->
  allow executing as a program (or `chmod +x "Hub Counter"`).

## Use

- The page is the same as `run.py hubgui`; `deploy/HUB_FEED.md` explains it.
- Your setup is `Documents/Hub Counter/cams.json` (in your home folder) and
  is opened again next time. Count logs (`hubfeed_*.csv`) and
  `hubcounter.log` go there too.
- Closing the browser tab leaves the counter running; open the program again
  to get the page back. **Quit** (top right) stops it.
- The page only answers on the computer it runs on. A Pi with no screen is
  set up differently (its page reachable from your laptop); that is not
  this app.

## Not in the app

The fuel-model blend (`--model`, the page's *Fuel model*) needs PyTorch,
which would make each build over 1 GB. The page says when it is missing.
For the blend, use the terminal version (`deploy/FRC_FMS.md`, step 1).

## Build it yourself (on the OS you want it for)

```bash
pip install pyinstaller opencv-python-headless numpy yt-dlp
pyinstaller apps/hubcounter/HubCounter.spec
python apps/hubcounter/smoke_test.py "dist/Hub Counter/Hub Counter"   # Mac: dist/Hub Counter.app/Contents/MacOS/Hub Counter
```
