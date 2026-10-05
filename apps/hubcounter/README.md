# Hub Counter.app

The hub fuel counter (`run.py hubgui`) as a Mac app: double-click it and the
setup page opens in your browser. Nothing to install, no terminal. Only the
hub counter is in it; the TBA scraper and training tools are not.

## Get it

From the repository's **Releases**, download `HubCounter-mac.zip` (from v0.4.1
on; between releases, from the **Actions** tab: the latest `mac-app` run ->
**Artifacts**). Unzip it and drag **Hub Counter** into Applications.
It is built for Apple silicon (M1-M4).

## First launch

The app is not signed by Apple, so macOS stops it the first time:

1. Double-click **Hub Counter**. macOS says it cannot verify it. Click **Done**.
2. Open **System Settings -> Privacy & Security**, scroll down, and click
   **Open Anyway** next to "Hub Counter was blocked". Confirm with your password.
3. Allow **camera** access when asked (or the cameras show nothing), and
   **local network** access (or counts never reach bioarena / frc-fms).

After that it opens with a double-click like any app.

## Use

- The page is the same as `run.py hubgui`; `deploy/HUB_FEED.md` explains it.
- Your setup is `Documents/Hub Counter/cams.json` and is opened again next
  time. Count logs (`hubfeed_*.csv`) and `hubcounter.log` go there too.
- Closing the browser tab leaves the counter running; open the app again to
  get the page back. **Quit** (top right) stops it.

## Not in the app

The fuel-model blend (`--model` / the page's *Fuel model*) needs PyTorch,
which would make the app over 1 GB. The page says when it is missing. For
the blend, use the terminal version (`deploy/FRC_FMS.md`, step 1).

## Build it yourself (on a Mac)

```bash
pip install pyinstaller opencv-python-headless numpy yt-dlp
pyinstaller apps/hubcounter/HubCounter.spec
open "dist/Hub Counter.app"
```
