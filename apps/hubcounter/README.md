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
- The page answers only on the computer it runs on until you press
  **Share** (below).

## Open it on other devices: Share on Wi-Fi

Press **Share** (top right). The page then also opens on any phone or
computer on the **same Wi-Fi**, at the address it shows (port 8791), after
the 6-digit PIN it shows. The scoreboard (`/board`, for a TV) opens without
a PIN. Only the computer running the counter can quit it or stop sharing.
Five wrong PINs from one device lock it out for a minute.

If the other device cannot connect, the network may keep devices apart
(common on venue and school Wi-Fi; use your own router), or the firewall
blocks port 8791 (allow Hub Counter when asked).

## No app: run it as a website

The app is the same page as the terminal version, so you can skip it. From
a checkout of this repository:

```bash
python3 run.py hubgui --setup cams.json          # opens http://127.0.0.1:8790
python3 run.py hubgui --setup cams.json --share  # also on Wi-Fi; prints the address and PIN
```

On a Mac, double-clicking `hubfeed.command` does the first one and installs
what it needs. On a Pi with no screen,
`python3 run.py hubgui --setup cams.json --share --pin 4821 --no-browser`
lets you set it up from your laptop's browser at `http://<pi-address>:8791`.

## The fuel model (colour + model blend)

Every build has PyTorch and the fuel model (`fuel_relabel.pt`) inside, so
each download is 300-400 MB. Open **Fuel model** on the page and press
**Use built-in model**; or pick another `.pt` with **Choose model**.

| computer | runs the model on | in practice |
|---|---|---|
| Mac, Apple silicon | its GPU (MPS) | fast enough; the one to use for the blend |
| Windows / Linux PC | the CPU | the builds carry the CPU PyTorch: a CUDA one is ~3 GB, over GitHub's limit. Usually too slow |
| Raspberry Pi | the CPU | too slow; use colour only |

When the model cannot keep up it turns itself off after about 10 s, the
page says so, and the hub counts by colour alone, so turning it on cannot
make a slow computer miss balls. For the blend on an NVIDIA PC, use the
terminal version (`deploy/FRC_FMS.md`, step 1) with CUDA PyTorch.

On the Central Valley broadcast the blend was worse than colour alone
(34.5% against 6.4%), so compare both on your own camera before choosing.

## Build it yourself (on the OS you want it for)

```bash
pip install torch torchvision     # Linux (x64 or Pi): add --index-url https://download.pytorch.org/whl/cpu
pip install pyinstaller opencv-python-headless numpy yt-dlp
pip install --no-deps ultralytics ultralytics-thop
pip install matplotlib pillow pyyaml requests psutil polars
mkdir -p models   # put v0.4.0's fuel_relabel.pt here, or the build has no built-in model
pyinstaller apps/hubcounter/HubCounter.spec
python apps/hubcounter/smoke_test.py "dist/Hub Counter/Hub Counter"   # Mac: dist/Hub Counter.app/Contents/MacOS/Hub Counter
```
