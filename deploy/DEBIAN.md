# Running the harvester on Debian (e.g. an old MacBook Pro)

Entirely command line. No IDE, no GUI, no desktop environment needed — SSH in
or use a TTY.

## Does Debian have Python?

Yes. Debian 12 ships **Python 3.11** and the code needs only 3.9+, so nothing
to do there.

Two things are *not* included by default and catch people out:

- `pip` and `venv` are separate packages (`python3-pip`, `python3-venv`).
- Debian marks the system Python **externally managed** (PEP 668), so a plain
  `pip install requests` fails with `externally-managed-environment`. Use the
  apt packages below and the problem never arises.

## Install

```bash
sudo apt update
sudo apt install -y ffmpeg tesseract-ocr python3-requests python3-numpy
```

That covers everything except yt-dlp. **Do not `apt install yt-dlp`** — the
packaged version lags by months and a stale yt-dlp simply stops working against
YouTube. Take it straight from upstream:

```bash
mkdir -p ~/.local/bin
curl -fsSL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
  -o ~/.local/bin/yt-dlp && chmod +x ~/.local/bin/yt-dlp
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

Or `pip install --user -U yt-dlp` if you prefer — either is fine, both stay
current. There is a scripted version of all of the above:

```bash
./deploy/debian_setup.sh            # apt where sensible, needs sudo
./deploy/debian_setup.sh --nosudo   # static binaries into ~/.local/bin
```

No root at all? ffmpeg can come from a static build (the script does this), but
tesseract cannot easily. Set `"score_labels": false` in `config.json` and you
still get frames — just no scoreboard labels.

## Which files

Only the harvester. Training and the API are not involved.

```
run.py
config.json
tbavid/          all of it (~76 KB)
.env             your TBA key, or export TBA_AUTH_KEY
```

Build the archive on the Mac:

```bash
./deploy/make_code_archive.sh --with-key     # for a machine you control
./deploy/make_code_archive.sh                # keyless, for Kaggle/Colab
```

### Getting it across, with a desktop on the Debian side

**Browser (easiest).** On the Mac, in the project folder:

```bash
python3 -m http.server 8000
```

Then open `http://<mac-ip>:8000` on the Debian box and click the file. Ctrl-C
the server afterwards — it has no authentication, so anything on the network
can read that folder while it runs. Use the **keyless** archive for this, and
type the key into `.env` by hand.

**File manager.** Nautilus/Thunar -> Other Locations -> Connect to Server ->
`smb://<mac-ip>` (enable File Sharing on the Mac first).

**scp**, if you would rather stay encrypted and are moving the keyed archive:

```bash
sudo apt install -y openssh-server        # on the Debian box
scp tbavid_code_WITH_KEY.tgz you@debian-ip:~/
```

A USB stick is also completely fine for 96 KB.

```bash
tar -xzf tbavid_code.tgz && cd TBACroppedOutVid   # or wherever you extract
echo 'TBA_AUTH_KEY=your-key-here' > .env
chmod 600 .env
```

The archive deliberately contains **no** `.env`, no database and no video.

## Run it

```bash
python3 run.py pull -n 3 --dry-run     # proves TBA access, downloads nothing
python3 run.py pull -n 3               # the real thing
python3 run.py verify                  # check counters against TBA
python3 run.py status
```

Note it is plain `python3`, not a venv — the apt packages are on the system
path.

### Settings that matter on an old machine

In `config.json`:

```json
"prefer_h264": true,
"keep_raw": false,
"keep_clean": false
```

`prefer_h264` avoids AV1, which has no hardware decode before ~2020 and is
punishing in software on a 2012 CPU. `keep_clean: false` skips an x264 encode
of a file that would be deleted anyway — worth several minutes a match here.

### Overnight

`deploy/overnight.sh` is macOS-specific (it uses `caffeinate`). On Debian:

```bash
nohup python3 run.py pull -n 20 --per-event-cap 2 > harvest.log 2>&1 &
tail -f harvest.log
```

### Keeping it awake with the lid shut

GNOME has no "when the lid is closed" setting — it was removed from Settings
and lid handling was delegated to systemd. So do it the systemd way.

**Best: scope it to the run.** `systemd-inhibit` exists for exactly this and
needs no config change, no root and no reboot. The inhibition lasts only as
long as the command:

```bash
systemd-inhibit --what=handle-lid-switch:sleep:idle \
  --why="harvesting match video" \
  nohup python3 run.py pull -n 20 --per-event-cap 2 > harvest.log 2>&1 &
```

Check what is currently holding things awake with `systemd-inhibit --list`.

**Permanent, if you would rather set it once.** A drop-in beats editing
`logind.conf` because package upgrades leave it alone:

```bash
sudo mkdir -p /etc/systemd/logind.conf.d
printf '[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchExternalPower=ignore\nHandleLidSwitchDocked=ignore\n' \
  | sudo tee /etc/systemd/logind.conf.d/99-nolid.conf
sudo systemctl restart systemd-logind
```

Restarting logind can end your graphical session, so save anything open first,
or just reboot.

**Also turn off idle suspend**, which *is* still in the GUI: Settings -> Power
-> Automatic Suspend -> off (at least on AC). That is a separate mechanism from
the lid and will stop a long run on its own.

**Simplest of all:** leave the lid open and let the screen blank. Blanking does
not suspend anything.

### A desktop costs you RAM you may want

A full GNOME session can hold 1.5-2 GB. If the machine has 4 GB that is a
meaningful slice, though harvesting is CPU-bound rather than memory-hungry and
will still run. If it starts swapping, log out of the desktop and work from a
TTY (Ctrl-Alt-F3) or over SSH — or install a lighter session (XFCE, LXQt).
Check with `free -h` while a pull is running.

## Expected speed

Measured on an M2 Mac mini: **~2.2 min per match**. Scaled by CPU benchmarks,
a 2012 Ivy Bridge i5/i7 should land around **9-10 min**, a 2009/2010 Core 2 Duo
around **22 min**. Those two are estimates, not measurements.

Time one match before planning a night around it:

```bash
time python3 run.py pull -n 1
```

The slowest stage is tesseract on the scoreboard. If that dominates, setting
`"score_labels": false` roughly halves the per-match time — at the cost of the
fuel-count labels.
