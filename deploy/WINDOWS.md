# Running on Windows

The pipeline itself is portable — pure Python, `pathlib` throughout, no
`shell=True`, no POSIX-only calls. What does not carry over is the `.sh`
tooling in `deploy/`, and one symlink issue that is now handled automatically.

## Install

**Python** — [python.org](https://www.python.org/downloads/) (3.9+). Tick
**"Add python.exe to PATH"** during install; nothing works otherwise.

**ffmpeg** — easiest via winget:

```powershell
winget install Gyan.FFmpeg
```

Or download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), unzip, and
add its `bin` folder to PATH.

**yt-dlp and Python packages:**

```powershell
pip install --user -U yt-dlp requests numpy
```

**tesseract** (optional — only for scoreboard labels):

```powershell
winget install UB-Mannheim.TesseractOCR
```

Check everything resolves:

```powershell
python -V; ffmpeg -version; yt-dlp --version; tesseract --version
```

If `yt-dlp` is "not recognised", pip's user scripts folder is not on PATH —
add `%APPDATA%\Python\Python3xx\Scripts`.

## Your API key

PowerShell, in the project folder:

```powershell
Set-Content -Path .env -Value "TBA_AUTH_KEY=your-key-here" -NoNewline
```

Or set it for the session: `$env:TBA_AUTH_KEY = "your-key-here"`

Note `.env` will not get restrictive permissions the way it does on
macOS/Linux; `chmod` has no real equivalent. Treat the file accordingly.

## Run it

Same commands, `python` rather than `python3`:

```powershell
python run.py pull -n 3 --dry-run
python run.py pull -n 20 --per-event-cap 2
python run.py verify
python run.py status
```

### Long runs

There is no `nohup`. Either leave the window open, or:

```powershell
Start-Process -NoNewWindow python -ArgumentList "run.py pull -n 20 --per-event-cap 2" `
  -RedirectStandardOutput harvest.log -RedirectStandardError harvest.err
Get-Content harvest.log -Wait          # the equivalent of tail -f
```

Stop the machine sleeping mid-run:

```powershell
powercfg /change standby-timeout-ac 0
```

(`powercfg /change standby-timeout-ac 30` puts it back.)

## What does not work on Windows

| | |
| --- | --- |
| `deploy/*.sh` | bash only — use Git Bash or WSL, or follow the steps by hand |
| `deploy/overnight.sh` | uses `caffeinate`; see `Start-Process` above |
| symlinks in `prepare_dataset.py` | **handled** — it detects Windows and copies instead |

Everything else — harvesting, crop detection, scoreboard OCR, the database,
the API, training — behaves the same.

## Splitting work with Windows teammates

Sharding is platform-independent, so mixed teams are fine:

```powershell
python run.py pull -n 20 --shard 2/4
```

The partition is `crc32(video_id) % 4`, which is identical on every OS. A
Windows teammate and a Linux one with different shard numbers will never pull
the same match.

To send results back:

```powershell
python run.py export-share $HOME\my_harvest
Compress-Archive -Path $HOME\my_harvest\* -DestinationPath harvest.zip
```

Whoever is collecting unzips it and runs `python run.py merge <folder>`.

## Caveat

None of this has been tested on Windows — the code audit is real (no
POSIX-only calls, no hardcoded paths) but it has only been *run* on macOS and
Debian. If something breaks, `python run.py status` and the traceback are the
useful things to report.
