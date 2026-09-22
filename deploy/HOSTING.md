# Running the API so nobody has to start it

`python3 serve.py` on a laptop is fine for a look. It is not fine as the thing
the scouting app depends on: it stops when the lid closes, and the scouting
hub then shows an empty `ALLIANCE·VID` column that looks exactly like *this
harvest has no footage of these robots* — two states that app goes out of its
way to tell apart (see its `server/vision.py`). A service that has to be
started by hand is a service that is down.

So put it on a host, under systemd, next to the scouting app's off-site
mirror. It comes back after a crash, after a reboot, and after the host is
rebuilt.

**Harvesting does not run there.** `run.py pull` and `run.py stream` want
ffmpeg, yt-dlp, numpy and hours of a machine's attention. The API wants
`python3` and one file, and that asymmetry is the whole reason `serve.py`
imports nothing but the standard library. Build the database where the videos
are; serve a copy.

## 1. Export a copy that can be served

```bash
python3 run.py db sync            # on the machine with the videos
python3 run.py db export          # -> data/serve/scouting.db
```

**`cp` is not good enough here, and it fails in a way that looks like a code
bug.** The working database runs in WAL mode, and a WAL database needs to
create its `-shm` companion file before *anything* can read it — a strictly
read-only connection included. Copy one to a host whose data directory is
mounted read-only, which is what the unit below does deliberately, and the API
starts cleanly and then answers every single request with `attempt to write a
readonly database`. Verified as an unprivileged user against a `0555`
directory, which is exactly what systemd's `ReadOnlyPaths=` produces.

`db export` goes through SQLite's own backup API — which also waits for a
consistent snapshot rather than catching the file mid-write, the other way `cp`
gets this wrong — and switches the copy out of WAL. What lands is one file with
nothing beside it.

## 2. Put it on the host

```bash
sudo useradd --system --home /opt/YOLOv26-FRC-Model frcharvest
sudo git clone https://github.com/ShadowOfTheVOID/YOLOv26-FRC-Model /opt/YOLOv26-FRC-Model

sudo cp deploy/frc-harvest.service /etc/systemd/system/
sudo systemctl enable --now frc-harvest
```

There are no secrets to set, and that is not an oversight: the API is GET-only
over a database rebuildable from its own manifest, so there is nothing to
authenticate to. Compare the mirror's unit, which carries two.

Then copy the database in and restart:

```bash
scp data/serve/scouting.db host:/tmp/scouting.db
sudo install -o frcharvest -g frcharvest -m 444 \
     /tmp/scouting.db /var/lib/frc-harvest/scouting.db
sudo systemctl restart frc-harvest
curl -s localhost:8781/health | head -c 200
```

Restart rather than reload: `serve.py` opens the file at start and the process
is cheap to replace. A new harvest is a new copy in and one restart.

## 3. TLS, if the scouting hub is not on the same host

The unit binds `127.0.0.1` only. Left there, the API is reachable by anything
on that host and nothing else, which is all you need if the mirror is on it
too.

To reach it from a hub elsewhere, put it behind the same Caddy the mirror uses
— `deploy/Caddyfile.harvest` here is a drop-in block for that file. Then set
the address in the scouting app's admin panel under **VIDEO HARVEST**.

Worth knowing before you expose it: **there is no authentication.** What is in
there is match video metadata, frame filenames, scoreboard readings and
detection boxes — nothing private, nothing that identifies a person — but it is
readable by anyone who finds the URL, and it is a database somebody could scrape
cheaply. The Caddyfile block includes a commented `basic_auth` line if you would
rather it were not.

## What still needs the hub running

The scouting hub is what polls this API and folds the numbers into the bundle it
pushes to the mirror. So with the hub off, the mirror keeps serving **the last
copy it was pushed** — which is the right answer, because this data barely
moves: harvesting is something you do between events, not during a match.

The mirror does not poll this API itself, and that is deliberate on its side:
it "does not scout, does not solve, holds no API key for anybody, and never
opens a connection of its own." Adding an outbound poller to it would trade
that away for freshness this source does not need.
