# Scouting database and API

`data/scouting.db` is a SQLite database rebuildable at any time from
`data/review/manifest.json` and `data/labels/*.csv`. Nothing lives only here.

```bash
.venv/bin/python run.py db sync     # build from manifest + fetch rosters from TBA
.venv/bin/python run.py serve       # read-only JSON API on 127.0.0.1:8781
.venv/bin/python run.py db export   # a copy a host can serve read-only
```

**Don't run `serve` off a laptop if a scouting app depends on it.** It stops
when the lid closes, and the app then shows an empty column that looks exactly
like "no footage of these robots" rather than "nothing is listening". Put it
under systemd on a host instead — one unit, no secrets, nothing to start by
hand: [deploy/HOSTING.md](deploy/HOSTING.md). Copy the database with `db
export` and not `cp`; that page says why, and it is not a small difference.

## Two ways rows get here

`db sync` builds everything from the manifest, which is the dataset's record.
`run.py live` writes straight into these tables instead, from a live broadcast,
keeping no video — so those rows have no `video_id`, no crop and no frames, and
carry `status='live'`. That is the right shape rather than a gap: nothing was
kept to point at, and a reading that cannot be re-read is worth being able to
tell apart from one that can.

## Tables

| table | one row per | notes |
| --- | --- | --- |
| `events` | event | holds `hub_blue` / `hub_red` geometry — see below |
| `matches` | match | crop box, coverage, final fuel per alliance, scoreboard status |
| `match_teams` | team-in-match | alliance + station + team number, from TBA |
| `score_events` | scoring event | `t_source`, alliance, `balls`, running `total` |
| `frames` | exported frame | timestamps in both clocks, fuel totals and lookahead |
| `detections` | detected object | `cls`, box, `alliance`, `team`, `track_id` |

Two views do the scouting arithmetic: `team_match_fuel` (a team's alliance fuel
per match) and `team_summary` (matches played, total and average alliance fuel,
counting only matches whose scoreboard read cleanly).

**`alliance_fuel` is alliance-level, not per-robot.** The scoreboard says an
alliance scored; it never says which of its three robots did. Per-robot credit
requires `detections.team`, which is what the identity layer below produces.

## API

`run.py serve` — GET only, CORS open, read-only connection.

| route | returns |
| --- | --- |
| `/health` | row counts |
| `/schema` | tables, columns and routes, so the app can discover the shape |
| `/events` | events with match counts |
| `/matches?event=&team=&scoreboard_ok=` | match list |
| `/matches/<match_key>` | match + teams + full scoring timeline |
| `/teams` | `team_summary` |
| `/teams/<number>` | summary + per-match rows |
| `/frames?match=&scored=1` | frames; `scored=1` keeps only those with fuel scored just after |
| `/detections?match=&frame=&team=&cls=` | detections joined to their frame |

All list routes take `limit` (max 5000) and `offset`.

```bash
curl -s 'http://127.0.0.1:8781/matches/2026gal_qm62' | jq '.teams, .match.blue_fuel'
curl -s 'http://127.0.0.1:8781/frames?match=2026gal_qm62&scored=1&limit=5' | jq
```

## Hub geometry

The camera is fixed for a whole event, so the two hub boxes are identical in
every frame of every match there. Record them once:

```bash
.venv/bin/python run.py db hub --event 2026gal --alliance blue --box 300,60,120,190
```

`train/autolabel_objects.py` then replays them into every frame of that event.
Read coordinates off any exported frame — they are in cleaned-frame pixels.

## Identifying individual robots

Bumper OCR does not work on this footage, and it is worth being precise about
why: at 1920x504 a bumper number is about **5 px tall** and motion-blurred.
Upscaled 6x and given to tesseract at three page-segmentation modes it returns
the empty string every time. That is missing detail, not a tuning problem.

`tbavid/identify.py` takes the route the data supports instead. TBA names the
exact six teams on the field, so identity is a six-way choice rather than an
open one, and a robot persists over hundreds of frames:

1. Score a bumper crop against **only those six** candidates.
2. Decide per **track**, not per frame — sum weak per-frame scores so the few
   sharp frames carry the decision.
3. Assign globally, so two tracks on one alliance cannot claim the same team.

`assign_tracks(con, match_key, alliance, scorer)` does this and writes
`detections.team`. With no `scorer` it records nothing rather than guessing: a
wrong team number in a scouting database is worse than a missing one.

**This needs a tracker upstream.** Train the detector, run
`model.track(..., tracker="bytetrack.yaml")`, write detections with
`track_id`, then call `assign_tracks`.

## Labelling robots and hubs

```bash
.venv-train/bin/python train/autolabel_objects.py --preview /tmp/check.jpg
.venv-train/bin/python train/autolabel_objects.py --append
```

Robots are found by alliance colour **and** departure from the temporal median
background — the ramps, walls and half the crowd are also blue and red, but
they don't move — then filtered by shape, since a bumper is wider than tall and
a standing person is not. Hubs come from the recorded event geometry.

These are proposals with real false positives (people in alliance colours near
the rail). The intended workflow is to correct a subset, train, and let the
detector label the rest — that bootstrap is cheaper than fighting the colour
gate.
