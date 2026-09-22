"""SQLite store behind the scouting app.

Everything the pipeline learns about a match ends up here: what was played,
who played it, when fuel scored, and which frames show it. The manifest and
label CSVs stay the pipeline's working files; this is the queryable copy the
app talks to, and it is always rebuildable from them.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import DATA, FRAME_DIR, LABEL_DIR, MANIFEST_PATH

DB_PATH = DATA / "scouting.db"

# One definition of the class list. prepare_dataset writes it into
# dataset.yaml, so the index a label carries and the index the model trains on
# cannot drift apart.
CLASSES = ["fuel", "robot_blue", "robot_red", "hub_blue", "hub_red"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS events (
    event_key     TEXT PRIMARY KEY,
    year          INTEGER,
    -- Hub geometry is per event, not per match: the camera is fixed for the
    -- whole event, so the two hub boxes are annotated once and reused.
    hub_blue      TEXT,
    hub_red       TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    match_key     TEXT PRIMARY KEY,
    event_key     TEXT NOT NULL REFERENCES events(event_key),
    video_id      TEXT UNIQUE,
    yt_key        TEXT,
    comp_level    TEXT,
    label         TEXT,
    title         TEXT,
    status        TEXT,
    source_secs   REAL,
    kept_secs     REAL,
    coverage      REAL,
    width         INTEGER,
    height        INTEGER,
    crop_x        INTEGER, crop_y INTEGER, crop_w INTEGER, crop_h INTEGER,
    clean_path    TEXT,
    -- Measured off the broadcast counter.
    blue_fuel     INTEGER,
    red_fuel      INTEGER,
    -- Official, from TBA. Related to the counter but NOT the same quantity:
    -- on a checked match the counter read 726 while TBA's auto+teleop came to
    -- 755, and the 2026 fuel-to-points rule is not pinned down here. Keep both
    -- and let the app choose rather than inventing a conversion.
    blue_points   INTEGER,
    red_points    INTEGER,
    tba_breakdown TEXT,
    scoreboard_ok INTEGER DEFAULT 0,
    scoreboard_note TEXT
);

CREATE TABLE IF NOT EXISTS match_teams (
    match_key     TEXT NOT NULL REFERENCES matches(match_key) ON DELETE CASCADE,
    alliance      TEXT NOT NULL CHECK (alliance IN ('blue','red')),
    station       INTEGER NOT NULL,
    team          INTEGER NOT NULL,
    PRIMARY KEY (match_key, alliance, station)
);

CREATE TABLE IF NOT EXISTS score_events (
    id            INTEGER PRIMARY KEY,
    match_key     TEXT NOT NULL REFERENCES matches(match_key) ON DELETE CASCADE,
    t_source      REAL NOT NULL,
    alliance      TEXT NOT NULL CHECK (alliance IN ('blue','red')),
    balls         INTEGER NOT NULL,
    total         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS frames (
    id            INTEGER PRIMARY KEY,
    match_key     TEXT NOT NULL REFERENCES matches(match_key) ON DELETE CASCADE,
    file          TEXT NOT NULL UNIQUE,
    t_clean       REAL,
    t_source      REAL,
    blue_total    INTEGER, red_total INTEGER,
    blue_next     INTEGER, red_next  INTEGER,
    -- Persisted rather than recomputed each export: a split that silently
    -- reshuffles between runs makes two models incomparable.
    split         TEXT
);

-- One row per training run, so a .pt file can always be traced back to the
-- data and settings that produced it.
CREATE TABLE IF NOT EXISTS training_runs (
    id            INTEGER PRIMARY KEY,
    started_at    TEXT,
    finished_at   TEXT,
    name          TEXT,
    model         TEXT,
    imgsz         INTEGER,
    epochs        INTEGER,
    batch         INTEGER,
    device        TEXT,
    train_frames  INTEGER,
    val_frames    INTEGER,
    train_boxes   INTEGER,
    classes       TEXT,
    weights       TEXT,
    map50         REAL,
    map50_95      REAL,
    precision     REAL,
    recall        REAL,
    notes         TEXT
);

-- One row per detected object. team is NULL until a bumper number is read or
-- a track is resolved, which is what makes per-robot scouting possible.
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY,
    frame_id      INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    cls           TEXT NOT NULL,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    conf          REAL,
    alliance      TEXT,
    team          INTEGER,
    track_id      INTEGER,
    source        TEXT
);

CREATE INDEX IF NOT EXISTS ix_match_event   ON matches(event_key);
CREATE INDEX IF NOT EXISTS ix_teams_team    ON match_teams(team);
CREATE INDEX IF NOT EXISTS ix_score_match   ON score_events(match_key, t_source);
CREATE INDEX IF NOT EXISTS ix_frames_match  ON frames(match_key, t_source);
CREATE INDEX IF NOT EXISTS ix_det_frame     ON detections(frame_id);
CREATE INDEX IF NOT EXISTS ix_det_team      ON detections(team);
CREATE INDEX IF NOT EXISTS ix_det_source    ON detections(source);
CREATE INDEX IF NOT EXISTS ix_frames_split  ON frames(split);

-- Alliance fuel per team-match. Attribution is at alliance level: the
-- scoreboard says an alliance scored, never which of its three robots did.
-- Per-robot credit needs detections.team populated; see robot_credit.
CREATE VIEW IF NOT EXISTS team_match_fuel AS
SELECT mt.team, mt.match_key, m.event_key, mt.alliance,
       CASE mt.alliance WHEN 'blue' THEN m.blue_fuel ELSE m.red_fuel END AS alliance_fuel,
       m.coverage, m.scoreboard_ok
FROM match_teams mt JOIN matches m ON m.match_key = mt.match_key;

CREATE VIEW IF NOT EXISTS team_summary AS
SELECT team,
       COUNT(*)                    AS matches,
       SUM(alliance_fuel)          AS total_alliance_fuel,
       ROUND(AVG(alliance_fuel),1) AS avg_alliance_fuel
FROM team_match_fuel
WHERE scoreboard_ok = 1
GROUP BY team;
"""


# Columns added after the first release. `CREATE TABLE IF NOT EXISTS` is a
# no-op on an existing table, so new columns need an explicit ALTER.
MIGRATIONS = [
    ("frames", "split", "TEXT"),
    ("matches", "blue_points", "INTEGER"),
    ("matches", "red_points", "INTEGER"),
    ("matches", "tba_breakdown", "TEXT"),
]


def _migrate(con: sqlite3.Connection) -> None:
    for table, column, decl in MIGRATIONS:
        exists = con.execute(
            "SELECT COUNT(*) FROM pragma_table_info(?) WHERE name=?",
            (table, column)).fetchone()[0]
        if not exists and con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table,)).fetchone()[0]:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    con.commit()


def write_live(con: sqlite3.Connection, event_key: str, match_key: str,
               series: Dict[str, list], totals: Dict[str, Optional[int]],
               teams: Optional[Dict[str, list]] = None,
               label: str = "", note: str = "") -> Dict[str, int]:
    """Write one live-scouted match: the timeline, the totals and the roster.

    Separate from `build`, which rebuilds everything from the manifest and
    would erase anything the manifest does not know about -- and the manifest
    is the DATASET's record, which a live read deliberately never enters. So
    these rows have no `video_id`, no crop and no frames, and that is the
    correct shape rather than a gap: nothing was kept to point at.

    `status` is 'live' so the two can always be told apart afterwards. A row
    read off a broadcast as it happened and a row read off a downloaded video
    are the same measurement of the same counter, but not the same evidence --
    the live one cannot be re-read, because the video is gone.

    Idempotent per match: re-running replaces that match's timeline rather than
    appending a second copy of it.
    """
    year = int(event_key[:4]) if event_key[:4].isdigit() else None
    con.execute("INSERT OR IGNORE INTO events(event_key, year) VALUES (?,?)",
                (event_key, year))

    # A live read is only as good as the OCR: no readings at all means the
    # counters were never found or never parsed, and that must not land as a
    # confident zero.
    ok = 1 if any(series.get(a) for a in ("blue", "red")) else 0
    con.execute("""
        INSERT INTO matches (match_key, event_key, comp_level, label, status,
                             blue_fuel, red_fuel, scoreboard_ok, scoreboard_note)
        VALUES (?,?,?,?,'live',?,?,?,?)
        ON CONFLICT(match_key) DO UPDATE SET
            event_key=excluded.event_key, status=excluded.status,
            blue_fuel=excluded.blue_fuel, red_fuel=excluded.red_fuel,
            scoreboard_ok=excluded.scoreboard_ok,
            scoreboard_note=excluded.scoreboard_note,
            label=COALESCE(excluded.label, matches.label)
        """, (match_key, event_key, _comp_level(label), label or None,
              totals.get("blue"), totals.get("red"), ok, note or None))

    if teams:
        con.execute("DELETE FROM match_teams WHERE match_key=?", (match_key,))
        for alliance in ("blue", "red"):
            for station, team in enumerate(teams.get(alliance) or [], start=1):
                try:
                    number = int(str(team).replace("frc", ""))
                except (TypeError, ValueError):
                    continue
                con.execute("INSERT OR IGNORE INTO match_teams"
                            "(match_key, alliance, station, team) VALUES (?,?,?,?)",
                            (match_key, alliance, station, number))

    con.execute("DELETE FROM score_events WHERE match_key=?", (match_key,))
    rows = 0
    for alliance, points in (series or {}).items():
        prev = None
        for point in points:
            t, v = point[0], point[1]
            if prev is not None and v > prev:
                con.execute("INSERT INTO score_events"
                            "(match_key,t_source,alliance,balls,total)"
                            " VALUES (?,?,?,?,?)",
                            (match_key, float(t), alliance, int(v - prev), int(v)))
                rows += 1
            prev = v
    con.commit()
    return {"score_events": rows, "scoreboard_ok": ok}


def _comp_level(label: str) -> Optional[str]:
    """'qm14' -> 'qm'. Advisory only; an unrecognised label stays NULL."""
    for level in ("qm", "sf", "qf", "ef", "f"):
        if label.startswith(level):
            return level
    return None


def export_for_serving(dest: Path, src: Path = None) -> Path:
    """Write a copy of the database that can be served from a read-only disk.

    Not `cp`. The working database runs in WAL mode, and a WAL database needs
    to create its `-shm` companion before anything -- including a strictly
    read-only connection -- can read it. Copy one onto a host that mounts its
    data directory read-only, which is what `deploy/frc-harvest.service` does
    on purpose, and the API starts cleanly and then fails every request with
    "attempt to write a readonly database". Verified as an unprivileged user
    against a 0555 directory, which is exactly what systemd's ReadOnlyPaths
    produces.

    So the copy is taken through SQLite's own backup API -- which waits for a
    consistent snapshot rather than catching the file mid-write, the other way
    `cp` gets this wrong -- and then switched out of WAL. A DELETE-journal
    database needs no sidecar files at all and reads fine with nothing but the
    read bit.
    """
    src = src or DB_PATH
    if not src.exists():
        raise SystemExit(f"{src} does not exist -- run `run.py db sync` first")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for stale in (dest, Path(f"{dest}-wal"), Path(f"{dest}-shm")):
        stale.unlink(missing_ok=True)

    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    target = sqlite3.connect(dest)
    try:
        source.backup(target)
        # Collapses any WAL the snapshot carried into the main file and leaves
        # nothing beside it for a reader to have to create.
        target.execute("PRAGMA journal_mode=DELETE")
        target.execute("VACUUM")
    finally:
        target.close()
        source.close()
    return dest


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    # Tables first so a fresh database exists, then migrations, and only then
    # the indexes and views -- some of them reference columns that migrations
    # add, so running the whole script up front fails on an older database.
    head, _, tail = SCHEMA.partition("CREATE INDEX")
    con.executescript(head)
    _migrate(con)
    con.executescript("CREATE INDEX" + tail)
    return con


def _kept_secs(analysis: dict) -> float:
    return round(sum(b - a for a, b in analysis.get("keep_ranges", [])), 3)


def build(con: sqlite3.Connection, manifest: Optional[dict] = None) -> Dict[str, int]:
    """(Re)load everything from the manifest and label CSVs. Idempotent."""
    if manifest is None:
        manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() \
                   else {"videos": {}}
    counts = {"matches": 0, "teams": 0, "score_events": 0, "frames": 0}

    for vid, e in manifest.get("videos", {}).items():
        a = e.get("analysis") or {}
        c = e.get("crop") or {}
        s = e.get("score") or {}
        mk = e.get("match_key") or vid
        ek = e.get("event_key") or ""
        year = int(ek[:4]) if ek[:4].isdigit() else None

        con.execute("INSERT OR IGNORE INTO events(event_key, year) VALUES (?,?)",
                    (ek, year))

        ok = 1 if (s and not s.get("error") and s.get("series")) else 0
        final = s.get("final") or {}
        con.execute("""
            INSERT INTO matches (match_key, event_key, video_id, yt_key, comp_level,
                label, title, status, source_secs, kept_secs, coverage, width, height,
                crop_x, crop_y, crop_w, crop_h, clean_path, blue_fuel, red_fuel,
                scoreboard_ok, scoreboard_note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(match_key) DO UPDATE SET
                event_key=excluded.event_key, video_id=excluded.video_id,
                yt_key=excluded.yt_key, status=excluded.status,
                kept_secs=excluded.kept_secs, coverage=excluded.coverage,
                crop_x=excluded.crop_x, crop_y=excluded.crop_y,
                crop_w=excluded.crop_w, crop_h=excluded.crop_h,
                clean_path=excluded.clean_path, blue_fuel=excluded.blue_fuel,
                red_fuel=excluded.red_fuel, scoreboard_ok=excluded.scoreboard_ok,
                scoreboard_note=excluded.scoreboard_note
        """, (mk, ek, vid, e.get("yt_key"), (e.get("label") or "")[:2],
              e.get("label"), e.get("title"), e.get("status"),
              e.get("source_duration"), _kept_secs(a), a.get("coverage"),
              a.get("width"), a.get("height"),
              c.get("x"), c.get("y"), c.get("w"), c.get("h"),
              e.get("clean_path"), final.get("blue"), final.get("red"),
              ok, s.get("error")))
        counts["matches"] += 1

        con.execute("DELETE FROM match_teams WHERE match_key=?", (mk,))
        for alliance, teams in (e.get("teams") or {}).items():
            for i, t in enumerate(teams, start=1):
                if str(t).isdigit():
                    con.execute("INSERT OR REPLACE INTO match_teams"
                                "(match_key, alliance, station, team) VALUES (?,?,?,?)",
                                (mk, alliance, i, int(t)))
                    counts["teams"] += 1

        con.execute("DELETE FROM score_events WHERE match_key=?", (mk,))
        for ev in s.get("events") or []:
            con.execute("INSERT INTO score_events(match_key,t_source,alliance,balls,total)"
                        " VALUES (?,?,?,?,?)",
                        (mk, ev["t"], ev["alliance"], ev["balls"], ev["total"]))
            counts["score_events"] += 1

        counts["frames"] += _load_frames(con, mk, vid)

    con.commit()
    return counts


def _load_frames(con: sqlite3.Connection, match_key: str, video_id: str) -> int:
    csv_path = LABEL_DIR / f"{video_id}.csv"
    if not csv_path.exists():
        return 0
    con.execute("DELETE FROM frames WHERE match_key=?", (match_key,))
    n = 0
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            con.execute("""INSERT OR REPLACE INTO frames
                (match_key,file,t_clean,t_source,blue_total,red_total,blue_next,red_next)
                VALUES (?,?,?,?,?,?,?,?)""",
                (match_key, row["frame"], _f(row.get("t_clean")), _f(row.get("t_source")),
                 _i(row.get("blue_total")), _i(row.get("red_total")),
                 _i(row.get("blue_scored_next")), _i(row.get("red_scored_next"))))
            n += 1
    return n


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def set_hub(con: sqlite3.Connection, event_key: str, alliance: str, box) -> None:
    """Record the one-time hub geometry for an event."""
    col = "hub_blue" if alliance == "blue" else "hub_red"
    con.execute("INSERT OR IGNORE INTO events(event_key) VALUES (?)", (event_key,))
    con.execute(f"UPDATE events SET {col}=? WHERE event_key=?",
                (json.dumps(list(box)), event_key))
    con.commit()


def add_detections(con: sqlite3.Connection, rows: Iterable[dict]) -> int:
    n = 0
    for r in rows:
        fid = con.execute("SELECT id FROM frames WHERE file=?", (r["file"],)).fetchone()
        if not fid:
            continue
        con.execute("""INSERT INTO detections
            (frame_id,cls,x,y,w,h,conf,alliance,team,track_id,source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (fid[0], r["cls"], r.get("x"), r.get("y"), r.get("w"), r.get("h"),
             r.get("conf"), r.get("alliance"), r.get("team"), r.get("track_id"),
             r.get("source", "auto")))
        n += 1
    con.commit()
    return n


def backfill_teams(con: sqlite3.Connection, client) -> int:
    """Fill in alliance rosters for matches ingested before rosters were captured.

    Also the repair path for any match whose TBA record changed after we pulled
    the video (surrogate swaps, replays).
    """
    rows = con.execute("""
        SELECT m.match_key FROM matches m
        LEFT JOIN match_teams t ON t.match_key = m.match_key
        WHERE t.match_key IS NULL
    """).fetchall()
    filled = 0
    for (mk,) in rows:
        match = client.get(f"/match/{mk}")
        if not match:
            print(f"  ! TBA has no match {mk}")
            continue
        alliances = match.get("alliances") or {}
        for alliance in ("blue", "red"):
            for i, tk in enumerate(alliances.get(alliance, {}).get("team_keys") or [],
                                   start=1):
                num = tk.replace("frc", "")
                if num.isdigit():
                    con.execute("INSERT OR REPLACE INTO match_teams"
                                "(match_key,alliance,station,team) VALUES (?,?,?,?)",
                                (mk, alliance, i, int(num)))
                    filled += 1
        # Names come free with the same request.
        con.execute("UPDATE events SET year=COALESCE(year,?) WHERE event_key=?",
                    (match.get("event_key", "")[:4] or None, match.get("event_key")))
    con.commit()
    return filled


def fetch_official(con: sqlite3.Connection, client) -> int:
    """Pull TBA's official score breakdown for every match in the database."""
    rows = con.execute("SELECT match_key FROM matches WHERE tba_breakdown IS NULL")
    n = 0
    for (mk,) in rows.fetchall():
        match = client.get(f"/match/{mk}")
        if not match:
            continue
        bd = match.get("score_breakdown") or {}
        alliances = match.get("alliances") or {}
        con.execute("""UPDATE matches SET blue_points=?, red_points=?, tba_breakdown=?
                       WHERE match_key=?""",
                    (alliances.get("blue", {}).get("score"),
                     alliances.get("red", {}).get("score"),
                     json.dumps(bd) if bd else None, mk))
        n += 1
    con.commit()
    return n


def summary(con: sqlite3.Connection) -> Dict[str, int]:
    q = lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    return {t: q(t) for t in ("events", "matches", "match_teams", "score_events",
                              "frames", "detections")}
