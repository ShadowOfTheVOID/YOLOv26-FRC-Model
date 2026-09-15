"""Read-only JSON API over the scouting database.

Stdlib only, so the harvester venv stays at requests + numpy and the scouting
app has no build step to depend on. Read-only by design: the database is
rebuildable from the manifest at any time, so nothing here should be the only
copy of anything.
"""
from __future__ import annotations

import json
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .db import DB_PATH, connect

MAX_LIMIT = 5000


def _rows(con, sql: str, params=()) -> List[dict]:
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def _page(q: Dict[str, List[str]]) -> Tuple[int, int]:
    limit = min(int((q.get("limit") or [200])[0]), MAX_LIMIT)
    return limit, int((q.get("offset") or [0])[0])


ROUTES: List[Tuple[re.Pattern, Callable]] = []


def route(pattern: str):
    def deco(fn):
        ROUTES.append((re.compile(f"^{pattern}$"), fn))
        return fn
    return deco


@route(r"/health")
def health(con, m, q):
    from .db import summary
    return {"ok": True, "counts": summary(con)}


@route(r"/schema")
def schema(con, m, q):
    """So an app developer can discover the shape without reading the source."""
    out = {}
    for (name, kind) in con.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table','view')"
            " AND name NOT LIKE 'sqlite_%'"):
        out[name] = {"kind": kind,
                     "columns": [r[1] for r in con.execute(f"PRAGMA table_info({name})")]}
    return {"objects": out, "routes": [p.pattern for p, _ in ROUTES]}


@route(r"/events")
def events(con, m, q):
    return {"events": _rows(con, """
        SELECT e.*, COUNT(m.match_key) AS matches
        FROM events e LEFT JOIN matches m ON m.event_key = e.event_key
        GROUP BY e.event_key ORDER BY e.event_key""")}


@route(r"/matches")
def matches(con, m, q):
    limit, offset = _page(q)
    where, params = [], []
    if q.get("event"):
        where.append("m.event_key = ?"); params.append(q["event"][0])
    if q.get("team"):
        where.append("m.match_key IN (SELECT match_key FROM match_teams WHERE team = ?)")
        params.append(int(q["team"][0]))
    if q.get("scoreboard_ok"):
        where.append("m.scoreboard_ok = ?"); params.append(int(q["scoreboard_ok"][0]))
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params += [limit, offset]
    return {"matches": _rows(con, f"""
        SELECT m.* FROM matches m {clause}
        ORDER BY m.event_key, m.match_key LIMIT ? OFFSET ?""", params)}


@route(r"/matches/([\w\-]+)")
def match_detail(con, m, q):
    mk = m.group(1)
    row = con.execute("SELECT * FROM matches WHERE match_key=?", (mk,)).fetchone()
    if not row:
        return None
    return {
        "match": dict(row),
        "teams": _rows(con, "SELECT alliance, station, team FROM match_teams"
                            " WHERE match_key=? ORDER BY alliance, station", (mk,)),
        "score_events": _rows(con, "SELECT t_source, alliance, balls, total"
                                   " FROM score_events WHERE match_key=?"
                                   " ORDER BY t_source", (mk,)),
    }


@route(r"/teams")
def teams(con, m, q):
    return {"teams": _rows(con, "SELECT * FROM team_summary ORDER BY team")}


@route(r"/teams/(\d+)")
def team_detail(con, m, q):
    t = int(m.group(1))
    summary = con.execute("SELECT * FROM team_summary WHERE team=?", (t,)).fetchone()
    rows = _rows(con, "SELECT * FROM team_match_fuel WHERE team=? ORDER BY match_key", (t,))
    if not summary and not rows:
        return None
    return {"team": t, "summary": dict(summary) if summary else None, "matches": rows}


@route(r"/frames")
def frames(con, m, q):
    limit, offset = _page(q)
    where, params = [], []
    if q.get("match"):
        where.append("match_key = ?"); params.append(q["match"][0])
    if (q.get("scored") or ["0"])[0] == "1":
        # Frames where fuel scored shortly after -- the ones worth labelling.
        where.append("(COALESCE(blue_next,0) + COALESCE(red_next,0)) > 0")
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params += [limit, offset]
    return {"frames": _rows(con, f"SELECT * FROM frames {clause}"
                                 f" ORDER BY match_key, t_source LIMIT ? OFFSET ?", params)}


@route(r"/detections")
def detections(con, m, q):
    limit, offset = _page(q)
    where, params = [], []
    if q.get("match"):
        where.append("f.match_key = ?"); params.append(q["match"][0])
    if q.get("frame"):
        where.append("f.file = ?"); params.append(q["frame"][0])
    if q.get("team"):
        where.append("d.team = ?"); params.append(int(q["team"][0]))
    if q.get("cls"):
        where.append("d.cls = ?"); params.append(q["cls"][0])
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params += [limit, offset]
    return {"detections": _rows(con, f"""
        SELECT d.*, f.file, f.match_key, f.t_source
        FROM detections d JOIN frames f ON f.id = d.frame_id
        {clause} ORDER BY f.match_key, f.t_source LIMIT ? OFFSET ?""", params)}


def make_handler(db_path: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TBAScouting/1.0"

        def log_message(self, *a):
            pass

        def _send(self, code: int, payload: dict):
            body = json.dumps(payload, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            # The scouting app is a separate origin; this API is read-only and
            # local, so a permissive CORS header costs nothing.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.end_headers()

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/health"
            q = parse_qs(parsed.query)
            for pattern, fn in ROUTES:
                m = pattern.match(path)
                if not m:
                    continue
                con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                con.row_factory = sqlite3.Row
                try:
                    result = fn(con, m, q)
                except (ValueError, sqlite3.Error) as exc:
                    self._send(400, {"error": str(exc)})
                    return
                finally:
                    con.close()
                if result is None:
                    self._send(404, {"error": "not found", "path": path})
                else:
                    self._send(200, result)
                return
            self._send(404, {"error": "no such route", "path": path,
                             "routes": [p.pattern for p, _ in ROUTES]})
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8781, db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        raise SystemExit(f"{db_path} does not exist -- run `run.py db build` first")
    httpd = ThreadingHTTPServer((host, port), make_handler(db_path))
    print(f"scouting API on http://{host}:{port}")
    for p, _ in ROUTES:
        print(f"  GET {p.pattern}")
    print("\nCtrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
