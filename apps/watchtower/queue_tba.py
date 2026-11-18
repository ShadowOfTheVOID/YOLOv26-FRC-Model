"""TBA's schedule for the match queue page (/queue), when Watchtower has none.

For an event Watchtower is not running (/queue?event=2026abcd) or before
Control has entered a schedule. Needs a TBA Read API key: event.yaml's
tba.read_key (the v0.x app's TBA tab wrote it there) or $TBA_AUTH_KEY.
Moved here from the v0.x app's settings.py when Control -> Setup took over
everything else that file edited.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

TBA_API = os.environ.get("WATCHTOWER_TBA_API", "https://www.thebluealliance.com/api/v3")


LEVELS = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}


def tba_matches(event_key: str, read_key: str = "") -> dict:
    """An event's schedule from TBA in the shape Watchtower's state gives
    the match queue page: {"name", "matches": [{key, comp_level, set_number,
    match_number, red: [254, ...], blue, surrogates, status, time,
    red_score, blue_score}]} in playing order, or {"error": why}.

    For a schedule Watchtower did not make (a real event, or before the
    scorekeeper has entered one). Played = TBA has a score for it
    ("committed"); TBA cannot know what is on the field, so the rest are
    "scheduled". Unlike the team list there is no public page to read, so
    this needs the Read API key. Playing order is level, then match number,
    then set: qf1m1, qf2m1, ... qf1m2 in a best-of-three bracket, and the
    double-elimination playoffs (all match 1) by set."""
    key = (event_key or "").strip().lower()
    if not re.fullmatch(r"\d{4}[a-z0-9]+", key):
        return {"error": "No TBA event key (like 2026catstd) for the match queue."}
    token = (read_key or "").strip() or os.environ.get("TBA_AUTH_KEY", "")
    if not token:
        return {"error": "TBA's schedule needs a Read API key: set tba.read_key in event.yaml or $TBA_AUTH_KEY."}

    def get(path):
        req = urllib.request.Request(f"{TBA_API}{path}",
                                     headers={"X-TBA-Auth-Key": token, "User-Agent": "Watchtower app"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))
    try:
        rows = get(f"/event/{key}/matches/simple") or []
        try:
            name = (get(f"/event/{key}/simple") or {}).get("name") or key
        except (urllib.error.URLError, OSError, ValueError):
            name = key
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"error": "TBA refused the Read API key. Check tba.read_key."}
        if e.code == 404:
            return {"error": f"TBA has no event {key}. Check the event key."}
        return {"error": f"TBA answered {e.code}. Trying again shortly."}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"error": f"Could not reach TBA ({getattr(e, 'reason', e)})."}

    def teams(al):
        return [int(t[3:]) if t[3:].isdigit() else t[3:] for t in (al or {}).get("team_keys") or []]
    out = []
    for m in rows:
        if m.get("comp_level") not in LEVELS:
            continue
        al = m.get("alliances") or {}
        rs, bs = ((al.get(a) or {}).get("score") for a in ("red", "blue"))
        played = rs is not None and bs is not None and rs >= 0 and bs >= 0
        out.append({"key": m.get("key", "").split("_")[-1], "comp_level": m["comp_level"],
                    "set_number": int(m.get("set_number") or 1), "match_number": int(m.get("match_number") or 1),
                    "red": teams(al.get("red")), "blue": teams(al.get("blue")),
                    "surrogates": [int(t[3:]) for a in ("red", "blue")
                                   for t in (al.get(a) or {}).get("surrogate_team_keys") or [] if t[3:].isdigit()],
                    "status": "committed" if played else "scheduled",
                    "time": m.get("predicted_time") or m.get("time"),
                    "red_score": rs if played else None, "blue_score": bs if played else None})
    out.sort(key=lambda m: (LEVELS[m["comp_level"]], m["match_number"], m["set_number"]))
    return {"name": name, "matches": out}
