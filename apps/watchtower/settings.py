"""Every event.yaml setting a scrimmage needs, edited from the app, never by hand.

Watchtower reads config/event.yaml once at start. The app's Home window shows
these fields; Save checks them with the same rules Watchtower applies at
start (fms.config.validate: PINs set, the control PIN its own) plus the
formats it expects, writes them, and restarts Watchtower.

Writing changes only the lines of the fields that changed, so the comments
fms.init's file carries (the only documentation of its other settings) stay.
A file reshaped so a field is not one line (a block-style team list, say) is
rewritten whole, with event.yaml.bak kept. Either way the result is read
back and must hold exactly what was saved, or nothing is written.

No GUI here: the window code calls load(), check() and save(), and the
tests call them directly.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path

# TBA's read API; WATCHTOWER_TBA_API points the tests at a stand-in.
TBA_API = os.environ.get("WATCHTOWER_TBA_API", "https://www.thebluealliance.com/api/v3")
# The public event page, whose Scouting tab lists the teams as CSV (in a
# <textarea id="team-list-csv">): no API key needed for that.
TBA_WEB = os.environ.get("WATCHTOWER_TBA_WEB", "https://www.thebluealliance.com")

# (key path in event.yaml, kind, label). Order is the order on the page.
FIELDS = [
    # Event and schedule
    (("event", "name"), "text", "Event name"),
    (("event", "date"), "date", "Date"),
    (("event", "utc_offset_hours"), "offset", "Time zone (hours from UTC on the day)"),
    (("event", "teams"), "teams", "Teams"),
    (("event", "qual_start"), "time", "Qualifications start"),
    (("event", "cycle_min"), "int", "Minutes per match (cycle)"),
    (("event", "lunch"), "time", "Lunch"),
    # Access
    (("server", "pins", "control"), "pin", "Scorekeeper PIN"),
    (("server", "pins", "ref"), "pin", "Ref PIN"),
    (("server", "pins", "emcee"), "pin", "Emcee PIN"),
    # Not Watchtower's either: the address phones are given when the FMS is
    # also reachable through a reverse proxy or tunnel (our Caddy at
    # watchtower.systemoverload.org, or any other). Empty unless set: nothing
    # is built in. Watchtower itself still listens on the Wi-Fi; this only
    # changes the link and QR on Phones & PINs.
    (("server", "public_url"), "url", "Public address (optional, e.g. https://watchtower.systemoverload.org)"),
    # The Blue Alliance
    (("event", "tba_event_key"), "tbakey", "TBA event key"),
    (("tba", "enabled"), "bool", "Send schedule and results to TBA with this write key"),
    # Both halves of the write key are masked on the page (password
    # fields): together they can rewrite the event's results on TBA.
    (("tba", "auth_id"), "secret", "Write key: Auth ID"),
    (("tba", "auth_secret"), "secret", "Write key: Auth Secret"),
    (("tba", "send_score_breakdown"), "bool", "Send score breakdowns (totals only is safer)"),
    # Not Watchtower's (it reads only its own tba keys): the app's own, a
    # fallback for "Import from TBA", which reads the event page's public
    # Scouting list first. From thebluealliance.com/account; $TBA_AUTH_KEY
    # is used when this is empty.
    (("tba", "read_key"), "secret", "Read API key"),
    # Game rules
    (("game", "auto_s"), "int", "Auto (s)"),
    (("game", "auto_teleop_gap_s"), "int", "Pause after auto (s)"),
    (("game", "transition_s"), "int", "Transition (s)"),
    (("game", "shift_s"), "int", "Shift length (s)"),
    (("game", "n_shifts"), "int", "Number of shifts"),
    (("game", "endgame_s"), "int", "Endgame (s)"),
    (("game", "score_grace_s"), "int", "Grace after a hub goes dark (s)"),
    (("game", "close_auto_margin"), "int", "Close-auto warning margin (fuel)"),
    (("game", "fuel_points"), "int", "Points per fuel"),
    (("game", "auto_tower_l1"), "int", "Auto tower level 1 points"),
    (("game", "tower"), "points4", "Tower points L0 / L1 / L2 / L3"),
    (("game", "fouls"), "fouls", "Foul points minor / major"),
    (("game", "rp", "win"), "int", "Ranking points: win"),
    (("game", "rp", "tie"), "int", "Ranking points: tie"),
    (("game", "rp", "energized_fuel"), "int", "Energized RP: fuel needed"),
    (("game", "rp", "supercharged_fuel"), "int", "Supercharged RP: fuel needed"),
    (("game", "rp", "traversal_tower_points"), "int", "Traversal RP: tower points needed"),
]
KEY = {".".join(p): (p, kind, label) for p, kind, label in FIELDS}


def _get(cfg: dict, path: tuple):
    for k in path:
        cfg = (cfg or {}).get(k) if isinstance(cfg, dict) else None
    return cfg


def load(event_yaml: Path) -> dict:
    """{"event.name": value, ...} for every field, Watchtower's defaults
    filled in where the file leaves one out."""
    import yaml
    from fms import config
    cfg = config._merge(config.DEFAULTS, yaml.safe_load(event_yaml.read_text(encoding="utf-8")) or {})
    out = {}
    for name, (path, kind, _) in KEY.items():
        v = _get(cfg, path)
        if kind == "teams":
            v = [int(t) for t in v or []]
        elif kind in ("text", "secret", "date", "time", "tbakey", "pin", "url"):
            v = "" if v is None else str(v)
        out[name] = v
    return out


def local_utc_offset(date: str = "") -> float:
    """This computer's offset from UTC on `date` (daylight saving included)."""
    t = time.mktime(time.strptime(date, "%Y-%m-%d")) + 12 * 3600 if date else time.time()
    lt = time.localtime(t)
    return round(lt.tm_gmtoff / 3600, 2)


def new_pins() -> dict:
    """Three different 6-digit PINs, as fms.init makes them."""
    pins: dict = {}
    for role in ("control", "ref", "emcee"):
        while True:
            p = f"{secrets.randbelow(10**6):06d}"
            if p not in pins.values():
                pins[role] = p
                break
    return pins


def parse_team_text(text: str) -> dict:
    """Team numbers (and names when given) from whatever was pasted:
    TBA's Scouting CSV (team_number,team_name,city,...), JSON (a list of
    numbers, "frc254" keys, or objects with team_number / key / nickname),
    or plain numbers. In the CSV only the team_number column counts: a
    plain digit search would take '75' out of a photo URL as a team."""
    import csv
    import io
    text = (text or "").strip()
    nums, names = [], {}

    def add(n, name=""):
        try:
            n = int(str(n).lower().replace("frc", "").strip())
        except ValueError:
            return
        if 0 < n < 100000 and n not in nums:
            nums.append(n)
            if name:
                names[str(n)] = str(name)

    if text[:1] in "[{":
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if data is not None:
            if isinstance(data, dict):
                items = data.get("teams") or list(data.values())
            else:
                items = data
            for it in items:
                if isinstance(it, dict):
                    add(it.get("team_number") or it.get("key") or it.get("team") or "",
                        it.get("nickname") or it.get("team_name") or it.get("name") or "")
                else:
                    add(it)
            return {"teams": nums, "names": names}
    first = text.splitlines()[0].lower() if text else ""
    if "team_number" in first:
        for r in csv.DictReader(io.StringIO(text)):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            add(r.get("team_number", ""), r.get("team_name") or r.get("nickname") or "")
        return {"teams": nums, "names": names}
    for t in re.findall(r"(?:frc)?(\d+)", text, re.I):
        add(t)
    return {"teams": nums, "names": names}


def clean_url(s: str) -> str:
    """'watchtower.systemoverload.org/' -> 'https://watchtower.systemoverload.org'.
    A bare name gets https (what a proxy with a certificate serves); a path
    is kept for a proxy that mounts the FMS under one. A user part, query or
    fragment is refused: the link goes to phones, never a key."""
    from urllib.parse import urlsplit
    s = (s or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = "https://" + s
    try:
        u = urlsplit(s)
        u.port                                         # raises on a bad port
    except ValueError:
        raise ValueError("not a web address") from None
    if u.scheme not in ("http", "https") or not u.hostname or " " in s:
        raise ValueError("like https://watchtower.systemoverload.org")
    if u.username or u.password or u.query or u.fragment:
        raise ValueError("just the address: no login, ? or # part")
    return f"{u.scheme}://{u.netloc}{u.path.rstrip('/')}"


def parse_teams(v) -> list:
    """'254, 1678 971' (any separators), TBA's CSV, JSON, or [254, ...] ->
    [254, 1678, 971], in order, no repeats."""
    if isinstance(v, str):
        return parse_team_text(v)["teams"]
    return parse_team_text(json.dumps(list(v or [])))["teams"]


def check(values: dict) -> tuple:
    """(clean values, {field: error}) for what the page sent."""
    clean, err = {}, {}
    for name, v in values.items():
        if name not in KEY:
            continue
        path, kind, label = KEY[name]
        try:
            clean[name] = _clean(kind, v)
        except ValueError as e:
            err[name] = f"{label}: {e}"
    pins = {r: clean.get(f"server.pins.{r}") for r in ("control", "ref", "emcee")}
    if pins["control"] and pins["control"] in (pins["ref"], pins["emcee"]):
        err["server.pins.control"] = ("Scorekeeper PIN: must differ from the ref and emcee PINs "
                                      "(anyone with those could run the event)")
    if clean.get("tba.enabled") and not (clean.get("event.tba_event_key") and clean.get("tba.auth_id")
                                         and clean.get("tba.auth_secret")):
        err["tba.enabled"] = ("Send to TBA: needs the TBA event key and the write key's Auth ID and Auth Secret "
                              "(see Write key on The Blue Alliance tab)")
    return clean, err


def _clean(kind: str, v):
    s = v.strip() if isinstance(v, str) else v
    if kind in ("text", "secret"):
        return "" if s is None else str(s)
    if kind == "date":
        if s and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            raise ValueError("use YYYY-MM-DD")
        return s or ""
    if kind == "time":
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", s or ""):
            raise ValueError("use HH:MM, 24-hour")
        return s
    if kind == "offset":
        f = float(s)
        if not -12 <= f <= 14:
            raise ValueError("between -12 and +14")
        return int(f) if f == int(f) else f
    if kind == "int":
        try:
            n = int(s)
        except (TypeError, ValueError):
            raise ValueError("a whole number") from None
        if n < 0:
            raise ValueError("0 or more")
        return n
    if kind == "bool":
        return bool(s) and s not in ("false", "0")
    if kind == "teams":
        return parse_teams(s)
    if kind == "pin":
        if not re.fullmatch(r"\d{4,8}", s or ""):
            raise ValueError("4 to 8 digits")
        return s
    if kind == "tbakey":
        if s and not re.fullmatch(r"\d{4}[a-z0-9]+", s.lower()):
            raise ValueError("like 2026catstd")
        return (s or "").lower()
    if kind == "url":
        return clean_url(s or "")
    if kind == "points4":
        vals = [int(x) for x in (s.values() if isinstance(s, dict) else re.findall(r"-?\d+", str(s)))]
        if len(vals) != 4:
            raise ValueError("four numbers")
        return dict(zip(("L0", "L1", "L2", "L3"), vals))
    if kind == "fouls":
        vals = [int(x) for x in (s.values() if isinstance(s, dict) else re.findall(r"-?\d+", str(s)))]
        if len(vals) != 2:
            raise ValueError("two numbers")
        return dict(zip(("minor", "major"), vals))
    raise ValueError(f"unknown kind {kind}")


def _render(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_render(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_render(x)}" for k, x in v.items()) + "}"
    return json.dumps(str(v), ensure_ascii=False)   # JSON strings are YAML strings


def _set_line(lines: list, path: tuple, rendered: str) -> bool:
    """Rewrite the one line holding `path` (keeping its comment); False if
    the file does not hold it on one line."""
    lo, hi = 0, len(lines)
    for depth, key in enumerate(path):
        ind = "  " * depth
        pat = re.compile(rf"^{ind}{re.escape(key)}:(\s*)([^#\n]*?)(\s*#.*)?(\n?)$")
        for i in range(lo, hi):
            m = pat.match(lines[i])
            if not m:
                continue
            if depth == len(path) - 1:
                if not m.group(2).strip() or m.group(2).strip()[0] in "|>":
                    return False                       # value is a block below
                lines[i] = f"{ind}{key}: {rendered}{m.group(3) or ''}{m.group(4)}"
                return True
            end = i + 1                                # this section's extent
            while end < hi and (not lines[end].strip() or lines[end].lstrip().startswith("#")
                                or lines[end].startswith(ind + "  ")):
                end += 1
            lo, hi = i + 1, end
            break
        else:
            return False
    return False


def _add_line(lines: list, path: tuple, rendered: str) -> bool:
    """Add a field missing from its section (e.g. tba.read_key, which
    fms.init's file never had) at the end of that section, so the rest of
    the file and its comments stay as they are. False if the section
    itself is missing or not block-shaped."""
    lo, hi, end = 0, len(lines), None
    for depth, key in enumerate(path[:-1]):
        ind = "  " * depth
        pat = re.compile(rf"^{ind}{re.escape(key)}:\s*(#.*)?\n?$")
        for i in range(lo, hi):
            if pat.match(lines[i]):
                end = i + 1
                while end < hi and (not lines[end].strip() or lines[end].lstrip().startswith("#")
                                    or lines[end].startswith(ind + "  ")):
                    end += 1
                lo, hi = i + 1, end
                break
        else:
            return False
    # Before trailing blank lines and column-0 comments: those head the
    # next section ("# ---- TBA trusted API"), and a line added after them
    # reads as part of that section, though YAML still files it here.
    while end > lo and (not lines[end - 1].strip() or lines[end - 1].startswith("#")):
        end -= 1
    if end and not lines[end - 1].endswith("\n"):
        lines[end - 1] += "\n"
    lines.insert(end, f"{'  ' * (len(path) - 1)}{path[-1]}: {rendered}\n")
    return True


def save(event_yaml: Path, values: dict) -> dict:
    """Write the changed fields. Returns {"changed": [...]} or raises
    ValueError with every problem, writing nothing."""
    import yaml
    from fms import config
    clean, err = check(values)
    if err:
        raise ValueError("\n".join(err.values()))
    current = load(event_yaml)
    changed = {k: v for k, v in clean.items() if current.get(k) != v}
    if not changed:
        return {"changed": []}
    text = event_yaml.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if all(_set_line(lines, KEY[k][0], _render(v)) or _add_line(lines, KEY[k][0], _render(v))
           for k, v in changed.items()):
        new = "".join(lines)
    else:
        data = yaml.safe_load(text) or {}
        for k, v in changed.items():
            d = data
            for p in KEY[k][0][:-1]:
                d = d.setdefault(p, {})
            d[KEY[k][0][-1]] = v
        new = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        event_yaml.with_suffix(".yaml.bak").write_text(text, encoding="utf-8")
    merged = config._merge(config.DEFAULTS, yaml.safe_load(new) or {})
    for k, v in clean.items():                         # read back what we wrote
        got = _get(merged, KEY[k][0])
        if KEY[k][1] == "teams":
            got = [int(t) for t in got or []]
        if got != v and str(got) != str(v):
            raise ValueError(f"could not write {KEY[k][2]} safely; nothing was saved")
    try:
        config.validate(merged, str(event_yaml))       # Watchtower's own start check
    except SystemExit as e:
        raise ValueError(str(e)) from None
    event_yaml.write_text(new, encoding="utf-8")
    return {"changed": sorted(changed)}


def _tba_page_teams(key: str) -> dict:
    """The teams from the event page's Scouting tab (public, no key)."""
    import html
    req = urllib.request.Request(f"{TBA_WEB}/event/{key}", headers={"User-Agent": "Watchtower app"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            page = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return {"error": f"TBA has no event {key}. Check the event key." if e.code == 404
                else f"TBA answered {e.code}. Try again in a minute."}
    except (urllib.error.URLError, OSError) as e:
        return {"error": f"Could not reach TBA ({getattr(e, 'reason', e)}). Is this computer online?"}
    m = re.search(r'<textarea[^>]*id="team-list-csv"[^>]*>(.*?)</textarea>', page, re.S)
    if not m:
        return {"error": "no team list on TBA's event page"}
    got = parse_team_text(html.unescape(m.group(1)))
    if not got["teams"]:
        return {"error": f"TBA lists no teams for {key} yet. Try again closer to the event."}
    got["teams"].sort()
    return got


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
        return {"error": "TBA's schedule needs a Read API key: add one on The Blue Alliance tab."}

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
            return {"error": "TBA refused the Read API key. Check it on The Blue Alliance tab."}
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


def tba_teams(event_key: str, read_key: str = "") -> dict:
    """The teams at a TBA event: {"teams": [254, ...], "names": {"254": "The
    Cheesy Poofs"}} sorted by number, or {"error": why}, in words a
    scorekeeper can act on. Reads the event page's public Scouting list;
    the Read API (key) only when the page has none. Needs internet."""
    key = (event_key or "").strip().lower()
    if not re.fullmatch(r"\d{4}[a-z0-9]+", key):
        return {"error": "Enter the TBA event key first (like 2026catstd), on The Blue Alliance tab."}
    # The event page's Scouting tab first: public, so no key is needed.
    page = _tba_page_teams(key)
    if "teams" in page or "no event" in page["error"] or "online" in page["error"]:
        return page
    token = (read_key or "").strip() or os.environ.get("TBA_AUTH_KEY", "")
    if not token:
        return page
    req = urllib.request.Request(f"{TBA_API}/event/{key}/teams/simple",
                                 headers={"X-TBA-Auth-Key": token, "User-Agent": "Watchtower app"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            rows = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"error": "TBA refused the Read API key. Check it on The Blue Alliance tab."}
        if e.code == 404:
            return {"error": f"TBA has no event {key}. Check the event key."}
        return {"error": f"TBA answered {e.code}. Try again in a minute."}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"error": f"Could not reach TBA ({getattr(e, 'reason', e)}). Is this computer online?"}
    teams = sorted({int(t["team_number"]) for t in rows or [] if t.get("team_number")})
    if not teams:
        return {"error": f"TBA lists no teams for {key} yet. Try again closer to the event."}
    names = {str(int(t["team_number"])): t.get("nickname") or "" for t in rows if t.get("team_number")}
    return {"teams": teams, "names": names}

