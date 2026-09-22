"""A scoreboard for a field with no FMS, where this count IS the score.

At a scrimmage there is no Field Management System, so nothing else is
counting. That makes `count.py` the authority rather than a cross-check, and
it changes what it has to be: a number on a laptop nobody can see, with no
match clock and no way to correct it, is not a scoreboard.

Three things this adds, and each exists because of something the counter
cannot do on its own.

**A match clock.** Balls only score during a match. People throw fuel around
between matches, robots get tested on the field, and a counter running
continuously would add all of it to the score. `Match` will not accept a ball
unless it is in a scoring phase.

**A way out.** Whatever is showing the score at the field already exists, so
there is deliberately no display here -- `serve()` hands the whole match over
as JSON and `--feed` writes a line per change to stdout. This is the thing that
knows what the score is, not the thing that draws it.

**A referee's override.** This is the important one. `count.py` is careful and
still fallible: it cannot see a ball occluded for its whole flight, and a hub
box that is slightly wrong costs real balls. Every other part of this
repository responds to uncertainty by recording nothing -- correct when a
scouting number can simply be absent, and useless when a match needs a final
score in the next thirty seconds. So the human at the table gets the last word,
and the record keeps both halves: what was detected, and what a person changed
it to. `adjusted` is never folded into `detected`, because "the camera missed
two" and "the camera saw two that were not there" are different facts about
the system and both are worth knowing after the event.

## Points

`points_per_ball` defaults to 1 for both phases, so the display shows the ball
count and nothing else. That is deliberate. `db.py` already refuses to convert
fuel to points -- "the 2026 fuel-to-points rule is not pinned down here" -- and
a scrimmage runs whatever rules its organiser chose anyway. Set the multipliers
for your field; nothing here invents them.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ALLIANCES = ("blue", "red")

# FRC's own shape, as a starting point. A scrimmage is exactly the place these
# get changed, so they are arguments rather than constants in the code below.
AUTO_S = 15.0
TELEOP_S = 135.0

IDLE, AUTO, TELEOP, ENDED = "idle", "auto", "teleop", "ended"
SCORING = (AUTO, TELEOP)


class Match:
    """One match's clock, score and log. Safe to drive from several threads.

    The counter thread calls `ball()` as it detects them, the referee's browser
    calls `adjust()` and the clock controls, and the display polls `state()` --
    all at once, which is why every one of them takes the lock.
    """

    def __init__(self, label: str = "", auto_s: float = AUTO_S,
                 teleop_s: float = TELEOP_S,
                 points_per_ball: Optional[Dict[str, float]] = None):
        self.lock = threading.Lock()
        self.label = label
        self.auto_s = float(auto_s)
        self.teleop_s = float(teleop_s)
        self.points_per_ball = dict(points_per_ball or {AUTO: 1.0, TELEOP: 1.0})
        self.reset(label)

    # -- clock ------------------------------------------------------------
    def reset(self, label: str = "") -> None:
        with self.lock:
            self.label = label or self.label
            self.started_at: Optional[float] = None
            self.ended_at: Optional[float] = None
            # Detected and adjusted are kept apart on purpose -- see the module
            # docstring. Per phase, because a scrimmage may score them apart.
            self.detected = {a: {AUTO: 0, TELEOP: 0} for a in ALLIANCES}
            self.adjusted = {a: {AUTO: 0, TELEOP: 0} for a in ALLIANCES}
            self.log: List[Dict] = []

    def start(self) -> None:
        with self.lock:
            self.started_at = time.monotonic()
            self.ended_at = None

    def stop(self) -> None:
        """End the match now. An abort and a buzzer look the same from here."""
        with self.lock:
            if self.started_at is not None and self.ended_at is None:
                self.ended_at = time.monotonic()

    def _elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return end - self.started_at

    def _phase(self) -> str:
        if self.started_at is None:
            return IDLE
        if self.ended_at is not None:
            return ENDED
        e = self._elapsed()
        if e < self.auto_s:
            return AUTO
        if e < self.auto_s + self.teleop_s:
            return TELEOP
        return ENDED

    def phase(self) -> str:
        with self.lock:
            return self._phase()

    # -- scoring ----------------------------------------------------------
    def ball(self, alliance: str, t: Optional[float] = None) -> bool:
        """A ball the detector says went in. False if it did not count.

        Refused outside a scoring phase rather than queued or clamped: fuel
        thrown around between matches is not a score, and a counter that ran
        through the changeover would quietly add it.
        """
        with self.lock:
            phase = self._phase()
            if alliance not in self.detected or phase not in SCORING:
                return False
            self.detected[alliance][phase] += 1
            self.log.append({"t": round(self._elapsed(), 2), "phase": phase,
                             "alliance": alliance, "by": "detector", "delta": 1})
            return True

    def adjust(self, alliance: str, delta: int, phase: Optional[str] = None,
               who: str = "ref") -> bool:
        """A person correcting the count. The last word, and recorded as one.

        Allowed after the buzzer, because that is when most corrections happen
        -- somebody watched a ball go in that the camera did not.
        """
        with self.lock:
            if alliance not in self.adjusted or not delta:
                return False
            phase = phase if phase in SCORING else (
                self._phase() if self._phase() in SCORING else TELEOP)
            # The total may not go below zero, but the adjustment itself may be
            # negative: "the camera saw one that did not happen" is a real
            # correction and must be recordable.
            current = self.detected[alliance][phase] + self.adjusted[alliance][phase]
            if current + delta < 0:
                delta = -current
            if not delta:
                return False
            self.adjusted[alliance][phase] += delta
            self.log.append({"t": round(self._elapsed(), 2), "phase": phase,
                             "alliance": alliance, "by": who, "delta": delta})
            return True

    # -- reading ----------------------------------------------------------
    def _balls(self, alliance: str) -> Dict[str, int]:
        return {p: self.detected[alliance][p] + self.adjusted[alliance][p]
                for p in (AUTO, TELEOP)}

    def _points(self, alliance: str) -> float:
        balls = self._balls(alliance)
        return sum(balls[p] * float(self.points_per_ball.get(p, 1.0))
                   for p in (AUTO, TELEOP))

    def state(self) -> Dict:
        with self.lock:
            phase = self._phase()
            elapsed = self._elapsed()
            total = self.auto_s + self.teleop_s
            return {
                "label": self.label,
                "phase": phase,
                "elapsed": round(elapsed, 1),
                "remaining": round(max(total - elapsed, 0.0), 1)
                             if phase in SCORING else 0.0,
                "running": phase in SCORING,
                "alliances": {
                    a: {
                        "balls": self._balls(a),
                        "total": sum(self._balls(a).values()),
                        "points": round(self._points(a), 1),
                        # Kept apart so the display -- and anyone reading this
                        # afterwards -- can see how much of the score a person
                        # put there.
                        "detected": sum(self.detected[a].values()),
                        "adjusted": sum(self.adjusted[a].values()),
                    } for a in ALLIANCES},
                "log": self.log[-40:],
                "pointsPerBall": self.points_per_ball,
            }

    def series(self) -> Dict[str, List[List[float]]]:
        """Cumulative per-alliance steps, the shape `db.write_live` wants.

        A referee's correction lands at the moment it was made, not when the
        ball it is about went in -- nobody knows that second -- so the timeline
        is approximate around a correction while the total is exact. Read
        `totals()` for the score and this for the shape of the match.
        """
        out: Dict[str, List[List[float]]] = {a: [] for a in ALLIANCES}
        running = {a: 0 for a in ALLIANCES}
        with self.lock:
            for row in self.log:
                a = row["alliance"]
                running[a] = max(running[a] + row["delta"], 0)
                out[a].append([row["t"], running[a]])
        return out

    def totals(self) -> Dict[str, int]:
        with self.lock:
            return {a: sum(self._balls(a).values()) for a in ALLIANCES}


def serve(match: Match, host: str = "0.0.0.0", port: int = 8780,
          on_change=None) -> ThreadingHTTPServer:
    """A JSON feed and control surface. No pages, no display, no browser.

    There is deliberately nothing to look at here. Whatever is showing the
    score at the field already exists; this is the thing that knows what the
    score IS, and it hands it over as JSON for that to read.

        GET  /state               the whole match: phase, clock, both alliances
        POST /start /stop /reset  the match clock
        POST /adjust  {"alliance": "red", "delta": 1}   a correction

    Binds every interface by default, because the display is on another machine
    on the field wifi. There is no authentication: it is a closed field network
    for an afternoon. Do not put it on the open internet.
    """
    class Handler(BaseHTTPRequestHandler):
        server_version = "FieldScore/1.0"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            raw = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in ("/", "/state"):
                return self._send(200, json.dumps(match.state()))
            self._send(404, json.dumps({"error": "no such route",
                                        "routes": ["/state", "/start", "/stop",
                                                   "/reset", "/adjust"]}))

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/start":
                match.start()
            elif path == "/stop":
                match.stop()
            elif path == "/reset":
                match.reset()
            elif path == "/adjust":
                # Every coercion inside the guard, not just the parse. `delta`
                # arriving as "x" raised out of the handler and dropped the
                # connection -- the ref page never sends that, but a scoreboard
                # is the last thing that should die on a stray request while a
                # match is waiting on it.
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("body is not an object")
                    alliance = str(body.get("alliance", ""))
                    delta = int(body.get("delta") or 0)
                except (ValueError, TypeError, json.JSONDecodeError):
                    return self._send(400, json.dumps({"error": "bad body"}))
                match.adjust(alliance, delta)
            else:
                return self._send(404, json.dumps({"error": "no such route"}))
            if on_change:
                on_change(match)
            return self._send(200, json.dumps(match.state()))

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True,
                     name="scoreboard").start()
    return httpd
