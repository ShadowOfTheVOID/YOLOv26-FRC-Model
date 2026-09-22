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

**A display.** Two big numbers and a clock, served to anything with a browser
-- a projector, a laptop on the scoring table, a phone. Stdlib HTTP, the same
as `review.py`, so there is nothing to install at a gym.

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


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scrimmage scoreboard</title>
<style>
 :root{--blue:#1d6fe0;--red:#d4332f;--bg:#07080a;--dim:#8a9099}
 *{box-sizing:border-box} html,body{margin:0;height:100%}
 body{background:var(--bg);color:#fff;font:600 16px/1.2 system-ui,sans-serif;
      display:flex;flex-direction:column;overflow:hidden}
 .clock{text-align:center;padding:2vh 0 0}
 .t{font:800 12vh/1 ui-monospace,monospace;letter-spacing:-.02em}
 .ph{letter-spacing:.35em;color:var(--dim);font-size:2.2vh;text-transform:uppercase}
 .mt{color:var(--dim);font-size:2vh;margin-top:.4vh}
 .scores{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:2vh;padding:2vh}
 .a{border-radius:2vh;display:flex;flex-direction:column;align-items:center;
    justify-content:center;position:relative}
 .blue{background:linear-gradient(160deg,#1d6fe0,#0d3f86)}
 .red{background:linear-gradient(160deg,#d4332f,#8b1a17)}
 .n{font:800 26vh/.85 ui-monospace,monospace}
 .lbl{letter-spacing:.35em;font-size:2.2vh;opacity:.85;text-transform:uppercase}
 .sub{font-size:2vh;opacity:.72;margin-top:1vh;min-height:2.4vh}
 .idle .n{opacity:.5}
 .ctrl{display:none}
 /* The ref page is the same document with ?ref=1, so a phone at the scoring
    table and the projector never disagree about what the score is. */
 body.ref .ctrl{display:flex;gap:1vh;margin-top:1.5vh}
 body.ref .n{font-size:16vh}
 button{font:800 2.4vh system-ui,sans-serif;padding:1.4vh 2.4vh;border:0;
        border-radius:1vh;background:rgba(255,255,255,.18);color:#fff;cursor:pointer}
 button:active{background:rgba(255,255,255,.34)}
 .bar{display:none;gap:1vh;justify-content:center;padding:0 0 2vh}
 body.ref .bar{display:flex}
 .bar button{background:#1b1f26}
 .go{background:#1d7a3a!important} .stop{background:#8b1a17!important}
</style>
<div class="clock">
  <div class="t" id="t">0:00</div>
  <div class="ph" id="ph">idle</div>
  <div class="mt" id="mt"></div>
</div>
<div class="scores">
  <div class="a blue" id="cb">
    <div class="lbl">Blue</div><div class="n" id="nb">0</div>
    <div class="sub" id="sb"></div>
    <div class="ctrl"><button onclick="adj('blue',-1)">&minus;1</button>
      <button onclick="adj('blue',1)">+1</button></div>
  </div>
  <div class="a red" id="cr">
    <div class="lbl">Red</div><div class="n" id="nr">0</div>
    <div class="sub" id="sr"></div>
    <div class="ctrl"><button onclick="adj('red',-1)">&minus;1</button>
      <button onclick="adj('red',1)">+1</button></div>
  </div>
</div>
<div class="bar">
  <button class="go" onclick="cmd('start')">START</button>
  <button class="stop" onclick="cmd('stop')">STOP</button>
  <button onclick="if(confirm('Clear the score?'))cmd('reset')">RESET</button>
</div>
<script>
const REF = new URLSearchParams(location.search).has('ref');
if (REF) document.body.classList.add('ref');
const $ = (i) => document.getElementById(i);
function mmss(s){s=Math.max(0,Math.round(s));return Math.floor(s/60)+':'+String(s%60).padStart(2,'0')}
async function cmd(what){ await fetch('/'+what,{method:'POST'}); tick(); }
async function adj(alliance,d){
  await fetch('/adjust',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({alliance,delta:d})}); tick();
}
async function tick(){
  let s; try { s = await (await fetch('/state')).json(); } catch(e){ return; }
  $('t').textContent = s.running ? mmss(s.remaining) : mmss(s.elapsed);
  $('ph').textContent = s.phase;
  $('mt').textContent = s.label || '';
  document.body.classList.toggle('idle', !s.running);
  for (const [a,ids] of [['blue',['nb','sb']],['red',['nr','sr']]]) {
    const x = s.alliances[a];
    $(ids[0]).textContent = x.points;
    // The two numbers are shown apart whenever a person has moved the score,
    // so nobody has to wonder whether the camera or the ref put it there.
    $(ids[1]).textContent = x.adjusted
      ? `${x.total} balls · ${x.detected} seen ${x.adjusted>0?'+':''}${x.adjusted} by ref`
      : `${x.total} balls`;
  }
}
tick(); setInterval(tick, 250);
</script>
"""


def serve(match: Match, host: str = "0.0.0.0", port: int = 8780,
          on_change=None) -> ThreadingHTTPServer:
    """Serve the display and the referee's controls. Returns the running server.

    Binds every interface by default, because the point is a projector and a
    phone on the same gym wifi looking at the same score. There is no
    authentication: it is a closed field network for an afternoon, and a
    password on the scoring table is a password somebody has to type while a
    match is waiting. Do not put this on the open internet.
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
            if path in ("/", "/ref"):
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/state":
                return self._send(200, json.dumps(match.state()))
            self._send(404, json.dumps({"error": "no such route"}))

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
