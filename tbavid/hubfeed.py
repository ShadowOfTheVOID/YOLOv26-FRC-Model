"""The hub FUEL counter feed to bioarena: the counter's half of the contract.

bioarena (Team 841's practice-field fork of cheesy-arena) has no PLC, so at
the 2026-10-10 scrimmage nothing counts balls into the hubs unless a camera
does. The spec it publishes ("Spec: Hub FUEL Counter Feed", sections 4 and 5)
is what this module implements:

  * UDP to 10.0.100.5:8411, one JSON object per datagram, carrying both hubs:
        {"v":1,"session":"c1f3a9d2","seq":4821,"red":57,"blue":0,"age_ms":38}
  * `red` / `blue` are cumulative since this process started and never go
    down. bioarena takes the match baseline itself, so nothing here knows or
    needs to know when a match starts.
  * A new `session` on every start, so bioarena tells a restart from a lost
    packet; `seq` + 1 per datagram, so it drops reordered ones.
  * Send the moment a count rises; otherwise a heartbeat every 100 ms. The
    heartbeat is bioarena's only liveness signal -- 1 s without one and the
    operator's screen says OFFLINE.
  * `age_ms` is measured from when the camera CAPTURED the frame that showed
    the ball, not from when the code finished with it.

Why the latency matters: the AUTO winner is decided at T+23.000 s from the
counts bioarena holds at that instant. A ball scored in AUTO whose datagram is
still inside this program then is credited to the next shift and does not
decide the winner. The budget is <= 100 ms camera-to-credited typical, 250 ms
p99, and the camera and vision code own almost all of it (spec 5.2).

Standard library only, deliberately: the sender must run on whatever laptop is
wired into the field switch, and `Receiver` -- a stand-in for bioarena's side,
for commissioning the link before the field computer exists -- must run
anywhere at all.
"""
from __future__ import annotations

import json
import random
import re
import socket
import threading
import time
from typing import Callable, Dict, Optional, Tuple

VERSION = 1
FMS_HOST = "10.0.100.5"        # the address driver stations are hardcoded to
FMS_PORT = 8411
HEARTBEAT_S = 0.100            # spec 4.3
OFFLINE_S = 1.0                # spec 5.3: bioarena's offline threshold
MAX_DATAGRAM = 512             # spec 4.2
MAX_INFO = 64
HUBS = ("red", "blue")
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def new_session() -> str:
    """Random hex, fresh per process start (spec 4.2)."""
    return f"{random.SystemRandom().getrandbits(32):08x}"


def encode(session: str, seq: int, red: int, blue: int,
           age_ms: Optional[int] = None, info: Optional[str] = None) -> bytes:
    msg = {"v": VERSION, "session": session, "seq": seq,
           "red": int(red), "blue": int(blue)}
    if age_ms is not None:
        msg["age_ms"] = max(0, int(age_ms))
    if info:
        msg["info"] = info[:MAX_INFO]
    data = json.dumps(msg, separators=(",", ":")).encode()
    if len(data) > MAX_DATAGRAM:
        raise ValueError(f"datagram is {len(data)} bytes, over {MAX_DATAGRAM}")
    return data


def parse_target(text: str) -> Tuple[str, int]:
    """'10.0.100.5' or '10.0.100.5:8411' -> (host, port)."""
    host, _, port = text.rpartition(":")
    if not host:
        return text, FMS_PORT
    return host, int(port)


class FeedSender:
    """Both hubs' cumulative counts, sent to bioarena.

    Thread-safe: with one camera per hub, two capture loops call `score`
    concurrently and a heartbeat thread calls `heartbeat`, and every datagram
    must carry the next `seq` and both current counts.

    `sock` and `clock` are injectable so the tests can run it without a
    network; `clock` must be the same clock the capture timestamps are on
    (time.monotonic), or `age_ms` is meaningless.
    """

    def __init__(self, target: Tuple[str, int] = (FMS_HOST, FMS_PORT),
                 session: Optional[str] = None, sock=None,
                 clock: Callable[[], float] = time.monotonic):
        self.target = target
        self.session = session or new_session()
        if not SESSION_RE.match(self.session):
            raise ValueError(f"session {self.session!r} is not 1-32 of "
                             f"[A-Za-z0-9_-]")
        self.clock = clock
        if sock is None:
            # One socket for the process lifetime, so bioarena's replies --
            # sent to our source port -- come back to it (spec 4.3).
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
        self.sock = sock
        self.counts: Dict[str, int] = {h: 0 for h in HUBS}
        self.seq = 0
        self.info = ""
        self.last_sent = float("-inf")
        self.last_increment: Optional[float] = None
        self.send_errors = 0
        self.last_error = ""
        # What bioarena said back. Shown, never depended on (spec 4.4).
        self.last_reply: Optional[Dict] = None
        self.last_reply_at = float("-inf")
        self.rtt_ms: Optional[float] = None
        self._sent_at: Dict[int, float] = {}
        self._lock = threading.Lock()

    # -- sending ----------------------------------------------------------
    def score(self, hub: str, n: int, captured_at: float) -> None:
        """`n` more balls into `hub`, seen in a frame captured at `captured_at`.

        Sent now, not at the next heartbeat: this is the datagram that carries
        the latency budget.
        """
        if hub not in self.counts:
            raise ValueError(f"no hub {hub!r}; bioarena knows {HUBS}")
        if n <= 0:
            return            # the counts may only rise within a session
        with self._lock:
            self.counts[hub] += n
            if self.last_increment is None or captured_at > self.last_increment:
                self.last_increment = captured_at
            self._send()

    def heartbeat(self) -> bool:
        """Send the unchanged counts if 100 ms have passed. True if it sent."""
        with self._lock:
            if self.clock() - self.last_sent < HEARTBEAT_S:
                return False
            self._send()
            return True

    def _send(self) -> None:
        self.seq += 1
        now = self.clock()
        age = None
        if self.last_increment is not None:
            age = round((now - self.last_increment) * 1000)
        data = encode(self.session, self.seq, self.counts["red"],
                      self.counts["blue"], age, self.info)
        try:
            self.sock.sendto(data, self.target)
        except OSError as e:
            # No route (cable out, wrong VLAN) must not stop the counting:
            # the counts are cumulative, so the next datagram that gets
            # through carries every ball scored meanwhile.
            self.send_errors += 1
            self.last_error = str(e)
        self.last_sent = now
        self._sent_at[self.seq] = now
        if len(self._sent_at) > 256:
            for k in sorted(self._sent_at)[:128]:
                del self._sent_at[k]

    # -- replies ----------------------------------------------------------
    def poll_replies(self) -> int:
        """Read every status reply waiting. Never blocks. Returns how many."""
        got = 0
        while True:
            try:
                data, _ = self.sock.recvfrom(2048)
            except (BlockingIOError, InterruptedError):
                return got
            except OSError:
                # Linux reports an earlier ICMP port-unreachable here, which
                # is just "bioarena is not listening" -- not ours to fix.
                return got
            try:
                reply = json.loads(data)
            except ValueError:
                continue
            if not isinstance(reply, dict):
                continue
            got += 1
            now = self.clock()
            with self._lock:
                sent = self._sent_at.pop(reply.get("seq"), None)
                if sent is not None:
                    self.rtt_ms = round((now - sent) * 1000, 1)
                self.last_reply = reply
                self.last_reply_at = now

    def linked(self) -> bool:
        return self.clock() - self.last_reply_at < OFFLINE_S


# -- a stand-in for bioarena, for commissioning --------------------------

class Receiver:
    """bioarena's acceptance rules (spec 4.5) and baseline arithmetic (6.2).

    Not bioarena: it has no match clock and no shifts. It exists so the
    counter can be brought up and checked on a laptop -- `run.py hubfeed-listen`
    -- before the field computer is available, and so the tests can hold the
    sender to the same rules the real receiver applies.
    """

    def __init__(self, counter_addr: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.counter_addr = counter_addr
        self.clock = clock
        self.session: Optional[str] = None
        self.seq = -1
        self.current = {h: 0 for h in HUBS}
        self.baseline = {h: 0 for h in HUBS}
        self.carried = {h: 0 for h in HUBS}
        self.last_accept = float("-inf")
        self.age_ms: Optional[int] = None
        self.info = ""
        self.restarts = 0
        self.dropped: Dict[str, int] = {}

    def reset_match(self) -> None:
        """What bioarena does at LoadMatch/StartMatch: zero match-relative."""
        self.baseline = dict(self.current)
        self.carried = {h: 0 for h in HUBS}

    def match_counts(self) -> Dict[str, int]:
        return {h: self.carried[h] + self.current[h] - self.baseline[h]
                for h in HUBS}

    def online(self) -> bool:
        return self.clock() - self.last_accept < OFFLINE_S

    def _drop(self, reason: str) -> Tuple[bool, str]:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        return False, reason

    def accept(self, data: bytes, source: str) -> Tuple[bool, str]:
        """Apply spec 4.5. Returns (accepted, reason or event)."""
        if self.counter_addr and source != self.counter_addr:
            return self._drop("unknown source")
        if len(data) > MAX_DATAGRAM:
            return self._drop("oversize")
        try:
            msg = json.loads(data)
        except ValueError:
            return self._drop("malformed json")
        if not isinstance(msg, dict) or msg.get("v") != VERSION:
            return self._drop("wrong version")
        session, seq = msg.get("session"), msg.get("seq")
        red, blue = msg.get("red"), msg.get("blue")
        if not (isinstance(session, str) and SESSION_RE.match(session)
                and all(isinstance(x, int) and not isinstance(x, bool)
                        and x >= 0 for x in (seq, red, blue))):
            return self._drop("missing or bad field")
        event = "count"
        if session != self.session:
            if self.session is not None:
                # Keep what the old session contributed to this match; the
                # new one starts from zero (spec 6.2).
                for h in HUBS:
                    self.carried[h] += self.current[h] - self.baseline[h]
                self.restarts += 1
                event = "hub counter restarted"
            self.session = session
            self.current = {h: 0 for h in HUBS}
            self.baseline = {h: 0 for h in HUBS}
        else:
            if seq <= self.seq:
                return self._drop("duplicate or reordered")
            if red < self.current["red"] or blue < self.current["blue"]:
                return self._drop("count went backwards")
        rose = red > self.current["red"] or blue > self.current["blue"]
        self.seq = seq
        self.current = {"red": red, "blue": blue}
        self.last_accept = self.clock()
        if rose and isinstance(msg.get("age_ms"), int):
            self.age_ms = msg["age_ms"]
        self.info = str(msg.get("info", ""))[:MAX_INFO]
        return True, event if (rose or event != "count") else "heartbeat"

    def status(self) -> Dict:
        """The reply's shape (spec 4.4), with no match to report."""
        m = self.match_counts()
        return {"v": VERSION, "seq": self.seq, "match_state": "PRE_MATCH",
                "match_time_s": 0, "shift": "NONE",
                "hub_active": {h: True for h in HUBS},
                "match_count": m, "credited": m,
                "auto_count": {h: 0 for h in HUBS}}


def listen(port: int = FMS_PORT, bind: str = "0.0.0.0",
           counter_addr: Optional[str] = None, out=print,
           stop: Optional[threading.Event] = None) -> Receiver:
    """Run `Receiver` on a real socket, printing every change. Blocks."""
    rx = Receiver(counter_addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind, port))
    sock.settimeout(0.25)
    out(f"listening for the hub counter on udp {bind}:{port}"
        + (f", accepting only {counter_addr}" if counter_addr else ""))
    was_online = False
    try:
        while stop is None or not stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                if was_online and not rx.online():
                    out("OFFLINE: nothing accepted for 1 s")
                    was_online = False
                continue
            ok, what = rx.accept(data, addr[0])
            if not ok:
                if rx.dropped[what] == 1:
                    out(f"dropped from {addr[0]}: {what} (said once per reason)")
                continue
            if not was_online:
                out(f"ONLINE: session {rx.session} from {addr[0]}:{addr[1]}")
                was_online = True
            if what != "heartbeat":
                m = rx.match_counts()
                out(f"{what}: red {m['red']}  blue {m['blue']}  "
                    f"age {rx.age_ms} ms  {rx.info}")
            sock.sendto(json.dumps(rx.status(), separators=(",", ":")).encode(),
                        addr)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return rx
