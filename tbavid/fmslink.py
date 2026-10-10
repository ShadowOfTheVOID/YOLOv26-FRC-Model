"""The hub counter talking to frc-fms (github.com/arnan-bajaj/frc-fms).

frc-fms is a scrimmage FMS whose vision side only reports timestamped fuel
events -- (t, hub, n), t in Unix seconds -- to

    POST {fms}/api/vision/events     header X-Vision-Key: <vision_key>
    {"events": {"red": [[t, n], ...], "blue": [...]}, "status": {...}, "source": "live"}

and decides from its own match timeline which period each ball belongs to.
That is a different contract from bioarena's UDP feed (cumulative counts,
heartbeat, `age_ms`), so it gets its own sender with the same face as
`hubfeed.FeedSender`: `hubcount.run`, `run.py hubfeed` and the web page use
either one unchanged. A target written as a URL picks this one:

    run.py hubfeed --setup cams.json --target http://VISIONKEY@192.168.1.10:8000

The key rides in the URL's user part so the page's one target box carries
it. Standard library only, like hubfeed.py.

Two things the FMS relies on that bioarena did not:
- **Timestamps are wall clock.** Frames are stamped on time.monotonic() (the
  clock `age_ms` is measured on); each is moved to time.time() with the
  offset between the two clocks, so a ball lands in the period it was seen
  in, not the one it was posted in.
- **Nothing may be dropped.** frc-fms adds events, it does not take a running
  total, so an event lost in a network blip is a ball lost. Events stay
  buffered until a POST that carried them succeeds -- frc-fms's own vision
  sender does the same -- and the POST runs on its own thread, so a slow
  FMS never holds a camera loop (a blocked loop misses balls silently).
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

HUBS = ("red", "blue")
POST_EVERY_S = 0.25        # frc-fms's own sender batches at this period
OFFLINE_S = 2.0            # no successful POST for this long = not linked


def is_fms_target(target: str) -> bool:
    return str(target).strip().lower().startswith(("http://", "https://"))


def split_target(target: str):
    """'http://KEY@host:8000' -> ('http://host:8000', 'KEY')."""
    u = urlsplit(target.strip())
    if not u.hostname:
        raise ValueError(f"not an frc-fms URL: {target!r}")
    key = u.username or ""
    host = u.hostname + (f":{u.port}" if u.port else "")
    return urlunsplit((u.scheme, host, u.path.rstrip("/"), "", "")), key


class FmsSender:
    """Fuel events to frc-fms, in the shape hubcount.run expects of a sender."""

    def __init__(self, target: str, key: str = "", post=None,
                 clock=time.monotonic, wall=time.time, start: bool = True):
        url, url_key = split_target(target)
        self.url = url
        self.key = key or url_key
        if not self.key:
            raise ValueError("frc-fms needs its vision_key: put it in the URL, "
                             "http://KEY@host:8000 (server.vision_key in event.yaml)")
        self.session = "frc-fms"
        self.peer = "frc-fms"      # what status lines call the other end
        self.counts: Dict[str, int] = {h: 0 for h in HUBS}
        self.info = ""
        self.hub_status: Dict[str, Dict] = {}
        self.send_errors = 0
        self.last_error = ""
        self.last_reply: Optional[Dict] = None
        self.last_reply_at = float("-inf")
        self.rtt_ms: Optional[float] = None
        self.clock, self.wall = clock, wall
        self._post = post or self._http_post
        self._buf: Dict[str, List[List[float]]] = {h: [] for h in HUBS}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if start:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="frc-fms sender")
            self._thread.start()

    # -- the FeedSender face ------------------------------------------------
    def score(self, hub: str, n: int, captured_at: float) -> None:
        """`n` balls into `hub`, seen in a frame captured at `captured_at`
        (time.monotonic()); queued with its wall-clock time."""
        if hub not in self.counts:
            raise ValueError(f"no hub {hub!r}; frc-fms knows {HUBS}")
        if n <= 0:
            return
        t = captured_at + (self.wall() - self.clock())
        with self._lock:
            self.counts[hub] += n
            self._buf[hub].append([round(t, 4), int(n)])

    def heartbeat(self) -> bool:
        return False          # the posting thread runs on its own period

    def poll_replies(self) -> int:
        return 0              # replies are read by the posting thread

    def linked(self) -> bool:
        return self.clock() - self.last_reply_at < OFFLINE_S

    def close(self) -> None:
        """Stop, after one last attempt to deliver what is queued."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.flush()

    # -- posting --------------------------------------------------------------
    def pending(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._buf.values())

    def flush(self) -> bool:
        """POST everything queued once. True if the FMS took it."""
        with self._lock:
            batch = {h: [e[:] for e in v] for h, v in self._buf.items()}
            # hubcount.run fills hub_status (fps, counter, error) for the
            # hubs it counts; before it does, nothing is claimed about fps.
            live = dict(self.hub_status)
            status = {h: dict(live.get(h) or {"counter": "tbavid colour"},
                              session_total=self.counts[h], info=self.info)
                      for h in (live or HUBS)}
        body = {"events": batch, "status": status, "source": "live"}
        sent_at = self.clock()
        try:
            reply = self._post(f"{self.url}/api/vision/events", body, self.key)
        except Exception as e:                      # network, HTTP, JSON
            self.send_errors += 1
            self.last_error = str(e)
            return False
        with self._lock:                            # drop only what was sent
            for h in batch:
                del self._buf[h][:len(batch[h])]
        self.last_reply = reply if isinstance(reply, dict) else {}
        self.last_reply_at = self.clock()
        self.rtt_ms = round((self.last_reply_at - sent_at) * 1000, 1)
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.flush()
            self._stop.wait(POST_EVERY_S)

    @staticmethod
    def _http_post(url: str, body: Dict, key: str) -> Dict:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "X-Vision-Key": key})
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read() or b"{}")


def split_targets(target: str) -> List[str]:
    """'http://KEY@m4:8000, 10.0.100.5:8411' -> both. Commas or spaces
    separate; neither appears in a host:port or in these URLs."""
    return [t for t in re.split(r"[,\s]+", str(target).strip()) if t]


def make_sender(target: str, udp_host: str = ""):
    """A bioarena UDP sender for host[:port], an frc-fms one for a URL, and a
    FanOut for several of them separated by commas. `udp_host` replaces the
    host of every UDP target (practice mode's local stand-in)."""
    from .hubfeed import FeedSender, parse_target
    targets = split_targets(target) or [target]
    if sum(not is_fms_target(t) for t in targets) > 1:
        # Checked before any sender exists: FmsSender starts a thread.
        raise ValueError("send counts to one bioarena only (one host:port); "
                         "add Watchtower as http://KEY@host:8000 beside it")
    senders = []
    for t in targets:
        if is_fms_target(t):
            senders.append(FmsSender(t))
        else:
            host, port = parse_target(t)
            senders.append(FeedSender((udp_host or host, port)))
    return senders[0] if len(senders) == 1 else FanOut(senders)


class FanOut:
    """Every ball to several field systems at once, in a sender's face.

    The scrimmage runs Watchtower (HTTP on the M4, port 8000) beside
    bioarena (UDP 8411), each on its own address. Watchtower only forwards
    to bioarena from its own vision runner (run_vision.py `feeds:`), which
    the Watchtower app does not run: with one target, the app's counter
    reached Watchtower and bioarena got nothing. Each target keeps its own
    sender and rules -- the UDP one its session, seq and heartbeat, the HTTP
    one its never-dropped queue -- so a dead Watchtower cannot stall the feed
    bioarena decides AUTO from, nor the reverse.

    The bioarena target is the primary: while it answers, the match state
    and round trip come from it, as bioarena is the official score. The
    Watchtower app always lists bioarena (the spec's 10.0.100.5:8411 unless
    vision.yaml says otherwise), so a venue with no bioarena must still read
    as linked: `linked` is any target answering, and the state comes from
    whichever leads (`lead`). Errors from any target are counted and named.
    """

    def __init__(self, senders):
        if len(senders) < 2:
            raise ValueError("FanOut wants two or more senders")
        self.senders = list(senders)
        udp = [x for x in self.senders if not isinstance(x, FmsSender)]
        if len(udp) > 1:
            # bioarena reads a second session as the counter restarting, and
            # two of ours alternating re-carry each other's totals.
            raise ValueError("send to one bioarena only (one UDP host:port)")
        self.primary = udp[0] if udp else self.senders[0]

    def lead(self):
        """The primary while it answers, else the first target that does."""
        if self.primary.linked():
            return self.primary
        return next((x for x in self.senders if x.linked()), self.primary)

    # what hubcount.run, hubapp and run.py read
    session = property(lambda self: self.primary.session)
    counts = property(lambda self: self.primary.counts)
    peer = property(lambda self: getattr(self.lead(), "peer", "bioarena"))
    last_reply = property(lambda self: self.lead().last_reply)
    rtt_ms = property(lambda self: self.lead().rtt_ms)
    # bioarena's own reply and link, whoever leads: the partner box's relay
    # and /board need bioarena's status, never Watchtower's {"record": ...}
    # (and /board must not flip views each time bioarena's 1 s link blinks).
    field_reply = property(lambda self: self.primary.last_reply)
    field_linked = property(lambda self: self.primary.linked())
    send_errors = property(lambda self: sum(x.send_errors for x in self.senders))

    @property
    def last_error(self) -> str:
        return "; ".join(f"{_name(x)}: {x.last_error}" for x in self.senders
                         if x.send_errors and x.last_error)

    def _set_all(name):
        def get(self):
            return getattr(self.primary, name)

        def put(self, value):
            for x in self.senders:
                setattr(x, name, value)
        return property(get, put)
    info = _set_all("info")
    hub_status = _set_all("hub_status")
    del _set_all

    def score(self, hub: str, n: int, captured_at: float) -> None:
        for x in self.senders:
            x.score(hub, n, captured_at)

    def heartbeat(self) -> bool:
        return any([x.heartbeat() for x in self.senders])

    def poll_replies(self) -> int:
        return sum(x.poll_replies() for x in self.senders)

    def linked(self) -> bool:
        return any(x.linked() for x in self.senders)

    def links(self) -> Dict[str, bool]:
        return {_name(x): x.linked() for x in self.senders}

    def describe(self) -> str:
        return ", ".join(_name(x) for x in self.senders)

    def close(self) -> None:
        for x in self.senders:
            if hasattr(x, "close"):
                x.close()

    def pending(self) -> int:
        return sum(x.pending() for x in self.senders if hasattr(x, "pending"))


def _name(sender) -> str:
    """'udp 10.0.100.5:8411' / 'http://m4:8000' -- never the key."""
    if isinstance(sender, FmsSender):
        return sender.url
    t = getattr(sender, "target", None)
    if isinstance(t, tuple) and len(t) == 2:
        return f"udp {t[0]}:{t[1]}"
    return str(getattr(sender, "peer", "bioarena"))


def sender_links(sender) -> Dict[str, bool]:
    """{destination: answering} for one sender or a FanOut's every target."""
    if isinstance(sender, FanOut):
        return sender.links()
    return {_name(sender): sender.linked()}


def sender_name(sender) -> str:
    """What the page and log call a sender's destination, without the key:
    'http://m4:8000', '10.0.100.5:8411', or a FanOut's list."""
    if isinstance(sender, FanOut):
        return sender.describe()
    if isinstance(sender, FmsSender):
        return sender.url
    return "{}:{}".format(*sender.target)
