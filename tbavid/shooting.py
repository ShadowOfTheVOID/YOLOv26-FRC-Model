"""Per-robot shooting: who shot, how many went in, how many missed.

`count.py` answers "how many balls went into each hub". Scouting needs the two
questions after it: *whose* were they, and what about the ones that did not go
in? `count.py` cannot see a miss at all -- a ball that never reaches a hub is
simply not an event there. This module follows each ball from the robot that
launched it to wherever it ended, and files the outcome against that robot.

Like `BallCounter`, it is pure state and arithmetic: robot tracks and ball
tracks in, per-robot tallies out. No model, no video, no database, so every
rule below is testable without a GPU.

## What a shot is

A ball track that **starts at a robot and then gets clear of it**. Both halves
matter:

  * *Starts at a robot*: its first position is inside a robot's box, grown by
    `launch_pad` (a ball leaving a shooter is usually first detected just
    above the bumpers, not inside them). That robot is the shooter; if two
    boxes contain the point, the nearer centre wins.
  * *Gets clear of it*: at some point it is at least `min_clear` robot-widths
    from the shooter's box **as the robot is now**, not as it was. Measuring
    from the start point instead would call every ball riding in a hopper a
    shot the moment its robot drove anywhere. A ball that stays with its robot
    was carried, not shot.

One exception: a ball that starts at a robot and ends in a hub is a shot
however little it cleared. A robot shooting from against the hub has almost
no room to clear it, and the hub itself proves the ball left.

## Made and missed

  * **made** -- the shot ended inside its own alliance's hub region, under the
    same hold as `BallCounter`: a new track appearing near where it vanished,
    within `reacquire_frames`, means it passed over rather than in, and the
    shot carries on.
  * **wrong_hub** -- it ended inside the other alliance's hub. Recorded, not
    judged: whether that scores for anyone is the game's rule, not this
    module's.
  * **missed** -- it ended anywhere else.

A ball that goes into a hub without ever being seen to leave a robot --
launch hidden behind another robot, or first detected mid-air -- is counted
as an **unattributed** make, so the per-hub totals still agree with
`BallCounter` even when the shooter is unknown.

## Broken tracks

The first detector measured a per-frame recall of 0.62 on its validation set,
so a ball in flight regularly drops out long enough for the tracker to end its
track, and comes back under a new id. Left alone, every such shot would read
as a miss by its shooter followed by an unattributed make -- wrong twice. So a
new track that appears within `stitch_px` of where a lost ball was heading
(its last velocity, extrapolated) is the same ball. That applies both to a
ball that is briefly missing and to a shot already settled as a miss but
still inside its `stitch_frames` hold; a miss is only final once that window
has passed with nothing picking it up.

A new track that starts inside a robot is never stitched: that is a launch,
and a fresh shot is more likely than a lost ball landing exactly on a robot.

## What this does not know

  * **Which team a robot is.** Tallies are per robot *track*, with its
    alliance. `by_team()` folds tracks into team numbers once something says
    which is which -- an operator at match start, or `identify.py`.
  * **Game state.** Whether a hub was active when a ball went in.
  * **Anything it never saw.** A shot occluded for its whole flight does not
    exist here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .count import Box, centre, hub_of

# A ball seen for fewer frames than this, across all its stitched pieces,
# never existed. Same meaning and value as count.MIN_TRACK_FRAMES.
MIN_TRACK_FRAMES = 3

# Frames a ball may go unseen before its track is treated as ended.
VANISH_FRAMES = 4

# The pass-over hold, shared with count.py.
REACQUIRE_FRAMES = 12
REACQUIRE_PX = 90.0

# How far a robot's box is grown, as a fraction of its size, when deciding
# whether a ball started at it.
LAUNCH_PAD = 0.3

# How far, in widths of the shooter's box, a ball must get from that robot
# before it has been shot rather than carried.
MIN_CLEAR = 1.0

# How long a miss is held open for the rest of a broken flight to turn up,
# and how close to the predicted position it must appear.
STITCH_FRAMES = 10
STITCH_PX = 80.0

# How long a robot's last box is remembered after it stops being detected, so
# a shooter that flickers out at the moment of launch is still found.
ROBOT_MEMORY = 15

Point = Tuple[float, float]


def grow(box: Box, pad: float) -> Box:
    x, y, w, h = box
    return (x - w * pad, y - h * pad, w * (1 + 2 * pad), h * (1 + 2 * pad))


def contains(box: Box, point: Point) -> bool:
    x, y, w, h = box
    return x <= point[0] <= x + w and y <= point[1] <= y + h


def gap(box: Box, point: Point) -> float:
    """Distance from a point to the nearest edge of a box; 0 inside it."""
    x, y, w, h = box
    dx = max(x - point[0], 0.0, point[0] - (x + w))
    dy = max(y - point[1], 0.0, point[1] - (y + h))
    return (dx * dx + dy * dy) ** 0.5


class _Ball:
    __slots__ = ("tid", "t0", "first", "last", "prev", "last_frame", "prev_frame",
                 "seen", "missing", "shooter", "shooter_alliance", "cleared",
                 "started_in", "shot")

    def __init__(self, tid: int, frame: int, t: float, point: Point,
                 shooter: Optional[int], shooter_alliance: Optional[str],
                 started_in: Optional[str]):
        self.tid = tid
        self.t0 = t
        self.first = point
        self.last = point
        self.prev = point
        self.last_frame = frame
        self.prev_frame = frame
        self.seen = 1
        self.missing = 0
        self.shooter = shooter          # robot track id it started at, if any
        # Taken at launch: the robot may have left memory by the time it lands.
        self.shooter_alliance = shooter_alliance
        self.cleared = 0.0              # furthest it got from the shooter's box
        self.started_in = started_in    # hub it was first seen inside, if any
        self.shot: Optional["_Shot"] = None

    def velocity(self) -> Point:
        dt = self.last_frame - self.prev_frame
        if dt <= 0:
            return (0.0, 0.0)
        return ((self.last[0] - self.prev[0]) / dt,
                (self.last[1] - self.prev[1]) / dt)

    def predict(self, frame: int) -> Point:
        vx, vy = self.velocity()
        dt = frame - self.last_frame
        return (self.last[0] + vx * dt, self.last[1] + vy * dt)


class _Shot:
    """One ball's flight from a robot, across however many track pieces."""
    __slots__ = ("robot", "alliance", "t0", "seen", "pieces", "confirmed")

    def __init__(self, robot: Optional[int], alliance: Optional[str], t0: float):
        self.robot = robot
        self.alliance = alliance
        self.t0 = t0
        self.seen = 0
        self.pieces = 1
        # False until the ball has been seen clear of its robot (or in a hub).
        # Until then it may be a ball being carried.
        self.confirmed = False


class _Held:
    """A settled-looking ending, waiting out its window before it is final."""
    __slots__ = ("kind", "shot", "point", "velocity", "frame", "t", "hub",
                 "started_in", "cleared")

    def __init__(self, kind: str, shot: Optional[_Shot], point: Point,
                 velocity: Point, frame: int, t: float, hub: Optional[str],
                 started_in: Optional[str] = None, cleared: float = 0.0):
        # "in_hub", "miss", or "maybe": a ball that ended still at its robot,
        # which is either carried or a flight broken before it got clear.
        self.kind = kind
        self.shot = shot            # None: an unattributed ball into a hub
        self.point = point
        self.velocity = velocity
        self.frame = frame
        self.t = t
        self.hub = hub
        self.started_in = started_in
        self.cleared = cleared

    def predict(self, frame: int) -> Point:
        dt = frame - self.frame
        return (self.point[0] + self.velocity[0] * dt,
                self.point[1] + self.velocity[1] * dt)


def _blank() -> Dict:
    return {"alliance": None, "shots": 0, "made": 0, "missed": 0, "wrong_hub": 0}


class ShotCounter:
    """Shots, makes and misses per robot track, from robots, balls and hubs."""

    def __init__(self, hubs: Dict[str, Box],
                 min_track_frames: int = MIN_TRACK_FRAMES,
                 vanish_frames: int = VANISH_FRAMES,
                 reacquire_frames: int = REACQUIRE_FRAMES,
                 reacquire_px: float = REACQUIRE_PX,
                 launch_pad: float = LAUNCH_PAD,
                 min_clear: float = MIN_CLEAR,
                 stitch_frames: int = STITCH_FRAMES,
                 stitch_px: float = STITCH_PX,
                 robot_memory: int = ROBOT_MEMORY):
        self.hubs = dict(hubs)
        self.min_track_frames = min_track_frames
        self.vanish_frames = vanish_frames
        self.reacquire_frames = reacquire_frames
        self.reacquire_px = reacquire_px
        self.launch_pad = launch_pad
        self.min_clear = min_clear
        self.stitch_frames = stitch_frames
        self.stitch_px = stitch_px
        self.robot_memory = robot_memory

        self.robots: Dict[int, Tuple[str, Box, int]] = {}   # tid -> alliance, box, frame
        self.balls: Dict[int, _Ball] = {}
        self.held: List[_Held] = []
        self.per_robot: Dict[int, Dict] = {}
        self.unattributed: Dict[str, int] = {a: 0 for a in hubs}
        self.into_hub: Dict[str, int] = {a: 0 for a in hubs}
        # Why balls did NOT become an outcome, so a count that looks wrong can
        # be read rather than guessed at.
        self.ignored: Dict[str, int] = {"too_short": 0, "carried": 0,
                                        "not_from_robot": 0, "started_inside": 0}
        self.stitched = 0
        self.reacquired = 0

    # -- the frame loop -----------------------------------------------------
    def update(self, frame: int, t: float,
               robots: Dict[int, Tuple[str, Box]],
               balls: Dict[int, Box]) -> List[Dict]:
        """One frame of robot tracks (tid -> (alliance, box)) and ball tracks
        (tid -> box). Returns the outcomes that became FINAL this frame."""
        for tid, (alliance, box) in robots.items():
            self.robots[tid] = (alliance, box, frame)
        for tid in [r for r, (_, _, f) in self.robots.items()
                    if frame - f > self.robot_memory]:
            del self.robots[tid]

        for tid, box in balls.items():
            point = centre(box)
            ball = self.balls.get(tid)
            if ball is None:
                self._arrive(tid, frame, t, point)
            else:
                ball.prev, ball.prev_frame = ball.last, ball.last_frame
                ball.last, ball.last_frame = point, frame
                ball.seen += 1
                ball.missing = 0
            self._measure_clearance(self.balls[tid])

        for tid in list(self.balls):
            ball = self.balls[tid]
            if ball.last_frame == frame:
                continue
            ball.missing += 1
            if ball.missing >= self.vanish_frames:
                del self.balls[tid]
                self._end(ball, frame, t)

        return self._settle(frame)

    def flush(self, frame: int, t: float) -> List[Dict]:
        """End of stream: end every ball and make every held outcome final."""
        for tid in list(self.balls):
            ball = self.balls.pop(tid)
            self._end(ball, frame, t)
        return self._settle(frame + max(self.reacquire_frames, self.stitch_frames) + 1)

    # -- arrivals -----------------------------------------------------------
    def _shooter_at(self, point: Point) -> Optional[int]:
        hits = [tid for tid, (_, box, _) in self.robots.items()
                if contains(grow(box, self.launch_pad), point)]
        if not hits:
            return None
        def dist(tid):
            cx, cy = centre(self.robots[tid][1])
            return (cx - point[0]) ** 2 + (cy - point[1]) ** 2
        return min(hits, key=dist)

    def _arrive(self, tid: int, frame: int, t: float, point: Point) -> None:
        """A track id not seen before: a new ball, or an old one found again."""
        at_robot = self._shooter_at(point)

        # 1. A lost ball of ours, briefly missing, reappearing where it was
        #    heading. Never a new track that starts at a robot: that is a launch.
        if at_robot is None:
            best, best_d = None, self.stitch_px
            for ball in self.balls.values():
                if ball.missing == 0:
                    continue
                px, py = ball.predict(frame)
                d = ((px - point[0]) ** 2 + (py - point[1]) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = ball, d
            if best is not None:
                del self.balls[best.tid]
                best.tid = tid
                best.prev, best.prev_frame = best.last, best.last_frame
                best.last, best.last_frame = point, frame
                best.seen += 1
                best.missing = 0
                self.balls[tid] = best
                self.stitched += 1
                return

        # 2. A held ending taken back: a miss whose flight turns out to go on,
        #    or a ball "in" a hub that came back out the other side.
        held = self._reclaim(point, frame, at_robot)
        if held is not None:
            ball = _Ball(tid, frame, t, point, None, None, None)
            if held.shot is not None:
                ball.shot = held.shot
                ball.shot.pieces += 1
                ball.shooter = held.shot.robot
                ball.shooter_alliance = held.shot.alliance
                # A confirmed shot already left its robot; a "maybe" carries on
                # measuring from where its first piece got to.
                ball.cleared = float("inf") if held.shot.confirmed else held.cleared
            else:
                # An unattributed ball that passed over a hub. It is still an
                # unattributed ball; it just has not gone in yet.
                ball.started_in = None
            self.balls[tid] = ball
            return

        # 3. A new ball.
        alliance = self.robots[at_robot][0] if at_robot is not None else None
        self.balls[tid] = _Ball(tid, frame, t, point, at_robot, alliance,
                                hub_of(self.hubs, point))

    def _reclaim(self, point: Point, frame: int,
                 at_robot: Optional[int]) -> Optional[_Held]:
        best, best_d = None, None
        for h in self.held:
            if h.kind == "in_hub":
                if frame - h.frame > self.reacquire_frames:
                    continue
                d = ((h.point[0] - point[0]) ** 2 + (h.point[1] - point[1]) ** 2) ** 0.5
                limit = self.reacquire_px
            else:
                if frame - h.frame > self.stitch_frames or at_robot is not None:
                    continue
                px, py = h.predict(frame)
                d = ((px - point[0]) ** 2 + (py - point[1]) ** 2) ** 0.5
                limit = self.stitch_px
            if d <= limit and (best_d is None or d < best_d):
                best, best_d = h, d
        if best is not None:
            self.held.remove(best)
            if best.kind == "in_hub":
                self.reacquired += 1
            else:
                self.stitched += 1
        return best

    def _measure_clearance(self, ball: _Ball) -> None:
        if ball.shooter is None or ball.cleared == float("inf"):
            return
        entry = self.robots.get(ball.shooter)
        if entry is None:
            return
        box = entry[1]
        width = max(box[2], 1.0)
        ball.cleared = max(ball.cleared, gap(box, ball.last) / width)

    # -- endings ------------------------------------------------------------
    def _end(self, ball: _Ball, frame: int, t: float) -> None:
        """A ball track has ended. Decide what it might have been."""
        hub = hub_of(self.hubs, ball.last)
        shot = ball.shot
        if shot is None and ball.shooter is not None:
            shot = _Shot(ball.shooter, ball.shooter_alliance, ball.t0)

        if shot is not None:
            shot.seen += ball.seen
            if not shot.confirmed:
                if ball.cleared >= self.min_clear or hub is not None:
                    shot.confirmed = True
                else:
                    # Still at its robot. Either it was carried, or the tracker
                    # broke a flight before the ball got clear -- which looks
                    # identical until the rest of the flight does or does not
                    # turn up. Deciding "carried" now lost that flight's shooter
                    # and made its landing someone else's.
                    self.held.append(_Held("maybe", shot, ball.last,
                                           ball.velocity(), frame, t, None,
                                           cleared=ball.cleared))
                    return
            if shot.seen < self.min_track_frames:
                self.ignored["too_short"] += 1
                return
            kind = "in_hub" if hub is not None else "miss"
            self.held.append(_Held(kind, shot, ball.last, ball.velocity(),
                                   frame, t, hub))
            return

        # Not a shot. It still counts for the hub if it went in, because the
        # per-hub totals must agree with BallCounter whether or not a shooter
        # was seen.
        if ball.seen < self.min_track_frames:
            self.ignored["too_short"] += 1
            return
        if hub is None:
            self.ignored["not_from_robot"] += 1
            return
        if ball.started_in == hub:
            self.ignored["started_inside"] += 1
            return
        self.held.append(_Held("in_hub", None, ball.last, ball.velocity(),
                               frame, t, hub))

    def _settle(self, frame: int) -> List[Dict]:
        """Held endings whose windows have passed are final."""
        out: List[Dict] = []
        for h in list(self.held):
            window = self.reacquire_frames if h.kind == "in_hub" else self.stitch_frames
            if frame - h.frame <= window:
                continue
            self.held.remove(h)
            if h.kind == "maybe":
                self.ignored["carried"] += 1
                continue
            out.append(self._record(h))
        return out

    def _record(self, h: _Held) -> Dict:
        if h.kind == "in_hub":
            self.into_hub[h.hub] = self.into_hub.get(h.hub, 0) + 1
        if h.shot is None:
            self.unattributed[h.hub] = self.unattributed.get(h.hub, 0) + 1
            return {"t": round(h.t, 3), "robot": None, "alliance": None,
                    "outcome": "made", "hub": h.hub}
        shot = h.shot
        stats = self.per_robot.setdefault(shot.robot, _blank())
        stats["alliance"] = stats["alliance"] or shot.alliance
        stats["shots"] += 1
        if h.kind == "miss":
            outcome = "missed"
        elif shot.alliance is None or h.hub == shot.alliance:
            outcome = "made"
        else:
            outcome = "wrong_hub"
        stats[outcome] += 1
        return {"t": round(h.t, 3), "robot": shot.robot, "alliance": shot.alliance,
                "outcome": outcome, "hub": h.hub if h.kind == "in_hub" else None,
                "launched": round(shot.t0, 3), "pieces": shot.pieces}

    # -- output -------------------------------------------------------------
    def hub_totals(self) -> Dict[str, int]:
        """Balls into each hub, attributed or not -- what BallCounter counts."""
        return dict(self.into_hub)

    def by_team(self, assignments: Dict[int, int]) -> Dict:
        """Fold robot tracks into teams. `assignments` maps track id -> team.

        A tracker that loses a robot and finds it again gives it a new id, so
        one team is often several tracks; this is where they become one row.
        Tracks nobody has assigned stay under None rather than vanishing --
        shots the scouting data cannot place are worth knowing about.
        """
        out: Dict = {}
        for tid, stats in self.per_robot.items():
            team = assignments.get(tid)
            row = out.setdefault(team, _blank())
            row["alliance"] = row["alliance"] or stats["alliance"]
            for key in ("shots", "made", "missed", "wrong_hub"):
                row[key] += stats[key]
        return out

    def report(self) -> List[str]:
        lines = []
        for tid, s in sorted(self.per_robot.items()):
            acc = f"{100 * s['made'] / s['shots']:.0f}%" if s["shots"] else "-"
            lines.append(f"robot {tid} ({s['alliance']}): {s['shots']} shots, "
                         f"{s['made']} made, {s['missed']} missed"
                         + (f", {s['wrong_hub']} into the other hub"
                            if s["wrong_hub"] else "") + f"  [{acc}]")
        if any(self.unattributed.values()):
            lines.append("unattributed makes (shooter not seen): "
                         + ", ".join(f"{a}={n}" for a, n in
                                     sorted(self.unattributed.items()) if n))
        named = {"too_short": "too few frames to be a ball",
                 "carried": "started at a robot but never left it",
                 "not_from_robot": "not launched by a robot, ended outside a hub",
                 "started_inside": "already in a hub when first seen"}
        for key, n in self.ignored.items():
            if n:
                lines.append(f"  ignored: {n} {named[key]}")
        if self.stitched:
            lines.append(f"  {self.stitched} broken flight(s) rejoined")
        if self.reacquired:
            lines.append(f"  {self.reacquired} ball(s) came back out of a hub")
        if self.held:
            lines.append(f"  {len(self.held)} outcome(s) still held open")
        return lines
