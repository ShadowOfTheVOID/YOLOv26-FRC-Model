"""Who scored: the broadcast's fuel counter, matched to the launches before it.

Watching balls land is where broadcast tracking fails: on the first 30 s of
Einstein 1, ShotCounter saw 12 of about 70 red balls go in and 0 of about 70
blue, because a 15 px ball in a stream is lost long before the funnel. But
two things a broadcast does well are enough without the landing:

  * the alliance fuel counters burned into the overlay (scoreboard.py /
    live.py read them) say exactly WHEN each ball was credited;
  * launches happen beside the robot, where tracking holds, and the bumper
    reader says WHICH robot launched.

A ball credited at time t was launched one delay earlier -- its flight plus
whatever the field and the overlay add. So each rise of the counter is given
to that alliance's launches nearest t - delay, each launch used at most once
(misses are launches too, and are simply left unused). A rise with no launch
in reach stays unattributed: a shooter the tracker never saw is not guessed.

The delay is not assumed. `best_delay` tries a range and keeps the one that
explains the most credited balls, per match, so a broadcast whose overlay
runs late fits itself.

Pure state: no video, no model. TBA's breakdown is the check on the totals.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DELAYS = [round(0.1 * i, 1) for i in range(0, 41)]   # 0.0 - 4.0 s
TOLERANCE_S = 0.6


def launches_from(events: Iterable[Dict]) -> List[Tuple[float, int, str]]:
    """(launch time, robot number, alliance) for every shot ShotCounter
    followed, made or missed -- made/missed is exactly what is not trusted
    here. Shots with no robot or alliance are left out."""
    out = []
    for e in events:
        if e.get("robot") is None or e.get("alliance") is None:
            continue
        t = e.get("launched", e.get("t"))
        if t is None:
            continue
        out.append((float(t), int(e["robot"]), str(e["alliance"])))
    return sorted(out)


def match(rises: Sequence[Dict], launches: Sequence[Tuple[float, int, str]],
          delay: float, tolerance: float = TOLERANCE_S,
          errors: Optional[List[float]] = None) -> Tuple[Dict, Dict[str, int]]:
    """Give each credited ball to an unused launch of its alliance.

    `rises` are scoreboard.to_events rows: {"t", "alliance", "balls"}.
    Returns ({robot: balls}, {alliance: balls nobody launched in reach}).
    Rises are handled in time order, each taking the launches nearest to
    t - delay, so a burst of rises consumes a burst of launches in order.
    `errors`, when given, collects each match's timing error.
    """
    used = set()
    credit: Dict[int, int] = defaultdict(int)
    unmatched: Dict[str, int] = defaultdict(int)
    for r in sorted(rises, key=lambda r: r["t"]):
        want = float(r["t"]) - delay
        near = sorted((abs(t - want), i) for i, (t, _, a) in enumerate(launches)
                      if a == r["alliance"] and i not in used and abs(t - want) <= tolerance)
        for k in range(int(r["balls"])):
            if k < len(near):
                err, i = near[k]
                used.add(i)
                credit[launches[i][1]] += 1
                if errors is not None:
                    errors.append(err)
            else:
                unmatched[r["alliance"]] += 1
    return dict(credit), dict(unmatched)


def best_delay(rises: Sequence[Dict], launches: Sequence[Tuple[float, int, str]],
               delays: Sequence[float] = DELAYS, tolerance: float = TOLERANCE_S) -> float:
    """The launch-to-credit delay that explains the most credited balls; among
    delays that explain as many (a whole window of them does, with a
    tolerance), the one whose matches are tightest."""
    best, best_key = delays[0], None
    for d in delays:
        errs: List[float] = []
        credit, _ = match(rises, launches, d, tolerance, errs)
        key = (sum(credit.values()), -round(sum(errs), 6))
        if best_key is None or key > best_key:
            best, best_key = d, key
    return best


SHARE_WINDOW_S = 1.0


def share(rises: Sequence[Dict], launches: Sequence[Tuple[float, int, str]],
          delay: float, window: float = SHARE_WINDOW_S) -> Tuple[Dict, Dict[str, int]]:
    """Split each credited ball among the robots of its alliance that were
    launching around t - delay, by how many launches each had there.

    `match` pairs one ball with one launch, and the tracker resolves far
    fewer launches than balls: on the first 30 s of Einstein 1 the counters
    credited 193 balls and ShotCounter followed 56 launches, so matching
    explained 50. A launch is better read as "this robot was shooting then";
    shared that way the same 30 s attributed 172 of 193 at a 1.0 s window
    (142 at 0.6 s, 186 at 2.0 s -- wider buys coverage by blaming whoever
    was nearby). Credits are fractional; round for display only.
    """
    credit: Dict[int, float] = defaultdict(float)
    unmatched: Dict[str, int] = defaultdict(int)
    for r in rises:
        want = float(r["t"]) - delay
        near = [robot for t, robot, a in launches
                if a == r["alliance"] and abs(t - want) <= window]
        if not near:
            unmatched[r["alliance"]] += int(r["balls"])
            continue
        for robot in near:
            credit[robot] += int(r["balls"]) / len(near)
    return dict(credit), dict(unmatched)


def by_team(credit: Dict[int, int], unmatched: Dict[str, int],
            teams: Dict[int, int], alliance_of: Dict[int, str]) -> Dict:
    """Robot numbers -> teams (TeamVotes / --teams). Robots with no team
    become "unnamed <alliance>", next to the balls nobody launched."""
    out: Dict = defaultdict(float)
    for robot, n in credit.items():
        team = teams.get(robot)
        out[team if team is not None else f"unnamed {alliance_of.get(robot, '?')}"] += n
    for alliance, n in unmatched.items():
        out[f"no launch seen ({alliance})"] += n
    return dict(out)


def attribute(rises: Sequence[Dict], events: Iterable[Dict], teams: Dict[int, int],
              delay: Optional[float] = None, tolerance: float = TOLERANCE_S,
              window: Optional[float] = SHARE_WINDOW_S) -> Dict:
    """Per-team credit for one match. The delay is fitted by one-to-one
    matching (it needs exact pairs); the credit is then shared over a
    `window` (see `share`), or one-to-one when `window` is None."""
    launches = launches_from(events)
    if delay is None:
        delay = best_delay(rises, launches, tolerance=tolerance)
    if window is None:
        credit, unmatched = match(rises, launches, delay, tolerance)
    else:
        credit, unmatched = share(rises, launches, delay, window)
    alliance_of = {r: a for _, r, a in launches}
    total = sum(int(r["balls"]) for r in rises)
    named = sum(n for r, n in credit.items() if teams.get(r) is not None)
    return {"delay_s": delay, "credited": total,
            "attributed": round(sum(credit.values()), 3), "named": round(named, 3),
            "by_team": {k: round(v, 2) for k, v in
                        by_team(credit, unmatched, teams, alliance_of).items()},
            "by_robot": credit}
