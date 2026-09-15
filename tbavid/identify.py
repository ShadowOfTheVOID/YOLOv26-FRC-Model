"""Attach team numbers to robot detections.

Why not just OCR the bumpers: measured on a 1920x504 Championship frame, a
bumper number is about 5 px tall and motion-blurred. Upscaled 6x and fed to
tesseract at three page-segmentation modes it returns the empty string every
time. No amount of threshold tuning recovers detail that was never sampled, so
reading a number off one frame is not a route to identity here.

What does work is that the problem is not open-ended. TBA tells us the exact
six teams on the field, so identity is a six-way choice, and a robot persists
across hundreds of frames. Two things follow:

  * Score a bumper crop against only the six candidates (template matching on
    rendered digit strings), never against all possible numbers.
  * Decide per TRACK, not per frame. Most frames are useless; a handful are
    sharp. Summing weak per-frame scores over a track lets those few carry the
    decision, and an assignment step keeps two tracks on the same alliance
    from claiming the same team.

This module is the identity layer. It needs a tracker upstream -- train the
detector, run `model.track(..., tracker="bytetrack.yaml")`, write detections
with `track_id`, then call `assign_tracks`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def vote(scores_by_frame: Iterable[Dict[int, float]]) -> Dict[int, float]:
    """Sum per-frame candidate scores into one score per team for a track."""
    total: Dict[int, float] = defaultdict(float)
    for frame_scores in scores_by_frame:
        for team, s in frame_scores.items():
            total[team] += s
    return dict(total)


def best_assignment(track_scores: Dict[int, Dict[int, float]],
                    teams: Sequence[int]) -> Dict[int, Optional[int]]:
    """Assign at most one team per track and one track per team.

    Greedy over the global score matrix rather than per-track argmax: three
    robots of one alliance look alike, and independent argmax happily gives two
    of them the same number.
    """
    pairs: List[Tuple[float, int, int]] = [
        (score, track, team)
        for track, scores in track_scores.items()
        for team, score in scores.items()
        if team in teams
    ]
    pairs.sort(reverse=True)

    out: Dict[int, Optional[int]] = {t: None for t in track_scores}
    used_teams: set = set()
    for score, track, team in pairs:
        if out.get(track) is None and team not in used_teams and score > 0:
            out[track] = team
            used_teams.add(team)
    return out


def assign_tracks(con, match_key: str, alliance: str,
                  scorer=None) -> Dict[int, Optional[int]]:
    """Resolve track_id -> team for one alliance of one match, and persist it.

    `scorer(detection_row) -> {team: score}` supplies the per-frame evidence.
    With no scorer this still resolves the degenerate but common case: if the
    alliance has exactly as many tracks as teams and no evidence to separate
    them, it records nothing rather than guessing, because a wrong team number
    in a scouting database is worse than a missing one.
    """
    teams = [r[0] for r in con.execute(
        "SELECT team FROM match_teams WHERE match_key=? AND alliance=? ORDER BY station",
        (match_key, alliance))]
    rows = con.execute("""
        SELECT d.* FROM detections d JOIN frames f ON f.id = d.frame_id
        WHERE f.match_key=? AND d.alliance=? AND d.track_id IS NOT NULL""",
        (match_key, alliance)).fetchall()
    if not teams or not rows:
        return {}

    by_track: Dict[int, List] = defaultdict(list)
    for r in rows:
        by_track[r["track_id"]].append(r)

    if scorer is None:
        return {t: None for t in by_track}

    track_scores = {t: vote(scorer(r) for r in dets) for t, dets in by_track.items()}
    assignment = best_assignment(track_scores, teams)

    for track, team in assignment.items():
        if team is not None:
            con.execute("""UPDATE detections SET team=? WHERE track_id=? AND id IN
                (SELECT d.id FROM detections d JOIN frames f ON f.id=d.frame_id
                 WHERE f.match_key=?)""", (team, track, match_key))
    con.commit()
    return assignment
