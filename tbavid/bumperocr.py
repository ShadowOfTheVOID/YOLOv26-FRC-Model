"""Bumper numbers -> team numbers, and the hub counter's totals -> per-team scores.

Two halves of per-robot scoring that were missing:

1. **The scorer `identify.py` asks for.** That module measured why free OCR
   fails at broadcast scale (a bumper number ~5 px tall returns the empty
   string from tesseract at every setting) and said what to do instead: score
   each crop against only the six teams on the field and decide per robot over
   the whole match, not per frame. `score_crop` is that scorer. It finds the
   alliance-coloured bumper band, takes the white numerals in it, and compares
   them to each candidate number rendered in a bold sans font (bumper numerals
   are required to be plain white or black block digits). When tesseract is
   installed its reading is added as a second opinion, scored against the
   same six candidates by digit similarity, never trusted as free text.
   Crops whose numerals are under `MIN_DIGIT_PX` tall return no evidence at
   all: a guess from 5 px of blur is worse than nothing.

2. **Whose balls the hub counter counted.** The colour hub counter is the
   accurate total (9-10% held-out error per hub, HUB_FEED.md); ShotCounter's
   per-robot attribution saw 19 of 74 balls on qm7. Neither alone is a
   per-robot score. `apportion` takes the hub counter's count per hub as the
   truth and splits it across teams in proportion to the makes ShotCounter
   attributed to each, with the unattributed share kept as its own row so
   nobody is credited with balls nobody saw them shoot.

Unvalidated on real bumpers: the plumbing is tested on rendered bumpers only,
and CLAUDE.md records twice that synthetic tests passed while real frames
failed. Check `run.py shots --ocr --annotate` by eye before trusting a name;
`--teams` still overrides anything read here.
"""
from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .identify import best_assignment

Box = Tuple[float, float, float, float]

# Below this the digits were never sampled well enough to tell apart. Measured
# on rendered, lightly blurred bumpers against six candidates that include
# look-alikes (1678/1671, 254/245): numerals 15 px tall in the white mask
# were right 6 of 6, 11 px wrong 2 of 6 (1671 read as 245, 971 as 8).
#
# Real frames (Einstein 8, 1080p, 2026-10-17): numerals there are 15-70 px,
# not identify.py's 5 px (that was a 1920x504 wide shot), and a person reads
# most of them. This scorer, on 25 held-out robot crops it named: 15 right of
# 23 checkable (65%), the misses mostly red look-alikes (8613/6800/254) and
# 1323/4414 called 4065. Over a 30 s clip the tracker made ~59 robot numbers
# of 6 robots, so no vote was clear and nobody was named. Not ready to name
# robots on its own; the template match is the weak part.
MIN_DIGIT_PX = 14
NORM_H = 32                      # numerals are compared at this height
# The bumper is the bottom of the robot box; the top is mechanism and air.
BUMPER_PART = 0.55
# Votes needed before a robot is named, and how far ahead of the runner-up.
# A wrong team number in scouting data is worse than a missing one.
MIN_VOTE = 1.5
MIN_MARGIN = 0.25
SAMPLE_S = 0.5                   # per robot; OCR every frame buys nothing

_FONTS = None


def _cv2():
    import cv2
    return cv2


def alliance_mask(hsv: np.ndarray, alliance: str) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 90) & (v > 60)
    if alliance == "red":
        return sat & ((h < 10) | (h > 165))
    return sat & (h > 95) & (h < 130)


def white_mask(hsv: np.ndarray) -> np.ndarray:
    return (hsv[..., 1] < 70) & (hsv[..., 2] > 160)


def bumper_band(hsv: np.ndarray, alliance: str) -> Optional[Tuple[int, int]]:
    """Rows of the crop that are mostly bumper colour, as (top, bottom)."""
    rows = alliance_mask(hsv, alliance).mean(axis=1)
    hit = np.flatnonzero(rows > 0.2)
    if hit.size == 0:
        return None
    # The longest run of bumper rows; a red light or ball above it is not it.
    runs, start = [], hit[0]
    for a, b in zip(hit, hit[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, hit[-1]))
    top, bottom = max(runs, key=lambda r: r[1] - r[0])
    # Grow into rows with any bumper colour at all. The numerals are the rows
    # with the LEAST bumper colour -- they are white -- and on Einstein 8 the
    # core run stopped above them: crop 6800's band was rows 28-42 and its
    # digits rows 34-61, so readable bumpers gave no evidence.
    while top > 0 and rows[top - 1] > 0.05:
        top -= 1
    while bottom + 1 < rows.size and rows[bottom + 1] > 0.05:
        bottom += 1
    return int(top), int(bottom) + 1


def numeral_groups(white: np.ndarray, min_h: int) -> List[Tuple[int, int, int, int]]:
    """Digit runs in a binary band: (x, y, w, h), one per printed number.

    Components at least `min_h` tall are digits; neighbours closer than one
    digit height are the same number. A bumper wraps the robot, so a crop can
    hold the number twice -- each run is scored on its own.
    """
    cv2 = _cv2()
    n, _, stats, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    digits = sorted((tuple(stats[i][:4]) for i in range(1, n)
                     if stats[i][3] >= min_h and stats[i][2] <= 6 * stats[i][3]),
                    key=lambda d: d[0])
    groups: List[List] = []
    for d in digits:
        if groups:
            gx, gy, gw, gh = groups[-1]
            if d[0] - (gx + gw) <= max(gh, d[3]):
                x2, y2 = max(gx + gw, d[0] + d[2]), max(gy + gh, d[1] + d[3])
                x1, y1 = min(gx, d[0]), min(gy, d[1])
                groups[-1] = [x1, y1, x2 - x1, y2 - y1]
                continue
        groups.append(list(d))
    return [tuple(g) for g in groups]


def render(text: str, font: int, thick: int) -> np.ndarray:
    """`text` in white on black, cropped to the ink, NORM_H tall."""
    cv2 = _cv2()
    scale = 2.0
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    img = np.zeros((h + base + 8, w + 8), np.uint8)
    cv2.putText(img, text, (4, h + 4), font, scale, 255, thick, cv2.LINE_AA)
    return _norm(img > 127)


def _norm(mask: np.ndarray) -> Optional[np.ndarray]:
    cv2 = _cv2()
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    w = max(1, int(round(m.shape[1] * NORM_H / m.shape[0])))
    return cv2.resize(m, (w, NORM_H), interpolation=cv2.INTER_AREA)


def _fonts():
    global _FONTS
    if _FONTS is None:
        cv2 = _cv2()
        _FONTS = [(cv2.FONT_HERSHEY_SIMPLEX, 6), (cv2.FONT_HERSHEY_DUPLEX, 7),
                  (cv2.FONT_HERSHEY_SIMPLEX, 9)]
    return _FONTS


def similarity(glyphs: np.ndarray, text: str) -> float:
    """Best normalised correlation of a numeral run against `text`, 0..1.

    The template is stretched to the run's width, and the stretch itself costs:
    "1678" squeezed into the width of "254" would otherwise still correlate.
    """
    cv2 = _cv2()
    best = 0.0
    for font, thick in _fonts():
        tpl = render(text, font, thick)
        if tpl is None:
            continue
        stretch = glyphs.shape[1] / tpl.shape[1]
        fit = cv2.resize(tpl, (glyphs.shape[1], NORM_H), interpolation=cv2.INTER_AREA)
        a, b = glyphs - glyphs.mean(), fit - fit.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if den <= 0:
            continue
        corr = float((a * b).sum()) / den
        best = max(best, corr * min(stretch, 1 / stretch))
    return max(0.0, best)


def digit_similarity(read: str, team: int) -> float:
    """How well an OCR reading supports `team`: 1 exact, less per edit."""
    read = "".join(c for c in read if c.isdigit())
    want = str(team)
    if not read:
        return 0.0
    # Levenshtein, tiny strings
    prev = list(range(len(want) + 1))
    for i, c in enumerate(read, 1):
        cur = [i]
        for j, d in enumerate(want, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c != d)))
        prev = cur
    return max(0.0, 1.0 - prev[-1] / len(want))


def tesseract_read(mask: np.ndarray) -> str:
    """Digits tesseract sees in a binary numeral run, or '' (and never raises)."""
    if shutil.which("tesseract") is None:
        return ""
    cv2 = _cv2()
    img = 255 - (mask > 0.5).astype(np.uint8) * 255          # black on white
    img = cv2.copyMakeBorder(cv2.resize(img, None, fx=2, fy=2), 16, 16, 16, 16,
                             cv2.BORDER_CONSTANT, value=255)
    ok, png = cv2.imencode(".png", img)
    if not ok:
        return ""
    try:
        out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "7",
                              "-c", "tessedit_char_whitelist=0123456789"],
                             input=png.tobytes(), capture_output=True, timeout=5)
        return out.stdout.decode("ascii", "ignore").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def score_crop(crop_bgr: np.ndarray, alliance: str, teams: Sequence[int],
               use_tesseract: bool = True) -> Dict[int, float]:
    """Per-candidate evidence from one robot crop, for identify.vote.

    Scores are relative -- each candidate's similarity minus the mean over
    the candidates -- so a blurry crop that matches every number equally adds
    nothing, and only a crop that prefers one number moves the vote.
    """
    cv2 = _cv2()
    if crop_bgr is None or crop_bgr.size == 0 or not teams:
        return {}
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    top = int(hsv.shape[0] * (1 - BUMPER_PART))
    band = bumper_band(hsv[top:], alliance)
    if band is None:
        return {}
    y0, y1 = top + band[0], top + band[1]
    if y1 - y0 < MIN_DIGIT_PX:
        return {}
    white = white_mask(hsv[y0:y1])
    raw: Dict[int, float] = defaultdict(float)
    for x, y, w, h in numeral_groups(white, max(MIN_DIGIT_PX, int(0.3 * (y1 - y0)))):
        glyphs = _norm(white[y:y + h, x:x + w])
        if glyphs is None:
            continue
        reading = tesseract_read(white[y:y + h, x:x + w]) if use_tesseract else ""
        for team in teams:
            s = similarity(glyphs, str(team)) + digit_similarity(reading, team)
            raw[team] = max(raw[team], s)
    if not raw:
        return {}
    mean = sum(raw.values()) / len(teams)
    return {t: s - mean for t, s in raw.items() if s - mean > 0}


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    H, W = img.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(W, int(x + w)), min(H, int(y + h))
    return img[y1:y2, x1:x2]


class TeamVotes:
    """Robot numbers (shooting.RobotNumbers) -> teams, voted over a match.

    `roster` is {"red": [t, t, t], "blue": [...]}: the six on the field, from
    TBA or typed in. Each robot is sampled at most every `sample_s`; the vote
    is summed per robot and resolved per alliance by identify.best_assignment,
    so two robots cannot both be named 254.
    """

    def __init__(self, roster: Dict[str, Sequence[int]], sample_s: float = SAMPLE_S,
                 scorer: Optional[Callable] = None, min_vote: float = MIN_VOTE,
                 min_margin: float = MIN_MARGIN, cropper: Callable = None):
        self.cropper = cropper or crop
        self.roster = {a: [int(t) for t in ts] for a, ts in roster.items()}
        self.sample_s = sample_s
        self.scorer = scorer or score_crop
        self.min_vote, self.min_margin = min_vote, min_margin
        self.votes: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.alliance: Dict[int, str] = {}
        self.last: Dict[int, float] = {}
        self.samples = 0

    def observe(self, img: np.ndarray, robots: Dict[int, Tuple[str, Box]], t: float) -> None:
        for num, (alliance, box) in robots.items():
            if alliance not in self.roster or t - self.last.get(num, -1e9) < self.sample_s:
                continue
            self.last[num] = t
            self.alliance[num] = alliance
            self.samples += 1
            for team, s in self.scorer(self.cropper(img, box), alliance,
                                       self.roster[alliance]).items():
                self.votes[num][team] += s

    def assignments(self) -> Dict[int, int]:
        """Robot number -> team, only where the vote is clear.

        Several robot numbers may get the same team: RobotNumbers misses some
        re-identifications, and a robot that came back under a new number is
        still that team. Within one pass of best_assignment no team repeats, so
        robots are assigned in rounds until no clear vote is left.
        """
        out: Dict[int, int] = {}
        for alliance, teams in self.roster.items():
            left = {n: dict(v) for n, v in self.votes.items()
                    if self.alliance.get(n) == alliance and self._clear(v)}
            while left:
                got = {n: t for n, t in best_assignment(left, teams).items() if t}
                if not got:
                    break
                out.update(got)
                for n in got:
                    del left[n]
        return out

    def _clear(self, v: Dict[int, float]) -> bool:
        ranked = sorted(v.values(), reverse=True) + [0.0, 0.0]
        return ranked[0] >= self.min_vote and ranked[0] - ranked[1] >= self.min_margin * ranked[0]

    def report(self) -> List[str]:
        named = self.assignments()
        lines = [f"bumper OCR: {self.samples} crop(s) read, "
                 f"{len(named)}/{len(self.alliance)} robot number(s) named"]
        for num in sorted(self.alliance):
            v = sorted(self.votes.get(num, {}).items(), key=lambda kv: -kv[1])[:2]
            ev = ", ".join(f"{t} {s:.1f}" for t, s in v) or "no readable bumper"
            lines.append(f"  robot {num} ({self.alliance[num]}): "
                         f"{named.get(num, 'unnamed')}  [{ev}]")
        return lines


def apportion(hub_counts: Dict[str, int], per_robot: Dict, assignments: Dict[int, int],
              unattributed: Optional[Dict[str, int]] = None) -> Dict:
    """Split each hub's counted balls across the teams whose shots went in it.

    `hub_counts` is the hub counter's per-hub total for the match ({"red": 63});
    `per_robot` and `unattributed` are ShotCounter's; `assignments` maps robot
    number -> team (TeamVotes, or --teams). A team's share of its alliance's
    hub is its `made` over every make into that hub -- robots with no name and
    balls with no shooter included, under the key None -- with integer shares
    by largest remainder, so each hub's shares sum exactly to its count.

    Per hub, not by_team: by_team folds every unnamed robot of BOTH alliances
    into one row, and its share of a hub would then be counted twice.
    """
    out: Dict = {}
    for hub, total in hub_counts.items():
        total = int(total)
        if total <= 0:
            continue
        made: Dict = defaultdict(int)
        for num, st in per_robot.items():
            if st.get("alliance") == hub and st.get("made"):
                made[assignments.get(num)] += st["made"]
        if unattributed and unattributed.get(hub):
            made[None] += unattributed[hub]
        seen = sum(made.values())
        blind = seen <= 0
        if blind:              # nobody seen scoring: the whole count is unnamed
            made, seen = {None: 1}, 1
        exact = {t: total * m / seen for t, m in made.items()}
        share = {t: int(v) for t, v in exact.items()}
        spare = total - sum(share.values())
        for t in sorted(exact, key=lambda k: (-(exact[k] - share[k]), str(k)))[:spare]:
            share[t] += 1
        for t, n in share.items():
            key = t if t is not None else f"unnamed {hub}"
            row = out.setdefault(key, {"alliance": hub, "scored": 0, "seen_made": 0})
            row["scored"] += n
            row["seen_made"] += 0 if blind else made[t]
    return out


def parse_roster(text: str) -> Dict[str, List[int]]:
    """'red=254,1678,971;blue=118,148,2056' -> {"red": [...], "blue": [...]}."""
    out: Dict[str, List[int]] = {}
    for part in filter(None, (p.strip() for p in (text or "").split(";"))):
        side, _, teams = part.partition("=")
        side = side.strip().lower()
        if side not in ("red", "blue"):
            raise ValueError(f"--roster: {side!r} is not red or blue (red=254,1678;blue=118)")
        try:
            out[side] = [int(t.strip().lower().replace("frc", ""))
                         for t in teams.split(",") if t.strip()]
        except ValueError:
            raise ValueError(f"--roster: {teams!r} is not a list of team numbers")
    return out


def parse_counts(text: str) -> Dict[str, int]:
    """'red=63,blue=40' -> {"red": 63, "blue": 40}."""
    out: Dict[str, int] = {}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        side, _, n = part.partition("=")
        if side.strip().lower() not in ("red", "blue") or not n.strip().isdigit():
            raise ValueError(f"--hub-counts: {part!r} should look like red=63")
        out[side.strip().lower()] = int(n)
    return out
