"""The Blue Alliance API v3 client.

TBA hosts no video -- match.videos[] is a reference list of
{type: "youtube"|"tba", key}. For "youtube" the key is a YouTube video ID,
which is the only place the actual pixels live. The legacy "tba" type is
effectively dead for modern seasons and is skipped.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import zlib
from typing import Dict, Iterator, List, Optional, Tuple

import requests

from . import formats
from .config import CACHE_DIR

BASE = "https://www.thebluealliance.com/api/v3"

# TBA's event_type codes for real competition: regional, district, district
# championship, championship division, championship final, district
# championship division, Festival of Champions. Everything else -- offseason
# (99), preseason (100) and unlabeled (-1) -- is exhibition play: mixed or
# stand-in rosters, non-standard field setups and demo rules, none of which is
# what a detector trained for competition footage should be learning from.
COMPETITIVE_EVENT_TYPES = frozenset({0, 1, 2, 3, 4, 5, 6})

# Real match play. TBA also carries practice matches ("pm") at some events, and
# those are run with half-built robots and no scoring pressure.
COMPETITIVE_COMP_LEVELS = frozenset({"qm", "ef", "qf", "sf", "f"})


def is_competitive_event(event: dict) -> bool:
    return event.get("event_type") in COMPETITIVE_EVENT_TYPES


def is_competitive_match(match: dict) -> bool:
    return match.get("comp_level") in COMPETITIVE_COMP_LEVELS


class TBAClient:
    def __init__(self, auth_key: str, min_interval_s: float = 0.15):
        self.session = requests.Session()
        self.session.headers.update(
            {"X-TBA-Auth-Key": auth_key, "User-Agent": "TBACroppedOutVid/1.0"}
        )
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -- caching -----------------------------------------------------------
    def _cache_path(self, path: str):
        digest = hashlib.sha256(path.encode()).hexdigest()[:32]
        return CACHE_DIR / f"{digest}.json"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call = time.monotonic()

    def get(self, path: str, tries: int = 4):
        """GET an API path, revalidating any cached copy with If-None-Match."""
        cache_file = self._cache_path(path)
        cached = None
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                cached = None

        headers = {}
        if cached and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]

        delay = 1.0
        for attempt in range(tries):
            self._throttle()
            try:
                resp = self.session.get(BASE + path, headers=headers, timeout=30)
            except requests.RequestException as exc:
                if attempt == tries - 1:
                    raise
                print(f"  ! {path}: {exc}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 304 and cached:
                return cached["body"]
            if resp.status_code == 200:
                body = resp.json()
                cache_file.write_text(
                    json.dumps({"etag": resp.headers.get("ETag"), "body": body})
                )
                return body
            if resp.status_code == 401:
                raise SystemExit(
                    "TBA rejected the API key (401). Check $TBA_AUTH_KEY / .env."
                )
            if resp.status_code == 404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == tries - 1:
                    break
                wait = float(resp.headers.get("Retry-After", delay))
                print(f"  ! {path}: HTTP {resp.status_code}; waiting {wait:.0f}s")
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()

        # Exhausted retries -- fall back to a stale cached copy if we have one.
        if cached:
            print(f"  ! {path}: giving up, using stale cache")
            return cached["body"]
        return None

    # -- endpoints ---------------------------------------------------------
    def events(self, year: int, competitive_only: bool = True) -> List[dict]:
        """Simple event records for a season, official competition by default.

        The whole record rather than the key, because two other decisions are
        made from fields it already carries and would otherwise cost a request
        per event: `event_type` filters the catalogue, and `district` picks the
        broadcast layout profile (see `formats.py`). `/events/{year}/simple`
        is one request for the season either way.
        """
        events = self.get(f"/events/{year}/simple") or []
        if not competitive_only:
            return [e for e in events if e.get("key")]
        return [e for e in events if e.get("key") and is_competitive_event(e)]

    def event_keys(self, year: int, competitive_only: bool = True) -> List[str]:
        """Event keys for a season, official competition only by default.

        /events/{year}/simple rather than /keys: the key alone does not say
        whether an event is an offseason or preseason one, and event_type
        does. It is still a single request per season.
        """
        if not competitive_only:
            return self.get(f"/events/{year}/keys") or []
        events = self.get(f"/events/{year}/simple") or []
        return [e["key"] for e in events if e.get("key") and is_competitive_event(e)]

    def event(self, event_key: str) -> dict:
        """One event record, for its district and type.

        `pick_unseen` gets both off the season list it already fetches, so this
        is only for the stream path, which is handed one event key and never
        walks the catalogue at all.
        """
        return self.get(f"/event/{event_key}") or {}

    def event_matches(self, event_key: str) -> List[dict]:
        return self.get(f"/event/{event_key}/matches") or []


def match_label(match: dict) -> str:
    """'2026casj_qm42' -> a filesystem-safe, human-readable stem."""
    level = match.get("comp_level", "??")
    num = match.get("match_number", 0)
    setn = match.get("set_number", 0)
    if level == "qm":
        return f"{level}{num}"
    return f"{level}{setn}m{num}"


def youtube_candidates(matches: List[dict], competitive_only: bool = True,
                       district: str = "", event_type: Optional[int] = None
                       ) -> Iterator[Dict[str, str]]:
    """Yield one candidate per distinct YouTube video ID in this event.

    `district` is the event's TBA district abbreviation ("ca", "fim", ...) or
    "" for a regional, and `event_type` is TBA's code for what kind of event it
    is. Both ride along on the candidate because the crop stage picks a
    broadcast layout profile from them, and by then the event record they came
    from is several steps out of scope. The type matters as much as the
    district: a district's weekend events and that district's own state
    championship share a district and are not the same broadcast.
    """
    seen_here = set()
    for match in matches:
        if competitive_only and not is_competitive_match(match):
            continue
        for video in match.get("videos") or []:
            if video.get("type") != "youtube":
                continue
            key = (video.get("key") or "").strip()
            if not key or key in seen_here:
                continue
            seen_here.add(key)
            alliances = match.get("alliances") or {}
            yield {
                "yt_key": key,
                "match_key": match.get("key", ""),
                "event_key": match.get("event_key", ""),
                "label": match_label(match),
                "district": district,
                "event_type": event_type,
                # The six teams on the field. This is what turns bumper-number
                # reading from open-ended OCR into a six-way choice.
                "teams": {
                    side: [t.replace("frc", "")
                           for t in (alliances.get(side, {}).get("team_keys") or [])]
                    for side in ("blue", "red")
                },
            }


def shard_of(yt_key: str, shards: int) -> int:
    """Which worker owns this video.

    crc32, not the builtin hash(): Python randomises string hashing per
    process, so hash() would put the same video in a different shard on every
    run and the partition would not hold.
    """
    return zlib.crc32(yt_key.encode()) % shards


def event_catalogue(client, year: int, competitive_only: bool) -> List[dict]:
    """`{key, district, event_type}` per event in a season's catalogue.

    `events()` carries the district and the type; `event_keys()` carries
    neither. Both are accepted so that a caller holding a narrower client -- a
    test fake, or anything that only ever needed keys -- still walks the same
    catalogue, just with nothing to pick a profile from, which reads as the
    generic layout rather than as a guess.

    A dict rather than a tuple because these three travel together from here to
    the crop stage, and two of them are only ever read by name.
    """
    getter = getattr(client, "events", None)
    if callable(getter):
        events = getter(year, competitive_only=competitive_only) or []
        return [{"key": e["key"], "district": formats.district_of(e),
                 "event_type": e.get("event_type")}
                for e in events if e.get("key")]
    return [{"key": k, "district": "", "event_type": None} for k in
            (client.event_keys(year, competitive_only=competitive_only) or [])]


def pick_unseen(
    client: TBAClient,
    year: int,
    count: int,
    ledger,
    retry_failed: bool = False,
    per_event_cap: int = 0,
    rng: Optional[random.Random] = None,
    shard: int = 0,
    shards: int = 1,
    competitive_only: bool = True,
) -> List[dict]:
    """Walk events in random order, collecting `count` never-pulled videos.

    Lazy on purpose: a season has ~1000 events, and fetching every one's
    match list just to sample a handful of videos would be thousands of
    requests. Shuffling the event list first keeps the sample spread across
    venues rather than clustered in whatever events sort first.

    With `competitive_only` (the default) the walk stays inside official
    competition: offseason and preseason events never enter the catalogue, and
    within an event only real match play is considered.
    """
    rng = rng or random.Random()
    catalogue = event_catalogue(client, year, competitive_only)
    if not catalogue:
        where = "competitive events" if competitive_only else "events"
        raise SystemExit(f"TBA returned no {where} for {year}.")
    rng.shuffle(catalogue)

    if shards < 1 or not (0 <= shard < shards):
        raise SystemExit(f"bad shard {shard}/{shards}: need 0 <= id < count")

    picked: List[dict] = []
    skipped = 0
    not_mine = 0
    events_walked = 0

    for event in catalogue:
        if len(picked) >= count:
            break
        event_key = event["key"]
        events_walked += 1
        matches = client.event_matches(event_key)
        if not matches:
            continue

        candidates = list(youtube_candidates(matches,
                                             competitive_only=competitive_only,
                                             district=event["district"],
                                             event_type=event["event_type"]))
        rng.shuffle(candidates)

        from_this_event = 0
        for cand in candidates:
            if len(picked) >= count:
                break
            if per_event_cap and from_this_event >= per_event_cap:
                break
            # Partition by video id so several people can harvest at once
            # with no shared state and no chance of two of them fetching the
            # same match. Every worker walks the same catalogue and skips
            # whatever is not theirs.
            if shards > 1 and shard_of(cand["yt_key"], shards) != shard:
                not_mine += 1
                continue
            if ledger.seen(cand["yt_key"], retry_failed=retry_failed):
                skipped += 1
                continue
            picked.append(cand)
            from_this_event += 1

    extra = f", {not_mine} owned by other shards" if shards > 1 else ""
    kind = "competitive " if competitive_only else ""
    print(f"  walked {events_walked} {kind}events, skipped {skipped} already-seen"
          f"{extra}, picked {len(picked)}")
    return picked
