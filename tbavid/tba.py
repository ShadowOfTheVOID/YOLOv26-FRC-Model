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
from typing import Dict, Iterator, List, Optional

import requests

from .config import CACHE_DIR

BASE = "https://www.thebluealliance.com/api/v3"


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
    def event_keys(self, year: int) -> List[str]:
        return self.get(f"/events/{year}/keys") or []

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


def youtube_candidates(matches: List[dict]) -> Iterator[Dict[str, str]]:
    """Yield one candidate per distinct YouTube video ID in this event."""
    seen_here = set()
    for match in matches:
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
) -> List[dict]:
    """Walk events in random order, collecting `count` never-pulled videos.

    Lazy on purpose: a season has ~1000 events, and fetching every one's
    match list just to sample a handful of videos would be thousands of
    requests. Shuffling the event list first keeps the sample spread across
    venues rather than clustered in whatever events sort first.
    """
    rng = rng or random.Random()
    keys = client.event_keys(year)
    if not keys:
        raise SystemExit(f"TBA returned no events for {year}.")
    rng.shuffle(keys)

    if shards < 1 or not (0 <= shard < shards):
        raise SystemExit(f"bad shard {shard}/{shards}: need 0 <= id < count")

    picked: List[dict] = []
    skipped = 0
    not_mine = 0
    events_walked = 0

    for event_key in keys:
        if len(picked) >= count:
            break
        events_walked += 1
        matches = client.event_matches(event_key)
        if not matches:
            continue

        candidates = list(youtube_candidates(matches))
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
    print(f"  walked {events_walked} events, skipped {skipped} already-seen"
          f"{extra}, picked {len(picked)}")
    return picked
