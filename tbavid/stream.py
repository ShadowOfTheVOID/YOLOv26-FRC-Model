"""Turn one whole-day event stream into per-match videos.

`tba.pick_unseen` walks TBA looking for `match.videos[]`, which is a per-match
link somebody uploaded. Plenty of events do not have those. A district weekend
publishes one continuous multi-hour broadcast per day and nothing per match,
so the picker finds nothing and `download.py` refuses the stream it is handed
with TOO_LONG.

This is the other way in: point it at the stream, let `audio.py` find the
matches inside it by listening for the field, cut each one out, and hand the
clips to the same pipeline every other video goes through. One command per
event day instead of one upload per match.

    ./run.py stream --url <stream> --event 2026casnf

## The part that refuses to guess

Audio says where the matches are. It cannot say WHICH they are, and a match
key is the single most dangerous thing in this codebase to get wrong: it is
what `db.py` joins a roster onto, so a clip labelled with the wrong key
attributes one alliance's fuel to six robots that were not on the field.
`identify.py` already says the rule this follows -- a wrong team number in a
scouting database is worse than a missing one.

So identity comes from TBA's schedule and only when the evidence actually
supports it:

  * the confirmed-cue count equals the number of matches TBA has for the event
    -> align in order, one to one;
  * it falls short, and the cues recovered from a single heard horn close the
    gap exactly -> align in order, with TBA's count corroborating the
    inference, which is the only thing that makes an inferred match safe;
  * `--from-match qm14` -> the operator has said where the day starts, and
    alignment runs from there;
  * anything else -> NO match keys are assigned.

That last case is not a failure, and this is the part worth understanding
before reading the code: **training frames do not need a match key, scouting
rows do.** A clip nobody can identify is still a cropped, main-camera-only
view of a real field with real robots on it, which is what the detector is
being trained on. It keeps its frames and enters the database under its own
video id -- never a real match key, so no roster joins onto it and no team is
credited with anything. What it loses is scouting attribution, which is
exactly what was not established.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import audio, download, tba
from .config import RAW_DIR
from .ffm import FFMPEG, probe

# Padding around the cues. The start cue might be "field ready" or the horn
# that starts autonomous, and nothing in the audio says which -- so the
# pre-roll has to cover either. The tail covers the scoreboard settling on its
# final count, which the OCR reads after the buzzer.
PRE_ROLL_S = 20.0
POST_ROLL_S = 15.0

# A cut is a stream copy, so it lands on the nearest keyframe rather than the
# exact second asked for. Broadcast keyframes run a couple of seconds apart and
# the padding above absorbs it; re-encoding sixty matches to gain those seconds
# would cost an hour of CPU for accuracy the crop stage does not need.
CUT_ARGS = ["-c", "copy", "-avoid_negative_ts", "make_zero"]


def match_sort_key(m: dict) -> Tuple:
    """Chronological order for an event's matches.

    `actual_time` when TBA has it, because that is when the match was really
    played and it is the only field that orders quals against playoffs
    correctly. Falling back to the schedule shape rather than to nothing: an
    event still in progress has real times for what has happened and none for
    what has not.
    """
    when = m.get("actual_time") or m.get("predicted_time") or m.get("time") or 0
    level = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}.get(m.get("comp_level"), 9)
    return (0 if when else 1, when or 0, level,
            m.get("set_number") or 0, m.get("match_number") or 0)


def event_matches(client, event_key: str, competitive_only: bool = True) -> List[dict]:
    """TBA's matches for one event, in the order they were played."""
    matches = client.event_matches(event_key) or []
    if competitive_only:
        matches = [m for m in matches if tba.is_competitive_match(m)]
    return sorted(matches, key=match_sort_key)


def align(windows: List[dict], matches: List[dict],
          from_match: str = "") -> Dict:
    """Decide which detected window is which TBA match, or decide not to.

    Returns `{"pairs": [(window, match_or_None)], "basis": str, "identified":
    n}`. The basis string is kept on every manifest entry, so a row in the
    scouting database can always be traced back to what established its
    identity -- a link, a count that agreed, or an operator's say-so.
    """
    confirmed = [w for w in windows if w["basis"] == "both cues"]
    inferred = [w for w in windows if w["basis"] != "both cues"]

    if from_match:
        labels = [tba.match_label(m) for m in matches]
        if from_match not in labels:
            raise SystemExit(
                f"--from-match {from_match} is not a match at {matches[0].get('event_key') if matches else '?'}.\n"
                f"  Matches TBA has: {', '.join(labels[:12])}"
                + (" ..." if len(labels) > 12 else ""))
        start = labels.index(from_match)
        run = matches[start:start + len(windows)]
        return {"pairs": list(zip(windows, run + [None] * (len(windows) - len(run)))),
                "basis": f"operator said the day starts at {from_match}",
                "identified": min(len(windows), len(run))}

    if matches and len(confirmed) == len(matches):
        return {"pairs": list(zip(confirmed, matches)),
                "basis": f"{len(confirmed)} confirmed cue pairs matched TBA's "
                         f"{len(matches)} matches exactly",
                "identified": len(confirmed),
                "dropped_inferred": len(inferred)}

    if matches and confirmed and len(confirmed) < len(matches) \
            and len(confirmed) + len(inferred) == len(matches):
        # The only circumstance an inferred match is safe to name. On its own a
        # lone unpaired horn could be a field fault rather than a match whose
        # other cue was drowned out; TBA's count agreeing to the exact match is
        # the corroboration that tells those apart.
        ordered = sorted(windows, key=lambda w: w["start"])
        return {"pairs": list(zip(ordered, matches)),
                "basis": f"{len(confirmed)} confirmed + {len(inferred)} recovered "
                         f"from a single cue closed the gap to TBA's "
                         f"{len(matches)} exactly",
                "identified": len(ordered)}

    ordered = sorted(windows, key=lambda w: w["start"])
    if not matches:
        why = "no event key given, so there was no schedule to align against"
    elif len(ordered) > len(matches):
        why = (f"heard {len(ordered)} match-shaped intervals but TBA has "
               f"{len(matches)} matches, so at least one is not a match")
    else:
        why = (f"heard {len(confirmed)} confirmed and {len(inferred)} recovered "
               f"against TBA's {len(matches)} matches, which does not add up to "
               f"a one-to-one order")
    return {"pairs": [(w, None) for w in ordered],
            "basis": f"unidentified: {why}", "identified": 0}


def cut(raw: Path, start: float, end: float, dest: Path) -> Optional[Path]:
    """Copy one time range out of the stream into its own file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(raw),
         "-t", f"{end - start:.3f}"] + CUT_ARGS + [str(dest)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        return None
    return dest


def download_stream(url: str, dest_dir: Path, cfg: dict
                    ) -> Tuple[Optional[Path], Optional[str], dict]:
    """Fetch a whole stream, with no duration guard.

    `download.download` rejects anything over `max_duration_s`, which is right
    for a picker that is looking for single matches and would otherwise fill a
    disk with a full-day stream somebody linked by accident. Here the full-day
    stream is the point, so that guard is deliberately not applied -- and the
    caller is told the size it is about to take on.
    """
    info, fail = download.probe_remote(_key_of(url), cfg) if _is_bare_key(url) else \
        _probe_url(url, cfg)
    if fail:
        return None, fail, {}
    duration = float((info or {}).get("duration") or 0)
    meta = {"title": (info or {}).get("title", ""), "duration": duration,
            "id": (info or {}).get("id", "")}

    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = meta["id"] or "stream"
    height = cfg["max_height"]
    fmt = (f"bv*[height<={height}][vcodec^=avc1]/bv*[height<={height}]"
           f"/b[height<={height}]/bv*/b") if cfg.get("prefer_h264") else \
          f"bv*[height<={height}]/b[height<={height}]/bv*/b"
    proc = subprocess.run(
        [download.YTDLP, "-f", fmt, "--no-playlist", "--no-warnings", "--no-part",
         "--retries", "5", "--fragment-retries", "5"]
        + download.cookie_args(cfg)
        + ["-o", str(dest_dir / f"{stem}.%(ext)s"), url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    found = sorted(dest_dir.glob(f"{stem}.*"))
    if proc.returncode != 0 or not found:
        for stray in found:
            stray.unlink(missing_ok=True)
        err = (proc.stderr or "").lower()
        return None, (download.L.UNAVAILABLE
                      if any(m in err for m in download.UNAVAILABLE_MARKERS)
                      else download.L.FAILED), meta
    return found[0], None, meta


def _is_bare_key(url: str) -> bool:
    return "/" not in url and ":" not in url and len(url) < 24


def _key_of(url: str) -> str:
    return url


def _probe_url(url: str, cfg: dict) -> Tuple[Optional[dict], Optional[str]]:
    """`yt-dlp -J` against a full URL, so Twitch and a bare YouTube id both work.

    Twitch matters specifically: FIRST California's Central Valley event goes
    out there rather than through the Webcast Unit's YouTube feed, and
    `download.probe_remote` only ever builds a youtube.com/watch URL.
    """
    proc = subprocess.run(
        [download.YTDLP, "-J", "--no-playlist", "--no-warnings"]
        + download.cookie_args(cfg) + [url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").lower()
        if any(m in err for m in download.UNAVAILABLE_MARKERS):
            return None, download.L.UNAVAILABLE
        return None, download.L.FAILED
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, download.L.FAILED


def listen(raw: Path, cfg: dict, duration_s: float = 0.0) -> Dict:
    """Find the matches in a downloaded stream. No network, no TBA.

    Split out from `ingest` so it can be run on its own -- `run.py stream
    --listen-only` prints what it heard without cutting or downloading
    anything, which is how you find out whether a venue's audio works at all
    before spending an hour of CPU on it.
    """
    conf = cfg.get("stream") or {}
    match_s = float(conf.get("match_s", 150.0))
    window_s = float(conf.get("match_window_s", 25.0))

    bursts = audio.find_bursts(raw)
    labels = audio.cluster_bursts(bursts)
    cue = audio.pick_cue_pair(bursts, labels, match_s, window_s)
    report = audio.cue_report(bursts, labels, cue)
    if not cue:
        return {"windows": [], "cue": None, "report": report,
                "bursts": len(bursts)}

    plan = audio.plan_matches(bursts, labels, cue)
    windows = audio.match_windows(
        plan,
        float(conf.get("pre_roll_s", PRE_ROLL_S)),
        float(conf.get("post_roll_s", POST_ROLL_S)),
        duration_s=duration_s,
    )
    # Said out loud whether or not it agrees with the config: a season whose
    # match length changed should be one edit somebody makes on purpose, not a
    # detector that quietly finds nothing.
    if abs(cue["interval_s"] - match_s) > window_s / 2:
        report.append(f"  NOTE: measured {cue['interval_s']:.1f}s between cues, but "
                      f"stream.match_s is {match_s:.0f}s -- if this event's matches "
                      f"really are that length, set it in config.json")
    return {"windows": windows, "cue": cue, "report": report,
            "bursts": len(bursts)}
