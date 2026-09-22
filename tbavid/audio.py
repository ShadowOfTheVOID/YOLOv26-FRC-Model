"""Find matches in a whole-day event stream by listening for the field.

A district event does not publish one video per match. It publishes one
multi-hour stream per day -- FIRST California's weekend events go out as
Webcast Unit YouTube streams, and Central Valley on Twitch -- and TBA's
`match.videos[]` for those events is empty, so the picker in `tba.py` finds
nothing to pull. `download.py` already refuses a stream it is handed, with
TOO_LONG, and says in its own docstring that TBA does sometimes link one.

The video cannot tell us where the matches are. Every stage in `shots.py`
works on what the camera is looking at, and between matches the camera is
looking at the same field from the same place. There is no cut to find.

The audio can. The field control system plays a sound to start a match and a
sound at the buzzer, and in a multi-hour stream those are the only events that
are all four of:

  * loud, against a floor of crowd noise and commentary;
  * TONAL -- a horn or a buzzer concentrates its energy in a few narrow
    frequency bands, which speech, applause and clapping do not;
  * repeated dozens of times across the whole broadcast;
  * separated by a FIXED interval, because a match is a fixed length.

That last one is what makes this work, and it is why nothing here contains a
frequency.

## Why there is no buzzer frequency in this file

The obvious implementation is a band-pass filter at whatever the horn's pitch
is. That number would have to come from somewhere, and the honest options are
measuring it off footage or guessing. A guess that is wrong finds nothing and
looks identical to a stream with no matches in it.

So the cue is found by what it DOES rather than by what it sounds like. Every
loud tonal burst in the stream is collected, the bursts are clustered by their
own spectral shape, and then every ordered pair of clusters (A, B) -- including
A with itself -- is scored by how many bursts of A have a burst of B a
match-length later, at a CONSISTENT interval. The winning pair is the field.

That handles the case a fixed filter would get wrong anyway: the start sound
and the buzzer need not be the same sound. If they are, one cluster pairs with
itself. If they are not, two clusters pair with each other. Neither case needs
to be known in advance, and `cue_report` prints which one happened along with
the interval it measured, so a season whose match length changes is a config
edit rather than a silent failure.

Nothing in here is pitch-specific, so it does not care whether the audio is
the FIRST Webcast Unit's feed, a Twitch re-encode, or a phone recording held
up in the stands.

## What it cannot do

It finds match-shaped intervals. It does not know WHICH match any of them is
-- that is `stream.py`'s job, using TBA's schedule, and it refuses to guess
rather than writing a wrong match key into a scouting database.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ffm import FFMPEG

# 8 kHz mono. A buzzer's fundamental and its first few harmonics are all well
# under 4 kHz, and this keeps a four-hour stream's audio to about 230 MB of
# samples read through a pipe rather than held on disk.
SAMPLE_RATE = 8000
FRAME = 1024            # 128 ms -- long enough to resolve a tone, short
HOP = 512               # 64 ms  -- enough to place its edges
BANDS = 24              # log-spaced bands, the burst's spectral signature

# A burst is loud AND tonal. Both thresholds come from the stream's own
# distribution (median + k*MAD, the idiom shots.find_cuts uses) rather than
# from an absolute level, so a quiet venue and a hot mix both work.
LOUD_SIGMA = 3.0
TONAL_SIGMA = 2.0

# Loudness is measured against a LOCAL baseline, not the whole stream's.
#
# This was a global median first, and on a synthetic broadcast with realistic
# level drift it missed three of eight start cues and found nothing at all in a
# quieter mix. A multi-hour stream does not hold one level: commentary comes and
# goes, the crowd swells between matches, and the mix is ridden by hand. A horn
# that is obvious against the ten seconds around it can sit below the median of
# a four-hour broadcast that also contains a finals crowd.
#
# Thirty seconds, on block medians interpolated back to frames. A block is
# ~470 frames and a burst is ~15, so a burst cannot meaningfully lift the
# median of the block it sits in -- which is the whole reason this is a median
# and not a mean.
BASELINE_WIN_S = 30.0

# A field sound is a short event. Commentary and crowd noise that happen to go
# tonal do not hold a stable peak this long, and a PA announcement holds it far
# longer.
MIN_BURST_S = 0.20
MAX_BURST_S = 4.0
BURST_GAP_S = 0.30      # bridge a burst that dips for one frame

# How close two bursts' spectra must be to be the same sound.
CLUSTER_DIST = 0.22

# A match is a fixed length, so the interval between its two cues is the same
# every time. This is the tolerance on "the same", not on the length itself --
# the length comes from cfg and is reported back measured.
PAIR_TOL_S = 6.0
MIN_PAIRS = 3           # fewer than this is not a pattern, it is a coincidence


class Burst:
    """One loud tonal event: when, how strong, and what it sounded like."""

    __slots__ = ("t0", "t1", "energy", "signature", "peak_hz")

    def __init__(self, t0: float, t1: float, energy: float,
                 signature: np.ndarray, peak_hz: float):
        self.t0 = t0
        self.t1 = t1
        self.energy = energy
        self.signature = signature
        self.peak_hz = peak_hz

    @property
    def mid(self) -> float:
        return (self.t0 + self.t1) / 2.0

    def __repr__(self) -> str:
        return (f"Burst({self.t0:.2f}-{self.t1:.2f}s, "
                f"{self.peak_hz:.0f}Hz, e={self.energy:.3f})")


def _band_edges(n_bins: int) -> List[Tuple[int, int]]:
    """Log-spaced band edges over the rFFT bins.

    Log rather than linear because a horn's harmonics are multiplicative: at
    linear spacing the low bands that carry the fundamental would be one bin
    wide and the signature would be dominated by empty high bands.
    """
    lo, hi = 2, n_bins            # bin 0/1 are DC and near-DC rumble
    edges = np.unique(np.geomspace(lo, hi, BANDS + 1).astype(int))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def read_pcm(path: Path, sample_rate: int = SAMPLE_RATE,
             chunk_s: float = 60.0):
    """Yield mono float32 blocks of a file's audio, via ffmpeg.

    Streamed rather than loaded: this runs over multi-hour broadcasts, and the
    whole point of the feature is not needing the disk for one.

    Yields nothing at all for a file with no audio track, which is a real case
    -- a silent re-upload -- and reads as "no cues found" rather than an error.
    """
    chunk_samples = max(int(sample_rate * chunk_s), FRAME)
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        tail = np.zeros(0, dtype=np.float32)
        while True:
            raw = proc.stdout.read(chunk_samples * 2)
            if not raw:
                break
            # An odd byte count would misalign every sample after it.
            if len(raw) % 2:
                raw = raw[:-1]
            block = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if tail.size:
                block = np.concatenate([tail, block])
            # Keep the last partial frame so a burst straddling a chunk
            # boundary is not cut in half by the read size.
            usable = ((block.size - FRAME) // HOP + 1) * HOP if block.size >= FRAME else 0
            if usable <= 0:
                tail = block
                continue
            yield block[:usable + FRAME - HOP]
            tail = block[usable:]
        if tail.size >= FRAME:
            yield tail
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


def frame_features(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(energy, tonality, bands) per frame of one block.

    `tonality` is peak-to-median ratio across the spectrum, in dB. A pure tone
    puts everything in one bin and scores high; applause and crowd noise are
    broadband and score near zero. This is the measure that separates a buzzer
    from a big cheer, which an energy threshold alone cannot do -- and a cheer
    is exactly what follows a match.
    """
    n = (block.size - FRAME) // HOP + 1
    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, np.zeros((0, 1), dtype=np.float32)

    # One strided view, one FFT call: per-frame Python would dominate runtime
    # over a four-hour stream.
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = block[idx] * np.hanning(FRAME).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)

    energy = spec.sum(axis=1)
    med = np.median(spec, axis=1) + 1e-9
    peak = spec.max(axis=1)
    tonality = 20.0 * np.log10(peak / med)

    edges = _band_edges(spec.shape[1])
    bands = np.stack([spec[:, a:b].sum(axis=1) for a, b in edges], axis=1)
    return energy, tonality, bands


def _runs(mask: np.ndarray, bridge: int) -> List[Tuple[int, int]]:
    """Contiguous True spans, bridging gaps of up to `bridge` frames."""
    on = np.where(mask)[0]
    if on.size == 0:
        return []
    spans = []
    start = prev = int(on[0])
    for i in on[1:]:
        i = int(i)
        if i - prev <= bridge + 1:
            prev = i
            continue
        spans.append((start, prev + 1))
        start = prev = i
    spans.append((start, prev + 1))
    return spans


def global_floor(x: np.ndarray, sigma: float) -> float:
    """median + sigma*MAD over everything, as a scalar."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med + sigma * mad * 1.4826


def local_floor(x: np.ndarray, frame_s: float, win_s: float,
                sigma: float) -> np.ndarray:
    """A per-frame threshold that follows the stream's own drifting level.

    Block medians rather than a true rolling window: a four-hour stream is
    ~225k frames, and a median over a sliding window of 470 of them is a
    quarter of a billion comparisons for a curve that is smooth by construction.
    Medians on non-overlapping blocks, linearly interpolated between block
    centres, give the same curve for a fraction of the work.
    """
    n = x.size
    blk = max(int(round(win_s / frame_s)), 1)
    nb = n // blk
    if nb < 2:
        return np.full(n, global_floor(x, sigma), dtype=np.float32)
    grid = x[:nb * blk].reshape(nb, blk)
    med = np.median(grid, axis=1)
    mad = np.median(np.abs(grid - med[:, None]), axis=1)
    thresh = med + sigma * mad * 1.4826
    centres = (np.arange(nb) + 0.5) * blk
    return np.interp(np.arange(n), centres, thresh).astype(np.float32)


def find_bursts(path: Path, sample_rate: int = SAMPLE_RATE,
                progress=None) -> List[Burst]:
    """Every loud tonal burst in a file, in time order.

    Two passes over the features, not the audio: the thresholds are relative to
    the stream's own medians, and those are not known until the whole stream
    has been seen. The features are ~40 bytes a frame, so a four-hour
    broadcast's worth is a few tens of megabytes held while the audio itself is
    streamed past once.
    """
    energies, tonalities, bands = [], [], []
    for block in read_pcm(path, sample_rate):
        e, t, b = frame_features(block)
        if e.size:
            energies.append(e)
            tonalities.append(t)
            bands.append(b)
            if progress:
                progress(sum(x.size for x in energies) * HOP / sample_rate)
    if not energies:
        return []

    energy = np.concatenate(energies)
    tonality = np.concatenate(tonalities)
    band = np.concatenate(bands, axis=0)

    frame_s = HOP / float(sample_rate)
    # Energy against the local baseline; tonality against the whole stream's,
    # because tonality is already a within-frame ratio (peak over median of the
    # spectrum) and so does not move with the mix the way energy does.
    mask = (energy > local_floor(energy, frame_s, BASELINE_WIN_S, LOUD_SIGMA)) & \
           (tonality > global_floor(tonality, TONAL_SIGMA))

    bridge = max(int(round(BURST_GAP_S / frame_s)), 1)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sample_rate)
    edges = _band_edges(freqs.size)
    band_hz = np.array([freqs[a:b].mean() for a, b in edges])

    out: List[Burst] = []
    for a, b in _runs(mask, bridge):
        dur = (b - a) * frame_s
        if dur < MIN_BURST_S or dur > MAX_BURST_S:
            continue
        rows = band[a:b]
        sig = rows.mean(axis=0)
        total = float(sig.sum())
        if total <= 0:
            continue
        # L1-normalised, so the signature is the SHAPE of the sound and two
        # instances of the same horn at different volumes match each other.
        sig = sig / total
        out.append(Burst(t0=a * frame_s, t1=b * frame_s,
                         energy=float(energy[a:b].max()),
                         signature=sig.astype(np.float32),
                         peak_hz=float(band_hz[int(np.argmax(sig))])))
    return out


def cluster_bursts(bursts: Sequence[Burst],
                   dist: float = CLUSTER_DIST) -> List[int]:
    """Group bursts by what they sound like. Returns a label per burst.

    Greedy against running cluster means, in descending energy order so the
    loudest, cleanest example of each sound defines its cluster rather than
    whichever one happened to come first in the broadcast.
    """
    labels = [-1] * len(bursts)
    centres: List[np.ndarray] = []
    counts: List[int] = []
    for i in sorted(range(len(bursts)), key=lambda k: -bursts[k].energy):
        sig = bursts[i].signature
        best, best_d = -1, dist
        for c, centre in enumerate(centres):
            d = float(np.abs(sig - centre).sum())     # L1 over an L1-normed pair
            if d < best_d:
                best, best_d = c, d
        if best < 0:
            centres.append(sig.copy())
            counts.append(1)
            labels[i] = len(centres) - 1
        else:
            counts[best] += 1
            centres[best] += (sig - centres[best]) / counts[best]
            labels[i] = best
    return labels


def pick_cue_pair(bursts: Sequence[Burst], labels: Sequence[int],
                  match_s: float, window_s: float,
                  tol_s: float = PAIR_TOL_S,
                  min_pairs: int = MIN_PAIRS) -> Optional[Dict]:
    """The interval that repeats like a match, and the burst pairs at it.

    Finds the spacing, not the sound. Every pair of bursts whose separation
    falls in the plausible window votes for its own spacing; the spacing with
    the most votes within `tol_s` is the match length, and the pairs that voted
    for it are the matches.

    ## Why this does not go via the timbre clusters

    It used to. Every ordered pair of clusters (A, B) was scored, on the
    reasoning that the start sound and the buzzer might be one sound or two.
    That works on clean audio and falls apart on a broadcast: run the same
    synthetic event through an AAC encode and one cue's bursts scatter across
    three clusters, because lossy coding rewrites exactly the quiet spectral
    detail an L1 or cosine signature distance is measuring. Measured on a
    planted stream, within-sound distances ran to 0.70 against across-sound
    distances from 0.42 -- no threshold separates those, so no amount of
    tuning CLUSTER_DIST was going to fix it. Worse, two field sounds an octave
    apart share most of their harmonics and are genuinely close in any
    band-energy descriptor.

    Dropping timbre from the decision loses nothing, because the consistency
    requirement was always what did the real work: a crowd can be loud twice,
    but it cannot be loud twice at the same spacing forty times running. It
    also removes the last tunable that had to match a venue's sound.

    The clusters are still computed and still reported -- "two different
    sounds" versus "the same sound both ends" is worth knowing when a detection
    looks wrong -- they just no longer gate anything.

    Returns None when nothing repeats often enough, which is the honest answer
    for a stream that is not an event broadcast.
    """
    if len(bursts) < 2:
        return None
    lo, hi = match_s - window_s, match_s + window_s

    # Candidate pairs, nearest partner per start: two cues bracket one match,
    # so a third burst in the window must not also count.
    deltas: List[Tuple[int, int, float]] = []
    for i, bi in enumerate(bursts):
        best_j, best_gap = -1, None
        for j in range(len(bursts)):
            if j == i:
                continue
            gap = bursts[j].mid - bi.mid
            if not (lo <= gap <= hi):
                continue
            off = abs(gap - match_s)
            if best_gap is None or off < best_gap:
                best_j, best_gap = j, off
        if best_j >= 0:
            deltas.append((i, best_j, bursts[best_j].mid - bi.mid))
    if len(deltas) < min_pairs:
        return None

    # Every candidate spacing is a hypothesis; the one with the most agreement
    # wins. Over the spacings themselves rather than a fixed grid, so the
    # answer is not quantised by a bin width.
    spacings = np.array([d for _, _, d in deltas])
    best = None
    for centre in spacings:
        near = np.abs(spacings - centre) <= tol_s
        if int(near.sum()) < min_pairs:
            continue
        refined = float(np.median(spacings[near]))
        keep = np.abs(spacings - refined) <= tol_s
        score = int(keep.sum())
        if best is None or score > best[0]:
            best = (score, refined, keep)
    if best is None:
        return None

    _, interval, keep = best
    candidates = [deltas[k] for k in range(len(deltas)) if keep[k]]

    # One burst, one role. Matches do not overlap, so a burst cannot be one
    # match's buzzer and the next match's start cue -- and without saying so,
    # an unrelated burst landing the match length before a real start cue gets
    # paired with it. Measured on a planted stream: a decoy 149.0s before a
    # genuine cue produced a ninth "match" among eight, and dragged the
    # interval spread from 0.0s to 1.0s where the spread is the confidence.
    #
    # Best offset first, so the pair that agrees most closely with the measured
    # interval claims its bursts and the coincidence is left over.
    claimed = set()
    kept: List[Tuple[int, int, float]] = []
    for i, j, d in sorted(candidates, key=lambda p: abs(p[2] - interval)):
        if i in claimed or j in claimed:
            continue
        claimed.add(i)
        claimed.add(j)
        kept.append((i, j, d))
    if len(kept) < min_pairs:
        return None
    kept.sort(key=lambda p: bursts[p[0]].mid)
    score = len(kept)
    starts = [labels[i] for i, _, _ in kept]
    ends = [labels[j] for _, j, _ in kept]
    a = max(set(starts), key=starts.count)
    b = max(set(ends), key=ends.count)
    return {"start_cluster": a, "end_cluster": b, "score": score,
            "interval_s": round(interval, 2),
            "spread_s": round(float(np.max(np.abs(
                np.array([d for _, _, d in kept]) - interval))), 2),
            "pairs": kept,
            # Reported, not decided on: see the docstring.
            "same_sound": a == b}


def plan_matches(bursts: Sequence[Burst], labels: Sequence[int],
                 cue: Dict, tol_s: float = PAIR_TOL_S) -> List[Dict]:
    """Every match the audio implies, including the ones missing a cue.

    A match whose start horn happened under a swell of crowd noise loses both
    cues to the pairing, and with them the whole match -- which matters here
    more than it looks, because the entire point of reading a stream is not
    hand-feeding matches one at a time.

    Once the interval is known it is recoverable. It was measured across every
    confirmed pair in this broadcast and held to a fraction of a second, so a
    lone end cue puts its own match's start at exactly interval before it. The
    inference is recorded per match rather than blended in: `basis` says
    whether both cues were heard, and a caller that would rather have only the
    certain ones can filter on it.

    Never invents a match out of nothing. Every entry here is anchored on at
    least one cue that was actually heard.
    """
    interval = float(cue["interval_s"])
    used_start = {i for i, _, _ in cue["pairs"]}
    used_end = {j for _, j, _ in cue["pairs"]}

    plan = [{"start": bursts[i].mid, "end": bursts[j].mid, "basis": "both cues"}
            for i, j, _ in cue["pairs"]]

    for k, lab in enumerate(labels):
        if lab == cue["end_cluster"] and k not in used_end and k not in used_start:
            plan.append({"start": bursts[k].mid - interval, "end": bursts[k].mid,
                         "basis": "start inferred from the buzzer"})
        elif lab == cue["start_cluster"] and k not in used_start and k not in used_end:
            plan.append({"start": bursts[k].mid, "end": bursts[k].mid + interval,
                         "basis": "buzzer inferred from the start cue"})

    plan.sort(key=lambda m: m["start"])
    # An inferred match that lands on a confirmed one is the same play heard
    # twice, and the confirmed reading wins. Confirmed entries sort first
    # within a collision because they are anchored at both ends.
    kept: List[Dict] = []
    for m in sorted(plan, key=lambda m: (m["start"], m["basis"] != "both cues")):
        if any(m["start"] < k["end"] and k["start"] < m["end"] for k in kept):
            continue
        kept.append(m)
    kept.sort(key=lambda m: m["start"])
    return kept


def match_windows(plan: Sequence[Dict], pre_roll_s: float, post_roll_s: float,
                  duration_s: float = 0.0) -> List[Dict]:
    """Pad each planned match out to a clip, non-overlapping, in time order.

    The padding is asymmetric on purpose. `pre_roll` has to cover the gap
    between whatever the start cue actually is -- field ready, or the horn that
    starts autonomous -- and the first ball leaving a robot, because a cue
    identified by its interval says nothing about which of those it is.
    """
    out: List[Dict] = []
    for m in plan:
        # The buzzer has to be in the recording. A stream that stops partway
        # through the last match yields a fragment, and a fragment aligned to a
        # real match key is worse than no clip at all: the scoreboard reader
        # takes its final count off the end of the video, so a 20-second stub
        # would write a confident, wrong final fuel for that match. Measured:
        # a 2000s recording of a match starting at 2000s produced exactly that.
        if duration_s and m["end"] > duration_s:
            continue
        a = max(m["start"] - pre_roll_s, 0.0)
        b = m["end"] + post_roll_s
        if duration_s:
            b = min(b, duration_s)
        if b <= a:
            continue
        if out and a < out[-1]["end"]:
            # Two windows overlapping after padding is the padding's doing, not
            # two readings of one match: trim rather than drop, so a tight
            # turnaround does not silently lose a match.
            a = out[-1]["end"]
            if b <= a:
                continue
        out.append({"start": round(a, 2), "end": round(b, 2),
                    "duration": round(b - a, 2), "basis": m["basis"],
                    "cue_start": round(m["start"], 2),
                    "cue_end": round(m["end"], 2)})
    return out


def cue_report(bursts: Sequence[Burst], labels: Sequence[int],
               cue: Optional[Dict]) -> List[str]:
    """Human-readable lines about what was heard, for the console and the report.

    Prints the measured interval whether or not it agrees with the configured
    match length: a season with a different match length should be a config
    edit somebody makes on purpose, not a silent miss.
    """
    lines = [f"{len(bursts)} loud tonal burst(s), "
             f"{len(set(labels))} distinct sound(s)"]
    if not bursts:
        lines.append("  no audio, or nothing in it stands out from the crowd noise")
        return lines
    tally: Dict[int, int] = {}
    for lab in labels:
        tally[lab] = tally.get(lab, 0) + 1
    for lab, n in sorted(tally.items(), key=lambda kv: -kv[1])[:6]:
        hz = np.median([b.peak_hz for b, l in zip(bursts, labels) if l == lab])
        lines.append(f"  sound {lab}: {n} time(s), peak around {hz:.0f} Hz")
    if not cue:
        lines.append("  no pair of sounds repeats at a consistent match-length "
                     "interval -- this does not look like an event broadcast")
        return lines
    lines.append(
        f"  field cues: sound {cue['start_cluster']} -> sound "
        f"{cue['end_cluster']}"
        + (" (the same sound both ends)" if cue["same_sound"]
           else " (two different sounds)")
        + f", {cue['score']} match(es) at {cue['interval_s']:.1f}s apart "
          f"(+/-{cue['spread_s']:.1f}s)")
    return lines
