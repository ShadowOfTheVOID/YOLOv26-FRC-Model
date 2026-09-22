#!/usr/bin/env python3
"""Regression suite. No network, no ffmpeg, no video files.

Every check here exists because something broke. The comments say what, so a
future change that reintroduces it fails loudly instead of producing a
plausible-looking wrong number -- which is how all of these got through the
first time.

    .venv/bin/python tests/test_pipeline.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tbavid import audio as A
from tbavid import formats as F
from tbavid import ledger as L
from tbavid import stream as S
from tbavid.crop import Profile, detect_bottom, detect_top
from tbavid.db import build, connect, summary
from tbavid.identify import best_assignment, vote
from tbavid.labels import Timeline, source_time
from tbavid.ledger import Ledger
from tbavid.render import build_filter
from tbavid.scoreboard import (banner_floor_px, clean_series, parse_pair,
                               plausible_step, probe_rows, tighten)
from tbavid.shots import (_apply_min_shot, build_shots, cluster_shots, find_cuts,
                          merge_ranges)
from tbavid.tba import (TBAClient, event_catalogue, is_competitive_event,
                        match_label, pick_unseen, youtube_candidates)

PASSED = 0
FAILED: list = []


def check(name: str, cond: bool) -> None:
    global PASSED
    if cond:
        PASSED += 1
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}")


H = 180


def prof(med, p25=None, edge=None, ref=0.005) -> Profile:
    med = np.asarray(med, float)
    return Profile(med,
                   np.asarray(med if p25 is None else p25, float),
                   np.zeros(H) if edge is None else np.asarray(edge, float),
                   ref)


def test_cuts():
    rng = np.random.default_rng(0)
    strip = np.concatenate([rng.integers(40, 60, (60, 36, 64, 3), dtype=np.uint8),
                            rng.integers(180, 200, (40, 36, 64, 3), dtype=np.uint8),
                            rng.integers(40, 60, (60, 36, 64, 3), dtype=np.uint8)])
    cuts = find_cuts(strip, 6.0, 0.045)
    check("finds both cuts",
          len(cuts) == 2 and abs(cuts[0] - 6.0) < 0.3 and abs(cuts[1] - 10.0) < 0.3)
    check("static video yields no cuts",
          find_cuts(rng.integers(40, 60, (100, 36, 64, 3), dtype=np.uint8),
                    6.0, 0.045) == [])
    check("shots span the duration",
          build_shots([3.0, 7.0], 10.0) == [(0.0, 3.0), (3.0, 7.0), (7.0, 10.0)])


def test_clustering():
    rng = np.random.default_rng(1)
    main = rng.random(6912)
    sigs = [main + rng.normal(0, 0.005, 6912) for _ in range(5)] + [main + 0.4, main + 0.45]
    a = cluster_shots(sigs, [40, 30, 25, 20, 15, 4, 3], 0.12, 0.20)
    check("main camera collapses to one cluster", len(set(a[:5])) == 1 and a[5] != a[0])
    check("adjacent ranges merge", merge_ranges([(0, 5), (5, 9), (12, 20)]) == [(0, 9), (12, 20)])

    # min_shot_s must apply to the merged run: applying it per shot let an
    # over-segmented main-camera stretch be discarded piece by piece.
    sh = [{"duration": 0.6, "is_main": True} for _ in range(5)]
    _apply_min_shot(sh, 1.5)
    check("fragmented main run survives min_shot", all(s["keep"] for s in sh))
    sh = [{"duration": 0.6, "is_main": True}, {"duration": 5, "is_main": False}]
    _apply_min_shot(sh, 1.5)
    check("isolated short blip is dropped", not sh[0]["keep"])


def test_crop_bands():
    med = np.full(H, 0.005); med[:28] = 1e-5
    p = prof(med)
    check("top band from the median", abs(detect_top(p) - 31 / 180) < 1e-9)
    # A district banner put the counters on the same row as the match clock, so
    # the static walk stopped above them: the crop sliced the digits and the
    # OCR saw nothing. The scoreboard now floors the crop depth.
    check("scoreboard floors a too-shallow banner",
          abs(detect_top(p, floor_px=300, height=1080) - 50 / 180) < 1e-9)
    runaway = prof(np.full(H, 1e-6))
    check("runaway walk is refused", detect_top(runaway) is None)
    check("runaway still honours the floor",
          abs(detect_top(runaway, floor_px=216, height=1080) - 36 / 180) < 1e-9)

    # A burned-in caption across the divider lifts its variance out of the
    # static range; a whole side-camera panel survived the crop because of it.
    # The edge signal survives the caption because the caption is not full width.
    noisy = np.full(H, 0.002)
    edge = np.full(H, 8.0); edge[117] = 120.0
    check("divider found by its edge",
          abs(detect_bottom(prof(noisy, noisy, edge)) - (180 - 114) / 180) < 1e-9)
    check("no divider without a hard edge",
          detect_bottom(prof(noisy, noisy, np.full(H, 8.0))) == 0.0)
    edge2 = np.full(H, 8.0); edge2[117] = 120.0; edge2[150] = 200.0
    check("topmost edge wins over the strongest",
          abs(detect_bottom(prof(noisy, noisy, edge2)) - (180 - 114) / 180) < 1e-9)
    # Dark, barely-moving bleachers inside the main panel are not a divider.
    quiet = np.full(H, 0.005); quiet[41:52] = 3e-4
    check("dark bleachers are not a divider", detect_bottom(prof(quiet, quiet)) == 0.0)


def test_formats():
    # A single-camera layout is selected by the district TBA already told us
    # about, not by a list of event keys somebody has to maintain. 2026casnf is
    # a real one: CA District San Francisco Event.
    fmt, why = F.select(district="ca", event_key="2026casnf", event_type=1)
    check("the California district feed selects its own profile",
          fmt.name == "ca_district" and "district ca" in why)

    # FIRST California's own state championship (2026cancmp) carries district
    # "ca" and is a different, larger production. Applying the weekend events'
    # no-split veto to it would keep a whole side-camera panel in the training
    # set if it does run one - the failure this file exists to prevent, pointed
    # the other way. It falls through to the profile with no opinion.
    check("a district's state championship is not its weekend feed",
          F.select(district="ca", event_key="2026cancmp", event_type=2)[0].name
          == "generic")
    # And a type we could not read is not a type that passed: an entry
    # harvested before the type was recorded must not inherit a veto.
    check("an unknown event type fails the gate rather than waiving it",
          F.select(district="ca", event_key="2026casnf")[0].name == "generic")
    check("a regional with no district falls through to generic",
          F.select(district="", event_key="2026roebling")[0].name == "generic")
    check("a Championship division is still the split-screen profile",
          F.select(district="", event_key="2026gal")[0].name == "champs_split")
    # Config outranks TBA: both routes are a human saying they have looked at
    # the footage. Per-event beats the whole run.
    cfg = {"crop": {"format": "champs_split", "formats": {"2026casnf": "generic"}}}
    check("crop.format forces a profile", F.select(district="ca", cfg=cfg)[0].name
          == "champs_split")
    check("crop.formats[event] beats crop.format",
          F.select(district="ca", event_key="2026casnf", cfg=cfg)[0].name == "generic")
    # Config is a human saying they looked at the footage, so it is not subject
    # to the type gate - that gate exists to stop the pipeline guessing, not to
    # overrule somebody who has measured a state championship.
    check("config outranks the event-type gate too",
          F.select(district="ca", event_key="2026cancmp", event_type=2,
                   cfg={"crop": {"format": "ca_district"}})[0].name == "ca_district")

    # TBA writes `district: null` for a regional, so the same read is also the
    # regional test. A string where an object belongs must not become a name.
    check("district comes off a TBA event record",
          F.district_of({"district": {"abbreviation": "CA"}}) == "ca")
    check("a regional reads as no district",
          F.district_of({"district": None}) == "" and F.district_of({}) == "")
    check("a district that is not an object is not a district",
          F.district_of({"district": "ca"}) == "")

    # This is the whole point of the profile. On a feed with no second-camera
    # panel, the strongest static edge in the lower half is the guardrail or the
    # front of the bleachers -- and taking it for a divider crops away the
    # bottom of the field while the frames still look plausible.
    ca = F.BY_NAME["ca_district"]
    top, bottom, notes = F.reconcile(ca, 0.16, 0.34)
    check("a divider on a single-camera layout is refused", bottom == 0.0)
    check("the refusal says so in the manifest",
          any("no second-camera panel" in n for n in notes))
    # ...and the same measurement on the layout that does split is kept.
    check("the split-screen profile keeps its divider",
          F.reconcile(F.BY_NAME["champs_split"], 0.16, 0.34)[1] == 0.34)

    # A measurement outside the expected range is still the measurement. The
    # pixels are what is being cropped; a profile is a description of a layout
    # that may have been re-cut between events.
    top, _, notes = F.reconcile(ca, 0.31, None)
    check("an out-of-range banner is kept, not clamped", top == 0.31)
    check("but the disagreement is recorded",
          any("outside the expected" in n for n in notes))

    # Inconclusive detection is where a profile supplies a number, and the
    # point of having one per layout instead of one global crop.top.
    check("an inconclusive banner falls back to the profile",
          F.reconcile(ca, None, None)[0] == ca.banner[2])

    # generic has no opinion, so it cannot veto: it must behave exactly like
    # the pre-profile pipeline on a feed nobody has characterised.
    check("generic takes a divider at face value",
          F.reconcile(F.BY_NAME["generic"], 0.16, 0.34)[1] == 0.34)

    # A typo in config.json must stop the run rather than silently harvest a
    # whole batch against the wrong layout.
    for bad in ({"crop": {"format": "ca-district"}},
                {"crop": {"formats": {"2026casnf": "nope"}}}):
        try:
            F.select(district="ca", event_key="2026casnf", cfg=bad)
            check("an unknown format name is refused", False)
        except SystemExit:
            check("an unknown format name is refused", True)


def test_format_tuning():
    # `split_mode: unlikely` is what a single-camera profile asks the detector
    # for: an edge on its own is not enough, because on a feed with no divider
    # there is always an edge down there. Either the static route agrees on the
    # row, or the edge is far past the threshold rather than just over it.
    noisy = np.full(H, 0.002)
    edge = np.full(H, 8.0); edge[117] = 120.0
    unlikely = F.BY_NAME["ca_district"].tuning()
    check("the profile asks for corroboration",
          unlikely["split_mode"] == "unlikely")
    check("a lone edge is not a divider under the veto",
          detect_bottom(prof(noisy, noisy, edge), tune=unlikely) == 0.0)
    check("...and is one without it",
          abs(detect_bottom(prof(noisy, noisy, edge)) - (180 - 114) / 180) < 1e-9)

    # A real divider is static as well as sharp, so both routes see it and the
    # veto lets it through -- a profile must not be able to crop a feed wrong
    # in the other direction either.
    static = np.full(H, 0.002); static[117:] = 1e-6
    check("two agreeing routes survive the veto",
          detect_bottom(prof(static, static, edge), tune=unlikely) > 0)

    # An overwhelming edge is believed on its own. 120 against a threshold of
    # 90 is not; 300 is.
    huge = np.full(H, 8.0); huge[117] = 300.0
    check("an overwhelming edge needs no corroboration",
          detect_bottom(prof(noisy, noisy, huge), tune=unlikely) > 0)

    # The banner walk reads the same thresholds out of the profile.
    med = np.full(H, 0.005); med[:28] = 1e-5
    check("banner detection is unchanged by a profile that does not retune it",
          detect_top(prof(med), tune=unlikely) == detect_top(prof(med)))
    check("a profile can lower the banner ceiling",
          detect_top(prof(med), tune={"max_banner_frac": 0.10}) is None)


def test_district_catalogue():
    # The district has to reach the crop stage, and the only place it is free
    # is the event list we already fetch to pick videos: /events/{year}/simple
    # carries `district`, so this costs no extra request.
    events = [{"key": "2026casnf", "event_type": 1,
               "district": {"abbreviation": "ca"}},
              {"key": "2026cancmp", "event_type": 2,
               "district": {"abbreviation": "ca"}},
              {"key": "2026roebling", "event_type": 0, "district": None}]
    c = TBAClient.__new__(TBAClient)
    c.get = lambda path: events
    # The type travels with the district, because on its own the district
    # cannot tell a weekend event from that district's state championship.
    check("the catalogue carries each event's district and type",
          event_catalogue(c, 2026, True) == [
              {"key": "2026casnf", "district": "ca", "event_type": 1},
              {"key": "2026cancmp", "district": "ca", "event_type": 2},
              {"key": "2026roebling", "district": "", "event_type": 0}])

    # A client that only has event_keys() -- anything written before the
    # district mattered -- must still walk the same catalogue.
    class KeysOnly:
        def event_keys(self, year, competitive_only=True):
            return ["2026casnf"]
    check("a keys-only client still walks, with nothing to select on",
          event_catalogue(KeysOnly(), 2026, True) ==
          [{"key": "2026casnf", "district": "", "event_type": None}])

    matches = [{"key": "2026casnf_qm1", "event_key": "2026casnf", "comp_level": "qm",
                "match_number": 1, "set_number": 0,
                "videos": [{"type": "youtube", "key": "V"}]}]
    got = list(youtube_candidates(matches, district="ca", event_type=1))
    check("and both ride along on the candidate",
          got[0]["district"] == "ca" and got[0]["event_type"] == 1)
    bare = list(youtube_candidates(matches))[0]
    check("a candidate with neither says so, rather than guessing",
          bare["district"] == "" and bare["event_type"] is None)

    # The whole chain in one go, because each link was added separately and the
    # profile is worth nothing if any one of them drops the type on the floor -
    # a silent fall back to `generic` is exactly the failure this is guarding.
    class Season:
        """events() and event_matches() as separate answers, unlike `c` above."""

        def events(self, year, competitive_only=True):
            return events

        def event_matches(self, ek):
            return [{"key": f"{ek}_qm1", "event_key": ek, "comp_level": "qm",
                     "match_number": 1, "set_number": 0,
                     "videos": [{"type": "youtube", "key": f"V{ek}"}]}]

    picked = pick_unseen(Season(), 2026, 9,
                         L.Ledger(Path(tempfile.mkdtemp()) / "l.json"),
                         rng=random.Random(0))
    check("the walk reaches every event in the catalogue", len(picked) == 3)
    got = {p["event_key"]: F.select(district=p["district"],
                                    event_key=p["event_key"],
                                    event_type=p["event_type"])[0].name
           for p in picked}
    check("and each one arrives at the crop stage with the right profile",
          got == {"2026casnf": "ca_district", "2026cancmp": "generic",
                  "2026roebling": "generic"})


def burst(t, hz=440.0, energy=1.0, sig=None):
    """A Burst at a time, without needing audio to make one."""
    v = np.zeros(A.BANDS, dtype=np.float32)
    v[int(hz) % A.BANDS] = 1.0
    return A.Burst(t0=t - 0.4, t1=t + 0.4, energy=energy,
                   signature=(sig if sig is not None else v), peak_hz=hz)


def test_audio_frames():
    # The measure that matters is tonality, not loudness. A cheer is the
    # loudest thing at an event after the buzzer, and an energy threshold on
    # its own cannot tell the two apart - which is the whole reason a
    # peak-to-median ratio is computed per frame.
    sr = A.SAMPLE_RATE
    t = np.arange(sr * 2) / sr
    tone = (0.5 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
    rng = np.random.default_rng(3)
    cheer = rng.normal(0, 0.5, t.size).astype(np.float32)
    _, tone_ton, _ = A.frame_features(tone)
    e_cheer, cheer_ton, _ = A.frame_features(cheer)
    check("a tone reads as tonal", float(np.median(tone_ton)) > 25.0)
    check("a cheer just as loud does not",
          float(np.median(cheer_ton)) < 15.0)
    check("and it is not quieter, so loudness alone could not separate them",
          float(np.median(e_cheer)) > 0)

    # A four-hour broadcast does not hold one level. Measured: a global median
    # missed three of eight start cues and found nothing in a quieter mix,
    # because a horn obvious against its own ten seconds can sit under the
    # median of a stream that also contains a finals crowd.
    frame_s = A.HOP / sr
    quiet = np.full(4000, 0.05); loud = np.full(4000, 1.0)
    drift = np.concatenate([quiet, loud]).astype(np.float32)
    drift[2000] = 0.30          # a cue in the quiet half
    local = A.local_floor(drift, frame_s, 8.0, 3.0)
    check("a local baseline catches a cue in the quiet half",
          drift[2000] > local[2000])
    check("...where one number for the whole stream would not",
          drift[2000] < A.global_floor(drift, 3.0))


def test_audio_cues():
    # Eight matches 150s long on a 380s cycle, plus decoys: a burst 150s from
    # nothing in particular, and a run of noise bursts at a DIFFERENT spacing.
    bursts = []
    for m in range(8):
        t0 = 100.0 + m * 380.0
        bursts.append(burst(t0, hz=880.0))
        bursts.append(burst(t0 + 150.0, hz=440.0))
    for k in range(5):
        bursts.append(burst(40.0 + k * 97.0, hz=1500.0))
    bursts.sort(key=lambda b: b.mid)
    labels = A.cluster_bursts(bursts)
    cue = A.pick_cue_pair(bursts, labels, match_s=150.0, window_s=25.0)
    check("the repeated interval is found", cue is not None)
    check("and it is the match length, not the decoys' spacing",
          abs(cue["interval_s"] - 150.0) < 0.5)
    check("every match is paired", cue["score"] == 8)
    check("a fixed interval reads as fixed", cue["spread_s"] < 0.5)

    # Timbre must not gate the decision. Lossy coding scatters one sound across
    # several clusters - measured on an AAC encode of a planted stream, where
    # within-sound signature distances ran to 0.70 against across-sound
    # distances from 0.42, so no threshold separates them. Give every burst a
    # different signature and the interval must still be found.
    rng = np.random.default_rng(5)
    scattered = [A.Burst(b.t0, b.t1, b.energy,
                         rng.random(A.BANDS).astype(np.float32), b.peak_hz)
                 for b in bursts]
    cue2 = A.pick_cue_pair(scattered, A.cluster_bursts(scattered), 150.0, 25.0)
    check("the interval survives signatures that cluster wrongly",
          cue2 is not None and cue2["score"] == 8)

    # A stream that is not an event broadcast must come back with nothing
    # rather than with a plausible-looking answer.
    random_bursts = [burst(x) for x in
                     np.cumsum(np.random.default_rng(9).uniform(20, 400, 12))]
    check("nothing repeating reads as no field",
          A.pick_cue_pair(random_bursts, A.cluster_bursts(random_bursts),
                          150.0, 25.0) is None)
    check("one burst cannot be a match", A.pick_cue_pair([burst(10.0)], [0],
                                                         150.0, 25.0) is None)


def test_audio_recovery():
    # A start horn under a crowd swell loses its match entirely, and that
    # matters more than it looks: reading a stream exists so matches are not
    # fed in one at a time. The interval is measured to a fraction of a second
    # across the whole broadcast, so one heard cue places the other.
    bursts = []
    for m in range(6):
        t0 = 100.0 + m * 380.0
        if m != 3:
            bursts.append(burst(t0, hz=880.0))
        bursts.append(burst(t0 + 150.0, hz=440.0))
    bursts.sort(key=lambda b: b.mid)
    labels = A.cluster_bursts(bursts)
    cue = A.pick_cue_pair(bursts, labels, 150.0, 25.0)
    check("the five intact matches pair", cue["score"] == 5)
    plan = A.plan_matches(bursts, labels, cue)
    check("and the sixth is recovered from its buzzer alone", len(plan) == 6)
    got = [m for m in plan if m["basis"] != "both cues"]
    check("the recovery is labelled, not blended in", len(got) == 1)
    check("and it lands where the missing match was",
          abs(got[0]["start"] - (100.0 + 3 * 380.0)) < 1.0)

    # Every match in the plan, recovered or not, is anchored on a cue that was
    # actually heard -- nothing is placed purely by extrapolating the cadence.
    heard_at = [b.mid for b in bursts]
    check("every match is anchored on a burst that was really there",
          all(any(abs(m["start"] - h) < 1.0 or abs(m["end"] - h) < 1.0
                  for h in heard_at) for m in plan))

    # An inferred match landing on a confirmed one is the same play heard
    # twice, and the confirmed reading has to win.
    doubled = list(bursts) + [burst(100.0 + 150.0 + 0.5, hz=440.0)]
    doubled.sort(key=lambda b: b.mid)
    dl = A.cluster_bursts(doubled)
    dp = A.plan_matches(doubled, dl, A.pick_cue_pair(doubled, dl, 150.0, 25.0))
    check("a duplicate reading does not become a second match", len(dp) == 6)

    wins = A.match_windows(plan, pre_roll_s=20.0, post_roll_s=15.0,
                           duration_s=2400.0)
    check("padding widens each clip", all(w["duration"] > 150.0 for w in wins))
    check("and never runs past the end of the stream",
          all(w["end"] <= 2400.0 for w in wins))
    # A stream that stops partway through the last match yields a fragment, and
    # the scoreboard reader takes its final count off the end of the video - so
    # a stub would write a confident wrong final fuel for a real match key.
    cut_short = A.match_windows(plan, 20.0, 15.0, duration_s=2100.0)
    check("a match the recording cuts off is dropped, not shipped as a stub",
          len(cut_short) == len(wins) - 1
          and all(w["duration"] > 150.0 for w in cut_short))
    check("windows come out in time order",
          [w["start"] for w in wins] == sorted(w["start"] for w in wins))
    check("and never overlap",
          all(a["end"] <= b["start"] for a, b in zip(wins, wins[1:])))


def test_stream_alignment():
    """Who each clip is, or the refusal to say.

    A wrong match key is the most damaging thing this pipeline can produce: it
    is what db.py joins a roster onto, so one mislabelled clip credits an
    alliance's fuel to six robots that were not on the field.
    """
    def win(t, basis="both cues"):
        return {"start": t, "end": t + 180.0, "duration": 180.0, "basis": basis,
                "cue_start": t + 20.0, "cue_end": t + 170.0}

    def tba_matches(n):
        return [{"key": f"2026casnf_qm{i}", "event_key": "2026casnf",
                 "comp_level": "qm", "match_number": i, "set_number": 0,
                 "actual_time": 1000 + i * 400} for i in range(1, n + 1)]

    four = [win(t) for t in (0.0, 400.0, 800.0, 1200.0)]
    plan = S.align(four, tba_matches(4))
    check("a count that matches TBA exactly aligns in order",
          plan["identified"] == 4
          and [m["key"] for _, m in plan["pairs"]]
          == ["2026casnf_qm1", "2026casnf_qm2", "2026casnf_qm3", "2026casnf_qm4"])

    # An unpaired horn can be a field fault rather than a match whose other cue
    # was drowned out, and on its own nothing tells those apart. When the
    # confirmed count already equals TBA's, the inferred one is the fault.
    plan = S.align(four + [win(1600.0, "buzzer inferred from the start cue")],
                   tba_matches(4))
    check("an extra inferred match is dropped when the count is already right",
          plan["identified"] == 4 and plan.get("dropped_inferred") == 1)

    # ...and when it closes the gap to TBA's count exactly, that agreement is
    # the corroboration that makes naming it safe.
    plan = S.align([win(0.0), win(400.0), win(800.0),
                    win(1200.0, "start inferred from the buzzer")],
                   tba_matches(4))
    check("an inferred match that closes the gap is named",
          plan["identified"] == 4 and "closed the gap" in plan["basis"])

    # More intervals than TBA has matches means at least one is not a match,
    # and there is no way to tell which. Nothing gets a key.
    plan = S.align(four + [win(1600.0), win(2000.0)], tba_matches(4))
    check("too many intervals names nothing",
          plan["identified"] == 0 and all(m is None for _, m in plan["pairs"]))
    check("and says why", "not a match" in plan["basis"])

    # Short of TBA's count with no inference to close it: the day is partial
    # and which matches these are is unknown.
    plan = S.align(four, tba_matches(9))
    check("a partial day names nothing without being told where it starts",
          plan["identified"] == 0)

    # The operator saying so is evidence. Anyone who knows the day starts at
    # qm6 knows something the audio cannot.
    plan = S.align(four, tba_matches(9), from_match="qm6")
    check("--from-match aligns from there",
          [m["key"] for _, m in plan["pairs"]]
          == [f"2026casnf_qm{i}" for i in (6, 7, 8, 9)])
    check("and records that a person said so", "operator" in plan["basis"])
    try:
        S.align(four, tba_matches(9), from_match="qm99")
        check("a match label that does not exist is refused", False)
    except SystemExit:
        check("a match label that does not exist is refused", True)

    # No event key at all is the --event-less run: clips still harvest, and
    # they must not acquire keys from nowhere.
    plan = S.align(four, [])
    check("with no schedule nothing is identified", plan["identified"] == 0)

    # Chronological order, because actual_time is the only field that puts
    # quals and playoffs in the order they were really played.
    ms = [{"comp_level": "f", "match_number": 1, "set_number": 1, "actual_time": 900},
          {"comp_level": "qm", "match_number": 2, "set_number": 0, "actual_time": 200},
          {"comp_level": "qm", "match_number": 1, "set_number": 0, "actual_time": 100}]
    check("matches sort by when they were played",
          [m["match_number"] for m in sorted(ms, key=S.match_sort_key)] == [1, 2, 1])
    # An event still in progress has real times for what happened and none for
    # what has not, and the unplayed must not sort in front of the played.
    ms2 = [{"comp_level": "qm", "match_number": 9, "set_number": 0},
           {"comp_level": "qm", "match_number": 1, "set_number": 0, "actual_time": 100}]
    check("matches with no time yet sort last",
          [m["match_number"] for m in sorted(ms2, key=S.match_sort_key)] == [1, 9])


def test_scoreboard():
    check("stray leading slash tolerated", parse_pair("/129/360") == (129, 360))
    check("bare integer has no denominator", parse_pair("173") == (173, None))
    check("OCR blowup rejected", not plausible_step(712, 7317))
    check("real volley allowed", plausible_step(2, 60))
    check("single-frame spike filtered out",
          clean_series([700, 701, 7317, 7317, 702], [0, 1, 2, 3, 4])
          == [(0, 700), (1, 701), (4, 702)])
    # A blowup must not be able to seed the series: anchoring on the first read
    # let one bad value sit above every genuine one and reject them all.
    check("blowup cannot seed the series",
          clean_series([9999, 3, 3, 4], [0, 1, 2, 3]) == [(1, 3), (3, 4)])
    check("empty input is safe", clean_series([None, None], [0, 1]) == [])
    # An event with a /100 denominator: OCR that drops a small numerator leaves
    # "/100", and reading that as the count locked the monotonic filter above
    # every real value for the rest of the match (counter said 100, official 19).
    check("ambiguous single number beside a slash is discarded",
          parse_pair("/100") is None and parse_pair("18/100") == (18, 100))

    check("banner floor from counter boxes",
          banner_floor_px({"counters": {"blue": [10, 181, 200, 46],
                                        "red": [1500, 181, 180, 43]}}) == 239)
    check("no counters means no floor", banner_floor_px({"error": "x"}) == 0)

    # Widening the search band changed vertical resolution and shrank a counter
    # box from 28px to 25px, clipping digits and turning 726 into 1722.
    check("probe resolution is scale-invariant",
          abs(204 / probe_rows(204) - 324 / probe_rows(324)) < 0.01)

    # Dilation joins the digits of one number but also reached ~14px sideways
    # and swallowed the static alliance icon, which OCR'd as a leading "1".
    raw = np.zeros((10, 30), np.uint8); raw[3:7, 10:21] = 255
    comp = np.zeros((10, 30), bool); comp[3:7, 0:25] = True
    _, xs = tighten(comp, raw)
    check("box tightens off the static icon", xs.min() == 10 and xs.max() == 20)


def test_download_options():
    import tempfile
    from tbavid.download import cookie_args
    # Documented in the Colab harvest notebook as the way past YouTube's
    # datacenter-IP blocking, so it has to actually reach the yt-dlp command.
    check("no cookies configured", cookie_args({}) == [])
    f = Path(tempfile.mkdtemp()) / "cookies.txt"
    f.write_text("# netscape\n")
    check("cookies passed when present",
          cookie_args({"ytdlp_cookies": str(f)}) == ["--cookies", str(f)])
    # A stale path in config must not make every download fail.
    check("missing cookie file ignored",
          cookie_args({"ytdlp_cookies": "/nope/cookies.txt"}) == [])


def test_labels():
    kr = [[3.0, 10.0], [20.0, 30.0]]
    check("clean time maps across the gap",
          source_time(7.0, kr) == 20.0 and source_time(6.9, kr) == 9.9)
    tl = Timeline({"blue": [[10.0, 5], [12.0, 9], [15.0, 20]]})
    check("windowed fuel count", tl.scored_between("blue", 10.0, 13.0) == 4)


def test_render():
    check("single range skips concat",
          "concat" not in build_filter([(0, 5)], "crop=1:1:0:0"))
    check("multiple ranges concat",
          "concat=n=3" in build_filter([(0, 1), (2, 3), (4, 5)], "crop=1:1:0:0"))


def test_identify():
    check("votes sum across frames", vote([{1: 0.2}, {1: 0.5, 2: 0.3}]) == {1: 0.7, 2: 0.3})
    # Three robots of one alliance look alike; independent argmax happily gives
    # two of them the same team number.
    a = best_assignment({1: {9: 0.9, 8: 0.4}, 2: {9: 0.8, 8: 0.5}}, [8, 9])
    check("no double assignment", len({v for v in a.values() if v}) == 2)


def test_ledger_and_picking():
    tmp = Path(tempfile.mkdtemp())
    led = Ledger(tmp / "s.json")
    led.record("A", L.OK); led.record("B", L.FAILED); led.record("C", L.TOO_LONG)
    check("ok is permanent", led.seen("A", retry_failed=True))
    check("failed is retryable", not led.seen("B", retry_failed=True))
    check("too_long stays skipped", led.seen("C", retry_failed=True))

    class FakeClient:
        def event_keys(self, year, competitive_only=True):
            return [f"2026e{i}" for i in range(6)]

        def event_matches(self, ek):
            return [{"key": f"{ek}_qm{n}", "event_key": ek, "comp_level": "qm",
                     "match_number": n, "set_number": 0,
                     "alliances": {"blue": {"team_keys": ["frc1", "frc2", "frc3"]},
                                   "red": {"team_keys": ["frc4", "frc5", "frc6"]}},
                     "videos": [{"type": "youtube", "key": f"{ek}v{n}"}]}
                    for n in range(1, 4)]

    c = FakeClient()
    led2 = Ledger(tmp / "t.json")
    first = pick_unseen(c, 2026, 4, led2, rng=random.Random(1))
    for p in first:
        led2.record(p["yt_key"], L.OK)
    second = pick_unseen(c, 2026, 4, led2, rng=random.Random(1))
    check("never pulls the same video twice",
          not (set(p["yt_key"] for p in first) & set(p["yt_key"] for p in second)))
    capped = pick_unseen(c, 2026, 6, Ledger(tmp / "u.json"), per_event_cap=1,
                         rng=random.Random(2))
    check("per-event cap honoured",
          max(Counter(p["event_key"] for p in capped).values()) == 1)
    check("rosters captured", first[0]["teams"]["blue"] == ["1", "2", "3"])

    # One stream can back several matches; keying on match_key would let the
    # same video in repeatedly. Legacy "tba" videos are not fetchable.
    matches = [{"key": "m1", "event_key": "e", "comp_level": "qm", "match_number": 1,
                "set_number": 0, "videos": [{"type": "youtube", "key": "S"}]},
               {"key": "m2", "event_key": "e", "comp_level": "qm", "match_number": 2,
                "set_number": 0, "videos": [{"type": "youtube", "key": "S"}]},
               {"key": "m3", "event_key": "e", "comp_level": "qm", "match_number": 3,
                "set_number": 0, "videos": [{"type": "tba", "key": "OLD"}]}]
    check("one stream yields one candidate, tba type skipped",
          [x["yt_key"] for x in youtube_candidates(matches)] == ["S"])
    check("match labels", match_label({"comp_level": "qm", "match_number": 42}) == "qm42"
          and match_label({"comp_level": "sf", "set_number": 3, "match_number": 1}) == "sf3m1")


def test_competitive_filter():
    # Offseason/preseason/unlabeled events run mixed rosters and demo rules;
    # training on them teaches the detector a field that does not exist at a
    # real event.
    events = [{"key": "2026casj", "event_type": 0},    # regional
              {"key": "2026dal", "event_type": 1},     # district
              {"key": "2026cmptx", "event_type": 4},   # championship final
              {"key": "2026foc", "event_type": 6},     # Festival of Champions
              {"key": "2026iri", "event_type": 99},    # offseason
              {"key": "2026week0", "event_type": 100}, # preseason
              {"key": "2026huh", "event_type": -1},    # unlabeled
              {"key": "2026nope"}]                     # no type at all
    kept = [e["key"] for e in events if is_competitive_event(e)]
    check("only official event types survive",
          kept == ["2026casj", "2026dal", "2026cmptx", "2026foc"])

    # event_keys() has to ask for /simple, not /keys: a bare key does not carry
    # event_type, so the filter would have nothing to read.
    c = TBAClient.__new__(TBAClient)   # no session or cache dir needed here
    c.get = lambda path: events if path.endswith("/simple") else ["ALL"]
    check("competitive event keys are filtered",
          c.event_keys(2026) == ["2026casj", "2026dal", "2026cmptx", "2026foc"])
    check("the escape hatch still returns every event",
          c.event_keys(2026, competitive_only=False) == ["ALL"])

    # Practice matches sit in the same event's match list as real play, and at
    # some events share a stream with it.
    matches = [{"key": "e_pm1", "event_key": "e", "comp_level": "pm",
                "match_number": 1, "set_number": 0,
                "videos": [{"type": "youtube", "key": "PRAC"}]},
               {"key": "e_qm1", "event_key": "e", "comp_level": "qm",
                "match_number": 1, "set_number": 0,
                "videos": [{"type": "youtube", "key": "QUAL"}]},
               {"key": "e_f1m1", "event_key": "e", "comp_level": "f",
                "match_number": 1, "set_number": 1,
                "videos": [{"type": "youtube", "key": "FINAL"}]}]
    check("practice matches are not candidates",
          [x["yt_key"] for x in youtube_candidates(matches)] == ["QUAL", "FINAL"])
    check("playoffs count as competitive",
          [x["label"] for x in youtube_candidates(matches)] == ["qm1", "f1m1"])
    check("including non-competitive brings practice back",
          [x["yt_key"] for x in youtube_candidates(matches, competitive_only=False)]
          == ["PRAC", "QUAL", "FINAL"])

    # A season with nothing but offseason events must stop, not silently walk
    # the offseason catalogue.
    class OffseasonOnly:
        def event_keys(self, year, competitive_only=True):
            return [] if competitive_only else ["2026iri"]

        def event_matches(self, ek):
            return []

    empty = False
    try:
        pick_unseen(OffseasonOnly(), 2026, 1, Ledger(Path(tempfile.mkdtemp()) / "c.json"))
    except SystemExit:
        empty = True
    check("no competitive events is a hard stop", empty)


def test_no_unbound_globals():
    """Every global a function reads must actually exist.

    _process() referenced `shard` and `shards`, which are parameters of
    fetch(), not of it. Nothing caught it: the crash lands after the
    download, shot analysis, crop, render and OCR have all run, so it needs
    a real video to reach and the suite deliberately has none. Reading the
    symbol table costs nothing and covers every function here.
    """
    import builtins
    import symtable

    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted((root / "tbavid").glob("*.py")) + \
                [root / "run.py", root / "serve.py"]:
        top = symtable.symtable(path.read_text(), str(path), "exec")
        defined = {s.get_name() for s in top.get_symbols()} | set(dir(builtins))

        def walk(table, trail):
            for child in table.get_children():
                name = f"{trail}.{child.get_name()}" if trail else child.get_name()
                for sym in child.get_symbols():
                    if (sym.is_global() and not sym.is_assigned()
                            and sym.get_name() not in defined):
                        offenders.append(f"{path.name}:{name}() -> {sym.get_name()}")
                walk(child, name)
        walk(top, "")

    check(f"no function reads an undefined global ({'; '.join(offenders)})"
          if offenders else "no function reads an undefined global",
          not offenders)


def test_packaging():
    """The archive builders, which ship the code to other people's machines."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "package", Path(__file__).resolve().parent.parent / "deploy" / "package.py")
    pkg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pkg)

    files = pkg.collect()
    arcs = {rel.as_posix() for _, rel in files}
    check("packages the code", {"run.py", "tbavid/tba.py"} <= arcs)
    # deploy/ is bundled wholesale, and *WITH_KEY* archives live in it. Without
    # this exclusion the "safe to upload" bundle quietly carries the key.
    check("never packages a WITH_KEY file",
          not any("WITH_KEY" in a for a in arcs))
    check("never packages caches or bulk data",
          not any(p in a.split("/") for a in arcs
                  for p in ("__pycache__", "data", "state", "dist")))
    # Windows builds must not emit backslash paths, or the archive unpacks as
    # one long filename on Linux.
    check("archive paths are posix", not any("\\" in a for a in arcs))

    # Release notes come from the CHANGELOG, so the two cannot drift apart.
    doc = ("# Changelog\n\n"
           "## v0.2.10 — 2026-10-01\n\nten.\n\n"
           "## v0.2.0 — 2026-09-16\n\nzero.\n\n### Added\n\n- a thing\n\n"
           "## v0.1.0-beta — 2026-09-15\n\nbeta.\n")
    check("extracts the right section",
          pkg.changelog_section("v0.2.0", doc) == "zero.\n\n### Added\n\n- a thing")
    # A prefix match would hand v0.2.0's tag v0.2.10's notes.
    check("v0.2.0 does not match v0.2.10",
          pkg.changelog_section("v0.2.10", doc) == "ten.")
    check("a leading v is optional",
          pkg.changelog_section("0.1.0-beta", doc) == "beta.")
    missing = False
    try:
        pkg.changelog_section("v9.9.9", doc)
    except SystemExit:
        missing = True
    check("an unknown version fails rather than releasing empty notes", missing)

    # The check that matters: a key anywhere in the payload stops the build.
    key = b"ZZZfakekeyfakekeyfakekeyfakekeyfakekey"
    refused = False
    try:
        pkg.assert_keyless([("config.json", b'{"x": "' + key + b'"}')], key, "test")
    except SystemExit:
        refused = True
    check("a key in the payload aborts the build", refused)
    pkg.assert_keyless([("config.json", b"clean")], key, "test")   # must not raise
    # No .env configured means no key to compare against, not "ship anything".
    pkg.assert_keyless([("config.json", b"anything")], b"", "test")


def test_audit():
    from tbavid import pipeline

    def vid(ek, label, frames):
        return {"event_key": ek, "label": label, "status": "ok",
                "exported": {"written": frames}}

    manifest = {"videos": {
        "a": vid("2026mimas", "qm33", 100),   # district, real play
        "b": vid("2026kylou", "qm8", 40),     # offseason event
        "c": vid("2026mimas", "pm1", 10),     # practice at a real event
        "d": vid("2025xxxxx", "qm1", 7),      # season TBA will not answer for
    }}

    class Stub:
        def event_keys(self, year, competitive_only=True):
            return ["2026mimas"] if year == 2026 else []

    out = pipeline.audit({}, manifest=manifest, client=Stub())
    check("audit counts every frame", out["frames"] == 157)
    # The offseason event and the practice match, not the unknown-season one:
    # a season TBA did not answer for is unknown, and condemning it would
    # delete good footage on no evidence.
    check("audit flags offseason and practice only",
          out["bad_videos"] == 2 and out["bad_frames"] == 50)
    check("unknown season is reported, not condemned",
          out["unknown_years"] == [2025]
          and "event type unknown" in out["events"]["2025xxxxx"]["reasons"])
    check("empty manifest audits cleanly",
          pipeline.audit({}, manifest={"videos": {}})["frames"] == 0)


def test_sharding():
    from tbavid.tba import shard_of
    keys = [f"vid{i:05d}" for i in range(4000)]
    # crc32, not hash(): Python randomises string hashing per process, so
    # hash() would reshuffle the partition on every run.
    owners = {k: shard_of(k, 4) for k in keys}
    check("shard assignment is stable", all(shard_of(k, 4) == owners[k] for k in keys))
    check("every video has exactly one owner in range",
          all(0 <= v < 4 for v in owners.values()))
    counts = Counter(owners.values())
    spread = max(counts.values()) - min(counts.values())
    check("shards are balanced", spread < len(keys) * 0.05)

    class FakeClient:
        def event_keys(self, y, competitive_only=True):
            return [f"2026e{i:02d}" for i in range(30)]

        def event_matches(self, ek):
            return [{"key": f"{ek}_qm{n}", "event_key": ek, "comp_level": "qm",
                     "match_number": n, "set_number": 0,
                     "alliances": {"blue": {"team_keys": []}, "red": {"team_keys": []}},
                     "videos": [{"type": "youtube", "key": f"{ek}v{n:02d}"}]}
                    for n in range(1, 11)]

    tmp = Path(tempfile.mkdtemp())
    c, picked = FakeClient(), {}
    for w in range(4):
        led = Ledger(tmp / f"w{w}.json")
        picked[w] = {p["yt_key"] for p in
                     pick_unseen(c, 2026, 20, led, rng=random.Random(w),
                                 shard=w, shards=4)}
    everything = [k for s in picked.values() for k in s]
    check("four workers never pick the same match",
          len(everything) == len(set(everything)) and len(everything) == 80)

    bad = False
    try:
        pick_unseen(c, 2026, 1, Ledger(tmp / "x.json"), shard=4, shards=4)
    except SystemExit:
        bad = True
    check("out-of-range shard is rejected", bad)


def test_db():
    con = connect(Path(tempfile.mkdtemp()) / "t.db")
    build(con, {"videos": {"v1": {
        "match_key": "2026x_qm1", "event_key": "2026x", "yt_key": "k", "label": "qm1",
        "status": "ok", "teams": {"blue": ["111", "222", "333"], "red": ["4", "5", "6"]},
        "analysis": {"width": 1920, "height": 1080, "coverage": 0.8,
                     "keep_ranges": [[0, 10]]},
        "crop": {"x": 0, "y": 100, "w": 1920, "h": 500},
        "score": {"series": {"blue": [[1, 0], [2, 5]]}, "final": {"blue": 5, "red": 1},
                  "events": [{"t": 2, "alliance": "blue", "balls": 5, "total": 5}]}}}})
    s = summary(con)
    check("db ingests a match",
          s["matches"] == 1 and s["match_teams"] == 6 and s["score_events"] == 1)
    check("team view resolves alliance fuel",
          con.execute("SELECT alliance_fuel FROM team_match_fuel WHERE team=111")
             .fetchone()[0] == 5)
    # A failed scoreboard read must degrade the record, never corrupt it.
    build(con, {"videos": {"v2": {
        "match_key": "2026x_qm2", "event_key": "2026x", "yt_key": "k2", "label": "qm2",
        "status": "ok", "teams": {},
        "analysis": {"width": 1920, "height": 1080, "coverage": 0.8,
                     "keep_ranges": [[0, 10]]},
        "crop": {"x": 0, "y": 100, "w": 1920, "h": 500},
        "score": {"error": "no counters"}}}})
    row = con.execute("SELECT scoreboard_ok, blue_fuel FROM matches WHERE match_key=?",
                      ("2026x_qm2",)).fetchone()
    check("failed scoreboard flagged, not faked", row[0] == 0 and row[1] is None)
    check("team_summary excludes unreadable matches",
          con.execute("SELECT COUNT(*) FROM team_summary").fetchone()[0] == 6)


def main() -> int:
    for fn in (test_cuts, test_clustering, test_crop_bands, test_formats,
               test_format_tuning, test_district_catalogue,
               test_audio_frames, test_audio_cues, test_audio_recovery,
               test_stream_alignment, test_scoreboard,
               test_download_options, test_labels, test_render, test_identify,
               test_ledger_and_picking, test_competitive_filter, test_audit,
               test_packaging, test_no_unbound_globals, test_sharding, test_db):
        fn()
    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
