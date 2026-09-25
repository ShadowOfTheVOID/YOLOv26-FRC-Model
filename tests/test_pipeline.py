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
from tbavid import identify
from tbavid import ledger as L
from tbavid import stream as S
from tbavid.crop import Profile, detect_bottom, detect_top
from tbavid.db import build, connect, export_for_serving, summary, write_live
from tbavid.live import Counter as LiveCounter
from tbavid.count import (BallCounter, health, hub_of, hubs_from_db,
                          learn_hubs)
from tbavid.shooting import ShotCounter
from tbavid.detect import (CLASSES, alliance_of, frame_path, model_source,
                           rows_from_boxes, write_rows)
from tbavid.field import AUTO, ENDED, IDLE, TELEOP, Match
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


def test_live_counter():
    """The live counter must agree with the batch one, read for read.

    `scoreboard.clean_series` is the tested rule and it builds a series from
    one call's worth of reads, seeding from the minimum of the first few. A
    live read arrives in chunks, so the state has to survive a boundary -
    called once per chunk it would re-seed every few seconds, losing the step
    across the join and letting one bad chunk re-anchor the match.

    So `Counter` is that rule made stateful, and two implementations of one
    rule drift unless something says they may not. This is that something.
    """
    rng = random.Random(11)
    mismatches = 0
    for _ in range(300):
        truth, v = [], 0
        for _ in range(rng.randint(5, 60)):
            v += rng.choice([0, 0, 1, 2, 3, 5, 8, 12])
            truth.append(v)
        vals = []
        for x in truth:
            r = rng.random()
            if r < 0.10:
                vals.append(None)                      # unreadable frame
            elif r < 0.18:
                vals.append(x * 10 + 1)                # a stray leading digit
            elif r < 0.22:
                vals.append(max(0, x - rng.randint(1, 30)))   # reads backwards
            else:
                vals.append(x)
        times = [i * 0.2 for i in range(len(vals))]
        if clean_series(vals, times) != LiveCounter().feed(vals, times):
            mismatches += 1
    check("the live counter agrees with clean_series on noisy reads",
          mismatches == 0)

    # And the thing clean_series cannot do. Split anywhere and the stitched
    # result must equal the one-pass read of the whole thing.
    vals = [100, 100, 103, 103, 108, None, 108, 112, 112, 118]
    times = [i * 0.2 for i in range(len(vals))]
    whole = clean_series(vals, times)
    for cut in range(1, len(vals)):
        c = LiveCounter()
        c.feed(vals[:cut], times[:cut])
        c.feed(vals[cut:], times[cut:])
        if c.points != whole:
            check(f"stitching at {cut} matches a single pass", False)
            break
    else:
        check("stitching at every boundary matches a single pass", True)

    # The failure this exists to prevent, stated as a fact: cleaning each chunk
    # on its own re-seeds mid-match.
    half = len(vals) // 2
    per_chunk = (clean_series(vals[:half], times[:half])
                 + clean_series(vals[half:], times[half:]))
    check("per-chunk cleaning would re-seed, which is why it is not used",
          per_chunk != whole)

    check("a chunk of nothing readable changes nothing",
          LiveCounter().feed([None, None], [0.0, 0.2]) == [])


def test_live_rows():
    """A live read files a timeline and a roster, and no dataset rows."""
    with tempfile.TemporaryDirectory() as tmp:
        con = connect(Path(tmp) / "live.db")
        series = {"blue": [[1.0, 10], [3.0, 14], [9.0, 30]],
                  "red": [[2.0, 4], [8.0, 9]]}
        got = write_live(con, "2026caclv", "2026caclv_qm14", series,
                         {"blue": 30, "red": 9},
                         teams={"blue": ["254", "frc1678", "8033"],
                                "red": ["971", "604", "1323"]},
                         label="qm14")
        check("the timeline is written as scoring events", got["score_events"] == 3)
        check("and the alliance totals land on the match",
              tuple(con.execute("SELECT blue_fuel, red_fuel FROM matches"
                                " WHERE match_key=?",
                                ("2026caclv_qm14",)).fetchone()) == (30, 9))
        check("the roster comes off TBA, with frc stripped",
              [r[0] for r in con.execute(
                  "SELECT team FROM match_teams WHERE match_key=? AND"
                  " alliance='blue' ORDER BY station",
                  ("2026caclv_qm14",))] == [254, 1678, 8033])
        check("a live row says it is live, so it cannot be confused with a "
              "re-readable one",
              con.execute("SELECT status FROM matches WHERE match_key=?",
                          ("2026caclv_qm14",)).fetchone()[0] == "live")
        check("nothing entered the dataset",
              con.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 0)
        check("and the match has no video to point at, because none was kept",
              tuple(con.execute("SELECT video_id, clean_path FROM matches"
                                " WHERE match_key=?",
                                ("2026caclv_qm14",)).fetchone()) == (None, None))
        check("the comp level is read off the label",
              con.execute("SELECT comp_level FROM matches WHERE match_key=?",
                          ("2026caclv_qm14",)).fetchone()[0] == "qm")

        # Re-filing the same match replaces its timeline rather than doubling it.
        write_live(con, "2026caclv", "2026caclv_qm14", series,
                   {"blue": 30, "red": 9}, label="qm14")
        check("re-filing a match does not append a second timeline",
              con.execute("SELECT COUNT(*) FROM score_events").fetchone()[0] == 3)

        # OCR that never read anything must not land as a confident zero.
        out = write_live(con, "2026caclv", "2026caclv_qm15", {"blue": [], "red": []},
                         {"blue": None, "red": None}, label="qm15")
        check("a match whose counters never read is flagged, not zeroed",
              out["scoreboard_ok"] == 0
              and con.execute("SELECT blue_fuel FROM matches WHERE match_key=?",
                              ("2026caclv_qm15",)).fetchone()[0] is None)
        # The scouting API's team view counts only clean reads, so an unread
        # match must not drag an average down.
        check("and it is excluded from the team view rather than averaged in",
              con.execute("SELECT COUNT(*) FROM team_match_fuel"
                          " WHERE match_key=?", ("2026caclv_qm15",)).fetchone()[0]
              == 0)
        con.close()


HUB = {"blue": (300.0, 100.0, 100.0, 100.0), "red": (700.0, 100.0, 100.0, 100.0)}


def _fly(tid, x0, y0, x1, y1, n, start=0):
    """A ball moving in a straight line over n frames."""
    return {tid: [(start + i, (x0 + (x1 - x0) * i / (n - 1),
                               y0 + (y1 - y0) * i / (n - 1), 12.0, 12.0))
                  for i in range(n)]}


def _play(paths, **kw):
    """Run a set of ball paths through a counter. Returns (events, counter)."""
    c = BallCounter(HUB, **kw)
    last = max(f for p in paths.values() for f, _ in p) if paths else 0
    events = []
    for frame in range(last + 30):
        balls = {tid: box for tid, path in paths.items()
                 for f, box in path if f == frame}
        events += c.update(frame, frame / 30.0, balls)
    events += c.flush(last + 40, (last + 40) / 30.0)
    return events, c


def test_ball_counting():
    """Counting fuel with no scoreboard, which means refusing the lookalikes.

    A ball going in stops being visible, so the event is a track vanishing
    inside a hub. Three other things look exactly like that, and almost all of
    count.py is saying no to them -- so almost all of this is too.
    """
    events, c = _play(_fly(1, 50, 150, 348, 150, 20))
    check("a ball flown into the hub counts",
          [(e["alliance"], e["total"]) for e in events] == [("blue", 1)])
    check("and nothing is credited to the other alliance", c.totals["red"] == 0)

    # A false detection that appears and is gone. A ball that was really there
    # was there for more than a frame.
    _, c = _play({2: [(5, (340.0, 145.0, 10.0, 10.0))]})
    check("a one-frame blob in the hub is not a ball",
          c.totals["blue"] == 0 and c.rejected["too_short"] == 1)

    # THE false positive that matters at a real field: a ball resting by the
    # hub that a robot drives in front of. It vanishes, and it is in the hub
    # region. It did not cross in, which is the difference.
    _, c = _play({3: [(i, (345.0, 150.0, 12.0, 12.0)) for i in range(15)]})
    check("a ball already in the hub, then occluded, is not a score",
          c.totals["blue"] == 0 and c.rejected["started_inside"] == 1)
    # ...and --allow-inside is the documented trade: recall for precision.
    _, c = _play({3: [(i, (345.0, 150.0, 12.0, 12.0)) for i in range(15)]},
                 require_entry=False)
    check("allow_inside counts it, which is what that flag is for",
          c.totals["blue"] == 1)

    # A ball passing OVER the hub vanishes behind it and comes back with a new
    # track id. This cannot be settled in the moment, so the score is held and
    # the reappearance withdraws it.
    over = _fly(4, 50, 150, 348, 150, 20)
    over.update(_fly(5, 360, 150, 600, 150, 20, start=24))
    _, c = _play(over)
    check("a ball that passes over the hub and reappears does not count",
          c.totals["blue"] == 0 and c.rejected["reacquired"] == 1)
    # The same flight with nothing reappearing is a score, so the rule above is
    # discriminating between two cases rather than just refusing both.
    _, c = _play(_fly(4, 50, 150, 348, 150, 20))
    check("...but the same flight with nothing coming back out does",
          c.totals["blue"] == 1)

    _, c = _play(_fly(8, 50, 400, 200, 400, 15))
    check("a ball that vanishes away from any hub is not a score",
          c.totals["blue"] == 0 and c.rejected["not_in_hub"] == 1)

    # Two in a row, and the running total is what a scoreboard would show.
    two = _fly(6, 50, 150, 348, 150, 15)
    two.update(_fly(7, 50, 160, 348, 158, 15, start=40))
    events, c = _play(two)
    check("consecutive scores carry a running total",
          [e["total"] for e in events] == [1, 2] and c.totals["blue"] == 2)

    # Each hub scores for its own alliance.
    events, c = _play(_fly(9, 900, 150, 748, 150, 20))
    check("the red hub scores for red",
          [(e["alliance"], e["total"]) for e in events] == [("red", 1)])

    # The shape db.write_live wants, so a count can be filed like any reading.
    check("the series is cumulative per alliance",
          c.series(events)["red"] == [[events[0]["t"], 1]])

    # Nothing silently disappears: every refusal is counted and reported,
    # because a counter that rejects quietly is one nobody can debug.
    lines = "\n".join(c.report())
    check("the report says what was counted", "counted 1 ball" in lines)
    _, c = _play({2: [(5, (340.0, 145.0, 10.0, 10.0))]})
    check("and why anything was not",
          any("too few frames" in l for l in c.report()))


def _still(tid, alliance, box, frames):
    """A robot standing in one place for the given frames."""
    return {tid: [(f, (alliance, box)) for f in range(frames)]}


def _play_shots(robots, balls, **kw):
    """Run robot and ball paths through a ShotCounter. Returns (events, counter)."""
    c = ShotCounter(HUB, **kw)
    frames = [f for p in list(robots.values()) + list(balls.values()) for f, _ in p]
    last = max(frames) if frames else 0
    events = []
    for frame in range(last + 30):
        r = {tid: v for tid, path in robots.items() for f, v in path if f == frame}
        b = {tid: box for tid, path in balls.items() for f, box in path if f == frame}
        events += c.update(frame, frame / 30.0, r, b)
    events += c.flush(last + 40, (last + 40) / 30.0)
    return events, c


# A blue robot standing below-left of the blue hub (300..400 x 100..200).
BLUE_BOT = (100.0, 300.0, 80.0, 60.0)
RED_BOT = (500.0, 300.0, 80.0, 60.0)


def _shot(tid, x1, y1, n=20, start=0, x0=134.0, y0=300.0, keep=None):
    """A ball leaving the blue robot's top edge for (x1, y1), top-left coords.

    `keep` drops frames from the flight, to make a track the detector broke.
    """
    path = _fly(tid, x0, y0, x1, y1, n, start=start)[tid]
    if keep is not None:
        path = [(f, b) for i, (f, b) in enumerate(path) if keep(i)]
    return {tid: path}


def test_shot_attribution():
    """Who shot, and did it go in -- including the misses count.py cannot see.

    Every case here is a way per-robot scouting numbers go wrong without anyone
    noticing: a ball in a hopper counted as a shot every time its robot drove,
    an intake counted as a make, a pass-over counted as in, and -- the one the
    first detector's 0.62 per-frame recall makes routine -- a flight the tracker
    broke in two, read as a miss by its shooter plus a make by nobody.
    """
    bot = _still(1, "blue", BLUE_BOT, 80)

    events, c = _play_shots(bot, _shot(10, 344, 144))
    s = c.per_robot.get(1, {})
    check("a shot from a robot into its hub is a make for that robot",
          (s.get("shots"), s.get("made"), s.get("missed")) == (1, 1, 0))
    check("and says which robot, alliance and outcome",
          [(e["robot"], e["alliance"], e["outcome"]) for e in events]
          == [(1, "blue", "made")])

    _, c = _play_shots(bot, _shot(11, 600, 420))
    s = c.per_robot.get(1, {})
    check("a shot that lands away from any hub is a miss -- which count.py never sees",
          (s.get("shots"), s.get("made"), s.get("missed")) == (1, 0, 1))

    # A ball riding in a hopper while its robot drives 200 px. It travels far
    # from where it started, but never gets clear of the robot it is in.
    driving = {1: [(f, ("blue", (100.0 + 5 * f, 300.0, 80.0, 60.0))) for f in range(40)]}
    carried = {12: [(f, (134.0 + 5 * f, 310.0, 12.0, 12.0)) for f in range(40)]}
    _, c = _play_shots(driving, carried)
    check("a ball carried in a moving robot is not a shot",
          not c.per_robot and c.ignored["carried"] == 1)

    # A ball on the floor rolled into a robot's intake: ends at a robot, not a hub.
    intake = {13: [(f, (400.0 - 8 * f, 350.0, 12.0, 12.0)) for f in range(30)]}
    _, c = _play_shots(bot, intake)
    check("a ball picked up off the floor is neither a shot nor a make",
          not c.per_robot and sum(c.hub_totals().values()) == 0)

    # Seen for the first time mid-air -- launch hidden behind another robot.
    # Nobody gets the credit, but the hub still has the ball.
    _, c = _play_shots(bot, _fly(14, 240, 150, 344, 144, 12))
    check("a make with no visible shooter is unattributed, not lost",
          not c.per_robot and c.unattributed["blue"] == 1
          and c.hub_totals()["blue"] == 1)

    # The flight broken for 3 frames: the old track is still 'missing' when the
    # new id appears where the ball was heading.
    short = _shot(15, 344, 144, keep=lambda i: not 10 <= i <= 12)
    short = {15: [(f, b) for f, b in short[15] if f < 10],
             16: [(f, b) for f, b in short[15] if f > 12]}
    _, c = _play_shots(bot, short)
    s = c.per_robot.get(1, {})
    check("a flight the tracker broke briefly is still one make",
          (s.get("shots"), s.get("made"), s.get("missed")) == (1, 1, 0)
          and c.unattributed["blue"] == 0 and c.stitched == 1)

    # Broken for long enough that the first piece had already ended as a miss.
    longer = _shot(17, 344, 144, n=24)
    longer = {17: [(f, b) for f, b in longer[17] if f < 8],
              18: [(f, b) for f, b in longer[17] if f > 14]}
    _, c = _play_shots(bot, longer)
    s = c.per_robot.get(1, {})
    check("...and a longer break takes back the miss it had looked like",
          (s.get("made"), s.get("missed")) == (1, 0) and c.unattributed["blue"] == 0)

    # Into the hub region, then out the other side: it passed over.
    over = _shot(19, 344, 144)
    over.update(_fly(20, 360, 144, 620, 144, 15, start=24))
    _, c = _play_shots(bot, over)
    s = c.per_robot.get(1, {})
    check("a shot that passes over the hub and lands beyond it is a miss",
          (s.get("made"), s.get("missed")) == (0, 1) and c.reacquired == 1)

    # Two robots; the ball starts at the red one and goes in the blue hub.
    both = dict(bot)
    both.update(_still(2, "red", RED_BOT, 80))
    events, c = _play_shots(both, _shot(21, 344, 144, x0=534.0))
    check("the ball is credited to the robot it left, not the nearest one later",
          2 in c.per_robot and 1 not in c.per_robot)
    check("and a red robot scoring in the blue hub is recorded as that, not a make",
          c.per_robot[2]["wrong_hub"] == 1 and c.per_robot[2]["made"] == 0
          and c.hub_totals()["blue"] == 1)

    # Shooting from against the hub: almost no room to clear the robot, but the
    # hub proves the ball left it.
    close = _still(3, "blue", (300.0, 212.0, 80.0, 60.0), 40)
    _, c = _play_shots(close, {22: [(f, (330.0, 206.0 - 8 * f, 12.0, 12.0))
                                    for f in range(8)]})
    check("a shot from against the hub still counts, though it barely cleared",
          c.per_robot.get(3, {}).get("made") == 1)

    # Tracks fold into teams, and an unassigned one is kept, not dropped.
    two = _shot(23, 344, 144)
    two.update(_shot(24, 600, 420, start=30))
    _, c = _play_shots(_still(1, "blue", BLUE_BOT, 80), two)
    teams = c.by_team({1: 254})
    check("by_team folds a robot's tracks into its team",
          teams[254]["shots"] == 2 and teams[254]["made"] == 1
          and teams[254]["missed"] == 1)
    check("the report reads per robot, with accuracy",
          any("2 shots, 1 made, 1 missed" in l and "50%" in l for l in c.report()))

    # The ordinary case on a real field: a ball rides in the hopper while the
    # robot drives, then is shot. One track, one shot -- the carrying is not a
    # second one, and the drive does not make it a shot early.
    drive = {4: [(f, ("blue", (60.0 + 3 * f, 300.0, 80.0, 60.0))) for f in range(60)]}
    ride = [(f, (94.0 + 3 * f, 310.0, 12.0, 12.0)) for f in range(20)]
    x, y = ride[-1][1][0], ride[-1][1][1]
    ride += [(20 + i, (x + (344 - x) * i / 14, y + (144 - y) * i / 14, 12.0, 12.0))
             for i in range(1, 15)]
    _, c = _play_shots(drive, {25: ride})
    s = c.per_robot.get(4, {})
    check("a ball carried and then shot is one shot, and a make",
          (s.get("shots"), s.get("made")) == (1, 1) and c.ignored["carried"] == 0)

    # Scouting and scoring must never disagree about what went in. The same
    # balls through BallCounter and ShotCounter give the same per-hub totals,
    # whoever shot them and whether a shooter was seen at all.
    both = dict(_still(1, "blue", BLUE_BOT, 90))
    both.update(_still(2, "red", RED_BOT, 90))
    balls = _shot(26, 344, 144)                            # blue robot, made
    balls.update(_fly(27, 240, 150, 344, 144, 12, start=30))   # nobody, made
    balls.update(_shot(28, 344, 150, x0=534.0, start=50))  # red robot, blue hub
    balls.update(_shot(29, 600, 420, start=20))            # blue robot, missed
    _, shots = _play_shots(both, balls)
    _, scores = _play(balls)
    check("per-hub totals from shots agree with the scoring counter",
          shots.hub_totals() == scores.totals
          and shots.hub_totals()["blue"] == 3)


def test_hub_geometry():
    """Where the hub is, learned or recorded."""
    check("a point in the hub names its alliance",
          hub_of(HUB, (350.0, 150.0)) == "blue"
          and hub_of(HUB, (750.0, 150.0)) == "red")
    check("a point outside both names neither", hub_of(HUB, (10.0, 10.0)) is None)
    check("padding widens the region",
          hub_of(HUB, (410.0, 150.0)) is None
          and hub_of(HUB, (410.0, 150.0), pad=20.0) == "blue")

    # The camera is fixed for a whole event, so the true box is constant and
    # the spread is detector jitter plus the odd frame where a bumper was
    # called a hub. A median ignores those; a mean is dragged by them.
    frames = [{"blue": (300.0, 100.0, 100.0, 100.0)} for _ in range(9)]
    frames.append({"blue": (900.0, 900.0, 40.0, 40.0)})        # one bad frame
    check("the hub box is the median, so one bad frame does not move it",
          learn_hubs(frames)["blue"] == (300.0, 100.0, 100.0, 100.0))
    check("nothing seen means nothing learned", learn_hubs([]) == {})

    with tempfile.TemporaryDirectory() as tmp:
        con = connect(Path(tmp) / "h.db")
        from tbavid.db import set_hub
        set_hub(con, "2026caclv", "blue", [300, 100, 100, 100])
        got = hubs_from_db(con, "2026caclv")
        check("a recorded hub box comes back as a box",
              got == {"blue": (300, 100, 100, 100)})
        check("an event with no hub recorded reads as none, not a guess",
              hubs_from_db(con, "2026nope") == {})
        con.close()


def test_scrimmage_scoreboard():
    """At a scrimmage there is no FMS, so this count IS the score.

    That changes what it has to be. A number nobody can see, with no clock and
    no way to correct it, is not a scoreboard - and the three things added for
    that are each here because of something the counter cannot do alone.
    """
    m = Match("Q1", auto_s=1.0, teleop_s=2.0,
              points_per_ball={AUTO: 4.0, TELEOP: 2.0})

    # People throw fuel around between matches and robots get tested on the
    # field. A counter running continuously would add all of it to the score.
    check("a ball before the match starts is refused",
          m.ball("blue") is False and m.phase() == IDLE)

    m.start()
    check("the clock starts in auto", m.phase() == AUTO)
    m.ball("blue"); m.ball("blue")
    blue = m.state()["alliances"]["blue"]
    check("auto balls score at the auto rate",
          blue["total"] == 2 and blue["points"] == 8.0)

    # Phases are what make per-phase scoring possible at all, and a scrimmage
    # is exactly where those rates get changed.
    m.started_at -= 1.05
    check("the clock rolls into teleop on its own", m.phase() == TELEOP)
    m.ball("blue")
    blue = m.state()["alliances"]["blue"]
    check("and teleop balls score at the teleop rate",
          blue["balls"] == {AUTO: 2, TELEOP: 1} and blue["points"] == 10.0)

    # THE feature that makes this usable as a real scoreboard. count.py is
    # careful and still fallible - it cannot see a ball occluded for its whole
    # flight. Everywhere else this repository answers uncertainty by recording
    # nothing, which is useless when a match needs a final score in thirty
    # seconds. So a person gets the last word.
    m.adjust("red", 2)
    red = m.state()["alliances"]["red"]
    check("a referee can add a ball the camera missed", red["total"] == 2)
    check("and the two are never folded together",
          red["detected"] == 0 and red["adjusted"] == 2)
    m.adjust("red", -1)
    check("a referee can take one away too",
          m.state()["alliances"]["red"]["total"] == 1)
    m.adjust("red", -50)
    check("but cannot drive a score below zero",
          m.state()["alliances"]["red"]["total"] == 0)
    check("a zero adjustment changes nothing", m.adjust("blue", 0) is False)
    check("an alliance that does not exist is refused",
          m.adjust("purple", 1) is False)

    # Most corrections happen after the buzzer - somebody saw a ball go in that
    # the camera did not - so the detector stops and the referee does not.
    m.started_at -= 5.0
    check("the match ends on its own", m.phase() == ENDED)
    before = m.state()["alliances"]["blue"]["total"]
    check("the detector is ignored after the buzzer", m.ball("blue") is False)
    m.adjust("blue", 1)
    check("but a referee can still correct it",
          m.state()["alliances"]["blue"]["total"] == before + 1)

    check("the totals are what db.write_live files",
          m.totals() == {"blue": 4, "red": 0})
    check("and the log says who put each ball there",
          {row["by"] for row in m.state()["log"]} == {"detector", "ref"})

    # Points default to one a ball, because db.py refuses to convert fuel to
    # points and a scrimmage runs whatever rules its organiser chose.
    plain = Match("Q2")
    plain.start()
    plain.ball("blue")
    check("with no rates configured the display shows ball count",
          plain.state()["alliances"]["blue"]["points"] == 1.0)

    # Reset is what happens between matches, and it must leave nothing behind.
    plain.reset("Q3")
    st = plain.state()
    check("reset clears the score, the log and the clock",
          st["alliances"]["blue"]["total"] == 0 and st["log"] == []
          and st["phase"] == IDLE and st["label"] == "Q3")


def test_nothing_to_verify_against():
    """At a scrimmage nothing else is counting, so this has to check itself.

    The failure that matters is silent. A box that cannot process frames as
    fast as the camera produces them misses balls between the frames it does
    see, and the score comes out low with no gap, no error and nothing that
    looks unusual - and with no FMS there is no second number anywhere that
    would disagree with it.
    """
    slow = health([1 / 15.0] * 30, expect_fps=30.0, frames=450, counter=None)
    check("a box at half the camera's rate is not keeping up",
          slow["keepingUp"] is False)
    check("and says roughly how much went past unseen",
          abs(slow["missedFrac"] - 0.5) < 0.05)

    fast = health([1 / 31.0] * 30, expect_fps=30.0, frames=900, counter=None)
    check("a box that is keeping up says so", fast["keepingUp"] is True)
    check("and reports nothing missed", fast["missedFrac"] == 0.0)

    # Without the camera's rate there is no way to tell a slow processor from
    # a slow camera. None, not True: claiming health it cannot know is the one
    # thing worse than saying nothing, because nothing else will correct it.
    blind = health([1 / 5.0] * 30, expect_fps=0.0, frames=100, counter=None)
    check("with no expected rate it declines to judge",
          blind["keepingUp"] is None and blind["missedFrac"] is None)
    check("but still reports what it measured", blind["fps"] == 5.0)

    check("no timings at all is not a claim of zero",
          health([], 30.0, 0, None)["keepingUp"] is None)

    # The counter's own refusals ride along, so "the score looks low" has an
    # answer other than a shrug.
    c = BallCounter(HUB)
    _, c = _play({2: [(5, (340.0, 145.0, 10.0, 10.0))]})
    h = health([1 / 30.0] * 5, 30.0, 150, c)
    check("rejections are reported beside the rate",
          h["rejected"]["too_short"] == 1 and h["held"] == 0)

    # The one quantity at a scrimmage that CAN be checked against something
    # outside: there is a timer on the wall, and the two can be compared by
    # looking.
    m = Match("Q1", auto_s=15.0, teleop_s=135.0)
    m.start()
    check("the clock can be moved to follow an outside one", m.sync(100.0))
    st = m.state()
    check("and lands where it was put",
          st["phase"] == TELEOP and abs(st["elapsed"] - 100.0) < 0.5)
    check("a negative clock is refused", m.sync(-5.0) is False)

    # Moving the clock must not move the score. Re-attributing balls to
    # whatever phase the new clock implies would invent information about when
    # they went in.
    m2 = Match("Q2", auto_s=15.0, teleop_s=135.0)
    m2.start()
    m2.ball("blue")                      # scored in auto
    m2.sync(100.0)                       # now teleop
    check("syncing the clock does not re-attribute balls already counted",
          m2.state()["alliances"]["blue"]["balls"] == {AUTO: 1, TELEOP: 0})

    # A sync backwards from after the buzzer restarts the clock rather than
    # leaving a match that says 'ended' at t=40.
    m3 = Match("Q3", auto_s=1.0, teleop_s=1.0)
    m3.start(); m3.stop()
    check("the match had ended", m3.phase() == ENDED)
    m3.sync(0.5)
    check("and syncing back into the match un-ends it", m3.phase() == AUTO)

    check("health reaches the consumer in the same payload as the score",
          "health" in m.state())


def test_counting_model_dataset():
    """Deriving the counting model's dataset from the five-class one.

    `count.py` reads fuel and the two hubs and ignores robots on every frame.
    Two unused classes are work done per frame for an output nothing reads,
    and at a scrimmage a model that cannot keep up misses balls silently - so
    dropping them is not cosmetic.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from subset_classes import CLASSES as SRC_CLASSES
    from subset_classes import COUNTING, index_map, remap_labels

    check("the subset tool shares the canonical class order",
          list(SRC_CLASSES) == list(CLASSES))
    check("and counting wants fuel plus both hubs",
          COUNTING == ["fuel", "hub_blue", "hub_red"])

    # THE thing that fails silently. Drop robot_blue (1) and hub_blue stops
    # being 3 and becomes 1. Get it wrong and the model trains happily on fuel
    # labelled as hubs, and the first sign is a scrimmage scoreboard counting
    # nonsense with nothing to check it against.
    m = index_map(COUNTING)
    check("indices are remapped to the new order", m == {0: 0, 3: 1, 4: 2})
    text, n = remap_labels(
        "0 .5 .5 .1 .1\n1 .2 .2 .1 .1\n3 .7 .7 .2 .2\n4 .9 .9 .2 .2\n", m)
    check("fuel stays 0, the hubs become 1 and 2, the robot goes",
          text == "0 .5 .5 .1 .1\n1 .7 .7 .2 .2\n2 .9 .9 .2 .2\n" and n == 3)
    check("the box geometry is carried through untouched",
          all(line.split()[1:] == orig.split()[1:] for line, orig in
              zip(text.splitlines(),
                  ["x .5 .5 .1 .1", "x .7 .7 .2 .2", "x .9 .9 .2 .2"])))

    # A label file is machine-written, so a line that does not parse is
    # corruption; carrying it into a second dataset would hide where it began.
    junk, n = remap_labels("not a label\n\n2\nx .1 .1 .1 .1\n", m)
    check("malformed rows are dropped, not passed through", junk == "" and n == 0)
    check("a frame with only robots loses every box",
          remap_labels("1 .2 .2 .1 .1\n2 .3 .3 .1 .1\n", m) == ("", 0))

    # A different order is a different model, and must be honoured as given.
    flipped = index_map(["hub_red", "fuel"])
    check("class order follows what was asked for, not the source order",
          flipped == {4: 0, 0: 1})
    try:
        index_map(["fuel", "banana"])
        check("a class that does not exist is refused", False)
    except SystemExit:
        check("a class that does not exist is refused", True)


def test_model_must_name_its_classes():
    """Two class orders now exist, so a model must say which it has.

    Five for the scouting detector, three for the counting one. Assuming the
    wrong one relabels every detection without failing anything - hub_blue
    read as robot_blue - and at a scrimmage nothing downstream would catch it.
    """
    from tbavid import count as C

    class NoNames:
        names = {}

    class RobotsOnly:
        names = {0: "robot_blue", 1: "robot_red"}

    out = C.run_source(NoNames(), "src", None)
    check("a model with no class names is refused, not guessed at",
          "no class names" in out.get("error", ""))
    out = C.run_source(RobotsOnly(), "src", None)
    check("and so is one with nothing it could count",
          "cannot count balls" in out.get("error", ""))
    check("the refusal names what is missing",
          "fuel" in out["error"] and "hub_blue" in out["error"])

    # Benchmark summary: the median is the speed, the p95 is where balls go.
    from benchmark import summarise
    stally = summarise([0.02] * 95 + [0.5] * 5, 30.0)
    check("a model that stalls still reports a healthy median",
          stally["fps"] == 50.0)
    check("but the p95 shows where the balls go",
          stally["worstFps"] == 2.0)
    slow = summarise([1 / 12.0] * 50, 30.0)
    check("a model too slow for the camera says so",
          slow["keepingUp"] is False and abs(slow["missedFrac"] - 0.6) < 0.05)
    check("and a fast enough one says that",
          summarise([1 / 45.0] * 50, 30.0)["keepingUp"] is True)
    check("with no target rate it makes no claim",
          "keepingUp" not in summarise([1 / 45.0] * 50))
    check("no timings at all is not a speed", summarise([]) == {})


def test_api_stays_stdlib():
    """The serving path must import nothing but the standard library.

    This is load-bearing rather than tidy. `deploy/frc-harvest.service` runs
    serve.py on a host with python3 and nothing else -- no pip install, no
    wheels, no build step -- and `deploy/HOSTING.md` promises exactly that. One
    `import numpy` added to api.py or db.py for convenience would break every
    such host, and only on deployment, where the symptom is a service that will
    not start on a machine nobody is sitting at.

    It matters more now that requirements-detect.txt exists: torch and
    ultralytics are in this repo's orbit, and the one place they must never
    reach is the box that answers the scouting app.
    """
    import ast
    third = {"numpy", "requests", "ultralytics", "torch", "cv2", "PIL", "scipy",
             "pandas", "yaml"}
    root = Path(__file__).resolve().parent.parent
    for rel in ("serve.py", "tbavid/api.py", "tbavid/db.py", "tbavid/config.py",
                "tbavid/identify.py"):
        found = set()
        for node in ast.walk(ast.parse((root / rel).read_text())):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names} & third
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in third:
                    found.add(node.module.split(".")[0])
        check(f"{rel} imports no third-party package", not found)

    # And the detector must NOT be importable without them, i.e. the lazy
    # import has to stay lazy -- `from tbavid import detect` is reached by
    # run.py's arg parsing on a box that may have no torch at all.
    import tbavid.detect as d
    check("detect.py itself imports torch only when a model is loaded",
          "ultralytics" not in {n.names[0].name.split(".")[0]
                                for n in ast.walk(ast.parse(
                                    (root / "tbavid" / "detect.py").read_text()))
                                if isinstance(n, ast.Import) and n.names}
          and d.CLASSES[0] == "fuel")


def test_detect_rows():
    """Turning a model's boxes into detection rows.

    Every decision here is one that would mislabel data rather than crash, so
    each is pinned: the class index lookup, the alliance, and which classes are
    allowed to carry a track.
    """
    # THE important one. train/prepare_dataset.py writes the dataset's
    # data.yaml, so its CLASSES list is what each index MEANS to the trained
    # weights. If detect.py's copy drifts, nothing fails -- every detection is
    # silently relabelled, and blue robots are recorded as red.
    src = (Path(__file__).resolve().parent.parent
           / "train" / "prepare_dataset.py").read_text()
    line = next(l for l in src.splitlines() if l.startswith("CLASSES"))
    trained = eval(line.split("=", 1)[1].strip())
    check("detect's class order matches the dataset's data.yaml",
          list(CLASSES) == list(trained))

    check("alliance comes off the class name",
          [alliance_of(c) for c in CLASSES]
          == [None, "blue", "red", "blue", "red"])

    names = {i: n for i, n in enumerate(CLASSES)}
    boxes = [
        (10.2, 20.8, 50.4, 60.1, 0.91, 1, 7),      # robot_blue, tracked
        (11.0, 21.0, 12.0, 22.0, 0.40, 0, 9),      # fuel, with a track id
        (5.0, 5.0, 25.0, 25.0, 0.70, 4, None),     # hub_red, untracked
    ]
    rows = rows_from_boxes(boxes, names, source="runs/x/weights/best.pt")
    check("a box becomes x/y/w/h in frame pixels",
          (rows[0]["x"], rows[0]["y"], rows[0]["w"], rows[0]["h"]) == (10, 21, 40, 39))
    check("the robot keeps its track", rows[0]["track_id"] == 7)
    # A ball at 3 fps moves further between samples than its own width, so a
    # track id on one means nothing -- and would reach identify.assign_tracks,
    # which is looking for robots.
    check("a ball's track id is dropped", rows[1]["track_id"] is None)
    check("fuel belongs to no alliance", rows[1]["alliance"] is None)
    check("an untracked hub still records", rows[2]["cls"] == "hub_red"
          and rows[2]["alliance"] == "red" and rows[2]["track_id"] is None)
    check("every row says which model produced it",
          all(r["source"] == "runs/x/weights/best.pt" for r in rows))

    # A model trained against a different data.yaml would otherwise be stored
    # under a class name invented here.
    check("an unknown class index is dropped, not renamed",
          rows_from_boxes([(0, 0, 10, 10, 0.9, 99, None)], names) == [])
    check("a zero-area box is dropped",
          rows_from_boxes([(10, 10, 10, 10, 0.9, 1, 1)], names) == [])
    # Coordinates can arrive either way round.
    flipped = rows_from_boxes([(50, 60, 10, 20, 0.9, 1, 1)], names)
    check("a box given corner-reversed still has positive extent",
          (flipped[0]["x"], flipped[0]["y"], flipped[0]["w"], flipped[0]["h"])
          == (10, 20, 40, 40))

    check("weights are identified by their run, not an absolute path",
          model_source(Path("/a/b/YOLOv26-FRC-Model/runs/h/weights/best.pt"))
          == "runs/h/weights/best.pt"
          and model_source(Path("best.pt")) == "best.pt")
    check("a bare frame name resolves into the frame directory",
          frame_path("a_b_c_000001.jpg").parent.name == "frames")
    check("and a path is left alone",
          frame_path("/tmp/x/y.jpg") == Path("/tmp/x/y.jpg"))


def test_detect_writes():
    """Detections land against a real frame and are replaceable per match."""
    with tempfile.TemporaryDirectory() as tmp:
        con = connect(Path(tmp) / "d.db")
        write_live(con, "2026caclv", "2026caclv_qm14", {"blue": [[1.0, 5]]},
                   {"blue": 5, "red": 0},
                   teams={"blue": ["254", "1678", "8033"],
                          "red": ["971", "604", "1323"]}, label="qm14")
        con.execute("INSERT INTO frames(match_key,file,t_clean,t_source)"
                    " VALUES (?,?,?,?)", ("2026caclv_qm14", "f_000001.jpg", 1.0, 1.0))
        fid = con.execute("SELECT id FROM frames").fetchone()[0]

        names = {i: n for i, n in enumerate(CLASSES)}
        rows = rows_from_boxes([(0, 0, 40, 40, 0.9, 1, 3),
                                (5, 5, 45, 45, 0.8, 2, 4)], names, source="m1")
        write_rows(con, fid, rows)
        con.commit()
        check("detections attach to a frame",
              con.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 2)
        check("and the API's join finds them",
              con.execute("SELECT COUNT(*) FROM detections d JOIN frames f"
                          " ON f.id=d.frame_id WHERE f.match_key=?",
                          ("2026caclv_qm14",)).fetchone()[0] == 2)

        # No scorer exists, so identity must record nothing rather than
        # assigning the one blue track to whichever team sorts first.
        assigned = identify.assign_tracks(con, "2026caclv_qm14", "blue")
        check("with no scorer, a track is left unnamed",
              assigned == {3: None})
        check("and no team is written to the detection",
              con.execute("SELECT COUNT(*) FROM detections WHERE team IS NOT NULL")
              .fetchone()[0] == 0)

        # Deleting a match's frames must take its detections with them, or a
        # re-export would leave detections pointing at frames that are gone.
        con.execute("DELETE FROM frames WHERE match_key=?", ("2026caclv_qm14",))
        con.commit()
        check("detections go when their frames do",
              con.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0)
        con.close()


def test_serving_export():
    """A copy that a read-only host can actually read.

    The working database is WAL, and a WAL database has to create its `-shm`
    companion before anything can read it -- a strictly read-only connection
    included. `cp` one onto a host that mounts its data directory read-only,
    which deploy/frc-harvest.service does on purpose, and the API starts
    cleanly and then answers every request with "attempt to write a readonly
    database". Confirmed as an unprivileged user against a 0555 directory,
    which is what systemd's ReadOnlyPaths= produces.

    So what is asserted here is the property that makes read-only serving
    work: no WAL, and nothing beside the file.
    """
    import sqlite3
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "work.db"
        con = connect(src)
        build(con, {"videos": {}})
        con.close()
        check("the working database is WAL, which is why this is needed",
              sqlite3.connect(src).execute("PRAGMA journal_mode").fetchone()[0]
              == "wal")

        dest = export_for_serving(Path(tmp) / "serve" / "scouting.db", src=src)
        mode = sqlite3.connect(dest).execute("PRAGMA journal_mode").fetchone()[0]
        check("the exported copy is not WAL", mode == "delete")
        check("and has no sidecar files for a reader to have to create",
              sorted(p.name for p in dest.parent.iterdir()) == ["scouting.db"])
        check("it opens read-only", sqlite3.connect(
            f"file:{dest}?mode=ro", uri=True).execute(
            "SELECT count(*) FROM matches").fetchone()[0] == 0)

        # Exporting twice must not leave the first attempt's WAL behind, which
        # would put a -wal file next to the copy and defeat the whole point.
        export_for_serving(dest, src=src)
        check("re-exporting leaves nothing stale beside it",
              sorted(p.name for p in dest.parent.iterdir()) == ["scouting.db"])

    try:
        export_for_serving(Path(tmp) / "x.db", src=Path(tmp) / "missing.db")
        check("exporting a database that is not there is refused", False)
    except SystemExit:
        check("exporting a database that is not there is refused", True)


def main() -> int:
    for fn in (test_cuts, test_clustering, test_crop_bands, test_formats,
               test_format_tuning, test_district_catalogue,
               test_audio_frames, test_audio_cues, test_audio_recovery,
               test_stream_alignment, test_scoreboard,
               test_download_options, test_labels, test_render, test_identify,
               test_ledger_and_picking, test_competitive_filter, test_audit,
               test_packaging, test_no_unbound_globals, test_sharding, test_db,
               test_serving_export, test_live_counter, test_live_rows,
               test_detect_rows, test_detect_writes, test_api_stays_stdlib,
               test_ball_counting, test_shot_attribution, test_hub_geometry, test_scrimmage_scoreboard,
               test_nothing_to_verify_against, test_counting_model_dataset,
               test_model_must_name_its_classes):
        fn()
    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
