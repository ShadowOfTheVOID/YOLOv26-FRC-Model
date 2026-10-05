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


class _Rows(list):
    def tolist(self):
        return list(self)


class _FakeBoxes:
    def __init__(self, rows):
        self.xyxy = _Rows([list(r[:4]) for r in rows])
        self.conf = _Rows([r[4] for r in rows])
        self.cls = _Rows([r[5] for r in rows])
        self.id = _Rows([r[6] for r in rows]) if rows else None

    def __len__(self):
        return len(self.xyxy)


class _FakeResult:
    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows)
        self.orig_img = None


class _FakeModel:
    """Stands in for an ultralytics model: class names, and a track() that
    yields scripted per-frame detections. Everything but the network runs."""

    def __init__(self, names, frames):
        self.names = names
        self.frames = frames

    def track(self, **kw):
        for rows in self.frames:
            yield _FakeResult(rows)


def _frames_of(names, robots, balls, n):
    """Robot and ball paths as per-frame detection rows for a _FakeModel."""
    index = {v: k for k, v in names.items()}
    frames = []
    for f in range(n):
        rows = []
        for tid, path in robots.items():
            for ff, (alliance, (x, y, w, h)) in path:
                if ff == f:
                    rows.append((x, y, x + w, y + h, 0.9,
                                 index[f"robot_{alliance}"], tid))
        for tid, path in balls.items():
            for ff, (x, y, w, h) in path:
                if ff == f:
                    rows.append((x, y, x + w, y + h, 0.9, index["fuel"], tid))
        frames.append(rows)
    return frames


def test_robot_in_a_pile_is_not_shooting():
    """A robot driving through fuel was credited with 42 shots and 41 misses.

    Measured on 2026nhdur qm7 with the first scouting model: the robot plowing
    the centre pile in auto. The balls it passed sat still while it drove away
    -- clear of the robot, so "shots" -- and each missed its hub. A shot has to
    move itself, fast: a robot-width inside 0.3 s. These paths are that case,
    in miniature, next to one real shot from the same moving robot.
    """
    drive = {1: [(f, ("blue", (100.0 + 5 * f, 300.0, 80.0, 60.0))) for f in range(70)]}
    pile = {}
    for i in range(5):
        bx = 190.0 + 50 * i                      # balls on the floor in its path
        start = int(max(0, (bx - 180) // 5))     # a fresh id as the robot reaches it
        pile[20 + i] = [(f, (bx, 345.0, 10.0, 10.0)) for f in range(start, start + 40)]
    events, c = _play_shots(drive, pile)
    shots = sum(r["shots"] for r in c.per_robot.values())
    check("balls a robot drives away from are not its shots", shots == 0)
    # The last ball is still beside the robot when the clip ends: carried.
    check("and are counted as not launched, so it can be read",
          c.ignored.get("not_launched", 0) == 4 and c.ignored["carried"] == 1)

    # Pushed along at robot speed: 1.5 px a frame against an 80 px robot is
    # about half a width per second at 30 fps -- a shove, not a launch.
    shoved = {30: [(f, (185.0 + 1.5 * (f - 5), 345.0, 10.0, 10.0)) for f in range(5, 70)]}
    events, c = _play_shots(_still(1, "blue", (100.0, 300.0, 80.0, 60.0), 80), shoved)
    check("a ball rolled away slowly is not a shot",
          sum(r["shots"] for r in c.per_robot.values()) == 0)

    # One robot boxed twice (0.58 and 0.37 on the held-out match) must be one
    # track to the counter, or its shots split between two "robots".
    from tbavid.shooting import one_box_per_robot
    kept = one_box_per_robot({5: ("blue", (623.0, 348.0, 60.0, 47.0), 0.58),
                              9: ("blue", (626.0, 352.0, 55.0, 40.0), 0.37),
                              7: ("red", (487.0, 410.0, 73.0, 85.0), 0.44)})
    check("two boxes on one robot become one, the confident one",
          set(kept) == {5, 7})
    kept = one_box_per_robot({1: ("blue", (100.0, 300.0, 80.0, 60.0), 0.5),
                              2: ("blue", (150.0, 300.0, 80.0, 60.0), 0.4)})
    check("robots side by side are both kept", set(kept) == {1, 2})

    # The report after a real run crashed on the new reason and lost every
    # result; each ignore reason must have a line.
    ev, c = _play_shots(drive, pile)
    check("the report names the not-launched balls",
          any("never moved itself" in line for line in c.report()))

    # Robots losing their tracker id: qm7 had 16099, 15856, 16580, 21187
    # appear mid-match, each a new "robot" with its own tally.
    from tbavid.shooting import RobotNumbers
    rn = RobotNumbers()
    a = rn.assign({1283: ("blue", (100.0, 300.0, 80.0, 60.0)),
                   2615: ("red", (500.0, 300.0, 80.0, 60.0))}, 0.0)
    b = rn.assign({1283: ("blue", (105.0, 300.0, 80.0, 60.0))}, 0.5)
    c2 = rn.assign({1283: ("blue", (110.0, 300.0, 80.0, 60.0)),
                    16580: ("red", (540.0, 310.0, 80.0, 60.0))}, 1.5)
    check("robots are numbered 1, 2 in the order they appear",
          set(a) == {1, 2} and a[1][0] == "blue")
    check("a new tracker id where a lost robot was keeps that robot's number",
          set(c2) == {1, 2} and rn.recovered == 1)
    d = rn.assign({21187: ("blue", (900.0, 300.0, 80.0, 60.0))}, 2.0)
    check("a new id far from any lost robot is a new robot", set(d) == {3})
    e = rn.assign({30000: ("blue", (500.0, 300.0, 80.0, 60.0))}, 2.2)
    check("a lost red robot's number is not given to a blue one",
          set(e) == {4})
    f = rn.assign({40000: ("red", (540.0, 310.0, 80.0, 60.0))}, 9.0)
    check("nor to anything after the robot has been gone too long",
          set(f) == {5})

    # Windows are frame counts chosen at 30 fps; at 60 fps they must double
    # or a ball missing for 0.07 s is already "gone".
    c30, c60 = ShotCounter(HUB), ShotCounter(HUB, fps=60.0)
    check("frame windows keep their length in time at 60 fps",
          (c60.vanish_frames, c60.robot_memory, c60.stitch_frames)
          == (2 * c30.vanish_frames, 2 * c30.robot_memory, 2 * c30.stitch_frames))
    from tbavid.count import BallCounter
    b30, b60 = BallCounter(HUB), BallCounter(HUB, fps=60.0)
    check("and the plain counter's too, since it is the score at a scrimmage",
          (b60.vanish_frames, b60.reacquire_frames)
          == (2 * b30.vanish_frames, 2 * b30.reacquire_frames))

    bot = _still(1, "blue", BLUE_BOT, 260)

    # A ball carried in the hopper for 2 s and then shot. The first launch
    # rule timed its 0.3 s from first sighting and threw this shot away.
    carried = [(f, (134.0, 310.0, 12.0, 12.0)) for f in range(60)]
    flown = _fly(50, 134.0, 310.0, 600.0, 420.0, 20, start=60)[50]
    events, c = _play_shots(bot, {50: carried + flown})
    r = c.per_robot.get(1, {})
    check("a ball shot after riding in the hopper is still a shot",
          r.get("shots") == 1 and r.get("missed") == 1)

    # A "flight" that wanders on for seconds, as chains through the qm7 piles
    # did, and ends in a hub: a miss at 2.5 s, not a make 5 s later.
    out_ = _fly(51, 134.0, 300.0, 250.0, 380.0, 10)[51]
    wander = [(10 + i, (250.0 + 3.0 * i, 380.0 - 1.8 * i, 12.0, 12.0))
              for i in range(150)]
    events, c = _play_shots(bot, {51: out_ + wander})
    r = c.per_robot.get(1, {})
    check("a shot still out of a hub 2.5 s after launch is a miss",
          r.get("shots") == 1 and r.get("missed") == 1 and r.get("made", 0) == 0
          and c.expired == 1)

    # 1058 driving hard through the qm7 pile: a ball pushed ahead at robot
    # speed covers a robot-width in 0.3 s but never leaves the robot behind.
    fast = {1: [(f, ("blue", (100.0 + 10 * f, 300.0, 80.0, 60.0))) for f in range(60)]}
    pushed = {60: [(f, (185.0 + 10 * f, 330.0, 12.0, 12.0)) for f in range(40)]
                  + [(40 + i, (585.0 + 2 * i, 330.0, 12.0, 12.0)) for i in range(20)]}
    events, c = _play_shots(fast, pushed)
    check("a ball pushed ahead of a fast robot is not a shot",
          sum(r["shots"] for r in c.per_robot.values()) == 0)
    # And the reverse at the same speed: still balls the fast robot drives
    # away from. Relative-only travel counted these (1058: 76 shots, 76 misses).
    left = {}
    for i in range(4):
        bx = 200.0 + 60 * i
        start = int(max(0, (bx - 180) // 10))
        left[70 + i] = [(f, (bx, 345.0, 10.0, 10.0)) for f in range(start, start + 40)]
    events, c = _play_shots(fast, left)
    check("still balls a fast robot drives away from are not shots",
          sum(r["shots"] for r in c.per_robot.values()) == 0)

    # The real thing, fired while driving: still a shot, still a miss.
    events, c = _play_shots(drive, _shot(40, 700, 100, start=10, x0=160.0))
    r = c.per_robot.get(1, {})
    check("a real shot from a moving robot still counts",
          r.get("shots") == 1 and r.get("missed", 0) + r.get("wrong_hub", 0) == 1)


def test_models_that_can_count_and_shoot():
    """What each command demands of a model -- and no more than it needs.

    count.py used to refuse any model without hub classes, even with both hub
    boxes handed to it. The first trained model is fuel-only, so that refusal
    was all that stood between it and a working counter.
    """
    from tbavid.count import model_can_count, run_source
    from tbavid.shooting import model_can_shoot, parse_teams, run_shots

    fuel_only = {0: "fuel"}
    five = dict(enumerate(CLASSES))

    check("a fuel-only model can count once it is handed the hub boxes",
          model_can_count(fuel_only, HUB) is None)
    check("...and without them it says to pass the boxes, not just no",
          "--hub-blue" in (model_can_count(fuel_only, None) or ""))
    check("a model with no fuel class cannot count at all",
          "no fuel class" in (model_can_count({0: "robot_blue"}, HUB) or ""))
    check("a model with hub classes can still learn the hubs itself",
          model_can_count(five, None) is None)

    # The whole counting loop, fed by the stand-in: a fuel-only model and two
    # drawn boxes, which is the scrimmage set-up.
    model = _FakeModel(fuel_only,
                       _frames_of(fuel_only, {}, _fly(1, 50, 150, 348, 150, 20), 60))
    out = run_source(model, None, lambda h: BallCounter(h), hubs=HUB)
    check("run_source counts with a fuel-only model and given hubs",
          "error" not in out and out["counter"].totals["blue"] == 1)

    check("shots refuse a model with no robot classes, and say why",
          "robot classes" in (model_can_shoot(fuel_only, HUB) or ""))
    check("the five-class model can attribute shots",
          model_can_shoot(five, HUB) is None)

    bot = _still(1, "blue", BLUE_BOT, 80)
    balls = _shot(10, 344, 144)
    balls.update(_shot(11, 600, 420, start=30))
    model = _FakeModel(five, _frames_of(five, bot, balls, 80))
    events = []
    out = run_shots(model, None, hubs=HUB, fps=30.0, on_event=events.append)
    s = out["counter"].per_robot.get(1, {}) if "error" not in out else {}
    check("run_shots attributes a make and a miss to the robot, end to end",
          (s.get("shots"), s.get("made"), s.get("missed")) == (2, 1, 1))
    # Wall-clock time would be milliseconds here; match time is the frame
    # over the source rate, and a shot ~20 frames in is ~0.7 s into the match.
    check("with fps given, shot times are match time, not processing time",
          len(events) == 2 and min(e["t"] for e in events) > 0.5)

    # The tracker's ids count fuel too: the first real run labelled robot 1307
    # "R1283", and it was read as a misread team number. Robots are numbered
    # from 1 in the order they appear, whatever id the tracker gave them.
    bots = _still(1283, "blue", BLUE_BOT, 80)
    bots.update(_still(2615, "red", RED_BOT, 80))
    model = _FakeModel(five, _frames_of(five, bots, _shot(10, 344, 144), 80))
    out = run_shots(model, None, hubs=HUB, fps=30.0)
    check("robots are numbered 1, 2, ... not by tracker id",
          "error" not in out and set(out["counter"].per_robot) == {1}
          and out["counter"].per_robot[1]["alliance"] == "blue")

    model = _FakeModel(fuel_only, _frames_of(fuel_only, {}, balls, 80))
    out = run_shots(model, None, hubs=HUB, fps=30.0)
    check("run_shots with a fuel-only model refuses rather than reporting nothing",
          "robot classes" in out.get("error", ""))

    check("--teams reads several track ids per team",
          parse_teams("3=254, 7=254,5=1678") == {3: 254, 7: 254, 5: 1678})
    try:
        parse_teams("3:254")
        check("a malformed --teams is refused", False)
    except ValueError:
        check("a malformed --teams is refused", True)


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


def test_robot_autolabel_rules():
    """The rules that turn YOLOE's robot proposals into labels, or drop a frame.

    Boxes are the ones measured on the real 2026nhdur frame. What matters most
    is the gate: a frame with robots found but not all labelled teaches the
    detector that the rest are floor, so it must be dropped whole.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from autolabel_robots import (alliance, has_robots, merge, screen, verdict,
                                  yolo_lines)

    red_a = (1488, 285, 1616, 346, 0.51)
    red_b = (1526, 211, 1626, 264, 0.34)
    both = (1494, 212, 1628, 347, 0.44)       # spans the two red robots
    hub = (1251, 12, 1437, 340, 0.30)         # the red hub, h/w 1.8
    wall = (1650, 30, 1915, 315, 0.14)        # alliance wall and ladder
    blue = (307, 238, 425, 304, 0.69)
    blue2 = (414, 285, 531, 353, 0.13)
    why = {d: r for d, r in screen([red_a, red_b, both, hub, wall, blue, blue2])}
    check("real robots pass", all(why[d] is None for d in (red_a, red_b, blue, blue2)))
    check("the hub is too tall to be a robot", why[hub] == "tall")
    check("a box around two robots is not a robot", why[both] == "spans two")
    check("the wall is ~10x the frame's robots", why[wall] == "too big")
    check("size needs others to compare with", screen([wall])[0][1] is None)
    # The KEEP frame from the first real previews: two hubs, the blue ladder
    # around robot 1307, and three real robots. With the hubs in the median
    # the ladder passed; it must be judged against robots only.
    hub_b, hub_r = (493, 10, 690, 425, 0.22), (1265, 10, 1440, 330, 0.25)
    ladder = (170, 147, 343, 348, 0.3)
    r1307, r7674 = (267, 227, 343, 307, 0.12), (380, 205, 493, 275, 0.68)
    r1058 = (1575, 283, 1688, 367, 0.50)
    why = {d: r for d, r in screen([hub_b, hub_r, ladder, r1307, r7674, r1058])}
    check("the ladder is too big once the hubs are out of the median",
          why[ladder] == "too big")
    check("and the near and far robots beside it are kept",
          all(why[d] is None for d in (r1307, r7674, r1058)))
    check("a box ending above the field line is the stands",
          screen([blue], field_top=320)[0][1] == "above field")

    # The red hub's top, painted as a robot in the user's preview: only one
    # other robot in the frame, and one is enough to compare with.
    hub_top, r2648 = (1253, 85, 1435, 300, 0.2), (943, 155, 1063, 252, 0.4)
    why = {d: r for d, r in screen([hub_top, r2648])}
    check("a hub top 3.4x the only robot is too big", why[hub_top] == "too big"
          and why[r2648] is None)
    near, far = (1175, 322, 1300, 380, 0.4), (1527, 211, 1608, 263, 0.5)
    check("but a near robot 1.7x a far one is not",
          all(r is None for _, r in screen([near, far])))
    # With fuel greyed YOLOE also boxes parts of robots.
    part = (456, 285, 531, 333, 0.15)
    why = {d: r for d, r in screen([blue2[:4] + (0.17,), part, red_a, blue])}
    check("a box inside a more confident robot box is a duplicate",
          why[part] == "duplicate" and why[blue2[:4] + (0.17,)] is None)
    check("and so is a less confident box around a confident one",
          dict(screen([(400, 270, 545, 365, 0.1), blue2, red_a]))[(400, 270, 545, 365, 0.1)]
          == "duplicate")

    from autolabel_robots import fuel_mask
    hsv = np.array([[[28, 200, 220], [28, 40, 200], [115, 200, 150], [20, 120, 60]]], np.uint8)
    check("fuel is greyed; floor, bumpers and shadowed fuel below the gate are not",
          fuel_mask(hsv).tolist() == [[True, False, False, False]])

    dup = (1490, 286, 1615, 347, 0.23)       # same robot from another tile
    check("NMS keeps the confident copy",
          merge([dup, red_a, blue]) == [blue, red_a])

    def band(hue, sat, val, n=100, grey=0):
        px = [(hue, sat, val)] * n + [(0, 10, 120)] * grey
        return np.array(px, np.uint8).reshape(-1, 1, 3)
    check("a blue bumper is blue", alliance(band(115, 180, 150))[0] == "robot_blue")
    check("red wraps round the hue circle",
          alliance(band(175, 180, 150))[0] == "robot_red"
          and alliance(band(4, 180, 150))[0] == "robot_red")
    # Robot 69: a navy bumper that compression turns grey-black. Guessing
    # would put a blue robot in the red class for the whole match.
    check("a crushed navy bumper is unknown, not guessed",
          alliance(band(118, 40, 40))[0] is None)
    check("a few coloured pixels on grey floor decide nothing",
          alliance(band(115, 180, 150, n=3, grey=97))[0] is None)
    mixed = np.concatenate([band(115, 180, 150, n=40), band(2, 180, 150, n=30)])
    check("a blue robot beside a red ramp is not a clear call",
          alliance(mixed)[0] is None)

    four = [("robot_blue", blue), ("robot_blue", blue2),
            ("robot_red", red_a), ("robot_red", red_b)]
    check("four readable robots is a training frame", verdict(four, 0, 4) is None)
    check("an unreadable robot no longer drops the frame -- it is painted out",
          verdict(four, 1, 4) is None)
    check("and it counts towards --min-robots",
          verdict(four[:1], 1, 2) is None and verdict(four[:1], 0, 2) is not None)
    check("two found drops the frame", verdict(four[:2], 0, 4) is not None)
    check("four of one alliance means one is not a robot",
          verdict([("robot_red", red_a)] * 4, 0, 4) is not None)

    line = yolo_lines([("robot_red", (100, 50, 300, 150, 0.5))], 1000, 500)[0]
    check("labels are class, centre and size, normalised",
          line == "2 0.200000 0.200000 0.200000 0.200000")
    # Painting out a robot of unknown alliance: grey where it was, the
    # labelled robot it overlaps left intact, and its fuel boxes gone with it.
    from autolabel_robots import PAINT, drop_covered, paint, strip_robots
    img = np.full((100, 200, 3), 50, np.uint8)
    out = paint(img, [(10, 10, 60, 60, 0.3)], [(40, 40, 90, 90, 0.5)])
    check("the unknown robot is painted grey", (out[20, 20] == PAINT).all())
    check("the labelled robot it overlaps is not", (out[50, 50] == 50).all())
    check("the source image is not modified", (img[20, 20] == 50).all())
    check("boxes off the edge are clipped, not an error",
          (paint(img, [(-5, -5, 250, 30, 0.1)], [])[0, 199] == PAINT).all())
    fuel = "0 0.100000 0.200000 0.02 0.02\n0 0.800000 0.800000 0.02 0.02\n"
    check("fuel centred in a painted robot goes, fuel elsewhere stays",
          drop_covered(fuel, [(10, 10, 60, 60, 0.3)], 200, 100)
          == "0 0.800000 0.800000 0.02 0.02\n")
    check("--restore strips robots and keeps fuel and hubs",
          strip_robots("0 .5 .5 .1 .1\n1 .2 .2 .1 .1\n2 .3 .3 .1 .1\n4 .9 .9 .2 .2\n")
          == "0 .5 .5 .1 .1\n4 .9 .9 .2 .2\n")
    check("fuel-only labels have no robots",
          not has_robots("0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n"))
    check("a second run sees the robots it wrote",
          has_robots("0 .5 .5 .1 .1\n" + line + "\n"))


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


class _FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class _FakeSock:
    """Collects what the sender sends; hands back queued replies."""
    def __init__(self, fail=False):
        self.sent = []
        self.replies = []
        self.fail = fail

    def sendto(self, data, addr):
        if self.fail:
            raise OSError("Network is unreachable")
        self.sent.append((data, addr))

    def recvfrom(self, n):
        if not self.replies:
            raise BlockingIOError
        return self.replies.pop(0), ("10.0.100.5", 8411)


def test_hub_feed_protocol():
    """The counter's half of bioarena's Hub FUEL Counter Feed (spec 4 and 5).

    bioarena makes the AUTO winner call at T+23.000 s from whatever counts it
    holds then, so a ball that is waiting for the next heartbeat instead of
    being sent does not decide the winner. And bioarena drops anything that
    breaks its acceptance rules, silently from the counter's side -- so the
    sender is held to them here with the same rules the receiver applies.
    """
    import json
    from tbavid import hubfeed as HF

    clock, sock = _FakeClock(), _FakeSock()
    s = HF.FeedSender(("10.0.100.5", 8411), session="c1f3a9d2", sock=sock,
                      clock=clock)
    check("the first heartbeat goes out at once", s.heartbeat())
    first = json.loads(sock.sent[-1][0])
    check("the datagram carries both hubs, v 1, session and seq",
          first == {"v": 1, "session": "c1f3a9d2", "seq": 1, "red": 0,
                    "blue": 0} and sock.sent[-1][1] == ("10.0.100.5", 8411))
    clock.t += 0.05
    check("no heartbeat before 100 ms", not s.heartbeat() and len(sock.sent) == 1)

    # THE latency rule: a score goes out on the frame it was confirmed, not
    # at the next heartbeat, and age_ms runs from the frame's CAPTURE time --
    # here 38 ms before the send -- not from when the code got to it.
    s.score("red", 1, captured_at=clock.t - 0.038)
    msg = json.loads(sock.sent[-1][0])
    check("a score is sent immediately, between heartbeats",
          len(sock.sent) == 2 and msg["red"] == 1 and msg["seq"] == 2)
    check("age_ms is measured from the capture timestamp", msg["age_ms"] == 38)
    s.score("blue", 3, captured_at=clock.t)
    msg = json.loads(sock.sent[-1][0])
    check("a clump scores several at once, and red is carried along",
          (msg["red"], msg["blue"], msg["seq"]) == (1, 3, 3))
    s.score("blue", 0, clock.t)
    s.score("blue", -2, clock.t)
    check("a count can never be sent going down",
          len(sock.sent) == 3 and s.counts["blue"] == 3)
    try:
        s.score("green", 1, clock.t)
        check("an unknown hub is refused", False)
    except ValueError:
        check("an unknown hub is refused", True)
    check("every datagram fits bioarena's 512-byte limit",
          len(HF.encode("x" * 32, 2 ** 40, 10 ** 6, 10 ** 6, 10 ** 6,
                        "y" * 200)) <= 512)
    try:
        HF.FeedSender(session="bad session!", sock=sock)
        check("a session outside [A-Za-z0-9_-]{1,32} is refused", False)
    except ValueError:
        check("a session outside [A-Za-z0-9_-]{1,32} is refused", True)
    check("sessions are fresh per start",
          HF.new_session() != HF.new_session()
          and HF.SESSION_RE.match(HF.new_session()) is not None)

    # A cable out on the field must not stop the counting: the counts are
    # cumulative, so the next datagram that gets through carries them all.
    down = HF.FeedSender(sock=_FakeSock(fail=True), clock=clock)
    down.score("red", 2, clock.t)
    check("a send error is recorded, not raised",
          down.send_errors == 1 and down.counts["red"] == 2)

    # The reply is shown, never depended on (spec 4.4).
    check("no reply means not linked", not s.linked())
    clock.t += 0.004
    sock.replies.append(json.dumps({"v": 1, "seq": 3,
                                    "match_state": "AUTO_PERIOD"}).encode())
    sock.replies.append(b"not json")
    check("replies are read without blocking, junk skipped",
          s.poll_replies() == 1 and s.linked())
    check("and the echoed seq gives the round trip", s.rtt_ms == 4.0)
    clock.t += 1.5
    check("the link goes down after a second of silence", not s.linked())
    check("targets parse with or without a port",
          HF.parse_target("10.0.100.5") == ("10.0.100.5", 8411)
          and HF.parse_target("127.0.0.1:9000") == ("127.0.0.1", 9000))


def test_hub_feed_receiver_rules():
    """bioarena's acceptance rules (spec 4.5) and restart arithmetic (6.2).

    `Receiver` is the stand-in `run.py hubfeed-listen` runs before the field
    computer exists. If it accepted what bioarena drops, the link would look
    fine on the bench and score nothing at the field.
    """
    import json
    from tbavid import hubfeed as HF

    clock = _FakeClock()
    rx = HF.Receiver("10.0.100.21", clock=clock)

    def d(**kw):
        m = {"v": 1, "session": "aa", "seq": 1, "red": 0, "blue": 0}
        m.update(kw)
        return json.dumps(m).encode()

    ok = "10.0.100.21"
    check("a datagram from another address is dropped (a robot's VLAN)",
          rx.accept(d(), "10.1.14.5") == (False, "unknown source"))
    check("v other than 1 is dropped", rx.accept(d(v=2), ok)[0] is False)
    check("a missing field is dropped",
          rx.accept(json.dumps({"v": 1, "session": "aa", "seq": 1,
                                "red": 0}).encode(), ok)[0] is False)
    check("junk is dropped", rx.accept(b"{", ok)[0] is False)
    check("oversize is dropped", rx.accept(b" " * 600, ok)[0] is False)
    check("the first good one is accepted", rx.accept(d(), ok)[0])
    check("and the counter is online", rx.online())
    check("a duplicate seq is dropped",
          rx.accept(d(), ok) == (False, "duplicate or reordered"))
    rx.accept(d(seq=5, red=4, blue=1, age_ms=40), ok)
    check("a reordered (older) seq is dropped",
          rx.accept(d(seq=4, red=3, blue=1), ok)[0] is False)
    check("a count going backwards is dropped",
          rx.accept(d(seq=6, red=2, blue=1), ok) == (False, "count went backwards"))
    check("age_ms is read from datagrams where a count rose", rx.age_ms == 40)
    rx.accept(d(seq=7, red=4, blue=1, age_ms=900), ok)
    check("and not from heartbeats", rx.age_ms == 40)

    # Match boundaries are bioarena's: ResetMatch baselines the running totals.
    rx.reset_match()
    rx.accept(d(seq=8, red=6, blue=1), ok)
    check("match counts are relative to ResetMatch",
          rx.match_counts() == {"red": 2, "blue": 0})
    # A counter restarted mid-match loses only what it missed while down.
    ok_, what = rx.accept(d(session="bb", seq=1, red=1, blue=0), ok)
    check("a new session is a restart, not a drop",
          ok_ and what == "hub counter restarted" and rx.restarts == 1)
    check("and what the old session scored this match is kept",
          rx.match_counts() == {"red": 3, "blue": 0})
    check("a new session may start its seq anywhere, even lower",
          rx.accept(d(session="bb", seq=2, red=1, blue=2), ok)[0]
          and rx.match_counts() == {"red": 3, "blue": 2})
    rx.reset_match()
    check("a restart before ResetMatch carries nothing into the next match",
          rx.match_counts() == {"red": 0, "blue": 0})
    status = rx.status()
    check("the reply echoes seq and has the spec's fields",
          status["seq"] == 2 and {"match_state", "shift", "hub_active",
                                  "match_count", "credited",
                                  "auto_count"} <= set(status))
    clock.t += 1.01
    check("a second with nothing accepted is offline", not rx.online())

    # End to end: whatever the sender produces, the receiver accepts.
    sock = _FakeSock()
    s = HF.FeedSender(sock=sock, clock=clock)
    rx2 = HF.Receiver()
    s.heartbeat()
    s.score("blue", 2, clock.t)
    s.score("red", 1, clock.t)
    clock.t += 0.2
    s.heartbeat()
    results = [rx2.accept(data, "10.0.100.21") for data, _ in sock.sent]
    check("everything the sender sends, bioarena's rules accept",
          all(r[0] for r in results)
          and rx2.match_counts() == {"red": 1, "blue": 2})


def test_hub_crossing_counter():
    """The live hub counter: yellow area across the funnel outline.

    Ported from experiments/area_hub_count.py because it is the only counter
    here that commits on the frame the ball crosses; BallCounter holds every
    score 400 ms, twice the feed's p99 budget. These fix the rules, not the
    accuracy -- the accuracy is measured on a real hub or not at all.
    """
    from tbavid import hubcount as HC

    square = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
    check("point in polygon",
          HC.point_in_poly(square, (150, 150))
          and not HC.point_in_poly(square, (250, 150))
          and not HC.point_in_poly(square, (150, 99)))
    try:
        HC.parse_poly("1,2,3,4")
        check("an outline needs three points", False)
    except ValueError:
        check("an outline needs three points", True)
    check("outlines parse", HC.parse_poly("1,2, 3,4,5,6") ==
          [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    def fly(counter, path, area=280.0):
        return [counter.update([(x, y, area)]) for x, y in path]

    # A ball dropping into the mouth: counted on the frame it crosses, which
    # is the latency the spec asks for -- not when it vanishes below.
    c = HC.CrossingCounter(square, ball_area=280.0)
    rises = fly(c, [(150, 40), (150, 60), (150, 80), (150, 105), (150, 130)])
    rises.append(c.update([]))
    check("a ball into the hub counts on the crossing frame",
          rises == [0, 0, 0, 1, 0, 0] and c.reported == 1)

    c = HC.CrossingCounter(square, ball_area=280.0)
    fly(c, [(150, 60), (150, 85), (150, 110)], area=3 * 280.0)
    check("a drum shooter's clump of three counts three", c.reported == 3)

    # An outline counts balls moving DOWN into it and ignores outward
    # crossings: on Einstein 1 blue, 60-100 s, the signed rule saw 161 in and
    # 171 out against 108 real balls -- bounces about the hood, not scores.
    c = HC.CrossingCounter(square, ball_area=280.0)
    fly(c, [(x, 150) for x in range(60, 241, 20)])
    check("a ball sideways across the mouth is not a score", c.reported == 0)
    c = HC.CrossingCounter(square, ball_area=280.0)
    fly(c, [(150, 240), (150, 215), (150, 190), (150, 165)])
    check("a ball rising into the outline from below is not a score",
          c.reported == 0)
    c = HC.CrossingCounter(square, ball_area=280.0)
    fly(c, [(150, 60), (150, 85), (150, 110), (150, 112), (150, 95), (150, 80)])
    check("a ball in then back out stays counted, the exit only noted",
          c.reported == 1 and c.exits == 1 and c.owed == 0)

    # The experiment's signed rule, which exit lines still use: a pass-over
    # is reported on entry, its exit owed against the next ball in.
    c = HC.CrossingCounter(square, ball_area=280.0, signed=True)
    fly(c, [(x, 150) for x in range(60, 241, 20)])
    check("signed: a pass-over is reported once, its exit owed",
          c.reported == 1 and c.net == 0 and c.owed == 1)
    rises = fly(c, [(150, 60), (150, 85), (150, 110)])
    check("signed: the next real ball settles it without a second report",
          sum(rises) == 0 and c.reported == 1 and c.owed == 0)
    fly(c, [(170, 60), (170, 85), (170, 110)])
    check("signed: after which balls count again", c.reported == 2)

    # One ball is learned from the crossings: balls at the mouth measured
    # 1.5-2.3x the still ball, and 56 of Einstein 1's 85 AUTO crossings on
    # blue were counted as two.
    c = HC.CrossingCounter(square, ball_area=280.0)
    for i in range(HC.LEARN_WARMUP + 5):
        fly(c, [(110 + 5 * i, 60), (110 + 5 * i, 85), (110 + 5 * i, 110)],
            area=2 * 280.0)
    check("before the warm-up a blob of two balls' area counts two, "
          "after it the blob size is learned as one ball",
          c.reported == 2 * HC.LEARN_WARMUP + 5)
    c = HC.CrossingCounter(square, ball_area=280.0, learn=False)
    for i in range(HC.LEARN_WARMUP + 5):
        fly(c, [(110 + 5 * i, 60), (110 + 5 * i, 85), (110 + 5 * i, 110)],
            area=2 * 280.0)
    check("without learning every one counts two",
          c.reported == 2 * (HC.LEARN_WARMUP + 5))
    check("a blob rounds up to the next ball only at 0.65 of one",
          (HC.balls_for(1.6 * 280, 280), HC.balls_for(1.7 * 280, 280),
           HC.balls_for(0.2 * 280, 280)) == (1, 2, 1))

    c = HC.CrossingCounter(square, ball_area=280.0)
    fly(c, [(150, 150), (152, 152), (151, 160)])
    check("a ball first seen already inside did not cross in", c.reported == 0)
    c = HC.CrossingCounter(square, ball_area=280.0)
    fly(c, [(150, 60), (150, 85), (150, 110)], area=20.0)
    check("a speck under the size floor is ignored", c.reported == 0)
    c = HC.CrossingCounter(square, ball_area=280.0)
    fly(c, [(150, 20), (150, 40), (150, 60), (150, 80)])
    check("a ball that never reaches the mouth is not counted", c.reported == 0)

    # Two balls side by side do not steal each other's crossings.
    c = HC.CrossingCounter(square, ball_area=280.0)
    for y in (60, 85, 110):
        c.update([(130, y, 280.0), (170, y, 280.0)])
    check("two balls in together count two", c.reported == 2)

    # Sizes scale off the ball: a close camera has much bigger balls.
    small = HC.region_of(square, 272.0, 1920, 1080)
    big = HC.region_of(square, 272.0 * 16, 1920, 1080)
    check("the search margin is the experiment's 60 px at its ball size",
          small == (40, 40, 260, 260))
    check("and four times wider for a ball four times wider",
          big == (0, 0, 440, 440))
    check("a region never leaves the frame",
          HC.region_of(square, 272.0 * 16, 300, 300)[2:] == (300, 300))
    try:
        HC.CrossingCounter(square, 0)
        check("a zero ball area is refused", False)
    except ValueError:
        check("a zero ball area is refused", True)

    # age_ms must run from capture: a V4L2 buffer stamp is on the same clock
    # as time.monotonic(), a file position is not.
    check("a plausible driver capture stamp is used",
          HC.frame_time(1000_000.0 - 30.0, 1000.0) == 1000.0 - 0.030)
    check("a file position is not mistaken for one",
          HC.frame_time(5000.0, 1000.0) == 1000.0
          and HC.frame_time(0.0, 1000.0) == 1000.0)

    h = HC.Health()
    for i in range(31):
        h.frame(10.0 + i / 30.0, 10.0 + i / 30.0 + 0.012)
    check("health measures the camera's rate and the lag",
          round(h.fps()) == 30 and round(h.lag_ms()) == 12)
    tally = HC.HubTally(HC.setup_from_flags(
        "0", {"red": square, "blue": square}, {}, 280.0))
    check("the info line fits bioarena's 64 characters",
          len(HC.info_line({str(i): h for i, _ in enumerate(range(9))},
                           tally)) <= 64)


def test_hub_multi_camera_setup():
    """Several cameras, several outlines per hub, combined into one count.

    From in front a ball that clips the rim and drops behind the hub looks
    like a score (Einstein 1: 832 net entries over the red hood, 415 real),
    so the way to accuracy is cameras close to each hub -- more than one per
    hub, at different distances. The feed still carries one number per hub
    that must never go down.
    """
    from tbavid import hubcount as HC

    sq = [[100, 100], [200, 100], [200, 200], [100, 200]]
    cfg = {"combine": {"red": "sum", "blue": "median"},
           "cameras": [
               {"name": "red-left", "source": "0", "ball_area": 1800,
                "zones": [{"hub": "red", "outline": sq}]},
               {"name": "red-right", "source": "1", "ball_area": 900,
                "zones": [{"hub": "red", "outline": "100,100,200,100,200,200"}]},
               {"name": "blue-a", "source": "2", "ball_area": 300,
                "zones": [{"hub": "blue", "outline": sq, "name": "ba"}]},
               {"name": "blue-b", "source": "3", "ball_area": 300,
                "zones": [{"hub": "blue", "outline": sq, "name": "bb"}]},
               {"name": "blue-c", "source": "4", "ball_area": 300,
                "zones": [{"hub": "blue", "outline": sq, "name": "bc"}]}]}
    setup = HC.setup_from_dict(cfg)
    check("a setup file parses: five cameras, both hubs",
          len(setup.cameras) == 5 and setup.hubs() == ["red", "blue"])
    check("each camera keeps its own ball size",
          [z.counter.ball_area for z in setup.zones("red")] == [1800.0, 900.0])

    t = HC.HubTally(setup)
    zr = setup.zones("red")
    zb = {z.name: z for z in setup.zones("blue")}
    zr[0].counter.reported, zr[1].counter.reported = 3, 2
    check("sum: two cameras on different chutes add up", t.value("red") == 5)
    check("and the whole rise is reported once", t.rise("red") == 5
          and t.rise("red") == 0)
    zr[1].counter.reported = 4
    check("a later rise on one camera reports only the new balls",
          t.rise("red") == 2 and t.sent["red"] == 7)

    # Three cameras on the same balls: one misses, one double-counts.
    zb["ba"].counter.reported, zb["bb"].counter.reported = 10, 12
    zb["bc"].counter.reported = 30
    check("median: the odd camera out is outvoted", t.value("blue") == 12)
    t.rise("blue")
    zb["bc"].counter.reported = 60
    check("and its runaway count does not move the hub", t.rise("blue") == 0)
    setup.combine["blue"] = "max"
    check("max takes the camera that missed fewest", t.value("blue") == 60)
    setup.combine["blue"] = "median"

    # Never-decreasing in, never-decreasing out, under every rule.
    import random as _r
    rng = _r.Random(7)
    ok = True
    for how in HC.COMBINE:
        setup.combine["blue"] = how
        for z in zb.values():
            z.counter.reported = 0
        tt = HC.HubTally(setup)
        last = 0
        for _ in range(300):
            rng.choice(list(zb.values())).counter.reported += rng.randint(0, 3)
            v = tt.value("blue")
            ok &= v >= last
            last = v
    check("sum, max and median of rising counts never fall", ok)

    def refused(c, why):
        try:
            HC.setup_from_dict(c)
            check(why, False)
        except ValueError:
            check(why, True)

    one = lambda **kw: {"cameras": [dict({"name": "a", "source": "0",
                                          "ball_area": 300, "zones": [
                                              {"hub": "red", "outline": sq}]},
                                         **kw)]}
    refused({"cameras": []}, "a setup with no cameras is refused")
    refused(one(ball_area=0), "a camera without a measured ball is refused")
    check("but accepted while measuring it, which is how it gets one",
          HC.setup_from_dict(one(ball_area=0), measuring=True).cameras[0]
          .ball_area == 0)
    refused(one(zones=[{"hub": "green", "outline": sq}]),
            "a zone for a hub that is not red or blue is refused")
    refused(one(zones=[{"hub": "red", "outline": [[1, 2], [3, 4]]}]),
            "an outline of two points is refused")
    refused(one(zones=[]), "a camera with no zones is refused")
    two = one()
    two["cameras"].append(dict(two["cameras"][0], name="b"))
    refused(two, "one device used by two cameras is refused")
    refused(dict(one(), combine={"red": "average"}),
            "an unknown combine rule is refused")

    # The single-camera flags still mean what they did.
    flags = HC.setup_from_flags("0", {"red": sq, "blue": sq}, {}, 280.0)
    check("--red/--blue on one --source are one camera, two zones",
          len(flags.cameras) == 1 and len(flags.cameras[0].zones) == 2)
    split = HC.setup_from_flags("0", {"red": sq, "blue": sq},
                                {"blue": "1"}, 280.0)
    check("--blue-source gives blue its own camera",
          sorted(c.source for c in split.cameras) == ["0", "1"])


def test_hub_calibration_and_gui_helpers():
    """Blur correction, the setup file round trip, and the window's helpers.

    The blur fraction that best matched the scoreboard was different on
    every Einstein match (0 on 4, 0.2-0.3 on 5, 0.5-0.7 on 1), so it is a
    per-camera setting chosen against a hand count, not a constant.
    """
    import math
    from tbavid import hubcount as HC
    from tbavid import hubapp as G

    sq = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
    A1 = 300.0
    d = math.sqrt(4 * A1 / math.pi)
    # One ball smeared 2 diameters along x: area A1 + d*2d, length 3d.
    L = 3 * d
    streak_area = A1 + d * (L - d)
    cov = (L * L / 16.0, d * d / 16.0, 0.0)          # uniform ellipse spread

    cov = (d * d / 16.0, L * L / 16.0, 0.0)          # smeared along y

    def cross(blur, area, cov_):
        c = HC.CrossingCounter(sq, A1, blur)
        c.update([(200.0, 60.0, area) + cov_])
        c.update([(200.0, 90.0, area) + cov_])            # outside, falling
        c.update([(200.0, 120.0, area) + cov_])           # crosses in
        return c.reported

    check("uncorrected, one smeared ball counts as its area in balls",
          cross(0.0, streak_area, cov) == HC.balls_for(streak_area, A1) > 1)
    check("fully corrected, the same streak is one ball",
          cross(1.0, streak_area, cov) == 1)
    check("a blob with no shape recorded is never corrected",
          cross(1.0, streak_area, ()) == HC.balls_for(streak_area, A1))
    try:
        HC.CrossingCounter(sq, A1, 1.5)
        check("a blur fraction over 1 is refused", False)
    except ValueError:
        check("a blur fraction over 1 is refused", True)

    # A replay of recorded blobs gives each hub's count at a blur setting.
    cam = HC.setup_from_dict({"cameras": [{
        "name": "c", "source": "x.mp4", "ball_area": A1, "blur": 0.5,
        "remove_static": True,
        "zones": [{"hub": "red", "outline": [list(p) for p in sq]}]}]}).cameras[0]
    frames = [{cam.zones[0].name: [(200.0, 60.0 + 30 * i, streak_area) + cov]}
              for i in range(3)]
    check("replay_counts: no correction", HC.replay_counts(cam, frames, 0.0)
          == {"red": HC.balls_for(streak_area, A1)})
    check("replay_counts: full correction", HC.replay_counts(cam, frames, 1.0)
          == {"red": 1})

    # What the window builds is what hubfeed --setup reads, and back again.
    setup = HC.setup_from_dict({"combine": {"red": "max", "blue": "sum"},
                                "cameras": [{"name": "c", "source": "0",
                                             "ball_area": A1, "blur": 0.3,
                                             "remove_static": True, "fps": 60,
                                             "zones": [{"name": "z", "hub": "red",
                                                        "outline": [list(p) for p in sq]}]}]})
    again = HC.setup_from_dict(HC.setup_to_dict(setup))
    c0, c1 = setup.cameras[0], again.cameras[0]
    check("a setup survives saving and loading",
          (c1.blur, c1.remove_static, c1.fps, c1.ball_area, again.combine["red"])
          == (0.3, True, 60.0, A1, "max")
          and c1.zones[0].counter.poly == c0.zones[0].counter.poly
          and c1.zones[0].counter.blur == 0.3)
    plain = HC.setup_from_dict({"cameras": [{"name": "c", "source": "0",
                                             "ball_area": A1, "zones": [
                                                 {"hub": "red", "outline": [list(p) for p in sq]}]}]})
    check("a camera with no blur given gets the measured default",
          plain.cameras[0].blur == HC.DEFAULT_BLUR)
    plain.cameras[0].blur = 0.0
    check("and a chosen blur of 0 survives saving and loading",
          HC.setup_from_dict(HC.setup_to_dict(plain)).cameras[0].blur == 0.0)
    # The old page saved every camera's blur slider, 0 unless moved: in a
    # file from before the current rules a 0 is the old default, not a choice.
    old = HC.setup_from_dict({"cameras": [{"name": "c", "source": "0",
                                           "ball_area": A1, "blur": 0,
                                           "zones": [{"hub": "red", "outline":
                                                      [list(p) for p in sq]}]}]})
    check("an old setup's blur 0 is read as unset, and says so",
          old.cameras[0].blur == HC.DEFAULT_BLUR and len(old.notes) == 1
          and "'c'" in old.notes[0])
    old2 = HC.setup_from_dict({"cameras": [{"name": "c", "source": "0",
                                            "ball_area": A1, "blur": 0.5,
                                            "zones": [{"hub": "red", "outline":
                                                       [list(p) for p in sq]}]}]})
    check("an old setup's chosen non-zero blur is kept",
          old2.cameras[0].blur == 0.5 and not old2.notes)

    # The window's pure helpers.
    check("a 1080p frame is shrunk to the view, a small one is not",
          G.fit_scale(1920, 1080) == 0.5 and G.fit_scale(640, 480) == 1.0)
    check("a click on the half-size preview is stored in frame pixels",
          G.to_frame(100, 50, 0.5) == (200.0, 100.0))
    check("and drawn back where it was clicked",
          G.to_canvas([(200.0, 100.0)], 0.5) == [100.0, 50.0])
    check("camera numbers are cameras, paths are recordings",
          not G.is_file_source("0") and G.is_file_source("hub.mp4")
          and not G.is_file_source("rtsp://cam/1"))
    check("names are made unique", G.next_name(["cam0", "cam02"], "cam0") == "cam03"
          and G.next_name([], "cam0") == "cam0")
    check("an empty setup says to add a camera",
          G.problems(G.new_setup()) == ["Add a camera (or a recording) first."])
    todo = G.problems({"cameras": [{"name": "c", "source": "0", "zones": []}]})
    check("a camera with no outline and no ball size says both",
          len(todo) == 2 and "outline" in todo[0] and "ball" in todo[1])


def test_hub_ui_controller_and_web():
    """The logic both front ends share, and the web page's HTTP surface.

    The web page is a view over HubController, so what a button does is
    tested here, without a browser. The web server controls cameras
    and lists the disk, so it must refuse other hosts and non-JSON posts.
    """
    import json
    import os
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from tbavid import hubapp as A
    from tbavid import hubweb as W

    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "hub practice.mp4")
        open(video, "wb").close()
        ctl = A.HubController()
        cam = ctl.add_camera(video)
        check("a recording is added under its file name",
              cam["name"] == "hub practice" and cam["zones"] == [])
        try:
            ctl.add_camera(video)
            check("the same source twice is refused", False)
        except ValueError:
            check("the same source twice is refused", True)
        try:
            ctl.add_camera(os.path.join(tmp, "missing.mp4"))
            check("a recording that does not exist is refused", False)
        except ValueError:
            check("a recording that does not exist is refused", True)
        z = ctl.add_zone(cam["name"], "red",
                         [[1, 1], [50, 1], [50, 50], [50, 50]])
        check("a double-click's repeated corner is dropped",
              len(z["outline"]) == 3 and z["hub"] == "red")
        try:
            ctl.add_zone(cam["name"], "red", [[1, 1], [2, 2]])
            check("an outline of two corners is refused", False)
        except ValueError:
            check("an outline of two corners is refused", True)
        c = ctl.update_camera(cam["name"], {"name": "red-exit", "ball_area": "1800",
                                            "blur": 0.3, "remove_static": True})
        check("camera settings are updated, the name included",
              (c["name"], c["ball_area"], c["blur"], c["remove_static"])
              == ("red-exit", 1800.0, 0.3, True))
        try:
            ctl.update_camera("red-exit", {"blur": 2})
            check("a blur over 1 is refused", False)
        except ValueError:
            check("a blur over 1 is refused", True)
        st = ctl.state()
        check("the state a front end draws is plain JSON",
              json.loads(json.dumps(st))["cfg"]["cameras"][0]["name"] == "red-exit"
              and st["problems"] == [] and not st["running"])
        path = ctl.save(os.path.join(tmp, "cams.json"))
        from tbavid.hubcount import load_setup
        check("what the controller saves, hubfeed --setup runs",
              load_setup(path).cameras[0].zones[0].hub == "red")
        ls = A.list_dir(tmp)
        check("the recording picker lists videos",
              ls["videos"] == ["hub practice.mp4"])

        # A Twitch or YouTube page is a stream, found by host, not a file.
        from tbavid import hubcount as HC
        check("Twitch and YouTube pages are streams",
              all(HC.is_stream_page(u) for u in (
                  "https://www.twitch.tv/firstinspires", "twitch.tv/frc",
                  "https://www.youtube.com/watch?v=abc", "youtu.be/abc")))
        check("files, camera numbers and other URLs are not",
              not any(HC.is_stream_page(u) for u in (
                  "match.mp4", "0", "rtsp://cam/1", "https://twitch.tv.evil.com/x",
                  "/Users/x/twitch.tv.mp4")))
        check("a stream is named after its channel",
              HC.stream_name("https://www.twitch.tv/firstinspires_newton/")
              == "firstinspires_newton"
              and HC.stream_name("https://www.twitch.tv/videos/414792150")
              == "vod-414792150")
        check("source kinds", (A.source_kind("0"), A.source_kind(video),
                               A.source_kind("twitch.tv/frc"))
              == ("camera", "file", "stream"))
        check("network cameras are wireless, streams are not",
              A.source_kind("rtsp://192.168.1.50:554/stream1") == "wireless"
              and A.source_kind("http://192.168.1.60:8080/video") == "wireless"
              and A.source_kind("https://www.twitch.tv/frc") == "stream"
              and not HC.is_network_camera("https://youtu.be/x"))
        wc = ctl.add_camera("rtsp://admin:hunter2@192.168.1.50:554/stream1")
        check("a Wi-Fi camera is named after its address",
              wc["name"] == "wifi-50" and ctl.state()["kinds"]["wifi-50"] == "wireless")
        check("and its password never reaches the on-screen log",
              not any("hunter2" in m[2] for m in ctl.messages_since(0))
              and A.redact("rtsp://admin:hunter2@h:554/s") == "rtsp://***@h:554/s")
        ctl.remove_camera("wifi-50")
        sc = ctl.add_camera("https://www.twitch.tv/firstinspires")
        check("a stream is added without looking it up (it may be offline)",
              sc["name"] == "firstinspires"
              and ctl.state()["kinds"]["firstinspires"] == "stream")
        ctl.remove_camera("firstinspires")

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), W.make_handler(ctl))
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"

        def call(path_, body=None, headers=None):
            req = urllib.request.Request(base + path_, data=body,
                                         headers=headers or {})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()

        code, page = call("/")
        check("the page is served", code == 200 and b"<title>Hub Counter</title>" in page)
        code, body = call("/api/state")
        check("the state is served as JSON",
              code == 200 and json.loads(body)["cfg"]["cameras"][0]["name"]
              == "red-exit")
        code, _ = call("/api/combine", json.dumps({"hub": "blue", "how": "max"}).encode(),
                       {"Content-Type": "application/json"})
        check("a JSON post changes the setup",
              code == 200 and ctl.cfg["combine"]["blue"] == "max")
        code, _ = call("/api/combine", json.dumps({"hub": "blue", "how": "avg"}).encode(),
                       {"Content-Type": "application/json"})
        check("a bad value is a 400, not a crash", code == 400)
        code, _ = call("/api/stop", b"hub=blue", {"Content-Type":
                                                  "application/x-www-form-urlencoded"})
        check("a form post (what another website could send) is refused",
              code == 415)
        code, _ = call("/api/state", headers={"Host": "evil.example:8790"})
        check("a request for another host name is refused (DNS rebinding)",
              code == 403)
        code, _ = call("/frame.jpg?cam=red-exit")
        check("no picture yet is a 404", code == 404)
        code, page = call("/board")
        check("the scoreboard page is served",
              code == 200 and b"<title>Hub Scoreboard</title>" in page)
        code, body = call("/api/board")
        b = json.loads(body) if code == 200 else {}
        check("a stopped counter's board says so and shows zeros",
              b.get("source") == "counter" and b.get("alert") == "counter stopped"
              and b.get("score") == {"red": 0, "blue": 0})
        httpd.shutdown()
        httpd.server_close()


def test_frc_fms_sender():
    """Fuel events to frc-fms (github.com/arnan-bajaj/frc-fms).

    frc-fms adds timestamped events and buckets them into periods by their
    wall-clock time, so a ball must carry the time it was SEEN, and an event
    lost in a failed POST is a ball lost -- unlike bioarena's cumulative feed,
    where the next datagram carries everything.
    """
    from tbavid import fmslink as FL

    check("a URL target is frc-fms, host:port is bioarena",
          FL.is_fms_target("http://k@10.0.0.2:8000") and not FL.is_fms_target("10.0.100.5:8411"))
    check("the vision key rides in the URL",
          FL.split_target("http://s3cret@192.168.1.10:8000/") == ("http://192.168.1.10:8000", "s3cret"))
    try:
        FL.FmsSender("http://192.168.1.10:8000", start=False)
        check("a URL without the vision key is refused", False)
    except ValueError:
        check("a URL without the vision key is refused", True)

    calls, fail = [], [True]
    def post(url, body, key):
        calls.append((url, body, key))
        if fail[0]:
            raise OSError("connection refused")
        return {"record": None}
    mono = [100.0]
    s = FL.FmsSender("http://k@fms:8000", post=post, clock=lambda: mono[0],
                     wall=lambda: 1_700_000_000.0 + mono[0], start=False)
    s.score("red", 2, captured_at=99.5)          # seen 0.5 s ago
    s.score("blue", 1, captured_at=100.0)
    check("an event carries the wall-clock time it was seen, not sent",
          s._buf["red"] == [[1_700_000_099.5, 2]])
    check("a failed POST keeps every event queued",
          not s.flush() and s.pending() == 2 and s.send_errors == 1 and not s.linked())
    fail[0] = False
    s.score("red", 1, captured_at=100.0)
    ok = s.flush()
    url, body, key = calls[-1]
    check("the next POST carries all of it, in frc-fms's shape, with the key header",
          ok and url == "http://fms:8000/api/vision/events" and key == "k"
          and body["events"] == {"red": [[1_700_000_099.5, 2], [1_700_000_100.0, 1]],
                                 "blue": [[1_700_000_100.0, 1]]}
          and body["source"] == "live" and s.pending() == 0 and s.linked())
    check("cumulative counts stay for the page and the board",
          s.counts == {"red": 3, "blue": 1})
    import tbavid.hubcount as HC2
    from types import SimpleNamespace as NS
    health = {"c": NS(fps=lambda: 60.0, lag_ms=lambda: 4.0)}
    tally = NS(zones={"red": [], "blue": []}, setup=NS(combine={}))
    line = HC2.status_line(s, health, tally)
    check("the console names frc-fms, not 'bioarena ?', when posting to it",
          "frc-fms took it" in line and "bioarena" not in line)

    # frc-fms's control page: its Vision panel shows "<fps> fps <counter>"
    # or `error` in red, and its "vision ok" pill needs every hub without
    # one. Posting only `counter` showed "undefined fps" and never went red.
    check("before the counter reports, no fps is claimed",
          "fps" not in body["status"]["red"] and body["status"]["red"]["counter"])
    sq = [[100, 100], [200, 100], [200, 140], [100, 140]]
    setup = HC2.setup_from_dict({"cameras": [
        {"name": "rc", "source": "0", "ball_area": 300,
         "model": {"weights": "m.pt"}, "zones": [{"hub": "red", "outline": sq}]},
        {"name": "bc", "source": "1", "ball_area": 300,
         "zones": [{"hub": "blue", "outline": sq}]}]})
    h = {"rc": NS(fps=lambda: 59.6), "bc": NS(fps=lambda: 60.0)}
    st = HC2.hub_status(setup, h, stale=[])
    check("each hub reports its camera's fps and which counter runs",
          st == {"red": {"fps": 59.6, "counter": "tbavid colour + model"},
                 "blue": {"fps": 60.0, "counter": "tbavid colour"}})
    st = HC2.hub_status(setup, h, stale=["bc"], model_off=["rc"])
    check("a dead camera is an error frc-fms shows in red",
          st["blue"]["error"] == "no frames from bc")
    check("a model dropped as too slow is named, but is not an error",
          "model off" in st["red"]["counter"] and "error" not in st["red"])
    h["bc"] = NS(fps=lambda: 14.0)
    check("a camera under MIN_FPS is an error: counts run low",
          "too slow" in HC2.hub_status(setup, h, stale=[])["blue"]["error"])
    s.hub_status = HC2.hub_status(setup, {"rc": NS(fps=lambda: 59.6),
                                          "bc": NS(fps=lambda: 60.0)}, stale=[])
    s.score("red", 1, captured_at=100.0)
    s.flush()
    sent = calls[-1][1]["status"]
    check("and that is what goes to frc-fms, with the running totals",
          sent["red"]["fps"] == 59.6 and sent["red"]["session_total"] == 4
          and sent["blue"]["counter"] == "tbavid colour")


def test_model_worker_and_fms_combo():
    """The model half's thread, shared by our counter and the frc-fms plugin.

    It must never make the camera wait, must say when it skips frames, and
    must give up when it cannot keep up -- a lagging model held the hub page
    at 52 against colour's 103 on 4 CPU cores.
    """
    import threading
    import time
    from tbavid import hubmodel as HM
    from tbavid import fms_counter as FC

    sq = [[100, 100], [200, 100], [200, 140], [100, 140]]
    box = lambda x, y: (x - 8, y - 8, x + 8, y + 8)
    gate = threading.Event()

    class Eye:
        """Stands in for ModelEye: a ball falling into the hub, one step per
        frame, and only as fast as `gate` lets it."""
        prev = None
        def __init__(self):
            self.y = 20
        def detect(self, frame):
            gate.wait(5)
            self.y += 15
            return ([(box(150, self.y), 0.6)] if self.y < 130 else []), []

    rises = []
    mc = HM.ModelCounter(sq, fps=30.0, min_track=3, vanish=2, reacquire=3)
    w = HM.ModelWorker(Eye(), {"red": mc}, on_rise=lambda n, tag: rises.append((n, tag)))
    gate.set()
    for i in range(20):
        w.offer(None, i / 30, tag=f"cap{i}")
        while w._job is not None:
            time.sleep(0.001)
    time.sleep(0.2)
    check("the worker counts the model's ball and hands back the frame's tag",
          mc.reported == 1 and rises and rises[0][0] == "red"
          and rises[0][1].startswith("cap"))
    gate.clear()
    offs = []
    slow = HM.ModelWorker(Eye(), {"red": HM.ModelCounter(sq)},
                          on_rise=lambda n, tag: None,
                          on_off=lambda o, k, tag: offs.append((o, k)))
    for i in range(HM.MODEL_WARMUP_FRAMES + 5):
        slow.offer(None, i / 30)            # never blocks, model stuck
    gate.set()
    time.sleep(0.3)
    check("offer never blocks; a stuck model's frames are counted as skipped",
          slow.skipped >= HM.MODEL_WARMUP_FRAMES)
    check("and a model that skipped most of them turns itself off",
          slow.off and len(offs) == 1)
    slow.offer(None, 0.0)
    check("after which it takes no more frames", slow.offered == HM.MODEL_WARMUP_FRAMES + 5)

    class Broken:
        prev = None
        def detect(self, frame):
            raise RuntimeError("MPS out of memory")
    errs = []
    b = HM.ModelWorker(Broken(), {}, on_rise=lambda n, tag: None, on_error=errs.append)
    b.offer(None, 0.0)
    time.sleep(0.2)
    check("a model error stops the worker and is reported, not swallowed",
          b.error == "MPS out of memory" and len(errs) == 1)
    for x in (w, slow, b):
        x.close()

    # The frc-fms plugin: same config file keys as ColourCounter, plus a model.
    base = {"hub": "red", "outline": sq, "ball_area": 300}
    try:
        FC.ComboCounter(dict(base))
        check("ComboCounter without a model is refused", False)
    except ValueError:
        check("ComboCounter without a model is refused", True)
    cc = FC.ComboCounter(dict(base, model="fuel_relabel.pt", model_weight=0.4))
    check("ComboCounter reads the model and its share from vision.yaml",
          cc.model["weight"] == 0.4 and cc.model["weights"] == "fuel_relabel.pt")
    check("plain ColourCounter stays colour only", FC.ColourCounter(dict(base)).model is None)
    # The blend inside the plugin, with the halves set by hand.
    from types import SimpleNamespace as NS
    cc.counters = [NS(reported=10)]
    cc.models = [NS(reported=4)]
    cc.worker = NS(off=False, error="")
    check("the plugin reports the blend of its halves (0.4 x 4 + 0.6 x 10)",
          cc._zone_count(0) == 8)
    cc.worker.off = True
    check("and colour alone once its model is off", cc._zone_count(0) == 10)
    live = FC.ComboCounter(dict(base, model="m.pt"))
    t0 = 1_700_000_000.0
    seen = [live._is_offline(t0 + i / 60, wall=t0 + i / 60 + 0.01) for i in range(5)]
    check("frames at camera speed, stamped now, are live: the model may skip",
          not any(seen))
    re = FC.ComboCounter(dict(base, model="m.pt"))
    seen = [re._is_offline(t0 + i * 0.5, wall=t0 + 1) for i in range(10)]
    check("a recording read faster than real time (rescore.py) waits for the model",
          seen[-1] and re.offline)
    # deploy/frc-fms.vision.yaml is copied over frc-fms's config by the
    # setup steps: every counter it names must exist, and the keys it uses
    # must be ones the plugin reads.
    import re
    example = (Path(__file__).resolve().parent.parent / "deploy" /
               "frc-fms.vision.yaml").read_text()
    named = re.findall(r'counter: "tbavid\.fms_counter:(\w+)"', example)
    check("the frc-fms example config names plugins that exist",
          named and all(hasattr(FC, n) for n in named))
    check("and per hub gives a cams.json setup and camera, which the plugin reads",
          "setup:" in example and "camera:" in example and "fps: 60" in example)
    old = FC.ComboCounter(dict(base, model="m.pt"))
    check("so does one read slower, stamped when it was recorded: a busy CPU "
          "made rescore.py look live and the model was dropped",
          old._is_offline(t0, wall=t0 + 3600) and old.offline)


def test_relabel_video_helpers():
    """The bootstrap labeller's pure parts: tiling, merging, clipping, output.

    Tiles are what fix the halved-frame miss (84% -> 93% of balls in flight
    found on Einstein 4), so they must cover every pixel; a tile that cuts a
    ball in half must not leave it in the image unlabelled.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    import relabel_video as RV

    t = RV.tiles_for(1920, 700)
    covered = all(any(x <= px < x + RV.TILE and y <= py < y + RV.TILE for x, y in t)
                  for px in range(0, 1920, 7) for py in range(0, 700, 7))
    check("tiles cover a 1920x700 main view, inside the frame",
          covered and all(x + RV.TILE <= 1920 and y + RV.TILE <= 700 for x, y in t))
    check("a frame smaller than a tile is one tile", RV.tiles_for(500, 400) == [(0, 0)])
    merged = RV.nms([((0, 0, 20, 20), 0.9), ((1, 1, 21, 21), 0.5), ((100, 100, 120, 120), 0.3)])
    check("overlapping tiles' duplicate of a ball is merged to the surer box",
          merged == [((0, 0, 20, 20), 0.9), ((100, 100, 120, 120), 0.3)])
    keep, grey = RV.clip_labels([(0, (10, 10, 30, 30), "model"),       # inside
                                 (0, (630, 10, 650, 30), "model"),     # half out
                                 (0, (636, 10, 656, 30), "motion"),    # mostly out
                                 (1, (600, 0, 700, 100), "robot")],    # robot, 40% in
                                0, 0, 640, 640)
    check("a ball at least half inside a tile is kept, shifted",
          [(c, b) for c, b, _ in keep] == [(0, (10, 10, 30, 30)), (0, (630, 10, 640, 30))])
    check("a ball mostly cut off is greyed, not left unlabelled; a cut robot is dropped",
          grey == [(636, 10, 640, 30)])
    import tempfile
    import train as TR
    tmp = Path(tempfile.mkdtemp())
    (tmp / "mix.yaml").write_text(
        "path: /workspace/frc\ntrain: [dataset-scout/images/train, dataset_einstein/images/train]\n"
        "val: [dataset-scout/images/val, dataset_einstein/images/val]\n")
    check("train.py finds the labels of every set a mixed yaml trains on",
          [str(p) for p in TR.label_dirs(tmp / "mix.yaml")] ==
          ["/workspace/frc/dataset-scout/labels/train", "/workspace/frc/dataset_einstein/labels/train",
           "/workspace/frc/dataset-scout/labels/val", "/workspace/frc/dataset_einstein/labels/val"])
    rel = tmp / "dataset_relabel-fuel"
    (rel / "images" / "train").mkdir(parents=True)
    (rel / "labels" / "train").mkdir(parents=True)
    (rel / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (rel / "dataset.yaml").write_text("path: dataset_relabel-fuel\ntrain: images/train\n"
                                      "val: images/train\n")
    TR.repoint_dataset(rel / "dataset.yaml")
    check("a relative path: (subset_classes.py from --src dataset_relabel) is made absolute",
          f"path: {rel.resolve()}" in (rel / "dataset.yaml").read_text()
          and any(d.rglob("*.txt") for d in TR.label_dirs(rel / "dataset.yaml")))
    row = [(0, (x, 50, x + 14, 64), "model") for x in range(0, 140, 20)]
    shirt = (0, (300, 40, 350, 75), "model")
    near = [(0, (x, 600, x + 30, 630), "model") for x in range(0, 300, 40)]
    kept = RV.drop_oversized(row + [shirt] + near + [(1, (400, 0, 600, 200), "robot")])
    check("a shirt-sized 'ball' among 14 px balls is dropped; big near balls and robots stay",
          shirt not in kept and len(kept) == len(row) + len(near) + 1)
    # --from-scraper: samples on a clean render's back-to-back clock, never
    # just after a cut, where "two frames earlier" is another shot.
    ranges = [(10.0, 12.0), (50.0, 51.0)]
    clean = RV.clean_sample_times(ranges, 0.5)
    check("clean-render samples skip the first 0.1 s after each cut",
          clean == [0.1, 0.6, 1.1, 1.6, 2.1, 2.6])
    check("raw samples are the same moments on the download's clock",
          RV.raw_sample_times(ranges, 0.5) == [10.1, 10.6, 11.1, 11.6, 50.1, 50.6])
    known = {"2026nhdur_qm7_abc": "val"}
    check("a match keeps the side it already has in the dataset being extended",
          RV.split_for("2026nhdur_qm7_abc", known, 0.0) == "val")
    new = [RV.split_for(f"2026x_qm{i}_k{i}", {}, 0.2) for i in range(200)]
    check("a new match goes by its hash: stable, about --val-frac of them to val",
          new == [RV.split_for(f"2026x_qm{i}_k{i}", {}, 0.2) for i in range(200)]
          and 20 <= new.count("val") <= 60)
    ds = Path(tempfile.mkdtemp())
    for split, name in (("train", "2026a_qm1_x_000010.jpg"), ("val", "2026a_qm2_y_000020.jpg")):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "images" / split / name).write_bytes(b"")
    check("known splits are read off prepare_dataset's frame names",
          RV.known_splits(ds) == {"2026a_qm1_x": "train", "2026a_qm2_y": "val"})
    manifest = {"videos": {
        "2026a_qm1_x": {"status": "ok", "yt_key": "x", "clean_path": "/v/a.mp4",
                        "analysis": {"keep_ranges": [[5, 9]]},
                        "crop": {"x": 0, "y": 40, "w": 1920, "h": 900}},
        "2026a_qm3_z": {"status": "quarantined", "analysis": {"keep_ranges": [[0, 1]]}},
        "2026b_qm4_w": {"status": "ok", "analysis": {"keep_ranges": []}}}}
    jobs = RV.scraper_jobs(manifest, [], RV.known_splits(ds), 0.2, False)
    check("the scraper's usable videos become jobs: quarantined and empty ones skipped",
          [(j["stem"], j["split"], j["crop"]) for j in jobs]
          == [("2026a_qm1_x", "train", (0, 40, 1920, 900))])
    check("--include-quarantined takes them too",
          len(RV.scraper_jobs(manifest, [], {}, 0.2, True)) == 2)
    check("--matches narrows by substring",
          RV.scraper_jobs(manifest, ["2026b"], {}, 0.2, True) == [])
    import subset_classes as SC
    block = "path: /x\ntrain: images/train\nnames:\n  0: fuel\n  1: robot_blue\n  2: robot_red\n"
    check("subset_classes reads the source's class order (block, map and list forms)",
          SC.yaml_names(block) == ["fuel", "robot_blue", "robot_red"]
          and SC.yaml_names("names: {0: fuel, 1: hub_red}") == ["fuel", "hub_red"]
          and SC.yaml_names("names: [robot_red, fuel]") == ["robot_red", "fuel"]
          and SC.index_map(["fuel"], SC.yaml_names("names: [robot_red, fuel]")) == {1: 0})
    moved = Path(tempfile.mkdtemp())
    (moved / "videos").mkdir()
    (moved / "videos" / "2026a_qm1_x.mp4").write_bytes(b"")
    (moved / "raw").mkdir()
    (moved / "raw" / "rawkey.mkv").write_bytes(b"")
    j = RV.resolve_source({"stem": "2026a_qm1_x", "clean": "/Users/old/data/videos/2026a_qm1_x.mp4",
                           "raw": None, "yt_key": "x", "crop": (0, 0, 10, 10)}, moved)
    r = RV.resolve_source({"stem": "2026a_qm2_y", "clean": None, "raw": "/gone/rawkey.mkv",
                           "yt_key": "rawkey", "crop": (0, 0, 10, 10)}, moved)
    check("a moved data folder: clean renders and raw downloads found by name under --data",
          (j["kind"], Path(j["path"]).name, r["kind"], Path(r["path"]).name)
          == ("clean", "2026a_qm1_x.mp4", "raw", "rawkey.mkv"))
    fj = RV.frame_jobs([Path("f/2026a_qm1_x_000030.jpg"), Path("f/2026a_qm1_x_000010.jpg"),
                        Path("f/2026a_qm2_y_000020.jpg"), Path("f/notaframe.jpg")],
                       {"2026a_qm2_y": "val"}, 0.0, [])
    check("exported frames group into one job per match, in frame order, keeping known splits",
          [(j["stem"], j["split"], [f.name for f in j["files"]]) for j in fj]
          == [("2026a_qm1_x", "train", ["2026a_qm1_x_000010.jpg", "2026a_qm1_x_000030.jpg"]),
              ("2026a_qm2_y", "val", ["2026a_qm2_y_000020.jpg"])])
    import numpy as np
    rng = np.random.default_rng(1)
    def balls(h, s_lo, s_hi, n=2000):
        return np.stack([rng.integers(h[0], h[1] + 1, n), rng.integers(s_lo, s_hi + 1, n),
                         rng.integers(200, 256, n)], 1)
    washed = RV.gate_from(balls((27, 31), 26, 90), 40)       # 2026inmis, measured
    rich = RV.gate_from(balls((27, 30), 159, 255), 40)       # Einstein 4, measured
    inside = lambda g, px: g[0][0] <= px[0] <= g[1][0] and px[1] >= g[0][1] and px[2] >= g[0][2]
    check("the colour gate follows a washed-out broadcast's fuel down (S 26-90 passes)",
          inside(washed, (29, 40, 230)) and not inside(RV.gate_from(balls((27, 30), 159, 255), 40), (29, 40, 230)))
    check("and still keeps the Einstein stone border out on either (H 13-19)",
          not inside(washed, (16, 100, 170)) and not inside(rich, (16, 100, 170)))
    check("grey never passes, and too few confident balls means no measured gate",
          washed[0][1] >= 20 and RV.gate_from(balls((27, 31), 26, 90), 5) is None)
    check("labels are YOLO-normalised",
          RV.to_yolo(0, (0, 0, 64, 32), 640, 320) == "0 0.050000 0.050000 0.100000 0.100000")


def test_hub_counter_app():
    """The Hub Counter app (every OS): the hub page as a double-click program, with Quit.

    A double-clicked app has no terminal to Ctrl-C in, so the page must be
    able to stop it; and the bundle must hold only the hub counter, not the
    scraper (its spec is checked here, the build runs in CI on a Mac).
    """
    import os
    import socket
    from tbavid import hubapp as A
    from tbavid import hubweb as W

    ctl = A.HubController()
    quit_calls = []
    ctl.on_quit = lambda: quit_calls.append(1)
    W.ACTIONS["quit"](ctl, {})
    check("the page's Quit stops the counter and shuts the server down",
          quit_calls == [1])

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "apps" / "hubcounter"))
    import main as APP
    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get("HOME")
        os.environ["HOME"] = tmp
        try:
            d = APP.data_dir()
        finally:
            if old is not None:
                os.environ["HOME"] = old
        check("the setup and logs live in ~/Documents/Hub Counter (the app is read-only)",
              d == Path(tmp) / "Documents" / "Hub Counter" and d.is_dir())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        check("a second launch sees the first one running and only reopens the page",
              APP.already_running(port))
    check("and a free port is not mistaken for a running app",
          not APP.already_running(port))

    # In a packaged app sys.executable is the app: "-m yt_dlp" would start a
    # second copy of it. The frozen path must call yt-dlp in-process.
    import types
    from tbavid import hubcount as HC
    calls = []
    class FakeYDL:
        def __init__(self, opts):
            calls.append(opts)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def extract_info(self, url, download=False):
            return {"url": "https://cdn.example/live.m3u8"}
    saved_mod, saved_frozen = sys.modules.get("yt_dlp"), getattr(sys, "frozen", None)
    sys.modules["yt_dlp"] = types.SimpleNamespace(YoutubeDL=FakeYDL)
    sys.frozen = True
    try:
        got = HC.resolve_stream("twitch.tv/somechannel")
    finally:
        if saved_mod is None:
            sys.modules.pop("yt_dlp", None)
        else:
            sys.modules["yt_dlp"] = saved_mod
        if saved_frozen is None:
            del sys.frozen
        else:
            sys.frozen = saved_frozen
    check("inside the app a stream is looked up in-process, not by re-running the app",
          got == "https://cdn.example/live.m3u8" and calls
          and calls[0]["format"] == "best[height<=1080]/best")

    spec = (root / "apps" / "hubcounter" / "HubCounter.spec").read_text()
    check("the app bundles the hub pages and keeps the scraper and torch out",
          "hubweb.html" in spec and "hubboard.html" in spec
          and '"tbavid.tba"' in spec and '"tbavid.pipeline"' in spec and '"torch"' in spec)
    check("and asks macOS for the camera and the local network, or it gets neither",
          "NSCameraUsageDescription" in spec and "NSLocalNetworkUsageDescription" in spec)


def test_hub_scoreboard_view():
    """The live scoreboard: bioarena's score when linked, the camera's otherwise.

    Linked, the big numbers are bioarena's `credited` (spec 4.4) -- the
    match's score, fuel into an inactive hub left out -- not the counter's
    session totals, which run across matches and include dark-hub fuel.
    """
    from tbavid.hubapp import board_view

    reply = {"v": 1, "seq": 9, "match_state": "TELEOP_PERIOD", "match_time_s": 47.3,
             "shift": "SHIFT2", "hub_active": {"red": True, "blue": False},
             "match_count": {"red": 60, "blue": 4}, "credited": {"red": 57, "blue": 0},
             "auto_count": {"red": 12, "blue": 0}}
    b = board_view({"red": 310, "blue": 290}, True, True, reply, [], 40.0, [])
    check("linked: the score is bioarena's credited count, with its clock",
          b["source"] == "bioarena" and b["score"] == {"red": 57, "blue": 0}
          and b["match_time_s"] == 47.3 and b["hub_active"]["blue"] is False
          and b["raw"] == {"red": 310, "blue": 290} and "alert" not in b)
    b = board_view({"red": 310, "blue": 290}, True, False, reply, [], 40.0, [])
    check("a reply from a bioarena that has gone quiet is not shown as the score",
          b["source"] == "counter" and b["score"] == {"red": 310, "blue": 290})
    b = board_view({"red": 3}, True, False, None, ["red-cam"], None, [])
    check("a blind camera is on the board",
          b["alert"] == "no picture from red-cam" and b["score"]["blue"] == 0)
    b = board_view({}, True, True, {"seq": 1}, [], None, [])
    check("a reply without scores falls back to the camera counts",
          b["source"] == "counter")
    b = board_view({"red": 1}, True, True, reply, [], None, [], slow={"red-cam": 19.6})
    check("a camera under 30 fps is flagged: 20 fps measured 25% error, 60 fps 10.5%",
          "red-cam 20 fps" in b.get("alert", ""))
    from tbavid.hubapp import slow_cameras
    from tbavid.hubcount import Health
    fast, slow_h, young = Health(), Health(), Health()
    for i in range(60):
        fast.frame(i / 60.0, i / 60.0 + 0.005)
        slow_h.frame(i / 20.0, i / 20.0 + 0.005)
    for i in range(5):
        young.frame(i / 10.0, i / 10.0)
    check("slow_cameras finds the 20 fps camera only, once it has a second of frames",
          slow_cameras({"a": fast, "b": slow_h, "c": young}) == {"b": 20.0})
    b = board_view({"red": 1}, True, True, reply, [], None, [], practice=True)
    check("practice replies are labelled as the stand-in, not bioarena",
          b["source"] == "practice" and b["score"]["red"] == 57)


def test_hub_exit_line_counter():
    """Counting balls as they come OUT of the hub.

    Every scored ball leaves through an exit, so the exit count is the score,
    and a ball that clips the rim and drops behind the hub -- the error that
    sank the funnel-mouth counts (Einstein 1: 832 net entries over the red
    hood, 415 real) -- never gets there.
    """
    from tbavid import hubcount as HC
    from tbavid import hubapp as A

    line, out = [(100.0, 100.0), (100.0, 200.0)], (160.0, 150.0)   # out = +x
    A1 = 300.0

    def run(path, area=A1, counter=None):
        c = counter or HC.ExitLineCounter(line, out, A1)
        for x, y in path:
            c.update([(float(x), float(y), area)])
        c.update([])
        return c

    c = run([(60, 150), (80, 150), (100 - 0.001, 150), (115, 150), (135, 150)])
    check("a ball rolling out through the exit counts once", c.reported == 1)
    c = run([(140, 150), (120, 150), (100.5, 150), (80, 150)])
    check("a ball going the other way does not count", c.reported == 0)
    c = run([(70, 150), (90, 150), (110, 150), (130, 150), (110, 150),
             (90, 150), (70, 150)])
    check("out and straight back in nets zero (after one report)",
          c.net == 0 and c.reported == 1 and c.owed == 1)
    c = run([(70, 60), (90, 60), (110, 60), (130, 60)])
    check("a ball passing beyond the end of the line does not count",
          c.reported == 0)
    c = run([(70, 150), (90, 150), (110, 150)], area=3 * A1)
    check("a clump of three through the exit counts three", c.reported == 3)
    # A ball exactly on the line counts once, on the frame it leaves it.
    c = run([(80, 150), (100, 150), (100, 150), (120, 150)])
    check("stopping on the line counts once, not twice", c.reported == 1)
    # Fast: 70 px a frame, over twice the ball's width, still caught because
    # the path is tested, not the side each end is on.
    c = HC.ExitLineCounter(line, out, A1)
    c.min_reach = 200
    run([(40, 150), (110, 150)], counter=c)
    check("a fast ball crossing between frames is caught", c.reported == 1)
    for bad, why in ((([(1, 1), (1, 1)], (5, 5)), "a line with one point"),
                     (([(0, 0), (10, 0)], (5, 0)), "an out point on the line")):
        try:
            HC.ExitLineCounter(bad[0], bad[1], A1)
            check(f"{why} is refused", False)
        except ValueError:
            check(f"{why} is refused", True)
    check("segment crossing: through the middle",
          HC.segments_cross((0, 5), (10, 5), (5, 0), (5, 10)))
    check("segment crossing: not reaching the line",
          not HC.segments_cross((0, 5), (4, 5), (5, 0), (5, 10)))

    # Exit lines in a setup file survive a round trip, and replay.
    cfg = {"cameras": [{"name": "c", "source": "x.mp4", "ball_area": A1,
                        "zones": [{"name": "red-exit", "hub": "red",
                                   "line": [list(p) for p in line],
                                   "out": list(out)}]}]}
    setup = HC.setup_from_dict(cfg)
    z = setup.cameras[0].zones[0]
    check("an exit zone is built as an exit line counter",
          z.kind == "exit" and isinstance(z.counter, HC.ExitLineCounter))
    again = HC.setup_to_dict(setup)
    check("and saved back as a line and an out point",
          again["cameras"][0]["zones"][0] == {"name": "red-exit", "hub": "red",
                                              "line": [[100.0, 100.0], [100.0, 200.0]],
                                              "out": [160.0, 150.0]})
    frames = [{"red-exit": [(float(x), 150.0, A1)]} for x in (70, 90, 110, 130)]
    check("replay counts exits too",
          HC.replay_counts(setup.cameras[0], frames, 0.0) == {"red": 1})

    # The website's controller makes one from three clicks.
    ctl = A.HubController()
    ctl.cfg["cameras"].append({"name": "c", "source": "0", "ball_area": A1,
                               "zones": []})
    zz = ctl.add_zone("c", "blue", [[100, 100], [100, 200], [160, 150]], "exit")
    check("three clicks make an exit line",
          zz["line"] == [[100.0, 100.0], [100.0, 200.0]] and zz["out"] == [160.0, 150.0]
          and zz["name"] == "c-blue-exit")
    try:
        ctl.add_zone("c", "blue", [[100, 100], [100, 200]], "exit")
        check("an exit line without its out point is refused", False)
    except ValueError:
        check("an exit line without its out point is refused", True)
    check("the exit is drawn and searched around all three points",
          A.zone_points(zz) == [[100.0, 100.0], [100.0, 200.0], [160.0, 150.0]])
    check("and the setup the website saves runs headless",
          HC.setup_from_dict(ctl.cfg).cameras[0].zones[0].kind == "exit")


def test_ball_tracker_follows_through_the_apex():
    """The renderer's tracker links balls by distance, not box overlap.

    ByteTrack lost flying balls at the top of their arc: a 12-20 px ball
    turning over stops overlapping a straight-line prediction. On 4 s of
    Einstein 4, 19 of 20 flights lost there still had a 0.1-0.8 detection
    where the ball was; ByteTrack took 1 of 25 flights through the apex, this
    took 106. These fix the rules on synthetic arcs.
    """
    from tbavid.trackvis import BallTracker, Coaster

    def box(x, y, w=16):
        return (x - w / 2, y - w / 2, x + w / 2, y + w / 2)

    # A ball thrown up and over: 12 px/frame sideways, rising, turning over at
    # the top, falling -- the case that broke.
    arc = [(100 + 12 * i, 400 - (30 * i - 1.5 * i * i)) for i in range(21)]
    t = BallTracker()
    ids = set()
    for i, (x, y) in enumerate(arc):
        conf = 0.15 if 8 <= i <= 12 else 0.6          # weak at the top
        got = t.update([(box(x, y), conf)])
        ids |= set(got)
    check("one ball over its apex keeps one id", len(ids) == 1)

    # Missed for three frames at the top, found again further along.
    t = BallTracker()
    ids = set()
    for i, (x, y) in enumerate(arc):
        if 9 <= i <= 11:
            t.update([])
            continue
        ids |= set(t.update([(box(x, y), 0.6)]))
    check("and through a three-frame gap", len(ids) == 1)

    # A weak detection alone cannot start a track.
    t = BallTracker()
    check("a weak detection does not start a track",
          t.update([(box(50, 50), 0.12)]) == {})

    # Two balls crossing paths keep their own ids (nearest match first).
    t = BallTracker()
    a_ids, b_ids = set(), set()
    for i in range(10):
        got = t.update([(box(100 + 10 * i, 200), 0.6), (box(100 + 10 * i, 260), 0.6)])
        for tid, bx in got.items():
            (a_ids if bx[1] < 230 else b_ids).add(tid)
    check("two balls side by side stay two tracks",
          len(a_ids) == 1 and len(b_ids) == 1 and a_ids != b_ids)
    # A far bigger object nearby is not taken for the ball.
    t = BallTracker()
    t.update([(box(100, 100), 0.6)])
    got = t.update([(box(104, 100, w=80), 0.9)])
    check("a robot-sized box is not linked to a ball's track",
          1 not in got)

    c = Coaster(coast=4)
    c.update(0, {7: box(100, 100)})
    c.update(1, {7: box(110, 100)})
    _, coasting = c.update(2, {})
    check("a lost moving ball is drawn where it should be",
          7 in coasting and abs((coasting[7][0] + coasting[7][2]) / 2 - 120) < 1e-6)
    for f in range(3, 7):
        _, coasting = c.update(f, {})
    check("but only for a few frames", 7 not in coasting)
    c = Coaster()
    for f in range(6):
        c.update(f, {1: box(100 + (f % 2) * 3, 100)})       # jitter in a pile
    check("a ball jittering in a pile draws no trail", c.moving_trail(1) == [])
    c = Coaster()
    for f in range(6):
        c.update(f, {1: box(100 + 10 * f, 100)})
    check("a ball travelling draws one", len(c.moving_trail(1)) == 6)


def test_hub_feed_needs_no_opencv_to_load():
    """The sender is stdlib only: it runs on whatever laptop is wired into the
    field switch, and CI has no cv2. hubcount keeps cv2 inside functions."""
    import ast
    root = Path(__file__).resolve().parent.parent
    third = {"numpy", "requests", "ultralytics", "torch", "cv2", "PIL"}
    tree = ast.parse((root / "tbavid" / "hubfeed.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names} & third
        elif isinstance(node, ast.ImportFrom) and node.module:
            found |= {node.module.split(".")[0]} & third
    check("hubfeed.py imports no third-party package", not found)
    for rel in ("tbavid/hubcount.py", "tbavid/hubapp.py", "tbavid/hubweb.py",
                "tbavid/hubmodel.py"):
        top = ast.parse((root / rel).read_text()).body
        check(f"{rel} imports cv2, numpy and tkinter only inside functions",
              not any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                      ({a.name.split(".")[0] for a in n.names}
                       & {"cv2", "numpy", "tkinter"}
                       if isinstance(n, ast.Import) else
                       (n.module or "").split(".")[0] in {"cv2", "numpy", "tkinter"})
                      for n in top))
    top = ast.parse((root / "tbavid" / "hubcount.py").read_text()).body
    check("hubcount.py imports cv2 and numpy only inside functions",
          not any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                  ({a.name.split(".")[0] for a in n.names} & {"cv2", "numpy"}
                   if isinstance(n, ast.Import) else
                   (n.module or "").split(".")[0] in {"cv2", "numpy"})
                  for n in top))


def test_hub_model_blend():
    """The fuel model as a second hub counter, blended with the colour one.

    Colour over-counts (balls clipping the rim and dropping behind the hub
    look like scores) and the model under-counts (the tracker drops balls
    fired in streams); on Einstein 1, held out from training, colour was
    9.1% off the official checkpoints, the model 12.1%, their mean 7.8%.
    The blend must keep the feed's rule: never down.
    """
    from tbavid import hubcount as HC
    from tbavid import hubmodel as HM

    sq = [[100, 100], [200, 100], [200, 140], [100, 140]]
    mc = HM.ModelCounter(sq, fps=30.0, min_track=3, vanish=2, reacquire=3)
    box = lambda x, y: (x - 8, y - 8, x + 8, y + 8)
    rises = 0
    for i, y in enumerate(range(20, 130, 15)):          # falls into the hub
        rises += mc.update(i / 30, [(box(150, y), 0.6)])
    for j in range(8):                                   # and is gone
        rises += mc.update((i + 1 + j) / 30, [])
    check("a ball that falls into the hub and vanishes is one score",
          mc.reported == 1 and rises == 1)
    mc2 = HM.ModelCounter(sq, fps=30.0, min_track=3, vanish=2, reacquire=3)
    for i, x in enumerate(range(-40, 360, 20)):          # passes over it
        mc2.update(i / 30, [(box(x, 60), 0.6)])
    for j in range(8):
        mc2.update(1 + j / 30, [])
    check("a ball that flies over the hub and leaves is not", mc2.reported == 0)
    far = HM.ModelCounter(sq)
    check("detections far from the hub are not the hub's",
          far.near(box(150, 120)) and not far.near(box(900, 120)))
    check("weak detections are ignored, as in the tuning",
          HM.KEEP_CONF > HM.DET_CONF)

    check("the crop covers the outline and stays in the frame",
          HM.crop_box(sq, 1920, 1080) == (0, 0, 640, 640)
          and HM.crop_box([[1800, 1000], [1900, 1000], [1900, 1060]], 1920, 1080)
          == (1280, 440, 1920, 1080))
    big = HM.crop_box([[0, 500], [600, 500], [600, 560]], 1920, 1080)
    check("a close camera's big outline gets a bigger crop",
          big[2] - big[0] == 1080)
    check("the model runs at ~30 fps: every other frame of a 60 fps camera",
          HM.model_stride(60) == 2 and HM.model_stride(30) == 1
          and HM.model_stride(0) == 1)

    import random as _r
    rng = _r.Random(3)
    c = m = last = 0
    ok = True
    for _ in range(500):
        c += rng.randint(0, 2)
        m += rng.randint(0, 2)
        v = HM.blend(c, m, 0.4)
        ok &= v >= last
        last = v
    check("a blend of rising counts never falls", ok)
    check("a model that skips most frames is dropped after its warm-up",
          HM.too_slow(HM.MODEL_WARMUP_FRAMES, HM.MODEL_WARMUP_FRAMES // 2)
          and not HM.too_slow(HM.MODEL_WARMUP_FRAMES - 1, HM.MODEL_WARMUP_FRAMES - 1)
          and not HM.too_slow(1000, 100))
    check("and colour alone takes over: weight 0 is the colour count",
          HM.blend(103, 1, 0.0) == 103)
    check("weight 0 is colour alone, 1 the model alone",
          HM.blend(10, 4, 0.0) == 10 and HM.blend(10, 4, 1.0) == 4
          and HM.blend(10, 4, 0.5) == 7)

    cfg = {"cameras": [{"name": "a", "source": "0", "ball_area": 300,
                        "model": {"weights": "fuel_relabel.pt", "weight": 0.4,
                                  "reacquire": 3},
                        "zones": [{"hub": "red", "outline": sq},
                                  {"hub": "blue", "line": [[0, 0], [10, 0]],
                                   "out": [5, 5]}]}]}
    setup = HC.setup_from_dict(cfg)
    cam = setup.cameras[0]
    check("a camera's model entry is read, defaults filled in",
          cam.model["weight"] == 0.4 and cam.model["counter"]["reacquire"] == 3
          and cam.model["counter"]["vanish"] == HM.DEFAULT_MODEL["vanish"])
    check("exit lines stay colour only: the model needs an outline's box",
          [z.name for z in cam.model_zones()] == ["a/red0"]
          and cam.zones[1].weight == 0.0)
    back = HC.setup_from_dict(HC.setup_to_dict(setup)).cameras[0].model
    check("and saved back unchanged", back == cam.model)
    zr = cam.zones[0]
    zr.counter.reported = 10
    check("before the model is attached the zone reads its colour count",
          zr.reported == 10)
    zr.model = HM.ModelCounter(sq)
    zr.model.reported = 5
    check("with it, the zone and the hub read the blend",
          zr.reported == 8 and HC.HubTally(setup).value("red") == 8)
    for bad, why in (({"weight": 0.4}, "a model with no weights file"),
                     ({"weights": "x.pt", "weight": 1.5}, "a weight above 1")):
        try:
            HC.setup_from_dict({"cameras": [dict(cfg["cameras"][0], model=bad)]})
            check(f"{why} is refused", False)
        except ValueError:
            check(f"{why} is refused", True)
    # The web page's side: a model chosen on the page is saved and run.
    import os
    from tbavid import hubapp as A
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "practice.mp4")
        weights = os.path.join(tmp, "fuel_relabel.pt")
        open(video, "wb").close()
        open(weights, "wb").close()
        ctl = A.HubController()
        name = ctl.add_camera(video)["name"]
        ctl.add_zone(name, "red", sq)
        ctl.update_camera(name, {"ball_area": 300})
        c = ctl.update_camera(name, {"model": weights})
        check("choosing a model on the page turns the blend on at the tuned share",
              c["model"] == {"weights": weights, "weight": HM.DEFAULT_WEIGHT})
        c = ctl.update_camera(name, {"model_weight": 0.3})
        check("and its share can be set", c["model"]["weight"] == 0.3)
        try:
            ctl.update_camera(name, {"model_weight": 2})
            check("a share above 1 is refused", False)
        except ValueError:
            check("a share above 1 is refused", True)
        check("the folder listing offers .pt files for the picker",
              A.list_dir(tmp)["models"] == ["fuel_relabel.pt"])
        if A._has_ultralytics():
            check("a model that is there is ready to start", ctl.state()["problems"] == [])
        else:
            check("without Ultralytics the page says why before Start",
                  any("Ultralytics" in p for p in ctl.state()["problems"]))
        os.remove(weights)
        check("a model file that has gone is caught before Start",
              any("not there" in p for p in ctl.state()["problems"]))
        saved = HC.load_setup(ctl.save(os.path.join(tmp, "cams.json")), measuring=True)
        check("what the page saves, hubfeed --setup reads with the model",
              saved.cameras[0].model["weight"] == 0.3)
        c = ctl.update_camera(name, {"model": ""})
        check("turning it off removes it", "model" not in c)
    plain = HC.setup_from_dict({"cameras": [dict(cfg["cameras"][0], model=None)]})
    check("no model entry: colour only, as before",
          plain.cameras[0].model is None and plain.cameras[0].model_zones() == []
          and "model" not in HC.setup_to_dict(plain)["cameras"][0])


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
               test_ball_counting, test_shot_attribution, test_robot_in_a_pile_is_not_shooting,
               test_models_that_can_count_and_shoot, test_hub_geometry, test_scrimmage_scoreboard,
               test_nothing_to_verify_against, test_counting_model_dataset,
               test_robot_autolabel_rules,
               test_model_must_name_its_classes, test_hub_feed_protocol,
               test_hub_feed_receiver_rules, test_hub_crossing_counter,
               test_hub_feed_needs_no_opencv_to_load, test_hub_multi_camera_setup,
               test_hub_calibration_and_gui_helpers, test_hub_ui_controller_and_web,
               test_hub_scoreboard_view, test_relabel_video_helpers, test_frc_fms_sender,
               test_hub_exit_line_counter, test_ball_tracker_follows_through_the_apex,
               test_hub_model_blend, test_model_worker_and_fms_combo,
               test_hub_counter_app):
        fn()
    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  - {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
