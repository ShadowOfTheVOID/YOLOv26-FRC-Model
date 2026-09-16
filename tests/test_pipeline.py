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

from tbavid import ledger as L
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
from tbavid.tba import (TBAClient, is_competitive_event, match_label, pick_unseen,
                        youtube_candidates)

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
    for fn in (test_cuts, test_clustering, test_crop_bands, test_scoreboard,
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
