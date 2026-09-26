#!/usr/bin/env python3
"""Pull FRC match videos from TBA, strip them to main-camera footage, and
export training frames.

    ./run.py pull --count 10        # full unattended run
    ./run.py pull --count 10 --review
    ./run.py review                 # inspect/fix a batch after the fact
    ./run.py status
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tbavid import pipeline
from tbavid.config import VIDEO_DIR, load_config

# Python block-buffers stdout when it is a pipe, so a backgrounded run showed
# nothing at all for minutes while it was in fact working. Progress output is
# the only way to tell a slow stage from a hung one.
sys.stdout.reconfigure(line_buffering=True)


def parse_shard(text):
    """'2/4' -> (1, 4). Human-facing numbering is 1-based; internal is 0-based."""
    if not text:
        return 0, 1
    try:
        a, b = text.split("/")
        idx, total = int(a), int(b)
    except ValueError:
        raise SystemExit(f"--shard wants N/M, e.g. 2/4 (got {text!r})")
    if total < 1 or not (1 <= idx <= total):
        raise SystemExit(f"--shard {text}: need 1 <= N <= M")
    return idx - 1, total


def competitive_only(args, cfg):
    """The flag wins over config.json; config.json defaults to competitive."""
    if getattr(args, "include_noncompetitive", False):
        return False
    return cfg.get("competitive_only", True)


def cmd_pull(args, cfg):
    shard, shards = parse_shard(getattr(args, "shard", None))
    produced = pipeline.fetch(cfg, args.count, retry_failed=args.retry_failed,
                              seed=args.seed, per_event_cap=args.per_event_cap,
                              dry_run=args.dry_run, shard=shard, shards=shards,
                              competitive_only=competitive_only(args, cfg))
    if args.dry_run or not produced:
        return 0
    if args.review:
        print("\n[3/4] review")
        pipeline.review_stage(cfg, port=args.port)
    print("\n[4/4] exporting frames")
    total = pipeline.export(cfg)
    print(f"\ndone: {len(produced)} video(s), {total} frames in data/frames/")
    return 0


def cmd_fetch(args, cfg):
    shard, shards = parse_shard(getattr(args, "shard", None))
    pipeline.fetch(cfg, args.count, retry_failed=args.retry_failed,
                   seed=args.seed, per_event_cap=args.per_event_cap,
                   dry_run=args.dry_run, shard=shard, shards=shards,
                   competitive_only=competitive_only(args, cfg))
    return 0


def cmd_review(args, cfg):
    pipeline.review_stage(cfg, port=args.port)
    print("\nexporting frames")
    total = pipeline.export(cfg)
    print(f"done: {total} frames written")
    return 0


def cmd_export(args, cfg):
    total = pipeline.export(cfg, only=args.only or None, force=args.force)
    print(f"done: {total} frames written")
    return 0


def cmd_scoreboard(args, cfg):
    n = pipeline.rescore(cfg, only=args.only or None)
    print(f"re-read {n} scoreboard(s); run `export --force` to rebuild labels")
    return 0


def cmd_verify(args, cfg):
    pipeline.verify_scores(cfg)
    return 0


def cmd_reprocess(args, cfg):
    n = pipeline.reprocess(cfg, only=args.only or None)
    print(f"\nreprocessed {n} video(s); exporting frames")
    total = pipeline.export(cfg, force=True)
    print(f"{total} frames written")
    return 0


def cmd_merge(args, cfg):
    from pathlib import Path as _P
    stats = pipeline.merge(cfg, _P(args.path))
    for k, v in stats.items():
        print(f"  {k.replace('_', ' '):22} {v}")
    if stats["videos_already_had"]:
        print(f"\n  ! {stats['videos_already_had']} video(s) were already here.")
        print("    With correct --shard settings that should be zero -- check")
        print("    everyone used the same M and a different N.")
    print("\nnow: python3 run.py db sync")
    return 0


def cmd_export_share(args, cfg):
    from pathlib import Path as _P
    counts = pipeline.export_share(cfg, _P(args.path))
    print(f"  frames {counts['frames']}, labels {counts['labels']} -> {args.path}")
    print("  plus manifest.json and seen.json")
    print("\nSend that folder to whoever is collecting; they run:")
    print(f"  python3 run.py merge <folder>")
    return 0


def cmd_db(args, cfg):
    from tbavid import db as dbmod

    # Before connect(), which would put the file back into WAL mode on the way
    # past and undo the one thing this command is for.
    if args.db_action == "export":
        dest = dbmod.export_for_serving(args.to)
        print(f"wrote {dest}\n"
              f"  Serveable from a read-only disk: no WAL, no sidecar files.\n"
              f"  Copy it to the host that runs the API:\n"
              f"    scp {dest} host:/var/lib/frc-harvest/scouting.db\n"
              f"  and see deploy/HOSTING.md.")
        return 0

    con = dbmod.connect()
    if args.db_action in ("build", "sync"):
        print("counts written:", dbmod.build(con))
    if args.db_action == "sync":
        from tbavid.config import tba_key
        from tbavid.tba import TBAClient
        client = TBAClient(tba_key(), cfg["tba_min_interval_s"])
        print("roster rows filled:", dbmod.backfill_teams(con, client))
        print("official scores fetched:", dbmod.fetch_official(con, client))
    if args.db_action == "hub":
        box = [int(v) for v in args.box.split(",")]
        dbmod.set_hub(con, args.event, args.alliance, box)
        print(f"{args.event} {args.alliance} hub = {box}")
    print("totals:", dbmod.summary(con))
    print(f"database: {dbmod.DB_PATH}")
    return 0


def cmd_serve(args, cfg):
    from tbavid import api
    api.serve(args.host, args.port)
    return 0


def cmd_audit(args, cfg):
    pipeline.audit(cfg)
    return 0


def cmd_status(args, cfg):
    pipeline.status(cfg)
    return 0


def cmd_prune(args, cfg):
    freed = pipeline.prune(cfg, raw=not args.clean_only, clean=args.clean or args.all)
    print(f"freed {freed/1e9:.2f} GB")
    return 0


def cmd_cropcheck(args, cfg):
    """Dump proof frames: the full frame with the crop box drawn, and the crop."""
    from tbavid import crop as crop_mod
    from tbavid.config import REVIEW_DIR

    manifest = pipeline.load_manifest()
    target = args.video
    if target is None:
        candidates = [v for v in manifest["videos"].values() if v.get("status") == "ok"]
        if not candidates:
            print("no rendered videos yet -- run `pull` first")
            return 1
        entry = candidates[0]
    else:
        entry = manifest["videos"].get(target)
        if entry is None:
            print(f"unknown video id {target}\n  known ids:")
            for vid in manifest["videos"]:
                print(f"    {vid}")
            return 1

    raw = pipeline._ensure_raw(entry, cfg)
    if raw is None:
        print("no source available for a crop check")
        return 1
    keep = [tuple(r) for r in entry["analysis"]["keep_ranges"]]
    times = crop_mod.sample_times(keep, args.frames)
    out = REVIEW_DIR / "cropcheck"
    written = crop_mod.cropcheck(raw, entry["crop"], out, times)
    print(f"crop: {entry['crop']['filter']}  (top_frac={entry['crop']['top_frac']})")
    for w in written:
        print(f"  {w}")
    print("\nCheck that the score banner is gone AND both hubs are still fully in frame.")
    return 0


def cmd_detect(args, cfg):
    """Run a trained .pt over harvested frames and record the detections."""
    from tbavid import db as dbmod
    from tbavid import detect as det
    from tbavid import identify

    model = det.load(args.weights)
    source = det.model_source(args.weights)
    con = dbmod.connect()

    if args.match:
        keys = [args.match]
    else:
        sql = "SELECT match_key FROM matches"
        params = ()
        if args.event:
            sql += " WHERE event_key=?"
            params = (args.event,)
        sql += " ORDER BY match_key"
        keys = [r[0] for r in con.execute(sql, params)]
    if not keys:
        print("no matches in the database -- run `db sync` first")
        return 1
    if args.limit:
        keys = keys[:args.limit]

    print(f"model: {source}  conf={args.conf}  tracker={args.tracker}")
    print(f"{len(keys)} match(es)\n")
    totals = {"frames": 0, "detections": 0, "missing": 0}
    for mk in keys:
        def progress(n, of, found):
            print(f"\r  {mk}: {n}/{of} frames, {found} detections", end="")

        got = det.detect_match(con, model, mk, source, conf=args.conf,
                               tracker=args.tracker, progress=progress)
        print(f"\r  {mk}: {got['frames']} frames, {got['detections']} detections, "
              f"{got['tracks']} track(s)"
              + (f", {got['missing']} frame file(s) missing" if got["missing"] else ""))
        for k in totals:
            totals[k] += got[k]

        if args.assign:
            # No scorer exists yet, so this reports nothing rather than
            # guessing -- see identify.py. Run anyway, because "the tracks are
            # there and unnamed" is the honest state and worth saying.
            named = 0
            for alliance in ("blue", "red"):
                for track, team in identify.assign_tracks(
                        con, mk, alliance).items():
                    named += 1 if team is not None else 0
            print(f"      identity: {named} track(s) named"
                  + ("" if named else " -- no scorer, so nothing was guessed"))

    print(f"\ntotal: {totals['detections']} detections over "
          f"{totals['frames']} frame(s)")
    if totals["missing"]:
        print(f"  {totals['missing']} frame(s) in the database had no file on "
              f"disk -- `prune` removes videos, not frames; re-run `export`.")
    print(f"\nThese are boxes and tracks, not team numbers. Naming a track "
          f"needs a\nscorer in identify.py, which does not exist yet -- see "
          f"SCOUTING.md.")
    con.close()
    return 0


def _lan_address():
    """This machine's address on the field network, for the printed URL.

    A scoreboard people have to find by asking is one nobody opens. No packet
    is sent: connect() on a UDP socket only picks the route.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))        # TEST-NET-1, deliberately unroutable
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _hub_arg(raw, what):
    if not raw:
        return None
    try:
        box = [float(v) for v in raw.split(",")]
    except ValueError:
        raise SystemExit(f"--{what} wants x,y,w,h in pixels (got {raw!r})")
    if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
        raise SystemExit(f"--{what} wants x,y,w,h with a positive size (got {raw!r})")
    return tuple(box)


def cmd_count(args, cfg):
    """Count scored fuel from the detector alone -- no scoreboard, no OCR."""
    from tbavid import count as counting
    from tbavid import db as dbmod
    from tbavid import detect as det

    hubs = {}
    for alliance, raw in (("blue", args.hub_blue), ("red", args.hub_red)):
        box = _hub_arg(raw, f"hub-{alliance}")
        if box:
            hubs[alliance] = box
    if not hubs and args.event:
        con = dbmod.connect()
        hubs = counting.hubs_from_db(con, args.event)
        con.close()
        if hubs:
            print(f"hub geometry from the database for {args.event}: "
                  + ", ".join(f"{a}={list(map(int, b))}" for a, b in hubs.items()))
    if not hubs:
        print(f"no hub geometry given, so it will be learned from the first "
              f"{args.learn} frames.\n"
              f"  Nothing is counted while it is being learned -- with no hub "
              f"there is no\n  'in' for a ball to go. `run.py db hub` records "
              f"it once per event instead.")

    model = det.load(args.weights)
    source = int(args.source) if str(args.source).isdigit() else args.source
    print(f"model: {det.model_source(args.weights)}  source: {source}\n")

    # At a scrimmage there is no FMS, so this count IS the score. That needs a
    # match clock (fuel thrown about between matches is not a score), a display
    # people can see, and a referee who can correct it.
    board = None
    if args.scoreboard:
        from tbavid.field import Match, serve as serve_board
        board = Match(args.match or "", auto_s=args.auto_s,
                      teleop_s=args.teleop_s,
                      points_per_ball={"auto": args.auto_points,
                                       "teleop": args.teleop_points})
        if args.port:
            serve_board(board, host=args.bind, port=args.port)
            where = args.bind if args.bind != "0.0.0.0" else _lan_address()
            print(f"score (JSON): http://{where}:{args.port}/state")
            print(f"  clock     : POST /start  /stop  /reset")
            print(f"  correction: POST /adjust {{\"alliance\":\"red\",\"delta\":1}}")
        if args.feed:
            print("feed        : one JSON line per change on stdout")
        print("Nothing scores until the match is started.\n")

    def factory(found):
        print("hubs: " + ", ".join(f"{a}={list(map(int, b))}"
                                   for a, b in sorted(found.items())) + "\n")
        return counting.BallCounter(
            found, min_track_frames=args.min_frames,
            reacquire_frames=args.reacquire,
            require_entry=not args.allow_inside, pad=args.pad)

    def emit(board):
        """One JSON line per change, for whatever is showing the score.

        Line-buffered and flushed, because the consumer of this is a pipe and
        a pipe block-buffers by default -- the score would arrive in bursts a
        few kilobytes apart, which on a scoreboard is the same as not arriving.
        """
        import json as _json
        sys.stdout.write(_json.dumps(board.state(), separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def on_event(e):
        if board is not None:
            # Refused outside a scoring phase, which is the point of having a
            # clock at all. Say so rather than dropping it silently.
            if not board.ball(e["alliance"]):
                if not args.feed:
                    print(f"  {e['t']:>8.2f}s  {e['alliance']:<4} +1  "
                          f"(not in a match -- ignored)")
                return
            if args.feed:
                return emit(board)
            st = board.state()["alliances"][e["alliance"]]
            print(f"  {e['t']:>8.2f}s  {e['alliance']:<4} +1  -> "
                  f"{st['total']} ball(s), {st['points']} pt")
            return
        if args.feed:
            sys.stdout.write(__import__("json").dumps(e, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            return
        print(f"  {e['t']:>8.2f}s  {e['alliance']:<4} +1  -> {e['total']}")

    warned = {"at": False}

    def on_health(h):
        if board is not None:
            board.set_health(h)
        if args.feed:
            return
        # Said once, loudly, and not repeated every second. With no FMS there
        # is nothing else that will ever notice this: a box that cannot keep up
        # with the camera misses balls between the frames it does see, and the
        # score comes out low with no gap and nothing that looks wrong.
        if h.get("keepingUp") is False and not warned["at"]:
            warned["at"] = True
            print(f"\n  !! {h['fps']:.0f} fps against a camera at "
                  f"{h['expectFps']:.0f} -- roughly "
                  f"{h['missedFrac'] * 100:.0f}% of frames are going past "
                  f"unseen.\n     Balls crossing the hub in those frames are "
                  f"not counted, and nothing\n     else here will notice. Lower "
                  f"the resolution, use a smaller model, or\n     accept that "
                  f"this score is a floor.\n")

    out = counting.run_source(
        model, source, factory, conf=args.conf, tracker=args.tracker,
        learn_frames=args.learn, hubs=hubs or None, on_event=on_event,
        max_frames=args.frames, expect_fps=args.expect_fps,
        on_health=on_health)

    if out.get("error"):
        print(f"\n  ! {out['error']}")
        return 1
    counter = out["counter"]
    print()
    for line in counter.report():
        print(line)
    h = out.get("health") or {}
    if h.get("keepingUp") is False:
        print(f"  !! ran at {h['fps']:.0f} fps against a camera at "
              f"{h['expectFps']:.0f}: this count is a floor, not a total")
    elif h.get("keepingUp") is None:
        print(f"  ran at {h.get('fps', 0):.0f} fps. Pass --expect-fps <camera "
              f"rate> and it will say whether that was fast enough -- nothing "
              f"else will.")

    if args.match:
        mk = args.match if "_" in args.match else f"{args.event or ''}_{args.match}"
        con = dbmod.connect()
        # With a scoreboard the match is the record, because it carries the
        # clock and the referee's corrections; without one the raw count is all
        # there is.
        series = board.series() if board else counter.series(out["events"])
        totals = board.totals() if board else dict(counter.totals)
        note = ("scrimmage scoreboard: detector count plus referee corrections"
                if board else
                "counted by the detector; no scoreboard was read")
        wrote = dbmod.write_live(
            con, args.event or "", mk, series, totals,
            label=args.match.split("_")[-1], note=note)
        con.close()
        print(f"\nfiled {mk}: {wrote['score_events']} scoring event(s)")
    else:
        print("\nNothing written. Pass --match to file this against a match.")
    return 0


def _source_fps(source) -> float:
    """A video file's frame rate, or 0.0 when it cannot be known."""
    if isinstance(source, int) or not Path(str(source)).is_file():
        return 0.0
    try:
        import cv2
        cap = cv2.VideoCapture(str(source))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def cmd_shots(args, cfg):
    """Per-robot shots from a model over video: who shot, made, missed."""
    import json

    from tbavid import count as counting
    from tbavid import db as dbmod
    from tbavid import detect as det
    from tbavid import shooting

    hubs = {}
    for alliance, raw in (("blue", args.hub_blue), ("red", args.hub_red)):
        box = _hub_arg(raw, f"hub-{alliance}")
        if box:
            hubs[alliance] = box
    if not hubs and args.event:
        con = dbmod.connect()
        hubs = counting.hubs_from_db(con, args.event)
        con.close()
        if hubs:
            print(f"hub geometry from the database for {args.event}: "
                  + ", ".join(f"{a}={list(map(int, b))}" for a, b in hubs.items()))
    try:
        teams = shooting.parse_teams(args.teams)
    except ValueError as exc:
        raise SystemExit(str(exc))

    source = int(args.source) if str(args.source).isdigit() else args.source
    fps = args.fps or _source_fps(source)
    if fps and fps < shooting.MIN_TRACKING_FPS:
        # Refusing would be wrong -- a 20 fps phone clip is still worth a
        # look -- but at the 3 fps of exported frames a ball crosses the field
        # between two samples, and every flight becomes noise.
        print(f"! {fps:.0f} fps is too slow to follow a ball in flight; shots "
              f"need at least {shooting.MIN_TRACKING_FPS:.0f}. Use the match "
              f"video, not exported frames.")
    elif not fps:
        print("  frame rate unknown, so shot times are wall-clock. Pass --fps "
              "for a recording.")

    model = det.load(args.weights)
    print(f"model: {det.model_source(args.weights)}  source: {source}\n")

    state = {"writer": None, "flash": None}

    def label(robot):
        if robot is None:
            return "unattributed"
        team = teams.get(robot)
        return f"robot {robot}" + (f" (team {team})" if team else "")

    def on_event(e):
        text = e["outcome"].replace("_", " ").upper()
        where = f" -> {e['hub']} hub" if e.get("hub") else ""
        print(f"{e['t']:8.1f}s  {label(e['robot']):<24} {text}{where}")
        state["flash"] = (e["t"], f"{label(e['robot'])}: {text}")

    def on_frame(frame, t, result, robots, balls, counter):
        if not args.annotate:
            return
        img = getattr(result, "orig_img", None)
        if img is None:
            return
        import cv2
        if state["writer"] is None:
            h, w = img.shape[:2]
            state["writer"] = cv2.VideoWriter(
                str(args.annotate), cv2.VideoWriter_fourcc(*"mp4v"),
                fps or 30.0, (w, h))
        out = img.copy()
        colour = {"blue": (255, 120, 0), "red": (0, 0, 255)}
        for alliance, (x, y, w, h) in ((counter.hubs if counter else hubs) or {}).items():
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                          colour.get(alliance, (255, 255, 255)), 2)
        for tid, (alliance, (x, y, w, h)) in robots.items():
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                          colour.get(alliance, (255, 255, 255)), 2)
            # The id is what --teams needs. Watching this video once, with the
            # match roster to hand, is how track ids become team numbers.
            cv2.putText(out, f"#{tid}" + (f" = {teams[tid]}" if tid in teams else ""),
                        (int(x), max(int(y) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour.get(alliance, (255, 255, 255)), 2)
        for (x, y, w, h) in balls.values():
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)),
                          (0, 220, 255), 1)
        if counter is not None:
            for row, (tid, st) in enumerate(sorted(counter.per_robot.items())):
                cv2.putText(out, f"#{tid}: {st['made']}/{st['shots']} made",
                            (10, 24 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            colour.get(st["alliance"], (255, 255, 255)), 2)
        if state["flash"] and t - state["flash"][0] < 1.5:
            cv2.putText(out, state["flash"][1], (10, out.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        state["writer"].write(out)

    result = shooting.run_shots(model, source, hubs=hubs or None,
                                conf=args.conf, tracker=args.tracker,
                                learn_frames=args.learn, fps=fps,
                                max_frames=args.frames, on_event=on_event,
                                on_frame=on_frame)
    if state["writer"] is not None:
        state["writer"].release()
        print(f"annotated video: {args.annotate}")
    if "error" in result:
        print(result["error"])
        return 1

    counter = result["counter"]
    print("\n" + "\n".join(counter.report()))
    if result.get("robots_recovered"):
        print(f"  {result['robots_recovered']} robot(s) kept their number "
              f"after the tracker lost them")
    by_team = counter.by_team(teams) if teams else {}
    if by_team:
        print("\nby team:")
        for team, row in sorted(by_team.items(),
                                key=lambda kv: (kv[0] is None, kv[0] or 0)):
            name = "unassigned tracks" if team is None else f"team {team}"
            acc = f"{100 * row['made'] / row['shots']:.0f}%" if row["shots"] else "-"
            print(f"  {name:<18} {row['shots']} shots, {row['made']} made, "
                  f"{row['missed']} missed  [{acc}]")
    if args.out:
        payload = {
            "source": str(source), "fps": fps, "frames": result["frames"],
            "hubs": result["hubs"], "events": result["events"],
            "per_robot": {str(k): v for k, v in counter.per_robot.items()},
            "by_team": {str(k): v for k, v in by_team.items()},
            "hub_totals": counter.hub_totals(),
            "unattributed": counter.unattributed,
            "ignored": counter.ignored,
            "stitched": counter.stitched, "reacquired": counter.reacquired,
            "robots_recovered": result.get("robots_recovered", 0),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")
    return 0


def cmd_live(args, cfg):
    """Scout a live feed: read the scoreboard as it happens, keep no video."""
    import json
    import urllib.request
    from tbavid import db as dbmod
    from tbavid import live
    from tbavid.config import OCR_WORK, ensure_dirs, tba_key
    from tbavid.ffm import require_tools
    from tbavid.tba import TBAClient, match_label

    require_tools()
    ensure_dirs()

    if not args.match and not args.hub:
        raise SystemExit(
            "live needs to know which match it is watching.\n"
            "  --match qm14                 name it yourself, or\n"
            "  --hub http://localhost:6059  ask the scouting hub what is on the "
            "field\n"
            "The counter cannot say which match it is, and a timeline filed "
            "against the\nwrong match key is worse than no timeline at all.")

    print(f"[1/3] resolving {args.url}")
    media = live.resolve(args.url, cfg)
    if not media:
        raise SystemExit("yt-dlp could not resolve that to a playable stream.\n"
                         "  Is it live right now, and is the URL the one you "
                         "would watch in a browser?")

    # The roster, so the timeline can be attributed to an alliance's three
    # robots. From TBA, like everywhere else -- the counter knows nothing.
    teams, label = None, args.match or ""
    if args.event:
        client = TBAClient(tba_key(), cfg["tba_min_interval_s"])
        for m in client.event_matches(args.event):
            if match_label(m) == label:
                teams = {side: [t.replace("frc", "") for t in
                                ((m.get("alliances") or {})
                                 .get(side, {}).get("team_keys") or [])]
                         for side in ("blue", "red")}
                break

    def current_match():
        """What the scouting hub says is on the field, if we are asking it."""
        if not args.hub:
            return label
        try:
            with urllib.request.urlopen(
                    args.hub.rstrip("/") + "/api/state", timeout=5) as res:
                state = json.loads(res.read().decode())
        except Exception:
            return None
        live_now = (state or {}).get("nexusLive") or {}
        return live_now.get("matchKey") or live_now.get("label") or None

    print(f"[2/3] finding the fuel counters ({args.bootstrap:.0f}s of footage)")
    print("      This needs the digits to actually move, so it has to run while")
    print("      a match is being played -- between matches it will find nothing.")

    seen = {"n": 0}

    def on_step(result, w):
        if result.get("error"):
            print(f"  ! {result['error']}")
            return True
        for e in result["events"]:
            print(f"  {e['t']:>7.1f}s  {e['alliance']:<4} +{e['balls']:<3} "
                  f"-> {e['total']}")
        seen["n"] += len(result["events"])
        t = result["totals"]
        print(f"  [{result['elapsed']:.0f}s] blue={t.get('blue')} red={t.get('red')}"
              f"  ({seen['n']} scoring event(s))")
        return True

    out = live.watch(media, cfg, OCR_WORK, on_step, chunk_s=args.chunk,
                     max_s=args.for_s, bootstrap_s=args.bootstrap)
    if out.get("error"):
        print(f"\n  ! {out['error']}")
        if not out.get("series"):
            return 1

    mk = args.match_key or (f"{args.event}_{label}" if args.event and label else label)
    if not mk:
        print("\nnothing to file this against, so nothing was written")
        return 1

    print(f"\n[3/3] filing {mk}")
    con = dbmod.connect()
    wrote = dbmod.write_live(con, args.event or "", mk,
                             out.get("series") or {}, out.get("totals") or {},
                             teams=teams, label=label,
                             note="read live off the broadcast; no video kept")
    con.close()
    print(f"  {wrote['score_events']} scoring event(s) written"
          f"{'' if wrote['scoreboard_ok'] else ' -- scoreboard_ok=0, the counters never read'}")
    print(f"  totals: {out.get('totals')}")
    print("\nNo video was kept. Nothing entered the dataset.")
    return 0


def cmd_stream(args, cfg):
    """Read a whole event-day stream instead of one upload per match."""
    if not (args.url or args.file):
        raise SystemExit("stream needs --url <stream> or --file <local video>")
    if args.url and args.file:
        raise SystemExit("--url and --file are two ways in; pick one")
    return pipeline.ingest_stream(
        cfg, url=args.url or "", local=args.file, event_key=args.event or "",
        from_match=args.from_match or "", listen_only=args.listen_only,
        limit=args.limit, force=args.force)


def cmd_formats(args, cfg):
    """List the broadcast layout profiles, or measure one against a download."""
    from tbavid import crop as crop_mod
    from tbavid import formats

    if args.calibrate:
        return _calibrate_format(args, cfg)

    if args.event or args.district or args.title:
        fmt, why = formats.select(district=args.district or "",
                                  event_key=args.event or "",
                                  title=args.title or "", cfg=cfg,
                                  event_type=args.event_type)
        print(f"{args.event or '(no event key)'}"
              f"{', district ' + args.district if args.district else ''}"
              f"{', type ' + str(args.event_type) if args.event_type is not None else ''}"
              f" -> {fmt.name}  ({why})\n")
        print(fmt.describe())
        print(f"  {fmt.notes}\n")
        # Without a type, a type-gated profile cannot match, and the answer
        # above is then "generic" for a reason that has nothing to do with the
        # district asked about. Say so rather than letting it read as a miss.
        if args.event_type is None and any(f.event_types for f in formats.FORMATS):
            gated = ", ".join(f.name for f in formats.FORMATS if f.event_types)
            print(f"  No --event-type given, so {gated} could not be considered:\n"
                  f"  a district's weekend events and that district's own state\n"
                  f"  championship share a district and are not the same broadcast.\n"
                  f"  TBA's event_type is what separates them "
                  f"(1 = district event, 2 = district championship).")
        return 0

    print("Broadcast layout profiles. The first whose selectors match wins;\n"
          "`generic` matches nothing and is only ever the fallback.\n")
    for fmt in formats.FORMATS:
        print(fmt.describe())
    print("Which one a video gets is decided by its event's TBA district, then\n"
          "its event key, then its video title. Override per run or per event:\n"
          '  "crop": {"format": "ca_district", "formats": {"2026casj": "generic"}}\n')
    print("To characterise a feed nobody has measured yet:\n"
          "  ./run.py formats --calibrate --video <id>")
    return 0


def _calibrate_format(args, cfg):
    """Report what one downloaded video's overlay actually measures."""
    from tbavid import crop as crop_mod
    from tbavid import formats

    manifest = pipeline.load_manifest()
    entries = manifest["videos"]
    if args.video:
        entry = entries.get(args.video)
        if entry is None:
            print(f"unknown video id {args.video}\n  known ids:")
            for vid in entries:
                print(f"    {vid}")
            return 1
        targets = [(args.video, entry)]
    elif args.event:
        targets = [(v, e) for v, e in entries.items()
                   if e.get("event_key") == args.event and not e.get("error")]
        if not targets:
            print(f"nothing downloaded from {args.event} yet")
            return 1
    else:
        print("--calibrate needs --video <id> or --event <key>: it measures a "
              "real download,\nnot a profile.")
        return 1

    for vid, entry in targets:
        raw = pipeline._ensure_raw(entry, cfg)
        if raw is None:
            print(f"{vid}: no source available to measure")
            continue
        fmt, why = formats.select(district=entry.get("district", ""),
                                  event_key=entry.get("event_key", ""),
                                  title=entry.get("title", ""), cfg=cfg)
        got = crop_mod.calibrate(raw, entry["analysis"], cfg, tune=fmt.tuning())
        print(f"\n{vid}")
        print(f"  profile in force : {fmt.name} ({fmt.provenance}) -- {why}")
        if got is None:
            print("  could not sample enough frames to measure")
            continue
        for k in ("letterbox", "top_frac", "bottom_frac_edge",
                  "bottom_frac_static", "static_split_row", "edge_peak",
                  "edge_peak_row", "edge_typical", "edge_ratio"):
            print(f"  {k:<18}: {got[k]}")
        edge_ok = got["edge_ratio"] and got["edge_ratio"] >= 3.0
        agree = (got["bottom_frac_edge"] or 0) > 0 and got["bottom_frac_static"] > 0
        if edge_ok and agree:
            print("  => both routes found a divider: this feed splits "
                  "(split_mode: expected)")
        elif edge_ok:
            print("  => an edge with no static band behind it: field furniture "
                  "rather than a divider (split_mode: unlikely)")
        else:
            print("  => no divider signal at all: single camera "
                  "(split_mode: unlikely)")
        print("\n  Look at the frames before writing any of this into a profile:\n"
              f"    ./run.py cropcheck --video {vid}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_pick_args(p):
        p.add_argument("-n", "--count", type=int, default=5,
                       help="how many never-pulled videos to fetch (default 5)")
        p.add_argument("--retry-failed", action="store_true",
                       help="reconsider videos that previously failed or were unavailable")
        p.add_argument("--seed", type=int, default=None,
                       help="fix the random event/match order (reproducible picks)")
        p.add_argument("--per-event-cap", type=int, default=0,
                       help="max videos from any single event (0 = no cap)")
        p.add_argument("--dry-run", action="store_true",
                       help="show what would be pulled without downloading")
        p.add_argument("--include-noncompetitive", action="store_true",
                       help="also consider offseason/preseason events and "
                            "practice matches (default: official competition "
                            "match play only)")
        p.add_argument("--shard", metavar="N/M",
                       help="split the catalogue across M people; you take "
                            "slice N (1-based). Needs no coordination -- the "
                            "partition is by video id, so no two shards can "
                            "ever pick the same match.")

    p = sub.add_parser("pull", help="full run: pick, download, clean, export frames")
    add_pick_args(p)
    p.add_argument("--review", action="store_true",
                   help="open the shot review UI before exporting frames")
    p.add_argument("--port", type=int, default=8731)
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("fetch", help="pick, download and clean only (no frames)")
    add_pick_args(p)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("review", help="review shot classification, then export frames")
    p.add_argument("--port", type=int, default=8731)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("export", help="export frames from already-cleaned videos")
    p.add_argument("--only", nargs="*", help="limit to these video ids")
    p.add_argument("--force", action="store_true", help="re-export even if already done")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("cropcheck", help="write proof frames for the crop box")
    p.add_argument("video", nargs="?", help="video id (default: the first ok video)")
    p.add_argument("--frames", type=int, default=3)
    p.set_defaults(func=cmd_cropcheck)

    p = sub.add_parser("scoreboard", help="re-read the scoreboard counters via OCR")
    p.add_argument("--only", nargs="*", help="limit to these video ids")
    p.set_defaults(func=cmd_scoreboard)

    p = sub.add_parser("verify", help="check OCR'd totals against TBA's official score")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("reprocess",
                       help="re-run crop/render/scoreboard on downloaded sources")
    p.add_argument("--only", nargs="*", help="limit to these video ids")
    p.set_defaults(func=cmd_reprocess)

    p = sub.add_parser("export-share",
                       help="bundle this machine's harvest for someone else")
    p.add_argument("path", help="output folder")
    p.set_defaults(func=cmd_export_share)

    p = sub.add_parser("merge", help="fold another person's harvest into this one")
    p.add_argument("path", help="folder produced by their `export-share`")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("db", help="build / sync the scouting database")
    p.add_argument("db_action", choices=["build", "sync", "hub", "export"], nargs="?",
                   default="build",
                   help="build: from manifest; sync: also fetch rosters from TBA; "
                        "hub: record an event's hub box; export: a copy that can "
                        "be served from a read-only disk")
    p.add_argument("--event", help="event key, for `hub`")
    p.add_argument("--alliance", choices=["blue", "red"], help="for `hub`")
    p.add_argument("--box", help="x,y,w,h in cleaned-frame pixels, for `hub`")
    p.add_argument("--to", type=Path, default=Path("data/serve/scouting.db"),
                   help="where `export` writes (default data/serve/scouting.db)")
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("serve", help="read-only JSON API for the scouting app")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                   help="0.0.0.0 when hosting; localhost by default")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8781)),
                   help="honours $PORT, which most hosts inject")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("count",
                       help="count scored fuel from the detector alone -- no "
                            "scoreboard, no OCR")
    p.add_argument("--weights", type=Path, required=True, help="the .pt")
    p.add_argument("--source", default="0",
                   help="camera index (0), a video file, or a stream URL")
    p.add_argument("--event", help="event key: for the recorded hub geometry, "
                                   "and for the match key if --match is short")
    p.add_argument("--match", help="file the result against this match "
                                   "(e.g. qm14). Omitted, nothing is written.")
    p.add_argument("--hub-blue", dest="hub_blue", metavar="X,Y,W,H",
                   help="blue hub box in pixels, instead of learning it")
    p.add_argument("--hub-red", dest="hub_red", metavar="X,Y,W,H",
                   help="red hub box in pixels")
    p.add_argument("--learn", type=int, default=90,
                   help="frames spent learning the hubs when none are given")
    p.add_argument("--conf", type=float, default=0.25, help="confidence floor")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--min-frames", dest="min_frames", type=int, default=3,
                   help="frames a ball must be seen for to be a ball")
    p.add_argument("--reacquire", type=int, default=12,
                   help="frames a score is held before counting, so a ball "
                        "that passed over the hub can withdraw it")
    p.add_argument("--allow-inside", dest="allow_inside", action="store_true",
                   help="also count a ball first seen already over the hub. "
                        "More recall, and it will count balls a robot merely "
                        "drove in front of.")
    p.add_argument("--pad", type=float, default=0.0,
                   help="grow the hub region by this many pixels")
    p.add_argument("--frames", type=int, default=0,
                   help="stop after N frames (0 = until the source ends)")
    p.add_argument("--scoreboard", action="store_true",
                   help="be the scoring authority at a scrimmage: a match "
                        "clock, so nothing counts between matches, and a "
                        "correction route for a referee. No display -- the "
                        "score comes out as JSON.")
    p.add_argument("--port", type=int, default=8780,
                   help="serve the JSON score on this port (0 = don't serve)")
    p.add_argument("--bind", default="0.0.0.0",
                   help="interface for the JSON feed (default every one, so "
                        "the display machine can reach it)")
    p.add_argument("--feed", action="store_true",
                   help="write one JSON line per change to stdout, for piping "
                        "into whatever shows the score")
    p.add_argument("--expect-fps", dest="expect_fps", type=float, default=0.0,
                   help="the camera's real frame rate. Given it, the counter "
                        "says whether it is keeping up -- and at a scrimmage "
                        "nothing else will, because a box that falls behind "
                        "misses balls and the score just comes out low.")
    p.add_argument("--auto-s", dest="auto_s", type=float, default=15.0,
                   help="seconds of autonomous (default 15)")
    p.add_argument("--teleop-s", dest="teleop_s", type=float, default=135.0,
                   help="seconds of teleop (default 135)")
    p.add_argument("--auto-points", dest="auto_points", type=float, default=1.0,
                   help="points per ball in auto (default 1, i.e. show balls)")
    p.add_argument("--teleop-points", dest="teleop_points", type=float,
                   default=1.0, help="points per ball in teleop (default 1)")
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("detect",
                       help="run a trained .pt over harvested frames and "
                            "record what it found")
    p.add_argument("--weights", type=Path, required=True,
                   help="the .pt, e.g. runs/<name>/weights/best.pt")
    p.add_argument("--match", help="one match key, instead of all of them")
    p.add_argument("--event", help="limit to one event")
    p.add_argument("--limit", type=int, default=0, help="first N matches only")
    p.add_argument("--conf", type=float, default=0.25,
                   help="confidence floor (default 0.25)")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="ultralytics tracker config (default bytetrack.yaml)")
    p.add_argument("--assign", action="store_true",
                   help="also try to name each track's team. Reports nothing "
                        "until identify.py has a scorer, which is the honest "
                        "answer rather than a guess.")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("shots",
                       help="per-robot shots from a model over video: who "
                            "shot, how many went in, how many missed")
    p.add_argument("--weights", type=Path, required=True,
                   help="the .pt -- it must detect robots as well as fuel")
    p.add_argument("--source", required=True,
                   help="a match video, or a camera index such as 0. Not "
                        "exported frames: a ball in flight cannot be followed "
                        "at 3 fps")
    p.add_argument("--event", help="event key, for hub geometry recorded "
                                   "with `run.py db hub`")
    p.add_argument("--hub-blue", dest="hub_blue", metavar="X,Y,W,H",
                   help="blue hub box in frame pixels")
    p.add_argument("--hub-red", dest="hub_red", metavar="X,Y,W,H",
                   help="red hub box in frame pixels")
    p.add_argument("--learn", type=int, default=90,
                   help="with no boxes given, frames to learn the hubs over "
                        "(needs a model with hub classes)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence floor")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--fps", type=float, default=0.0,
                   help="the source's frame rate, when the file does not say")
    p.add_argument("--frames", type=int, default=0,
                   help="stop after N frames (0 = the whole source)")
    p.add_argument("--teams", default="",
                   help="robot track ids to team numbers, e.g. 3=254,7=254,"
                        "5=1678. Read the ids off --annotate; one team can "
                        "have several")
    p.add_argument("--annotate", type=Path,
                   help="write a video with robot ids, balls, hubs and each "
                        "outcome drawn on -- for assigning teams and for "
                        "checking shots by eye")
    p.add_argument("--out", type=Path, help="write the results as JSON")
    p.set_defaults(func=cmd_shots)

    p = sub.add_parser("live",
                       help="scout a live feed: read the scoreboard as it "
                            "happens and keep no video")
    p.add_argument("--url", required=True,
                   help="the live stream, as you would watch it (Twitch or YouTube)")
    p.add_argument("--event", help="TBA event key, for the roster")
    p.add_argument("--match", help="which match is on the field, e.g. qm14")
    p.add_argument("--match-key", dest="match_key",
                   help="the full TBA match key, if it is not <event>_<match>")
    p.add_argument("--hub", metavar="URL",
                   help="ask a running scouting hub what is on the field "
                        "instead of naming the match (e.g. http://localhost:6059)")
    p.add_argument("--chunk", type=float, default=20.0,
                   help="seconds read per pass (default 20)")
    p.add_argument("--bootstrap", type=float, default=90.0,
                   help="seconds of footage used to find the counters (default 90)")
    p.add_argument("--for", dest="for_s", type=float, default=0.0,
                   help="stop after this many seconds (0 = until the stream ends)")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("stream",
                       help="pull one whole event-day stream and cut every "
                            "match out of it by listening for the field")
    p.add_argument("--url", help="the stream: a YouTube or Twitch URL, or a "
                                 "bare YouTube video id")
    p.add_argument("--file", type=Path,
                   help="a stream already on disk, instead of downloading one")
    p.add_argument("--event", help="TBA event key. Without it the clips are "
                                   "harvested but carry no match keys, because "
                                   "there is no schedule to align against.")
    p.add_argument("--from-match", dest="from_match", metavar="LABEL",
                   help="which match this stream starts at (e.g. qm14), for a "
                        "day the cue count cannot line up on its own")
    p.add_argument("--listen-only", action="store_true", dest="listen_only",
                   help="report what the audio contains and stop: no cutting, "
                        "no processing, nothing written")
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N matches found (0 = all)")
    p.add_argument("--force", action="store_true",
                   help="reprocess clips already in the manifest")
    p.set_defaults(func=cmd_stream)

    p = sub.add_parser("formats",
                       help="broadcast layout profiles: list, explain, calibrate")
    p.add_argument("--event", help="event key, to show the profile it selects")
    p.add_argument("--district", help="TBA district abbreviation, e.g. ca")
    p.add_argument("--title", help="video title, for the title selectors")
    p.add_argument("--event-type", type=int, default=None, dest="event_type",
                   help="TBA event_type (1 = district event, 2 = district "
                        "championship). Some profiles only apply to one kind "
                        "of event and cannot match without it.")
    p.add_argument("--calibrate", action="store_true",
                   help="measure a real download instead of listing profiles; "
                        "needs --video or --event")
    p.add_argument("--video", help="manifest video id to measure")
    p.set_defaults(func=cmd_formats)

    p = sub.add_parser("status", help="ledger and dataset summary")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("audit",
                       help="report which harvested videos are not competitive "
                            "(reports only, deletes nothing)")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("prune", help="delete sources/cleaned videos to reclaim disk")
    p.add_argument("--clean", action="store_true",
                   help="also delete cleaned videos whose frames are exported")
    p.add_argument("--clean-only", action="store_true", help="keep raw sources")
    p.add_argument("--all", action="store_true", help="raw sources and cleaned videos")
    p.set_defaults(func=cmd_prune)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
