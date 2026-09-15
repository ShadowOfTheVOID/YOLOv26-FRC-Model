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


def cmd_pull(args, cfg):
    shard, shards = parse_shard(getattr(args, "shard", None))
    produced = pipeline.fetch(cfg, args.count, retry_failed=args.retry_failed,
                              seed=args.seed, per_event_cap=args.per_event_cap,
                              dry_run=args.dry_run, shard=shard, shards=shards)
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
                   dry_run=args.dry_run, shard=shard, shards=shards)
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
    p.add_argument("db_action", choices=["build", "sync", "hub"], nargs="?",
                   default="build",
                   help="build: from manifest; sync: also fetch rosters from TBA; "
                        "hub: record an event's hub box")
    p.add_argument("--event", help="event key, for `hub`")
    p.add_argument("--alliance", choices=["blue", "red"], help="for `hub`")
    p.add_argument("--box", help="x,y,w,h in cleaned-frame pixels, for `hub`")
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("serve", help="read-only JSON API for the scouting app")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                   help="0.0.0.0 when hosting; localhost by default")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8781)),
                   help="honours $PORT, which most hosts inject")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="ledger and dataset summary")
    p.set_defaults(func=cmd_status)

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
