#!/usr/bin/env python3
"""Build an Ultralytics dataset directory out of data/frames/.

Splits by MATCH, never by frame. Frames sampled 1/3 s apart from the same
match are near-identical, so a random frame-level split puts near-copies of
val images into train and hands you a validation score that means nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FRAMES = ROOT / "data" / "frames"
OUT = ROOT / "dataset"

# Kept deliberately small. Every class is a labelling cost multiplied by every
# frame, and alliance colour is what lets a score be attributed to a side.
CLASSES = ["fuel", "robot_blue", "robot_red", "hub_blue", "hub_red"]


def match_of(frame: Path) -> str:
    """'2026gal_qm62_cly0s0kOhEg_000123.jpg' -> '2026gal_qm62_cly0s0kOhEg'."""
    return frame.stem.rsplit("_", 1)[0]


def _scoreboard_ok_videos():
    """video_ids whose scoreboard read cleanly, from the scouting database."""
    try:
        from tbavid.db import connect
        con = connect()
        rows = con.execute(
            "SELECT video_id FROM matches WHERE scoreboard_ok=1 "
            "AND video_id IS NOT NULL").fetchall()
        con.close()
        return {r[0] for r in rows}
    except Exception:
        return None


def _scoring_frames() -> set:
    """Frames with fuel scored just after them, from the scouting database."""
    try:
        from tbavid.db import connect
        con = connect()
        rows = con.execute(
            "SELECT file FROM frames "
            "WHERE COALESCE(blue_next,0) + COALESCE(red_next,0) > 0").fetchall()
        con.close()
        return {r[0] for r in rows}
    except Exception as exc:
        print(f"  (no scoring data available: {exc})")
        return set()


def subsample(files: list, cap: int, scoring: set) -> list:
    """Pick `cap` frames spread across the match, favouring scoring moments.

    Evenly spaced rather than the first N: frames are sequential, so a prefix
    would be the opening seconds of the match over and over. Where the
    scoreboard says fuel landed just after a frame, prefer it -- those are the
    frames that actually show the event being detected.
    """
    if len(files) <= cap:
        return files
    hot = [f for f in files if f.name in scoring]
    cold = [f for f in files if f.name not in scoring]
    take_hot = min(len(hot), cap * 2 // 3) if hot else 0

    def spread(seq, n):
        if n <= 0 or not seq:
            return []
        if len(seq) <= n:
            return list(seq)
        step = len(seq) / n
        return [seq[int(i * step)] for i in range(n)]

    chosen = spread(hot, take_hot) + spread(cold, cap - take_hot)
    return sorted(set(chosen))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of MATCHES held out (default 0.2)")
    ap.add_argument("--val-matches", nargs="*", default=None,
                    help="explicit match ids for val, overriding --val-frac")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (forced on Windows, "
                         "which needs Developer Mode or admin rights to create "
                         "a symlink; also required for a portable archive)")
    ap.add_argument("--clean", action="store_true", help="wipe dataset/ first")
    ap.add_argument("--per-match", type=int, default=0,
                    help="cap frames per match (0 = all). 60-100 diverse frames "
                         "beats 480 consecutive near-duplicates and trains in a "
                         "fraction of the time")
    ap.add_argument("--scoreboard-ok-only", action="store_true",
                    help="use only matches whose scoreboard read cleanly. A "
                         "failed read means an overlay layout the crop "
                         "detector has not seen, and those frames can still "
                         "contain burned-in scoreboard -- so this doubles as a "
                         "filter for trustworthy crops.")
    ap.add_argument("--prefer-scoring", action="store_true",
                    help="when capping, bias selection toward frames where fuel "
                         "scored shortly after")
    args = ap.parse_args()

    frames = sorted(FRAMES.glob("*.jpg"))
    if not frames:
        print(f"no frames in {FRAMES} -- run `run.py pull` first")
        return 1

    by_match = defaultdict(list)
    for f in frames:
        by_match[match_of(f)].append(f)
    matches = sorted(by_match)

    if args.scoreboard_ok_only:
        good = _scoreboard_ok_videos()
        if good is None:
            print("  (no database -- cannot filter; run run.py db sync)")
        else:
            dropped = [m for m in matches if m not in good]
            matches = [m for m in matches if m in good]
            frames = [f for m in matches for f in by_match[m]]
            print(f"  scoreboard filter: keeping {len(matches)}, dropping "
                  f"{len(dropped)} with unverified layouts")
            for m in dropped[:8]:
                print(f"    dropped {m}")
            if not matches:
                print("  nothing left -- drop the flag or fix the crops")
                return 1

    if args.per_match:
        scoring = _scoring_frames() if args.prefer_scoring else set()
        for m in matches:
            by_match[m] = subsample(sorted(by_match[m]), args.per_match, scoring)
        frames = [f for m in matches for f in by_match[m]]

    if args.val_matches is not None:
        val = {m for m in args.val_matches if m in by_match}
        unknown = set(args.val_matches) - val
        if unknown:
            print(f"unknown match ids ignored: {sorted(unknown)}")
    else:
        # Hash-based so the same match always lands on the same side as the
        # dataset grows -- a fresh random draw each run would quietly migrate
        # matches across the split and contaminate every model you compare.
        ranked = sorted(matches,
                        key=lambda m: hashlib.sha256(m.encode()).hexdigest())
        n_val = max(1, round(len(matches) * args.val_frac)) if len(matches) > 1 else 0
        val = set(ranked[:n_val])

    use_copy = args.copy
    if not use_copy and platform.system() == "Windows":
        # Creating a symlink on Windows needs Developer Mode or an elevated
        # shell; without it Path.symlink_to raises and the whole build dies.
        print("  Windows detected -- copying images instead of symlinking")
        use_copy = True

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)

    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for match in matches:
        split = "val" if match in val else "train"
        for src in by_match[match]:
            dst = OUT / "images" / split / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if use_copy:
                shutil.copy2(src, dst)
            else:
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    # Fall back rather than abort: a dataset that costs extra
                    # disk beats one that does not build.
                    shutil.copy2(src, dst)
                    use_copy = True
            counts[split] += 1

    yaml = OUT / "dataset.yaml"
    yaml.write_text(
        "# Generated by train/prepare_dataset.py\n"
        f"path: {OUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )

    print(f"{len(matches)} match(es), {len(frames)} frames")
    for m in matches:
        print(f"  {'VAL  ' if m in val else 'train'}  {m}  ({len(by_match[m])} frames)")
    print(f"\ntrain={counts['train']}  val={counts['val']}")
    print(f"wrote {yaml}")

    if len(matches) < 2:
        print("\n!! Only one match. There is no honest val split from a single "
              "match -- every val frame is a near-copy of a train frame.\n"
              "   Pull at least 8-10 matches before you trust any metric.")
    labelled = len(list((OUT / "labels").rglob("*.txt")))
    if labelled == 0:
        print("\n!! No label files yet. Ultralytics reads dataset/labels/<split>/"
              "<frame>.txt; with none present every image counts as empty and "
              "the model learns to detect nothing. Annotate first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
