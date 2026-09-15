"""Stage orchestration: fetch -> (review) -> export."""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from . import crop, download, labels, ledger as L, render, review, scoreboard, shots
from .config import (DATA, FRAME_DIR, LABEL_DIR, MANIFEST_PATH, OCR_WORK,
                     RAW_DIR, REVIEW_DIR, THUMB_DIR, VIDEO_DIR, ensure_dirs,
                     tba_key)
from .ffm import probe, require_tools
from .ledger import Ledger
from .tba import TBAClient, pick_unseen

QUARANTINE_DIR = REVIEW_DIR / "quarantine"


# -- manifest --------------------------------------------------------------
def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            print("  ! manifest unreadable, starting a new one")
    return {"version": 1, "videos": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(MANIFEST_PATH)


def video_id(cand: dict) -> str:
    return f"{cand['event_key']}_{cand['label']}_{cand['yt_key']}"


def _extras(cand: dict, meta: dict, **more) -> dict:
    """Ledger detail fields. yt_key is the ledger's own key, so drop it here."""
    out = {k: v for k, v in cand.items() if k not in ("yt_key", "teams")}
    out.update(meta)
    out.update(more)
    return out


# -- fetch -----------------------------------------------------------------
def fetch(cfg: dict, count: int, retry_failed: bool = False,
          seed: Optional[int] = None, per_event_cap: int = 0,
          dry_run: bool = False, shard: int = 0, shards: int = 1) -> List[str]:
    """Pick, download, classify, crop and render. Returns the new video ids."""
    require_tools()
    ensure_dirs()
    led = Ledger()

    who = f" (shard {shard + 1} of {shards})" if shards > 1 else ""
    print(f"[1/4] picking {count} unseen {cfg['season']} videos from TBA{who}")
    client = TBAClient(tba_key(), cfg["tba_min_interval_s"])
    picks = pick_unseen(client, cfg["season"], count, led,
                        retry_failed=retry_failed, per_event_cap=per_event_cap,
                        rng=random.Random(seed), shard=shard, shards=shards)
    if not picks:
        print("  nothing new to pull -- every candidate is already in the ledger")
        return []

    for cand in picks:
        print(f"    {video_id(cand)}")
    if dry_run:
        print("  dry run: nothing downloaded, ledger untouched")
        return []

    manifest = load_manifest()
    produced: List[str] = []

    for n, cand in enumerate(picks, start=1):
        vid = video_id(cand)
        yt = cand["yt_key"]
        print(f"\n[2/4] ({n}/{len(picks)}) {vid}")

        raw, fail, meta = download.download(yt, RAW_DIR, cfg)
        if fail:
            print(f"  ! download {fail}: {meta.get('title', '')[:60]}")
            led.record(yt, fail, **_extras(cand, meta))
            continue
        print(f"  downloaded {raw.name} ({meta['duration']:.0f}s)")

        entry = _process(raw, vid, cand, meta, cfg)
        if entry.get("error"):
            print(f"  ! {entry['error']}")
            led.record(yt, L.FAILED, **_extras(cand, meta, error=entry["error"]))
            _cleanup_raw(raw, cfg)
            continue

        manifest["videos"][vid] = entry
        save_manifest(manifest)
        led.record(yt, L.REJECTED if entry["status"] == "quarantined" else L.OK,
                   **_extras(cand, meta, video_id=vid,
                             coverage=entry["analysis"]["coverage"]))
        _cleanup_raw(raw, cfg)
        produced.append(vid)

    return produced


def _process(raw: Path, vid: str, cand: dict, meta: dict, cfg: dict) -> dict:
    """Analyse, crop and render one downloaded file into a manifest entry."""
    analysis = shots.analyze(raw, cfg, thumb_dir=THUMB_DIR, thumb_prefix=f"{vid}_")
    if analysis.get("error"):
        return {"error": analysis["error"]}

    cov = analysis["coverage"]
    print(f"  {analysis['n_shots']} shots / {analysis['n_clusters']} clusters, "
          f"main camera covers {cov*100:.0f}% of runtime")

    quarantined = cov < cfg["reject_threshold"]
    if quarantined:
        print(f"  !! QUARANTINED: main-camera coverage {cov*100:.0f}% is below "
              f"{cfg['reject_threshold']*100:.0f}% -- classification is not "
              f"trustworthy, so no frames will be exported.\n"
              f"     Run `run.py review` to inspect and rescue it.")

    if not analysis["keep_ranges"]:
        return {"error": "no shots survived classification"}

    # Locate the fuel counters first: where they sit sets a floor on how deep
    # the banner crop must go, and that floor is what stops the crop slicing
    # through the digits on layouts where the counters share a row with the
    # match clock.
    located = {}
    if cfg.get("score_labels", True):
        located = scoreboard.locate(raw, analysis, cfg, OCR_WORK)
        if located.get("error"):
            print(f"  scoreboard: {located['error']}")

    try:
        crop_box = crop.resolve_crop(raw, analysis, cfg,
                                     banner_floor_px=scoreboard.banner_floor_px(located),
                                     event_key=cand.get("event_key", ""))
    except ValueError as exc:
        return {"error": str(exc)}
    for note in crop_box["notes"]:
        print(f"  crop: {note}")
    print(f"  crop: {analysis['width']}x{analysis['height']} -> "
          f"{crop_box['w']}x{crop_box['h']}")

    ranges = [tuple(r) for r in analysis["keep_ranges"]]
    out_dir = QUARANTINE_DIR if quarantined else VIDEO_DIR
    dest = out_dir / f"{vid}.mp4"

    if cfg.get("keep_clean", True) or quarantined:
        ok, err = render.render_clean(raw, dest, ranges, crop_box["filter"], cfg)
        if not ok:
            return {"error": f"render failed: {err}"}
        info = probe(dest)
        print(f"  rendered {os.path.relpath(dest)} "
              f"({info['duration']:.0f}s of {analysis['duration']:.0f}s)")
    else:
        # Nothing is going to read this file, so do not spend an x264 pass
        # making it. export() samples straight from the source instead.
        dest = None
        print(f"  skipping intermediate render (keep_clean=false)")

    score = None
    if cfg.get("score_labels", True):
        score = scoreboard.read(raw, analysis, located, cfg, OCR_WORK)
        if not score.get("error"):
            final = ", ".join(f"{a}={v}" for a, v in sorted(score["final"].items()))
            print(f"  scoreboard: {len(score['events'])} scoring events ({final})")

    return {
        **{k: cand[k] for k in ("yt_key", "match_key", "event_key", "label")},
        "teams": cand.get("teams") or {},
        "shard": f"{shard + 1}/{shards}" if shards > 1 else None,
        "worker": cfg.get("worker") or None,
        "title": meta.get("title", ""),
        "source_duration": meta.get("duration", analysis["duration"]),
        "clean_path": str(dest) if dest else None,
        "raw_path": str(raw) if cfg["keep_raw"] else None,
        "analysis": analysis,
        "crop": crop_box,
        "score": score,
        "status": "quarantined" if quarantined else "ok",
        "reviewed": False,
        "exported": None,
    }


def _cleanup_raw(raw: Path, cfg: dict) -> None:
    if not cfg["keep_raw"]:
        raw.unlink(missing_ok=True)


# -- re-render after review ------------------------------------------------
def rerender(cfg: dict, vids: List[str]) -> None:
    manifest = load_manifest()
    for vid in vids:
        entry = manifest["videos"].get(vid)
        if not entry:
            continue
        raw = _ensure_raw(entry, cfg)
        if raw is None:
            print(f"  ! {vid}: no source available to re-render")
            continue
        ranges = [tuple(r) for r in entry["analysis"]["keep_ranges"]]
        if not ranges:
            print(f"  ! {vid}: every shot was dropped, nothing to render")
            continue
        dest = VIDEO_DIR / f"{vid}.mp4"
        ok, err = render.render_clean(raw, dest, ranges, entry["crop"]["filter"], cfg)
        if not ok:
            print(f"  ! {vid}: render failed: {err}")
            continue
        old = Path(entry["clean_path"])
        if old != dest:
            old.unlink(missing_ok=True)
        entry["clean_path"] = str(dest)
        entry["status"] = "ok"
        print(f"  re-rendered {vid}")
        _cleanup_raw(raw, cfg)
    save_manifest(manifest)


def _ensure_raw(entry: dict, cfg: dict) -> Optional[Path]:
    """Return the raw source, re-downloading if it was pruned."""
    stated = entry.get("raw_path")
    if stated and Path(stated).exists():
        return Path(stated)
    existing = sorted(RAW_DIR.glob(f"{entry['yt_key']}.*"))
    if existing:
        return existing[0]
    print(f"  source for {entry['yt_key']} was pruned; re-downloading")
    raw, fail, _ = download.download(entry["yt_key"], RAW_DIR, cfg)
    return None if fail else raw


# -- export ----------------------------------------------------------------
def export(cfg: dict, only: Optional[List[str]] = None, force: bool = False) -> int:
    ensure_dirs()
    manifest = load_manifest()
    total = 0
    for vid, entry in manifest["videos"].items():
        if only and vid not in only:
            continue
        if entry.get("status") != "ok":
            continue
        if entry.get("exported") and not force:
            continue
        for stale in FRAME_DIR.glob(f"{vid}_*.jpg"):
            stale.unlink()

        clean = Path(entry["clean_path"]) if entry.get("clean_path") else None
        if clean and clean.exists():
            result = render.export_frames(clean, FRAME_DIR, vid, cfg)
        else:
            # No intermediate video: sample from the source with the same
            # trim/crop graph the renderer would have used.
            raw = _ensure_raw(entry, cfg)
            if raw is None:
                print(f"  ! {vid}: no source and no cleaned video")
                continue
            result = render.export_frames_direct(
                raw, FRAME_DIR, vid,
                [tuple(r) for r in entry["analysis"]["keep_ranges"]],
                entry["crop"]["filter"], cfg)
        total += result.get("written", 0)
        msg = (f"  {vid}: {result.get('written', 0)} frames "
               f"({result.get('duplicates', 0)} near-duplicates dropped)")

        score = entry.get("score") or {}
        if result.get("frames") and not score.get("error") and score.get("series"):
            csv_path = LABEL_DIR / f"{vid}.csv"
            labels.write_frame_labels(csv_path, result["frames"],
                                      entry["analysis"]["keep_ranges"],
                                      score["series"], cfg)
            entry["labels"] = str(csv_path)
            msg += f", labels -> {csv_path.name}"
        # The frame list is large and reconstructible; keep the manifest small.
        result.pop("frames", None)
        entry["exported"] = result
        if clean and not cfg.get("keep_clean", True) and result.get("written"):
            # Frames are the training input; the cleaned video is an
            # intermediate worth ~122 MB a match. Re-renderable from the raw,
            # or re-downloadable, so it is safe to drop once frames exist.
            clean.unlink(missing_ok=True)
            msg += ", cleaned video removed"
        print(msg)
    save_manifest(manifest)
    return total


# -- review ----------------------------------------------------------------
def review_stage(cfg: dict, port: int = 8731) -> List[str]:
    manifest = load_manifest()
    if not manifest["videos"]:
        print("  manifest is empty -- run `fetch` first")
        return []
    changed = review.serve(manifest, port=port)
    if changed:
        print(f"[re-render] {len(changed)} video(s) changed")
        rerender(cfg, changed)
    return changed


# -- housekeeping ----------------------------------------------------------
def reprocess(cfg: dict, only: Optional[List[str]] = None) -> int:
    """Re-run analysis, crop, render and scoreboard on already-downloaded videos.

    The path back to a correct dataset after a detector fix: the source files
    are the expensive part and they are already on disk.
    """
    manifest = load_manifest()
    done = 0
    for vid, entry in list(manifest["videos"].items()):
        if only and vid not in only:
            continue
        raw = _ensure_raw(entry, cfg)
        if raw is None:
            print(f"  ! {vid}: no source available")
            continue
        cand = {k: entry.get(k, "") for k in ("yt_key", "match_key", "event_key", "label")}
        cand["teams"] = entry.get("teams") or {}
        meta = {"title": entry.get("title", ""),
                "duration": entry.get("source_duration") or 0.0}
        print(f"\n[reprocess] {vid}")
        fresh = _process(raw, vid, cand, meta, cfg)
        if fresh.get("error"):
            print(f"  ! {fresh['error']}")
            continue
        for stale in (VIDEO_DIR / f"{vid}.mp4", QUARANTINE_DIR / f"{vid}.mp4"):
            if str(stale) != fresh["clean_path"]:
                stale.unlink(missing_ok=True)
        fresh["teams"] = entry.get("teams") or fresh.get("teams") or {}
        manifest["videos"][vid] = fresh
        save_manifest(manifest)
        _cleanup_raw(raw, cfg)
        done += 1
    return done


def rescore(cfg: dict, only: Optional[List[str]] = None) -> int:
    """Re-read the scoreboard for videos already fetched."""
    manifest = load_manifest()
    done = 0
    for vid, entry in manifest["videos"].items():
        if only and vid not in only:
            continue
        raw = _ensure_raw(entry, cfg)
        if raw is None:
            print(f"  ! {vid}: no source available")
            continue
        score = scoreboard.extract(raw, entry["analysis"], entry["crop"], cfg, OCR_WORK)
        entry["score"] = score
        entry["exported"] = None          # labels are stale now
        if score.get("error"):
            print(f"  ! {vid}: {score['error']}")
        else:
            print(f"  {vid}: {len(score['events'])} events, final={score['final']}")
            done += 1
        _cleanup_raw(raw, cfg)
    save_manifest(manifest)
    return done


# A HEURISTIC, not the rule. On a checked match it reproduced red exactly (61)
# and missed blue by 29 (726 vs 755), so the 2026 fuel-to-points relation is
# something else and one match is not enough to pin it. Reported as a
# comparison so a systematic drift would show up across many matches; never
# used to correct or override what the counter actually displayed.
def expected_fuel(fields: dict) -> Optional[int]:
    try:
        return ((fields["totalAutoPoints"] - fields.get("autoTowerPoints", 0))
                + fields["totalTeleopPoints"])
    except (KeyError, TypeError):
        return None


def verify_scores(cfg: dict) -> None:
    """Check OCR'd fuel totals against TBA's official score breakdown."""
    manifest = load_manifest()
    client = TBAClient(tba_key(), cfg["tba_min_interval_s"])
    for vid, entry in manifest["videos"].items():
        score = entry.get("score") or {}
        if score.get("error") or not score.get("final"):
            continue
        match = client.get(f"/match/{entry['match_key']}")
        if not match:
            print(f"{vid}: TBA has no match {entry['match_key']}")
            continue
        breakdown = (match.get("score_breakdown") or {})
        print(f"\n{vid}  (TBA {entry['match_key']})")
        for alliance, total in sorted(score["final"].items()):
            fields = breakdown.get(alliance) or {}
            want = expected_fuel(fields)
            if want is None:
                ints = {k: v for k, v in fields.items() if isinstance(v, int)}
                print(f"  {alliance}: OCR {total}; no usable TBA fields {ints}")
            elif total == want:
                print(f"  {alliance}: counter {total} == heuristic {want}   match")
            else:
                print(f"  {alliance}: counter {total} vs heuristic {want}   "
                      f"delta {total - want:+d}  (counter is the measured value)")


def merge(cfg: dict, incoming: Path) -> Dict[str, int]:
    """Fold another person's harvest into this one.

    Expects the layout `run.py export-share` produces: frames/, labels/,
    manifest.json and seen.json. Everything is keyed by YouTube video id, so
    merging is a union -- and if the shards were disjoint there is nothing to
    collide in the first place. Overlaps are reported rather than silently
    resolved, because an overlap means the sharding was set up wrong and you
    want to know.
    """
    incoming = Path(incoming)
    if not incoming.exists():
        raise SystemExit(f"{incoming} does not exist")

    stats = {"videos_added": 0, "videos_already_had": 0,
             "frames_copied": 0, "labels_copied": 0, "ledger_added": 0}

    mine = load_manifest()
    other_path = incoming / "manifest.json"
    if other_path.exists():
        other = json.loads(other_path.read_text()).get("videos", {})
        for vid, entry in other.items():
            if vid in mine["videos"]:
                stats["videos_already_had"] += 1
                continue
            # Paths in their manifest point at their disk; ours will be
            # regenerated by `db build`, so blank them rather than lie.
            entry = dict(entry)
            entry["raw_path"] = None
            mine["videos"][vid] = entry
            stats["videos_added"] += 1
        save_manifest(mine)

    led = Ledger()
    other_led = incoming / "seen.json"
    if other_led.exists():
        for yt, rec in json.loads(other_led.read_text()).items():
            if yt not in led.entries:
                led.entries[yt] = rec
                stats["ledger_added"] += 1
        led.save()

    for name, dest in (("frames", FRAME_DIR), ("labels", LABEL_DIR)):
        src = incoming / name
        if not src.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.is_file() and not (dest / f.name).exists():
                shutil.copy2(f, dest / f.name)
                stats[f"{name}_copied"] += 1

    return stats


def export_share(cfg: dict, dest: Path) -> Dict[str, int]:
    """Bundle this machine's harvest for someone else to merge."""
    dest = Path(dest)
    (dest / "frames").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    counts = {"frames": 0, "labels": 0}
    for name, src in (("frames", FRAME_DIR), ("labels", LABEL_DIR)):
        if src.exists():
            for f in sorted(src.iterdir()):
                if f.is_file():
                    shutil.copy2(f, dest / name / f.name)
                    counts[name] += 1
    if MANIFEST_PATH.exists():
        shutil.copy2(MANIFEST_PATH, dest / "manifest.json")
    from .config import LEDGER_PATH
    if LEDGER_PATH.exists():
        shutil.copy2(LEDGER_PATH, dest / "seen.json")
    return counts


def prune(cfg: dict, raw: bool = True, clean: bool = False) -> int:
    """Free disk. Frames, labels and the database are never touched.

    Sources can be re-downloaded and cleaned videos re-rendered, so both are
    recoverable; the exported frames are the part that would actually cost you
    to regenerate.
    """
    freed = 0
    manifest = load_manifest()
    if raw and RAW_DIR.exists():
        for f in sorted(RAW_DIR.iterdir()):
            if f.is_file():
                freed += f.stat().st_size
                f.unlink()
        for entry in manifest["videos"].values():
            entry["raw_path"] = None
    if clean:
        for entry in manifest["videos"].values():
            path = entry.get("clean_path")
            if path and Path(path).exists() and (entry.get("exported") or {}).get("written"):
                freed += Path(path).stat().st_size
                Path(path).unlink()
    save_manifest(manifest)
    return freed


def status(cfg: dict) -> None:
    led = Ledger()
    manifest = load_manifest()
    counts = led.counts()
    print(f"ledger: {len(led.entries)} videos ever considered")
    for k in sorted(counts):
        print(f"  {k:<12} {counts[k]}")

    vids = manifest.get("videos", {})
    ok = sum(1 for v in vids.values() if v.get("status") == "ok")
    quar = sum(1 for v in vids.values() if v.get("status") == "quarantined")
    exported = sum(1 for v in vids.values() if v.get("exported"))
    frames = len(list(FRAME_DIR.glob("*.jpg"))) if FRAME_DIR.exists() else 0
    print(f"\nmanifest: {len(vids)} videos ({ok} ok, {quar} quarantined, "
          f"{exported} exported)")
    print(f"frames:   {frames} jpgs in {FRAME_DIR}")
    for label, d in (("raw", RAW_DIR), ("videos", VIDEO_DIR), ("frames", FRAME_DIR)):
        if d.exists():
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  {label:<8} {size/1e9:.2f} GB")
    if quar:
        print("\nquarantined videos are excluded from frames/; "
              "run `run.py review` to inspect them")
