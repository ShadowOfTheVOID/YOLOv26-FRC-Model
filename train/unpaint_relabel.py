#!/usr/bin/env python3
"""Copy a relabel_video.py dataset with the grey paint taken back off.

Why. relabel_video.py paints grey over every ball-like yellow blob that its
detector neither boxed nor saw move: left in unboxed, those balls would
teach "this is not fuel". That is the right call for training on its
labels, and the wrong one for judging SAM 3 against them -- the painted
blobs are exactly the hard balls (in piles, in hoppers, high against the
crowd) that SAM 3 is being tried for. On painted images SAM 3 never gets
the chance, and the comparison is rigged towards the old labels.

Everything needed to cut the same pixels again is already recorded:

  * the image name is `<video>_<frame index>` for a whole frame, plus
    `t<x0>-<y0>` for a flight tile (relabel_video.py's naming);
  * relabel_report.json keeps the run's settings: `rows` (the Einstein
    split-screen cut) and `mask` (the scoreboard's yellow icon and arrows,
    greyed with 118 before labelling -- those stay greyed, or SAM 3 would
    be asked about a drawing of a ball);
  * frames are read the way relabel_video.py reads them: seek to two frames
    before, read three, keep the last. The same seek on the same file gives
    the same frame.

Labels are copied unchanged. The output is a dataset of its own, ready for
autolabel_sam3.py --compare or --out:

    python train/unpaint_relabel.py --data ds_relabel/dataset.yaml \\
        --videos /content/drive/MyDrive/frc-bakeoff/videos --out ds_unpainted
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

MASK_GREY = 118          # relabel_video.py's mask value
NAME = re.compile(r"^(?P<stem>.+)_(?P<fi>\d{6})(?:t(?P<x0>\d+)-(?P<y0>\d+))?$")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm", ".avi")


def parse_name(name: str) -> Optional[Tuple[str, int, Optional[Tuple[int, int]]]]:
    """'e4_007373' -> ('e4', 7373, None); 'e4_007373t1280-13' -> ('e4', 7373, (1280, 13)).
    None for a name relabel_video.py did not write."""
    m = NAME.match(Path(name).stem)
    if not m:
        return None
    tile = (int(m["x0"]), int(m["y0"])) if m["x0"] is not None else None
    return m["stem"], int(m["fi"]), tile


def settings(report: dict) -> Tuple[Optional[Tuple[int, int]], list]:
    """(rows, masks) from relabel_report.json's recorded arguments."""
    s = report.get("settings", {})
    rows = tuple(int(v) for v in s["rows"].split(":")) if s.get("rows") else None
    masks = [tuple(int(v) for v in m.split(",")) for m in s.get("mask") or []]
    return rows, masks


def cut(frame, rows, masks, tile, size):
    """The pixels relabel_video.py labelled, before its grey paint."""
    frame = frame.copy()
    for x0, y0, x1, y1 in masks:
        frame[y0:y1, x0:x1] = MASK_GREY
    if rows:
        frame = frame[rows[0]:rows[1]]
    if tile:
        x0, y0 = tile
        w, h = size
        frame = frame[y0:y0 + h, x0:x0 + w]
    return frame


def read_frame(cap, fi: int):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, fi - 2))
    cap.read()
    cap.read()
    ok, frame = cap.read()
    return frame if ok else None


def find_video(videos: Path, stem: str) -> Optional[Path]:
    for ext in VIDEO_EXT:
        p = videos / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="relabel_video.py dataset.yaml")
    ap.add_argument("--videos", type=Path, required=True, help="folder with the same video files")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import cv2
    src = args.data.resolve().parent
    report_path = src / "relabel_report.json"
    if not report_path.exists():
        raise SystemExit(f"{report_path} missing: not a relabel_video.py dataset")
    report = json.loads(report_path.read_text())
    rows, masks = settings(report)
    out = args.out.resolve()
    if out == src:
        raise SystemExit("--out must be a new directory")

    caps: Dict[str, object] = {}
    done = skipped = 0
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((src / "images" / split).glob("*.jpg")):
            parsed = parse_name(img.name)
            lab = src / "labels" / split / (img.stem + ".txt")
            if not parsed:
                skipped += 1
                continue
            stem, fi, tile = parsed
            if stem not in caps:
                path = find_video(args.videos, stem)
                caps[stem] = cv2.VideoCapture(str(path)) if path else None
                if not path:
                    print(f"  ! no video for '{stem}' in {args.videos}: its frames are left out")
            cap = caps[stem]
            frame = read_frame(cap, fi) if cap is not None else None
            if frame is None:
                skipped += 1
                continue
            painted = cv2.imread(str(img))
            h, w = painted.shape[:2]
            raw = cut(frame, rows, masks, tile, (w, h))
            if raw.shape[:2] != (h, w):
                # a different video file than the one labelled: the boxes would not fit
                print(f"  ! {img.name}: {raw.shape[1]}x{raw.shape[0]} from the video, "
                      f"{w}x{h} in the dataset -- left out")
                skipped += 1
                continue
            cv2.imwrite(str(out / "images" / split / img.name), raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if lab.exists():
                shutil.copy(lab, out / "labels" / split / lab.name)
            done += 1
    for cap in caps.values():
        if cap is not None:
            cap.release()
    yaml = (src / "dataset.yaml").read_text()
    yaml = re.sub(r"(?m)^path:.*$", f"path: {out}", yaml)
    (out / "dataset.yaml").write_text(yaml)
    shutil.copy(report_path, out / "relabel_report.json")
    print(f"{done} images re-cut without paint into {out}; {skipped} left out")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
