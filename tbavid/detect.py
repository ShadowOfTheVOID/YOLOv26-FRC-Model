"""Run a trained detector over harvested frames and record what it found.

This is the step that was missing. `train/train.py` produces a `.pt` and,
until now, nothing in this package loaded one -- `ultralytics` appeared in
exactly one file, on the training side. So a finished model had nowhere to go,
the `detections` table was never written by anything, and `identify.py` sat
waiting on "a tracker upstream" that did not exist. Everything downstream of
the detector was designed and unreachable.

    python3 run.py detect --weights runs/harvest/weights/best.pt

## Over frames, not over video

Detections reference `frames.id`, so a detection can only exist for a frame the
dataset actually holds. Running the model over the exported frames rather than
over the cleaned video keeps that true by construction, and those frames
already carry `t_source` and the fuel lookahead columns -- which is what makes
a detection joinable to "did this alliance score in the next 1.5 seconds".

The cost is frame rate: frames are sampled at `sample_fps` (3 by default), so
tracking sees 3 fps rather than 30. ByteTrack copes with that for objects the
size of a robot; it is not enough for a ball in flight, and nothing here
pretends a fuel track means anything.

## What a `.pt` does and does not buy

It gives per-frame boxes and, through the tracker, tracks that persist across a
match. That is enough for robot positions, hub occupancy and per-alliance robot
counts.

It does **not** give you which team. `identify.assign_tracks` turns tracks into
team numbers only when handed a `scorer`, and writing one is still open: the
measurement in `identify.py` stands, a bumper number is ~5 px tall at broadcast
resolution and OCR returns the empty string every time. With no scorer,
`assign_tracks` deliberately records nothing rather than guessing, and
`run.py detect --assign` reports that as the answer instead of hiding it.

So: a `.pt` moves this from "no detections at all" to "detections and tracks,
unnamed". It does not close the per-robot gap on its own, and `SCOUTING.md`
already says what would.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import FRAME_DIR

# Must stay in lockstep with train/prepare_dataset.py, which writes the
# dataset's data.yaml and so decides what each class INDEX means to the
# trained weights. A mismatch here would not fail -- it would silently relabel
# every detection, calling blue robots red. tests/test_pipeline.py reads that
# file and asserts the two agree, because nothing else would notice.
CLASSES: Tuple[str, ...] = ("fuel", "robot_blue", "robot_red", "hub_blue", "hub_red")

# Objects worth tracking across frames. A ball is not: at 3 fps it moves too
# far between samples for identity to mean anything, and 127k of them per match
# would dominate the table for no gain.
TRACKED = ("robot_blue", "robot_red")

DEFAULT_CONF = 0.25
DEFAULT_TRACKER = "bytetrack.yaml"


def alliance_of(cls_name: str) -> Optional[str]:
    """'robot_blue' -> 'blue'. Fuel belongs to nobody until somebody scores it."""
    if cls_name.endswith("_blue"):
        return "blue"
    if cls_name.endswith("_red"):
        return "red"
    return None


def load(weights: Path):
    """A YOLO model, or a message a person can act on.

    Imported here rather than at module scope on purpose. `requirements.txt` is
    requests and numpy; torch and ultralytics are hundreds of megabytes and are
    needed only by whoever is running a model. `serve.py` and the whole
    scouting API must keep working on a host that has neither.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed in this environment.\n"
            "  Detection needs it and torch; harvesting and serving do not,\n"
            "  which is why they are not in requirements.txt:\n"
            "    python3 -m venv .venv-detect\n"
            "    .venv-detect/bin/pip install -r requirements-detect.txt")
    weights = Path(weights)
    if not weights.exists():
        raise SystemExit(
            f"no weights at {weights}\n"
            "  Train some first:  python3 train/train.py\n"
            "  which writes runs/<name>/weights/best.pt")
    model = YOLO(str(weights))
    # A model must say what its class indices mean, and this refuses rather
    # than falling back to CLASSES below. Since train/subset_classes.py exists
    # there are two class orders in play -- five for the scouting detector,
    # three for the counting one -- and assuming the wrong one relabels every
    # detection without failing: hub_blue read as robot_blue, and a scrimmage
    # scoreboard counting robots as fuel.
    if not getattr(model, "names", None):
        raise SystemExit(
            f"{weights} carries no class names, so there is no way to know "
            f"what its\n  class indices mean. Guessing would relabel every "
            f"detection silently.\n  Re-export it from a checkpoint that has "
            f"them.")
    return model


def frame_path(file: str) -> Path:
    """Where a `frames.file` value actually lives.

    The column holds the basename for everything this pipeline exports, but a
    merged-in harvest or a hand-edited CSV can carry a path, and silently
    looking in the wrong directory would read as "the model found nothing".
    """
    p = Path(file)
    if p.is_absolute() or len(p.parts) > 1:
        return p
    return FRAME_DIR / file


def rows_from_boxes(boxes: Iterable[Tuple], names: Dict[int, str],
                    source: str = "") -> List[Dict]:
    """Turn one frame's raw boxes into detection rows.

    Separated from the model call so it is testable without torch: every
    decision that could silently mislabel a detection -- the class lookup, the
    alliance, whether a track id survives -- happens here.

    `boxes` yields (x1, y1, x2, y2, conf, cls_index, track_id).
    """
    out: List[Dict] = []
    for x1, y1, x2, y2, conf, cls_index, track_id in boxes:
        name = names.get(int(cls_index))
        if name is None:
            # A model trained against a different data.yaml. Dropped rather
            # than stored under a made-up class name.
            continue
        x, y = int(round(min(x1, x2))), int(round(min(y1, y2)))
        w, h = int(round(abs(x2 - x1))), int(round(abs(y2 - y1)))
        if w <= 0 or h <= 0:
            continue
        out.append({
            "cls": name, "x": x, "y": y, "w": w, "h": h,
            "conf": round(float(conf), 4),
            "alliance": alliance_of(name),
            # Only kept for the classes tracking is meaningful for, so a stray
            # id on a ball cannot reach identify.assign_tracks.
            "track_id": (int(track_id) if track_id is not None
                         and name in TRACKED else None),
            "source": source or None,
        })
    return out


def _boxes_of(result) -> List[Tuple]:
    """Pull (xyxy, conf, cls, id) out of one ultralytics Result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    ids = boxes.id
    ids = ids.tolist() if ids is not None else [None] * len(boxes)
    out = []
    for xyxy, conf, cls, tid in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                    boxes.cls.tolist(), ids):
        out.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls, tid))
    return out


def match_frames(con, match_key: str) -> List[Tuple[int, str]]:
    """(frame_id, file) for one match, in the order they were played.

    Time order matters: the tracker carries state from one call to the next, so
    handing it frames out of order would build tracks out of nothing.
    """
    return [(r["id"], r["file"]) for r in con.execute(
        "SELECT id, file FROM frames WHERE match_key=? ORDER BY t_source, id",
        (match_key,))]


def write_rows(con, frame_id: int, rows: Sequence[Dict]) -> int:
    for r in rows:
        con.execute("""INSERT INTO detections
            (frame_id, cls, x, y, w, h, conf, alliance, team, track_id, source)
            VALUES (?,?,?,?,?,?,?,?,NULL,?,?)""",
                    (frame_id, r["cls"], r["x"], r["y"], r["w"], r["h"],
                     r["conf"], r["alliance"], r["track_id"], r["source"]))
    return len(rows)


def detect_match(con, model, match_key: str, source: str,
                 conf: float = DEFAULT_CONF,
                 tracker: str = DEFAULT_TRACKER,
                 progress=None) -> Dict:
    """Run the model over one match's frames and record the detections.

    Idempotent: a match's existing detections are cleared first, so re-running
    with better weights replaces that match rather than accumulating two
    models' opinions in one table. `source` is what tells them apart
    afterwards, and is why the column exists.
    """
    frames = match_frames(con, match_key)
    if not frames:
        return {"frames": 0, "detections": 0, "missing": 0, "tracks": 0}

    con.execute("""DELETE FROM detections WHERE frame_id IN
                   (SELECT id FROM frames WHERE match_key=?)""", (match_key,))

    # The model's own names, always -- CLASSES is the canonical five-class
    # order and only a last resort for a caller that built its own stub.
    names = dict(getattr(model, "names", None) or
                 {i: n for i, n in enumerate(CLASSES)})
    total, missing, tracks = 0, 0, set()
    for n, (frame_id, file) in enumerate(frames, 1):
        path = frame_path(file)
        if not path.exists():
            missing += 1
            continue
        # persist=True is what carries tracker state between calls; without it
        # every frame would start a fresh track and identify.assign_tracks
        # would see one track per frame instead of one per robot.
        result = model.track(source=str(path), persist=True, tracker=tracker,
                             conf=conf, verbose=False)
        result = result[0] if isinstance(result, list) else result
        rows = rows_from_boxes(_boxes_of(result), names, source=source)
        total += write_rows(con, frame_id, rows)
        tracks.update(r["track_id"] for r in rows if r["track_id"] is not None)
        if progress:
            progress(n, len(frames), total)
    con.commit()
    return {"frames": len(frames), "detections": total, "missing": missing,
            "tracks": len(tracks)}


def model_source(weights: Path) -> str:
    """A short, stable identifier for the weights that produced a detection.

    The run directory and the file, not an absolute path: `runs/x/weights` is
    the same model on the laptop that trained it and the box that ran it, and
    an absolute path would make the same detection look like two.
    """
    weights = Path(weights)
    parts = weights.parts
    if "runs" in parts:
        return "/".join(parts[parts.index("runs"):])
    return weights.name
