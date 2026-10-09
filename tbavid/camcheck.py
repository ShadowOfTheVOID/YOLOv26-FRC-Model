"""Has this camera moved since its outlines were drawn?

Cameras are taken down and put back between the practice field and the
event, tripods get bumped, and camera numbers swap when cameras are
re-plugged. An outline drawn on yesterday's picture then counts the wrong
pixels, and nothing in the count says so: it just reads low. Until now the
only guard was "always check the picture" by eye.

So each camera keeps a small grey copy of the picture its outlines were
drawn on (`"reference"` in cams.json, about 15 KB). A new picture is
compared with it by phase correlation, which gives the shift, then by the
correlation of the two pictures' edges once aligned, which says whether
it is the same view at all. Edges, not brightness: the venue's lights and
the camera's auto exposure change the brightness of every pixel, not where
the edges are. The outcome is one of

- "ok": same view, moved less than half a ball;
- "moved": same view, shifted by (dx, dy) frame pixels -- the page offers
  to move the outlines by exactly that (a shift only: a camera turned or
  zoomed needs its outlines drawn again);
- "different": not the picture the outlines were drawn on -- another
  camera, or one turned far;
- "size": the picture size changed, so the outlines and ball size are in
  other pixels.

numpy and the standard library only, so it is tested without OpenCV.
"""
from __future__ import annotations

import base64
import zlib
from typing import Dict, Optional

import numpy as np

REF_WIDTH = 160          # thumbnail width; 1280 px frames -> 8 px per pixel
# Same view when the aligned edge pictures correlate at least this well. A
# synthetic scene under a 40% brightness change and noise stays above 0.8;
# two unrelated scenes sit near 0. The page says "check the picture", so a
# miss here costs a look, not a count.
SAME_VIEW = 0.35
# Moved when the shift is more than half a ball (the outline's tolerance
# measured on Einstein: +-10 px cost <= 5 points), or 1% of the width when
# no ball has been measured yet.
MOVE_BALLS = 0.5
MOVE_FRAC = 0.01


def _grey(frame) -> np.ndarray:
    a = np.asarray(frame, dtype=np.float32)
    if a.ndim == 3:
        # BGR, as OpenCV gives it
        a = 0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2]
    return a


def shrink(grey: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Box-average a grey image down to tw x th (area resampling)."""
    h, w = grey.shape
    ys = np.linspace(0, h, th + 1).astype(int)[:-1]
    xs = np.linspace(0, w, tw + 1).astype(int)[:-1]
    rows = np.add.reduceat(grey, ys, axis=0)
    cells = np.add.reduceat(rows, xs, axis=1)
    ny = np.diff(np.append(ys, h))[:, None]
    nx = np.diff(np.append(xs, w))[None, :]
    return cells / (ny * nx)


def make_reference(frame) -> Dict:
    """The small grey copy kept with a camera's outlines."""
    g = _grey(frame)
    h, w = g.shape
    tw = min(REF_WIDTH, w)
    th = max(1, round(h * tw / w))
    small = np.clip(shrink(g, tw, th), 0, 255).astype(np.uint8)
    data = base64.b64encode(zlib.compress(small.tobytes(), 9)).decode()
    return {"w": int(w), "h": int(h), "tw": int(tw), "th": int(th), "grey": data}


def _unpack(ref: Dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(ref["grey"]))
    return np.frombuffer(raw, np.uint8).reshape(ref["th"], ref["tw"]).astype(np.float32)


def _edges(a: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    return np.hypot(gx, gy)


def _phase_shift(a: np.ndarray, b: np.ndarray):
    """(dx, dy) that moves `a` onto `b`, in pixels, sub-pixel."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    r = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)

    def sub(c, n, get):
        # a parabola through the peak and its neighbours
        l, m, rr = get((c - 1) % n), get(c), get((c + 1) % n)
        d = l - 2 * m + rr
        off = 0.5 * (l - rr) / d if d else 0.0
        v = c + max(-0.5, min(0.5, off))
        return v - n if v > n / 2 else v
    dx = sub(px, w, lambda i: r[py, i])
    dy = sub(py, h, lambda i: r[i, px])
    return dx, dy


def _overlap_corr(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> float:
    """Correlation of a and b where they overlap once a is moved by (dx, dy)."""
    h, w = a.shape
    if abs(dx) >= w - 4 or abs(dy) >= h - 4:
        return 0.0
    ya, yb = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xa, xb = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    p, q = a[ya, xa].ravel(), b[yb, xb].ravel()
    p, q = p - p.mean(), q - q.mean()
    den = float(np.sqrt((p * p).sum() * (q * q).sum()))
    return float((p * q).sum() / den) if den else 0.0


def compare(ref: Optional[Dict], frame, ball_area: float = 0.0) -> Optional[Dict]:
    """How `frame` sits against the reference; None without a reference.
    dx, dy are in the frame's own pixels: add them to every outline point."""
    if not ref or not ref.get("grey"):
        return None
    g = _grey(frame)
    h, w = g.shape
    if (w, h) != (ref["w"], ref["h"]):
        return {"status": "size", "dx": 0.0, "dy": 0.0, "similarity": 0.0,
                "was": f"{ref['w']}x{ref['h']}", "now": f"{w}x{h}"}
    a = _unpack(ref)
    b = shrink(g, ref["tw"], ref["th"])
    dx, dy = _phase_shift(a, b)
    ea, eb = _edges(a), _edges(b)
    sim = max(_overlap_corr(ea, eb, int(round(dx)), int(round(dy))),
              _overlap_corr(ea, eb, 0, 0))
    k = ref["w"] / ref["tw"]
    fdx, fdy = round(float(dx) * k, 1), round(float(dy) * k, 1)
    limit = (MOVE_BALLS * np.sqrt(4 * ball_area / np.pi) if ball_area > 0
             else MOVE_FRAC * w)
    # never finer than one thumbnail pixel: below that the shift is noise
    limit = max(limit, k)
    if sim < SAME_VIEW:
        status = "different"
    elif float(np.hypot(fdx, fdy)) > limit:
        status = "moved"
    else:
        status = "ok"
    return {"status": status, "dx": fdx, "dy": fdy, "similarity": round(sim, 2)}


def shift_points(pts, dx: float, dy: float):
    return [[round(p[0] + dx, 1), round(p[1] + dy, 1)] for p in pts]
