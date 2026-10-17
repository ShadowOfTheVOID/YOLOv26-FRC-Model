"""A small bumper-number reader: the scorer bumperocr.TeamVotes uses with weights.

Why a learned reader replaced template matching (bumperocr.score_crop): on
Einstein 1 -- a match none of this was trained or tuned on -- templates
picked the right one of an alliance's three teams on 56% of robot crops.
Bumper fonts are italic, condensed and stylised ("9470" with a square 7 and
0), digits touch, so splitting them into blobs worked on 2 of 99 crops.

This reads the whole number strip at once: a CNN + bidirectional GRU trained
with CTC on synthetic digit strings plus real Einstein 4/5/8 strips
(train/train_bumper_ocr.py). A candidate team is scored by how likely the
strip is to read as exactly that number, so it is still a choice among the
six on the field, never free text. Held out on Einstein 1 (79 hand-labelled
crops): 83% right on the 56% of crops it is sure of (margin >= 2 nats).

Needs torch, which the detect environment already has for Ultralytics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import bumperocr as B

H, W = 32, 128
# A crop counts only when the best team beats the runner-up by this much
# negative log-likelihood; below it the read is a coin toss (precision on
# Einstein 1: 74% at 0, 83% at 2, 87% at 5).
MIN_MARGIN_NATS = 2.0
EVIDENCE_CAP = 6.0


def regions(crop: np.ndarray, alliance: str, limit: int = 4) -> List[np.ndarray]:
    """White number strips in the bumper band, H x W, widest first.

    Several are returned because the widest is often a decal on the robot
    (8613's frame lettering), not the number; the reader scores each and the
    best one counts.
    """
    import cv2
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    top = int(hsv.shape[0] * (1 - B.BUMPER_PART))
    band = B.bumper_band(hsv[top:], alliance)
    if band is None:
        return []
    y0, y1 = top + band[0], top + band[1]
    w = B.white_mask(hsv[y0:y1]).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(w, 8)
    # Real digits are 7-20 px here; MIN_DIGIT_PX (set on rendered digits)
    # threw most of them away.
    comps = sorted((tuple(st[i][:4]) for i in range(1, n)
                    if 7 <= st[i][3] <= 0.9 * (y1 - y0) and st[i][2] <= 8 * st[i][3]
                    and st[i][4] >= 12), key=lambda c: c[0])
    groups: List[List[int]] = []
    for c in comps:
        if groups:
            g = groups[-1]
            tall = max(g[3], c[3])
            if (c[0] - (g[0] + g[2]) <= 0.8 * tall
                    and abs((c[1] + c[3] / 2) - (g[1] + g[3] / 2)) < 0.8 * tall):
                x1, y1_ = min(g[0], c[0]), min(g[1], c[1])
                x2, y2 = max(g[0] + g[2], c[0] + c[2]), max(g[1] + g[3], c[1] + c[3])
                groups[-1] = [x1, y1_, x2 - x1, y2 - y1_]
                continue
        groups.append(list(c))
    out = []
    for x, y, gw, gh in sorted(groups, key=lambda g: -g[2]):
        if gw < 1.2 * gh:
            continue
        p = max(2, gh // 6)
        out.append(normalise(w[max(0, y - p):y + gh + p, max(0, x - p):x + gw + p].astype(np.float32)))
        if len(out) >= limit:
            break
    return out


def normalise(mask: np.ndarray) -> np.ndarray:
    import cv2
    h, w = mask.shape
    nw = min(W, max(8, int(round(w * H / h))))
    r = cv2.resize(mask, (nw, H), interpolation=cv2.INTER_AREA)
    o = np.zeros((H, W), np.float32)
    o[:, :nw] = r
    return o


def build():
    import torch.nn as nn
    import torch.nn.functional as F

    def c(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(c(1, 32), nn.MaxPool2d(2), c(32, 64), nn.MaxPool2d(2),
                                   c(64, 96), c(96, 96), nn.MaxPool2d((2, 1)),
                                   c(96, 128), nn.MaxPool2d((4, 1)))
            self.rnn = nn.GRU(128, 96, bidirectional=True, batch_first=True)
            self.out = nn.Linear(192, 11)          # 0 is the CTC blank

        def forward(self, x):                      # B,1,32,128 -> B,32,11 log-probs
            f = self.f(x).squeeze(2).permute(0, 2, 1)
            f, _ = self.rnn(f)
            return F.log_softmax(self.out(f), -1)
    return Net()


def encode(text: str) -> List[int]:
    return [int(ch) + 1 for ch in text]


def seq_nll(logp, cands: Sequence[str]):
    """-log P(strip reads exactly `c`) for each candidate; logp is T x 11."""
    import torch
    import torch.nn.functional as F
    T = logp.shape[0]
    lp = logp.unsqueeze(1).expand(T, len(cands), 11)
    tg = torch.cat([torch.tensor(encode(c)) for c in cands])
    return F.ctc_loss(lp, tg, torch.full((len(cands),), T, dtype=torch.long),
                      torch.tensor([len(c) for c in cands]), reduction="none",
                      zero_infinity=True)


class Reader:
    """`Reader(weights).score` is a TeamVotes scorer: crop -> {team: evidence}."""

    def __init__(self, weights: str, min_margin: float = MIN_MARGIN_NATS):
        import torch
        self.torch = torch
        self.net = build()
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval()
        self.min_margin = min_margin

    def nll(self, crop: np.ndarray, alliance: str, teams: Sequence[int]) -> Optional[np.ndarray]:
        strips = regions(crop, alliance) if crop is not None and crop.size else []
        if not strips or not teams:
            return None
        cands = [str(t) for t in teams]
        with self.torch.no_grad():
            lp = self.net(self.torch.tensor(np.stack(strips))[:, None])
            return np.min(np.stack([seq_nll(lp[i], cands).numpy() for i in range(len(strips))]), 0)

    def score(self, crop: np.ndarray, alliance: str, teams: Sequence[int]) -> Dict[int, float]:
        nll = self.nll(crop, alliance, teams)
        if nll is None or len(teams) < 2:
            return {}
        order = np.argsort(nll)
        margin = float(nll[order[1]] - nll[order[0]])
        if margin < self.min_margin:
            return {}
        # About one MIN_VOTE per two sure reads: a robot is named after a
        # couple of clear looks, never on one.
        return {int(teams[order[0]]): min(margin, EVIDENCE_CAP) / 4.0}
