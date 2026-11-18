"""Run an exported fuel model (.onnx) with ONNX Runtime: no torch, no GPU.

Why. The Hub Counter app bundled torch + Ultralytics for the model half of
the hub count, and they were most of its size (the Linux arm64 build passed
GitHub's 2 GB asset limit with PyPI's CUDA torch; the CPU builds are still
several hundred MB). ONNX Runtime is ~15-20 MB, runs on the CPU by default
and leaves the GPU to whatever else the computer is doing. The weights are
the same numbers: `train/export_onnx.py` checks every export against the
.pt on the same images and refuses one whose detections differ.

Three kinds of export are read, told apart by their outputs' shapes, so
whichever model wins the 2026-11 experiments drops in as one file:

  * YOLO with NMS left to us -- one output (B, 4 + nc, N): centre-x,
    centre-y, w, h in input pixels, then a score per class. fuel_relabel.pt
    is this kind (its head's end2end is off). Decoded exactly as
    ultralytics.utils.nms.non_max_suppression does at predict time: best
    class over `conf`, then greedy NMS at IoU 0.5 per class
    (PREDICT_IOU in Ultralytics 8.4), at most `max_det`.
  * YOLO26 end-to-end -- one output (B, K, 6): x1, y1, x2, y2, score,
    class; no NMS.
  * RF-DETR -- two outputs, boxes (B, Q, 4) as normalised centre-x,
    centre-y, w, h and logits (B, Q, C). Class c is logit channel c, 0-based
    like our label files; channels past the model's classes are RF-DETR's
    no-object slot ("__background__", train_rfdetr.detection_names) and are
    never reported. Scores are sigmoids, top Q pairs, no NMS: RF-DETR's own
    postprocess (decode_rfdetr).

Preprocessing follows each library: YOLO letterboxes (cv2 INTER_LINEAR,
grey 114 padding, centred) and scales to 0-1 RGB; RF-DETR resizes straight
to its square resolution and normalises with ImageNet's mean and std.
The hub counter's crops are square, so the letterbox never pads there.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

NMS_IOU = 0.5           # Ultralytics 8.4's PREDICT_IOU
MAX_NMS = 30000         # boxes NMS looks at, as Ultralytics' max_nms
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

Dets = Tuple[np.ndarray, np.ndarray, np.ndarray]   # xyxy (N,4), score (N,), class (N,)


def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Ultralytics' LetterBox for a fixed square input: (image, gain, (pad_x, pad_y))."""
    import cv2
    h, w = img.shape[:2]
    gain = min(size / h, size / w)
    nh, nw = max(round(h * gain), 1), max(round(w * gain), 1)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(py - 0.1), round(py + 0.1)
    left, right = round(px - 0.1), round(px + 0.1)
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))
    return img, gain, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS, highest score first: the indices kept (torchvision.ops.nms)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        union = area[i] + area[rest] - inter
        order = rest[inter <= iou * np.maximum(union, 1e-9)]
    return np.array(keep, dtype=np.int64)


def decode_yolo_raw(pred: np.ndarray, conf: float, max_det: int,
                    classes: Optional[Sequence[int]] = None, iou: float = NMS_IOU) -> Dets:
    """One image's (4 + nc, N) output -> detections in input pixels."""
    p = pred.T
    cls_scores = p[:, 4:]
    j = cls_scores.argmax(1)
    s = cls_scores[np.arange(len(p)), j]
    keep = s > conf
    if classes is not None:
        keep &= np.isin(j, list(classes))
    p, s, j = p[keep], s[keep], j[keep]
    if len(p) > MAX_NMS:
        top = np.argsort(-s, kind="stable")[:MAX_NMS]
        p, s, j = p[top], s[top], j[top]
    xy, wh = p[:, :2], p[:, 2:4] / 2
    boxes = np.concatenate([xy - wh, xy + wh], 1)
    # class-aware NMS by offsetting each class far apart, as Ultralytics (max_wh 7680)
    k = nms(boxes + (j * 7680.0)[:, None], s, iou)[:max_det]
    return boxes[k], s[k], j[k]


def decode_yolo_e2e(pred: np.ndarray, conf: float, max_det: int,
                    classes: Optional[Sequence[int]] = None) -> Dets:
    """One image's (K, 6) end-to-end output -> detections in input pixels."""
    keep = pred[:, 4] > conf
    if classes is not None:
        keep &= np.isin(pred[:, 5].astype(int), list(classes))
    p = pred[keep][:max_det]
    return p[:, :4].copy(), p[:, 4].copy(), p[:, 5].astype(int)


def decode_rfdetr(boxes: np.ndarray, logits: np.ndarray, size: int, nc: int, conf: float,
                  max_det: int, classes: Optional[Sequence[int]] = None,
                  num_select: int = 0) -> Dets:
    """One image's (Q, 4) normalised cxcywh + (Q, C) logits -> detections in input pixels.

    As RF-DETR 1.12's PostProcess._select_topk: every (query, channel) pair
    ranked by sigmoid, highest first and ties by index, the top `num_select`
    (default Q, the model's own: 300, or 900 for the tuned one) kept, and
    only then the threshold. The no-object channels take part in that
    ranking -- they can push a real class out of the top Q -- and are
    dropped after it, as detection_names drops "__background__"."""
    c_all = logits.shape[1]
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    flat = prob.reshape(-1)
    order = np.argsort(-flat, kind="stable")[:num_select or len(logits)]
    order = order[flat[order] > conf]
    q, j = order // c_all, order % c_all
    keep = j < nc
    if classes is not None:
        keep &= np.isin(j, list(classes))
    q, j, order = q[keep][:max_det], j[keep][:max_det], order[keep][:max_det]
    b = boxes[q].astype(np.float64) * size
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
    return xyxy, flat[order], j


class OnnxDetector:
    """predict(images) -> per image (xyxy, score, class) in that image's pixels.

    `providers` defaults to the CPU only: the point is to leave the GPU alone.
    Pass e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"] to use one.
    """

    def __init__(self, path: str, providers: Optional[List[str]] = None, threads: int = 0):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=opts,
                                            providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        self.fixed_batch = inp.shape[0] if isinstance(inp.shape[0], int) else 0
        outs = self.session.get_outputs()
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.names = parse_names(meta.get("names", ""))
        if len(outs) == 2:
            self.kind = "rfdetr"
            # boxes are the 4-wide output; RF-DETR's export names them dets/labels
            shapes = [o.shape for o in outs]
            self.box_out = 0 if shapes[0][-1] == 4 else 1
            c = shapes[1 - self.box_out][-1]
            self.nc = len(self.names) or (int(c) - 1 if isinstance(c, int) else 1)
        else:
            shape = outs[0].shape
            self.kind = "yolo_e2e" if shape[-1] == 6 else "yolo"
            self.nc = len(self.names) or (int(shape[1]) - 4 if isinstance(shape[1], int) else 1)

    def _prep(self, img: np.ndarray):
        if self.kind == "rfdetr":
            import cv2
            h, w = img.shape[:2]
            # resized as 0-1 floats, as RF-DETR resizes its tensor (bilinear,
            # no antialias): resizing the uint8 image rounded every pixel and
            # moved 1-3 boxes per image across the threshold
            x = cv2.resize(img[:, :, ::-1].astype(np.float32) / 255.0, (self.size, self.size),
                           interpolation=cv2.INTER_LINEAR)
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            return x.transpose(2, 0, 1), (self.size / w, self.size / h), (0, 0)
        x, gain, pad = letterbox(img, self.size)
        x = x[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return x, (gain, gain), pad

    def predict(self, images: Sequence[np.ndarray], conf: float = 0.25, max_det: int = 300,
                classes: Optional[Sequence[int]] = None) -> List[Dets]:
        """BGR images (as OpenCV reads them) -> detections, boxes clipped to each image."""
        prepped = [self._prep(im) for im in images]
        batch = np.ascontiguousarray(np.stack([p[0] for p in prepped]))
        outs = self._run(batch)
        results = []
        for i, (im, (_, (gx, gy), (px, py))) in enumerate(zip(images, prepped)):
            if self.kind == "yolo":
                b, s, c = decode_yolo_raw(outs[0][i], conf, max_det, classes)
            elif self.kind == "yolo_e2e":
                b, s, c = decode_yolo_e2e(outs[0][i], conf, max_det, classes)
            else:
                b, s, c = decode_rfdetr(outs[self.box_out][i], outs[1 - self.box_out][i],
                                        self.size, self.nc, conf, max_det, classes)
            b = b.astype(np.float64)
            b[:, [0, 2]] = (b[:, [0, 2]] - px) / gx
            b[:, [1, 3]] = (b[:, [1, 3]] - py) / gy
            h, w = im.shape[:2]
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
            results.append((b, s.astype(np.float64), c))
        return results

    def _run(self, batch: np.ndarray) -> List[np.ndarray]:
        """One run, or one per image when the export has a fixed batch that differs."""
        if not self.fixed_batch or self.fixed_batch == len(batch):
            return self.session.run(None, {self.input: batch})
        parts = [self.session.run(None, {self.input: batch[i:i + 1]}) for i in range(len(batch))]
        return [np.concatenate([p[k] for p in parts]) for k in range(len(parts[0]))]


def parse_names(raw: str) -> List[str]:
    """Ultralytics writes names as "{0: 'fuel', 1: 'robot'}" in the metadata."""
    import ast
    if not raw:
        return []
    try:
        d = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(d, dict):
        return [str(d[k]) for k in sorted(d)]
    return [str(n) for n in d] if isinstance(d, (list, tuple)) else []
