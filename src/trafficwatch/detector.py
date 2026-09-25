"""YOLO detector (Ultralytics, COCO weights) mapped to four superclasses."""
from __future__ import annotations

from pathlib import Path

import numpy as np

VEHICLE, TWO_WHEELER, PERSON, ANIMAL = 0, 1, 2, 3
SUPER_NAMES = ("vehicle", "two_wheeler", "person", "animal")
# COCO id -> superclass. Car, bus and truck are one class: the detector often
# flips between them on the same object, and no rule needs the difference.
COCO_TO_SUPER = {0: PERSON, 1: TWO_WHEELER, 2: VEHICLE, 3: TWO_WHEELER, 5: VEHICLE, 7: VEHICLE,
                 16: ANIMAL, 17: ANIMAL, 18: ANIMAL, 19: ANIMAL}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def superclass_nms(det: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS inside each superclass (det rows: x1, y1, x2, y2, conf, sc)."""
    keep = []
    for sc in np.unique(det[:, 5]):
        idx = np.where(det[:, 5] == sc)[0]
        idx = idx[np.argsort(-det[idx, 4], kind="stable")]
        while len(idx):
            keep.append(idx[0])
            if len(idx) == 1:
                break
            overlap = box_iou(det[idx[:1], :4], det[idx[1:], :4])[0]
            idx = idx[1:][overlap < iou]
    return det[np.sort(keep)]


class Detector:
    """Batch inference: list of BGR frames -> list of (N, 6) float32 arrays
    with columns x1, y1, x2, y2, conf, superclass (pixels of the input frame)."""

    def __init__(self, weights: str | Path, imgsz: int, conf: float, iou: float = 0.6,
                 half: bool = True, agnostic_iou: float = 0.7, device: str | None = None):
        import torch
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.is_file():  # never let Ultralytics fetch weights by name
            raise FileNotFoundError(f"weights not found: {weights} (run weights/download.sh)")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = bool(half) and self.device.startswith("cuda")
        self.model = YOLO(str(weights), task="detect")
        self.imgsz, self.conf, self.iou, self.agnostic_iou = imgsz, conf, iou, agnostic_iou
        self.classes = sorted(COCO_TO_SUPER)

    def warmup(self, height: int, width: int) -> None:
        self([np.zeros((height, width, 3), np.uint8)])

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        results = self.model.predict(frames, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                     half=self.half, device=self.device, classes=self.classes,
                                     max_det=300, verbose=False)
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                out.append(np.zeros((0, 6), np.float32))
                continue
            cls = boxes.cls.cpu().numpy().astype(int)
            sc = np.array([COCO_TO_SUPER[c] for c in cls], np.float32)
            det = np.column_stack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), sc]).astype(np.float32)
            out.append(superclass_nms(det, self.agnostic_iou))
        return out
