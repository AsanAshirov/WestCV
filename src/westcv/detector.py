"""Road-user detector: Ultralytics YOLO with COCO weights, no fine-tuning."""
from __future__ import annotations

import sys

import numpy as np
import torch
from ultralytics import YOLO

# COCO ids we keep: road users, traffic lights, animals (road_obstacle candidates).
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
           9: "traffic light", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}


def fp16_ok(device) -> bool:
    """GTX 16xx cards (TU116/TU117) return empty results in FP16; T4 and RTX cards do not."""
    return device != "cpu" and "GTX 16" not in torch.cuda.get_device_name(device)


class Detector:
    def __init__(self, weights: str, imgsz: int = 1920, conf: float = 0.1, device=None):
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            print("[westcv] WARNING: no CUDA device (driver too old for the torch build?): the detector runs "
                  "on CPU and will not fit the time budget", file=sys.stderr, flush=True)
        self.quantize = 16 if fp16_ok(self.device) else 32
        self.imgsz, self.conf = imgsz, conf
        self.model = YOLO(weights)

    def __call__(self, frames: list[np.ndarray], imgsz: int | None = None) -> list[np.ndarray]:
        """Per frame an (N, 6) float32 array: x1, y1, x2, y2 (frame pixels), confidence, COCO class."""
        results = self.model.predict(frames, imgsz=imgsz or self.imgsz, conf=self.conf, classes=list(CLASSES),
                                     quantize=self.quantize, device=self.device, verbose=False)
        out = []
        for r in results:
            b = r.boxes
            out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                       b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32))
        return out

    def warmup(self, shape=(1080, 1920, 3), imgsz: int | None = None) -> None:
        self([np.zeros(shape, np.uint8)], imgsz)
