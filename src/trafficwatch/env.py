"""Offline mode and determinism. Import this module before ultralytics or torch."""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Forced, not setdefault: the evaluation machine has no internet and any
# implicit download (weights by name, fonts, pip auto-install) must fail fast.
for _key, _value in {
    "YOLO_OFFLINE": "1",
    "YOLO_AUTOINSTALL": "False",
    "YOLO_VERBOSE": "False",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "MPLBACKEND": "Agg",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}.items():
    os.environ[_key] = _value
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "trafficwatch_yolo"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights" / "torch"))


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
