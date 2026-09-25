"""Pipeline configuration (configs/pipeline.yaml). All thresholds are in seconds
and in box-size units (BS = sqrt(w*h) of the object's box), never in frames."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from .env import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("TW_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Paths in the config are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
