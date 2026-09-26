"""Pipeline configuration (configs/pipeline.yaml). All thresholds are in seconds
and in box-size units (BS = sqrt(w*h) of the object's box), never in frames."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from .env import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """A config may start with `extends: other.yaml` (relative to itself) and override
    only the keys it changes; nested sections are merged key by key."""
    path = Path(path or os.environ.get("TW_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base = cfg.pop("extends", None)
    return _merge(load_config(path.parent / base), cfg) if base else cfg


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def resolve(path: str | Path) -> Path:
    """Paths in the config are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
