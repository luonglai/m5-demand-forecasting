"""Load the YAML run configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    mode = cfg["storage"]["mode"]
    if mode not in ("path", "table"):
        raise ValueError(f"storage.mode must be 'path' or 'table', got {mode!r}")
    if mode == "table" and not (cfg["storage"]["catalog"] and cfg["storage"]["schema"]):
        raise ValueError("storage.catalog and storage.schema are required when mode = table")
    cutoffs = cfg["forecast"]["cutoffs"]
    if cutoffs != sorted(cutoffs) or len(cutoffs) < 2:
        raise ValueError("forecast.cutoffs needs at least two ascending values (validation, test)")
