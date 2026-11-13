"""Load config.yaml (or the file named by TRIAGE_CONFIG) and resolve paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _config_path() -> Path:
    name = os.environ.get("TRIAGE_CONFIG", "config.yaml")
    p = Path(name)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config() -> dict[str, Any]:
    with open(_config_path(), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve every entry under paths: relative to the repository root.
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else REPO_ROOT / p)
    cfg.setdefault("smoke", {"tiny_models": False})
    return cfg


CFG = load_config()


def path(key: str) -> Path:
    """Return a resolved path from the paths: section, creating parent dirs."""
    p = Path(CFG["paths"][key])
    p.mkdir(parents=True, exist_ok=True) if key.endswith("_dir") else p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve(p: str) -> str:
    """Resolve a possibly relative path (e.g. a local model dir) against the repo root.
    Anything that is not an existing local path is returned unchanged, so a
    HuggingFace model id like 'BAAI/bge-small-en-v1.5' keeps its forward slash on Windows.
    """
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    if (REPO_ROOT / pp).exists():
        return str(REPO_ROOT / pp)
    return p