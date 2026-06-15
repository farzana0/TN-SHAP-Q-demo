"""Byte-stable persistence helpers for the demo notebooks.

`save_results` writes one JSON file per notebook to results/<NN>.json (sorted keys,
numpy scalars coerced to python floats) so the paper can read headline numbers
mechanically.  `savefig` saves figures at >=150 dpi with fixed PNG metadata so repeated
headless runs are byte-identical.
"""
from __future__ import annotations

import json
import os

import numpy as np

# repo root = parent of this package directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
RES_DIR = os.path.join(REPO_ROOT, "results")


def _coerce(obj):
    if isinstance(obj, dict):
        return {str(k): _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _coerce(obj.tolist())
    return obj


def save_results(nb_id: str, data: dict) -> str:
    """Write results/<nb_id>.json (e.g. nb_id='01'); returns the path."""
    os.makedirs(RES_DIR, exist_ok=True)
    path = os.path.join(RES_DIR, f"{nb_id}.json")
    with open(path, "w") as f:
        json.dump(_coerce(data), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def savefig(fig, name: str, dpi: int = 200) -> str:
    """Save fig to figures/<name> at >=150 dpi with fixed metadata; returns the path."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    meta = {"Software": None} if name.lower().endswith(".png") else None
    fig.savefig(path, dpi=max(150, dpi), bbox_inches="tight", metadata=meta)
    return path
