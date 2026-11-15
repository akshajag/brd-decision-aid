"""Load config/assumptions.yaml and expose parameter values with their metadata."""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS_PATH = ROOT / "config" / "assumptions.yaml"
PARAM_SECTIONS = ("economics", "health", "interventions")


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with open(ASSUMPTIONS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_assumptions() -> dict:
    """Return a deep copy so callers can modify values (what-if) safely."""
    return copy.deepcopy(_load_raw())


def param_values(cfg: dict) -> dict[str, float]:
    """Flatten economics/health/interventions to {key: value}."""
    out = {}
    for sec in PARAM_SECTIONS:
        for k, spec in cfg[sec].items():
            out[k] = float(spec["value"])
    return out


def param_specs(cfg: dict) -> dict[str, dict]:
    """{key: spec-with-section} for UI sliders and Monte Carlo ranges."""
    out = {}
    for sec in PARAM_SECTIONS:
        for k, spec in cfg[sec].items():
            out[k] = {**spec, "section": sec}
    return out
