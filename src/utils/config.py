"""Configuration loading, device selection, and run control."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml


# ── Project root (two levels up from this file) ──────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    """Load a YAML file. Relative paths resolve from project root."""
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def deep_merge(base, override):
    """Recursively merge *override* into *base* (mutates base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_scenario(scenario_path):
    """Load scenario config merged with defaults.

    Returns (cfg, dynamics_cfg) where dynamics_cfg is loaded from the
    path specified in cfg['dynamics'].
    """
    cfg = load_config(scenario_path)
    defaults = load_config("configs/defaults.yaml")
    merged = deep_merge(defaults, cfg)
    dyn_cfg = load_config(merged["dynamics"])
    return merged, dyn_cfg


# ── Skip / run blocks (for main.py) ─────────────────────────────────

class _SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, label):
    """Context manager to skip or run a code block.

    Usage:
        with skip_run("run", "My Experiment") as check, check():
            do_stuff()
    """
    @contextmanager
    def check():
        if flag == "skip":
            sys.stderr.write(f"\x1b[90m  skip │ {label}\x1b[0m\n")
            raise _SkipWith()
        else:
            sys.stdout.write(f"\x1b[1;32m  run  │ {label}\x1b[0m\n")
            yield

    try:
        yield check
    except _SkipWith:
        pass
