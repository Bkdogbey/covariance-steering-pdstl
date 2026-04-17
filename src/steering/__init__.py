"""Steering factory.

Usage:
    from steering import get_steerer
    steerer = get_steerer("closed_loop", dynamics)
"""

from steering.base import BaseSteerer, RolloutResult
from steering.open_loop import OpenLoopSteerer
from steering.closed_loop import ClosedLoopSteerer

_REGISTRY = {
    "open_loop": OpenLoopSteerer,
    "closed_loop": ClosedLoopSteerer,
}


def get_steerer(name, dynamics):
    """Instantiate a steerer by name."""
    return _REGISTRY[name](dynamics)
