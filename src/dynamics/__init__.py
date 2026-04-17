"""Dynamics factory.

Usage:
    from dynamics import get_dynamics
    dyn = get_dynamics(dyn_cfg, device)
"""

from dynamics.base import BaseDynamics
from dynamics.single_integrator import SingleIntegrator
from dynamics.double_integrator import DoubleIntegrator

_REGISTRY = {
    "single_integrator": SingleIntegrator,
    "double_integrator": DoubleIntegrator,
}


def get_dynamics(cfg, device="cpu"):
    """Instantiate dynamics from a config dict."""
    cls = _REGISTRY[cfg["type"]]
    return cls(dt=cfg["dt"], u_max=cfg["u_max"], D_diag=cfg["D_diag"], device=device)
