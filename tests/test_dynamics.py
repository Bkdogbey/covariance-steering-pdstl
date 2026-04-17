"""Tests for the dynamics module."""

import torch
import pytest
from dynamics import get_dynamics
from dynamics.single_integrator import SingleIntegrator
from dynamics.double_integrator import DoubleIntegrator


def test_single_integrator_shapes():
    dyn = SingleIntegrator()
    assert dyn.A.shape == (2, 2)
    assert dyn.B.shape == (2, 2)
    assert dyn.DDT.shape == (2, 2)
    assert dyn.nx == 2
    assert dyn.nu == 2


def test_double_integrator_shapes():
    dyn = DoubleIntegrator()
    assert dyn.A.shape == (4, 4)
    assert dyn.B.shape == (4, 2)
    assert dyn.DDT.shape == (4, 4)
    assert dyn.nx == 4
    assert dyn.nu == 2


def test_factory():
    cfg = {"type": "double_integrator", "dt": 0.2, "u_max": 2.5, "D_diag": 0.03}
    dyn = get_dynamics(cfg)
    assert isinstance(dyn, DoubleIntegrator)


def test_step_preserves_shapes():
    dyn = DoubleIntegrator()
    mu = torch.zeros(4)
    Sigma = torch.eye(4) * 0.01
    u = torch.tensor([1.0, 0.0])
    mu_next, Sigma_next = dyn.step(mu, Sigma, u)
    assert mu_next.shape == (4,)
    assert Sigma_next.shape == (4, 4)


def test_bound_control():
    dyn = DoubleIntegrator(u_max=2.0)
    v = torch.tensor([100.0, -100.0])  # extreme values
    u = dyn.bound_control(v)
    assert (u.abs() <= 2.0 + 1e-6).all()
