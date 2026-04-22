"""Tests for Monte Carlo verification of covariance steering + pdSTL plans."""

import pytest
import torch
from dynamics.single_integrator import SingleIntegrator
from dynamics.double_integrator import DoubleIntegrator
from steering.closed_loop import ClosedLoopSteerer
from planning.base import PlanResult
from monte_carlo import sample_trajectories, eval_spec_empirical, mc_verify
from stl.predicates import RectangularGoalPredicate
from stl.operators import Eventually


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def simple_plan(device):
    """Minimal PlanResult for a SingleIntegrator over T=5 steps."""
    dyn = SingleIntegrator(dt=0.2, u_max=1.0, D_diag=0.05, device=device)
    T, nx, nu = 5, 2, 2
    V = torch.zeros(T, nu, device=device)
    K = torch.zeros(T, nu, nx, device=device)
    mu0 = torch.zeros(nx, device=device)
    Sigma0 = torch.eye(nx, device=device) * 0.01

    steerer = ClosedLoopSteerer(dyn)
    rollout = steerer(V, K, mu0, Sigma0)

    result = PlanResult(
        mu_trace=rollout.mu_trace.detach(),
        Sigma_trace=rollout.Sigma_trace.detach(),
        V=V, K=K, best_p=0.7,
    )
    return dyn, result, mu0, Sigma0


@pytest.fixture(scope="module")
def trivial_spec():
    """Eventually reach an enormous region — always satisfied."""
    return Eventually(
        RectangularGoalPredicate({"x": [-1000.0, 1000.0], "y": [-1000.0, 1000.0]}),
        interval=[0, 5],
    )


# ── Tests ─────────────────────────────────────────────────────────────

def test_sample_shape(device, simple_plan):
    dyn, result, mu0, Sigma0 = simple_plan
    N = 50
    samples = sample_trajectories(dyn, result, mu0, Sigma0, n_samples=N, device=device)
    T = result.V.shape[0]
    assert samples.shape == (N, T + 1, dyn.nx)


def test_zero_noise_zero_feedback_collapses_to_mean(device):
    """With near-zero noise and K=0 and Sigma0=0, samples equal the nominal mean."""
    dyn = SingleIntegrator(dt=0.2, u_max=1.0, D_diag=1e-9, device=device)
    T, nx, nu = 5, 2, 2
    V = torch.randn(T, nu, device=device) * 0.3
    K = torch.zeros(T, nu, nx, device=device)
    mu0 = torch.tensor([1.0, 0.5], device=device)
    Sigma0 = torch.zeros(nx, nx, device=device)

    steerer = ClosedLoopSteerer(dyn)
    rollout = steerer(V, K, mu0, Sigma0)
    result = PlanResult(
        mu_trace=rollout.mu_trace.detach(),
        Sigma_trace=rollout.Sigma_trace.detach(),
        V=V, K=K, best_p=0.5,
    )

    N = 20
    samples = sample_trajectories(dyn, result, mu0, Sigma0, n_samples=N, device=device)
    nominal = result.mu_trace.squeeze(0)  # [T+1, nx]

    torch.testing.assert_close(
        samples,
        nominal.unsqueeze(0).expand(N, -1, -1),
        atol=1e-3, rtol=1e-3,
    )


def test_empirical_range_and_types(device, simple_plan, trivial_spec):
    dyn, result, mu0, Sigma0 = simple_plan
    N = 30
    samples = sample_trajectories(dyn, result, mu0, Sigma0, n_samples=N, device=device)
    p_emp, successes = eval_spec_empirical(samples, trivial_spec, device=device)

    assert isinstance(p_emp, float)
    assert 0.0 <= p_emp <= 1.0
    assert successes.shape == (N,)
    assert successes.dtype == torch.bool


def test_trivially_satisfiable_spec(device, simple_plan, trivial_spec):
    dyn, result, mu0, Sigma0 = simple_plan
    N = 100
    samples = sample_trajectories(dyn, result, mu0, Sigma0, n_samples=N, device=device)
    p_emp, successes = eval_spec_empirical(samples, trivial_spec, device=device)

    assert p_emp == pytest.approx(1.0, abs=1e-6), (
        f"Trivially satisfiable spec should give P̂(φ)=1.0, got {p_emp:.4f}"
    )


def test_mc_verify_returns_correct_structure(device, simple_plan, trivial_spec):
    dyn, result, mu0, Sigma0 = simple_plan
    T = result.V.shape[0]
    N = 20
    mc = mc_verify(result, dyn, trivial_spec, mu0, Sigma0, n_samples=N, device=str(device))

    assert set(mc.keys()) == {"p_analytic", "p_empirical", "samples", "successes"}
    assert isinstance(mc["p_analytic"], float)
    assert isinstance(mc["p_empirical"], float)
    assert mc["samples"].shape == (N, T + 1, dyn.nx)
    assert mc["successes"].shape == (N,)
    assert mc["successes"].dtype == torch.bool
