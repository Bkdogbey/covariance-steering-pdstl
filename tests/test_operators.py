"""Tests for STL operators."""

import torch
import pytest
from stl.base import GaussianBelief, BeliefTrajectory
from stl.operators import Always, Eventually, And
from stl.predicates import RectangularGoalPredicate, RectangularObstaclePredicate


def _make_trajectory(T=10, nx=2, device="cpu"):
    """Create a simple belief trajectory moving right along x-axis."""
    beliefs = []
    for t in range(T + 1):
        mu = torch.tensor([[float(t), 0.0]], device=device)  # [1, 2]
        var = torch.tensor([[0.1, 0.1]], device=device)       # [1, 2]
        beliefs.append(GaussianBelief(mu, var))
    return BeliefTrajectory(beliefs)


def test_predicate_output_shape():
    bt = _make_trajectory(T=10)
    pred = RectangularGoalPredicate({"x": [8.0, 10.0], "y": [-1.0, 1.0]})
    trace = pred(bt)
    assert trace.shape == (1, 11, 2)


def test_predicate_bounds_valid():
    """Probability bounds must be in [0, 1] with lower ≤ upper."""
    bt = _make_trajectory(T=10)
    pred = RectangularGoalPredicate({"x": [8.0, 10.0], "y": [-1.0, 1.0]})
    trace = pred(bt)
    assert (trace >= -1e-6).all()
    assert (trace <= 1.0 + 1e-6).all()
    assert (trace[..., 0] <= trace[..., 1] + 1e-6).all()


def test_always_output_shape():
    bt = _make_trajectory(T=10)
    pred = RectangularObstaclePredicate({"x": [4.0, 6.0], "y": [-1.0, 1.0]})
    spec = Always(pred, interval=[1, 10])
    trace = spec(bt)
    assert trace.shape == (1, 11, 2)


def test_eventually_finds_goal():
    """Agent passes through goal at t=9,10 → Eventually should have high probability."""
    bt = _make_trajectory(T=10)
    pred = RectangularGoalPredicate({"x": [8.0, 10.0], "y": [-1.0, 1.0]})
    spec = Eventually(pred, interval=[0, 10])
    trace = spec(bt)
    p_at_0 = trace[0, 0, 0].item()
    assert p_at_0 > 0.5, f"Expected high goal probability, got {p_at_0}"


def test_and_composition():
    bt = _make_trajectory(T=10)
    goal = RectangularGoalPredicate({"x": [8.0, 10.0], "y": [-1.0, 1.0]})
    safe = RectangularObstaclePredicate({"x": [4.0, 6.0], "y": [2.0, 4.0]})
    spec = And(Eventually(goal, interval=[0, 10]), Always(safe, interval=[1, 10]))
    trace = spec(bt)
    assert trace.shape == (1, 11, 2)
    assert (trace[..., 0] <= trace[..., 1] + 1e-6).all()
