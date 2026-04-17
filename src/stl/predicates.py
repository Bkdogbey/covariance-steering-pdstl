"""Probabilistic predicates over Gaussian beliefs.

Each predicate reads (μ, Σ) from a belief trajectory and returns
[B, T, 2] probability bounds via the Gaussian CDF.
"""

import math
import torch
from stl.operators import STL_Formula


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _extract(belief_trajectory, diagonal_only=True):
    """Stack means and variances from a belief trajectory.

    Returns:
        mu:  [B, T, D]
        var: [B, T, D] (diagonal) or [B, T, D, D] (full)
    """
    means, vars_ = [], []
    for b in belief_trajectory:
        means.append(b.mean_full)
        if diagonal_only and b.var_full.ndim > 2:
            vars_.append(torch.diagonal(b.var_full, dim1=-2, dim2=-1))
        else:
            vars_.append(b.var_full)
    return torch.stack(means, dim=1), torch.stack(vars_, dim=1)


def _normal_cdf(value, mean, var):
    """P(X ≤ value) for X ~ N(mean, var)."""
    std = torch.sqrt(var + 1e-6)
    z = (value - mean) / std
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


# ═════════════════════════════════════════════════════════════════════
# Predicates
# ═════════════════════════════════════════════════════════════════════

class RectangularGoalPredicate(STL_Formula):
    """P(x ∈ [x_min,x_max] × [y_min,y_max]).

    Decomposes into independent per-axis interval probabilities.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, bt, **kw):
        mu, var = _extract(bt)
        p_x = _normal_cdf(self.x_max, mu[..., 0], var[..., 0]) - \
              _normal_cdf(self.x_min, mu[..., 0], var[..., 0])
        p_y = _normal_cdf(self.y_max, mu[..., 1], var[..., 1]) - \
              _normal_cdf(self.y_min, mu[..., 1], var[..., 1])
        p = torch.clamp(p_x * p_y, 0.0, 1.0)
        return torch.stack([p, p], dim=-1)


class RectangularObstaclePredicate(STL_Formula):
    """P(x ∉ obstacle) = max(P(left), P(right), P(below), P(above))."""

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, bt, **kw):
        mu, var = _extract(bt)
        mx, my = mu[..., 0], mu[..., 1]
        vx, vy = var[..., 0], var[..., 1]
        probs = torch.stack([
            _normal_cdf(self.x_min, mx, vx),       # left
            1.0 - _normal_cdf(self.x_max, mx, vx), # right
            _normal_cdf(self.y_min, my, vy),        # below
            1.0 - _normal_cdf(self.y_max, my, vy),  # above
        ], dim=0)
        p, _ = probs.max(dim=0)
        return torch.stack([p, p], dim=-1)


class CircularObstaclePredicate(STL_Formula):
    """P(||x - center|| > radius) via projected variance."""

    def __init__(self, circle_def, device="cpu"):
        super().__init__()
        self.register_buffer(
            "center",
            torch.tensor(circle_def["center"], dtype=torch.float32, device=device),
        )
        self.radius = circle_def["radius"]

    def robustness_trace(self, bt, **kw):
        mu, sigma = _extract(bt, diagonal_only=False)
        diff = mu[..., :2] - self.center
        dist = torch.norm(diff, dim=-1)
        d = diff / (dist.unsqueeze(-1) + 1e-6)
        if sigma.ndim == 3:
            sigma_proj = torch.sum(d ** 2 * sigma[..., :2], dim=-1)
        else:
            sigma_proj = torch.einsum("bti,btij,btj->bt", d, sigma[..., :2, :2], d)
        p = 1.0 - _normal_cdf(self.radius, dist, sigma_proj)
        return torch.stack([p, p], dim=-1)


class MovingRectangularObstaclePredicate(STL_Formula):
    """P(x ∉ moving rectangle) with time-varying center."""

    def __init__(self, obs_def, device="cpu"):
        super().__init__()
        self.register_buffer(
            "x_traj",
            torch.as_tensor(obs_def["x_traj"], dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "y_traj",
            torch.as_tensor(obs_def["y_traj"], dtype=torch.float32, device=device),
        )
        self.width = obs_def["width"]
        self.height = obs_def["height"]

    def robustness_trace(self, bt, **kw):
        mu, var = _extract(bt)
        mx, my = mu[..., 0], mu[..., 1]
        vx, vy = var[..., 0], var[..., 1]
        xlo = self.x_traj - self.width / 2
        xhi = self.x_traj + self.width / 2
        ylo = self.y_traj - self.height / 2
        yhi = self.y_traj + self.height / 2
        probs = torch.stack([
            _normal_cdf(xlo, mx, vx),
            1.0 - _normal_cdf(xhi, mx, vx),
            _normal_cdf(ylo, my, vy),
            1.0 - _normal_cdf(yhi, my, vy),
        ], dim=0)
        p, _ = probs.max(dim=0)
        return torch.stack([p, p], dim=-1)
