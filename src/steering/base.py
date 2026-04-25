"""Abstract steering interface.

A Steerer takes (V, K, μ₀, Σ₀) and a dynamics model, and produces
the full belief trajectory (μ_trace, Σ_trace). The planner doesn't
care whether covariance is open-loop or closed-loop — it gets the
same output signature either way.
"""

import torch
import torch.nn as nn
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class RolloutResult:
    """Output of a steering rollout."""
    mu_trace: torch.Tensor     # [1, T+1, nx]
    Sigma_trace: torch.Tensor  # [1, T+1, nx, nx]


class BaseSteerer(nn.Module):
    """Abstract base. Subclasses implement _step_covariance."""

    def __init__(self, dynamics):
        super().__init__()
        self.dyn = dynamics

    @abstractmethod
    def _step_covariance(self, Sigma, K_t, V_t):
        """Propagate covariance one step.

        Args:
            Sigma: [nx, nx]  current covariance
            K_t:   [nu, nx]  feedback gain at this step
            V_t:   [nu]      pre-saturation feedforward at this step

        Returns:
            Sigma_next: [nx, nx]
        """
        ...

    def forward(self, V, K, mu0, Sigma0):
        """Roll out the full trajectory.

        Args:
            V:      [T, nu]       unconstrained feedforward params
            K:      [T, nu, nx]   feedback gains
            mu0:    [nx]          initial mean
            Sigma0: [nx, nx]      initial covariance

        Returns:
            RolloutResult with mu_trace [1, T+1, nx], Sigma_trace [1, T+1, nx, nx]
        """
        T = V.shape[0]
        mu, Sigma = mu0, Sigma0
        mus, Sigmas = [mu], [Sigma]

        for t in range(T):
            u_ff = self.dyn.bound_control(V[t])
            mu = self.dyn.A @ mu + self.dyn.B @ u_ff
            Sigma = self._step_covariance(Sigma, K[t], V[t])
            mus.append(mu)
            Sigmas.append(Sigma)

        return RolloutResult(
            mu_trace=torch.stack(mus).unsqueeze(0),
            Sigma_trace=torch.stack(Sigmas).unsqueeze(0),
        )
