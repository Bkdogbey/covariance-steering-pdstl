"""Abstract dynamics interface.

Every dynamics model exposes:
    A, B, DDT  — system matrices (properties)
    nx, nu     — state/control dimensions
    step(mu, Sigma, u) → (mu_next, Sigma_next)  open-loop single step
    bound_control(v) → u                         smooth saturation
"""

import torch
import torch.nn as nn
from abc import abstractmethod


class BaseDynamics(nn.Module):
    """Abstract base for discrete-time LTI systems: x_{k+1} = A x_k + B u_k + D w_k."""

    def __init__(self, dt, u_max, D_diag, nx, nu, device="cpu"):
        super().__init__()
        self.dt = dt
        self.u_max = u_max
        self.nx = nx
        self.nu = nu
        self.device = device
        self._build_matrices(D_diag)

    @abstractmethod
    def _build_matrices(self, D_diag):
        """Subclass must set self._A, self._B, self._DDT as buffers."""
        ...

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def DDT(self):
        return self._DDT

    def bound_control(self, v):
        """Smooth saturation: v ∈ ℝ → u ∈ [-u_max, u_max]."""
        return self.u_max * torch.tanh(v)

    def step(self, mu, Sigma, u):
        """Open-loop single step (no feedback gain).

        Args:
            mu:    [nx]       state mean
            Sigma: [nx, nx]   state covariance
            u:     [nu]       control input

        Returns:
            mu_next, Sigma_next
        """
        mu_next = self._A @ mu + self._B @ u
        Sigma_next = self._A @ Sigma @ self._A.T + self._DDT
        return mu_next, Sigma_next
