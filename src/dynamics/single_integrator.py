"""Single integrator: x_{k+1} = x_k + u_k dt + D w_k."""

import torch
from dynamics.base import BaseDynamics


class SingleIntegrator(BaseDynamics):

    def __init__(self, dt=0.2, u_max=1.0, D_diag=0.05, device="cpu"):
        super().__init__(dt=dt, u_max=u_max, D_diag=D_diag, nx=2, nu=2, device=device)

    def _build_matrices(self, D_diag):
        self.register_buffer("_A", torch.eye(2, device=self.device))
        self.register_buffer("_B", torch.eye(2, device=self.device) * self.dt)
        D = torch.ones(2, device=self.device) * D_diag
        self.register_buffer("_DDT", torch.diag(D ** 2))
