"""Double integrator: position-velocity model with acceleration control."""

import torch
from dynamics.base import BaseDynamics


class DoubleIntegrator(BaseDynamics):

    def __init__(self, dt=0.2, u_max=2.5, D_diag=0.03, device="cpu"):
        super().__init__(dt=dt, u_max=u_max, D_diag=D_diag, nx=4, nu=2, device=device)

    def _build_matrices(self, D_diag):
        dt = self.dt
        self.register_buffer("_A", torch.tensor([
            [1., 0., dt, 0.],
            [0., 1., 0., dt],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ], device=self.device))

        self.register_buffer("_B", torch.tensor([
            [0.5 * dt**2, 0.],
            [0., 0.5 * dt**2],
            [dt, 0.],
            [0., dt],
        ], device=self.device))

        D = torch.ones(4, device=self.device) * D_diag
        self.register_buffer("_DDT", torch.diag(D ** 2))
