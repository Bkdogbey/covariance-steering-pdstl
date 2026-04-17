"""Estimate process noise covariance D from flight data.

Used to bridge hardware experiments with the covariance steering pipeline.
Feed Crazyflie tracking residuals → fit diagonal D → plug into dynamics.
"""

import torch
import numpy as np


def estimate_D_from_residuals(residuals, dt=0.2):
    """Estimate diagonal process noise matrix from tracking residuals.

    Args:
        residuals: np.ndarray [N, nx] — per-timestep tracking errors
        dt: float — timestep (to normalize to per-step noise)

    Returns:
        D_diag: np.ndarray [nx] — diagonal of D such that DDᵀ ≈ sample covariance
    """
    cov = np.cov(residuals, rowvar=False)  # [nx, nx]
    # D_diag² ≈ diag(cov), so D_diag = sqrt(diag(cov))
    return np.sqrt(np.diag(cov))


def make_DDT(D_diag, device="cpu"):
    """Build DDᵀ matrix from diagonal D vector."""
    D = torch.tensor(D_diag, dtype=torch.float32, device=device)
    return torch.diag(D ** 2)
