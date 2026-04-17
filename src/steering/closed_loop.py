"""Closed-loop covariance steering (Okamoto et al., Theorem 1).

Control law:  u_k = v_k + K_k y_k
Covariance:   Σ_{k+1} = (A + B K_k) Σ_k (A + B K_k)^T + D D^T

The feedback gain K_k appears inside the covariance propagation,
making Σ a differentiable function of K — enabling gradient-based
joint optimization of (V, K) through the pdSTL computation graph.
"""

from steering.base import BaseSteerer


class ClosedLoopSteerer(BaseSteerer):

    def _step_covariance(self, Sigma, K_t):
        A_cl = self.dyn.A + self.dyn.B @ K_t      # [nx, nx]
        return A_cl @ Sigma @ A_cl.T + self.dyn.DDT
