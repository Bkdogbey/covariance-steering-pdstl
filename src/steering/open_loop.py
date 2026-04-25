"""Open-loop covariance propagation (K ≡ 0 baseline).

Σ_{k+1} = A Σ_k A^T + D D^T

The feedback gain K is accepted but ignored, so the planner
interface stays identical to closed-loop.
"""

from steering.base import BaseSteerer


class OpenLoopSteerer(BaseSteerer):

    def _step_covariance(self, Sigma, K_t, V_t):
        A = self.dyn.A
        return A @ Sigma @ A.T + self.dyn.DDT
