"""Closed-loop covariance steering (Okamoto et al., Theorem 1).

Control law:  u_k = sat(v_k + K_k (x_k - μ_k))
Covariance:   Σ_{k+1} = (A + B K_eff_k) Σ_k (A + B K_eff_k)^T + D D^T

where K_eff_k = diag(sech²(v_k) · u_max) @ K_k is the effective gain
after linearising the tanh saturation around the mean. This ensures the
covariance model is consistent with the actual saturating policy: when
v_k is deep in saturation (|v_k| large), sech²(v_k) → 0 and K has no
effect on covariance, which is correct.
"""

import torch
from steering.base import BaseSteerer


class ClosedLoopSteerer(BaseSteerer):

    def _step_covariance(self, Sigma, K_t, V_t):
        # Jacobian of tanh saturation at V_t: sech²(v) * u_max
        gain_scale = self.dyn.u_max * (1.0 - torch.tanh(V_t) ** 2)  # [nu]
        K_eff = K_t * gain_scale.unsqueeze(-1)                        # [nu, nx]
        A_cl = self.dyn.A + self.dyn.B @ K_eff                        # [nx, nx]
        return A_cl @ Sigma @ A_cl.T + self.dyn.DDT
