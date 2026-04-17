"""MPC planner: receding-horizon covariance steering.

At each step:
  1. Solve a SingleShotPlanner over horizon H
  2. Apply first control + feedback gain
  3. Shift horizon forward, warm-start from previous solution
"""

import torch
from planning.base import BasePlanner, PlanResult
from planning.single_shot import SingleShotPlanner


class MPCPlanner(BasePlanner):

    def solve(self, mu0, Sigma0, spec=None, init_V=None, verbose=True):
        H = self.cfg["horizon"]
        max_steps = self.cfg.get("sim_steps", 200)

        curr_mu = mu0
        curr_Sigma = Sigma0

        # Trace accumulators
        mu_list = [curr_mu]
        Sigma_list = [curr_Sigma]
        V_list, K_list = [], []
        p_trace = []
        prev_V = init_V

        for step in range(max_steps):
            # Build local spec (may change per step, e.g. moving obstacles)
            local_spec = spec or self.env.get_specification(H)

            # Solve fixed-horizon subproblem
            sub = SingleShotPlanner(self.dyn, self.steerer, self.env, self.cfg)
            result = sub.solve(curr_mu, curr_Sigma, T=H, spec=local_spec,
                               init_V=prev_V, verbose=False)

            p_trace.append(result.best_p)
            V_list.append(result.V[0])
            K_list.append(result.K[0])

            # Apply first control
            u_ff = self.dyn.bound_control(result.V[0])
            curr_mu = self.dyn.A @ curr_mu + self.dyn.B @ u_ff

            # Closed-loop covariance step with K_0
            A_cl = self.dyn.A + self.dyn.B @ result.K[0]
            curr_Sigma = A_cl @ curr_Sigma @ A_cl.T + self.dyn.DDT

            # Add process noise sample (simulation)
            noise = torch.distributions.MultivariateNormal(
                torch.zeros_like(curr_mu), self.dyn.DDT
            ).sample()
            curr_mu = curr_mu + noise

            mu_list.append(curr_mu)
            Sigma_list.append(curr_Sigma)

            # Warm-start: shift previous solution
            prev_V = torch.cat([result.V[1:], result.V[-1:]], dim=0)

            if verbose and step % 10 == 0:
                print(f"  MPC step {step:3d} | P(φ)={result.best_p:.4f} "
                      f"| pos=[{curr_mu[0]:.2f}, {curr_mu[1]:.2f}]")

            # Check termination (can be overridden)
            if self._check_done(curr_mu, step):
                if verbose:
                    print(f"  MPC done at step {step}")
                break

        return PlanResult(
            mu_trace=torch.stack(mu_list).unsqueeze(0),
            Sigma_trace=torch.stack(Sigma_list).unsqueeze(0),
            V=torch.stack(V_list),
            K=torch.stack(K_list),
            best_p=max(p_trace) if p_trace else 0.0,
            history=p_trace,
        )

    def _check_done(self, mu, step):
        """Override for custom termination. Default: goal reached."""
        if self.env.goal is None:
            return False
        gx = (self.env.goal["x"][0] + self.env.goal["x"][1]) / 2.0
        gy = (self.env.goal["y"][0] + self.env.goal["y"][1]) / 2.0
        goal = torch.tensor([gx, gy], device=mu.device)
        return torch.norm(mu[:2] - goal).item() < 0.5
