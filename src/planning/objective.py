"""Shared loss computation for all planner types.

J = w_phi * L_phi  +  w_u * ||u||²  +  w_du * ||Δu||²
  + w_K  * ||K||²_F  +  w_dist * ||μ_T - goal||²
"""

import torch


def compute_loss(p_sat, V, K, mu_trace, env, dyn, weights):
    """Compute the total scalar objective.

    Args:
        p_sat:     scalar tensor — SRM(φ, B) at t=0
        V:         [T, nu] — unconstrained feedforward params
        K:         [T, nu, nx] — feedback gains
        mu_trace:  [1, T+1, nx] — mean trajectory
        env:       Environment — for goal center
        dyn:       BaseDynamics — for bound_control
        weights:   dict with w_phi, w_u, w_du, w_K, w_dist

    Returns:
        J: scalar tensor (differentiable)
    """
    w = weights
    device = V.device

    # STL satisfaction
    loss_phi = -torch.log(p_sat + 1e-4)

    # Control effort
    u_seq = dyn.bound_control(V)
    loss_u = torch.sum(u_seq ** 2)

    # Control smoothness
    u_diff = u_seq[1:] - u_seq[:-1]
    loss_du = torch.sum(u_diff ** 2) + torch.sum(u_seq[0] ** 2)

    # Feedback gain regularization
    loss_K = torch.sum(K ** 2)

    # Terminal distance to goal
    loss_dist = torch.tensor(0.0, device=device)
    if env.goal is not None:
        gx = (env.goal["x"][0] + env.goal["x"][1]) / 2.0
        gy = (env.goal["y"][0] + env.goal["y"][1]) / 2.0
        goal_xy = torch.tensor([gx, gy], device=device)
        loss_dist = torch.sum((mu_trace[0, -1, :2] - goal_xy) ** 2)

    return (
        w["w_phi"]  * loss_phi
        + w["w_u"]  * loss_u
        + w["w_du"] * loss_du
        + w["w_K"]  * loss_K
        + w["w_dist"] * loss_dist
    )
