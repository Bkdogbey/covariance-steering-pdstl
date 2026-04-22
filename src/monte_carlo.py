"""Monte Carlo verification for covariance steering + pdSTL plans.

Simulates N stochastic trajectories using the optimized (V, K) and evaluates
STL satisfaction empirically to validate analytic probability estimates.

Control law used in simulation:
    u_k = u_max * tanh(v_k) + K_k @ (x_k - mu_k)
    x_{k+1} = A x_k + B u_k + L_D w_k,  w_k ~ N(0, I)

where mu_k is the nominal mean from the planner's best rollout.
"""

import torch
from stl.base import GaussianBelief, BeliefTrajectory


def sample_trajectories(dynamics, result, mu0, Sigma0, n_samples, device="cpu"):
    """Simulate N stochastic trajectories using the optimized (V, K).

    Args:
        dynamics:  BaseDynamics with A, B, DDT, bound_control
        result:    PlanResult with mu_trace [1,T+1,nx], V [T,nu], K [T,nu,nx]
        mu0:       [nx] initial mean
        Sigma0:    [nx, nx] initial covariance
        n_samples: number of Monte Carlo rollouts
        device:    torch device string or object

    Returns:
        Tensor [N, T+1, nx] — one state trajectory per sample
    """
    device = torch.device(device)
    T = result.V.shape[0]
    nx = dynamics.nx

    jitter = 1e-9 * torch.eye(nx, device=device)
    L0 = torch.linalg.cholesky(Sigma0.to(device) + jitter)
    L_D = torch.linalg.cholesky(dynamics.DDT.to(device) + jitter)

    # Sample initial states from N(mu0, Sigma0): [N, nx]
    x = mu0.to(device) + (L0 @ torch.randn(nx, n_samples, device=device)).T

    # Nominal mean trajectory from planner: [T+1, nx]
    mu_nom = result.mu_trace[0].to(device)

    # Pre-compute bounded feedforward controls: [T, nu]
    u_ff_seq = dynamics.bound_control(result.V.to(device))
    K_seq = result.K.to(device)

    states = [x.clone()]

    with torch.no_grad():
        for t in range(T):
            delta_x = x - mu_nom[t]                          # [N, nx]
            u_fb = delta_x @ K_seq[t].T                      # [N, nu]
            u = u_ff_seq[t] + u_fb                           # [N, nu]

            w = torch.randn(n_samples, nx, device=device)    # [N, nx]
            x = x @ dynamics.A.T + u @ dynamics.B.T + w @ L_D.T
            states.append(x.clone())

    return torch.stack(states, dim=1)  # [N, T+1, nx]


def eval_spec_empirical(samples, spec, device="cpu", eps=1e-8):
    """Evaluate STL spec on N point trajectories and return empirical P(phi).

    Each sample trajectory is evaluated by treating it as a Gaussian with
    near-zero covariance (eps), making predicates return effectively 0 or 1.

    Args:
        samples: [N, T+1, nx] sample trajectories
        spec:    STL_Formula (same spec used during planning)
        device:  torch device
        eps:     covariance floor for point-trajectory evaluation

    Returns:
        (p_empirical, successes) where successes is a bool tensor [N]
    """
    device = torch.device(device)
    N, T_plus_1, nx = samples.shape
    successes = torch.zeros(N, dtype=torch.bool)

    with torch.no_grad():
        for i in range(N):
            beliefs = []
            for t in range(T_plus_1):
                mean = samples[i, t].to(device).unsqueeze(0)      # [1, nx]
                var = torch.full((1, nx), eps, device=device)     # [1, nx]
                beliefs.append(GaussianBelief(mean, var))
            bt = BeliefTrajectory(beliefs)
            p = spec(bt)[0, 0, 0].item()
            successes[i] = p > 0.5

    p_empirical = successes.float().mean().item()
    return p_empirical, successes


def mc_verify(result, dynamics, spec, mu0, Sigma0, n_samples=500, device="cpu"):
    """Run Monte Carlo verification on a solved plan.

    Args:
        result:    PlanResult from planner.solve()
        dynamics:  BaseDynamics
        spec:      STL_Formula (same spec used during planning)
        mu0:       [nx] initial mean
        Sigma0:    [nx, nx] initial covariance
        n_samples: number of MC rollouts
        device:    torch device string

    Returns:
        dict with keys:
            p_analytic  — float, analytic P(phi) from planner
            p_empirical — float, empirical fraction of satisfied samples
            samples     — [N, T+1, nx] simulated trajectories
            successes   — [N] bool tensor
    """
    samples = sample_trajectories(dynamics, result, mu0, Sigma0, n_samples, device)
    p_empirical, successes = eval_spec_empirical(samples, spec, device)
    return {
        "p_analytic":  result.best_p,
        "p_empirical": p_empirical,
        "samples":     samples,
        "successes":   successes,
    }
