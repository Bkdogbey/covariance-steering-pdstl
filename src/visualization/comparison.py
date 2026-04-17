"""Side-by-side comparison plots for open-loop vs closed-loop steering."""

import matplotlib.pyplot as plt
from visualization.trajectory import plot_trajectory


def plot_comparison(result_ol, result_cl, env, T, save_path=None):
    """Two-panel trajectory comparison + convergence.

    Args:
        result_ol: PlanResult from open-loop run
        result_cl: PlanResult from closed-loop run
        env: Environment
        T: horizon
        save_path: optional file path to save figure
    """
    mu_ol = result_ol.mu_trace.cpu().squeeze().numpy()
    S_ol = result_ol.Sigma_trace.cpu().squeeze().numpy()
    mu_cl = result_cl.mu_trace.cpu().squeeze().numpy()
    S_cl = result_cl.Sigma_trace.cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_trajectory(
        axes[0], mu_ol, S_ol, env, T,
        title=f"Open-Loop (K≡0)  |  P(φ)={result_ol.best_p:.3f}",
    )
    plot_trajectory(
        axes[1], mu_cl, S_cl, env, T,
        title=f"Cov Steering  |  P(φ)={result_cl.best_p:.3f}",
    )

    fig.suptitle("pdSTL + Covariance Steering", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
