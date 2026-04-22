"""Monte Carlo verification plots."""

import numpy as np
import matplotlib.pyplot as plt
from visualization.trajectory import draw_env


def plot_mc_verification(mc_result, env, cfg, result, save_path=None):
    """Two-panel figure: sample trajectories + analytic vs empirical bar chart.

    Args:
        mc_result:  dict from mc_verify() with keys p_analytic, p_empirical,
                    samples [N,T+1,nx], successes [N] bool
        env:        Environment (for draw_env)
        cfg:        scenario config dict (for label, horizon)
        result:     PlanResult (for nominal mean trajectory)
        save_path:  optional file path to save the figure

    Returns:
        matplotlib Figure
    """
    label = cfg.get("label", "Scenario")
    T = cfg["horizon"]
    N = mc_result["samples"].shape[0]
    p_analytic = mc_result["p_analytic"]
    p_empirical = mc_result["p_empirical"]
    n_success = int(mc_result["successes"].sum())

    samples_np = mc_result["samples"].cpu().numpy()         # [N, T+1, nx]
    successes_np = mc_result["successes"].cpu().numpy()     # [N] bool
    mu_np = result.mu_trace.detach().cpu().squeeze().numpy()  # [T+1, nx]

    fig, (ax_traj, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: sample trajectories ──────────────────────────────
    draw_env(ax_traj, env)

    fail_idx = np.where(~successes_np)[0]
    succ_idx = np.where(successes_np)[0]

    for i in fail_idx:
        ax_traj.plot(samples_np[i, :, 0], samples_np[i, :, 1],
                     color="#d62728", alpha=0.15, lw=0.6, zorder=1)
    for i in succ_idx:
        ax_traj.plot(samples_np[i, :, 0], samples_np[i, :, 1],
                     color="#2ca02c", alpha=0.15, lw=0.6, zorder=2)

    # Nominal mean trajectory on top
    ax_traj.plot(mu_np[:, 0], mu_np[:, 1],
                 color="#1f77b4", lw=2.5, zorder=3, label="Nominal mean")
    ax_traj.plot(mu_np[0, 0], mu_np[0, 1], "ko", ms=7, zorder=4)
    ax_traj.plot(mu_np[-1, 0], mu_np[-1, 1], "bs", ms=7, zorder=4)

    ax_traj.set_title(
        f"MC Samples  (N={N},  {n_success} pass / {N - n_success} fail)\n"
        f"green = success   red = failure",
        fontsize=11,
    )
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_aspect("equal")
    ax_traj.autoscale_view()

    # ── Right panel: bar chart ────────────────────────────────────────
    labels = ["Analytic P(φ)", "Empirical P̂(φ)"]
    values = [p_analytic, p_empirical]
    colors = ["#1f77b4", "#2ca02c"]

    bars = ax_bar.bar(labels, values, color=colors, width=0.4, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
        )

    ax_bar.axhline(p_analytic, color="#1f77b4", linestyle="--", lw=1.2, alpha=0.6)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_ylabel("Probability", fontsize=12)
    ax_bar.set_title("Analytic vs Empirical P(φ)", fontsize=12, fontweight="bold")

    gap = abs(p_analytic - p_empirical)
    ax_bar.text(0.5, 0.05, f"|Δ| = {gap:.3f}", transform=ax_bar.transAxes,
                ha="center", fontsize=11, color="#555555")

    fig.suptitle(f"{label}  —  Monte Carlo Verification  (T={T})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig
