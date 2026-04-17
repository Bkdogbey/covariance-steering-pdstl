"""Convergence curve plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(histories, labels=None, colors=None, save_path=None):
    """Plot P(φ) and/or loss convergence for one or more runs.

    Args:
        histories: list of dicts, each with keys 'p_sat', 'loss', optionally 'det_trace'
        labels: list of strings
        colors: list of color strings
    """
    n = len(histories)
    labels = labels or [f"Run {i}" for i in range(n)]
    colors = colors or ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"][:n]

    has_det = any("det_trace" in h for h in histories)
    n_rows = 3 if has_det else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.2 * n_rows), sharex=True)

    for h, lbl, c in zip(histories, labels, colors):
        if "p_sat" in h:
            axes[0].plot(h["p_sat"], label=lbl, color=c, lw=1.5)
        if "loss" in h:
            axes[1].plot(h["loss"], label=lbl, color=c, lw=1.5)
        if has_det and "det_trace" in h:
            axes[2].semilogy(h["det_trace"], label=lbl, color=c, lw=1.5)

    axes[0].set_ylabel("P(φ)", fontsize=12)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    if has_det:
        axes[2].set_ylabel("det(Σ_pos)", fontsize=12)
        axes[2].set_xlabel("Iteration", fontsize=12)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel("Iteration", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
