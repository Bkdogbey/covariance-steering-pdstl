"""
Covariance Steering + pdSTL — Main Entry Point
================================================

All experiments are controlled from here using skip_run blocks.
Toggle "run" / "skip" to select which experiments to execute.

Usage:
    cd cov-steer-stl
    PYTHONPATH=src python main.py
"""

import torch
import numpy as np
import matplotlib
import os

# Use non-interactive backend if no display
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import get_device, load_config, load_scenario, skip_run
from dynamics import get_dynamics
from steering import get_steerer
from planning import get_planner, build_environment
from visualization import plot_comparison, plot_convergence, plot_trajectory


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def setup_scenario(scenario_path, device=None):
    """Load config → build dynamics, steerer, env, initial belief.

    Returns:
        cfg, dynamics, steerer, env, mu0, Sigma0
    """
    device = device or get_device()
    cfg, dyn_cfg = load_scenario(scenario_path)

    dynamics = get_dynamics(dyn_cfg, device)
    steering_mode = cfg.get("planner", {}).get("steering", "closed_loop")
    steerer = get_steerer(steering_mode, dynamics)
    env = build_environment(cfg, device)

    init = cfg["initial_state"]
    mu0 = torch.tensor(init["mean"], dtype=torch.float32, device=device)
    Sigma0 = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

    return cfg, dynamics, steerer, env, mu0, Sigma0


def run_scenario(scenario_path, verbose=True):
    """Run a single scenario end-to-end. Returns PlanResult."""
    device = get_device()
    cfg, dynamics, steerer, env, mu0, Sigma0 = setup_scenario(scenario_path, device)
    planner = get_planner(cfg, dynamics, steerer, env)
    result = planner.solve(mu0, Sigma0, verbose=verbose)
    return result, env, cfg


def run_comparison(scenario_path, save_dir="data/results"):
    """Run open-loop vs closed-loop on the same scenario, plot side-by-side."""
    device = get_device()
    cfg, dyn_cfg = load_scenario(scenario_path)
    dynamics = get_dynamics(dyn_cfg, device)
    env = build_environment(cfg, device)
    T = cfg["horizon"]

    init = cfg["initial_state"]
    mu0 = torch.tensor(init["mean"], dtype=torch.float32, device=device)
    Sigma0 = torch.diag(torch.tensor(init["cov_diag"], dtype=torch.float32, device=device))

    # ── Open-loop baseline ───────────────────────────────────────
    print("\n── Open-Loop (K ≡ 0) ──")
    steerer_ol = get_steerer("open_loop", dynamics)
    # For open-loop, zero out K learning rate so it stays at zero
    cfg_ol = {**cfg, "optimizer": {**cfg["optimizer"], "lr_k": 0.0}}
    from planning.single_shot import SingleShotPlanner
    planner_ol = SingleShotPlanner(dynamics, steerer_ol, env, cfg_ol)
    result_ol = planner_ol.solve(mu0, Sigma0, T=T, verbose=True)

    # ── Closed-loop covariance steering ──────────────────────────
    print("\n── Closed-Loop (K optimised) ──")
    steerer_cl = get_steerer("closed_loop", dynamics)
    planner_cl = SingleShotPlanner(dynamics, steerer_cl, env, cfg)
    result_cl = planner_cl.solve(mu0, Sigma0, T=T, verbose=True)

    # ── Print summary ────────────────────────────────────────────
    S_end_ol = result_ol.Sigma_trace[0, -1, :2, :2].cpu().numpy()
    S_end_cl = result_cl.Sigma_trace[0, -1, :2, :2].cpu().numpy()
    det_ol = np.linalg.det(S_end_ol)
    det_cl = np.linalg.det(S_end_cl)
    print(f"\n  Summary:")
    print(f"    Open-loop  P(φ) = {result_ol.best_p:.4f},  det(Σ_end) = {det_ol:.2e}")
    print(f"    Cov-steer  P(φ) = {result_cl.best_p:.4f},  det(Σ_end) = {det_cl:.2e}")
    if det_cl > 1e-15:
        print(f"    Covariance reduction: {det_ol / det_cl:.1f}x")
    print(f"    ||K||_F = {result_cl.K.norm().item():.4f}")

    # ── Plots ────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    label = cfg.get("label", "comparison").lower().replace(" ", "_")

    fig = plot_comparison(result_ol, result_cl, env, T,
                          save_path=os.path.join(save_dir, f"{label}_trajectories.png"))
    plt.show()

    fig = plot_convergence(
        [
            {"p_sat": result_ol.history, "loss": result_ol.history},
            {"p_sat": result_cl.history, "loss": result_cl.history},
        ],
        labels=["Open-Loop", "Cov Steering"],
        save_path=os.path.join(save_dir, f"{label}_convergence.png"),
    )
    plt.show()

    return result_ol, result_cl


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════

# ── 1. Narrow Gap: Open-Loop vs Covariance Steering ─────────────
with skip_run("run", "Narrow Gap — Open-Loop vs Cov Steering") as check, check():
    run_comparison("configs/scenarios/narrow_gap.yaml")

# ── 2. Obstacle Field (single run) ──────────────────────────────
with skip_run("skip", "Obstacle Field") as check, check():
    result, env, cfg = run_scenario("configs/scenarios/obstacle_field.yaml")
    mu_np = result.mu_trace.cpu().squeeze().numpy()
    S_np = result.Sigma_trace.cpu().squeeze().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_trajectory(ax, mu_np, S_np, env, cfg["horizon"],
                    title=f"Obstacle Field  |  P(φ)={result.best_p:.3f}")
    plt.tight_layout()
    plt.show()

# ── 3. Lane Change MPC ──────────────────────────────────────────
with skip_run("skip", "Lane Change MPC") as check, check():
    result, env, cfg = run_scenario("configs/scenarios/lane_change.yaml")
    print(f"  Lane change done. P(φ) = {result.best_p:.4f}")
