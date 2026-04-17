# Architecture

## Overview

This project integrates **covariance steering** (Okamoto et al.) with
**probabilistic differentiable Signal Temporal Logic** (pdSTL) for
safe trajectory planning under uncertainty. The optimizer jointly
tunes a feedforward control sequence **V** and feedback gains **K**
so that the closed-loop belief trajectory satisfies a temporal logic
specification with high probability.

## Data Flow

```
main.py
  │
  ├── load_scenario(yaml)
  │     ├── dynamics config  →  get_dynamics(cfg)  →  BaseDynamics
  │     ├── scenario config  →  build_environment() →  Environment
  │     └── steering mode    →  get_steerer(name)   →  BaseSteerer
  │
  └── get_planner(cfg, dynamics, steerer, env)  →  BasePlanner
        │
        │  planner.solve(μ₀, Σ₀)
        │    │
        │    │  ┌─── Optimization Loop ───────────────────────────┐
        │    │  │                                                  │
        │    │  │  (V, K) ─→ steerer.forward(V, K, μ₀, Σ₀)       │
        │    │  │              │                                   │
        │    │  │              │  for t = 0..T-1:                  │
        │    │  │              │    μ_{t+1} = A μ_t + B tanh(v_t)  │
        │    │  │              │    Σ_{t+1} = A_cl Σ_t A_cl' + DD' │
        │    │  │              │    where A_cl = A + B K_t          │
        │    │  │              │                                   │
        │    │  │              └─→ (μ_trace, Σ_trace)              │
        │    │  │                      │                           │
        │    │  │              wrap as BeliefTrajectory             │
        │    │  │                      │                           │
        │    │  │              spec(traj) ─→ STL evaluation         │
        │    │  │                      │    (RNN backward pass)    │
        │    │  │                      │                           │
        │    │  │              p_sat = SRM(φ, B) at t=0            │
        │    │  │                      │                           │
        │    │  │              compute_loss(p_sat, V, K, ...)      │
        │    │  │                      │                           │
        │    │  │              J.backward()  →  ∇V, ∇K             │
        │    │  │              optimizer.step()                    │
        │    │  │                                                  │
        │    │  └──────────────────────────────────────────────────┘
        │    │
        │    └─→ PlanResult(μ_trace, Σ_trace, V, K, best_p)
        │
        └── visualization
```

## Module Responsibilities

### `src/dynamics/`
Owns the system matrices **(A, B, DDᵀ)** and provides a single-step
open-loop update. Each model is a subclass of `BaseDynamics`.

| File | Class | State | Control |
|------|-------|-------|---------|
| `single_integrator.py` | `SingleIntegrator` | [x, y] | [vx, vy] |
| `double_integrator.py` | `DoubleIntegrator` | [px, py, vx, vy] | [ax, ay] |

**Adding a new model:** subclass `BaseDynamics`, implement `_build_matrices()`,
register in `__init__._REGISTRY`.

### `src/steering/`
Owns the **covariance propagation logic**. This is where the covariance
steering contribution lives. Both steerers share the same interface:

```python
result = steerer(V, K, μ₀, Σ₀)  →  RolloutResult(μ_trace, Σ_trace)
```

| File | Σ update | K used? |
|------|----------|---------|
| `open_loop.py` | `A Σ Aᵀ + DDᵀ` | No (ignored) |
| `closed_loop.py` | `(A+BK) Σ (A+BK)ᵀ + DDᵀ` | **Yes** |

The planner doesn't know which one it's using — same output signature.

### `src/stl/`
Ported from the pdSTL paper. Three sub-modules:

- **`base.py`** — `GaussianBelief`, `BeliefTrajectory`
- **`operators.py`** — `Always`, `Eventually`, `Until`, `And`, `Or`, `Negation`
  with backward RNN evaluation (O(T) complexity)
- **`predicates.py`** — `RectangularGoalPredicate`, `RectangularObstaclePredicate`,
  `CircularObstaclePredicate`, `MovingRectangularObstaclePredicate`

Predicates read `(μ, Σ)` from beliefs and return `[B, T, 2]` probability
bounds via the Gaussian CDF. Because Σ depends on K (in closed-loop mode),
gradients flow from the STL loss through the predicate probabilities back to K.

### `src/planning/`
Ties everything together. Key design: **no loss logic in planner subclasses.**

- **`objective.py`** — single `compute_loss()` function used by all planners
- **`base.py`** — `BasePlanner` with `_optimize_step()`, parameter init, belief wrapping
- **`single_shot.py`** — fixed-horizon loop calling `_optimize_step()` repeatedly
- **`mpc.py`** — receding-horizon, internally creates `SingleShotPlanner` per window
- **`environment.py`** — `Environment` class + `build_environment(cfg)`

**Adding a new planner type:** subclass `BasePlanner`, implement `solve()`,
register in `__init__._REGISTRY`.

### `src/visualization/`
Stateless plotting functions that accept numpy arrays or `PlanResult` objects.

### `src/utils/`
Config loading, device selection, `skip_run` for experiment control.

## Config Structure

```
configs/
├── defaults.yaml              # optimizer weights, learning rates
├── dynamics/
│   ├── single_integrator.yaml # A, B, D params
│   └── double_integrator.yaml
└── scenarios/
    ├── narrow_gap.yaml        # initial state, obstacles, goal, planner type
    └── ...
```

Scenario YAML references a dynamics config by path. The `load_scenario()`
function merges scenario overrides on top of `defaults.yaml`.

## Key Equations

**Control law** (Okamoto, Theorem 1):
```
u_k = v_k + K_k y_k
```
where `v_k` is feedforward and `K_k` is the feedback gain.

**Mean propagation** (depends on V only):
```
μ_{k+1} = A μ_k + B u_max tanh(v_k)
```

**Covariance propagation** (depends on K):
```
Σ_{k+1} = (A + B K_k) Σ_k (A + B K_k)ᵀ + D Dᵀ
```

**Predicate probability** (connects Σ to STL):
```
P(aᵀx ≤ c) = Φ((c - aᵀμ_t) / √(aᵀΣ_t a))
```

**Loss** (shared by all planners):
```
J = w_φ · (−log SRM) + w_u · ||u||² + w_K · ||K||²_F + w_d · ||μ_T − goal||²
```

## Gradient Path

The chain that makes this work:

```
K  →  A_cl = A + BK
   →  Σ_{k+1} = A_cl Σ A_cl' + DD'
   →  var_diag = diag(Σ)
   →  P(safe) = 1 - Φ(threshold | μ, var_diag)
   →  SRI bounds via Always/Eventually RNN
   →  SRM = lower bound at t=0
   →  J = -log(SRM)
   →  ∂J/∂K via autograd
```

Every operation is differentiable. The Gaussian CDF `Φ` is differentiable
via `torch.erf`. The RNN temporal operators use smooth log-sum-exp
approximations. The `tanh` control saturation is smooth everywhere.
