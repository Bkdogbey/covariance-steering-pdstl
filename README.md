# Covariance Steering + pdSTL

Differentiable covariance steering integrated with probabilistic Signal Temporal Logic for safe trajectory planning under uncertainty.

## Quickstart

```bash
pip install torch numpy matplotlib pyyaml scipy
cd cov-steer-stl
PYTHONPATH=src python main.py
```

## What it does

Jointly optimises a feedforward control sequence **V** and feedback gains **K** so that the closed-loop belief trajectory satisfies a temporal logic specification with high probability. The feedback gains actively shape the covariance — shrinking uncertainty where constraints are tight — rather than passively routing around it.

## Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline explanation.

```
main.py              ← single entry point, toggle experiments here
src/
  dynamics/          ← system models (A, B, D matrices)
  steering/          ← open-loop vs closed-loop covariance propagation
  stl/               ← probabilistic STL operators and predicates
  planning/          ← optimisers (single-shot, MPC)
  visualization/     ← trajectory plots, convergence curves
  utils/             ← config loading, device selection
configs/             ← YAML configs for dynamics, scenarios, defaults
```

## Adding a new component

- **Dynamics model:** subclass `BaseDynamics`, implement `_build_matrices()`, register in `dynamics/__init__.py`
- **Steering mode:** subclass `BaseSteerer`, implement `_step_covariance()`, register in `steering/__init__.py`
- **Planner type:** subclass `BasePlanner`, implement `solve()`, register in `planning/__init__.py`
- **Scenario:** write a YAML in `configs/scenarios/`, reference it in `main.py`
