# Covariance Steering + pdSTL: Full Technical Reference

---

## 1. System Model

We use a discrete-time double integrator with Gaussian process noise:

$$x_{k+1} = A x_k + B u_k + w_k, \qquad w_k \sim \mathcal{N}(0,\, DD^T)$$

**State and control:**

$$x_k = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \end{bmatrix} \in \mathbb{R}^4, \qquad u_k = \begin{bmatrix} a_x \\ a_y \end{bmatrix} \in \mathbb{R}^2$$

**System matrices** (timestep $\Delta t = 0.2\,\text{s}$):

$$A = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \qquad B = \begin{bmatrix} \tfrac{1}{2}\Delta t^2 & 0 \\ 0 & \tfrac{1}{2}\Delta t^2 \\ \Delta t & 0 \\ 0 & \Delta t \end{bmatrix}$$

**Process noise:** $D = d \cdot I_4$, $d = 0.05$, so $DD^T = d^2 I_4$.

**Control saturation:** $u_k = u_{\max} \tanh(v_k)$, where $v_k \in \mathbb{R}^2$ are the unconstrained optimization variables and $u_{\max} = 2.5\,\text{m/s}^2$.

---

## 2. Belief Propagation

The state distribution is tracked as a Gaussian belief $b_k = (\mu_k, \Sigma_k)$.

### 2.1 Mean Dynamics

Since $\mathbb{E}[K_k(x_k - \mu_k)] = 0$, the mean only depends on the feedforward:

$$\mu_{k+1} = A\mu_k + B\,u_{\max}\tanh(v_k)$$

### 2.2 Open-Loop Covariance (Baseline, $K \equiv 0$)

$$\Sigma_{k+1} = A\Sigma_k A^T + DD^T$$

Covariance grows monotonically — the system has no mechanism to reject noise.

### 2.3 Closed-Loop Covariance Steering

The closed-loop policy is:

$$u_k = \mathrm{sat}\!\left(v_k + K_k(x_k - \mu_k)\right)$$

Linearising the tanh saturation about the mean gives the effective gain:

$$K^{\mathrm{eff}}_k = \mathrm{diag}\!\left(u_{\max}\,\mathrm{sech}^2(v_k)\right) K_k \in \mathbb{R}^{n_u \times n_x}$$

where $\mathrm{sech}^2(v) = 1 - \tanh^2(v)$.  The covariance then propagates as:

$$\Sigma_{k+1} = A_k^{\mathrm{cl}}\,\Sigma_k\,(A_k^{\mathrm{cl}})^T + DD^T, \qquad A_k^{\mathrm{cl}} = A + B K^{\mathrm{eff}}_k$$

**Why this matters:** When $|v_k|$ is large (control near saturation), $\mathrm{sech}^2(v_k) \to 0$ and the feedback gain has no effect on covariance — the model correctly captures that a saturated actuator cannot respond to deviations.  The gradient flows back through $\mathrm{sech}^2(v_k)$, coupling the optimization of $V$ and $K$.

---

## 3. Probabilistic STL (pdSTL)

### 3.1 Atomic Predicates

All predicates evaluate $P(\text{condition} \mid x_k \sim \mathcal{N}(\mu_k, \Sigma_k))$ using the normal CDF $\Phi$.

**Gaussian CDF:**

$$\Phi(z) = \frac{1}{2}\left(1 + \mathrm{erf}\!\left(\frac{z}{\sqrt{2}}\right)\right)$$

**Rectangular goal** $x_k \in [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$:

$$P_{\mathrm{goal}} = \bigl[\Phi(x_{\max};\mu_x,\sigma_x^2) - \Phi(x_{\min};\mu_x,\sigma_x^2)\bigr] \cdot \bigl[\Phi(y_{\max};\mu_y,\sigma_y^2) - \Phi(y_{\min};\mu_y,\sigma_y^2)\bigr]$$

**Rectangular obstacle avoidance** $x_k \notin [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$:

$$P_{\mathrm{obs}} = \max\!\left(\Phi(x_{\min};\mu_x,\sigma_x^2),\; 1-\Phi(x_{\max};\mu_x,\sigma_x^2),\; \Phi(y_{\min};\mu_y,\sigma_y^2),\; 1-\Phi(y_{\max};\mu_y,\sigma_y^2)\right)$$

(Takes the best of the four escape directions — conservative lower bound on $P(\text{outside})$.)

**Circular obstacle avoidance** $\|x_k - c\| > r$: projects variance along the direction to the obstacle center $d = (x-c)/\|x-c\|$:

$$\sigma_{\mathrm{proj}}^2 = d^T \Sigma_{[:2,:2]}\, d, \qquad P_{\mathrm{circ}} = 1 - \Phi(r;\, \|{\mu_{[:2]} - c}\|,\, \sigma_{\mathrm{proj}}^2)$$

### 3.2 Logical Operators

All operators work on probability bound traces of shape $[B, T, 2]$ (lower, upper bounds).

| Operator | Formula |
|---|---|
| $\phi_1 \wedge \phi_2$ | $p_L = p_{L,1} \cdot p_{L,2}$;  $p_U = \min(p_{U,1}, p_{U,2})$ |
| $\phi_1 \vee \phi_2$ | $p_L = \max(p_{L,1}, p_{L,2})$;  $p_U = \min(p_{U,1}+p_{U,2},\, 1)$ |
| $\neg\phi$ | $p_L = 1 - p_{U}$;  $p_U = 1 - p_{L}$ |

### 3.3 Temporal Operators

Implemented as backward RNNs (reverse-process-reverse) for $O(T)$ complexity.

**Smooth min/max** (scale $\lambda > 0$):

$$\widetilde{\min}(x) = -\frac{1}{\lambda}\log\!\sum_i e^{-\lambda x_i}, \qquad \widetilde{\max}(x) = \frac{1}{\lambda}\log\!\sum_i e^{\lambda x_i}$$

| Operator | Semantics | Aggregation |
|---|---|---|
| $\square_{[a,b]}\,\phi$ (Always) | $\phi$ holds for all $t' \in [t+a, t+b]$ | Smooth min over window |
| $\lozenge_{[a,b]}\,\phi$ (Eventually) | $\phi$ holds for some $t' \in [t+a, t+b]$ | Smooth max over window |
| $\phi_1\,\mathcal{U}_{[a,b]}\,\phi_2$ (Until) | $\phi_2$ holds at some $\tau \in [t+a,t+b]$; $\phi_1$ holds until $\tau$ | Nested min/max |

**Output:** $\mathrm{SRM}(\phi, \mathcal{B}) = p_{L}[0, 0] \in [0,1]$ — the lower-bound satisfaction probability at $t=0$.

---

## 4. Optimization Problem

We jointly optimize the feedforward sequence $V = [v_0,\ldots,v_{T-1}] \in \mathbb{R}^{T \times n_u}$ and feedback gains $K = [K_0,\ldots,K_{T-1}] \in \mathbb{R}^{T \times n_u \times n_x}$:

$$\min_{V,\,K}\; J(V,K)$$

### 4.1 Loss Function

$$J = \underbrace{w_\phi \cdot (-\log(p_{\mathrm{sat}} + \epsilon))}_{\text{(1) STL satisfaction}}
  + \underbrace{w_{\Sigma_T} \cdot \mathrm{tr}(\Sigma_T^{[xy]})}_{\text{(2) terminal covariance}}
  + \underbrace{w_d \cdot \|\mu_T^{[xy]} - g\|^2}_{\text{(3) goal distance}}
  + \underbrace{w_{du} \cdot \sum_{k}\|\Delta u_k\|^2}_{\text{(4) smoothness}}
  + \underbrace{w_K \cdot \|K\|_F^2}_{\text{(5) gain regularization}}
  + \underbrace{w_r \cdot \sum_{\mathrm{obs}} \mathrm{ReLU}(m - \mathrm{sdf})^2}_{\text{(6) obstacle repulsion}}$$

where $\Sigma_T^{[xy]} = \Sigma_{T,00} + \Sigma_{T,11}$ is the position-block trace, $g$ is the goal center, $\Delta u_k = u_{k+1} - u_k$ (with $\Delta u_0 = u_0$), and $\mathrm{sdf}$ is the signed distance to each obstacle.

**Term roles:**

| # | Term | Role |
|---|---|---|
| 1 | $-\log p_{\mathrm{sat}}$ | Primary objective — drives all constraint satisfaction |
| 2 | $\mathrm{tr}(\Sigma_T^{[xy]})$ | Pulls terminal distribution tight (covariance steering endpoint) |
| 3 | $\|\mu_T - g\|^2$ | Provides gradient when mean is already inside goal (CDF saturates) |
| 4 | $\|\Delta u\|^2$ | Prevents control chattering |
| 5 | $\|K\|_F^2$ | Keeps $A+BK^{\mathrm{eff}}$ stable; prevents unbounded covariance collapse |
| 6 | $\mathrm{ReLU}(m-\mathrm{sdf})^2$ | Geometric guide to help the mean find feasible corridors |

**Note:** Trajectory-wide covariance trace $\sum_k \mathrm{tr}(\Sigma_k)$ and control-effort norm $\sum_k \|u_k\|^2$ were removed — both are subsumed by terms 1 and 4 respectively.

### 4.2 Differentiability Chain

$$K \;\xrightarrow{A_k^{\mathrm{cl}} = A + BK^{\mathrm{eff}}}\; \Sigma_{k+1} \;\xrightarrow{\Phi(\cdot)}\; p_{\mathrm{pred}} \;\xrightarrow{\widetilde{\min}/\widetilde{\max}}\; p_{\mathrm{sat}} \;\xrightarrow{-\log}\; J$$

Every step is differentiable through PyTorch autograd. The saturation Jacobian $\mathrm{sech}^2(v_k)$ also couples $V \to K^{\mathrm{eff}} \to \Sigma \to J$.

---

## 5. Planners

### 5.1 Single-Shot Planner

Optimize $(V, K)$ once over the full horizon $T$:

```
for iter in range(max_iters):
    roll out steerer → (μ_trace, Σ_trace)
    evaluate pdSTL spec → p_sat
    compute J → backprop → Adam step
    save best (V, K) by p_sat
    early-stop if p_sat ≥ α for patience steps
```

Parameters are initialized as $V \sim \mathcal{N}(0, 0.1^2)$, $K = 0$.

### 5.2 Receding-Horizon MPC Planner

At each step $t = 0,\ldots,T-1$:

1. **Local solve:** optimize $(V_{t:t+H}, K_{t:t+H})$ over window $H \le T-t$ from current belief $(\mu_t, \Sigma_t)$, running `mpc.iters` gradient steps.
2. **Warm start:** shift previous $V^*$ forward by one, pad with zero.
3. **Random restarts:** `n_starts - 1` random initializations per step to escape local minima.
4. **Apply:** advance belief one step using only the first control $v_t^*, K_t^*$.
5. **Assemble:** stack per-step results into full $(V, K, \mu\text{-trace}, \Sigma\text{-trace})$.

Global $P(\phi)$ is evaluated on the assembled full trajectory after all $T$ steps.

**Comparison:**

| | Single-Shot | MPC |
|---|---|---|
| Computation | One solve, more iterations | Many short solves |
| Noise handling | Fixed plan, no correction | Re-plans each step |
| K role | Shapes Σ globally | Shapes Σ within each window |
| Horizon | Full $T$ | Window $H \ll T$ |

---

## 6. Scenarios

### Narrow Gap

- **Start:** $\mu_0 = (-1, 0, 1, 0)$, $\Sigma_0 = 10^{-3} I$
- **Goal:** $p_x \in [9,10]$, $p_y \in [0,1]$
- **Obstacles:** 5 axis-aligned rectangles forming a corridor with a narrow lateral gap
- **Horizon:** $T = 80$ (single-shot), $T = 40$ (MPC)
- **Spec:** $\square_{[0,T]}\,\phi_{\mathrm{obs}} \wedge \lozenge_{[T,T]}\,\phi_{\mathrm{goal}}$

### Double Slit

- **Start:** $\mu_0 = (-3, 0, 1, 0)$, $\Sigma_0 = 0$
- **Goal:** $p_x \in [4.5,5.5]$, $p_y \in [0.5,1.5]$
- **Obstacles:** Vertical wall at $x \in [0,1.5]$ with two gaps: upper (width 0.5 m, narrow) and lower (width 1.0 m, wider)
- **Horizon:** $T = 40$
- **Expected behavior:** Open-loop covariance grows $\Rightarrow$ must use the wide lower gap. Covariance steering squeezes $\sigma_y$ $\Rightarrow$ can thread the narrow upper gap aligned with the goal.

### Obstacle Field

- **Dynamics:** Single integrator ($n_x=2$, $n_u=2$)
- **Start:** $\mu_0 = (0, 5)$, $\Sigma_0 = 10^{-3} I$
- **Goal:** $p_x \in [11,12]$, $p_y \in [6,7]$
- **Horizon:** $T = 130$

---

## 7. Key Design Decisions

**Saturation-consistent effective gain.** Using $K^{\mathrm{eff}} = \mathrm{diag}(\mathrm{sech}^2(v)\,u_{\max})\,K$ instead of $K$ directly ensures the covariance model is consistent with the actual nonlinear policy. A controller near saturation cannot respond to state deviations — the model reflects this.

**Repulsion as curriculum, not objective.** The SDF-based repulsion term guides the mean into feasible corridors during early optimization, when the STL loss landscape is nearly flat. It is kept small ($w_r \ll w_\phi$) so that pdSTL dominates once a rough path is found.

**$w_K$ must scale with horizon.** With $T=80$, $K \in \mathbb{R}^{80 \times 2 \times 4}$ has 640 parameters. Small $w_K$ allows $K$ to grow large enough to make $A + BK^{\mathrm{eff}}$ unstable. A safe choice is $w_K \approx 0.01$–$0.02$ for $T=80$.

**Terminal-only covariance penalty.** Penalising $\mathrm{tr}(\Sigma_T)$ targets the endpoint of the belief trajectory where goal satisfaction is evaluated. The trajectory-wide sum $\sum_k \mathrm{tr}(\Sigma_k)$ uniformly shrinks uncertainty everywhere, fighting the optimizer in regions no STL predicate observes.
