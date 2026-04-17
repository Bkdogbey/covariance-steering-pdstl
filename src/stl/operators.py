"""Probabilistic differentiable STL operators.

All operators work on [B, T, 2] traces where dim=-1 is [lower, upper] bounds.
Temporal operators use backward RNN (flip → process → flip) for forward-looking
semantics with O(T) complexity.
"""

import numpy as np
import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════
# Smooth min / max
# ═════════════════════════════════════════════════════════════════════

class Minish(nn.Module):
    """Smooth minimum via negative log-sum-exp."""

    def forward(self, x, scale, dim=1, keepdim=True):
        if scale > 0:
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        return x.min(dim=dim, keepdim=keepdim)[0]


class Maxish(nn.Module):
    """Smooth maximum via log-sum-exp."""

    def forward(self, x, scale, dim=1, keepdim=True):
        if scale > 0:
            return torch.logsumexp(x * scale, dim=dim, keepdim=keepdim) / scale
        return x.max(dim=dim, keepdim=keepdim)[0]


# ═════════════════════════════════════════════════════════════════════
# Base formula
# ═════════════════════════════════════════════════════════════════════

class STL_Formula(nn.Module):
    """Base class. Subclasses implement robustness_trace → [B, T, 2]."""

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kw):
        raise NotImplementedError

    def forward(self, belief_trajectory, **kw):
        return self.robustness_trace(belief_trajectory, **kw)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Negation(self)


# ═════════════════════════════════════════════════════════════════════
# Logical operators
# ═════════════════════════════════════════════════════════════════════

class Negation(STL_Formula):
    """¬φ : swap and complement bounds."""

    def __init__(self, sub):
        super().__init__()
        self.sub = sub

    def robustness_trace(self, bt, scale=-1, keepdim=True, **kw):
        t = self.sub(bt, scale=scale, keepdim=keepdim, **kw)
        return torch.stack([1.0 - t[..., 1], 1.0 - t[..., 0]], dim=-1)


class And(STL_Formula):
    """φ₁ ∧ φ₂ : product lower bound, min upper bound."""

    def __init__(self, sub1, sub2):
        super().__init__()
        self.sub1 = sub1
        self.sub2 = sub2

    def robustness_trace(self, bt, scale=-1, keepdim=True, **kw):
        t1 = self.sub1(bt, scale=scale, keepdim=keepdim, **kw)
        t2 = self.sub2(bt, scale=scale, keepdim=keepdim, **kw)
        lower = t1[..., 0:1] * t2[..., 0:1]
        upper = torch.minimum(t1[..., 1:2], t2[..., 1:2])
        return torch.cat([lower, upper], dim=-1)


class Or(STL_Formula):
    """φ₁ ∨ φ₂ : max lower bound, clamped sum upper bound."""

    def __init__(self, sub1, sub2):
        super().__init__()
        self.sub1 = sub1
        self.sub2 = sub2

    def robustness_trace(self, bt, scale=-1, keepdim=True, **kw):
        t1 = self.sub1(bt, scale=scale, keepdim=keepdim, **kw)
        t2 = self.sub2(bt, scale=scale, keepdim=keepdim, **kw)
        lower = torch.maximum(t1[..., 0:1], t2[..., 0:1])
        upper = torch.minimum(t1[..., 1:2] + t2[..., 1:2], torch.ones_like(t1[..., 1:2]))
        return torch.cat([lower, upper], dim=-1)


# ═════════════════════════════════════════════════════════════════════
# Temporal operators (backward RNN, sliding window)
# ═════════════════════════════════════════════════════════════════════

class _TemporalOp(STL_Formula):
    """Shared RNN machinery for Always / Eventually."""

    def __init__(self, sub, interval=None):
        super().__init__()
        self.sub = sub
        self.interval = interval
        self._interval = [0, np.inf] if interval is None else list(interval)
        self.rnn_dim = self._compute_rnn_dim()
        self.operation = None  # set by subclass

        M = np.diag(np.ones(self.rnn_dim - 1), k=1)
        self.register_buffer("M", torch.tensor(M, dtype=torch.float32), persistent=False)
        b = torch.zeros(self.rnn_dim, 1, dtype=torch.float32)
        b[-1] = 1.0
        self.register_buffer("b", b, persistent=False)

    def _compute_rnn_dim(self):
        if self.interval is None:
            return 1
        a, b = self._interval
        return int(a) if np.isinf(b) else int(b + 1)

    # ── shift register helpers ───────────────────────────────────────

    def _init_hidden(self, x):
        """x: [B, T, 2] reversed. Returns (h0, count)."""
        h0 = x[:, :1, :].expand(-1, self.rnn_dim, -1).clone()
        if np.isinf(self._interval[1]) and self._interval[0] > 0:
            return ((x[:, :1, :], h0), 0.0)
        return (h0, 0.0)

    def _shift(self, h0, x):
        """Apply M @ h0 + b * x for [B, rnn_dim, 2] state."""
        B, R, C = h0.shape
        flat = h0.permute(0, 2, 1).reshape(-1, R)
        shifted = torch.matmul(flat, self.M.t()).reshape(B, C, R).permute(0, 2, 1)
        return shifted + self.b.view(1, -1, 1) * x.squeeze(1).unsqueeze(1)

    # ── single RNN step ──────────────────────────────────────────────

    def _rnn_cell(self, x, hc, scale):
        h0, c = hc
        if self.interval is None:
            inp = torch.cat([h0, x], dim=1)
            out = self.operation(inp, scale, dim=1, keepdim=True)
            return out, (out, None)
        elif np.isinf(self._interval[1]) and self._interval[0] > 0:
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)
            out = self.operation(dh, scale, dim=1, keepdim=True)
            return out, ((out, self._shift(h0, x)), None)
        else:
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h = self._shift(h0, x)
            window = new_h[:, : b - a + 1, :]
            out = self.operation(window, scale, dim=1, keepdim=True)
            return out, (new_h, None)

    # ── full trace ───────────────────────────────────────────────────

    def robustness_trace(self, bt, scale=-1, keepdim=True, **kw):
        sub_trace = self.sub(bt, scale=scale, keepdim=keepdim, **kw)  # [B, T, 2]
        rev = torch.flip(sub_trace, dims=[1])
        outs = []
        hc = self._init_hidden(rev)
        for xi in torch.split(rev, 1, dim=1):
            o, hc = self._rnn_cell(xi, hc, scale)
            outs.append(o)
        return torch.flip(torch.cat(outs, dim=1), dims=[1])


class Always(_TemporalOp):
    """□[a,b] φ — minimum over window."""

    def __init__(self, sub, interval=None):
        super().__init__(sub, interval)
        self.operation = Minish()


class Eventually(_TemporalOp):
    """◇[a,b] φ — maximum over window."""

    def __init__(self, sub, interval=None):
        super().__init__(sub, interval)
        self.operation = Maxish()


class Until(STL_Formula):
    """φ₁ U[a,b] φ₂ — φ₂ holds at some τ ∈ [t+a, t+b], φ₁ holds until τ."""

    def __init__(self, left, right, interval=None):
        super().__init__()
        self.left = left
        self.right = right
        self._interval = [0, np.inf] if interval is None else list(interval)
        self.min_op = Minish()
        self.max_op = Maxish()

    def robustness_trace(self, bt, scale=-1, keepdim=True, **kw):
        phi = self.left(bt, scale=scale, keepdim=True, **kw)
        psi = self.right(bt, scale=scale, keepdim=True, **kw)
        B, T, _ = phi.shape
        a = int(self._interval[0])
        b = T - 1 if np.isinf(self._interval[1]) else int(self._interval[1])

        results = []
        for t in range(T):
            start, end = t + a, min(t + b, T - 1)
            if start > end:
                results.append(torch.zeros(B, 2, device=phi.device))
                continue
            candidates = []
            for tau in range(start, end + 1):
                if tau == t:
                    min_phi = torch.ones_like(phi[:, 0, :])
                else:
                    seg = phi[:, t:tau, :]
                    min_phi = self.min_op(seg, scale, dim=1, keepdim=False)
                pair = torch.stack([min_phi, psi[:, tau, :]], dim=1)
                candidates.append(self.min_op(pair, scale, dim=1, keepdim=False).unsqueeze(1))
            best = self.max_op(torch.cat(candidates, dim=1), scale, dim=1, keepdim=False)
            results.append(best)
        return torch.stack(results, dim=1)
