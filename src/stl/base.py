"""Belief representations for probabilistic STL evaluation."""

from abc import ABC, abstractmethod


class Belief(ABC):
    """Abstract belief state. Predicates read .mean_full and .var_full."""

    @abstractmethod
    def value(self):
        raise NotImplementedError

    @abstractmethod
    def probability_of(self, residual):
        raise NotImplementedError


class GaussianBelief(Belief):
    """Concrete Gaussian belief wrapping (μ, Σ) tensors.

    Attributes:
        mean_full: [B, D]           mean vector
        var_full:  [B, D] or [B, D, D]  covariance (diagonal or full)
    """

    def __init__(self, mean_full, var_full):
        self.mean_full = mean_full
        self.var_full = var_full

    def value(self):
        return self.mean_full

    def probability_of(self, residual):
        raise NotImplementedError(
            "GaussianBelief is used with predicates that read mean/var directly."
        )


class BeliefTrajectory:
    """Sequence of beliefs indexed by time."""

    def __init__(self, beliefs):
        self.beliefs = beliefs

    def __getitem__(self, t):
        return self.beliefs[t]

    def __len__(self):
        return len(self.beliefs)

    def suffix(self, t):
        return BeliefTrajectory(self.beliefs[t:])
