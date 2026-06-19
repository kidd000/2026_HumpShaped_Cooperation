"""
Parameter grid for evolution simulation sweeps.

Defines parameter ranges and generates all combinations for systematic
exploration of the parameter space.

Sweep parameters:
- K: Production function steepness (1-128, or inf for step function)
- seeds: Random seeds for replication

All other parameters follow the main-text model and are fixed:
the inflection point is x0 = 0.5, the belief-inheritance noise is a
symmetric Gaussian with sigma = 0.01, group size N and the remaining
parameters (M, T_gen, mu_strat, mpcr, hump_threshold) are held constant
across the sweep.
"""

from dataclasses import dataclass, field
from typing import Iterator
import json

from .params import SimParams


@dataclass
class EvolutionGrid:
    """
    Parameter grid for evolution simulation sweeps.

    Attributes
    ----------
    K_values : list[float]
        Values of production function steepness to sweep.
    seeds : list[int]
        Random seeds for replication.
    M : int
        Number of groups (fixed across sweep).
    N : int
        Group size (fixed across sweep).
    T_gen : int
        Generations to simulate (fixed across sweep).
    convergence_threshold : float
        Adiabatic convergence threshold (fixed across sweep).
    max_round_iterations : int
        Maximum iterations for convergence (fixed across sweep).
    mu_strat : float
        Strategy mutation rate (fixed across sweep).
    sigma_belief : float
        Symmetric Gaussian belief-inheritance noise (fixed across sweep).
    mpcr : float
        Marginal per capita return (fixed across sweep).
    x0 : float
        Production function inflection point (fixed at 0.5 in the main model).
    n_strategies : int
        Number of strategies (3 = no Hump, 4 = all). Default 4.
    hump_threshold : float
        Hump strategy peak threshold (fixed at 0.5 in the main model).
    """

    # Sweep parameters
    K_values: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    seeds: list[int] = field(default_factory=lambda: list(range(20)))

    # Fixed parameters
    M: int = 250
    N: int = 4
    T_gen: int = 50000
    convergence_threshold: float = 0.01
    max_round_iterations: int = 1000
    mu_strat: float = 0.01
    sigma_belief: float = 0.01
    mpcr: float = 0.4
    x0: float = 0.5
    n_strategies: int = 4
    hump_threshold: float = 0.5

    @property
    def n_K(self) -> int:
        """Number of K values."""
        return len(self.K_values)

    @property
    def n_seeds(self) -> int:
        """Number of seeds."""
        return len(self.seeds)

    @property
    def total_conditions(self) -> int:
        """Total number of parameter combinations."""
        return self.n_K * self.n_seeds

    @property
    def unique_conditions(self) -> int:
        """Number of unique parameter combinations (excluding seeds)."""
        return self.n_K

    def iter_params(self) -> Iterator[SimParams]:
        """
        Iterate over all parameter combinations.

        Yields
        ------
        SimParams
            Simulation parameters for each combination.
        """
        for K in self.K_values:
            for seed in self.seeds:
                yield SimParams(
                    K=K,
                    x0=self.x0,
                    M=self.M,
                    N=self.N,
                    T_gen=self.T_gen,
                    convergence_threshold=self.convergence_threshold,
                    max_round_iterations=self.max_round_iterations,
                    mu_strat=self.mu_strat,
                    sigma_belief=self.sigma_belief,
                    mpcr=self.mpcr,
                    seed=seed,
                    n_strategies=self.n_strategies,
                    hump_threshold=self.hump_threshold,
                )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'K_values': self.K_values,
            'seeds': self.seeds,
            'M': self.M,
            'N': self.N,
            'T_gen': self.T_gen,
            'convergence_threshold': self.convergence_threshold,
            'max_round_iterations': self.max_round_iterations,
            'mu_strat': self.mu_strat,
            'sigma_belief': self.sigma_belief,
            'mpcr': self.mpcr,
            'x0': self.x0,
            'n_strategies': self.n_strategies,
            'hump_threshold': self.hump_threshold,
            'total_conditions': self.total_conditions,
            'unique_conditions': self.unique_conditions,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "EvolutionGrid":
        """Create from dictionary (ignoring computed properties)."""
        valid_keys = ['K_values', 'seeds', 'M', 'N', 'T_gen',
                      'convergence_threshold', 'max_round_iterations',
                      'mu_strat', 'sigma_belief', 'mpcr', 'x0',
                      'n_strategies', 'hump_threshold']
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> "EvolutionGrid":
        """Create from JSON string."""
        return cls.from_dict(json.loads(s))

    def summary(self) -> str:
        """Return human-readable summary."""
        return f"""EvolutionGrid:
  Sweep parameters:
    K:     {self.K_values} ({self.n_K} values)
    seeds: {self.seeds} ({self.n_seeds} values)

  Fixed parameters:
    M={self.M}, N={self.N}, T_gen={self.T_gen}, n_strategies={self.n_strategies}
    x0={self.x0}, hump_threshold={self.hump_threshold}
    convergence={self.convergence_threshold}, max_iter={self.max_round_iterations}
    mu_strat={self.mu_strat}, sigma_belief={self.sigma_belief}, mpcr={self.mpcr}

  Total: {self.total_conditions} conditions ({self.unique_conditions} unique × {self.n_seeds} seeds)"""

    def __repr__(self) -> str:
        return f"EvolutionGrid(total={self.total_conditions})"
