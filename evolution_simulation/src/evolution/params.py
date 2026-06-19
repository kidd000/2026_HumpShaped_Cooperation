"""
Simulation parameters for evolution simulation.

This module defines the SimParams dataclass that holds all parameters
for the global pooling evolution simulation. Parameters are categorized as:
- Environment/game structure (K, x0, mpcr)
- Population structure (M, N, T_gen)
- Convergence parameters (convergence_threshold, max_round_iterations)
- Evolution parameters (mu_strat, sigma_belief)
- Random seed
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# Strategy type constants
STRATEGY_ALLC = 0
STRATEGY_ALLD = 1
STRATEGY_CC = 2
STRATEGY_HUMP = 3

STRATEGY_NAMES = {
    STRATEGY_ALLC: "All-C",
    STRATEGY_ALLD: "All-D",
    STRATEGY_CC: "CC",
    STRATEGY_HUMP: "Hump",
}

N_STRATEGIES = 4


@dataclass(frozen=True)
class SimParams:
    """
    Immutable simulation parameters.

    Frozen dataclass ensures hashability for caching and prevents
    accidental modification during simulation.

    Attributes
    ----------
    K : float
        Production function steepness (>= 1, or inf for step function).
        Higher = more step-like. K=inf gives exact step function.
        Default 50.0.
    x0 : float
        Production function inflection point (0-1).
        Default 0.5.
    mpcr : float
        Marginal per capita return (0-1).
        Default 0.4.
    M : int
        Number of groups.
        Default 250.
    N : int
        Group size (players per group).
        Default 4.
    T_gen : int
        Number of generations to simulate.
        Default 50000.
    convergence_threshold : float
        Threshold for adiabatic convergence within generation.
        Stop when |group_mean_t - group_mean_{t-1}| < threshold.
        Default 0.01.
    max_round_iterations : int
        Maximum iterations per generation (prevents infinite loops).
        Default 1000.
    mu_strat : float
        Strategy mutation rate (0-1).
        Default 0.01.
    sigma_belief : float
        Standard deviation of the symmetric Gaussian noise added to the
        inherited initial belief: b_offspring = clip_[0,1](b_parent + N(0, sigma_belief)).
        Default 0.01.
    seed : Optional[int]
        Random seed for reproducibility. None for random.
    """

    # Environment / Game structure
    K: float = 50.0
    x0: float = 0.5
    mpcr: float = 0.4

    # Population structure
    M: int = 250
    N: int = 4
    T_gen: int = 50000

    # Convergence parameters (adiabatic approximation)
    convergence_threshold: float = 0.01
    max_round_iterations: int = 1000

    # Convergence detection parameters (3-stage: delta, cycle, fallback)
    cycle_decimals: int = 6        # Decimal places for state hashing in cycle detection
    fallback_window: int = 10      # Window size for fallback averaging on timeout
    max_cycle_check: int = 50      # Maximum number of past states to check for cycles

    # Evolution parameters
    mu_strat: float = 0.01
    sigma_belief: float = 0.01  # Symmetric Gaussian belief-inheritance noise
    n_strategies: int = 4  # Number of strategies (3=no Hump, 4=all)
    hump_threshold: float = 0.5  # Hump strategy peak threshold
    enabled_strategies: tuple[int, ...] = (0, 1, 2, 3)  # Strategy IDs available for mutation

    # Random seed
    seed: Optional[int] = None

    @property
    def N_total(self) -> int:
        """Total number of agents."""
        return self.M * self.N

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SimParams":
        """Create from dictionary."""
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "SimParams":
        """Create from JSON string."""
        return cls.from_dict(json.loads(s))

    def with_updates(self, **kwargs) -> "SimParams":
        """
        Create a new SimParams with updated values.

        Since SimParams is frozen, this returns a new instance.

        Parameters
        ----------
        **kwargs
            Parameter names and new values.

        Returns
        -------
        SimParams
            New instance with updated values.

        Example
        -------
        >>> params = SimParams()
        >>> params_high_k = params.with_updates(K=100.0, sigma_belief=0.02)
        """
        current = self.to_dict()
        current.update(kwargs)
        return SimParams.from_dict(current)

    def validate(self) -> list[str]:
        """
        Validate parameter ranges.

        Returns
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Environment parameters (K can be inf for step function)
        if not (self.K >= 1.0):
            errors.append(f"K must be >= 1 (or inf), got {self.K}")
        if not (0.0 <= self.x0 <= 1.0):
            errors.append(f"x0 must be in [0, 1], got {self.x0}")
        if not (0.0 < self.mpcr < 1.0):
            errors.append(f"mpcr must be in (0, 1), got {self.mpcr}")

        # Population structure
        if self.M < 1:
            errors.append(f"M must be >= 1, got {self.M}")
        if self.N < 2:
            errors.append(f"N must be >= 2, got {self.N}")
        if self.T_gen < 1:
            errors.append(f"T_gen must be >= 1, got {self.T_gen}")

        # Convergence parameters
        if not (0.0 < self.convergence_threshold <= 1.0):
            errors.append(f"convergence_threshold must be in (0, 1], got {self.convergence_threshold}")
        if self.max_round_iterations < 1:
            errors.append(f"max_round_iterations must be >= 1, got {self.max_round_iterations}")

        # Convergence detection parameters
        if not (1 <= self.cycle_decimals <= 12):
            errors.append(f"cycle_decimals must be in [1, 12], got {self.cycle_decimals}")
        if self.fallback_window < 1:
            errors.append(f"fallback_window must be >= 1, got {self.fallback_window}")
        if self.max_cycle_check < 2:
            errors.append(f"max_cycle_check must be >= 2, got {self.max_cycle_check}")

        # Evolution parameters
        if not (0.0 <= self.mu_strat <= 1.0):
            errors.append(f"mu_strat must be in [0, 1], got {self.mu_strat}")
        if self.sigma_belief < 0.0:
            errors.append(f"sigma_belief must be >= 0, got {self.sigma_belief}")
        if not (3 <= self.n_strategies <= 4):
            errors.append(f"n_strategies must be 3 or 4, got {self.n_strategies}")
        if not (0.0 < self.hump_threshold < 1.0):
            errors.append(f"hump_threshold must be in (0, 1), got {self.hump_threshold}")

        # enabled_strategies validation
        if len(self.enabled_strategies) == 0:
            errors.append("enabled_strategies must not be empty")
        if any(s < 0 or s > 3 for s in self.enabled_strategies):
            errors.append(f"enabled_strategies elements must be in [0, 3], got {self.enabled_strategies}")
        if len(set(self.enabled_strategies)) != len(self.enabled_strategies):
            errors.append(f"enabled_strategies must not contain duplicates, got {self.enabled_strategies}")
        if STRATEGY_ALLD not in self.enabled_strategies:
            errors.append(f"enabled_strategies must include AllD ({STRATEGY_ALLD}) for initialization")

        return errors

    def summary(self) -> str:
        """Return a human-readable summary of parameters."""
        return f"""SimParams:
  Environment:  K={self.K}, x0={self.x0}, mpcr={self.mpcr}
  Population:   M={self.M} groups, N={self.N} per group, total={self.N_total}
  Generations:  T_gen={self.T_gen}
  Convergence:  threshold={self.convergence_threshold}, max_iter={self.max_round_iterations}
  Detection:    cycle_decimals={self.cycle_decimals}, fallback_window={self.fallback_window}, max_cycle_check={self.max_cycle_check}
  Evolution:    mu_strat={self.mu_strat}, n_strategies={self.n_strategies}, hump_threshold={self.hump_threshold}, enabled={self.enabled_strategies}
  Belief noise: sigma={self.sigma_belief}
  Seed:         {self.seed}"""


# Default parameters instance
DEFAULT_PARAMS = SimParams()
