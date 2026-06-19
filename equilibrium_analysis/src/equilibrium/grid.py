"""
Parameter Grid Module
=====================

Defines parameter grids for large-scale equilibrium computation.

Grid Parameters:
- x0: Production function center (0.01 ~ 0.99)
- K: Production function selectivity (1.0 ~ 50.0)
- init_coop: Initial cooperation rate (0.01 ~ 0.99)

Note: Boundary values (0.0, 1.0) are excluded to avoid numerical issues.
"""

import numpy as np
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional


@dataclass
class ParameterGrid:
    """
    Parameter grid for equilibrium sweep computation.

    Attributes
    ----------
    x0_values : np.ndarray
        Production function center values
    K_values : np.ndarray
        Production function selectivity values
    init_coop_values : np.ndarray
        Initial cooperation rate values

    Examples
    --------
    >>> grid = ParameterGrid.default()
    >>> print(grid.total_size)
    970299
    >>> for x0, K, init_coop in grid.iter_params():
    ...     # compute equilibria
    ...     pass
    """

    x0_values: np.ndarray
    K_values: np.ndarray
    init_coop_values: np.ndarray

    @classmethod
    def default(cls, exclude_boundaries: bool = True) -> 'ParameterGrid':
        """
        Create default parameter grid.

        Parameters
        ----------
        exclude_boundaries : bool
            If True, exclude 0.0 and 1.0 from x0 and init_coop

        Returns
        -------
        ParameterGrid
            Default grid with 99 x 99 x 99 points
        """
        if exclude_boundaries:
            x0_values = np.linspace(0.01, 0.99, 99)
            init_coop_values = np.linspace(0.01, 0.99, 99)
        else:
            x0_values = np.linspace(0.0, 1.0, 101)
            init_coop_values = np.linspace(0.0, 1.0, 101)

        K_values = np.arange(1.0, 50.5, 0.5)  # 1.0, 1.5, ..., 50.0 (99 points)

        return cls(
            x0_values=x0_values,
            K_values=K_values,
            init_coop_values=init_coop_values
        )

    @classmethod
    def small(cls) -> 'ParameterGrid':
        """
        Create small grid for testing.

        Returns
        -------
        ParameterGrid
            Small grid with 10 x 10 x 10 points
        """
        return cls(
            x0_values=np.linspace(0.1, 0.9, 10),
            K_values=np.array([1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
            init_coop_values=np.linspace(0.1, 0.9, 10)
        )

    @classmethod
    def x0_fixed(cls, x0: float = 0.5, exclude_boundaries: bool = True) -> 'ParameterGrid':
        """
        Create grid with fixed x0 value (2D sweep over K and init_coop).

        Parameters
        ----------
        x0 : float
            Fixed x0 value
        exclude_boundaries : bool
            If True, exclude 0.0 and 1.0 from init_coop

        Returns
        -------
        ParameterGrid
            Grid with single x0 value
        """
        if exclude_boundaries:
            init_coop_values = np.linspace(0.01, 0.99, 99)
        else:
            init_coop_values = np.linspace(0.0, 1.0, 101)

        K_values = np.arange(1.0, 50.5, 0.5)

        return cls(
            x0_values=np.array([x0]),
            K_values=K_values,
            init_coop_values=init_coop_values
        )

    @property
    def n_x0(self) -> int:
        """Number of x0 values."""
        return len(self.x0_values)

    @property
    def n_K(self) -> int:
        """Number of K values."""
        return len(self.K_values)

    @property
    def n_init_coop(self) -> int:
        """Number of init_coop values."""
        return len(self.init_coop_values)

    @property
    def total_size(self) -> int:
        """Total number of parameter combinations."""
        return self.n_x0 * self.n_K * self.n_init_coop

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape (n_x0, n_K, n_init_coop)."""
        return (self.n_x0, self.n_K, self.n_init_coop)

    def iter_params(self) -> Iterator[Tuple[float, float, float]]:
        """
        Iterate over all parameter combinations.

        Yields
        ------
        Tuple[float, float, float]
            (x0, K, init_coop) parameter tuple
        """
        for x0 in self.x0_values:
            for K in self.K_values:
                for init_coop in self.init_coop_values:
                    yield (x0, K, init_coop)

    def summary(self) -> str:
        """Return human-readable summary of the grid."""
        lines = [
            "ParameterGrid Summary",
            "=" * 40,
            f"x0:        {self.x0_values[0]:.2f} ~ {self.x0_values[-1]:.2f} ({self.n_x0} points)",
            f"K:         {self.K_values[0]:.1f} ~ {self.K_values[-1]:.1f} ({self.n_K} points)",
            f"init_coop: {self.init_coop_values[0]:.2f} ~ {self.init_coop_values[-1]:.2f} ({self.n_init_coop} points)",
            "-" * 40,
            f"Total combinations: {self.total_size:,}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ParameterGrid(n_x0={self.n_x0}, n_K={self.n_K}, n_init_coop={self.n_init_coop}, total={self.total_size:,})"


# Convenience functions
def create_default_grid() -> ParameterGrid:
    """Create default parameter grid (99 x 99 x 99)."""
    return ParameterGrid.default()


def create_x0_fixed_grid(x0: float = 0.5) -> ParameterGrid:
    """Create grid with fixed x0 for 2D analysis."""
    return ParameterGrid.x0_fixed(x0)
