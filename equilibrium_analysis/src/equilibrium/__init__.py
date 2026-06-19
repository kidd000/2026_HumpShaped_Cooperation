"""
Reference replicator-dynamics library for Hump-shaped strategy research.

This is a reference implementation of the focal-player expected payoffs,
convergence database, and equilibrium solver for the public goods game with
4 strategies: AllC, AllD, CC (Conditional Cooperation), and H (Hump-shaped).
It is not required to reproduce the Figure 4 panels (see the self-contained
scripts under ``scripts/``); it supports the broader analysis.

Modules
-------
simulation
    Production function, strategies, composition generation, within-group
    convergence, multinomial / focal expected payoffs, replicator dynamics.
grid
    Parameter grid definitions for systematic parameter sweeps.
storage
    HDF5-based storage for equilibrium computation results.
solver
    Numerical methods for finding and classifying equilibria.

Example
-------
>>> from equilibrium.simulation import (
...     generate_compositions,
...     simulate_composition,
...     multinomial_prob,
... )
>>> # Generate all group compositions for N=4
>>> compositions = generate_compositions(n_strategies=4, group_size=4)
>>> print(f"Number of compositions: {len(compositions)}")
Number of compositions: 35
"""

# Core simulation exports
from .simulation import (
    # Constants
    STRATEGY_NAMES,
    STRATEGY_IDS,
    N_STRATEGIES,
    DEFAULT_GROUP_SIZE,
    DEFAULT_MPCR,
    DEFAULT_THRESHOLD,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_CYCLE_DECIMALS,
    DEFAULT_FALLBACK_WINDOW,
    # Production function
    supply_sigmoid,
    supply_sigmoid_raw,
    # Strategy functions
    get_strategy_response,
    # Composition utilities
    generate_compositions,
    compositions_to_array,
    # Within-group simulation functions
    simulate_composition,
    aggregate_payoffs_by_strategy,
    # Convergence database functions
    build_convergence_database,
    summarize_database,
    # Probability
    multinomial_prob,
    # Expected payoffs (focal-player formulation)
    expected_payoffs,
    compute_fitness,
    # Replicator dynamics
    replicator_dynamics,
    replicator_dynamics_rk45,
    replicator_dynamics_3strategy,
    replicator_dynamics_constrained,
    find_trajectory_endpoint,
)

# Grid exports
from .grid import (
    ParameterGrid,
    create_default_grid,
    create_x0_fixed_grid,
)

# Storage exports
from .storage import (
    EQ_TYPE_STABLE,
    EQ_TYPE_UNSTABLE,
    EQ_TYPE_SADDLE,
    EQ_TYPE_EMPTY,
    EquilibriumResult,
    EquilibriumStore,
    open_store,
)

# Solver exports
from .solver import (
    compute_selection_gradients,
    find_equilibria_3d,
    classify_equilibrium_jacobian,
    compute_equilibria_for_params,
    is_interior_equilibrium,
)

__all__ = [
    # Constants
    "STRATEGY_NAMES",
    "STRATEGY_IDS",
    "N_STRATEGIES",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_MPCR",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MAX_ROUNDS",
    "DEFAULT_CYCLE_DECIMALS",
    "DEFAULT_FALLBACK_WINDOW",
    # Production
    "supply_sigmoid",
    "supply_sigmoid_raw",
    # Strategies
    "get_strategy_response",
    # Compositions
    "generate_compositions",
    "compositions_to_array",
    # Within-group simulation
    "simulate_composition",
    "aggregate_payoffs_by_strategy",
    "build_convergence_database",
    "summarize_database",
    # Probability
    "multinomial_prob",
    # Expected payoffs (focal-player formulation)
    "expected_payoffs",
    "compute_fitness",
    # Replicator dynamics
    "replicator_dynamics",
    "replicator_dynamics_rk45",
    "replicator_dynamics_3strategy",
    "replicator_dynamics_constrained",
    "find_trajectory_endpoint",
    # Grid
    "ParameterGrid",
    "create_default_grid",
    "create_x0_fixed_grid",
    # Storage
    "EQ_TYPE_STABLE",
    "EQ_TYPE_UNSTABLE",
    "EQ_TYPE_SADDLE",
    "EQ_TYPE_EMPTY",
    "EquilibriumResult",
    "EquilibriumStore",
    "open_store",
    # Solver
    "compute_selection_gradients",
    "find_equilibria_3d",
    "classify_equilibrium_jacobian",
    "compute_equilibria_for_params",
    "is_interior_equilibrium",
]
