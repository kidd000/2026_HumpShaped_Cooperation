"""
Mutation module for strategy evolution.

Implements random strategy mutation where each agent has a probability
mu_strat of having their strategy randomly changed to any of the 4 strategies.
"""

import numpy as np
from numba import njit

from .strategies import N_STRATEGIES


@njit(cache=True)
def apply_mutation(
    strategies: np.ndarray,
    mu_strat: float,
    n_strategies: int = N_STRATEGIES
) -> tuple[np.ndarray, int]:
    """
    Apply random mutation to strategies.

    Each agent has probability mu_strat of mutating. When mutation occurs,
    the new strategy is chosen uniformly at random from all strategies
    (including potentially the same strategy).

    Parameters
    ----------
    strategies : np.ndarray
        Strategy IDs for all agents (modified in-place).
    mu_strat : float
        Mutation probability per individual.
    n_strategies : int
        Number of available strategies (default 4).

    Returns
    -------
    tuple[np.ndarray, int]
        (strategies, n_mutations) - modified strategies and count of mutations.
    """
    n_mutations = 0

    for i in range(len(strategies)):
        if np.random.random() < mu_strat:
            # Mutate to random strategy
            new_strategy = np.random.randint(0, n_strategies)
            if new_strategy != strategies[i]:
                n_mutations += 1
            strategies[i] = new_strategy

    return strategies, n_mutations


@njit(cache=True)
def apply_mutation_from_pool(
    strategies: np.ndarray,
    mu_strat: float,
    strategy_pool: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Apply mutation sampling from an explicit strategy pool.

    Used for ablation conditions where arbitrary strategy subsets are enabled
    (e.g., w/o CC uses pool [0, 1, 3]).

    Parameters
    ----------
    strategies : np.ndarray
        Strategy IDs for all agents (modified in-place).
    mu_strat : float
        Mutation probability per individual.
    strategy_pool : np.ndarray
        Array of allowed strategy IDs to sample from.

    Returns
    -------
    tuple[np.ndarray, int]
        (strategies, n_mutations) - modified strategies and count of mutations.
    """
    n_pool = len(strategy_pool)
    n_mutations = 0

    for i in range(len(strategies)):
        if np.random.random() < mu_strat:
            new_strategy = strategy_pool[np.random.randint(0, n_pool)]
            if new_strategy != strategies[i]:
                n_mutations += 1
            strategies[i] = new_strategy

    return strategies, n_mutations


@njit(cache=True)
def count_strategies(strategies: np.ndarray, n_strategies: int = N_STRATEGIES) -> np.ndarray:
    """
    Count the number of agents with each strategy.

    Parameters
    ----------
    strategies : np.ndarray
        Strategy IDs for all agents.
    n_strategies : int
        Number of strategies.

    Returns
    -------
    np.ndarray
        Count for each strategy (shape: n_strategies).
    """
    counts = np.zeros(n_strategies, dtype=np.int32)
    for s in strategies:
        counts[s] += 1
    return counts


