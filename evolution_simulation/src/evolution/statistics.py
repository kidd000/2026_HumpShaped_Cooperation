"""
Statistics module for evolution simulation.

Computes generation-level and strategy-level statistics.

Main outputs:
1. GenerationStats: Population-level statistics per generation
2. StrategyStats: Strategy-specific statistics per generation
"""

import numpy as np
from numba import njit

from .strategies import N_STRATEGIES


@njit(cache=True)
def compute_strategy_frequencies(
    strategies: np.ndarray,
    n_strategies: int = N_STRATEGIES
) -> np.ndarray:
    """
    Compute frequency of each strategy in the population.

    Parameters
    ----------
    strategies : np.ndarray
        Strategy IDs for all agents.
    n_strategies : int
        Number of strategies.

    Returns
    -------
    np.ndarray
        Frequency of each strategy (shape: n_strategies), sums to 1.
    """
    n_total = len(strategies)
    counts = np.zeros(n_strategies, dtype=np.float64)

    for s in strategies:
        counts[s] += 1.0

    return counts / n_total


@njit(cache=True)
def compute_strategy_counts(
    strategies: np.ndarray,
    n_strategies: int = N_STRATEGIES
) -> np.ndarray:
    """
    Count agents with each strategy.

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


@njit(cache=True)
def compute_belief_stats(beliefs: np.ndarray) -> tuple[float, float]:
    """
    Compute mean and standard deviation of beliefs.

    Parameters
    ----------
    beliefs : np.ndarray
        Belief values for all agents.

    Returns
    -------
    tuple[float, float]
        (mean_belief, std_belief)
    """
    mean_belief = np.mean(beliefs)
    std_belief = np.std(beliefs)
    return mean_belief, std_belief


@njit(cache=True)
def _compute_mean_by_strategy(
    strategies: np.ndarray,
    values: np.ndarray,
    n_strategies: int = N_STRATEGIES
) -> np.ndarray:
    """
    Compute mean of values grouped by strategy type.

    Parameters
    ----------
    strategies : np.ndarray
        Strategy IDs for all agents.
    values : np.ndarray
        Values to average per strategy.
    n_strategies : int
        Number of strategies.

    Returns
    -------
    np.ndarray
        Mean value for each strategy (shape: n_strategies).
        NaN if no agents have that strategy.
    """
    value_sum = np.zeros(n_strategies, dtype=np.float64)
    counts = np.zeros(n_strategies, dtype=np.int32)

    for i in range(len(strategies)):
        s = strategies[i]
        value_sum[s] += values[i]
        counts[s] += 1

    result = np.empty(n_strategies, dtype=np.float64)
    for s in range(n_strategies):
        if counts[s] > 0:
            result[s] = value_sum[s] / counts[s]
        else:
            result[s] = np.nan

    return result


@njit(cache=True)
def compute_mean_coop_by_strategy(
    strategies: np.ndarray,
    mean_coop: np.ndarray,
    n_strategies: int = N_STRATEGIES
) -> np.ndarray:
    """Compute mean cooperation rate for each strategy type."""
    return _compute_mean_by_strategy(strategies, mean_coop, n_strategies)


@njit(cache=True)
def compute_mean_fitness_by_strategy(
    strategies: np.ndarray,
    fitness: np.ndarray,
    n_strategies: int = N_STRATEGIES
) -> np.ndarray:
    """Compute mean fitness for each strategy type."""
    return _compute_mean_by_strategy(strategies, fitness, n_strategies)


@njit(cache=True)
def compute_mean_belief_by_strategy(
    strategies: np.ndarray,
    beliefs: np.ndarray,
    n_strategies: int = N_STRATEGIES
) -> np.ndarray:
    """Compute mean belief for each strategy type."""
    return _compute_mean_by_strategy(strategies, beliefs, n_strategies)


@njit(cache=True)
def compute_generation_stats(
    strategies: np.ndarray,
    beliefs: np.ndarray,
    fitness: np.ndarray,
    mean_coop: np.ndarray,
    n_strategies: int = N_STRATEGIES
) -> tuple:
    """
    Compute all generation-level statistics.

    Returns
    -------
    tuple
        (freq_by_strategy, mean_belief, std_belief, mean_fitness,
         mean_coop_rate, mean_coop_by_strategy)
    """
    # Strategy frequencies
    freq_by_strategy = compute_strategy_frequencies(strategies, n_strategies)

    # Belief statistics
    mean_belief, std_belief = compute_belief_stats(beliefs)

    # Fitness statistics
    mean_fitness = np.mean(fitness)

    # Cooperation statistics
    mean_coop_rate = np.mean(mean_coop)
    mean_coop_by_strat = compute_mean_coop_by_strategy(strategies, mean_coop, n_strategies)

    return (
        freq_by_strategy,
        mean_belief,
        std_belief,
        mean_fitness,
        mean_coop_rate,
        mean_coop_by_strat
    )


@njit(cache=True)
def compute_strategy_stats(
    strategies: np.ndarray,
    beliefs: np.ndarray,
    fitness: np.ndarray,
    mean_coop: np.ndarray,
    n_strategies: int = N_STRATEGIES
) -> tuple:
    """
    Compute strategy-level statistics.

    Returns
    -------
    tuple
        (counts, mean_beliefs, mean_fitnesses, mean_coops)
        Each is shape (n_strategies,)
    """
    counts = compute_strategy_counts(strategies, n_strategies)
    mean_beliefs = compute_mean_belief_by_strategy(strategies, beliefs, n_strategies)
    mean_fitnesses = compute_mean_fitness_by_strategy(strategies, fitness, n_strategies)
    mean_coops = compute_mean_coop_by_strategy(strategies, mean_coop, n_strategies)

    return counts, mean_beliefs, mean_fitnesses, mean_coops


