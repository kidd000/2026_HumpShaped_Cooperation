"""
Selection module for global pooling evolutionary dynamics.

Implements fitness-proportional selection where all individuals compete globally:
- Individuals are selected proportional to their fitness (global pool)
- Groups are randomly reformed each generation (no persistent group structure)

Main functions: global_pool_selection(), random_grouping()
"""

import numpy as np
from numba import njit


@njit(cache=True)
def normalize_fitness(fitness: np.ndarray) -> np.ndarray:
    """
    Normalize fitness values to selection probabilities.

    Handles negative fitness by shifting to positive range.

    Parameters
    ----------
    fitness : np.ndarray
        Raw fitness values.

    Returns
    -------
    np.ndarray
        Normalized probabilities (sum to 1).
    """
    n = len(fitness)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    min_fit = np.min(fitness)
    if min_fit < 0:
        # Shift to make all positive
        shifted = fitness - min_fit + 1e-10
    else:
        shifted = fitness + 1e-10  # Avoid division by zero

    total = np.sum(shifted)
    return shifted / total


@njit(cache=True)
def sample_one_proportional(probs: np.ndarray) -> int:
    """
    Sample a single index proportional to probabilities.

    Parameters
    ----------
    probs : np.ndarray
        Probability for each index (should sum to 1).

    Returns
    -------
    int
        Sampled index.
    """
    r = np.random.random()
    cumsum = 0.0
    for i in range(len(probs)):
        cumsum += probs[i]
        if r < cumsum:
            return i
    return len(probs) - 1


@njit(cache=True)
def global_pool_selection(
    strategies: np.ndarray,
    beliefs: np.ndarray,
    fitness: np.ndarray,
    sigma_belief: float,
    M: int,
    N: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Global pooling selection without group structure.

    Process:
    1. Pool all individuals (ignore group boundaries)
    2. Select parents proportional to fitness (global relative fitness)
    3. Offspring inherit strategy and belief (with Gaussian noise)

    Parameters
    ----------
    strategies : np.ndarray
        Current strategies (shape: M*N).
    beliefs : np.ndarray
        Current beliefs (shape: M*N).
    fitness : np.ndarray
        Individual fitness values (shape: M*N).
    sigma_belief : float
        Standard deviation of the symmetric Gaussian belief-inheritance noise.
    M : int
        Number of groups.
    N : int
        Group size.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (new_strategies, new_beliefs)

    """
    n_total = M * N

    # Prepare output arrays
    new_strategies = np.empty(n_total, dtype=np.int32)
    new_beliefs = np.empty(n_total, dtype=np.float64)

    # Normalize fitness to global selection probabilities
    global_probs = normalize_fitness(fitness)

    # Generate offspring
    for i in range(n_total):
        # Select parent proportional to global fitness
        parent_idx = sample_one_proportional(global_probs)

        # Inherit strategy
        new_strategies[i] = strategies[parent_idx]

        # Symmetric Gaussian belief noise: epsilon ~ N(0, sigma)
        noise = np.random.normal(0.0, sigma_belief)

        new_belief = beliefs[parent_idx] + noise

        # Clamp to [0, 1]
        if new_belief < 0.0:
            new_belief = 0.0
        elif new_belief > 1.0:
            new_belief = 1.0

        new_beliefs[i] = new_belief

    return new_strategies, new_beliefs


@njit(cache=True)
def random_grouping(M: int, N: int) -> np.ndarray:
    """
    Randomly assign individuals to groups.

    Each generation, individuals are randomly assigned to groups,
    implementing the "temporary grouping and dissolution" model.

    Parameters
    ----------
    M : int
        Number of groups.
    N : int
        Group size.

    Returns
    -------
    np.ndarray
        Group IDs for each individual (shape: M*N).
        Each group will have exactly N members.

    """
    n_total = M * N
    group_ids = np.empty(n_total, dtype=np.int32)

    # Create shuffled indices using Fisher-Yates shuffle
    indices = np.arange(n_total, dtype=np.int32)
    for i in range(n_total - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        indices[i], indices[j] = indices[j], indices[i]

    # Assign to groups: first N go to group 0, next N to group 1, etc.
    for i in range(n_total):
        group_ids[indices[i]] = i // N

    return group_ids


