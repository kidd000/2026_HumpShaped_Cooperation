"""
Composition lookup tables for group strategy compositions.

For N=4 players with 4 strategies, there are C(4+4-1, 4) = 35 possible
compositions. This module provides pre-computed lookup tables for
efficient O(1) access in Numba-compiled code.

Usage
-----
>>> from evolution.composition import COMPOSITION_TABLE, COMPOSITION_ID_LOOKUP
>>> # Get composition for ID=0
>>> n_c, n_d, n_cc, n_h = COMPOSITION_TABLE[0]
>>> # Get ID for composition (1, 1, 1, 1)
>>> comp_id = COMPOSITION_ID_LOOKUP[1, 1, 1, 1]
"""

import numpy as np
from numba import njit


def generate_composition_table(N: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate composition lookup tables.

    Parameters
    ----------
    N : int
        Group size (default 4).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (table, lookup) where:
        - table: shape (n_compositions, 4), dtype=int32
          Maps composition_id -> (n_AllC, n_AllD, n_CC, n_Hump)
        - lookup: shape (N+1, N+1, N+1, N+1), dtype=int32
          Maps (n_AllC, n_AllD, n_CC, n_Hump) -> composition_id
          Invalid combinations have value -1.
    """
    compositions = []

    # Generate all valid compositions where n_c + n_d + n_cc + n_h = N
    for n_c in range(N + 1):
        for n_d in range(N + 1 - n_c):
            for n_cc in range(N + 1 - n_c - n_d):
                n_h = N - n_c - n_d - n_cc
                compositions.append((n_c, n_d, n_cc, n_h))

    table = np.array(compositions, dtype=np.int32)

    # Create reverse lookup
    lookup = np.full((N + 1, N + 1, N + 1, N + 1), -1, dtype=np.int32)
    for comp_id, (n_c, n_d, n_cc, n_h) in enumerate(compositions):
        lookup[n_c, n_d, n_cc, n_h] = comp_id

    return table, lookup


# Pre-computed tables for N=4 (35 compositions)
COMPOSITION_TABLE, COMPOSITION_ID_LOOKUP = generate_composition_table(N=4)

# Number of compositions
N_COMPOSITIONS = len(COMPOSITION_TABLE)


@njit(cache=True)
def count_strategies(strategies: np.ndarray) -> tuple[int, int, int, int]:
    """
    Count each strategy type in an array.

    Parameters
    ----------
    strategies : np.ndarray
        Array of strategy IDs (0=AllC, 1=AllD, 2=CC, 3=Hump).

    Returns
    -------
    tuple[int, int, int, int]
        (n_AllC, n_AllD, n_CC, n_Hump)
    """
    n_c = 0
    n_d = 0
    n_cc = 0
    n_h = 0

    for s in strategies:
        if s == 0:
            n_c += 1
        elif s == 1:
            n_d += 1
        elif s == 2:
            n_cc += 1
        else:  # s == 3
            n_h += 1

    return n_c, n_d, n_cc, n_h


