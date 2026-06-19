"""
Strategy definitions for evolution simulation.

Four strategy types:
- AllC (ID=0): Always cooperate fully (c=1.0)
- AllD (ID=1): Never cooperate (c=0.0)
- CC (ID=2): Conditional cooperator - match others' average
- Hump (ID=3): Hump-shaped - increase up to x=0.5, then decrease

All functions are Numba-compatible for JIT compilation.
"""

import numpy as np
from numba import njit

STRATEGY_ALLC = 0
STRATEGY_ALLD = 1
STRATEGY_CC = 2
STRATEGY_HUMP = 3

N_STRATEGIES = 4

STRATEGY_NAMES = ["AllC", "AllD", "CC", "Hump"]


@njit(cache=True)
def allc_response(others_mean: float) -> float:
    """
    AllC strategy: Always cooperate fully.

    Parameters
    ----------
    others_mean : float
        Average cooperation rate of others (ignored).

    Returns
    -------
    float
        Cooperation rate (always 1.0).
    """
    return 1.0


@njit(cache=True)
def alld_response(others_mean: float) -> float:
    """
    AllD strategy: Never cooperate.

    Parameters
    ----------
    others_mean : float
        Average cooperation rate of others (ignored).

    Returns
    -------
    float
        Cooperation rate (always 0.0).
    """
    return 0.0


@njit(cache=True)
def cc_response(others_mean: float) -> float:
    """
    Conditional Cooperator strategy: Match others' average.

    Parameters
    ----------
    others_mean : float
        Average cooperation rate of others.

    Returns
    -------
    float
        Cooperation rate (equals others_mean, clamped to [0, 1]).
    """
    if others_mean < 0.0:
        return 0.0
    elif others_mean > 1.0:
        return 1.0
    return others_mean


@njit(cache=True)
def hump_response(others_mean: float, threshold: float = 0.5) -> float:
    """
    Hump-shaped strategy: Increase up to threshold, then decrease.

    f(x) = x                   if x <= threshold
    f(x) = 2 * threshold - x   if threshold < x < 2 * threshold
    f(x) = 0                   if x >= 2 * threshold

    Maximum cooperation rate is threshold (at x=threshold).

    Parameters
    ----------
    others_mean : float
        Average cooperation rate of others.
    threshold : float
        Peak point of the hump (default: 0.5).

    Returns
    -------
    float
        Cooperation rate (0 to threshold).
    """
    if others_mean < 0.0:
        return 0.0
    elif others_mean <= threshold:
        return others_mean
    elif others_mean < 2.0 * threshold:
        return 2.0 * threshold - others_mean
    else:
        return 0.0


@njit(cache=True)
def get_cooperation_rate(
    strategy_id: int, others_mean: float, hump_threshold: float = 0.5
) -> float:
    """
    Get cooperation rate for a given strategy and others' mean.

    Parameters
    ----------
    strategy_id : int
        Strategy ID (0=AllC, 1=AllD, 2=CC, 3=Hump).
    others_mean : float
        Average cooperation rate of others in the group.
    hump_threshold : float
        Threshold for Hump strategy (default: 0.5).

    Returns
    -------
    float
        Cooperation rate in [0, 1].
    """
    if strategy_id == STRATEGY_ALLC:
        return allc_response(others_mean)
    elif strategy_id == STRATEGY_ALLD:
        return alld_response(others_mean)
    elif strategy_id == STRATEGY_CC:
        return cc_response(others_mean)
    else:  # STRATEGY_HUMP
        return hump_response(others_mean, hump_threshold)


