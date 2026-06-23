"""
Production function for Public Goods Game.

Implements the sigmoid (logistic) production function:
    S(c) = 1 / (1 + exp(-K * (c - x0)))

where:
    c: group average cooperation rate [0, 1]
    K: steepness parameter (1-100, higher = more step-like)
    x0: inflection point (0-1, threshold for cooperation benefit)

Payoff function:
    π_i = E * (1 + MPCR * S(c_bar) * N - c_i)

where:
    E: endowment (default 1)
    MPCR: marginal per capita return (default 0.4)
    N: group size (default 4)
    c_bar: group average cooperation rate
    c_i: individual's cooperation rate
"""

import numpy as np
from numba import njit

# Default constants
DEFAULT_ENDOWMENT = 1.0
DEFAULT_MPCR = 0.4
DEFAULT_GROUP_SIZE = 4


@njit(cache=True)
def sigmoid_production(c: float, K: float, x0: float) -> float:
    """
    Compute normalized sigmoid production function value.

    S_norm(c) = (S(c) - S(0)) / (S(1) - S(0))

    where S(c) = 1 / (1 + exp(-K * (c - x0)))

    This normalization ensures S(0) = 0 and S(1) = 1 for all K values.

    Special case: K = inf returns a step function:
        S(c) = 1 if c >= x0, else 0

    Parameters
    ----------
    c : float
        Group average cooperation rate [0, 1].
    K : float
        Steepness parameter (1-100, or inf for step function).
    x0 : float
        Inflection point (0-1).

    Returns
    -------
    float
        Normalized production value in [0, 1].
    """
    # Handle K = infinity (step function); S(x0) = 1/2 per SI Eq. S3
    if np.isinf(K):
        if c > x0:
            return 1.0
        elif c < x0:
            return 0.0
        else:
            return 0.5

    # Helper function to compute raw sigmoid with overflow protection
    def raw_sigmoid(x: float) -> float:
        exponent = -K * (x - x0)
        if exponent > 700:
            return 0.0
        elif exponent < -700:
            return 1.0
        return 1.0 / (1.0 + np.exp(exponent))

    # Compute raw sigmoid values
    S_c = raw_sigmoid(c)
    S_0 = raw_sigmoid(0.0)
    S_1 = raw_sigmoid(1.0)

    # Normalize to [0, 1] range
    denominator = S_1 - S_0
    if denominator < 1e-10:
        # Edge case: S_1 ≈ S_0 (very small K or extreme x0)
        return 0.5
    
    return (S_c - S_0) / denominator


