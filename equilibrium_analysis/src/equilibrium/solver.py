"""
Equilibrium Solver Module
=========================

Reference utilities for replicator dynamics on the public goods game:

- Selection gradients (pi_s - pi_bar) for each strategy
- Jacobian-based stability classification of an equilibrium point
- Interior-equilibrium predicate
"""

import numpy as np
from typing import Dict, Tuple

from .storage import EQ_TYPE_STABLE, EQ_TYPE_UNSTABLE, EQ_TYPE_SADDLE


def compute_selection_gradients(
    p_d: float, p_cc: float, p_h: float,
    k: float, x0: float, init_coop: float,
    convergence_db: dict,
    p_allc: float = 0.0
) -> Tuple[float, float, float]:
    """
    Compute selection gradients (π_s - π̄) for each strategy.

    Parameters
    ----------
    p_d, p_cc, p_h : float
        Proportions in the 3-strategy sub-simplex (should sum to 1)
    k, x0, init_coop : float
        Model parameters
    convergence_db : dict
        Pre-computed convergence database
    p_allc : float
        Fixed proportion of AllC strategy

    Returns
    -------
    Tuple[float, float, float]
        (grad_D, grad_CC, grad_H) - selection gradients
    """
    from .simulation import compute_fitness

    scale = 1.0 - p_allc
    actual_p_d = p_d * scale
    actual_p_cc = p_cc * scale
    actual_p_h = p_h * scale

    try:
        payoffs, mean_payoff = compute_fitness(
            p_allc, actual_p_d, actual_p_cc, actual_p_h,
            k, x0, init_coop, convergence_db
        )
    except (ValueError, KeyError):
        return np.nan, np.nan, np.nan

    grad_d = payoffs['AllD'] - mean_payoff
    grad_cc = payoffs['CC'] - mean_payoff
    grad_h = payoffs['H'] - mean_payoff

    return grad_d, grad_cc, grad_h


def classify_equilibrium_jacobian(
    point: Dict[str, float],
    k: float, x0: float, init_coop: float,
    convergence_db: dict,
    epsilon: float = 1e-4
) -> int:
    """
    Classify equilibrium using numerical Jacobian.

    Parameters
    ----------
    point : dict
        Equilibrium point {'AllC', 'AllD', 'CC', 'H'}
    k, x0, init_coop : float
        Model parameters
    convergence_db : dict
        Pre-computed convergence database
    epsilon : float
        Perturbation for numerical differentiation

    Returns
    -------
    int
        EQ_TYPE_STABLE, EQ_TYPE_UNSTABLE, or EQ_TYPE_SADDLE
    """
    p_allc = point.get('AllC', 0.0)
    scale = 1.0 - p_allc

    # Get proportions in sub-simplex
    if scale > 0:
        p_d = point['AllD'] / scale
        p_cc = point['CC'] / scale
        p_h = point['H'] / scale
    else:
        return EQ_TYPE_STABLE  # All AllC

    # Check if corner (vertex equilibrium)
    if p_d > 0.99 or p_cc > 0.99 or p_h > 0.99:
        # Perturb slightly towards interior and check if absent strategies can invade
        corner_configs = {
            p_d > 0.99: ((0.98, 0.01, 0.01), (1, 2)),   # D corner: check CC, H
            p_cc > 0.99: ((0.01, 0.98, 0.01), (0, 2)),   # CC corner: check D, H
            p_h > 0.99: ((0.01, 0.01, 0.98), (0, 1)),    # H corner: check D, CC
        }
        perturbed_coords, check_indices = corner_configs[True]
        grads = compute_selection_gradients(
            *perturbed_coords, k, x0, init_coop, convergence_db, p_allc
        )
        if np.isnan(grads[0]):
            return EQ_TYPE_SADDLE
        if all(grads[i] <= 0 for i in check_indices):
            return EQ_TYPE_STABLE
        return EQ_TYPE_UNSTABLE

    # Interior equilibrium: compute Jacobian numerically
    # Replicator dynamics: dp_s/dt = p_s * (π_s - π̄)
    # We use 2D representation (p_d, p_cc) since p_h = 1 - p_d - p_cc

    def dynamics(p_d_val, p_cc_val):
        p_h_val = 1.0 - p_d_val - p_cc_val
        if p_h_val < 0:
            return 0, 0

        g_d, g_cc, g_h = compute_selection_gradients(
            p_d_val, p_cc_val, p_h_val,
            k, x0, init_coop, convergence_db, p_allc
        )

        if np.isnan(g_d):
            return 0, 0

        # dp/dt = p * g  (but we work in sub-simplex)
        dp_d = p_d_val * g_d
        dp_cc = p_cc_val * g_cc

        return dp_d, dp_cc

    # Numerical Jacobian
    J = np.zeros((2, 2))

    f0_d, f0_cc = dynamics(p_d, p_cc)

    # ∂f/∂p_d
    f1_d, f1_cc = dynamics(p_d + epsilon, p_cc)
    J[0, 0] = (f1_d - f0_d) / epsilon
    J[1, 0] = (f1_cc - f0_cc) / epsilon

    # ∂f/∂p_cc
    f2_d, f2_cc = dynamics(p_d, p_cc + epsilon)
    J[0, 1] = (f2_d - f0_d) / epsilon
    J[1, 1] = (f2_cc - f0_cc) / epsilon

    # Eigenvalues
    try:
        eigenvalues = np.linalg.eigvals(J)
        real_parts = np.real(eigenvalues)

        if np.all(real_parts < -1e-8):
            return EQ_TYPE_STABLE
        elif np.all(real_parts > 1e-8):
            return EQ_TYPE_UNSTABLE
        else:
            return EQ_TYPE_SADDLE
    except np.linalg.LinAlgError:
        return EQ_TYPE_SADDLE


def is_interior_equilibrium(point: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check if equilibrium is interior (all strategies present).

    Parameters
    ----------
    point : np.ndarray
        [p_AllC, p_AllD, p_CC, p_H]
    threshold : float
        Minimum proportion for interior classification

    Returns
    -------
    bool
        True if interior equilibrium
    """
    return np.all(point > threshold)
