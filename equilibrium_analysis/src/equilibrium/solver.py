"""
Equilibrium Solver Module
=========================

Numerical methods for finding equilibria in replicator dynamics.

Methods:
- Root-finding (scipy.optimize.root) for equilibrium detection
- Jacobian-based stability classification

Note:
    Uses scipy.optimize.root instead of minimize for better convergence.
    Root-finding directly solves g(p) = 0 with quadratic convergence,
    while minimize(|g|²) has linear convergence and flat minima issues.
"""

import numpy as np
from scipy.optimize import root
from typing import List, Dict, Tuple, Optional

from .storage import EquilibriumResult, EQ_TYPE_STABLE, EQ_TYPE_UNSTABLE, EQ_TYPE_SADDLE


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


def find_equilibria_3d(
    k: float, x0: float, init_coop: float,
    convergence_db: dict,
    p_allc: float = 0.0,
    n_initial_points: int = 20,
    tol: float = 1e-8,
    dedup_threshold: float = 0.02
) -> List[Dict[str, float]]:
    """
    Find equilibria numerically using root-finding.

    Uses scipy.optimize.root (Powell's hybrid method) which has quadratic
    convergence near roots, unlike minimize which has linear convergence.

    Parameters
    ----------
    k, x0, init_coop : float
        Model parameters
    convergence_db : dict
        Pre-computed convergence database
    p_allc : float
        Fixed proportion of AllC (default: 0)
    n_initial_points : int
        Grid resolution for initial points
    tol : float
        Tolerance for equilibrium detection (default: 1e-8)
    dedup_threshold : float
        Threshold for deduplication

    Returns
    -------
    List[Dict[str, float]]
        List of equilibrium points {'AllC', 'AllD', 'CC', 'H'}
    """
    scale = 1.0 - p_allc

    def equilibrium_equations(x):
        """
        Equilibrium conditions: π_D - π_H = 0, π_CC - π_H = 0

        Using differences between strategies (rather than π_s - π̄)
        is numerically more stable as it avoids computing mean payoff.

        Returns [0, 0] at equilibrium points.
        """
        p_d, p_cc = x
        p_h = 1.0 - p_d - p_cc

        # Outside simplex: return large values to guide solver back
        if p_d < -0.1 or p_cc < -0.1 or p_h < -0.1 or p_d > 1.1 or p_cc > 1.1 or p_h > 1.1:
            return [1e6, 1e6]

        gd, gcc, gh = compute_selection_gradients(
            p_d, p_cc, p_h,
            k, x0, init_coop, convergence_db, p_allc
        )

        if np.isnan(gd) or np.isnan(gcc) or np.isnan(gh):
            return [1e6, 1e6]

        # Return [π_D - π_H, π_CC - π_H] (at equilibrium, all equal to π̄)
        # Note: g_s = π_s - π̄, so g_D - g_H = π_D - π_H
        return [gd - gh, gcc - gh]

    raw_equilibria = []

    # Try grid of initial points
    for i in range(n_initial_points):
        for j in range(n_initial_points - i):
            p_d0 = i / n_initial_points
            p_cc0 = j / n_initial_points
            p_h0 = 1.0 - p_d0 - p_cc0

            if p_h0 < 0:
                continue

            try:
                # Use Powell's hybrid method (modified Newton)
                sol = root(
                    equilibrium_equations,
                    [p_d0, p_cc0],
                    method='hybr',
                    tol=tol
                )

                # Check if solution is valid
                if sol.success and np.max(np.abs(sol.fun)) < tol * 100:
                    p_d, p_cc = sol.x
                    p_h = 1.0 - p_d - p_cc

                    # Check if inside simplex (with small tolerance)
                    if p_d >= -tol and p_cc >= -tol and p_h >= -tol:
                        p_d = max(0, p_d)
                        p_cc = max(0, p_cc)
                        p_h = max(0, p_h)

                        # Normalize to ensure sum = 1
                        total = p_d + p_cc + p_h
                        if total > 0:
                            p_d /= total
                            p_cc /= total
                            p_h /= total

                        raw_equilibria.append({
                            'AllC': p_allc,
                            'AllD': p_d * scale,
                            'CC': p_cc * scale,
                            'H': p_h * scale
                        })
            except Exception:
                pass

    def _is_duplicate(candidate, existing_list):
        for ueq in existing_list:
            dist = (abs(candidate['AllD'] - ueq['AllD']) +
                   abs(candidate['CC'] - ueq['CC']) +
                   abs(candidate['H'] - ueq['H']))
            if dist < dedup_threshold:
                return True
        return False

    # Deduplicate raw equilibria
    equilibria = []
    for eq in raw_equilibria:
        if not _is_duplicate(eq, equilibria):
            equilibria.append(eq)

    # Add corner equilibria
    for p_d, p_cc, p_h in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        corner_eq = {
            'AllC': p_allc,
            'AllD': p_d * scale,
            'CC': p_cc * scale,
            'H': p_h * scale
        }
        if not _is_duplicate(corner_eq, equilibria):
            equilibria.append(corner_eq)

    return equilibria


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


def compute_equilibria_for_params(
    k: float, x0: float, init_coop: float,
    convergence_db: dict,
    p_allc: float = 0.0,
    n_initial_points: int = 15
) -> EquilibriumResult:
    """
    Compute equilibria for a single parameter set.

    This is the main function for sweep computation.

    Parameters
    ----------
    k, x0, init_coop : float
        Model parameters
    convergence_db : dict
        Pre-computed convergence database
    p_allc : float
        Fixed AllC proportion
    n_initial_points : int
        Grid resolution for initial points

    Returns
    -------
    EquilibriumResult
        Result with equilibrium points and types
    """
    # Find equilibria
    equilibria = find_equilibria_3d(
        k, x0, init_coop, convergence_db,
        p_allc=p_allc,
        n_initial_points=n_initial_points
    )

    if not equilibria:
        return EquilibriumResult.empty()

    # Classify each equilibrium
    n_eq = len(equilibria)
    points = np.zeros((n_eq, 4), dtype=np.float32)
    types = np.zeros(n_eq, dtype=np.int8)

    for i, eq in enumerate(equilibria):
        points[i] = [eq['AllC'], eq['AllD'], eq['CC'], eq['H']]
        types[i] = classify_equilibrium_jacobian(
            eq, k, x0, init_coop, convergence_db
        )

    return EquilibriumResult(points=points, types=types, n_equilibria=n_eq)


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
