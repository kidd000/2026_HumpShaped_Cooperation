"""
Interaction module for within-generation dynamics.

Uses adiabatic approximation: iterate interaction rounds until convergence
(instead of a fixed T_round). Convergence detection has 3 stages:
1. Exact/threshold convergence (delta < threshold)
2. Limit cycle detection via state hashing
3. Fallback averaging on timeout

Main output: equilibrium payoffs and cooperation rates used as fitness.
"""

import numpy as np
from numba import njit

from .strategies import get_cooperation_rate
from .production import sigmoid_production


@njit(cache=True)
def run_single_round(
    strategies: np.ndarray,
    beliefs: np.ndarray,
    group_ids: np.ndarray,
    prev_group_means: np.ndarray,
    is_first_round: bool,
    K: float,
    x0: float,
    mpcr: float,
    E: float,
    N: int,
    M: int,
    hump_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single round of interaction for all agents.

    Parameters
    ----------
    strategies : np.ndarray
        Strategy IDs for all agents (shape: M*N).
    beliefs : np.ndarray
        Beliefs for all agents (shape: M*N).
    group_ids : np.ndarray
        Group membership for all agents (shape: M*N).
    prev_group_means : np.ndarray
        Previous round's group mean cooperation (shape: M).
    is_first_round : bool
        Whether this is the first round.
    K : float
        Production function steepness.
    x0 : float
        Production function inflection point.
    mpcr : float
        Marginal per capita return.
    E : float
        Endowment.
    N : int
        Group size.
    M : int
        Number of groups.
    hump_threshold : float
        Threshold for Hump strategy (default: 0.5).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (cooperation_rates, payoffs, group_means)
        - cooperation_rates: shape (M*N,)
        - payoffs: shape (M*N,)
        - group_means: shape (M,) - current round's group means
    """
    n_total = M * N
    cooperation_rates = np.empty(n_total, dtype=np.float64)

    # Step 1: Determine cooperation rates
    for i in range(n_total):
        if is_first_round:
            # Use belief as initial reference
            cooperation_rates[i] = get_cooperation_rate(strategies[i], beliefs[i], hump_threshold)
        else:
            # Use previous group mean
            g = group_ids[i]
            cooperation_rates[i] = get_cooperation_rate(strategies[i], prev_group_means[g], hump_threshold)

    # Step 2: Compute group means
    group_means = np.zeros(M, dtype=np.float64)
    for i in range(n_total):
        g = group_ids[i]
        group_means[g] += cooperation_rates[i]
    for g in range(M):
        group_means[g] /= N

    # Step 3: Compute payoffs
    payoffs = np.empty(n_total, dtype=np.float64)
    for i in range(n_total):
        g = group_ids[i]
        c_bar = group_means[g]
        S = sigmoid_production(c_bar, K, x0)
        payoffs[i] = E * (1.0 + mpcr * S * N - cooperation_rates[i])

    return cooperation_rates, payoffs, group_means


# Convergence type constants
CONV_EXACT = 0
CONV_LIMIT_CYCLE = 1
CONV_TIMEOUT = 2


@njit(cache=True)
def _round_state(group_means: np.ndarray, decimals: int) -> np.ndarray:
    """Round group means to specified decimal places for state comparison."""
    multiplier = 10.0 ** decimals
    result = np.empty_like(group_means)
    for i in range(len(group_means)):
        result[i] = np.round(group_means[i] * multiplier) / multiplier
    return result


@njit(cache=True)
def _states_equal(state1: np.ndarray, state2: np.ndarray) -> bool:
    """Check if two states are equal."""
    for i in range(len(state1)):
        if state1[i] != state2[i]:
            return False
    return True


@njit(cache=True)
def run_generation_adiabatic(
    strategies: np.ndarray,
    beliefs: np.ndarray,
    group_ids: np.ndarray,
    K: float,
    x0: float,
    mpcr: float,
    E: float,
    N: int,
    M: int,
    convergence_threshold: float = 0.01,
    max_iterations: int = 1000,
    hump_threshold: float = 0.5,
    cycle_decimals: int = 6,
    fallback_window: int = 10,
    max_cycle_check: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Run generation with adiabatic approximation and 3-stage convergence detection.

    1. Stage 1: Exact/threshold convergence (max_change < threshold)
    2. Stage 2: Limit cycle detection via state hashing
    3. Stage 3: Fallback averaging on timeout

    Parameters
    ----------
    strategies : np.ndarray
        Strategy IDs for all agents (shape: M*N).
    beliefs : np.ndarray
        Beliefs for all agents (shape: M*N).
    group_ids : np.ndarray
        Group membership for all agents (shape: M*N).
    K : float
        Production function steepness.
    x0 : float
        Production function inflection point.
    mpcr : float
        Marginal per capita return.
    E : float
        Endowment.
    N : int
        Group size.
    M : int
        Number of groups.
    convergence_threshold : float
        Stop when max|group_mean_t - group_mean_{t-1}| < threshold.
        Default 0.01.
    max_iterations : int
        Maximum iterations to prevent infinite loops.
        Default 1000.
    hump_threshold : float
        Threshold for Hump strategy (default: 0.5).
    cycle_decimals : int
        Decimal places for state hashing in cycle detection.
        Default 6.
    fallback_window : int
        Window size for fallback averaging on timeout.
        Default 10.
    max_cycle_check : int
        Maximum number of past states to check for cycles.
        Default 50.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]
        (equilibrium_payoffs, equilibrium_coop, equilibrium_group_means,
         iterations, convergence_type, cycle_length)
        - equilibrium_payoffs: Payoff at convergence (shape: M*N)
        - equilibrium_coop: Cooperation rates at convergence (shape: M*N)
        - equilibrium_group_means: Group means at convergence (shape: M)
        - iterations: Number of iterations until convergence
        - convergence_type: 0=exact, 1=limit_cycle, 2=timeout
        - cycle_length: Cycle period (0 if not a cycle)
    """
    n_total = M * N

    # Initialize group means
    group_means = np.zeros(M, dtype=np.float64)
    prev_group_means = np.zeros(M, dtype=np.float64)

    # Track final round values
    final_coop_rates = np.zeros(n_total, dtype=np.float64)
    final_payoffs = np.zeros(n_total, dtype=np.float64)

    # State history for cycle detection (max_cycle_check x M)
    state_history = np.zeros((max_cycle_check, M), dtype=np.float64)
    n_states_stored = 0

    # History for fallback averaging
    coop_history = np.zeros((fallback_window, n_total), dtype=np.float64)
    payoff_history = np.zeros((fallback_window, n_total), dtype=np.float64)
    group_mean_history = np.zeros((fallback_window, M), dtype=np.float64)

    iterations = 0
    convergence_type = CONV_TIMEOUT
    cycle_length = 0

    for t in range(max_iterations):
        is_first = (t == 0)

        coop_rates, payoffs, group_means = run_single_round(
            strategies, beliefs, group_ids, group_means, is_first,
            K, x0, mpcr, E, N, M, hump_threshold
        )

        iterations = t + 1

        # Store in circular buffer for fallback averaging
        buf_idx = t % fallback_window
        for i in range(n_total):
            coop_history[buf_idx, i] = coop_rates[i]
            payoff_history[buf_idx, i] = payoffs[i]
        for g in range(M):
            group_mean_history[buf_idx, g] = group_means[g]

        # Check convergence (skip first iteration)
        if t > 0:
            # Stage 1: Check for exact or threshold convergence
            max_change = 0.0
            for g in range(M):
                change = abs(group_means[g] - prev_group_means[g])
                if change > max_change:
                    max_change = change

            if max_change < convergence_threshold:
                convergence_type = CONV_EXACT
                for i in range(n_total):
                    final_coop_rates[i] = coop_rates[i]
                    final_payoffs[i] = payoffs[i]
                break

            # Stage 2: Check for limit cycle
            current_state = _round_state(group_means, cycle_decimals)

            for j in range(n_states_stored):
                if _states_equal(current_state, state_history[j]):
                    # Cycle detected!
                    cycle_length = n_states_stored - j
                    convergence_type = CONV_LIMIT_CYCLE

                    # Compute cycle average
                    cycle_start_idx = j
                    for i in range(n_total):
                        avg_coop = 0.0
                        avg_pay = 0.0
                        for k in range(cycle_length):
                            hist_idx = (cycle_start_idx + k) % fallback_window
                            avg_coop += coop_history[hist_idx, i]
                            avg_pay += payoff_history[hist_idx, i]
                        final_coop_rates[i] = avg_coop / cycle_length
                        final_payoffs[i] = avg_pay / cycle_length

                    for g in range(M):
                        avg_gm = 0.0
                        for k in range(cycle_length):
                            hist_idx = (cycle_start_idx + k) % fallback_window
                            avg_gm += group_mean_history[hist_idx, g]
                        group_means[g] = avg_gm / cycle_length
                    break

            if convergence_type == CONV_LIMIT_CYCLE:
                break

            # Store current state for future cycle detection
            if n_states_stored < max_cycle_check:
                for g in range(M):
                    state_history[n_states_stored, g] = current_state[g]
                n_states_stored += 1

        # Store for next iteration comparison
        for g in range(M):
            prev_group_means[g] = group_means[g]

        # Store current values (in case we hit max_iterations)
        for i in range(n_total):
            final_coop_rates[i] = coop_rates[i]
            final_payoffs[i] = payoffs[i]

    # Stage 3: Fallback averaging on timeout
    if convergence_type == CONV_TIMEOUT:
        n_avg = min(iterations, fallback_window)
        for i in range(n_total):
            avg_coop = 0.0
            avg_pay = 0.0
            for k in range(n_avg):
                avg_coop += coop_history[k, i]
                avg_pay += payoff_history[k, i]
            final_coop_rates[i] = avg_coop / n_avg
            final_payoffs[i] = avg_pay / n_avg

        for g in range(M):
            avg_gm = 0.0
            for k in range(n_avg):
                avg_gm += group_mean_history[k, g]
            group_means[g] = avg_gm / n_avg

    return final_payoffs, final_coop_rates, group_means, iterations, convergence_type, cycle_length


