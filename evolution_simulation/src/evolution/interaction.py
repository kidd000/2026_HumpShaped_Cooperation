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
    prev_coop_rates: np.ndarray,
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
    prev_coop_rates : np.ndarray
        Previous round's individual cooperation rates (shape: M*N).
        Used to form each agent's leave-one-out reference.
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
            # React to the leave-one-out mean of the OTHER N-1 group members.
            # The production term (Step 3) still uses the full group mean.
            g = group_ids[i]
            if N > 1:
                others_mean = (N * prev_group_means[g] - prev_coop_rates[i]) / (N - 1)
            else:
                others_mean = prev_group_means[g]
            cooperation_rates[i] = get_cooperation_rate(strategies[i], others_mean, hump_threshold)

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

    # Initialize group means and per-agent cooperation rates
    group_means = np.zeros(M, dtype=np.float64)
    prev_group_means = np.zeros(M, dtype=np.float64)
    prev_coop_rates = np.zeros(n_total, dtype=np.float64)

    # Track final round values
    final_coop_rates = np.zeros(n_total, dtype=np.float64)
    final_payoffs = np.zeros(n_total, dtype=np.float64)

    # Aligned linear history (depth max_cycle_check). Row j holds iteration j's
    # rounded individual-cooperation state and its raw coop/payoff/group means,
    # so cycle detection and cycle averaging share one consistent index space.
    state_history = np.zeros((max_cycle_check, n_total), dtype=np.float64)
    coop_history = np.zeros((max_cycle_check, n_total), dtype=np.float64)
    payoff_history = np.zeros((max_cycle_check, n_total), dtype=np.float64)
    gmean_history = np.zeros((max_cycle_check, M), dtype=np.float64)
    hist_len = 0

    iterations = 0
    convergence_type = CONV_TIMEOUT
    cycle_length = 0

    for t in range(max_iterations):
        is_first = (t == 0)

        coop_rates, payoffs, group_means = run_single_round(
            strategies, beliefs, group_ids, group_means, prev_coop_rates, is_first,
            K, x0, mpcr, E, N, M, hump_threshold
        )

        iterations = t + 1

        # Rounded individual-cooperation state for cycle detection
        current_state = _round_state(coop_rates, cycle_decimals)

        # Check convergence (skip first iteration)
        if t > 0:
            # Stage 1: individual-level threshold convergence
            #   max_i |c_i^t - c_i^{t-1}| < convergence_threshold
            max_change = 0.0
            for i in range(n_total):
                change = abs(coop_rates[i] - prev_coop_rates[i])
                if change > max_change:
                    max_change = change

            if max_change < convergence_threshold:
                convergence_type = CONV_EXACT
                for i in range(n_total):
                    final_coop_rates[i] = coop_rates[i]
                    final_payoffs[i] = payoffs[i]
                break

            # Stage 2: limit cycle on the individual cooperation vector
            found = -1
            for j in range(hist_len):
                if _states_equal(current_state, state_history[j]):
                    found = j
                    break

            if found >= 0:
                # One full period = stored rows found .. hist_len-1
                cycle_length = hist_len - found
                convergence_type = CONV_LIMIT_CYCLE
                for i in range(n_total):
                    avg_coop = 0.0
                    avg_pay = 0.0
                    for k in range(cycle_length):
                        idx = found + k
                        avg_coop += coop_history[idx, i]
                        avg_pay += payoff_history[idx, i]
                    final_coop_rates[i] = avg_coop / cycle_length
                    final_payoffs[i] = avg_pay / cycle_length
                for g in range(M):
                    avg_gm = 0.0
                    for k in range(cycle_length):
                        avg_gm += gmean_history[found + k, g]
                    group_means[g] = avg_gm / cycle_length
                break

        # Record this iteration into the aligned linear history
        if hist_len < max_cycle_check:
            for i in range(n_total):
                state_history[hist_len, i] = current_state[i]
                coop_history[hist_len, i] = coop_rates[i]
                payoff_history[hist_len, i] = payoffs[i]
            for g in range(M):
                gmean_history[hist_len, g] = group_means[g]
            hist_len += 1

        # Store for next iteration comparison
        for g in range(M):
            prev_group_means[g] = group_means[g]
        for i in range(n_total):
            prev_coop_rates[i] = coop_rates[i]

        # Store current values (in case we hit max_iterations)
        for i in range(n_total):
            final_coop_rates[i] = coop_rates[i]
            final_payoffs[i] = payoffs[i]

    # Stage 3: Fallback averaging on timeout (most recent rows of linear history)
    if convergence_type == CONV_TIMEOUT:
        n_avg = min(hist_len, fallback_window)
        if n_avg < 1:
            n_avg = 1
        start = hist_len - n_avg
        for i in range(n_total):
            avg_coop = 0.0
            avg_pay = 0.0
            for k in range(start, hist_len):
                avg_coop += coop_history[k, i]
                avg_pay += payoff_history[k, i]
            final_coop_rates[i] = avg_coop / n_avg
            final_payoffs[i] = avg_pay / n_avg

        for g in range(M):
            avg_gm = 0.0
            for k in range(start, hist_len):
                avg_gm += gmean_history[k, g]
            group_means[g] = avg_gm / n_avg

    return final_payoffs, final_coop_rates, group_means, iterations, convergence_type, cycle_length


