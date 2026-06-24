"""
Hump-shaped Strategy Evolutionary Simulation
=============================================

Replicator dynamics analysis for public goods game with 4 strategies:
- AllC: Always cooperate (c=1)
- AllD: Always defect (c=0)
- CC: Conditional Cooperation (c=mean_others)
- H: Hump-shaped (c=x if x<=0.5 else 1-x)

All values are ratio-based (0-1), with E=1 normalization.
"""

import numpy as np
from numba import njit
from itertools import combinations_with_replacement


# =============================================================================
# Constants
# =============================================================================

STRATEGY_NAMES = ["AllC", "AllD", "CC", "H"]
STRATEGY_IDS = {"AllC": 0, "AllD": 1, "CC": 2, "H": 3}
N_STRATEGIES = 4
DEFAULT_GROUP_SIZE = 4
DEFAULT_MPCR = 0.4
DEFAULT_THRESHOLD = 1e-2   # SI: converge when max_i |c_i^t - c_i^{t-1}| < 1e-2
DEFAULT_MAX_ROUNDS = 1000  # SI: max within-group rounds before timeout averaging

# New convergence detection constants
DEFAULT_CYCLE_DECIMALS = 6  # Decimal places for state hashing
DEFAULT_FALLBACK_WINDOW = 10  # Window size for fallback averaging


# =============================================================================
# Production Function (S-shaped)
# =============================================================================

@njit(cache=True)
def supply_sigmoid_raw(c: float, k: float, x0: float) -> float:
    """
    Raw S-shaped production function (NOT normalized).

    Parameters
    ----------
    c : float
        Group average cooperation rate (0-1)
    k : float
        Steepness parameter (1-50)
    x0 : float
        Inflection point (0-1)

    Returns
    -------
    float
        Production output (NOT guaranteed to be 0-1 at boundaries)

    Note
    ----
    This function does NOT pass through (0,0) and (1,1).
    Use supply_sigmoid() for the normalized version.
    """
    return 1.0 / (1.0 + np.exp(-k * (c - x0)))


@njit(cache=True)
def supply_sigmoid(c: float, k: float, x0: float) -> float:
    """
    Normalized S-shaped production function.

    Guarantees S(0) = 0 and S(1) = 1 for any k and x0.

    S_norm(c) = (S_raw(c) - S_raw(0)) / (S_raw(1) - S_raw(0))

    Parameters
    ----------
    c : float
        Group average cooperation rate (0-1)
    k : float
        Steepness parameter (1-50)
    x0 : float
        Inflection point (0-1)

    Returns
    -------
    float
        Production output (0-1), guaranteed to pass through (0,0) and (1,1)
    """
    # Raw sigmoid values
    s_c = 1.0 / (1.0 + np.exp(-k * (c - x0)))
    s_0 = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    s_1 = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))

    # Normalize to [0, 1]
    denominator = s_1 - s_0
    if denominator < 1e-10:
        # Edge case: when k is very small, function is nearly flat
        return c  # Fall back to linear
    return (s_c - s_0) / denominator


# =============================================================================
# Strategy Functions
# =============================================================================


@njit(cache=True)
def strategy_hump(mean_others: float, threshold: float = 0.5) -> float:
    """
    Hump-shaped: Piecewise linear with configurable threshold.

    f(x) = x              if x <= threshold
    f(x) = 2*threshold-x  if threshold < x < 2*threshold
    f(x) = 0              if x >= 2*threshold

    Peak at x=threshold with value threshold.

    Parameters
    ----------
    mean_others : float
        Mean cooperation rate of others (0-1)
    threshold : float
        Threshold for hump peak (default: 0.5, can be synced with x0)

    Returns
    -------
    float
        Cooperation rate (0 to threshold)
    """
    if mean_others <= threshold:
        return mean_others
    elif mean_others < 2.0 * threshold:
        return 2.0 * threshold - mean_others
    else:
        return 0.0


@njit(cache=True)
def get_strategy_response(strategy_id: int, mean_others: float,
                          hump_threshold: float = 0.5) -> float:
    """
    Get cooperation rate based on strategy ID.

    Parameters
    ----------
    strategy_id : int
        0=AllC, 1=AllD, 2=CC, 3=H
    mean_others : float
        Mean cooperation rate of others (0-1)
    hump_threshold : float
        Threshold for Hump-shaped strategy (default: 0.5)

    Returns
    -------
    float
        Cooperation rate (0-1)
    """
    if strategy_id == 0:  # AllC
        return 1.0
    elif strategy_id == 1:  # AllD
        return 0.0
    elif strategy_id == 2:  # CC
        return mean_others
    else:  # H (strategy_id == 3)
        return strategy_hump(mean_others, hump_threshold)


# =============================================================================
# Payoff Function
# =============================================================================

# =============================================================================
# Composition Generation
# =============================================================================

def generate_compositions(
    n_strategies: int = N_STRATEGIES,
    group_size: int = DEFAULT_GROUP_SIZE
) -> list:
    """
    Generate all possible group compositions.

    For 4 strategies and 4 players: C(4+4-1, 4) = 35 compositions.

    Parameters
    ----------
    n_strategies : int
        Number of strategies (default: 4)
    group_size : int
        Number of players per group (default: 4)

    Returns
    -------
    list of tuple
        List of compositions, each as (n_AllC, n_AllD, n_CC, n_H)
    """
    compositions = set()
    for combo in combinations_with_replacement(range(n_strategies), group_size):
        comp = [0] * n_strategies
        for s in combo:
            comp[s] += 1
        compositions.add(tuple(comp))
    return sorted(list(compositions))


def compositions_to_array(compositions: list) -> np.ndarray:
    """
    Convert compositions list to numpy array for Numba compatibility.

    Parameters
    ----------
    compositions : list of tuple
        List of compositions

    Returns
    -------
    np.ndarray
        Shape (n_compositions, n_strategies), dtype int32
    """
    return np.array(compositions, dtype=np.int32)


# =============================================================================
# Simulation Helpers
# =============================================================================

def _assign_strategy_ids(n_allc: int, n_alld: int, n_cc: int, n_h: int) -> np.ndarray:
    """Create strategy ID array from counts. 0=AllC, 1=AllD, 2=CC, 3=H."""
    N = n_allc + n_alld + n_cc + n_h
    strategy_ids = np.zeros(N, dtype=np.int32)
    idx = 0
    for sid, count in enumerate([n_allc, n_alld, n_cc, n_h]):
        for _ in range(count):
            strategy_ids[idx] = sid
            idx += 1
    return strategy_ids


def _compute_group_payoffs_from_rates(
    coop_rates: np.ndarray, k: float, x0: float, mpcr: float = DEFAULT_MPCR
) -> np.ndarray:
    """Compute payoffs for all players from cooperation rates."""
    N = len(coop_rates)
    group_mean = np.mean(coop_rates)
    S_val = supply_sigmoid(group_mean, k, x0)
    payoffs = np.zeros(N)
    for i in range(N):
        payoffs[i] = 1.0 + mpcr * N * S_val - coop_rates[i]
    return payoffs


def _simulate_round(
    coop_rates: np.ndarray, strategy_ids: np.ndarray, hump_threshold: float
) -> tuple:
    """Execute one simultaneous-update round. Returns (new_rates, max_change)."""
    N = len(coop_rates)
    new_rates = np.zeros(N)
    max_change = 0.0
    for i in range(N):
        others_sum = np.sum(coop_rates) - coop_rates[i]
        mean_others = others_sum / (N - 1) if N > 1 else 0.0
        new_rates[i] = get_strategy_response(strategy_ids[i], mean_others, hump_threshold)
        change = abs(new_rates[i] - coop_rates[i])
        if change > max_change:
            max_change = change
    return new_rates, max_change


def simulate_composition(
    n_allc: int,
    n_alld: int,
    n_cc: int,
    n_h: int,
    k: float,
    x0: float,
    init_coop: float,
    mpcr: float = DEFAULT_MPCR,
    threshold: float = DEFAULT_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    hump_threshold: float = 0.5,
    cycle_decimals: int = DEFAULT_CYCLE_DECIMALS,
    fallback_window: int = DEFAULT_FALLBACK_WINDOW
) -> dict:
    """
    Simulate a single group composition with 3-stage convergence detection.

    Convergence Detection Priority:
    1. Exact convergence: max_change == 0
    2. Limit cycle: detect repeated state and compute cycle average
    3. Fallback: compute average of last `fallback_window` rounds

    Parameters
    ----------
    n_allc, n_alld, n_cc, n_h : int
        Number of players for each strategy
    k : float
        Production function steepness (1-50)
    x0 : float
        Production function inflection point (0-1)
    init_coop : float
        Initial cooperation rate for all players (0-1)
    mpcr : float
        Marginal per capita return (default: 0.4)
    threshold : float
        Convergence threshold for near-zero detection (default: 1e-6)
    max_rounds : int
        Maximum rounds before stopping (default: 100)
    hump_threshold : float
        Threshold for Hump-shaped strategy (default: 0.5)
    cycle_decimals : int
        Decimal places for state hashing (default: 6)
    fallback_window : int
        Window size for fallback averaging (default: 10)

    Returns
    -------
    dict
        {
            'final_rates': np.ndarray,      # Final/averaged cooperation rates
            'payoffs': np.ndarray,          # Final/averaged payoffs
            'rounds': int,                   # Number of rounds executed
            'converged': bool,               # Whether any convergence was detected
            'convergence_type': str,         # 'exact', 'limit_cycle', or 'timeout'
            'cycle_length': int,             # Length of cycle (0 if not cycle)
            'cycle_start': int               # Round where cycle started (-1 if not cycle)
        }
    """
    N = n_allc + n_alld + n_cc + n_h

    if N == 0:
        return {
            'final_rates': np.zeros(0),
            'payoffs': np.zeros(0),
            'rounds': 0,
            'converged': True,
            'convergence_type': 'exact',
            'cycle_length': 0,
            'cycle_start': -1
        }

    strategy_ids = _assign_strategy_ids(n_allc, n_alld, n_cc, n_h)
    # Strategy-specific initial cooperation: AllC=1, AllD=0 from the start;
    # CC and Hump start from the common initial belief b0 (= init_coop).
    coop_rates = np.full(N, init_coop)
    coop_rates[strategy_ids == 0] = 1.0  # AllC
    coop_rates[strategy_ids == 1] = 0.0  # AllD

    # State history for cycle detection. The common pre-update state is NOT
    # registered, so it cannot pollute a detected limit cycle's average.
    # payoff_history is kept aligned with rate_history (payoff_history[j] is the
    # payoff vector of rate_history[j]) so that limit-cycle / timeout fitness can
    # average per-step payoffs rather than recomputing S at the mean cooperation.
    state_history = {}
    rate_history = []
    payoff_history = []

    convergence_type = 'timeout'
    cycle_length = 0
    cycle_start = -1
    rounds = 0

    for round_num in range(max_rounds):
        rounds = round_num + 1
        coop_rates_new, max_change = _simulate_round(coop_rates, strategy_ids, hump_threshold)
        coop_rates = coop_rates_new

        if max_change < threshold:
            convergence_type = 'exact'
            break

        state_hash = tuple(np.round(coop_rates, decimals=cycle_decimals))
        if state_hash in state_history:
            cycle_start = state_history[state_hash]
            cycle_length = len(rate_history) - cycle_start
            convergence_type = 'limit_cycle'
            break

        state_history[state_hash] = len(rate_history)
        rate_history.append(coop_rates.copy())
        payoff_history.append(
            _compute_group_payoffs_from_rates(coop_rates, k, x0, mpcr)
        )

    # Fitness evaluation per SI Sec. 2. Production S is nonlinear, so we average
    # per-step payoffs over the relevant window, NOT S(mean(c)).
    #   (i) exact      -> payoff at the terminal step
    #   (ii) cycle     -> payoffs averaged over one full period
    #   (iii) timeout  -> payoffs averaged over the final `fallback_window` steps
    if convergence_type == 'limit_cycle':
        coop_rates = np.mean(np.array(rate_history[cycle_start:]), axis=0)
        payoffs = np.mean(np.array(payoff_history[cycle_start:]), axis=0)
    elif convergence_type == 'timeout':
        window = min(fallback_window, len(rate_history))
        if window < 1:
            payoffs = _compute_group_payoffs_from_rates(coop_rates, k, x0, mpcr)
        else:
            coop_rates = np.mean(np.array(rate_history[-window:]), axis=0)
            payoffs = np.mean(np.array(payoff_history[-window:]), axis=0)
    else:  # exact
        payoffs = _compute_group_payoffs_from_rates(coop_rates, k, x0, mpcr)

    return {
        'final_rates': coop_rates,
        'payoffs': payoffs,
        'rounds': rounds,
        'converged': convergence_type != 'timeout',
        'convergence_type': convergence_type,
        'cycle_length': cycle_length,
        'cycle_start': cycle_start
    }


@njit(cache=True)
def aggregate_payoffs_by_strategy(
    coop_rates: np.ndarray,
    payoffs: np.ndarray,
    n_allc: int,
    n_alld: int,
    n_cc: int,
    n_h: int
) -> tuple:
    """
    Aggregate cooperation rates and payoffs by strategy.

    Returns
    -------
    tuple
        (strategy_coop_rates, strategy_payoffs)
        Each is shape (4,) for [AllC, AllD, CC, H]
        Values are means for each strategy (0 if strategy not present)
    """
    strategy_coop = np.zeros(4)
    strategy_payoff = np.zeros(4)
    counts = np.array([n_allc, n_alld, n_cc, n_h])

    idx = 0
    # AllC
    if n_allc > 0:
        for _ in range(n_allc):
            strategy_coop[0] += coop_rates[idx]
            strategy_payoff[0] += payoffs[idx]
            idx += 1
        strategy_coop[0] /= n_allc
        strategy_payoff[0] /= n_allc

    # AllD
    if n_alld > 0:
        for _ in range(n_alld):
            strategy_coop[1] += coop_rates[idx]
            strategy_payoff[1] += payoffs[idx]
            idx += 1
        strategy_coop[1] /= n_alld
        strategy_payoff[1] /= n_alld

    # CC
    if n_cc > 0:
        for _ in range(n_cc):
            strategy_coop[2] += coop_rates[idx]
            strategy_payoff[2] += payoffs[idx]
            idx += 1
        strategy_coop[2] /= n_cc
        strategy_payoff[2] /= n_cc

    # H
    if n_h > 0:
        for _ in range(n_h):
            strategy_coop[3] += coop_rates[idx]
            strategy_payoff[3] += payoffs[idx]
            idx += 1
        strategy_coop[3] /= n_h
        strategy_payoff[3] /= n_h

    return strategy_coop, strategy_payoff, counts


def build_convergence_database(
    k_values: np.ndarray,
    x0_values: np.ndarray,
    init_coop_values: np.ndarray,
    compositions: list = None,
    mpcr: float = DEFAULT_MPCR,
    threshold: float = DEFAULT_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    hump_threshold: float = 0.5,
    cycle_decimals: int = DEFAULT_CYCLE_DECIMALS,
    fallback_window: int = DEFAULT_FALLBACK_WINDOW,
) -> dict:
    """
    Build convergence database with 3-stage convergence detection.

    Uses simulate_composition with:
    1. Exact convergence detection
    2. Limit cycle detection
    3. Fallback averaging

    Parameters
    ----------
    k_values : np.ndarray
        Array of k (steepness) values
    x0_values : np.ndarray
        Array of x0 (inflection point) values
    init_coop_values : np.ndarray
        Array of initial cooperation rates
    compositions : list, optional
        List of compositions to simulate (default: all 35)
    mpcr : float
        Marginal per capita return
    threshold : float
        Convergence threshold
    max_rounds : int
        Maximum rounds
    hump_threshold : float
        Threshold for Hump-shaped strategy (default: 0.5)
    cycle_decimals : int
        Decimal places for state hashing (default: 6)
    fallback_window : int
        Window size for fallback averaging (default: 10)

    Returns
    -------
    dict
        Nested dictionary: db[(k, x0, init_coop)][composition] = result_dict
        Result dict includes: final_rates, payoffs, strategy_coop, strategy_payoff,
        counts, rounds, converged, convergence_type, cycle_length, cycle_start,
        hump_threshold
    """
    if compositions is None:
        compositions = ALL_COMPOSITIONS

    db = {}

    for k in k_values:
        for x0_val in x0_values:
            effective_hump_threshold = hump_threshold

            for init_coop in init_coop_values:
                key = (float(k), float(x0_val), float(init_coop))
                db[key] = {}

                for comp in compositions:
                    n_c, n_d, n_cc, n_h = comp
                    result = simulate_composition(
                        n_c, n_d, n_cc, n_h,
                        float(k), float(x0_val), float(init_coop),
                        mpcr, threshold, max_rounds,
                        effective_hump_threshold,
                        cycle_decimals, fallback_window
                    )

                    rates = result['final_rates']
                    payoffs = result['payoffs']
                    s_coop, s_pay, counts = aggregate_payoffs_by_strategy(
                        rates, payoffs, n_c, n_d, n_cc, n_h
                    )

                    db[key][comp] = {
                        'final_rates': rates.copy(),
                        'payoffs': payoffs.copy(),
                        'strategy_coop': s_coop.copy(),
                        'strategy_payoff': s_pay.copy(),
                        'counts': counts.copy(),
                        'rounds': result['rounds'],
                        'converged': result['converged'],
                        'convergence_type': result['convergence_type'],
                        'cycle_length': result['cycle_length'],
                        'cycle_start': result['cycle_start'],
                        'hump_threshold': effective_hump_threshold
                    }

    return db


def summarize_database(db: dict) -> dict:
    """
    Summarize convergence database statistics.

    Returns
    -------
    dict
        Summary statistics
    """
    total_sims = 0
    converged_count = 0
    total_rounds = 0
    max_rounds_seen = 0

    for params, compositions in db.items():
        for comp, result in compositions.items():
            total_sims += 1
            if result['converged']:
                converged_count += 1
            total_rounds += result['rounds']
            if result['rounds'] > max_rounds_seen:
                max_rounds_seen = result['rounds']

    return {
        'total_simulations': total_sims,
        'converged': converged_count,
        'convergence_rate': converged_count / total_sims if total_sims > 0 else 0,
        'avg_rounds': total_rounds / total_sims if total_sims > 0 else 0,
        'max_rounds': max_rounds_seen,
        'n_param_combinations': len(db)
    }


# =============================================================================
# Phase 2-1: Multinomial Probability
# =============================================================================

@njit(cache=True)
def factorial(n: int) -> int:
    """
    Calculate factorial of n.

    Parameters
    ----------
    n : int
        Non-negative integer

    Returns
    -------
    int
        n! = n * (n-1) * ... * 1
    """
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@njit(cache=True)
def multinomial_prob(
    n_c: int, n_d: int, n_cc: int, n_h: int,
    p_c: float, p_d: float, p_cc: float, p_h: float
) -> float:
    """
    Calculate multinomial distribution probability.

    P(n | p) = n! / (n_c! n_d! n_cc! n_h!) × p_c^n_c × p_d^n_d × p_cc^n_cc × p_h^n_h

    Parameters
    ----------
    n_c, n_d, n_cc, n_h : int
        Number of players for each strategy in the composition
    p_c, p_d, p_cc, p_h : float
        Population proportion for each strategy (must sum to 1)

    Returns
    -------
    float
        Probability of this composition given strategy proportions

    Notes
    -----
    For n=4 (group size), this gives the probability that a randomly
    formed group has this specific composition.
    """
    n = n_c + n_d + n_cc + n_h

    # Multinomial coefficient: n! / (n_c! n_d! n_cc! n_h!)
    coeff = factorial(n) / (factorial(n_c) * factorial(n_d) *
                            factorial(n_cc) * factorial(n_h))

    # Probability term: p_c^n_c × p_d^n_d × p_cc^n_cc × p_h^n_h
    # Handle zero proportions: 0^0 = 1, 0^n = 0 for n > 0
    prob = 1.0
    if n_c > 0:
        if p_c <= 0:
            return 0.0
        prob *= p_c ** n_c
    if n_d > 0:
        if p_d <= 0:
            return 0.0
        prob *= p_d ** n_d
    if n_cc > 0:
        if p_cc <= 0:
            return 0.0
        prob *= p_cc ** n_cc
    if n_h > 0:
        if p_h <= 0:
            return 0.0
        prob *= p_h ** n_h

    return coeff * prob


def expected_payoffs(
    p_c: float, p_d: float, p_cc: float, p_h: float,
    k: float, x0: float, init_coop: float,
    convergence_db: dict
) -> dict:
    """
    Calculate expected payoffs using focal player (N-1 others) method.

    For focal player of strategy s:
        pi_s(p) = sum_{others} P(others | p) * payoff_s(others + {focal_s})

    where N-1 co-players are drawn from Multinomial(N-1; p). This correctly
    accounts for the pivotal player effect: AllC and AllD see different group
    means because AllC cooperates (c=1) and AllD defects (c=0).
    """
    payoffs = {'AllC': 0.0, 'AllD': 0.0, 'CC': 0.0, 'H': 0.0}
    total_prob = {'AllC': 0.0, 'AllD': 0.0, 'CC': 0.0, 'H': 0.0}

    param_key = (float(k), float(x0), float(init_coop))

    if param_key not in convergence_db:
        raise ValueError(f"Parameter combination {param_key} not in database")

    comp_db = convergence_db[param_key]
    strategy_names = ['AllC', 'AllD', 'CC', 'H']

    # For each full composition, consider each strategy s as the focal player.
    # The N-1 co-players are: full_comp minus 1 focal player of type s.
    for comp in comp_db.keys():
        n_c, n_d, n_cc, n_h = comp
        counts = [n_c, n_d, n_cc, n_h]

        comp_result = comp_db[comp]
        strategy_payoffs = comp_result['strategy_payoff']

        for s in range(4):
            if counts[s] < 1:
                continue  # focal player of type s not present in this composition

            # N-1 co-players (others) = full composition minus 1 focal player of type s
            o = list(counts)
            o[s] -= 1

            # Probability of this others composition under Multinomial(N-1; p)
            prob = multinomial_prob(o[0], o[1], o[2], o[3], p_c, p_d, p_cc, p_h)

            if prob <= 0:
                continue

            s_name = strategy_names[s]
            payoffs[s_name] += prob * strategy_payoffs[s]
            total_prob[s_name] += prob

    # Normalize by total probability mass (should be ~1 for each strategy)
    for s in payoffs:
        if total_prob[s] > 0:
            payoffs[s] /= total_prob[s]
        else:
            payoffs[s] = 0.0

    return payoffs


def compute_fitness(
    p_c: float, p_d: float, p_cc: float, p_h: float,
    k: float, x0: float, init_coop: float,
    convergence_db: dict
) -> tuple:
    """
    Compute fitness values for replicator dynamics.

    Returns both individual strategy payoffs and population mean payoff.

    Parameters
    ----------
    (same as expected_payoffs)

    Returns
    -------
    tuple
        (payoffs_dict, mean_payoff)
        - payoffs_dict: {'AllC': float, ...} expected payoff per strategy
        - mean_payoff: float, population-weighted average payoff
    """
    payoffs = expected_payoffs(p_c, p_d, p_cc, p_h, k, x0, init_coop, convergence_db)

    # Mean population payoff
    mean_payoff = (p_c * payoffs['AllC'] + p_d * payoffs['AllD'] +
                   p_cc * payoffs['CC'] + p_h * payoffs['H'])

    return payoffs, mean_payoff


# =============================================================================
# Phase 3-1: Replicator Dynamics
# =============================================================================

def replicator_dynamics(
    p0: dict,
    k: float,
    x0: float,
    init_coop: float,
    convergence_db: dict,
    dt: float = 0.01,
    T: float = 100.0,
    threshold: float = 1e-8,
    record_interval: int = 1
) -> dict:
    """
    Simulate replicator dynamics for strategy proportions.

    Replicator equation:
        dp_s/dt = p_s * (π_s - π̄)

    where:
    - p_s: proportion of strategy s in population
    - π_s: expected payoff of strategy s
    - π̄: mean population payoff

    Parameters
    ----------
    p0 : dict
        Initial strategy proportions {'AllC': float, 'AllD': float, 'CC': float, 'H': float}
        Must sum to 1.0
    k : float
        Production function steepness
    x0 : float
        Production function inflection point
    init_coop : float
        Initial cooperation rate for group simulations
    convergence_db : dict
        Pre-computed convergence database
    dt : float
        Time step for Euler integration (default: 0.01)
    T : float
        Maximum simulation time (default: 100.0)
    threshold : float
        Convergence threshold for stopping (default: 1e-8)
    record_interval : int
        Record trajectory every N steps (default: 1)

    Returns
    -------
    dict
        - 'trajectory': list of dict, strategy proportions at each recorded time
        - 'times': list of float, time points
        - 'final': dict, final strategy proportions
        - 'converged': bool, whether dynamics converged
        - 'steps': int, number of steps taken
    """
    # Initialize proportions
    p = {
        'AllC': float(p0.get('AllC', 0.0)),
        'AllD': float(p0.get('AllD', 0.0)),
        'CC': float(p0.get('CC', 0.0)),
        'H': float(p0.get('H', 0.0))
    }

    # Normalize to ensure sum = 1
    total = sum(p.values())
    if total > 0:
        for s in p:
            p[s] /= total

    trajectory = [p.copy()]
    times = [0.0]

    max_steps = int(T / dt)
    converged = False
    step = 0

    for step in range(1, max_steps + 1):
        t = step * dt

        # Compute fitness
        try:
            payoffs, mean_payoff = compute_fitness(
                p['AllC'], p['AllD'], p['CC'], p['H'],
                k, x0, init_coop, convergence_db
            )
        except ValueError:
            # Parameter not in database
            break

        # Replicator update: dp_s/dt = p_s * (π_s - π̄)
        dp = {}
        max_change = 0.0
        for s in p:
            dp[s] = p[s] * (payoffs[s] - mean_payoff) * dt
            if abs(dp[s]) > max_change:
                max_change = abs(dp[s])

        # Apply updates
        for s in p:
            p[s] += dp[s]

        # Enforce simplex constraints: p_s >= 0
        for s in p:
            if p[s] < 0:
                p[s] = 0.0

        # Renormalize to sum = 1
        total = sum(p.values())
        if total > 0:
            for s in p:
                p[s] /= total
        else:
            # All zero (shouldn't happen), reset to uniform
            for s in p:
                p[s] = 0.25

        # Record trajectory
        if step % record_interval == 0:
            trajectory.append(p.copy())
            times.append(t)

        # Check convergence
        if max_change < threshold:
            converged = True
            break

    # Ensure final state is recorded
    if len(trajectory) == 0 or trajectory[-1] != p:
        trajectory.append(p.copy())
        times.append(step * dt)

    return {
        'trajectory': trajectory,
        'times': times,
        'final': p.copy(),
        'converged': converged,
        'steps': step
    }


def replicator_dynamics_rk45(
    p0: dict,
    k: float,
    x0: float,
    init_coop: float,
    convergence_db: dict,
    t_max: float = 500.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = 1.0
) -> dict:
    """
    Simulate replicator dynamics using RK45 adaptive integrator.

    Uses scipy.integrate.solve_ivp with RK45 for adaptive step-size integration.
    State is represented as 3D vector (p_AllC, p_AllD, p_CC); p_H = 1 - sum,
    so the simplex sum constraint is enforced automatically.

    Replicator equation:
        dp_s/dt = p_s * (pi_s - pi_bar)

    Parameters
    ----------
    p0 : dict
        Initial strategy proportions {'AllC': float, 'AllD': float, 'CC': float, 'H': float}
        Must sum to 1.0
    k : float
        Production function steepness
    x0 : float
        Production function inflection point
    init_coop : float
        Initial cooperation rate for group simulations (adiabatic parameter b0)
    convergence_db : dict
        Pre-computed convergence database
    t_max : float
        Maximum integration time (default: 500.0)
    rtol : float
        Relative tolerance for RK45 (default: 1e-8)
    atol : float
        Absolute tolerance for RK45 (default: 1e-10)
    max_step : float
        Maximum allowed step size (default: 1.0)

    Returns
    -------
    dict
        - 'trajectory': list of dict, strategy proportions at each recorded time
        - 'times': list of float, time points
        - 'final': dict, final strategy proportions {'AllC', 'AllD', 'CC', 'H'}
        - 'converged': bool, True if convergence event triggered (max|dp/dt| < 1e-10)
        - 'steps': int, number of accepted RK45 steps
        - 'total_clamp': float, total clamping magnitude applied to enforce p_s >= 0
    """
    from scipy.integrate import solve_ivp

    # Normalize initial conditions
    p_c0 = float(p0.get('AllC', 0.0))
    p_d0 = float(p0.get('AllD', 0.0))
    p_cc0 = float(p0.get('CC', 0.0))
    p_h0 = float(p0.get('H', 0.0))

    total0 = p_c0 + p_d0 + p_cc0 + p_h0
    if total0 > 0:
        p_c0 /= total0
        p_d0 /= total0
        p_cc0 /= total0
    else:
        p_c0 = p_d0 = p_cc0 = 0.25

    # Initial 3D state: (p_AllC, p_AllD, p_CC); p_H = 1 - sum
    y0 = np.array([p_c0, p_d0, p_cc0], dtype=float)

    # Accumulate clamping magnitude across all evaluations
    total_clamp = [0.0]

    def _clamped_payoffs(y):
        """Clamp y to simplex, compute fitness, return (payoffs, mean, clamp_amount)."""
        p_c, p_d, p_cc_val = y
        p_h_val = 1.0 - p_c - p_d - p_cc_val

        # Measure violation before clamping
        clamp_amount = (abs(min(0.0, p_c)) + abs(min(0.0, p_d)) +
                        abs(min(0.0, p_cc_val)) + abs(min(0.0, p_h_val)))

        # Clamp negatives for payoff calculation
        p_c_c = max(0.0, p_c)
        p_d_c = max(0.0, p_d)
        p_cc_c = max(0.0, p_cc_val)
        p_h_c = max(0.0, p_h_val)

        # Renormalize after clamping
        total_p = p_c_c + p_d_c + p_cc_c + p_h_c
        if total_p > 0:
            p_c_c /= total_p
            p_d_c /= total_p
            p_cc_c /= total_p
            p_h_c /= total_p

        payoffs, mean_payoff = compute_fitness(
            p_c_c, p_d_c, p_cc_c, p_h_c,
            k, x0, init_coop, convergence_db
        )
        return payoffs, mean_payoff, clamp_amount, p_c_c, p_d_c, p_cc_c

    def ode_rhs(t, y):
        """Replicator ODE right-hand side in 3D (p_AllC, p_AllD, p_CC)."""
        try:
            payoffs, mean_payoff, clamp_amount, p_c_c, p_d_c, p_cc_c = _clamped_payoffs(y)
        except (ValueError, KeyError):
            return np.zeros(3)

        total_clamp[0] += clamp_amount

        dp_c = p_c_c * (payoffs['AllC'] - mean_payoff)
        dp_d = p_d_c * (payoffs['AllD'] - mean_payoff)
        dp_cc = p_cc_c * (payoffs['CC'] - mean_payoff)

        return np.array([dp_c, dp_d, dp_cc])

    def convergence_event(t, y):
        """
        Event function for convergence detection.
        Returns max|dp/dt| - 1e-10; triggers (crosses zero downward) when converged.
        """
        try:
            payoffs, mean_payoff, _, p_c_c, p_d_c, p_cc_c = _clamped_payoffs(y)
        except (ValueError, KeyError):
            return 1.0  # Non-zero → no event

        p_h_c = max(0.0, 1.0 - p_c_c - p_d_c - p_cc_c)

        dp_c = p_c_c * (payoffs['AllC'] - mean_payoff)
        dp_d = p_d_c * (payoffs['AllD'] - mean_payoff)
        dp_cc = p_cc_c * (payoffs['CC'] - mean_payoff)
        dp_h = p_h_c * (payoffs['H'] - mean_payoff)

        max_rate = max(abs(dp_c), abs(dp_d), abs(dp_cc), abs(dp_h))
        return max_rate - 1e-10

    convergence_event.terminal = True
    convergence_event.direction = -1  # Trigger when crossing zero from above (converging)

    # Run RK45 integrator
    sol = solve_ivp(
        ode_rhs,
        t_span=(0.0, t_max),
        y0=y0,
        method='RK45',
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=convergence_event,
        dense_output=False
    )

    # Build trajectory as list of dicts (clamp & renormalize each recorded point)
    trajectory = []
    for i in range(len(sol.t)):
        p_c_i = float(sol.y[0, i])
        p_d_i = float(sol.y[1, i])
        p_cc_i = float(sol.y[2, i])
        p_h_i = 1.0 - p_c_i - p_d_i - p_cc_i

        p_c_i = max(0.0, p_c_i)
        p_d_i = max(0.0, p_d_i)
        p_cc_i = max(0.0, p_cc_i)
        p_h_i = max(0.0, p_h_i)

        total_i = p_c_i + p_d_i + p_cc_i + p_h_i
        if total_i > 0:
            p_c_i /= total_i
            p_d_i /= total_i
            p_cc_i /= total_i
            p_h_i /= total_i

        trajectory.append({
            'AllC': p_c_i,
            'AllD': p_d_i,
            'CC': p_cc_i,
            'H': p_h_i
        })

    final = trajectory[-1] if trajectory else {
        'AllC': float(y0[0]),
        'AllD': float(y0[1]),
        'CC': float(y0[2]),
        'H': max(0.0, 1.0 - float(y0[0]) - float(y0[1]) - float(y0[2]))
    }

    # Convergence: event was triggered
    converged = len(sol.t_events[0]) > 0

    if total_clamp[0] > 1e-6:
        import warnings
        warnings.warn(
            f"replicator_dynamics_rk45: total clamp = {total_clamp[0]:.3e} "
            f"(k={k}, x0={x0}, init_coop={init_coop})",
            RuntimeWarning
        )

    return {
        'trajectory': trajectory,
        'times': list(sol.t),
        'final': final,
        'converged': converged,
        'steps': max(0, len(sol.t) - 1),
        'total_clamp': total_clamp[0]
    }


def replicator_dynamics_3strategy(
    p0: dict,
    k: float,
    x0: float,
    init_coop: float,
    convergence_db: dict,
    dt: float = 0.01,
    T: float = 100.0,
    threshold: float = 1e-8,
    record_interval: int = 1
) -> dict:
    """
    Simulate replicator dynamics with AllC excluded (3 strategies only).

    Uses only AllD, CC, H strategies (p_AllC = 0).
    This is for ternary plot visualization.

    Parameters
    ----------
    p0 : dict
        Initial proportions {'AllD': float, 'CC': float, 'H': float}
        Must sum to 1.0 (AllC is implicitly 0)

    Returns
    -------
    dict
        Same as replicator_dynamics
    """
    # Set AllC = 0 and normalize others
    p0_full = {
        'AllC': 0.0,
        'AllD': float(p0.get('AllD', 0.0)),
        'CC': float(p0.get('CC', 0.0)),
        'H': float(p0.get('H', 0.0))
    }

    # Normalize the 3 strategies
    total = p0_full['AllD'] + p0_full['CC'] + p0_full['H']
    if total > 0:
        p0_full['AllD'] /= total
        p0_full['CC'] /= total
        p0_full['H'] /= total

    return replicator_dynamics(
        p0_full, k, x0, init_coop, convergence_db,
        dt, T, threshold, record_interval
    )


def replicator_dynamics_constrained(
    q0: dict,
    p_allc: float,
    k: float,
    x0: float,
    init_coop: float,
    convergence_db: dict,
    dt: float = 0.01,
    T: float = 100.0,
    threshold: float = 1e-8,
    record_interval: int = 1
) -> dict:
    """
    Simulate replicator dynamics with AllC proportion FIXED externally.

    This represents a constrained scenario where AllC proportion is maintained
    at a fixed level (e.g., through external intervention), while the other
    three strategies compete on the remaining (1 - p_allc) proportion.

    The dynamics evolve on the 3-strategy sub-simplex:
        q_D + q_CC + q_H = 1
    where q_s = p_s / (1 - p_allc) are the normalized proportions.

    Parameters
    ----------
    q0 : dict
        Initial normalized proportions {'AllD': float, 'CC': float, 'H': float}
        Must sum to 1.0
    p_allc : float
        Fixed AllC proportion (0 < p_allc < 1)
    k, x0, init_coop : float
        Model parameters
    convergence_db : dict
        Pre-computed convergence database
    dt, T, threshold, record_interval : float/int
        Simulation parameters

    Returns
    -------
    dict
        - 'trajectory': list of dict with normalized q values
        - 'trajectory_full': list of dict with actual p values
        - 'times': list of float
        - 'final_q': dict, final normalized proportions
        - 'final_p': dict, final actual proportions (including p_allc)
        - 'converged': bool
        - 'steps': int
    """
    # Initialize normalized proportions
    q = {
        'AllD': float(q0.get('AllD', 0.0)),
        'CC': float(q0.get('CC', 0.0)),
        'H': float(q0.get('H', 0.0))
    }

    # Normalize to ensure sum = 1
    total = sum(q.values())
    if total > 0:
        for s in q:
            q[s] /= total
    else:
        q = {'AllD': 1/3, 'CC': 1/3, 'H': 1/3}

    scale = 1.0 - p_allc

    trajectory_q = [q.copy()]
    trajectory_p = [{
        'AllC': p_allc,
        'AllD': q['AllD'] * scale,
        'CC': q['CC'] * scale,
        'H': q['H'] * scale
    }]
    times = [0.0]

    converged = False
    max_steps = int(T / dt)

    for step in range(1, max_steps + 1):
        t = step * dt

        # Compute actual proportions for fitness calculation
        p_actual = {
            'AllC': p_allc,
            'AllD': q['AllD'] * scale,
            'CC': q['CC'] * scale,
            'H': q['H'] * scale
        }

        # Get fitness values
        try:
            payoffs, _ = compute_fitness(
                p_actual['AllC'], p_actual['AllD'], p_actual['CC'], p_actual['H'],
                k, x0, init_coop, convergence_db
            )
        except (ValueError, KeyError):
            break

        # Constrained mean payoff (only for D, CC, H)
        mean_payoff_3 = (q['AllD'] * payoffs['AllD'] +
                         q['CC'] * payoffs['CC'] +
                         q['H'] * payoffs['H'])

        # Replicator update for 3 strategies
        dq = {}
        max_change = 0.0
        for s in q:
            dq[s] = q[s] * (payoffs[s] - mean_payoff_3) * dt
            if abs(dq[s]) > max_change:
                max_change = abs(dq[s])

        # Apply updates
        for s in q:
            q[s] += dq[s]

        # Enforce simplex constraints
        for s in q:
            if q[s] < 0:
                q[s] = 0.0

        # Renormalize to sum = 1
        total = sum(q.values())
        if total > 0:
            for s in q:
                q[s] /= total

        # Record trajectory
        if step % record_interval == 0:
            trajectory_q.append(q.copy())
            trajectory_p.append({
                'AllC': p_allc,
                'AllD': q['AllD'] * scale,
                'CC': q['CC'] * scale,
                'H': q['H'] * scale
            })
            times.append(t)

        # Check convergence
        if max_change < threshold:
            converged = True
            break

    # Final state
    final_q = q.copy()
    final_p = {
        'AllC': p_allc,
        'AllD': q['AllD'] * scale,
        'CC': q['CC'] * scale,
        'H': q['H'] * scale
    }

    return {
        'trajectory': trajectory_q,
        'trajectory_full': trajectory_p,
        'times': times,
        'final_q': final_q,
        'final_p': final_p,
        'converged': converged,
        'steps': step
    }


def find_trajectory_endpoint(
    p0: dict,
    k: float,
    x0: float,
    init_coop: float,
    convergence_db: dict,
    dt: float = 0.01,
    T: float = 100.0,
    threshold: float = 1e-8
) -> dict:
    """
    Find the endpoint of replicator dynamics trajectory (no recording).

    Faster than full replicator_dynamics when only the final state is needed.

    Returns
    -------
    dict
        {'final': dict, 'converged': bool, 'steps': int}
    """
    p = {
        'AllC': float(p0.get('AllC', 0.0)),
        'AllD': float(p0.get('AllD', 0.0)),
        'CC': float(p0.get('CC', 0.0)),
        'H': float(p0.get('H', 0.0))
    }

    total = sum(p.values())
    if total > 0:
        for s in p:
            p[s] /= total

    max_steps = int(T / dt)
    converged = False
    step = 0

    for step in range(1, max_steps + 1):
        try:
            payoffs, mean_payoff = compute_fitness(
                p['AllC'], p['AllD'], p['CC'], p['H'],
                k, x0, init_coop, convergence_db
            )
        except ValueError:
            break

        dp = {}
        max_change = 0.0
        for s in p:
            dp[s] = p[s] * (payoffs[s] - mean_payoff) * dt
            if abs(dp[s]) > max_change:
                max_change = abs(dp[s])

        for s in p:
            p[s] += dp[s]
            if p[s] < 0:
                p[s] = 0.0

        total = sum(p.values())
        if total > 0:
            for s in p:
                p[s] /= total

        if max_change < threshold:
            converged = True
            break

    return {
        'final': p.copy(),
        'converged': converged,
        'steps': step
    }


# =============================================================================
# Module Initialization
# =============================================================================

# Pre-generate 35 compositions for default settings
ALL_COMPOSITIONS = generate_compositions()
ALL_COMPOSITIONS_ARRAY = compositions_to_array(ALL_COMPOSITIONS)
