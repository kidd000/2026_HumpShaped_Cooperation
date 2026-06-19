"""
Simulation engine for evolutionary dynamics (global pooling model).

Each generation:
1. Randomly assign individuals to temporary groups
2. Run interaction rounds until convergence (adiabatic approximation)
3. Compute fitness and statistics
4. Perform global pool selection (fitness-proportional)
5. Apply mutation
"""

import numpy as np

from .params import SimParams, N_STRATEGIES, STRATEGY_ALLD
from .interaction import run_generation_adiabatic
from .selection import (
    global_pool_selection,
    random_grouping,
)
from .mutation import apply_mutation, apply_mutation_from_pool
from .statistics import (
    compute_generation_stats,
    compute_strategy_stats,
)


def initialize_population(
    params: SimParams,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize population with AllD strategy and zero beliefs.

    All agents start as defectors (AllD) with belief = 0 (expecting
    others to not cooperate). This represents a pessimistic initial
    state from which cooperation must emerge through evolution.

    Parameters
    ----------
    params : SimParams
        Simulation parameters.
    rng : np.random.Generator
        Random number generator (unused but kept for API consistency).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (strategies, beliefs, group_ids)
    """
    n_total = params.N_total

    # All agents start as AllD (strategy = 1)
    strategies = np.full(n_total, STRATEGY_ALLD, dtype=np.int32)

    # All beliefs start at 0 (pessimistic about others' cooperation)
    beliefs = np.zeros(n_total, dtype=np.float64)

    # Assign to groups (agent i belongs to group i // N)
    group_ids = np.repeat(np.arange(params.M, dtype=np.int32), params.N)

    return strategies, beliefs, group_ids


def run_simulation(params: SimParams) -> dict:
    """
    Run the full evolution simulation.

    Parameters
    ----------
    params : SimParams
        Simulation parameters.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'generation_stats': dict of arrays for generation-level stats
        - 'strategy_stats': dict of arrays for strategy-level stats
        - 'individual_snapshots': dict of arrays for individual-level data
        - 'params': the SimParams used
    """
    # Validate parameters
    errors = params.validate()
    if errors:
        raise ValueError(f"Invalid parameters: {errors}")

    # Setup RNG
    if params.seed is not None:
        rng = np.random.default_rng(params.seed)
        np.random.seed(params.seed)  # For Numba functions
    else:
        rng = np.random.default_rng()

    M, N = params.M, params.N
    n_total = params.N_total
    T_gen = params.T_gen
    E = 1.0  # Endowment (normalized)

    convergence_threshold = params.convergence_threshold
    max_round_iterations = params.max_round_iterations

    # Initialize population
    strategies, beliefs, group_ids = initialize_population(params, rng)

    # Precompute enabled strategy pool for mutation
    # Use pool mutation only when enabled_strategies has been explicitly customized
    _enabled = params.enabled_strategies
    _default_enabled = (0, 1, 2, 3)
    use_pool_mutation = (_enabled != _default_enabled)
    if use_pool_mutation:
        strategy_pool = np.array(_enabled, dtype=np.int32)

    # Preallocate result arrays
    # Generation stats
    gen_stats = {
        'generation': np.empty(T_gen, dtype=np.int32),
        'freq_allc': np.empty(T_gen, dtype=np.float64),
        'freq_alld': np.empty(T_gen, dtype=np.float64),
        'freq_cc': np.empty(T_gen, dtype=np.float64),
        'freq_hump': np.empty(T_gen, dtype=np.float64),
        'mean_belief': np.empty(T_gen, dtype=np.float64),
        'std_belief': np.empty(T_gen, dtype=np.float64),
        'mean_fitness': np.empty(T_gen, dtype=np.float64),
        'mean_coop_rate': np.empty(T_gen, dtype=np.float64),
        'mean_coop_allc': np.empty(T_gen, dtype=np.float64),
        'mean_coop_alld': np.empty(T_gen, dtype=np.float64),
        'mean_coop_cc': np.empty(T_gen, dtype=np.float64),
        'mean_coop_hump': np.empty(T_gen, dtype=np.float64),
    }

    # Strategy stats (T_gen * N_STRATEGIES rows)
    n_strat_rows = T_gen * N_STRATEGIES
    strat_stats = {
        'generation': np.empty(n_strat_rows, dtype=np.int32),
        'strategy': np.empty(n_strat_rows, dtype=np.int32),
        'count': np.empty(n_strat_rows, dtype=np.int32),
        'mean_belief': np.empty(n_strat_rows, dtype=np.float64),
        'mean_fitness': np.empty(n_strat_rows, dtype=np.float64),
        'mean_coop': np.empty(n_strat_rows, dtype=np.float64),
    }

    # Individual snapshots (T_gen * n_total rows)
    n_ind_rows = T_gen * n_total
    ind_snapshots = {
        'generation': np.empty(n_ind_rows, dtype=np.int32),
        'agent_id': np.empty(n_ind_rows, dtype=np.int32),
        'strategy': np.empty(n_ind_rows, dtype=np.int32),
        'group_id': np.empty(n_ind_rows, dtype=np.int32),
        'belief': np.empty(n_ind_rows, dtype=np.float64),
        'mean_coop': np.empty(n_ind_rows, dtype=np.float64),
        'fitness': np.empty(n_ind_rows, dtype=np.float64),
    }

    # Main simulation loop
    for gen in range(T_gen):
        # Step 1: Run interaction until convergence (3-stage: exact, limit_cycle, timeout)
        fitness, mean_coop, final_group_means, _, conv_type, cycle_len = run_generation_adiabatic(
            strategies, beliefs, group_ids,
            params.K, params.x0, params.mpcr, E, N, M,
            convergence_threshold, max_round_iterations,
            params.hump_threshold,
            params.cycle_decimals, params.fallback_window, params.max_cycle_check
        )

        # Step 2: Compute and store statistics
        (freq_by_strat, mean_belief, std_belief, mean_fit,
         mean_coop_rate, mean_coop_by_strat) = compute_generation_stats(
            strategies, beliefs, fitness, mean_coop
        )

        # Store generation stats
        gen_stats['generation'][gen] = gen
        for idx, suffix in enumerate(['allc', 'alld', 'cc', 'hump']):
            gen_stats[f'freq_{suffix}'][gen] = freq_by_strat[idx]
            gen_stats[f'mean_coop_{suffix}'][gen] = mean_coop_by_strat[idx]
        gen_stats['mean_belief'][gen] = mean_belief
        gen_stats['std_belief'][gen] = std_belief
        gen_stats['mean_fitness'][gen] = mean_fit
        gen_stats['mean_coop_rate'][gen] = mean_coop_rate

        # Store strategy stats
        counts, mean_beliefs, mean_fitnesses, mean_coops = compute_strategy_stats(
            strategies, beliefs, fitness, mean_coop
        )
        for s in range(N_STRATEGIES):
            idx = gen * N_STRATEGIES + s
            strat_stats['generation'][idx] = gen
            strat_stats['strategy'][idx] = s
            strat_stats['count'][idx] = counts[s]
            strat_stats['mean_belief'][idx] = mean_beliefs[s]
            strat_stats['mean_fitness'][idx] = mean_fitnesses[s]
            strat_stats['mean_coop'][idx] = mean_coops[s]

        # Store individual snapshots
        start_idx = gen * n_total
        end_idx = start_idx + n_total
        ind_snapshots['generation'][start_idx:end_idx] = gen
        ind_snapshots['agent_id'][start_idx:end_idx] = np.arange(n_total)
        ind_snapshots['strategy'][start_idx:end_idx] = strategies
        ind_snapshots['group_id'][start_idx:end_idx] = group_ids
        ind_snapshots['belief'][start_idx:end_idx] = beliefs
        ind_snapshots['mean_coop'][start_idx:end_idx] = mean_coop
        ind_snapshots['fitness'][start_idx:end_idx] = fitness

        # Step 3: Global pool selection
        strategies, beliefs = global_pool_selection(
            strategies, beliefs, fitness,
            params.sigma_belief, M, N
        )

        # Step 4: Random regrouping for next generation
        group_ids = random_grouping(M, N)

        # Step 5: Mutation
        if use_pool_mutation:
            apply_mutation_from_pool(strategies, params.mu_strat, strategy_pool)
        else:
            apply_mutation(strategies, params.mu_strat, params.n_strategies)

    return {
        'generation_stats': gen_stats,
        'strategy_stats': strat_stats,
        'individual_snapshots': ind_snapshots,
        'params': params
    }


