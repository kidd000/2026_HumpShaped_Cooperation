"""
I/O module for saving and loading simulation results.

Output formats:
- generation_stats.parquet: Generation-level statistics
- strategy_stats.parquet: Strategy-level statistics per generation
- individual_snapshots.parquet: Individual agent data per generation
- metadata.json: Simulation parameters and execution info

Directory structure:
output/
└── evolution_YYYYMMDD_HHMMSS/
    ├── metadata.json
    ├── generation_stats.parquet
    ├── strategy_stats.parquet
    └── individual_snapshots.parquet
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .params import SimParams


def create_output_dir(
    base_dir: str = "output/simulation",
    prefix: str = "evolution",
    timestamp: Optional[str] = None
) -> Path:
    """
    Create timestamped output directory.

    Parameters
    ----------
    base_dir : str
        Base directory for outputs.
    prefix : str
        Prefix for the output folder name.
    timestamp : str, optional
        Custom timestamp. If None, uses current time.

    Returns
    -------
    Path
        Path to created directory.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_results(
    results: dict,
    output_dir: Optional[Path] = None,
    base_dir: str = "output/simulation",
    save_individual_snapshots: bool = True
) -> Path:
    """
    Save simulation results to Parquet and JSON files.

    Parameters
    ----------
    results : dict
        Results dictionary from run_simulation().
    output_dir : Path, optional
        Output directory. If None, creates timestamped directory.
    base_dir : str
        Base directory for outputs (used if output_dir is None).
    save_individual_snapshots : bool
        Whether to save individual snapshots (can be large).

    Returns
    -------
    Path
        Path to output directory.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = create_output_dir(base_dir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    params: SimParams = results['params']

    # Save metadata
    metadata = {
        **params.to_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save generation_stats
    gen_stats_df = pd.DataFrame(results['generation_stats'])
    gen_stats_path = output_dir / "generation_stats.parquet"
    gen_stats_df.to_parquet(gen_stats_path, index=False)

    # Save strategy_stats
    strat_stats_df = pd.DataFrame(results['strategy_stats'])
    strat_stats_path = output_dir / "strategy_stats.parquet"
    strat_stats_df.to_parquet(strat_stats_path, index=False)

    # Save individual_snapshots (optional, can be large)
    if save_individual_snapshots:
        ind_snapshots_df = pd.DataFrame(results['individual_snapshots'])
        ind_snapshots_path = output_dir / "individual_snapshots.parquet"
        ind_snapshots_df.to_parquet(ind_snapshots_path, index=False)

    return output_dir


def save_condition_result(
    results: dict,
    output_path: str | Path,
    params: 'SimParams',
    save_trajectory: bool = True,
    save_individual: bool = False
) -> Path:
    """
    Save a single condition result to a single Parquet file.

    This is optimized for parameter sweeps where each condition
    is saved to a separate file for easy parallel processing.

    Parameters
    ----------
    results : dict
        Results dictionary from run_simulation().
    output_path : str or Path
        Path to the output Parquet file.
    params : SimParams
        Simulation parameters.
    save_trajectory : bool
        If True, save full trajectory. If False, only save final state.
    save_individual : bool
        If True, save individual_snapshots to separate file.
        File will be saved as {output_path}_individual.parquet

    Returns
    -------
    Path
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract generation_stats from results
    gen_stats = results['generation_stats']
    T_gen = len(gen_stats['generation'])

    if save_trajectory:
        # Full trajectory
        data = {
            'generation': gen_stats['generation'],
            'p_AllC': gen_stats['freq_allc'],
            'p_AllD': gen_stats['freq_alld'],
            'p_CC': gen_stats['freq_cc'],
            'p_H': gen_stats['freq_hump'],
            'mean_coop': gen_stats['mean_coop_rate'],
            'mean_belief': gen_stats['mean_belief'],
            'mean_fitness': gen_stats['mean_fitness'],
            'mean_coop_hump': gen_stats['mean_coop_hump'],
        }
    else:
        # Only final state (single row)
        final_idx = -1
        data = {
            'generation': [gen_stats['generation'][final_idx]],
            'p_AllC': [gen_stats['freq_allc'][final_idx]],
            'p_AllD': [gen_stats['freq_alld'][final_idx]],
            'p_CC': [gen_stats['freq_cc'][final_idx]],
            'p_H': [gen_stats['freq_hump'][final_idx]],
            'mean_coop': [gen_stats['mean_coop_rate'][final_idx]],
            'mean_belief': [gen_stats['mean_belief'][final_idx]],
            'mean_fitness': [gen_stats['mean_fitness'][final_idx]],
            'mean_coop_hump': [gen_stats['mean_coop_hump'][final_idx]],
        }

    # Add parameter columns (repeated for each row)
    n_rows = len(data['generation'])
    data['K'] = [params.K] * n_rows
    data['x0'] = [params.x0] * n_rows
    data['seed'] = [params.seed] * n_rows
    data['M'] = [params.M] * n_rows
    data['N'] = [params.N] * n_rows
    data['T_gen'] = [params.T_gen] * n_rows
    data['mpcr'] = [params.mpcr] * n_rows
    data['mu_strat'] = [params.mu_strat] * n_rows
    data['sigma_belief'] = [params.sigma_belief] * n_rows
    data['convergence_threshold'] = [params.convergence_threshold] * n_rows
    data['max_round_iterations'] = [params.max_round_iterations] * n_rows

    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)

    # Save individual snapshots if requested
    if save_individual and 'individual_snapshots' in results:
        ind_snapshots = results['individual_snapshots']
        if ind_snapshots is not None:
            # Create individual output path by replacing .parquet with _individual.parquet
            ind_output_path = output_path.with_name(
                output_path.stem + '_individual.parquet'
            )
            ind_df = pd.DataFrame(ind_snapshots)
            # Add parameter columns for reference
            ind_df['K'] = params.K
            ind_df['x0'] = params.x0
            ind_df['seed'] = params.seed
            ind_df['sigma_belief'] = params.sigma_belief
            ind_df.to_parquet(ind_output_path, index=False)

    return output_path


def load_condition_result(path: str | Path) -> pd.DataFrame:
    """
    Load a single condition result from Parquet file.

    Parameters
    ----------
    path : str or Path
        Path to the Parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame with simulation results.
    """
    return pd.read_parquet(path)


def load_results(output_dir: str | Path) -> dict:
    """
    Load simulation results from saved files.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing saved results.

    Returns
    -------
    dict
        Results dictionary with same structure as run_simulation() output.
    """
    output_dir = Path(output_dir)

    # Load metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    param_keys = ['K', 'x0', 'mpcr', 'M', 'N', 'T_gen',
                  'convergence_threshold', 'max_round_iterations',
                  'mu_strat', 'sigma_belief', 'seed']
    param_dict = {k: metadata[k] for k in param_keys if k in metadata}
    params = SimParams.from_dict(param_dict)

    # Load generation_stats
    gen_stats_path = output_dir / "generation_stats.parquet"
    gen_stats_df = pd.read_parquet(gen_stats_path)
    gen_stats = {col: gen_stats_df[col].values for col in gen_stats_df.columns}

    # Load strategy_stats
    strat_stats_path = output_dir / "strategy_stats.parquet"
    strat_stats_df = pd.read_parquet(strat_stats_path)
    strat_stats = {col: strat_stats_df[col].values for col in strat_stats_df.columns}

    # Load individual_snapshots (if exists)
    ind_snapshots_path = output_dir / "individual_snapshots.parquet"
    if ind_snapshots_path.exists():
        ind_snapshots_df = pd.read_parquet(ind_snapshots_path)
        ind_snapshots = {col: ind_snapshots_df[col].values for col in ind_snapshots_df.columns}
    else:
        ind_snapshots = None

    return {
        'generation_stats': gen_stats,
        'strategy_stats': strat_stats,
        'individual_snapshots': ind_snapshots,
        'params': params,
        'metadata': metadata
    }


def get_final_state(results: dict) -> dict:
    """
    Extract final generation state from results.

    Parameters
    ----------
    results : dict
        Results dictionary.

    Returns
    -------
    dict
        Dictionary with final generation statistics.
    """
    gen_stats = results['generation_stats']
    final_idx = len(gen_stats['generation']) - 1

    return {
        'generation': gen_stats['generation'][final_idx],
        'freq_allc': gen_stats['freq_allc'][final_idx],
        'freq_alld': gen_stats['freq_alld'][final_idx],
        'freq_cc': gen_stats['freq_cc'][final_idx],
        'freq_hump': gen_stats['freq_hump'][final_idx],
        'mean_belief': gen_stats['mean_belief'][final_idx],
        'mean_coop_rate': gen_stats['mean_coop_rate'][final_idx],
        'mean_fitness': gen_stats['mean_fitness'][final_idx],
        'mean_coop_hump': gen_stats['mean_coop_hump'][final_idx],
    }


def summarize_results(results: dict) -> str:
    """
    Generate a text summary of simulation results.

    Parameters
    ----------
    results : dict
        Results dictionary.

    Returns
    -------
    str
        Formatted summary string.
    """
    params = results['params']
    gen_stats = results['generation_stats']
    final = get_final_state(results)

    summary = f"""
Simulation Summary
==================

Parameters:
  K={params.K}, x0={params.x0}, mpcr={params.mpcr}
  M={params.M} groups, N={params.N} per group
  T_gen={params.T_gen}, convergence_threshold={params.convergence_threshold}
  mu_strat={params.mu_strat}, sigma_belief={params.sigma_belief}
  seed={params.seed}

Final State (generation {final['generation']}):
  Strategy frequencies:
    AllC: {final['freq_allc']:.3f}
    AllD: {final['freq_alld']:.3f}
    CC:   {final['freq_cc']:.3f}
    Hump: {final['freq_hump']:.3f}

  Mean belief: {final['mean_belief']:.3f}
  Mean cooperation: {final['mean_coop_rate']:.3f}
  Mean fitness: {final['mean_fitness']:.1f}

  Hump strategy:
    Mean cooperation: {final['mean_coop_hump']:.3f}
"""
    return summary.strip()


