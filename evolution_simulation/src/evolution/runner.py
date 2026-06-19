"""
Evolution simulation runner with parallel execution support.

Provides infrastructure for running parameter sweeps across multiple
CPU cores using multiprocessing.
"""

import json
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import traceback

from .params import SimParams
from .grid import EvolutionGrid
from .engine import run_simulation
from .io import save_condition_result


def run_single_condition(args: tuple) -> dict:
    """
    Run a single simulation condition (worker function for multiprocessing).

    Parameters
    ----------
    args : tuple
        (params, output_dir, save_trajectory, overwrite, save_individual)

    Returns
    -------
    dict
        Result summary with status and timing.
    """
    params, output_dir, save_trajectory, overwrite, save_individual = args

    condition_id = f"K={params.K}_seed={params.seed}"
    output_path = Path(output_dir) / f"{condition_id}.parquet"

    result = {
        'condition_id': condition_id,
        'K': params.K,
        'seed': params.seed,
        'status': 'unknown',
        'elapsed_sec': 0.0,
        'error': None,
        'output_path': str(output_path),
    }

    # Skip if already completed and not overwriting
    if output_path.exists() and not overwrite:
        result['status'] = 'skipped'
        return result

    try:
        import time
        start_time = time.perf_counter()

        # Run simulation
        sim_results = run_simulation(params)

        elapsed = time.perf_counter() - start_time
        result['elapsed_sec'] = elapsed

        # Save results
        save_condition_result(
            sim_results,
            output_path,
            params,
            save_trajectory=save_trajectory,
            save_individual=save_individual
        )

        # Add final state summary
        gen_stats = sim_results['generation_stats']
        final_idx = -1
        result['final_p_AllC'] = float(gen_stats['freq_allc'][final_idx])
        result['final_p_AllD'] = float(gen_stats['freq_alld'][final_idx])
        result['final_p_CC'] = float(gen_stats['freq_cc'][final_idx])
        result['final_p_H'] = float(gen_stats['freq_hump'][final_idx])
        result['final_mean_coop'] = float(gen_stats['mean_coop_rate'][final_idx])
        result['final_mean_fitness'] = float(gen_stats['mean_fitness'][final_idx])

        result['status'] = 'completed'

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    return result


@dataclass
class EvolutionRunner:
    """
    Runner for parallel evolution simulations.

    Attributes
    ----------
    grid : EvolutionGrid
        Parameter grid defining conditions to run.
    output_dir : str
        Directory for saving results.
    n_workers : int
        Number of parallel workers (default: CPU count - 1).
    save_trajectory : bool
        Whether to save full trajectory data (default: True).
    save_individual : bool
        Whether to save individual_snapshots data (default: False).
        Warning: This creates large files (~1.5GB per condition for T_gen=25000).
    overwrite : bool
        Whether to overwrite existing results (default: False).
    """

    grid: EvolutionGrid
    output_dir: str
    n_workers: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    save_trajectory: bool = True
    save_individual: bool = False  # Save individual_snapshots (large files!)
    overwrite: bool = False

    # Results tracking
    _results: list = field(default_factory=list, repr=False)
    _start_time: Optional[datetime] = field(default=None, repr=False)
    _end_time: Optional[datetime] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize output directory."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save grid configuration
        grid_path = self.output_dir / "grid_config.json"
        with open(grid_path, 'w') as f:
            json.dump(self.grid.to_dict(), f, indent=2)

    @property
    def n_total(self) -> int:
        """Total number of conditions."""
        return self.grid.total_conditions

    @property
    def n_completed(self) -> int:
        """Number of completed conditions."""
        return sum(1 for r in self._results if r['status'] == 'completed')

    @property
    def n_skipped(self) -> int:
        """Number of skipped conditions."""
        return sum(1 for r in self._results if r['status'] == 'skipped')

    @property
    def n_failed(self) -> int:
        """Number of failed conditions."""
        return sum(1 for r in self._results if r['status'] == 'failed')

    def _prepare_args(self) -> list:
        """Prepare arguments for worker processes."""
        args_list = []
        for params in self.grid.iter_params():
            args_list.append((
                params,
                str(self.output_dir),
                self.save_trajectory,
                self.overwrite,
                self.save_individual,
            ))
        return args_list

    def run(self) -> list:
        """
        Run all simulations in parallel.

        Returns
        -------
        list
            List of result dictionaries.
        """
        self._start_time = datetime.now()
        self._results = []

        args_list = self._prepare_args()

        if self.n_workers == 1:
            # Sequential execution (useful for debugging)
            for args in args_list:
                self._results.append(run_single_condition(args))
        else:
            # Parallel execution
            with mp.Pool(processes=self.n_workers) as pool:
                for result in pool.imap_unordered(run_single_condition, args_list):
                    self._results.append(result)

        self._end_time = datetime.now()

        # Save summary
        self._save_summary()

        return self._results

    def _save_summary(self) -> None:
        """Save sweep summary to JSON and CSV."""
        summary_path = self.output_dir / "sweep_summary.json"

        # Compute statistics
        completed_results = [r for r in self._results if r['status'] == 'completed']

        elapsed_total = (self._end_time - self._start_time).total_seconds()
        elapsed_compute = sum(r['elapsed_sec'] for r in completed_results)

        summary = {
            'sweep_info': {
                'start_time': self._start_time.isoformat(),
                'end_time': self._end_time.isoformat(),
                'elapsed_total_sec': elapsed_total,
                'elapsed_compute_sec': elapsed_compute,
                'n_workers': self.n_workers,
                'speedup': elapsed_compute / elapsed_total if elapsed_total > 0 else 0,
            },
            'conditions': {
                'total': self.n_total,
                'completed': self.n_completed,
                'skipped': self.n_skipped,
                'failed': self.n_failed,
            },
            'grid': self.grid.to_dict(),
            'results': self._results,
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Also save as CSV for easy analysis
        csv_path = self.output_dir / "sweep_results.csv"
        self._save_results_csv(csv_path)

    def _save_results_csv(self, path: Path) -> None:
        """Save results to CSV file."""
        import csv

        if not self._results:
            return

        # Get all keys from completed results
        completed = [r for r in self._results if r['status'] == 'completed']
        if completed:
            fieldnames = list(completed[0].keys())
        else:
            fieldnames = ['condition_id', 'K', 'seed', 'status',
                         'elapsed_sec', 'error', 'output_path']

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self._results)


