#!/usr/bin/env python
"""
CLI script for running evolution simulation parameter sweeps.

Usage:
    # Default grid (K powers of two up to 128, 20 seeds)
    uv run python scripts/run_evolution_sweep.py

    # Custom K values over several group sizes
    uv run python scripts/run_evolution_sweep.py --K 1,16,128 --N 3,4,5,6

    # Custom output directory and worker count
    uv run python scripts/run_evolution_sweep.py -o output/my_sweep -w 8

Results are written under the output directory; the multi-N sweep adds an
N{N}/ subdirectory per group size. Returns a non-zero exit code if any
condition fails.
"""

import argparse
import sys
from pathlib import Path

# Add package src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution import EvolutionGrid, EvolutionRunner


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(',')]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',')]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run evolution simulation parameter sweep')
    parser.add_argument('--K', type=str, default=None,
                        help='K values (comma-separated, e.g. "1,2,4,8,16,32,64,128")')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Random seeds (comma-separated)')
    parser.add_argument('--no-hump', action='store_true',
                        help='Run without Hump (3 strategies: AllC, AllD, CC)')
    parser.add_argument('--M', type=int, default=None, help='Number of groups (default: 250)')
    parser.add_argument('--N', type=str, default=None,
                        help='Group size(s); multiple comma-separated values run a per-N sweep')
    parser.add_argument('--T-gen', type=int, default=None, help='Number of generations (default: 50000)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory (default: output/simulation/data/x050/sweep[_no_hump])')
    parser.add_argument('-w', '--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--no-trajectory', action='store_true',
                        help='Only save final state (faster, smaller files)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    return parser


def build_grid(args) -> EvolutionGrid:
    grid = EvolutionGrid()
    if args.K:
        grid.K_values = parse_float_list(args.K)
    if args.seeds:
        grid.seeds = parse_int_list(args.seeds)

    grid.n_strategies = 3 if args.no_hump else 4
    if args.M is not None:
        grid.M = args.M
    if args.T_gen is not None:
        grid.T_gen = args.T_gen
    return grid


def output_dir_for(args, grid: EvolutionGrid) -> str:
    if args.output:
        return args.output
    x0_str = f"{int(grid.x0 * 100):03d}"
    subdir = "sweep_no_hump" if args.no_hump else "sweep"
    return f"output/simulation/data/x{x0_str}/{subdir}"


def run_one(grid: EvolutionGrid, output_dir: str, args) -> int:
    kwargs = {
        'grid': grid,
        'output_dir': output_dir,
        'save_trajectory': not args.no_trajectory,
        'overwrite': args.overwrite,
    }
    if args.workers:
        kwargs['n_workers'] = args.workers
    results = EvolutionRunner(**kwargs).run()
    return 1 if any(r['status'] == 'failed' for r in results) else 0


def main() -> int:
    args = build_parser().parse_args()
    grid = build_grid(args)
    N_values = parse_int_list(args.N) if args.N else None
    base_output = output_dir_for(args, grid)

    # Single group size
    if not N_values or len(N_values) == 1:
        if N_values:
            grid.N = N_values[0]
        return run_one(grid, base_output, args)

    # Multi-N sweep: one output subdirectory per group size
    exit_code = 0
    for N in N_values:
        grid.N = N
        exit_code |= run_one(grid, str(Path(base_output) / f"N{N}"), args)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
