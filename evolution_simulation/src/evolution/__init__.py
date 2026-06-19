"""
Evolution simulation package (global pooling model).

Agent-based evolutionary simulation with 4 strategies (All-C, All-D, CC, Hump-shaped)
in a public goods game. Uses global pool selection and random regrouping each generation.

Modules
-------
params : Simulation parameters (SimParams)
strategies : Strategy response functions (@njit)
production : Production function and payoffs (@njit)
interaction : Within-generation interaction dynamics (@njit)
selection : Global pool selection and random grouping (@njit)
mutation : Strategy mutation (@njit)
statistics : Statistics computation (@njit)
composition : Group composition lookup tables
engine : Main simulation engine
io : File I/O for results
grid, runner : Parameter sweep infrastructure

Usage
-----
>>> from evolution import SimParams, run_simulation, save_results
>>> params = SimParams(M=100, N=4, T_gen=1000, seed=42)
>>> results = run_simulation(params)
>>> save_results(results, base_dir="output")
"""

from .params import SimParams, STRATEGY_ALLC, STRATEGY_ALLD, STRATEGY_CC, STRATEGY_HUMP, N_STRATEGIES
from .engine import run_simulation, initialize_population
from .io import save_results, load_results, get_final_state, summarize_results, save_condition_result, load_condition_result
from .grid import EvolutionGrid
from .runner import EvolutionRunner

__all__ = [
    # Parameters
    "SimParams",
    "STRATEGY_ALLC",
    "STRATEGY_ALLD",
    "STRATEGY_CC",
    "STRATEGY_HUMP",
    "N_STRATEGIES",
    # Engine
    "run_simulation",
    "initialize_population",
    # I/O
    "save_results",
    "load_results",
    "get_final_state",
    "summarize_results",
    "save_condition_result",
    "load_condition_result",
    # Sweep
    "EvolutionGrid",
    "EvolutionRunner",
]
