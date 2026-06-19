# Evolution Simulation

Agent-based evolutionary simulation of cooperation strategies in a public goods
game. This package reproduces the individual-based simulations of the paper
(the N x K sweeps behind the main-text figures).

---

## Model

### Strategies

Each strategy is a response function `f(x)` of the co-players' mean cooperation
rate `x`:

| Strategy | ID | f(x) | Description |
|----------|----|------|-------------|
| AllC | 0 | 1 | Always fully cooperate |
| AllD | 1 | 0 | Always defect |
| CC | 2 | x | Conditional cooperation (match the others' mean) |
| Hump | 3 | min(x, 1 - x) | Piecewise-linear, peaks at 0.5 |

The Hump response equals `x` for `x <= 0.5` and `1 - x` for `x > 0.5`, the same
function written as `H` in the companion `equilibrium_analysis` package.

### Production function

An S-shaped (sigmoid) production function maps the group cooperation rate to
collective output:

```
S(c) = 1 / (1 + exp(-K (c - x0)))
```

- `K`: steepness (larger is more step-like; `inf` gives an exact step function)
- `x0`: inflection point, fixed at 0.5 in the main model

### Payoff

```
pi_i = E (1 + MPCR * S(c_bar) * N - c_i)
```

with endowment `E = 1`, `MPCR = 0.4`, group size `N`, individual cooperation
`c_i`, and group-mean cooperation `c_bar`.

### Evolutionary dynamics

Each generation applies global-pool selection followed by random regrouping:

1. **Global-pool selection** -- all individuals are pooled (group boundaries are
   ignored) and parents are sampled in proportion to fitness.
2. **Belief inheritance** -- each offspring inherits its parent's strategy and
   initial belief, with a symmetric Gaussian perturbation added to the belief:
   `b_offspring = clip_[0,1](b_parent + N(0, sigma_belief))`, `sigma_belief = 0.01`.
3. **Mutation** -- with probability `mu_strat` an individual switches to a
   uniformly random strategy.
4. **Random regrouping** -- individuals are reassigned to fresh groups of size `N`.

The whole population is initialized as AllD with initial belief `b = 0`.

---

## Installation

```bash
# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

Requires Python >= 3.10, `numpy`, `pandas`, `numba`, and `pyarrow`.

---

## Usage

### Command line

The main entry point is `scripts/run_evolution_sweep.py`, which sweeps the
production steepness `K` (and, optionally, the group size `N`) over a set of
random seeds. All other parameters follow the main-text model and are fixed
(`x0 = 0.5`, `sigma_belief = 0.01`, `MPCR = 0.4`, hump threshold `= 0.5`).

```bash
# main-text sweep (K in powers of two up to 128, 20 seeds)
uv run python scripts/run_evolution_sweep.py

# multi-N sweep with custom K values (one output directory per N)
uv run python scripts/run_evolution_sweep.py --K 1,2,4,8,16,32,64,128 --N 3,4,5,6,7,8

# 3-strategy run without Hump
uv run python scripts/run_evolution_sweep.py --no-hump
```

### Python API

```python
from evolution import EvolutionGrid, EvolutionRunner

grid = EvolutionGrid(
    K_values=[1.0, 4.0, 16.0, 64.0],
    seeds=[0, 1, 2, 3, 4],
    M=250,        # number of groups
    N=4,          # group size
    T_gen=50000,  # generations
)

runner = EvolutionRunner(grid=grid, output_dir="output/my_sweep", save_trajectory=True)
results = runner.run()
```

Single condition:

```python
from evolution import SimParams, run_simulation

params = SimParams(K=16.0, x0=0.5, sigma_belief=0.01, M=250, N=4, T_gen=50000, seed=42)
results = run_simulation(params)
```

### Parameters

Swept by `EvolutionGrid`:

| Parameter | Description | Main-text values |
|-----------|-------------|------------------|
| `K_values` | Production steepness | [1, 2, 4, 8, 16, 32, 64, 128] |
| `seeds` | Random seeds | 0..19 (20 seeds) |

Fixed:

| Parameter | Description | Value |
|-----------|-------------|-------|
| `M` | Number of groups | 250 |
| `N` | Group size | 3..16 (swept via `--N`) |
| `T_gen` | Generations | 50000 |
| `x0` | Production inflection point | 0.5 |
| `mu_strat` | Mutation rate | 0.01 |
| `sigma_belief` | Belief-inheritance noise (s.d.) | 0.01 |
| `mpcr` | Marginal per capita return | 0.4 |
| `hump_threshold` | Hump peak threshold | 0.5 |
| `n_strategies` | 3 (no Hump) or 4 | 4 |

---

## Output

Each condition is written to a Parquet file named `K={K}_seed={seed}.parquet`
under the output directory (the multi-N sweep adds an `N{N}/` subdirectory).

| Column | Description |
|--------|-------------|
| `generation` | Generation index |
| `p_AllC`, `p_AllD`, `p_CC`, `p_H` | Strategy frequencies |
| `mean_coop` | Mean cooperation rate |
| `mean_belief` | Mean belief |
| `mean_fitness` | Mean fitness |
| `K`, `x0`, `seed`, `sigma_belief`, ... | Parameter values |
