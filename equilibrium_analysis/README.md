# Equilibrium Analysis

Analytical reproduction of the Figure 4 panels for the public goods game with
four strategies (AllC, AllD, CC, H), plus a reference implementation of the
replicator-dynamics machinery.

The two scripts under `scripts/` reproduce the data behind Figure 4B-D and are
self-contained (numpy + scipy). The `src/equilibrium/` library is provided as a
reference implementation of the focal-player expected payoffs, convergence
database, and equilibrium solver used in the broader analysis (e.g. the
Supplementary Information); it is not required for the Figure 4 panels.

---

## Model

### Strategies

Each strategy is a response function `f(x)` of the co-players' mean cooperation
rate `x`:

| Strategy | Symbol | f(x) | Description |
|----------|--------|------|-------------|
| All-C | AllC | 1 | Always fully cooperate |
| All-D | AllD | 0 | Always defect |
| Conditional cooperation | CC | x | Match the others' mean |
| Hump-shaped | H | x if x <= 0.5 else 1 - x | Piecewise-linear, peaks at 0.5 |

`H` is the same function written as `Hump` in the companion
`evolution_simulation` package.

### Production and payoff

The collective output is an S-shaped function of the group cooperation rate
`c_bar`, and payoffs are ratio-based:

```
S(c) = 1 / (1 + exp(-K (c - x0)))           # x0 = 0.5 in the main model
pi_i = E (1 + MPCR * S(c_bar) * N - c_i)     # E = 1, MPCR = 0.4
```

### Focal-player (pivotal) expected payoffs

A focal individual's expected payoff is obtained by drawing its `N - 1`
co-players from the population and adding the focal player itself, so a
strategy's payoff reflects its marginal effect on group production. This
pivotal formulation is what lets a rare AllC switch a group's production on,
and it underlies both the edge-fitness computation below and the reference
library.

---

## Reproducing Figure 4

Figure 4 is evaluated at the representative analytical condition `N = 12`,
the step-function production limit `K -> infinity`, and initial belief
`b0 = 0.56`. Panel A is a schematic (no data).

### Panels B and D -- fitness along the AllC-exploiter edges

`compute_edge_fitness_step_limit.py` computes, along the AllC-AllD and
AllC-Hump edges, the resident and invader fitnesses, the population mean, the
invasion fitnesses, and the equilibria (sign change of the fitness
difference). It uses the step-function (`K -> infinity`) production limit and
is self-contained (no convergence database needed).

```bash
uv run python scripts/compute_edge_fitness_step_limit.py --N 12 --b0 0.56 --x0 0.5
```

Output: `edge_AllD_AllC_Kinf_N12.csv` and `edge_AllC_HS_Kinf_N12.csv` under the
output directory (columns include `alpha, pi_left, pi_right, pi_mean,
pi_invader1, pi_invader2, is_equilibrium, eq_stable`).

### Panel C -- critical-K boundaries K*(N)

`compute_critical_K_boundaries.py` computes the closed-form critical steepness
`K*(N)` above which a polymorphic equilibrium containing AllC exists, for the
AllC-AllD edge (`K -> infinity` pivotal condition) and the AllC-Hump edge
(`b0` marginalized over Uniform[0,1]). Self-contained (numpy + scipy).

```bash
uv run python scripts/compute_critical_K_boundaries.py -o analytical_boundaries.csv
```

Output CSV columns: `N, K_critical, edge_type` (`edge_type` is `AllC-AllD` or
`AllC-Hump`).

> No precomputed data is shipped; run the scripts above to regenerate it.

---

## Reference library (`src/equilibrium/`)

Provided for completeness; not needed to reproduce Figure 4.

| Module | Role |
|--------|------|
| `simulation` | Production function, strategies, composition generation, within-group convergence, multinomial / focal expected payoffs, replicator dynamics |
| `solver` | Selection gradients, Jacobian-based stability classification, interior-equilibrium predicate |
| `grid` | Parameter grid definitions |
| `storage` | HDF5 storage for large equilibrium sweeps |

These implement the focal expected payoffs and replicator equilibrium search
over the full strategy simplex, used for the broader/SI analyses.

---

## Installation

```bash
# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

The Figure 4 scripts need only `numpy`, `scipy`, and `pandas`. The reference
library additionally uses `numba` and `h5py`. Requires Python >= 3.10.
