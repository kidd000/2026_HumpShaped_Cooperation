# Code for "Partial exploiters sustain cooperation: the hump-shaped strategy stably coexists with unconditional cooperators"

This repository contains the code behind the paper's two modelling components.
Each lives in its own self-contained package with its own README, dependencies,
and entry points.

| Package | What it does |
|---------|--------------|
| [`evolution_simulation/`](evolution_simulation/) | Agent-based evolutionary simulation of the four strategies in a public goods game (the individual-based N x K sweeps). |
| [`equilibrium_analysis/`](equilibrium_analysis/) | Analytical reproduction of Figure 4 (edge fitness and critical-K boundaries) with focal-player (pivotal) expected payoffs, plus a reference replicator-dynamics library. |

The two packages are independent: the simulation is a stochastic, finite-population
model in which beliefs evolve, while the equilibrium analysis is a deterministic
replicator model in which the expected payoffs are computed analytically from a
multinomial weighting over group compositions. The four strategies are the same
in both (the Hump response `min(x, 1-x)` is written `Hump` in the simulation and
`H` in the equilibrium analysis).

See each package's README for installation and usage.

---

## Reproducing the main-text figures

The sections below describe how the data behind each figure is produced from
the packages above, so the figures can be reconstructed with any plotting tool.
Figure 1 is a schematic and uses no data.

### Figure 2. Evolutionary outcomes across the N x K space

Produced from `evolution_simulation`. Run the N x K sweep over group sizes
`N = 3..16` and steepness `K = 1, 2, ..., 128` (`run_evolution_sweep.py`,
20 seeds per condition). Averaging the strategy frequencies over the final
10 generations of each run gives the steady-state composition for every `(N, K)`,
which feeds the three panels:

- **A** -- the most prevalent strategy at each `(N, K)` (the strategy with the
  highest steady-state frequency, or "mixed" when none exceeds 40%).
- **B** -- the full evolutionary time series for three representative cells,
  `(N, K) = (4, 16)`, `(12, 16)`, and `(16, 16)`.
- **C** -- heatmaps of the mean cooperation rate and of each strategy's
  steady-state frequency (CC, Hump, AllC, AllD) across `(N, K)`.

### Figure 3. The effect of Hump on cooperation and fitness

Also from `evolution_simulation`, comparing the full four-strategy runs against
three-strategy runs without Hump (`--no-hump`), at `K = 16` and `K = 1`:

- **A** -- population cooperation rate and mean fitness as a function of `N`,
  with and without Hump.
- **B** -- the steady-state strategy composition (stacked bars) as a function of
  `N`, with and without Hump.

### Figure 4. The pivotal-AllC mechanism (analytical)

Produced from `equilibrium_analysis` at the representative analytical condition
`N = 12`, the step-function production limit `K -> infinity`, and initial belief
`b0 = 0.56`. The two scripts there are self-contained (numpy + scipy).

- **A** -- a schematic of the pivotal-AllC effect (no data).
- **B** & **D** -- the payoff difference, fitness, and invadability along the
  AllC-AllD and AllC-Hump edges, from `compute_edge_fitness_step_limit.py`.
- **C** -- the region of the `N x K` space in which polymorphic equilibria
  containing AllC exist (the critical-`K` boundaries `K*(N)`), from
  `compute_critical_K_boundaries.py`.

Both rest on the focal-player (pivotal) expected payoffs.
