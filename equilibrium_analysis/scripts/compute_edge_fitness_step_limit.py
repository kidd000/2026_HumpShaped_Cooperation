#!/usr/bin/env python3
"""Compute edge fitness under the step-function limit (K -> infinity).

Generates 2-strategy edge fitness CSVs for Figure 4 (panels B and D) under the
analytical step-function production function. Self-contained (numpy + scipy);
no convergence database needed.

Edges:
    - AllD-AllC  (the AllC-AllD edge)
    - AllC-HS    (the AllC-Hump edge)

Output CSV columns: alpha, pi_left, pi_right, pi_mean, pi_invader1,
pi_invader2, is_equilibrium, eq_stable.

Usage::

    uv run python scripts/compute_edge_fitness_step_limit.py \\
        --N 12 --b0 0.56 --x0 0.5 --resolution 101
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom

# ---------------------------------------------------------------------------
# Constants and CSV schema
# ---------------------------------------------------------------------------
MPCR = 0.4

DEFAULT_N = 12
DEFAULT_B0 = 0.56
DEFAULT_X0 = 0.5
DEFAULT_RESOLUTION = 101
DEFAULT_OUTPUT_DIR = "output/edge_fitness"

CSV_COLUMNS = [
    "alpha",
    "pi_left",
    "pi_right",
    "pi_mean",
    "pi_invader1",
    "pi_invader2",
    "is_equilibrium",
    "eq_stable",
    "alpha_equilibrium",
]


# ---------------------------------------------------------------------------
# Production function (K -> infinity limit)
# ---------------------------------------------------------------------------

def step_function(c: float | np.ndarray, x0: float) -> np.ndarray:
    """Step production function S_inf(c).

    Matches the K -> infinity limit of the sigmoid ``1/(1+exp(-K(c-x0)))``:
        S(c) = 1          if c > x0
        S(c) = 0.5        if c == x0
        S(c) = 0          if c < x0

    The ``c == x0`` tie is resolved via ``np.isclose`` (atol=1e-12) to keep
    consistency with ``compute_allc_alld_edge_analytical.py``.
    """
    c = np.asarray(c, dtype=float)
    return np.where(
        c > x0,
        1.0,
        np.where(np.isclose(c, x0, atol=1e-12), 0.5, 0.0),
    )


# ---------------------------------------------------------------------------
# Edge computations (implemented in Phase 1-2 / Phase 1-3)
# ---------------------------------------------------------------------------

def _hump_reaction(x: np.ndarray | float) -> np.ndarray | float:
    """Piecewise-linear Hump reaction function f(x) = x if x<=0.5 else 1-x."""
    x = np.asarray(x, dtype=float)
    return np.where(x <= 0.5, x, 1.0 - x)


def compute_edge_AllD_AllC(
    alpha_grid: np.ndarray,
    n: int,
    x0: float,
) -> pd.DataFrame:
    """AllD-AllC edge under K -> infinity.

    Edge-dict convention::

        left=AllC, right=AllD,  alpha = proportion of AllD.
        invader1 = CC,           invader2 = H (Hump).

    For each alpha, co-players are drawn independently as AllC (prob 1-alpha)
    or AllD (prob alpha). Let ``k`` = number of AllD co-players among the
    N-1 co-players, distributed as Binom(N-1, alpha). Per-focal payoffs::

        focal AllC (k AllD others):  c = (N-k)/N,   pi = MPCR*N*S(c)
        focal AllD (k AllD others):  c = (N-1-k)/N, pi = 1 + MPCR*N*S(c)
        focal Hump (invader):         co-player mean x = (N-1-k)/(N-1)
                                      f = x if x<=0.5 else 1-x
                                      c_group = ((N-1-k) + f)/N
                                      pi = 1 + MPCR*N*S(c_group) - f
        focal CC (invader):           c_CC = (N-1-k)/(N-1)
                                      c_group = c_CC  (algebraic simplification)
                                      pi = 1 + MPCR*N*S(c_CC) - c_CC

    Boundary NaN: pi_left -> NaN at alpha=1, pi_right -> NaN at alpha=0.
    """
    k_vals = np.arange(n)  # k_AllD = 0, 1, ..., N-1

    c_focal_allc = (n - k_vals) / n
    c_focal_alld = (n - 1 - k_vals) / n
    x_coplayer_mean = (n - 1 - k_vals) / max(n - 1, 1)  # 0 if N==1
    f_hump = _hump_reaction(x_coplayer_mean)
    c_focal_hump = ((n - 1 - k_vals) + f_hump) / n
    c_focal_cc = x_coplayer_mean  # CC's group mean simplifies to its own contribution

    pi_k_allc = MPCR * n * step_function(c_focal_allc, x0)
    pi_k_alld = 1.0 + MPCR * n * step_function(c_focal_alld, x0)
    pi_k_hump = 1.0 + MPCR * n * step_function(c_focal_hump, x0) - f_hump
    pi_k_cc = 1.0 + MPCR * n * step_function(c_focal_cc, x0) - c_focal_cc

    def diff_fn(a):
        """pi_left - pi_right (= pi_AllC - pi_AllD) at arbitrary alpha."""
        probs = binom.pmf(k_vals, n - 1, a)
        return float(np.dot(probs, pi_k_allc - pi_k_alld))

    rows = []
    for a in alpha_grid:
        probs = binom.pmf(k_vals, n - 1, a)
        pi_allc = float(np.dot(probs, pi_k_allc))
        pi_alld = float(np.dot(probs, pi_k_alld))
        pi_hump = float(np.dot(probs, pi_k_hump))
        pi_cc = float(np.dot(probs, pi_k_cc))

        pi_left = pi_allc
        pi_right = pi_alld
        if a == 0.0:
            pi_right = float("nan")
        if a == 1.0:
            pi_left = float("nan")

        # pi_mean = resident population-weighted average ((1-a)*left + a*right)
        left_w = 0.0 if a == 1.0 else (1.0 - a) * pi_allc
        right_w = 0.0 if a == 0.0 else a * pi_alld
        pi_mean = left_w + right_w

        rows.append({
            "alpha": round(float(a), 8),
            "pi_left": pi_left,
            "pi_right": pi_right,
            "pi_mean": pi_mean,
            "pi_invader1": pi_cc,    # invader1 = CC
            "pi_invader2": pi_hump,  # invader2 = H
            "is_equilibrium": False,
            "eq_stable": False,
            "alpha_equilibrium": float("nan"),
        })
    return pd.DataFrame(rows), diff_fn


def _solve_cH_with_AllD(j: int, n: int, b0: float,
                        tol: float = 1e-10, max_iter: int = 500) -> float:
    """Iterative Hump equilibrium when group = 1 AllD + j AllC + (N-1-j) Hump.

    Each resident Hump sees co-players {1 AllD, j AllC, (N-2-j) other Humps}
    within its group of size N. At fixed point: c_H = f(x) where
    x = (j + (N-2-j)*c_H)/(N-1). Starting from c_H = min(b0, 1-b0).
    """
    if n - 1 - j <= 0:
        return 0.0  # no Hump co-players -> c_H irrelevant
    c_H = min(b0, 1.0 - b0)
    for _ in range(max_iter):
        x = (j + (n - 2 - j) * c_H) / (n - 1)
        new_c_H = x if x <= 0.5 else 1.0 - x
        if abs(new_c_H - c_H) < tol:
            return float(new_c_H)
        c_H = new_c_H
    return float(c_H)


def _solve_cH_cC_with_CC(j: int, n: int, b0: float,
                         tol: float = 1e-10, max_iter: int = 500) -> tuple[float, float]:
    """Joint iteration when group = 1 CC + j AllC + (N-1-j) Hump.

    CC tracks its co-player mean; Humps respond to a mixture that includes
    the CC's contribution. Both conditional strategies start from the common
    initial belief b0 (c_H=min(b0,1-b0) for Hump, c_CC=b0 for CC).
    """
    if n - 1 - j <= 0:
        # No Hump co-players: group = 1 CC + (N-1) AllC => c_CC = j/(N-1) = 1
        c_CC = float(j) / max(n - 1, 1)
        return 0.0, c_CC
    c_H = min(b0, 1.0 - b0)
    c_CC = b0
    for _ in range(max_iter):
        # Simultaneous (Jacobi) update per the paper Methods: every player
        # responds to the PREVIOUS round's co-player mean. The CC contribution to
        # a focal Hump's co-player mean is therefore the previous c_CC, not the
        # just-updated new_c_CC. (The fixed point is identical; only the
        # transient path differs from a sequential/Gauss-Seidel update.)
        new_c_CC = (j + (n - 1 - j) * c_H) / (n - 1)
        x_h = (c_CC + j + (n - 2 - j) * c_H) / (n - 1)
        new_c_H = x_h if x_h <= 0.5 else 1.0 - x_h
        if abs(new_c_H - c_H) < tol and abs(new_c_CC - c_CC) < tol:
            return float(new_c_H), float(new_c_CC)
        c_H, c_CC = new_c_H, new_c_CC
    return float(c_H), float(c_CC)


def compute_edge_AllC_HS(
    alpha_grid: np.ndarray,
    n: int,
    x0: float,
    b0: float,
) -> pd.DataFrame:
    """AllC-Hump edge under K -> infinity.

    Edge-dict convention::

        left=AllC, right=H (Hump),  alpha = proportion of Hump.
        invader1 = AllD,             invader2 = CC.

    Let ``j`` = number of AllC co-players (~Binom(N-1, 1-alpha)). Resident
    Hump cooperation rate uses the closed form from SI Methods Sec. 4.B
    (group-level AllC count ``j_group``):

        c_H(j_group) = (N-1-j_group) / (2(N-1)-j_group)   if j_group >= 1
        c_H(0)        = min(b0, 1-b0)                      (all-Hump group)

    - focal AllC  (j_group = j+1, always >= 1):
          group mean = (j+1 + (N-1-j)*c_H) / N,   pi = MPCR*N*S(group mean)
    - focal Hump  (j_group = j):
          group mean = (j + (N-j)*c_H) / N
          pi = 1 + MPCR*N*S(group mean) - c_H
    - invader AllD: iterative Hump c_H via ``_solve_cH_with_AllD``
    - invader CC:   joint iteration via ``_solve_cH_cC_with_CC``
    """
    j_vals = np.arange(n)  # j_AllC co-players = 0, 1, ..., N-1

    # --- Resident focal AllC (j_group = j+1 always >= 1) ---
    c_H_focal_AllC = np.zeros(n)
    for idx, j in enumerate(j_vals):
        jg = j + 1
        denom = 2 * (n - 1) - jg
        if denom <= 0 or jg >= n:
            c_H_focal_AllC[idx] = 0.0  # all-AllC group, no Hump
        else:
            c_H_focal_AllC[idx] = (n - 1 - jg) / denom
    gm_focal_AllC = (j_vals + 1 + (n - 1 - j_vals) * c_H_focal_AllC) / n
    pi_k_AllC = MPCR * n * step_function(gm_focal_AllC, x0)

    # --- Resident focal Hump (j_group = j) ---
    c_H_focal_H = np.zeros(n)
    for idx, j in enumerate(j_vals):
        if j == 0:
            c_H_focal_H[idx] = min(b0, 1.0 - b0)
        else:
            c_H_focal_H[idx] = (n - 1 - j) / (2 * (n - 1) - j)
    gm_focal_H = (j_vals + (n - j_vals) * c_H_focal_H) / n
    pi_k_H = 1.0 + MPCR * n * step_function(gm_focal_H, x0) - c_H_focal_H

    def diff_fn(a):
        """pi_left - pi_right (= pi_AllC - pi_Hump) at arbitrary alpha."""
        probs = binom.pmf(j_vals, n - 1, 1.0 - a)
        return float(np.dot(probs, pi_k_AllC - pi_k_H))

    # --- Invader AllD (iterate Hump c_H per j) ---
    pi_k_AllD_inv = np.zeros(n)
    for idx, j in enumerate(j_vals):
        if n - 1 - j <= 0:  # no Hump co-players
            group_mean = (j) / n  # focal AllD (0) + j AllC (1) + 0 Hump
            pi_k_AllD_inv[idx] = 1.0 + MPCR * n * float(step_function(group_mean, x0))
        else:
            c_H_iv = _solve_cH_with_AllD(int(j), n, b0)
            group_mean = (j + (n - 1 - j) * c_H_iv) / n
            pi_k_AllD_inv[idx] = 1.0 + MPCR * n * float(step_function(group_mean, x0))

    # --- Invader CC (joint iteration per j) ---
    pi_k_CC_inv = np.zeros(n)
    for idx, j in enumerate(j_vals):
        c_H_iv, c_CC_iv = _solve_cH_cC_with_CC(int(j), n, b0)
        group_mean = (c_CC_iv + j + (n - 1 - j) * c_H_iv) / n
        pi_k_CC_inv[idx] = (1.0 + MPCR * n * float(step_function(group_mean, x0))
                            - c_CC_iv)

    # --- Binomial expectation over alpha grid (alpha = Hump proportion) ---
    rows = []
    for a in alpha_grid:
        probs = binom.pmf(j_vals, n - 1, 1.0 - a)
        pi_allc = float(np.dot(probs, pi_k_AllC))
        pi_hump = float(np.dot(probs, pi_k_H))
        pi_alld_inv = float(np.dot(probs, pi_k_AllD_inv))
        pi_cc_inv = float(np.dot(probs, pi_k_CC_inv))

        pi_left = pi_allc
        pi_right = pi_hump
        if a == 0.0:
            pi_right = float("nan")
        if a == 1.0:
            pi_left = float("nan")

        left_w = 0.0 if a == 1.0 else (1.0 - a) * pi_allc
        right_w = 0.0 if a == 0.0 else a * pi_hump
        pi_mean = left_w + right_w

        rows.append({
            "alpha": round(float(a), 8),
            "pi_left": pi_left,
            "pi_right": pi_right,
            "pi_mean": pi_mean,
            "pi_invader1": pi_alld_inv,
            "pi_invader2": pi_cc_inv,
            "is_equilibrium": False,
            "eq_stable": False,
            "alpha_equilibrium": float("nan"),
        })
    return pd.DataFrame(rows), diff_fn


# ---------------------------------------------------------------------------
# Equilibrium detection (Phase 1-5)
# ---------------------------------------------------------------------------

def annotate_equilibria(df: pd.DataFrame, diff_fn=None) -> pd.DataFrame:
    """Add ``is_equilibrium``, ``eq_stable`` and ``alpha_equilibrium`` columns.

    ``diff = pi_left - pi_right``; a sign change between consecutive alpha grid
    points flags an equilibrium at the closer endpoint (kept for plotting on the
    grid). Stability: stable iff ``diff`` goes negative -> positive. When
    ``diff_fn`` (a callable of continuous alpha) is supplied, the equilibrium
    location is refined by bisection (Brent) within the sign-change bracket and
    stored in ``alpha_equilibrium`` (otherwise the grid alpha is recorded).
    """
    df = df.copy()
    df["is_equilibrium"] = False
    df["eq_stable"] = False
    df["alpha_equilibrium"] = float("nan")

    alpha = df["alpha"].to_numpy()
    pi_left = df["pi_left"].to_numpy()
    pi_right = df["pi_right"].to_numpy()
    diff = pi_left - pi_right

    for i in range(len(diff) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if not (np.isfinite(d0) and np.isfinite(d1)):
            continue
        if d0 * d1 < 0:
            idx = i if abs(d0) < abs(d1) else i + 1
            df.loc[idx, "is_equilibrium"] = True
            df.loc[idx, "eq_stable"] = bool(d0 < 0 and d1 > 0)
            a_star = float(alpha[idx])
            if diff_fn is not None:
                try:
                    a_star = float(brentq(diff_fn, alpha[i], alpha[i + 1],
                                          xtol=1e-10, rtol=1e-12))
                except ValueError:
                    a_star = float(alpha[idx])
            df.loc[idx, "alpha_equilibrium"] = a_star
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=DEFAULT_N,
                        help=f"Group size (default: {DEFAULT_N})")
    parser.add_argument("--b0", type=float, default=DEFAULT_B0,
                        help=f"Initial cooperation belief (default: {DEFAULT_B0})")
    parser.add_argument("--x0", type=float, default=DEFAULT_X0,
                        help=f"Step threshold (default: {DEFAULT_X0})")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help=f"alpha grid points (default: {DEFAULT_RESOLUTION})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    return parser.parse_args()


def write_csv(df: pd.DataFrame, output_dir: Path, edge_name: str, n: int) -> Path:
    """Write the DataFrame to ``edge_{edge_name}_Kinf_N{n}.csv``."""
    path = output_dir / f"edge_{edge_name}_Kinf_N{n}.csv"
    df[CSV_COLUMNS].to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_grid = np.linspace(0.0, 1.0, args.resolution)

    # AllD-AllC edge
    df_allc_alld, diff_alld = compute_edge_AllD_AllC(alpha_grid, args.N, args.x0)
    df_allc_alld = annotate_equilibria(df_allc_alld, diff_alld)
    write_csv(df_allc_alld, output_dir, "AllD_AllC", args.N)

    # AllC-Hump edge
    df_allc_hs, diff_hs = compute_edge_AllC_HS(alpha_grid, args.N, args.x0, args.b0)
    df_allc_hs = annotate_equilibria(df_allc_hs, diff_hs)
    write_csv(df_allc_hs, output_dir, "AllC_HS", args.N)


if __name__ == "__main__":
    main()
