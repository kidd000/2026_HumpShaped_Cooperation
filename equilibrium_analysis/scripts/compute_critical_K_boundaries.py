#!/usr/bin/env python3
"""
Critical-K boundaries K*(N) for the AllC-exploiter edges (Figure 4C).

Computes, for each group size N, the critical production steepness K*(N) above
which a polymorphic equilibrium containing AllC exists, for the two edges:

  - AllC-AllD : the smallest K (bisection over [1, 500]) admitting an interior
    AllC-AllD equilibrium; existence is gated by the K -> infinity pivotal factor
    MPCR * N * C(N-1, floor(N/2)) * 2^{-(N-1)} > 1.
  - AllC-Hump : the b0-marginalized condition (b0 averaged over Uniform[0,1]).

Both edges use the normalized logistic production (SI Eq. S2; S(0)=0, S(1)=1).
The script is self-contained (numpy + scipy only) and does not need the
convergence database. It reproduces ``analytical_boundaries.csv``, the input to
the Figure 4C panel.

Usage:
    uv run python scripts/compute_critical_K_boundaries.py
    uv run python scripts/compute_critical_K_boundaries.py --n-max 50 -o boundaries.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import binom

MPCR = 0.4
N_ALPHA = 201  # resolution for the delta(alpha) scan


# --- Production function -----------------------------------------------------

def _sigmoid(c, k, x0=0.5):
    # Normalized logistic (SI Eq. S2): S(0)=0, S(1)=1, with a linear fallback
    # when the raw sigmoid is nearly flat (small K).
    c = np.asarray(c, dtype=float)
    s_c = 1.0 / (1.0 + np.exp(-k * (c - x0)))
    s_0 = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    s_1 = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
    denom = s_1 - s_0
    return np.where(denom < 1e-10, c, (s_c - s_0) / denom)


def _step(c, x0=0.5):
    """Step-function limit of the sigmoid at K -> infinity."""
    c = np.asarray(c, dtype=float)
    return np.where(c > x0, 1.0,
                    np.where(np.isclose(c, x0, atol=1e-12), 0.5, 0.0))


def _prod(c, k, x0=0.5):
    return _step(c, x0) if np.isinf(k) else _sigmoid(c, k, x0)


# --- AllC-AllD edge ----------------------------------------------------------

def _pivotal_factor_alld(n):
    """MPCR * N * C(N-1, floor(N/2)) * 2^{-(N-1)} (the K->inf pivotal term)."""
    return MPCR * n * comb(n - 1, n // 2, exact=True) * (0.5 ** (n - 1))


def _min_delta_alld(k, n, mpcr=MPCR, x0=0.5):
    """min_alpha delta(alpha) along the AllC-AllD edge."""
    alpha = np.linspace(0.01, 0.99, N_ALPHA)
    kv = np.arange(n)
    s_c = _prod((n - kv) / n, k, x0)
    s_d = _prod((n - 1 - kv) / n, k, x0)
    dk = 1.0 + mpcr * n * (s_d - s_c)
    da = np.array([np.dot(binom.pmf(kv, n - 1, a), dk) for a in alpha])
    return float(da.min())


def critical_K_alld(n, k_lo=1.0, k_hi=500.0, precision=0.001, mpcr=MPCR):
    """Binary search for K*(N) on the AllC-AllD edge; nan if no equilibrium."""
    if _min_delta_alld(k_hi, n, mpcr) >= 0:
        return np.nan
    if _min_delta_alld(k_lo, n, mpcr) < 0:
        return k_lo
    lo, hi = k_lo, k_hi
    while hi - lo > precision:
        mid = (lo + hi) / 2.0
        if _min_delta_alld(mid, n, mpcr) < 0:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2.0, 3)


# --- AllC-Hump edge (b0-marginalized) ----------------------------------------

def _cbar_hump(h, N):
    """Group mean cooperation with h Hump and N-h AllC players (h < N)."""
    return (N**2 - 2*N + h) / (N * (N - 2 + h))


def _cH_hump(h, N):
    """Hump convergence cooperation rate (h Hump players, h < N)."""
    if h <= 0:
        return 0.0
    return (h - 1) / (N - 2 + h)


def _S_allH_marginalized(K, x0=0.5, n_b0=2001):
    """<S(min(b0, 1-b0); K)> averaged over b0 ~ Uniform[0, 1].

    Numerical marginalization using the normalized production _sigmoid (SI Eq. S2).
    """
    if np.isinf(K):
        return 0.0
    b0 = np.linspace(0.0, 1.0, n_b0)
    c = np.minimum(b0, 1.0 - b0)
    return float(np.mean(_sigmoid(c, K, x0)))


def _min_delta_hump_marg(K, N, mpcr=MPCR, x0=0.5):
    """min_alpha delta(alpha) along the AllC-Hump edge (b0-marginalized)."""
    dk = np.zeros(N)
    for k in range(N):
        if k <= N - 2:
            cbar_B = _cbar_hump(k, N)
            cbar_A = _cbar_hump(k + 1, N)
            c_H_focal = _cH_hump(k + 1, N)
            delta_cost = 1.0 - c_H_focal
            dk[k] = delta_cost + mpcr * N * (_prod(cbar_A, K, x0) - _prod(cbar_B, K, x0))
        else:
            cbar_fc = _cbar_hump(N - 1, N)
            dk[k] = 0.75 + mpcr * N * (_S_allH_marginalized(K, x0) - _prod(cbar_fc, K, x0))
    alpha = np.linspace(0.01, 0.99, N_ALPHA)
    da = np.array([np.dot(binom.pmf(np.arange(N), N - 1, a), dk) for a in alpha])
    return float(da.min())


def critical_K_hump_marg(N, k_lo=1.0, k_hi=500.0, precision=0.001, mpcr=MPCR, x0=0.5):
    """Binary search for K*(N) on the AllC-Hump edge (b0-marginalized)."""
    if _min_delta_hump_marg(k_hi, N, mpcr, x0) >= 0:
        return np.nan
    if _min_delta_hump_marg(k_lo, N, mpcr, x0) < 0:
        return k_lo
    lo, hi = k_lo, k_hi
    while hi - lo > precision:
        mid = (lo + hi) / 2.0
        if _min_delta_hump_marg(mid, N, mpcr, x0) < 0:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2.0, 3)


# --- Assembly ----------------------------------------------------------------

def compute_boundaries(n_max=50):
    """Return the boundary table with columns N, K_critical, edge_type."""
    rows = []
    # AllC-AllD: K* exists only where the pivotal factor exceeds 1; reported
    # from N = 11 (small N have no AllC-AllD equilibrium).
    for n in range(3, n_max + 1):
        k_star = critical_K_alld(n) if _pivotal_factor_alld(n) > 1.0 else np.nan
        if n >= 11 and not np.isnan(k_star):
            rows.append({"N": n, "K_critical": float(k_star), "edge_type": "AllC-AllD"})
    # AllC-Hump (b0-marginalized): defined for all N; clamp nan to K_LO = 1
    # (an equilibrium exists for any K there).
    for n in range(3, n_max + 1):
        k_star = critical_K_hump_marg(n)
        if np.isnan(k_star):
            k_star = 1.0
        rows.append({"N": n, "K_critical": float(k_star), "edge_type": "AllC-Hump"})
    return pd.DataFrame(rows, columns=["N", "K_critical", "edge_type"])


def main():
    parser = argparse.ArgumentParser(
        description="Compute analytical critical-K boundaries K*(N) for Figure 4C."
    )
    parser.add_argument("--n-max", type=int, default=50, help="Maximum group size N (default: 50)")
    parser.add_argument(
        "-o", "--output", type=str,
        default="output/analytical_boundaries.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = compute_boundaries(n_max=args.n_max)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
