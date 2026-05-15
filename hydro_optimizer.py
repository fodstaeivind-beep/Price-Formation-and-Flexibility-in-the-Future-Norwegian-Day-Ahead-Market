"""
hydro_optimizer.py
==================
Optimal redistribution of wind-displaced hydro across hours.

Scope
-----
This optimizer does NOT model the full Norwegian hydro system.
It only redistributes the hydro volume that was displaced by extra
wind — i.e. water that was saved in high-wind hours and can be
shifted to low-wind hours.

All other hydro remains inside the aggregated supply curve unchanged.

The only inputs needed are:
    - displaced_mw : hourly series of displaced hydro from Pass 1
    - s_max        : maximum virtual reservoir buffer (MWh)

The only output is:
    - reinjection_mw : optimal hourly reinjection schedule

Optimization problem
--------------------
Maximize sum(wind_price * reinjection) across all hours,
subject to:
    - reinjection(t) >= 0                              (can only add, not remove)
    - reinjection(t) <= max_reinject_mw                (per-hour cap)
    - virtual_reservoir(t) in [-s_max, s_max]          (allow pre-injection up to -s_max)
    - reservoir(T) in [-0.05*s_max, 0.05*s_max]        (close to zero at year-end)
    - sum(reinjection) in [0.95, 1.05] * sum(displaced) (balance: reinject ~= saved)

STANDALONE USAGE
----------------
    python hydro_optimizer.py

Runs a synthetic demo using a simple high/low wind pattern.

INTEGRATION USAGE
-----------------
    from hydro_optimizer import optimize_hydro_shift

    reinjection_mw = optimize_hydro_shift(
        displaced_mw = pass1_df["displaced_mw"].to_numpy(),
        s_max        = 10_000_000,   # 10 TWh buffer — tune to your data
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# CORE OPTIMIZER
# ──────────────────────────────────────────────

def optimize_hydro_shift(
    displaced_mw: np.ndarray,
    s_max: float,
    wind_price: np.ndarray | None = None,
    max_reinject_mw: float | None = None,
) -> np.ndarray:
    """
    Compute the optimal reinjection schedule for wind-displaced hydro.

    Objective
    ---------
    Maximize sum(wind_price * reinjection) — reinject water preferentially
    in high-price hours to reduce both peak prices and price variance.

    The per-hour cap (max_reinject_mw) prevents the optimizer from
    concentrating all water in a handful of extreme-price hours,
    forcing it to spread reinjection across many expensive hours instead.

    Constraints
    -----------
    - r(t) <= max_reinject_mw           per-hour reinjection cap
    - reservoir(t) >= -s_max            allow pre-injection (borrow up to s_max)
    - reservoir(t) <= s_max             physical storage buffer cap
    - reservoir(T) in [-0.05, 0.05]*s_max   year-end close-out (near zero)
    - sum(r) in [0.95, 1.05]*total_d    balance: total reinjected ~ total saved

    The reservoir can go negative, meaning hydro is reinjected before the
    equivalent saving has occurred. This is physically valid if you assume
    the operator knows future wind and can pre-release water to be saved
    later. The [-s_max, s_max] bounds prevent unlimited borrowing.

    Parameters
    ----------
    displaced_mw : array of shape (T,)
        Hourly displaced hydro from Pass 1 [MW].
    s_max : float
        Maximum virtual reservoir buffer [MWh].
    wind_price : array of shape (T,), optional
        Hourly wind-only scenario prices. Used as weights to concentrate
        reinjection in high-price hours. Should be capped before passing
        to avoid extreme outliers dominating.
    max_reinject_mw : float, optional
        Maximum reinjection allowed in any single hour [MW].
        If None, no per-hour cap is applied.

    Returns
    -------
    reinjection_mw : array of shape (T,), >= 0
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix, vstack as sp_vstack, csc_matrix

    d = np.asarray(displaced_mw, dtype=float).clip(min=0.0)
    T = len(d)
    total_d = float(d.sum())

    if total_d == 0.0:
        return np.zeros(T)

    cum_d = np.cumsum(d)

    # Objective: maximize sum(wind_price * r) = minimize -sum(wind_price * r)
    if wind_price is not None:
        p = np.asarray(wind_price, dtype=float)
    else:
        p = np.ones(T)
    c = -p

    # ── Variable bounds ──────────────────────────────────────────────
    # r(t) in [0, max_reinject_mw] if cap set, else [0, inf)
    ub = max_reinject_mw if max_reinject_mw is not None else None
    bounds = [(0, ub)] * T

    # ── Sparse constraint matrix ─────────────────────────────────────
    # Reservoir >= -s_max: cumsum(r)[t] <= cum_d[t] + s_max
    A_lo = lil_matrix((T, T))
    for t in range(T):
        A_lo[t, 0:t+1] = 1.0
    b_lo = cum_d + s_max

    # Reservoir <= s_max: -cumsum(r)[t] <= s_max - cum_d[t]
    A_hi = lil_matrix((T, T))
    for t in range(T):
        A_hi[t, 0:t+1] = -1.0
    b_hi = s_max - cum_d

    # End-of-year lower bound: -sum(r) <= 0.05*s_max - total_d
    #   i.e. reservoir(T) = total_d - sum(r) >= -0.05*s_max
    A_eoy_lo = lil_matrix((1, T))
    A_eoy_lo[0, 0:T] = -1.0
    b_eoy_lo = np.array([0.0 * s_max - total_d])

    # End-of-year upper bound: sum(r) <= total_d + 0.05*s_max
    #   i.e. reservoir(T) = total_d - sum(r) >= -0.05*s_max
    A_eoy_hi = lil_matrix((1, T))
    A_eoy_hi[0, 0:T] = 1.0
    b_eoy_hi = np.array([total_d + 0.0 * s_max])

    # Min use: -sum(r) <= -0.95*total_d  (reinject at least 95% of saved)
    A_min = lil_matrix((1, T))
    A_min[0, 0:T] = -1.0
    b_min = np.array([-0.95 * total_d])

    # Max use: sum(r) <= 1.05*total_d  (don't reinject more than 105% of saved)
    A_max = lil_matrix((1, T))
    A_max[0, 0:T] = 1.0
    b_max = np.array([1.05 * total_d])

    A_ub = csc_matrix(sp_vstack([A_lo, A_hi, A_eoy_lo, A_eoy_hi, A_min, A_max]))
    b_ub = np.concatenate([b_lo, b_hi, b_eoy_lo, b_eoy_hi, b_min, b_max])

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.status != 0:
        print(f"  Warning: optimizer did not converge ({result.message}). Returning uniform reinjection.")
        return np.full(T, total_d / T)

    return np.clip(result.x, 0.0, None)


# ──────────────────────────────────────────────
# DIAGNOSTICS
# ──────────────────────────────────────────────

def shift_diagnostics(displaced_mw: np.ndarray, reinjection_mw: np.ndarray, s_max: float) -> dict:
    """
    Compute summary statistics for the hydro shift.

    Returns a dict with key metrics useful for reporting.
    """
    d = np.asarray(displaced_mw, dtype=float).clip(min=0.0)
    r = np.asarray(reinjection_mw, dtype=float).clip(min=0.0)

    reservoir = np.cumsum(d - r)

    return {
        "total_displaced_mwh":       float(d.sum()),
        "total_reinjected_mwh":      float(r.sum()),
        "reinjection_rate_pct":      float(100 * r.sum() / (d.sum() + 1e-9)),
        "reservoir_max_mwh":         float(reservoir.max()),
        "reservoir_utilisation_pct": float(100 * reservoir.max() / (s_max + 1e-9)),
        "hours_with_displacement":   int((d > 0).sum()),
        "hours_with_reinjection":    int((r > 0).sum()),
        "mean_displaced_mw":         float(d[d > 0].mean()) if (d > 0).any() else 0.0,
        "mean_reinjection_mw":       float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        "variance_before":           float(np.var(d)),
        "variance_after":            float(np.var(d - r)),
        "variance_reduction_pct":    float(100 * (np.var(d) - np.var(d - r)) / (np.var(d) + 1e-9)),
    }


def plot_shift(displaced_mw: np.ndarray, reinjection_mw: np.ndarray, s_max: float):
    """Plot displaced vs reinjected hydro and virtual reservoir level."""
    d         = np.asarray(displaced_mw).clip(min=0.0)
    r         = np.asarray(reinjection_mw).clip(min=0.0)
    reservoir = np.cumsum(d - r)
    hours     = np.arange(len(d))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].fill_between(hours, d, alpha=0.4, color="steelblue",  label="Displaced (saved)")
    axes[0].fill_between(hours, r, alpha=0.4, color="darkorange", label="Reinjected (shifted)")
    axes[0].set_ylabel("MW")
    axes[0].set_title("Displaced vs Reinjected Hydro")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(hours, reservoir / 1e6, alpha=0.4, color="royalblue")
    axes[1].plot(hours, reservoir / 1e6, color="royalblue", lw=0.8)
    axes[1].axhline(s_max / 1e6, color="red", lw=1, ls="--", alpha=0.7,
                    label=f"S_MAX ({s_max/1e6:.0f} TWh)")
    axes[1].set_ylabel("TWh")
    axes[1].set_xlabel("Hour of year")
    axes[1].set_title("Virtual Reservoir Level (cumulative saved − reinjected)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hydro_shift_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to hydro_shift_diagnostics.png")


# ──────────────────────────────────────────────
# STANDALONE DEMO
# ──────────────────────────────────────────────

def make_synthetic_displaced(seed: int = 42) -> np.ndarray:
    """
    Synthetic displaced hydro series for testing.
    Non-zero only when wind is high (roughly 30% of hours).
    """
    rng = np.random.default_rng(seed)
    T   = 8760
    wind_high = rng.random(T) < 0.30
    displaced = np.where(wind_high, rng.uniform(100, 800, T), 0.0)
    return displaced


def main():
    print("=" * 55)
    print("HYDRO SHIFT OPTIMIZER — Standalone Demo")
    print("=" * 55)

    displaced = make_synthetic_displaced()
    s_max     = 10_000_000   # 10 TWh buffer

    print(f"\nInput displaced hydro:")
    print(f"  Hours with displacement : {(displaced > 0).sum()}")
    print(f"  Total displaced         : {displaced.sum():.0f} MWh")
    print(f"  Mean (when > 0)         : {displaced[displaced > 0].mean():.1f} MW")

    reinjection = optimize_hydro_shift(displaced, s_max=s_max)
    stats       = shift_diagnostics(displaced, reinjection, s_max)

    print(f"\nOptimizer results:")
    for k, v in stats.items():
        print(f"  {k:<35}: {v:.1f}")

    plot_shift(displaced, reinjection, s_max)


if __name__ == "__main__":
    main()