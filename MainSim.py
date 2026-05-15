"""
mainsim_v3.py
=============
Revised simulation pipeline.

Key change from v2:
    Wind is no longer loaded from an Excel file and scaled by a fixed
    WIND_FACTOR. Instead, the scenario wind comes from the wind simulation
    output (production_by_zone.csv produced by SimulateNorwayByZoneActual.py).
    The actual wind baseline is still read from the Excel file.

    delta_wind per hour = simulated_total_MWh - actual_wind_MWh
    (clamped to zero if simulated <= actual)


Key change from v1:
    Hydro that can be shifted is ONLY the volume displaced by extra wind —
    i.e. the hydro bids that sit between the baseline and scenario clearing
    price in the aggregated supply curve. All other hydro remains as-is
    inside the original curve.

    Wind injection uses empirical bin weights (wind-only coefficients from
    stacked regression of accumulated bin volume on wind, R-o-R, and reservoir
    hydro with hour-of-day and day-of-week fixed effects). These weights capture
    the observed displacement pattern across the supply curve. Only wind
    coefficients are used — hydro coefficients are excluded to avoid
    double-counting with the optimizer.

Pipeline per hour
-----------------
1.  Load baseline supply + demand curves.
2.  Clear baseline market  -> base_price.
3.  Inject extra wind using empirical bin weights -> scenario supply curve.
4.  Clear scenario market (wind only, no hydro shift yet) -> wind_price.
5.  Read displaced hydro from supply curve between wind_price and base_price.
6.  Store (timestamp, displaced_hydro_mw) for the full year.

After the hourly pass
---------------------
7.  Run HydroOptimizer on the displaced_hydro series:
        - maximize the shift of displaced hydro FROM high-wind hours
          TO low-wind hours
        - subject to a virtual reservoir (cumulative saved water) that
          cannot exceed S_MAX and must return to zero by year-end
8.  Second pass: for each hour apply the optimizer's reinjection decision
    to the scenario supply curve, then clear the final market price.
"""

from __future__ import annotations

import json
from pathlib import Path

from hydro_optimizer import optimize_hydro_shift, shift_diagnostics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# SETTINGS  (unchanged from v1)
# ==============================

DATA_DIR    = Path("nordpool_bidcurves/2024")
TZ          = "Europe/Oslo"
WIND_XLSX   = Path("Prod_data/ProductionByLocation_2024_NO_None_MWh.xlsx")  # actual wind baseline
WIND_SIM_CSV = Path("Results_windsim/out_2024_WEC/production_by_zone1.csv")               # simulated wind scenario
DEMAND_FACTOR = 1.257

# Empirical wind displacement bin weights (wind-only coefficients from
# stacked regression). Must correspond to PRICE_BINS intervals.
# Original regression weights. Bin 4 volume is placed in 0-10 EUR/MWh sub-range.
PRICE_BINS  = [-501, -498, -50, 0, 110, 4000]
BIN_WEIGHTS = [0.385, 0.055, 0.560, 0.0, 0.0]
 
 
# Reservoir cap for the *shiftable* (displaced) hydro [MWh]
# This is NOT total Norwegian reservoir — it is the buffer for
# water saved when wind displaces hydro.
# A reasonable upper bound: ~2 weeks of average hydro generation
#   32 GW * 24 h * 14 days = ~10 TWh
S_MAX = 10_000_000   # 10 TWh  (tune to your data)
 
# Per-hour reinjection cap as a multiple of mean displaced volume.
# Prevents the optimizer concentrating all water in a handful of hours.
# Higher = more concentration in expensive hours.
# Lower  = more spreading across all high-price hours.
# Tune this to balance peak price reduction vs average price reduction.
REINJECT_CAP_FACTOR = 4.0
 
 
# ==============================
# WIND LOADER  (unchanged)
# ==============================
 
def load_wind_series_from_excel(xlsx_path: Path) -> pd.Series:
    raw     = pd.read_excel(xlsx_path, header=None)
    header0 = raw.iloc[0].tolist()
    header1 = raw.iloc[1].tolist()
 
    cols = []
    for i, (h0, h1) in enumerate(zip(header0, header1)):
        cols.append(str(h0).strip() if i < 2 else str(h1).strip())
 
    df = raw.iloc[2:].copy()
    df.columns = cols
    df = df.reset_index(drop=True)
 
    df["Delivery Start (CET)"] = pd.to_datetime(
        df["Delivery Start (CET)"], dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["Delivery Start (CET)"])
 
    wind_col = "Wind Onshore (Average MWh)"
    if wind_col not in df.columns:
        raise KeyError(f"Could not find '{wind_col}' in columns: {df.columns.tolist()}")
 
    df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce").fillna(0.0)
    idx = df["Delivery Start (CET)"].dt.tz_localize(
        TZ, ambiguous="infer", nonexistent="shift_forward"
    )
    s = pd.Series(df[wind_col].astype(float).to_numpy(), index=idx, name="wind_MW")
    return s.sort_index()
 
 
def load_wind_sim_series(csv_path: Path) -> pd.Series:
    """
    Load the simulated wind production from production_by_zone.csv.
    Returns a Series of Total_MWh indexed by timezone-aware timestamps (Oslo).
    The CSV timestamp column is assumed to be naive hourly (local Oslo time).
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if "Total_MWh" not in df.columns:
        raise KeyError(f"'Total_MWh' column not found in {csv_path}. "
                       "Run SimulateNorwayByZoneActual.py first.")
    df = df.dropna(subset=["timestamp", "Total_MWh"])
    idx = df["timestamp"].dt.tz_localize(TZ, ambiguous=False, nonexistent="shift_forward")
    s = pd.Series(df["Total_MWh"].astype(float).to_numpy(), index=idx, name="wind_sim_MW")
    s = s[~s.index.duplicated(keep="first")]  # drop duplicate DST hour if present
    return s.sort_index()
 
 
# ==============================
# CURVE HELPERS  (unchanged)
# ==============================
 
def curve_to_series(curve_points: list[dict]) -> pd.Series:
    if not curve_points:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(curve_points)
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["price", "volume"])
    return df.groupby("price")["volume"].max().sort_index()
 
 
def to_incremental(cum_supply: pd.Series) -> pd.DataFrame:
    df = cum_supply.reset_index()
    df.columns = ["price", "cum_volume"]
    df = df.sort_values("price").reset_index(drop=True)
    df["increment"] = df["cum_volume"].diff().fillna(df["cum_volume"]).clip(lower=0.0)
    return df[["price", "increment"]]
 
 
def to_cumulative(increment_df: pd.DataFrame) -> pd.Series:
    df = increment_df.sort_values("price").copy()
    df["cum_volume"] = df["increment"].cumsum()
    return df.set_index("price")["cum_volume"]
 
 
def scale_cumulative_curve(cum_curve: pd.Series, factor: float) -> pd.Series:
    if factor == 1.0:
        return cum_curve
    return (cum_curve.astype(float) * float(factor)).clip(lower=0.0)
 
 
def evaluate_supply_at_prices(supply_cum: pd.Series, prices: pd.Index) -> pd.Series:
    return supply_cum.reindex(prices).sort_index().ffill().fillna(0.0)
 
 
def evaluate_demand_at_prices(demand_cum: pd.Series, prices: pd.Index) -> pd.Series:
    return demand_cum.reindex(prices).sort_index().bfill().fillna(0.0)
 
 
def clear_market_price(supply_cum: pd.Series, demand_cum: pd.Series) -> float:
    price_grid = pd.Index(sorted(set(supply_cum.index).union(set(demand_cum.index))))
    Qs = evaluate_supply_at_prices(supply_cum, price_grid)
    Qd = evaluate_demand_at_prices(demand_cum, price_grid)
    crossing = Qs >= Qd
    if not crossing.any():
        return float(price_grid[-1])
    return float(crossing.idxmax())
 
 
# ==============================
# WIND INJECTION (empirical bin weights)
# ==============================
 
def add_extra_wind_to_supply_bins(supply_cum: pd.Series, delta_wind_mw: float) -> pd.Series:
    """
    Inject delta_wind_mw into the supply curve using empirical bin weights.
 
    All bins use the original regression weights applied to the full delta_wind.
    The only modification is that all volume allocated to bin 4 (0-110 EUR/MWh)
    is placed specifically in the 0-10 EUR/MWh sub-range, rather than spread
    across the full bin. This reflects that wind bids cluster at the low end
    of the price range.
 
    Bin weights (original regression):
        [-501,-498]: 0.212  [-498,-50]: 0.025  [-50,0]: 0.370
        [0,110]:     0.393   [110,4000]:  0.0
    """
    if delta_wind_mw <= 0:
        return supply_cum
 
    inc = to_incremental(supply_cum)
    df  = inc.copy()
    df["bin"] = pd.cut(df["price"], bins=PRICE_BINS, labels=False, include_lowest=True)
 
    for i, w in enumerate(BIN_WEIGHTS):
        if w == 0:
            continue
 
        extra = delta_wind_mw * w
 
        if i == 3:
            # Bin 4: place all volume in 0-10 EUR/MWh sub-range
            mask = (df["price"] >= 0.0) & (df["price"] <= 10.0)
            if mask.sum() == 0:
                # No points in sub-range — fall back to full bin 4
                mask = df["bin"] == 3
        else:
            mask = df["bin"] == i
 
        if mask.sum() == 0:
            continue
 
        total = df.loc[mask, "increment"].sum()
        if total > 0:
            df.loc[mask, "increment"] += extra * (df.loc[mask, "increment"] / total)
        else:
            df.loc[mask, "increment"] += extra / mask.sum()
 
    df["increment"] = df["increment"].clip(lower=0.0)
    df = df.drop(columns=["bin"])
 
    return to_cumulative(df)
 
 
 
# ==============================
# NEW: EXTRACT DISPLACED HYDRO
# ==============================
 
def extract_displaced_hydro(
    supply_cum: pd.Series,
    price_high: float,
    price_low: float,
) -> float:
    """
    Return the hydro volume in the supply curve that sits between
    price_low and price_high (i.e. bids that were accepted at baseline
    price but are now below the new lower scenario price).
 
    This is the volume that wind has displaced — water that can be
    saved and shifted to another hour.
 
    Parameters
    ----------
    supply_cum  : baseline cumulative supply curve
    price_high  : baseline clearing price  (higher)
    price_low   : scenario clearing price after wind injection (lower)
 
    Returns
    -------
    displaced_mw : float >= 0
    """
    if price_low >= price_high:
        return 0.0
 
    s = supply_cum.sort_index()
 
    # Cumulative volume supplied up to each price
    vol_at_high = float(s[s.index <= price_high].iloc[-1]) if (s.index <= price_high).any() else 0.0
    vol_at_low  = float(s[s.index <= price_low ].iloc[-1]) if (s.index <= price_low ).any() else 0.0
 
    return max(vol_at_high - vol_at_low, 0.0)
 
 
# Optimizer imported from hydro_optimizer.py
 
 
# ==============================
# REINJECT HYDRO INTO SUPPLY
# ==============================
 
def reinject_hydro_to_supply(
    supply_cum: pd.Series,
    reinject_mw: float,
    bid_price: float = None,   # kept for backward compatibility but unused
    n_steps: int = 50,         # number of price points across the range
) -> pd.Series:
    if reinject_mw <= 0:
        return supply_cum

    s = supply_cum.copy().astype(float)

    # Distribute equally across 200-600 EUR/MWh
    price_points = np.linspace(500, 1500, n_steps)
    volume_per_step = reinject_mw / n_steps

    for p in price_points:
        # Add volume at this price point and shift all higher prices up
        mask = s.index >= p
        s.loc[mask] += volume_per_step

        if p not in s.index:
            vol_below = float(s[s.index < p].iloc[-1]) if (s.index < p).any() else 0.0
            s.loc[p] = vol_below + volume_per_step
            s = s.sort_index()

    return s
 
 
# ==============================
# MAIN
# ==============================
 
def main():
    # ── Load wind ──────────────────────────────────────────────────
    wind_actual = load_wind_series_from_excel(WIND_XLSX).sort_index()
    wind_sim    = load_wind_sim_series(WIND_SIM_CSV).sort_index()
 
    # ── PASS 1: collect displaced hydro for each hour ──────────────
    print("Pass 1: computing displaced hydro per hour...")
 
    pass1_records = []   # will become a lookup by timestamp
 
    for file in sorted(DATA_DIR.glob("*.json")):
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
 
        for pos in data.get("aggregatedOrderPositions", []):
            ts = pd.to_datetime(pos["deliveryStart"], utc=True).tz_convert(TZ)
 
            supply_pts = pos.get("aggregatedSupplyCurve", [])
            demand_pts = pos.get("aggregatedDemandCurve", [])
            if not supply_pts or not demand_pts:
                continue
 
            supply_cum = curve_to_series(supply_pts)
            demand_cum = curve_to_series(demand_pts)
 
            # Baseline clearing price
            base_price = clear_market_price(supply_cum, demand_cum)

            # Cleared volume at base price
            price_grid_b = pd.Index(sorted(set(supply_cum.index).union(set(demand_cum.index))))
            Qs_b = evaluate_supply_at_prices(supply_cum, price_grid_b)
            Qd_b = evaluate_demand_at_prices(demand_cum, price_grid_b)
            if base_price >= 4000:
                base_cleared_mwh = float(Qd_b[Qd_b.index >= base_price].iloc[0]) if (Qd_b.index >= base_price).any() else 0.0
            else:
                base_cleared_mwh = float(Qs_b[Qs_b.index <= base_price].iloc[-1]) if (Qs_b.index <= base_price).any() else 0.0

            # Wind lookup — actual baseline
            if ts in wind_actual.index:
                wind_base = float(wind_actual.loc[ts])
            else:
                idx_near  = wind_actual.index.get_indexer([ts], method="nearest")
                wind_base = float(wind_actual.iloc[idx_near[0]])
 
            # Simulated scenario wind
            if ts in wind_sim.index:
                wind_scenario = float(wind_sim.loc[ts])
            else:
                idx_near      = wind_sim.index.get_indexer([ts], method="nearest")
                wind_scenario = float(wind_sim.iloc[idx_near[0]])
 
            delta_wind = max(wind_scenario - wind_base, 0.0)
 
            # Inject wind using empirical bin weights -> wind-only scenario supply
            supply_wind = add_extra_wind_to_supply_bins(supply_cum, delta_wind) 
            # Scale demand
            demand_scen = scale_cumulative_curve(demand_cum, DEMAND_FACTOR)
 
            # Clearing price after wind injection (before any hydro shift)
            wind_price = clear_market_price(supply_wind, demand_scen)

            # Cleared volume at wind price
            price_grid_w = pd.Index(sorted(set(supply_wind.index).union(set(demand_scen.index))))
            Qs_w = evaluate_supply_at_prices(supply_wind, price_grid_w)
            Qd_w = evaluate_demand_at_prices(demand_scen, price_grid_w)
            if wind_price >= 4000:
                wind_cleared_mwh = float(Qd_w[Qd_w.index >= wind_price].iloc[0]) if (Qd_w.index >= wind_price).any() else 0.0
            else:
                wind_cleared_mwh = float(Qs_w[Qs_w.index <= wind_price].iloc[-1]) if (Qs_w.index <= wind_price).any() else 0.0

            # Displaced hydro: bids between wind_price and base_price
            if base_price <= 0 or base_price == wind_price:
                # Base < 0: must-run generation only, nothing to displace
                # Base == wind: no price change, nothing to displace
                displaced = 0.0
            elif wind_price < 0:
                # Base > 0, Wind < 0: only read volume between 0 and base price
                displaced = extract_displaced_hydro(supply_cum, base_price, 0.0)
            else:
                # Base > 0, Wind > 0: read volume between wind price and base price
                displaced = extract_displaced_hydro(supply_cum, base_price, wind_price)
 
            pass1_records.append({
                "ts":                ts,
                "base_price":        base_price,
                "wind_price":        wind_price,
                "displaced_mw":      displaced,
                "delta_wind":        delta_wind,
                "wind_base_mw":      wind_base,
                "wind_cleared_mwh":  wind_cleared_mwh,
                "base_cleared_mwh":  base_cleared_mwh,
            })
 
    if not pass1_records:
        print("No records produced in pass 1. Check data paths and curve keys.")
        return
 
    pass1_df = pd.DataFrame(pass1_records).sort_values("ts").reset_index(drop=True)
    print(f"  Hours processed: {len(pass1_df)}")
    print(f"  Total displaced hydro: {pass1_df['displaced_mw'].sum():.0f} MWh")
    print(f"  Mean displaced hydro:  {pass1_df['displaced_mw'].mean():.1f} MW/h")
    total_base_twh = pass1_df["base_cleared_mwh"].sum() / 1_000_000
    print(f"  Total cleared volume (base, full year):          {total_base_twh:.2f} TWh")
    total_wind_twh = pass1_df["wind_cleared_mwh"].sum() / 1_000_000
    print(f"  Total cleared volume (wind scenario, full year): {total_wind_twh:.2f} TWh")
 
    # ── HYDRO OPTIMIZER ────────────────────────────────────────────
    print("\nRunning hydro shift optimizer...")
 
    displaced_series  = pass1_df["displaced_mw"].to_numpy()
    wind_price_series = pass1_df["wind_price"].to_numpy()
 
    # Cap wind_price weights at 95th percentile and also apply a per-hour
    # reinjection cap so the optimizer spreads water across many high-price
    # hours rather than dumping everything into a handful of extreme hours.
    #
    # The per-hour cap is set to mean_displaced * REINJECT_CAP_FACTOR.
    # A factor of ~10 allows meaningful reinjection in expensive hours
    # while preventing the optimizer from concentrating 5 TWh in 7 hours.
    price_cap   = float(np.percentile(wind_price_series[wind_price_series < 4000], 95))  # cap at 95th percentile, excluding extreme outliers
    wind_price_capped = np.clip(wind_price_series, 0.0, price_cap)
 
    mean_displaced     = float(displaced_series[displaced_series > 0].mean())
    per_hour_cap       = mean_displaced * REINJECT_CAP_FACTOR
    print(f"  Wind price weight cap (95th pct, excl 4000): {price_cap:.1f} EUR/MWh")
    print(f"  Per-hour reinjection cap: {per_hour_cap:.0f} MW")
 
    reinjection = optimize_hydro_shift(
        displaced_series,
        s_max          = S_MAX,
        wind_price     = wind_price_capped,
        max_reinject_mw = per_hour_cap,
    )
 
    pass1_df["reinjection_mw"] = reinjection
    pass1_df["net_shift_mw"]   = reinjection - displaced_series  # negative = saving hour
 
    stats = shift_diagnostics(displaced_series, reinjection, S_MAX)
    for k, v in stats.items():
        print(f"  {k:<35}: {v:.1f}")
 
    # ── PASS 2: final prices with hydro reinjection ────────────────
    print("\nPass 2: computing final scenario prices with hydro reinjection...")
 
    # Build a fast lookup: ts -> reinjection_mw and wind_price (bid price)
    reinject_lookup = pass1_df.set_index("ts")[["reinjection_mw", "wind_price", "base_price", "delta_wind"]].to_dict("index")
 
    final_records = []
 
    for file in sorted(DATA_DIR.glob("*.json")):
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
 
        for pos in data.get("aggregatedOrderPositions", []):
            ts = pd.to_datetime(pos["deliveryStart"], utc=True).tz_convert(TZ)
 
            if ts not in reinject_lookup:
                continue
 
            info = reinject_lookup[ts]
 
            supply_pts = pos.get("aggregatedSupplyCurve", [])
            demand_pts = pos.get("aggregatedDemandCurve", [])
            if not supply_pts or not demand_pts:
                continue
 
            supply_cum  = curve_to_series(supply_pts)
            demand_scen = scale_cumulative_curve(curve_to_series(demand_pts), DEMAND_FACTOR)
 
            # Step 1: inject wind using empirical bin weights
            supply_wind = add_extra_wind_to_supply_bins(supply_cum, info["delta_wind"])
 
            # Step 2: reinject shifted hydro at base_price of this hour.
            # Bidding at base_price ensures the hydro clears in the market —
            # it represents what this hour's market will bear before wind/demand
            # changes. Bidding at wind_price (the scenario price of the source
            # hour) was wrong because it placed hydro above the clearing price
            # in most destination hours, so it was never dispatched.
            supply_final = reinject_hydro_to_supply(
                supply_wind,
                reinject_mw = info["reinjection_mw"],
                bid_price   = info["base_price"],
            )
 

            final_price = clear_market_price(supply_final, demand_scen)

            # Find the cleared volume at the final price
            price_grid = pd.Index(sorted(set(supply_final.index).union(set(demand_scen.index))))
            Qs = evaluate_supply_at_prices(supply_final, price_grid)
            Qd = evaluate_demand_at_prices(demand_scen, price_grid)
            if final_price >= 4000:
                cleared_volume_mwh = float(Qd[Qd.index >= final_price].iloc[0]) if (Qd.index >= final_price).any() else 0.0
            else:
                cleared_volume_mwh = float(Qs[Qs.index <= final_price].iloc[-1]) if (Qs.index <= final_price).any() else 0.0

            final_records.append({
                "time_oslo":           ts,
                "base_price":          info["base_price"],
                "wind_price":          info["wind_price"],
                "final_price":         final_price,
                "delta_wind_mw":       info["delta_wind"],
                "displaced_mw":        pass1_df.loc[pass1_df["ts"] == ts, "displaced_mw"].iloc[0],
                "reinjection_mw":      info["reinjection_mw"],
                "cleared_volume_mwh":  cleared_volume_mwh,
                "wind_cleared_mwh":    pass1_df.loc[pass1_df["ts"] == ts, "wind_cleared_mwh"].iloc[0],
                "base_cleared_mwh":    pass1_df.loc[pass1_df["ts"] == ts, "base_cleared_mwh"].iloc[0],
            })
 
    df = pd.DataFrame(final_records).sort_values("time_oslo").reset_index(drop=True)
 
    if df.empty:
        print("No final records produced.")
        return
 
    # ── SUMMARY ────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(df[["base_price", "wind_price", "final_price"]].describe().round(2))
 
    print(f"\nPrice variance (base):       {df['base_price'].var():.1f}")
    print(f"Price variance (wind only):  {df['wind_price'].var():.1f}")
    print(f"Price variance (final):      {df['final_price'].var():.1f}")
    pct = 100 * (df["wind_price"].var() - df["final_price"].var()) / (df["wind_price"].var() + 1e-9)
    print(f"Variance reduction from hydro shift: {pct:.1f}%")
    total_twh = df["cleared_volume_mwh"].sum() / 1_000_000
    print(f"\nTotal cleared volume (scenario, full year): {total_twh:.2f} TWh")
 
    df.to_csv("price_simulation_v2.csv", index=False)
    print("\nResults saved to price_simulation_v2.csv")
 
    # ── IDENTIFY HOURS TO PLOT ─────────────────────────────────────
    ts_high_wind = df.loc[df["delta_wind_mw"].idxmax(), "time_oslo"]
    ts_low_wind  = df.loc[df["delta_wind_mw"].idxmin(), "time_oslo"]
    print(f"\nPlot hours:")
    print(f"  Highest delta wind: {ts_high_wind} ({df['delta_wind_mw'].max():.0f} MW)")
    print(f"  Lowest  delta wind: {ts_low_wind}  ({df['delta_wind_mw'].min():.0f} MW)")
 
    # ── COLLECT CURVES FOR PLOT HOURS (third pass, two hours only) ─
    plot_curves = {}   # ts -> {"supply_base", "supply_scen", "demand_base", "demand_scen"}
    plot_targets = {pd.Timestamp(ts_high_wind), pd.Timestamp(ts_low_wind)}
 
    for file in sorted(DATA_DIR.glob("*.json")):
        if len(plot_curves) == 2:
            break
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for pos in data.get("aggregatedOrderPositions", []):
            ts = pd.to_datetime(pos["deliveryStart"], utc=True).tz_convert(TZ)
            if ts not in plot_targets:
                continue
            supply_pts = pos.get("aggregatedSupplyCurve", [])
            demand_pts = pos.get("aggregatedDemandCurve", [])
            if not supply_pts or not demand_pts:
                continue
 
            info         = reinject_lookup[ts]
            supply_cum   = curve_to_series(supply_pts)
            demand_cum   = curve_to_series(demand_pts)
            demand_scen  = scale_cumulative_curve(demand_cum, DEMAND_FACTOR)
            supply_wind  = add_extra_wind_to_supply_bins(supply_cum, info["delta_wind"])
            supply_final = reinject_hydro_to_supply(
                supply_wind,
                reinject_mw = info["reinjection_mw"],
                bid_price   = info["base_price"],
            )
            plot_curves[ts] = {
                "supply_base":  supply_cum,
                "supply_scen":  supply_final,
                "demand_base":  demand_cum,
                "demand_scen":  demand_scen,
            }
            if len(plot_curves) == 2:
                break
 
    # ── PLOTS ──────────────────────────────────────────────────────
    price_bins_plot = PRICE_BINS  # reuse the existing bin boundaries
 
    def plot_bid_curves(ax, curves, ts, row_df):
        """Plot base and scenario supply + demand curves for one hour."""
        sb = curves["supply_base"].sort_index()
        ss = curves["supply_scen"].sort_index()
        db = curves["demand_base"].sort_index()
        ds = curves["demand_scen"].sort_index()
 
        # Supply: price on y-axis, volume on x-axis
        ax.step(sb.values / 1000, sb.index, where="post",
                color="steelblue", lw=1.5, label="Supply base")
        ax.step(ss.values / 1000, ss.index, where="post",
                color="steelblue", lw=1.5, ls="--", label="Supply scenario")
 
        # Demand: descending — plot reversed
        db_s = db.sort_index(ascending=False)
        ds_s = ds.sort_index(ascending=False)
        ax.step(db_s.values / 1000, db_s.index, where="post",
                color="darkorange", lw=1.5, label="Demand base")
        ax.step(ds_s.values / 1000, ds_s.index, where="post",
                color="darkorange", lw=1.5, ls="--", label="Demand scenario")
 
        # Clearing prices
        r = row_df.iloc[0]
        ax.axhline(r["base_price"],  color="steelblue",  lw=1, ls=":", alpha=0.8,
                   label=f"Base price: {r['base_price']:.1f} €/MWh")
        ax.axhline(r["final_price"], color="darkorange", lw=1, ls=":", alpha=0.8,
                   label=f"Scenario price: {r['final_price']:.1f} €/MWh")
 
        ax.set_xlabel("Volume (GW)")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.set_title(f"{pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M')}  |  "
                     f"Δwind = {r['delta_wind_mw']:.0f} MW")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
 
    def plot_hours_per_price_bin(ax, df_results, bins, labels):
        """Bar chart: number of hours where final_price falls in each bin."""
        counts_base  = pd.cut(df_results["base_price"],  bins=bins, labels=labels).value_counts().reindex(labels)
        counts_scen  = pd.cut(df_results["final_price"], bins=bins, labels=labels).value_counts().reindex(labels)
 
        x     = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, counts_base.values,  width, label="Base",     color="steelblue",  alpha=0.8)
        ax.bar(x + width/2, counts_scen.values,  width, label="Scenario", color="darkorange", alpha=0.8)
 
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Number of hours")
        ax.set_title("Hours per Price Bin — Base vs Scenario")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
 
        # Annotate with counts
        for rect_b, rect_s, vb, vs in zip(
            ax.patches[:len(labels)], ax.patches[len(labels):],
            counts_base.values, counts_scen.values
        ):
            ax.text(rect_b.get_x() + rect_b.get_width()/2, rect_b.get_height() + 10,
                    str(int(vb)), ha="center", va="bottom", fontsize=7)
            ax.text(rect_s.get_x() + rect_s.get_width()/2, rect_s.get_height() + 10,
                    str(int(vs)), ha="center", va="bottom", fontsize=7)
 
    # Price bin labels for the bar chart
    bin_boundaries = [-501, 0, 50, 75, 100, 200, 600, 4000]
    bin_labels     = ["<0", "0-50", "50-75", "75-100", "100-200", "200-600", ">600"]
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
 
    # Plot 1: highest delta wind hour
    if pd.Timestamp(ts_high_wind) in plot_curves:
        plot_bid_curves(
            axes[0], plot_curves[pd.Timestamp(ts_high_wind)],
            ts_high_wind,
            df[df["time_oslo"] == ts_high_wind]
        )
    else:
        axes[0].set_title("High wind hour — curves not found")
 
    # Plot 2: lowest delta wind hour
    if pd.Timestamp(ts_low_wind) in plot_curves:
        plot_bid_curves(
            axes[1], plot_curves[pd.Timestamp(ts_low_wind)],
            ts_low_wind,
            df[df["time_oslo"] == ts_low_wind]
        )
    else:
        axes[1].set_title("Low wind hour — curves not found")
 
    # Plot 3: hours per price bin
    plot_hours_per_price_bin(axes[2], df, bin_boundaries, bin_labels)
 
    plt.suptitle("Norwegian Electricity Market — Wind & Demand Scenario Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig("price_simulation_v2.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to price_simulation_v2.png")
 
 
if __name__ == "__main__":
    main()