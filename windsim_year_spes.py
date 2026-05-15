#!/usr/bin/env python3
"""
SimulateNorwayByZoneActual.py

Multi-zone wind production for Norway (or any zonal setup).

- Most zones: uses ACTUAL data for the requested --year (no sampling).
- Bootstrap zones (--bootstrap-zone): uses moving-block bootstrap drawn from
  the full historical pool (all years in the CSV), matching the original
  SimulateNorwayByZone.py approach. Useful when a zone's data does not cover
  the requested year (e.g. OS2/OU2 only available 1961-2021).

What it does
------------
- Reads one wind *profile* CSV per zone (columns: Timestamp, Profile)
- For normal zones: filters data to the requested --year
- For bootstrap zones: builds seasonal pools from ALL available data and draws
  variable-length blocks (exponential distribution, mean 72h winter/autumn,
  48h spring/summer) to fill the full year
- Multiplies the capacity factor by the capacity (MW) you set for each zone
- Writes per-zone CSVs and a combined CSV with all zones + a Total column

Usage example
-------------
python SimulateNorwayByZoneActual.py \\
  --zone-file NO1=winddata/NO1_wind_onshore_stock.csv \\
  --zone-file NO2=winddata/NO2_wind_onshore_stock.csv \\
  --zone-file NO3=winddata/NO3_wind_onshore_stock.csv \\
  --zone-file NO4=winddata/NO4_wind_onshore_stock.csv \\
  --zone-file NO5=winddata/NO5_wind_onshore_stock.csv \\
  --zone-file OS2=winddata/NO2_wind_offshore_UtsiraNord.csv \\
  --zone-file OU2=winddata/NO2_wind_offshore_SorligeNordsjo2.csv \\
  --capacity NO1=385 --capacity NO2=1431 --capacity NO3=2108 --capacity NO4=1159 --capacity NO5=0 \\
  capacity OS2=1500 --capacity OU2=1000 year 2030 --bootstrap-zone OS2 --bootstrap-zone OU2 --seed 42 --out-dir out_2030

# python windsim_year_spes.py --zone-file NO1=winddata/NO1_wind_onshore_stock.csv --zone-file NO2=winddata/NO2_wind_onshore_stock.csv  --zone-file NO3=winddata/NO3_wind_onshore_stock.csv --zone-file NO4=winddata/NO4_wind_onshore_stock.csv  --zone-file NO5=winddata/NO5_wind_onshore_stock.csv --zone-file OS2=winddata/NO2_wind_offshore_UtsiraNord.csv --zone-file OU2=winddata/NO2_wind_offshore_SorligeNordsjo2.csv --capacity NO1=500 --capacity NO2=2000 --capacity NO3=2500 --capacity NO4=4500 --capacity NO5=500   --capacity OS2=2500 --capacity OU2=2500    --year 2024 --bootstrap-zone OS2 --bootstrap-zone OU2 --out-dir out_2024_WEC  
Output
------
- out_2030/production_NO1.csv  ... per-zone hourly capacity_factor and production_MWh
- out_2030/production_by_zone.csv ... wide table with one production_MWh_<ZONE> per zone + Total_MWh

Notes
-----
- Missing hours for actual-data zones are filled with NaN (no imputation).
- Bootstrap zones sample from their full historical record regardless of year.
- --seed ensures bootstrap zones are reproducible across runs.
"""

import argparse
import hashlib
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

SEASONS = ["winter", "spring", "Summer", "Autumn"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Autumn"


def load_hourly(csv_path: str, clip: bool = True) -> pd.Series:
    """Load CSV with Timestamp/Profile columns; return hourly Series."""
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns or "Profile" not in df.columns:
        raise ValueError(f"{csv_path}: CSV must contain 'Timestamp' and 'Profile' columns.")
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    prof = pd.to_numeric(df["Profile"], errors="coerce")
    df = (pd.DataFrame({"timestamp": ts, "profile": prof})
            .dropna(subset=["timestamp"])
            .sort_values("timestamp"))
    df = df.groupby("timestamp", as_index=False)["profile"].mean()
    s = df.set_index("timestamp")["profile"].sort_index().asfreq("H")
    if clip:
        s = s.clip(lower=0, upper=1)
    return s


def parse_kv_list(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"Expected KEY=VALUE, got '{it}'.")
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            raise ValueError(f"Malformed KEY=VALUE: '{it}'.")
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Bootstrap helpers (from original SimulateNorwayByZone.py)
# ---------------------------------------------------------------------------

def build_seasonal_pools(series: pd.Series) -> Dict[str, np.ndarray]:
    """Build per-season arrays of valid values from the full historical series."""
    month_map = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11],
    }
    months = series.index.month
    return {
        s: series[months.isin(m)].dropna().to_numpy()
        for s, m in month_map.items()
    }


def get_mean_block_len(season: str) -> float:
    return 72.0 if season in ("winter", "Autumn") else 48.0


def stable_zone_seed(global_seed: int | None, zone: str) -> int | None:
    if global_seed is None:
        return None
    h = int(hashlib.sha256((str(global_seed) + "|" + zone).encode()).hexdigest(), 16)
    return h % (2**32 - 1)


def draw_profile_from_pool(pool: np.ndarray, hours: int, mean_block_len: float,
                           seed: int | None) -> np.ndarray:
    """Draw variable-length blocks (exponential distribution) to fill `hours`."""
    if pool.size == 0:
        raise ValueError("Empty pool: no valid data for this season.")
    rng = np.random.default_rng(seed)
    cf: List[float] = []
    while len(cf) < hours:
        block_len = max(1, int(rng.exponential(mean_block_len)))
        block_len = min(block_len, pool.size)
        start = rng.integers(0, pool.size - block_len + 1)
        cf.extend(pool[start: start + block_len])
    return np.array(cf[:hours])


def simulate_bootstrap_year(series: pd.Series, year: int, seed: int | None) -> pd.Series:
    """Generate a full year via season-aware moving-block bootstrap."""
    pools = build_seasonal_pools(series)
    year_idx = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="H")

    # Split year into contiguous season segments
    s_list = [season_from_month(ts.month) for ts in year_idx]
    segments: List[tuple] = []
    seg_start = 0
    for i in range(1, len(year_idx) + 1):
        if i == len(year_idx) or s_list[i] != s_list[i - 1]:
            segments.append((seg_start, i, s_list[i - 1]))
            seg_start = i

    draws: List[np.ndarray] = []
    for seg_num, (a, b, seg_season) in enumerate(segments):
        pool = pools.get(seg_season, np.empty(0, dtype=float))
        if pool.size == 0:
            raise SystemExit(
                f"Bootstrap: no valid data for season '{seg_season}'. Check data coverage."
            )
        seg_seed = None if seed is None else (seed + seg_num) % (2**32 - 1)
        cf_seg = draw_profile_from_pool(pool, hours=b - a,
                                        mean_block_len=get_mean_block_len(seg_season),
                                        seed=seg_seed)
        draws.append(cf_seg)

    return pd.Series(np.concatenate(draws), index=year_idx)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_daily_production(combined: pd.DataFrame, out_dir: str, year: int) -> None:
    """
    Plot daily total wind production with a rolling min–max shaded band,
    matching the style of the reference figure (Norwegian labels).
    """
    daily = combined["Total_MWh"].resample("D").sum()

    # Rolling 30-day min/max band for the grey envelope
    window = 30
    roll_min = daily.rolling(window, center=True, min_periods=1).min()
    roll_max = daily.rolling(window, center=True, min_periods=1).max()

    fig, ax = plt.subplots(figsize=(16, 5))

    # Shaded band
    ax.fill_between(daily.index, roll_min, roll_max,
                    color="grey", alpha=0.45, linewidth=0)

    # Daily production line
    ax.plot(daily.index, daily.values, color="black", linewidth=0.9,
            label="Norge totalt (daglig sum)")

    # Axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(daily.index[0], daily.index[-1])
    ax.set_ylim(bottom=0)

    ax.set_title("Wind power generation per day – Norge totalt", fontsize=13)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Daily generation [MWh]", fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(axis="x", color="white", linewidth=0.8)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"daily_production_{year}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


def plot_hourly_february(combined: pd.DataFrame, out_dir: str, year: int) -> None:
    """
    Plot hourly total wind production for the first two weeks of February.
    """
    feb_start = pd.Timestamp(f"{year}-02-01 00:00")
    feb_end   = pd.Timestamp(f"{year}-02-14 23:00")
    hourly = combined["Total_MWh"].loc[feb_start:feb_end]

    if hourly.empty:
        print(f"WARNING: No hourly data found for February {year}, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.fill_between(hourly.index, 0, hourly.values,
                    color="steelblue", alpha=0.35, linewidth=0)
    ax.plot(hourly.index, hourly.values, color="steelblue", linewidth=0.9,
            label="Norge totalt (timeproduksjon)")

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlim(hourly.index[0], hourly.index[-1])
    ax.set_ylim(bottom=0)

    ax.set_title(f"Wind power generation per hour – first two weeks of February {year}",
                 fontsize=13)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Hourly generation [MWh]", fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(axis="x", color="lightgrey", linewidth=0.6, linestyle="--")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"hourly_february_{year}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Multi-zone wind production: actual data for most zones, "
            "moving-block bootstrap for specified zones (e.g. OS2, OU2)."
        )
    )
    ap.add_argument("--zone-file", action="append", required=True,
                    help="Repeat as ZONE=path/to/profile.csv for each zone")
    ap.add_argument("--capacity", action="append", required=True,
                    help="Repeat as ZONE=MW for each zone (must match --zone-file zones)")
    ap.add_argument("--year", type=int, required=True,
                    help="Calendar year to simulate (e.g. 2030)")
    ap.add_argument("--bootstrap-zone", action="append", default=[],
                    help="Zone(s) to simulate via bootstrap instead of actual data "
                         "(repeat for each, e.g. --bootstrap-zone OS2 --bootstrap-zone OU2)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Global random seed for bootstrap zones (ensures reproducibility)")
    ap.add_argument("--clip", dest="clip", action="store_true",
                    help="Clip Profile to [0,1] (default)")
    ap.add_argument("--no-clip", dest="clip", action="store_false",
                    help="Do not clip Profile")
    ap.add_argument("--out-dir", type=str, default="out_wind_actual",
                    help="Directory to write outputs")
    ap.add_argument("--loss-factor", type=float, default=1.15,
                    help="Divide production (MWh) by this factor for losses (default 1.15)")
    ap.set_defaults(clip=True)
    args = ap.parse_args()

    zone_to_file = parse_kv_list(args.zone_file)
    zone_to_cap_str = parse_kv_list(args.capacity)
    bootstrap_zones = set(args.bootstrap_zone)

    # Validate zone consistency
    zones_files = set(zone_to_file.keys())
    zones_caps = set(zone_to_cap_str.keys())
    if zones_files != zones_caps:
        msg = []
        if zones_caps - zones_files:
            msg.append(f"Missing --zone-file for: {sorted(zones_caps - zones_files)}")
        if zones_files - zones_caps:
            msg.append(f"Missing --capacity for: {sorted(zones_files - zones_caps)}")
        raise SystemExit("Zone mismatch. " + "; ".join(msg))

    unknown = bootstrap_zones - zones_files
    if unknown:
        raise SystemExit(f"--bootstrap-zone refers to unknown zones: {sorted(unknown)}")

    zone_to_cap: Dict[str, float] = {z: float(v) for z, v in zone_to_cap_str.items()}
    os.makedirs(args.out_dir, exist_ok=True)

    year_idx = pd.date_range(f"{args.year}-01-01 00:00", f"{args.year}-12-31 23:00", freq="H")
    print(f"Year: {args.year}  ({len(year_idx)} hours)")
    print(f"Actual-data zones : {sorted(zones_files - bootstrap_zones)}")
    print(f"Bootstrap zones   : {sorted(bootstrap_zones)}")
    print(f"Loss factor       : {args.loss_factor}")
    if bootstrap_zones:
        print("Bootstrap method  : moving-block (mean 72h winter/autumn, 48h spring/summer)")

    per_zone_series: Dict[str, pd.Series] = {}

    for zone, csv_path in zone_to_file.items():
        full_series = load_hourly(csv_path, clip=args.clip)

        if zone in bootstrap_zones:
            z_seed = stable_zone_seed(args.seed, zone)
            cf_year = simulate_bootstrap_year(full_series, args.year, seed=z_seed)
            print(f"{zone}: bootstrap | historical pool: {full_series.dropna().shape[0]} hours")
        else:
            cf_year = full_series[full_series.index.year == args.year]
            if cf_year.empty:
                raise SystemExit(
                    f"{zone}: No data found for year {args.year} in '{csv_path}'. "
                    "Check your CSV covers this year, or add zone to --bootstrap-zone."
                )
            cf_year = cf_year.reindex(year_idx)
            nan_count = int(cf_year.isna().sum())
            if nan_count:
                print(f"  WARNING — {zone}: {nan_count} missing hours filled with NaN")

        cf_series = cf_year.rename(f"capacity_factor_{zone}")
        prod_series = (cf_series * zone_to_cap[zone] / args.loss_factor).rename(f"production_MWh_{zone}")

        out_df = pd.concat(
            [cf_series.rename("capacity_factor"), prod_series.rename("production_MWh")], axis=1
        )
        out_df.index.name = "timestamp"
        out_path = os.path.join(args.out_dir, f"production_{zone}.csv")
        out_df.to_csv(out_path)
        print(f"{zone}: {zone_to_cap[zone]:.1f} MW | Saved: {out_path}")
        per_zone_series[zone] = prod_series

    # Combined output
    combined = pd.concat(per_zone_series.values(), axis=1)
    combined.index.name = "timestamp"
    combined["Total_MWh"] = combined.sum(axis=1)

    combined_out = os.path.join(args.out_dir, "production_by_zone.csv")
    combined.to_csv(combined_out)

    total_gwh = combined["Total_MWh"].sum() / 1000.0
    avg = combined["Total_MWh"].mean()
    print(f"\nSaved combined output to: {combined_out}")
    print(f"~> Total production (all zones): {total_gwh:,.2f} GWh")
    print(f"Min hourly: {combined['Total_MWh'].min():.3f} MWh at {combined['Total_MWh'].idxmin()}")
    print(f"Max hourly: {combined['Total_MWh'].max():.3f} MWh at {combined['Total_MWh'].idxmax()}")
    print(f"Avg hourly: {avg:.3f} MWh")

    try:
        daily = combined["Total_MWh"].resample("D").sum()
        print(f"Min daily: {daily.min()/1000:.3f} GWh on {daily.idxmin().date()}")
        print(f"Max daily: {daily.max()/1000:.3f} GWh on {daily.idxmax().date()}")
    except Exception:
        pass

    # Daily production plot
    plot_daily_production(combined, args.out_dir, args.year)

    # Hourly two-week February plot
    plot_hourly_february(combined, args.out_dir, args.year)


if __name__ == "__main__":
    main()