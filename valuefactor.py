import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
# Each run is a (production_file, price_file) pair, in order.
# Edit these paths to match your actual file names.

RUNS = {
    "H1": ("Results_windsim/out_2024_WEC/production_by_zone1.csv", "Resultater_maisim/price_simulation_v2(H1).csv"),
    "H2": ("Results_windsim/out_2024_WEC2/production_by_zone2.csv", "Resultater_maisim/price_simulation_v2(H2).csv"),
    "H3": ("Results_windsim/out_2024_WEC3/production_by_zone3.csv", "Resultater_maisim/price_simulation_v2(H3).csv"),
    "H4": ("Results_windsim/out_2024_WEC4/production_by_zone4.csv", "Resultater_maisim/price_simulation_v2(H4).csv"),
    "H5": ("Results_windsim/out_2024_WEC5/production_by_zone5.csv", "Resultater_maisim/price_simulation_v2(H5).csv"),
}

# Production zones to calculate value factor for
ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5", "OS2", "OU2"]

# Onshore and offshore zone groupings
ONSHORE_ZONES  = ["NO1", "NO2", "NO3", "NO4", "NO5"]
OFFSHORE_ZONES = ["OS2", "OU2"]

# Column name for the price to use from the price file
PRICE_COL = "final_price"

# ── Value factor formula (equation 2) ────────────────────────────────────────
# V_yr = Σ(Pᵢ · pᵢ) / (Σ(Pᵢ) · p̄)

def value_factor(production: pd.Series, price: pd.Series, p_bar: float) -> float:
    """
    production : hourly wind production (MWh)
    price      : hourly spot price (EUR/MWh)
    p_bar      : time-weighted average price for the period
    """
    return (production * price).sum() / (production.sum() * p_bar)


# ── Load and process each run ─────────────────────────────────────────────────

def load_run(prod_path: str, price_path: str) -> pd.DataFrame:
    prod = pd.read_csv(prod_path)
    prod["ts"] = pd.to_datetime(prod["timestamp"])

    price = pd.read_csv(price_path)
    price["ts"] = (
        pd.to_datetime(price["time_oslo"], utc=True)
        .dt.tz_convert("Europe/Oslo")
        .dt.tz_localize(None)
    )

    df = pd.merge(prod, price[["ts", PRICE_COL]], on="ts", how="inner")
    return df


def calculate_run(df: pd.DataFrame) -> dict:
    p = df[PRICE_COL]
    p_bar = p.mean()
    result = {"p_bar": round(p_bar, 2)}

    for z in ZONES:
        col = f"production_MWh_{z}"
        q = df[col]
        result[z] = value_factor(q, p, p_bar) if q.sum() > 0 else None

    # Zone-level value factor using Total_MWh column (pre-aggregated)
    result["Total"] = value_factor(df["Total_MWh"], p, p_bar)

    # Aggregated value factor: treat all wind as a single portfolio.
    # Revenue = Σ_zones(Pᵢ_zone · pᵢ_zone), but since all zones share one
    # system price here, this simplifies to using Total_MWh × final_price.
    # The key difference from "Total" above is that p̄ is computed as the
    # production-share-weighted average of zone prices when zones have
    # separate prices. With a single price column it equals "Total".
    # This section is ready to extend if per-zone prices are added.
    total_production = df["Total_MWh"]
    result["Aggregated"] = value_factor(total_production, p, p_bar)

    # ── Onshore aggregated value factor (NO1–NO5) ─────────────────────────
    # Sum production across all onshore zones and compute a single VF.
    onshore_cols = [f"production_MWh_{z}" for z in ONSHORE_ZONES]
    onshore_production = df[onshore_cols].sum(axis=1)
    result["Onshore"] = (
        value_factor(onshore_production, p, p_bar)
        if onshore_production.sum() > 0 else None
    )

    # ── Offshore aggregated value factor (OS2 + OU2) ──────────────────────
    # Sum production across all offshore zones and compute a single VF.
    offshore_cols = [f"production_MWh_{z}" for z in OFFSHORE_ZONES]
    offshore_production = df[offshore_cols].sum(axis=1)
    result["Offshore"] = (
        value_factor(offshore_production, p, p_bar)
        if offshore_production.sum() > 0 else None
    )

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

all_results = []

for run_name, (prod_file, price_file) in RUNS.items():
    print(f"Processing {run_name}...")
    df = load_run(prod_file, price_file)
    result = calculate_run(df)
    result["run"] = run_name
    all_results.append(result)

results_df = pd.DataFrame(all_results).set_index("run")

# ── Print per-run results ─────────────────────────────────────────────────────

vf_cols = ZONES + ["Total", "Aggregated", "Onshore", "Offshore"]

print("\n=== Value factor per run (by zone) ===")
print(results_df[["p_bar"] + ZONES + ["Total"]].to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=== Value factor per run (aggregated wind portfolio) ===")
print(results_df[["p_bar", "Aggregated", "Onshore", "Offshore"]].to_string(float_format=lambda x: f"{x:.4f}"))

# ── Summary statistics across runs ───────────────────────────────────────────

def summarise(series: pd.Series) -> dict:
    vals = series.dropna()
    return {
        "mean":  vals.mean(),
        "std":   vals.std(),
        "min":   vals.min(),
        "max":   vals.max(),
        "range": vals.max() - vals.min(),
    }

print("\n=== Summary across runs (by zone) ===")
zone_summary = pd.DataFrame(
    {z: summarise(results_df[z]) for z in ZONES + ["Total"]}
).T
print(zone_summary.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=== Summary across runs (aggregated wind portfolio) ===")
agg_summary = pd.DataFrame({
    "Aggregated": summarise(results_df["Aggregated"]),
    "Onshore":    summarise(results_df["Onshore"]),
    "Offshore":   summarise(results_df["Offshore"]),
}).T
print(agg_summary.to_string(float_format=lambda x: f"{x:.4f}"))

# ── Optional: save results to CSV ────────────────────────────────────────────

results_df.to_csv("value_factor_per_run.csv")
zone_summary.to_csv("value_factor_summary_by_zone.csv")
agg_summary.to_csv("value_factor_summary_aggregated.csv")
print("\nResults saved to:")
print("  value_factor_per_run.csv")
print("  value_factor_summary_by_zone.csv")
print("  value_factor_summary_aggregated.csv")