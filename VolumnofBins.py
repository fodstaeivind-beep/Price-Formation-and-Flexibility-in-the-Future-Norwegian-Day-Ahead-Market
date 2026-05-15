"""
Create an Excel file with hourly supply-curve bin volumes + hourly wind production (2025, Norway).

Inputs
------
1) Daily JSON bid-curve files:
   folder: nordpool_bidcurves/2025/
   names : yyyy-mm-dd.json

   Each file contains:
     aggregatedOrderPositions[] (hourly)
       deliveryStart (UTC, e.g. 2025-03-10T23:00:00Z)
       aggregatedSupplyCurve[] (cumulative volume vs price):
         { "price": ..., "volume": ... }

2) Wind Excel file:
   ProductionByLocation_2025_NO_None_MWh.xlsx
   - has headers on row 2 -> read with header=1
   - wind column: "Wind Onshore (Average MWh)"
   - time columns appear as "Unnamed: 0" and optionally "Unnamed: 1"
     (script handles both cases)
   - timestamps appear to be day-first, e.g. "13.01.2025 00:00:00"

What it outputs
---------------
An Excel file per month (option 2):
  supply_bins_2025_MM.xlsx

Columns:
  datetime_utc, datetime_oslo, wind_mwh,
  bin1_p_lt_-498, bin2_-498_to_-50, bin3_-50_to_0, bin4_0_to_110, bin5_110_to_4000

How bin volumes are computed
----------------------------
aggregatedSupplyCurve is cumulative Q(p). Using interpolation:
  V1 = Q(-498)
  V2 = Q(-50)  - Q(-498)
  V3 = Q(0)    - Q(-50)
  V4 = Q(110)  - Q(0)
  V5 = Q(4000) - Q(110)
"""

from __future__ import annotations

import os
import glob
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# -----------------------
# SETTINGS (EDIT IF NEEDED)
# -----------------------
JSON_DIR = "nordpool_bidcurves/2024"
WIND_XLSX = "ProductionByLocation_2024_NO_None_MWh.xlsx"
LOCAL_TZ = "Europe/Oslo"

# Your requested price breakpoints for 5 bins:
P_BREAKS = [-498.0, -50.0, 0.0, 110.0, 4000.0]

# Wind column name you provided:
WIND_VALUE_COL = "Wind Onshore (Average MWh)"
Hydro_value_col_ror = "Hydro Run Of River And Poundage (Average MWh)"
Hydro_value_col_res = "Hydro Water Reservoir (Average MWh)"

# -----------------------
# SUPPLY CURVE BINNING
# -----------------------
def _interp_Q(prices: np.ndarray, volumes: np.ndarray, p_star: float) -> float:
    """Interpolate cumulative volume Q(p) at p_star; prices must be sorted ascending."""
    return float(np.interp(p_star, prices, volumes))


def compute_bin_volumes_from_aggregated_curve(
    curve: List[Dict], p_breaks: List[float]
) -> Tuple[float, float, float, float, float]:
    """
    aggregatedSupplyCurve is cumulative volume Q(p).
    Bin volumes are differences: Q(b) - Q(a).
    """
    if not curve:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    prices = np.array([pt["price"] for pt in curve], dtype=float)
    volumes = np.array([pt["volume"] for pt in curve], dtype=float)

    # sort ascending by price for interpolation
    idx = np.argsort(prices)
    prices, volumes = prices[idx], volumes[idx]

    q_m498 = _interp_Q(prices, volumes, p_breaks[0])
    q_m50 = _interp_Q(prices, volumes, p_breaks[1])
    q_0 = _interp_Q(prices, volumes, p_breaks[2])
    q_110 = _interp_Q(prices, volumes, p_breaks[3])
    q_4000 = _interp_Q(prices, volumes, p_breaks[4])

    v1 = q_m498
    v2 = q_m50 - q_m498
    v3 = q_0 - q_m50
    v4 = q_110 - q_0
    v5 = q_4000 - q_110

    return (v1, v2, v3, v4, v5)


def parse_one_json(filepath: str) -> pd.DataFrame:
    """Parse one yyyy-mm-dd.json file into an hourly dataframe of bin volumes."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for pos in data.get("aggregatedOrderPositions", []):
        ts_utc = pos.get("deliveryStart")
        curve = pos.get("supplyCurve", [])

        v1, v2, v3, v4, v5 = compute_bin_volumes_from_aggregated_curve(curve, P_BREAKS)

        rows.append(
            {
                "datetime_utc": ts_utc,
                "bin1_p_lt_-498": v1,
                "bin2_-498_to_-50": v2,
                "bin3_-50_to_0": v3,
                "bin4_0_to_110": v4,
                "bin5_110_to_4000": v5,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["datetime_oslo"] = df["datetime_utc"].dt.tz_convert(LOCAL_TZ)
    return df


def build_supply_bins_for_month(year: int, month: int) -> pd.DataFrame:
    """Load all JSON files for a given month and return hourly supply-bin volumes."""
    pattern = os.path.join(JSON_DIR, f"{year}-{month:02d}-*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No JSON files found for pattern: {pattern}")

    dfs = []
    for fp in files:
        d = parse_one_json(fp)
        if not d.empty:
            dfs.append(d)

    supply = pd.concat(dfs, ignore_index=True).sort_values("datetime_utc")
    return supply


# -----------------------
# WIND READING
# -----------------------
def read_wind_xlsx(path: str) -> pd.DataFrame:
    """
    Read production data and return hourly (UTC):
      datetime_utc, wind_mwh, hydro_res_mwh, hydro_ror_mwh
    """
    df = pd.read_excel(path, header=1)

    # Validate required columns
    required = [WIND_VALUE_COL, Hydro_value_col_res, Hydro_value_col_ror]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    c0 = "Unnamed: 0"
    c1 = "Unnamed: 1"
    if c0 not in df.columns:
        raise ValueError(f"Expected '{c0}' not found. Columns: {df.columns.tolist()}")

    # Try parse Unnamed: 0 directly as datetime (day-first)
    dt0 = pd.to_datetime(df[c0], dayfirst=True, errors="coerce")

    if dt0.notna().mean() > 0.9:
        df["datetime_local"] = dt0
    else:
        # Otherwise combine Unnamed: 0 + Unnamed: 1
        if c1 not in df.columns:
            raise ValueError(
                "Could not parse 'Unnamed: 0' as timestamps and 'Unnamed: 1' is missing."
            )
        combined = df[c0].astype(str).str.strip() + " " + df[c1].astype(str).str.strip()
        df["datetime_local"] = pd.to_datetime(combined, dayfirst=True, errors="coerce")

    # Keep rows where we have a timestamp (production values may be missing; merge handles that)
    df = df.dropna(subset=["datetime_local"]).copy()

    # Convert local Oslo time -> UTC hourly
    df["datetime_utc"] = (
        pd.to_datetime(df["datetime_local"])
        .dt.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.floor("H")
    )

    # Rename production columns to consistent output names
    df = df.rename(
        columns={
            WIND_VALUE_COL: "wind_mwh",
            Hydro_value_col_res: "hydro_res_mwh",
            Hydro_value_col_ror: "hydro_ror_mwh",
        }
    )

    # If duplicates exist within an hour, take mean (matches "Average MWh")
    prod = (
        df.groupby("datetime_utc", as_index=False)[
            ["wind_mwh", "hydro_res_mwh", "hydro_ror_mwh"]
        ]
        .mean()
        .sort_values("datetime_utc")
    )

    return prod


# -----------------------
# MONTHLY PROCESS (OPTION 2)
# -----------------------
def process_month(year: int, month: int) -> None:
    supply = build_supply_bins_for_month(year, month)
    prod = read_wind_xlsx(WIND_XLSX)

    out = supply.merge(prod, on="datetime_utc", how="left")

    out = out[
        [
        "datetime_utc",
        "datetime_oslo",
        "wind_mwh",
        "hydro_res_mwh",
        "hydro_ror_mwh",
        "bin1_p_lt_-498",
        "bin2_-498_to_-50",
        "bin3_-50_to_0",
        "bin4_0_to_110",
        "bin5_110_to_4000",
        ]
    ]

    out["datetime_utc"] = out["datetime_utc"].dt.tz_localize(None)
    out["datetime_oslo"] = out["datetime_oslo"].dt.tz_localize(None)

    filename = f"supply_bins_{year}_{month:02d}.xlsx"
    out.to_excel(filename, index=False)

    print(f"✅ Wrote {filename} ({len(out)} rows)")
    print(f"Wind missing share: {out['wind_mwh'].isna().mean():.1%}")
    print(f"Hydro reservoir missing share: {out['hydro_res_mwh'].isna().mean():.1%}")
    print(f"Hydro RoR missing share: {out['hydro_ror_mwh'].isna().mean():.1%}")


# -----------------------
# RUN ONE MONTH (EDIT MONTH HERE)
# -----------------------
if __name__ == "__main__":
    process_month(2024, 12)  # change 1 -> 2,3,...,12 for other months
    

