#!/usr/bin/env python3
"""
Diagnose dyadic missingness — why regressions use fewer rows than the raw IRS merge.

Scans ny_fl_dyadic_2122.csv for NaNs on columns used in Layer 1 / Layer 2 complete-case logic.
Writes dyadic_missingness.csv plus a tiny text blurb — handy when someone asks why regression N < row count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyfl.paths import DATA_PROCESSED, ROOT

DYADIC = DATA_PROCESSED / "ny_fl_dyadic_2122.csv"
OUT_CSV = ROOT / "outputs" / "tables" / "dyadic_missingness.csv"
OUT_TXT = DATA_PROCESSED / "dyadic_missingness_summary.txt"

# Columns referenced in Layer 1 / Layer 2 complete-case logic (plus identifiers)
TRACK_COLS = [
    "origin_fips5",
    "dest_fips5",
    "n_returns",
    "log_flow",
    "o_dem_two_party_share",
    "d_dem_two_party_share",
    "abs_diff_dem_two_party_share",
    "dest_more_dem",
    "log_distance_km",
    "log_o_population",
    "log_d_population",
    "dyadic_housing_gap",
    "dyadic_affordability_gap",
    "abs_diff_log_pop_density",
    "abs_diff_pct_nh_white",
    "dest_lower_rent_than_origin",
    "abs_diff_affordability_index",
    "dest_lower_rent_pressure_than_origin",
]


def main() -> None:
    """Count NaNs per important column and ballpark how many rows the big regressions actually use."""
    if not DYADIC.is_file():
        print(f"Skip missingness diagnostic: {DYADIC} not found (run 03 first).")
        return

    df = pd.read_csv(DYADIC, dtype={"origin_fips5": str, "dest_fips5": str})
    n = len(df)
    rows: list[dict[str, object]] = []
    # Per-column NaN counts — if a regressor is often missing, listwise deletion shrinks N a lot.
    for c in TRACK_COLS:
        if c not in df.columns:
            rows.append({"column": c, "present": False, "n_missing": None, "pct_missing": None})
            continue
        miss = df[c].isna().sum()
        rows.append(
            {
                "column": c,
                "present": True,
                "n_missing": int(miss),
                "pct_missing": round(100.0 * float(miss) / max(n, 1), 2),
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # Roughly mirror how 05 drops rows so the N printed here isn’t a total lie.
    import numpy as np

    m = df.copy()
    m["log_flow"] = np.log1p(m["n_returns"])
    if {"o_dem_two_party_share", "d_dem_two_party_share"}.issubset(m.columns):
        m["dem_origin_x_dest"] = m["o_dem_two_party_share"] * m["d_dem_two_party_share"]
    if "distance_km" in m.columns:
        m["log_distance_km"] = np.log1p(m["distance_km"].clip(lower=0))
    for c in ["o_population", "d_population"]:
        if c in m.columns:
            m[f"log_{c}"] = np.log1p(m[c].clip(lower=0))

    l1_cols = [
        "o_dem_two_party_share",
        "d_dem_two_party_share",
        "dem_origin_x_dest",
        "log_distance_km",
        "log_o_population",
        "log_d_population",
    ]
    if "dyadic_housing_gap" in m.columns:
        l1_cols.append("dyadic_housing_gap")
    if "dyadic_affordability_gap" in m.columns:
        l1_cols.append("dyadic_affordability_gap")
    l1_cols = [c for c in l1_cols if c in m.columns]
    ok1 = m["log_flow"].notna() & m[l1_cols].notna().all(axis=1)
    if "origin_fips5" in m.columns:
        ok1 = ok1 & m["origin_fips5"].notna()

    l2_cols = [
        "abs_diff_dem_two_party_share",
        "abs_diff_log_pop_density",
        "abs_diff_pct_nh_white",
        "dest_more_dem",
        "log_distance_km",
        "log_o_population",
        "log_d_population",
    ]
    for opt in (
        "dest_lower_rent_than_origin",
        "dyadic_housing_gap",
        "dyadic_affordability_gap",
        "abs_diff_affordability_index",
        "dest_lower_rent_pressure_than_origin",
    ):
        if opt in m.columns:
            l2_cols.append(opt)
    l2_cols = [c for c in l2_cols if c in m.columns]
    ok2 = m["n_returns"].notna() & m[l2_cols].notna().all(axis=1)
    if "origin_fips5" in m.columns:
        ok2 = ok2 & m["origin_fips5"].notna()

    l2_reduced = [
        "abs_diff_dem_two_party_share",
        "abs_diff_log_pop_density",
        "abs_diff_pct_nh_white",
        "dest_more_dem",
        "log_distance_km",
        "log_o_population",
        "log_d_population",
    ]
    l2_reduced = [c for c in l2_reduced if c in m.columns]
    ok2r = m["n_returns"].notna() & m[l2_reduced].notna().all(axis=1)
    if "origin_fips5" in m.columns:
        ok2r = ok2r & m["origin_fips5"].notna()

    txt_lines = [
        f"Dyadic rows (all positive-flow pairs): {n}",
        f"Layer-1 full-spec complete cases (approx.): {int(ok1.sum())}",
        f"Layer-2 full-spec complete cases (approx.): {int(ok2.sum())}",
        f"Layer-2 reduced-spec complete cases (approx.): {int(ok2r.sum())}",
        "",
        f"Per-column missingness: {OUT_CSV}",
    ]
    OUT_TXT.write_text("\n".join(txt_lines) + "\n")
    print("\n".join(txt_lines))


if __name__ == "__main__":
    main()
