#!/usr/bin/env python3
"""
Script 07 — Layer 1 OLS ablations: drop blocks of predictors and compare fit.

Same log(1 + returns) family as script 05 main OLS; writes ablation_ols_results.txt.
Pairs with script 09 (count knockouts). We rip out politics / housing / distance in chunks and see how R² moves — rough intuition, not causal magic.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

from nyfl.paths import DATA_PROCESSED

DYADIC = DATA_PROCESSED / "ny_fl_dyadic_2122.csv"
OUT = DATA_PROCESSED / "ablation_ols_results.txt"


def _prep(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    if "distance_km" in m.columns:
        m["log_distance_km"] = np.log1p(m["distance_km"].clip(lower=0))
    for c in ["o_population", "d_population"]:
        if c in m.columns:
            m[f"log_{c}"] = np.log1p(m[c].clip(lower=0))
    return m


BLOCKS: dict[str, list[str] | None] = {
    "full": None,
    "no_political": [
        "d_dem_two_party_share",
        "o_dem_two_party_share",
        "abs_diff_dem_two_party_share",
        "dest_more_dem",
    ],
    "no_housing": [
        "abs_diff_median_home_value",
        "abs_diff_median_gross_rent",
        "dyadic_housing_gap",
        "dyadic_affordability_gap",
        "abs_diff_affordability_index",
        "dest_lower_rent_pressure_than_origin",
    ],
    "no_distance": ["distance_km"],
}
ONLY_GRAVITY_KEEP = [
    "n_returns",
    "distance_km",
    "o_population",
    "d_population",
]


def build_design(m: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """log flow for y, stack whatever columns we have for X, then complete cases (N can shift when columns drop)."""
    # Dependent variable: log migration volume (Layer 1 style); BLOCKS below drop columns from X only.
    y = pd.Series(np.log1p(m["n_returns"]), name="log_flow", index=m.index)
    base = [
        "d_dem_two_party_share",
        "o_dem_two_party_share",
        "abs_diff_dem_two_party_share",
        "dest_more_dem",
        "distance_km",
        "log_distance_km",
        "o_population",
        "d_population",
        "log_o_population",
        "log_d_population",
        "abs_diff_median_home_value",
        "dyadic_housing_gap",
        "dyadic_affordability_gap",
        "abs_diff_affordability_index",
        "abs_diff_log_pop_density",
        "abs_diff_pct_nh_white",
        "dest_lower_rent_than_origin",
        "dest_lower_rent_pressure_than_origin",
    ]
    cols = [c for c in base if c in m.columns]
    X = m[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    n = len(X)
    min_non_null = max(5, int(0.05 * n)) if n else 0
    keep = [c for c in X.columns if X[c].notna().sum() >= min_non_null]
    X = X[keep].dropna()
    y = y.loc[X.index]
    X = sm.add_constant(X)
    return y, X


def main() -> None:
    """Loop the block drops, dump everything to ablation_ols_results.txt."""
    m = _prep(pd.read_csv(DYADIC))
    lines: list[str] = []

    # Each iteration: remove a thematic block of regressors, refit OLS, record R² and summary.
    for label, drop in BLOCKS.items():
        try:
            m2 = m.copy()
            if drop:
                for c in drop:
                    if c in m2.columns:
                        m2 = m2.drop(columns=[c])
            y, X = build_design(m2)
            if len(y) < 5 or X.shape[1] < 2:
                lines.append(f"\n=== {label} === SKIPPED (insufficient rows)\n")
                continue
            # HC1: heteroskedasticity-robust SEs (not clustered — quick ablation, not main inference).
            model = OLS(y, X).fit(cov_type="HC1")
            buf = io.StringIO()
            buf.write(model.summary().as_text())
            lines.append(f"\n=== {label} ===\n")
            lines.append(buf.getvalue())
            lines.append(f"R-squared: {model.rsquared:.4f}  N: {int(model.nobs)}\n")
        except Exception as e:
            lines.append(f"\n=== {label} === ERROR: {e}\n")

    # Gravity-only specification
    try:
        keep = [c for c in ONLY_GRAVITY_KEEP if c in m.columns]
        m2 = m[keep].copy()
        y, X = build_design(m2)
        if len(y) >= 5 and X.shape[1] >= 2:
            model = OLS(y, X).fit(cov_type="HC1")  # same HC1 as block ablations above
            buf = io.StringIO()
            buf.write(model.summary().as_text())
            lines.append("\n=== only_gravity ===\n")
            lines.append(buf.getvalue())
            lines.append(f"R-squared: {model.rsquared:.4f}  N: {int(model.nobs)}\n")
    except Exception as e:
        lines.append(f"\n=== only_gravity === ERROR: {e}\n")

    text = "".join(lines)
    OUT.write_text(text)
    print(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
