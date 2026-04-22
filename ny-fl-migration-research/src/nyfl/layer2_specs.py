"""
Shared Layer 2 count-model design: Poisson/NB/GEE with origin-population offset.

We removed log_o_population from the mean equation and use offset = log(origin population)
as the standard gravity exposure term. We also drop abs_diff_affordability_index when it is
nearly collinear with dyadic_affordability_gap (|corr| > 0.95) to avoid rank-deficient designs
that produced NaN cluster SEs and a degenerate pseudo R^2 of 1 for Poisson.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Layer2Design:
    y: pd.Series
    X: pd.DataFrame
    offset: pd.Series  # log origin population (already log-scaled for offset)
    groups: pd.Series
    dropped_for_collinearity: list[str]
    x_columns: list[str]
    standardized: list[str]  # columns z-scored for conditioning


def _base_predictor_names(m: pd.DataFrame, include_housing_ml: bool) -> list[str]:
    # Core gravity + demographics + politics; full=True appends ML gaps and rent-pressure dummies when present.
    cols = [
        "abs_diff_dem_two_party_share",
        "abs_diff_log_pop_density",
        "abs_diff_pct_nh_white",
        "dest_more_dem",
        "log_distance_km",
        "log_d_population",
    ]
    if include_housing_ml:
        if "dest_lower_rent_than_origin" in m.columns:
            cols.append("dest_lower_rent_than_origin")
        if "dest_lower_rent_pressure_than_origin" in m.columns:
            cols.append("dest_lower_rent_pressure_than_origin")
        if "dyadic_housing_gap" in m.columns:
            cols.append("dyadic_housing_gap")
        if "dyadic_affordability_gap" in m.columns:
            cols.append("dyadic_affordability_gap")
        if "abs_diff_affordability_index" in m.columns:
            cols.append("abs_diff_affordability_index")
    return [c for c in cols if c in m.columns]


def build_layer2_design(
    m: pd.DataFrame,
    *,
    full: bool,
) -> Layer2Design | None:
    """Build y, X, offset, cluster ids for the count models.

    `full=True` pulls in ML gaps + rent stuff when those columns exist; `full=False` is the slimmer spec.
    We z-score continuous X’s; origin pop only shows up as log(offset), not twice in the design matrix.
    """
    if "n_returns" not in m.columns or "o_population" not in m.columns:
        return None

    # y = observed migrant count (same IRS n_returns as Layer 1, but on the original scale).
    y = m["n_returns"].astype(float)
    cols = _base_predictor_names(m, include_housing_ml=full)
    need = {"log_distance_km", "log_d_population"}.intersection(cols)
    if len(need) < 1 or len(cols) < 2:
        return None

    X = m[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Exposure: origin population enters as offset, not as another log term in X (standard gravity fix).
    pop = pd.to_numeric(m["o_population"], errors="coerce").clip(lower=1.0)
    offset = np.log(pop)

    ok = (
        y.notna()
        & X.notna().all(axis=1)
        & offset.notna()
        & m["o_population"].notna()
    )
    if "origin_fips5" in m.columns:
        ok = ok & m["origin_fips5"].notna()
    y = y.loc[ok]
    X = X.loc[ok].copy()
    offset = offset.loc[ok]
    groups = m.loc[ok, "origin_fips5"].astype(str) if "origin_fips5" in m.columns else pd.Series(range(len(y)), index=y.index, dtype=str)

    dropped: list[str] = []
    if (
        "abs_diff_affordability_index" in X.columns
        and "dyadic_affordability_gap" in X.columns
    ):
        c = X["abs_diff_affordability_index"].corr(X["dyadic_affordability_gap"])
        if pd.notna(c) and abs(float(c)) > 0.95:
            X = X.drop(columns=["abs_diff_affordability_index"])
            dropped.append("abs_diff_affordability_index (|corr|>0.95 vs dyadic_affordability_gap)")

    if len(y) < 10 or X.shape[1] < 1:
        return None

    # Z-score continuous columns (exclude 0/1 dummies) to prevent near-singular Poisson information
    # matrices that produced NaN cluster/GEE SEs and a degenerate Cox–Snell pseudo R² of 1.0.
    standardized: list[str] = []
    for col in list(X.columns):
        s = pd.to_numeric(X[col], errors="coerce")
        if s.notna().sum() < 3:
            continue
        u = pd.Series(s.dropna().unique())
        if len(u) <= 2 and set(np.round(u.astype(float), 6)).issubset({0.0, 1.0}):
            continue
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if sd and sd > 1e-12:
            X[col] = (s - mu) / sd
            standardized.append(col)

    return Layer2Design(
        y=y,
        X=X,
        offset=offset,
        groups=groups,
        dropped_for_collinearity=dropped,
        x_columns=list(X.columns),
        standardized=standardized,
    )


def _apply_zscore(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Z-score continuous columns; skip binary 0/1 — used by the raw-ACS housing spec only."""
    X = X.copy()
    standardized: list[str] = []
    for col in list(X.columns):
        s = pd.to_numeric(X[col], errors="coerce")
        if s.notna().sum() < 3:
            continue
        u = pd.Series(s.dropna().unique())
        if len(u) <= 2 and set(np.round(u.astype(float), 6)).issubset({0.0, 1.0}):
            continue
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if sd and sd > 1e-12:
            X[col] = (s - mu) / sd
            standardized.append(col)
    return X, standardized


def build_layer2_raw_acs_housing_design(m: pd.DataFrame) -> Layer2Design | None:
    """Housing enters only through raw ACS |Δ| fields — script 12 uses this next to the ML spec."""
    if "n_returns" not in m.columns or "o_population" not in m.columns:
        return None

    cols = [
        "abs_diff_dem_two_party_share",
        "abs_diff_log_pop_density",
        "abs_diff_pct_nh_white",
        "dest_more_dem",
        "log_distance_km",
        "log_d_population",
    ]
    for c in (
        "abs_diff_median_home_value",
        "abs_diff_median_gross_rent",
        "abs_diff_pct_owner_occupied",
    ):
        if c in m.columns:
            cols.append(c)

    cols = [c for c in cols if c in m.columns]
    if len(cols) < 2 or "log_distance_km" not in cols:
        return None

    y = m["n_returns"].astype(float)
    X = m[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pop = pd.to_numeric(m["o_population"], errors="coerce").clip(lower=1.0)
    offset = np.log(pop)

    ok = y.notna() & X.notna().all(axis=1) & offset.notna() & m["o_population"].notna()
    if "origin_fips5" in m.columns:
        ok = ok & m["origin_fips5"].notna()
    y = y.loc[ok]
    X = X.loc[ok].copy()
    offset = offset.loc[ok]
    groups = m.loc[ok, "origin_fips5"].astype(str) if "origin_fips5" in m.columns else pd.Series(range(len(y)), index=y.index, dtype=str)

    if len(y) < 10:
        return None

    X, standardized = _apply_zscore(X)

    return Layer2Design(
        y=y,
        X=X,
        offset=offset,
        groups=groups,
        dropped_for_collinearity=[],
        x_columns=list(X.columns),
        standardized=standardized,
    )
