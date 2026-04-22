#!/usr/bin/env python3
"""
Script 03 — Dyadic dataset: each row = one NY county -> FL county flow with covariates.

Steps:
  1) Merge IRS flows to origin county row and destination county row (prefix o_ / d_).
  2) Distance: great-circle km from gazetteer centroids; optional external pair-distance CSV can override.
  3) Dyadic features: absolute differences in partisanship, race, rent, home value, density, affordability, etc.
  4) Directional dummies (e.g. dest more Democratic than origin; dest cheaper rent).
  5) If script 04 already ran: merge ML predictions and form dyadic_housing_gap, dyadic_affordability_gap.
  6) Log transforms for gravity models; human-readable origin_county_label / dest_county_label.

04 has to run before 03 if you want ML-based dyadic gaps — see run_all.sh.

We build |Δ| measures between origin and dest, plus a few directional dummies. The ML columns are just
(predicted dest − predicted origin) from the county GBDT, not raw ACS fields.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from nyfl.county_labels import label_for_fips, load_fips_label_map
from nyfl.geo import haversine_km
from nyfl.paths import DATA_PROCESSED, DATA_RAW

FLOW_CSV = DATA_PROCESSED / "ny_fl_flows_2122.csv"
COV_CSV = DATA_PROCESSED / "county_covariates_ny_fl.csv"
HOUSING_PRED_CSV = DATA_PROCESSED / "county_housing_gbdt_predictions.csv"
DEFAULT_EXTERNAL_DIST = DATA_RAW / "nber_pair_distances.csv"


def _load_external_pair_distances() -> pd.DataFrame | None:
    """If you have a pre-made OD distance table, use it instead of haversine (handy for robustness).

    Set PAIR_DISTANCE_CSV or NBER_PAIR_DISTANCES_CSV, or drop `data/raw/nber_pair_distances.csv` in place.
    """
    env = os.environ.get("PAIR_DISTANCE_CSV") or os.environ.get("NBER_PAIR_DISTANCES_CSV")
    if env:
        path = Path(env)
        if not path.is_file():
            print(f"Warning: pair distance file not found ({path}) — using haversine only.")
            return None
    else:
        path = DEFAULT_EXTERNAL_DIST
        if not path.is_file():
            return None
    ext = pd.read_csv(path, dtype={"origin_fips5": str, "dest_fips5": str})
    ext["origin_fips5"] = ext["origin_fips5"].astype(str).str.zfill(5)
    ext["dest_fips5"] = ext["dest_fips5"].astype(str).str.zfill(5)
    dcol = "distance_km" if "distance_km" in ext.columns else "distance_km_nber"
    if dcol not in ext.columns:
        raise ValueError(f"{path} needs a distance column: distance_km or distance_km_nber")
    ext = ext.rename(columns={dcol: "distance_km_external"})
    return ext[["origin_fips5", "dest_fips5", "distance_km_external"]].drop_duplicates(
        ["origin_fips5", "dest_fips5"]
    )


def main() -> None:
    """Stick flows onto origin county row + destination county row, add distance, gaps, logs — the main analysis table."""
    # IRS flows: origin/dest FIPS and n_returns (and AGI) per directed pair.
    flows = pd.read_csv(FLOW_CSV, dtype={"origin_fips5": str, "dest_fips5": str})
    # One row per county (NY or FL); will be joined twice — as origin and as destination.
    cov = pd.read_csv(COV_CSV, dtype={"county_fips5": str})
    cov["county_fips5"] = cov["county_fips5"].astype(str).str.zfill(5)

    # Prefix every covariate column with o_ or d_ so origin and dest attributes sit on one dyad row.
    o = cov.add_prefix("o_").rename(columns={"o_county_fips5": "origin_fips5"})
    d = cov.add_prefix("d_").rename(columns={"d_county_fips5": "dest_fips5"})

    # Left join: keep every flow row; unmatched FIPS would produce NaNs (diagnose script tracks this).
    m = flows.merge(o, on="origin_fips5", how="left").merge(d, on="dest_fips5", how="left")

    # Distance (km): haversine on gazetteer centroids; optional external pair file (e.g. NBER-style) overrides when matched
    if {"o_lat", "o_lon", "d_lat", "d_lon"}.issubset(m.columns):
        m["distance_km_haversine"] = [
            haversine_km(a, b, c, d)
            for a, b, c, d in zip(m["o_lat"], m["o_lon"], m["d_lat"], m["d_lon"])
        ]
    else:
        m["distance_km_haversine"] = np.nan
    m["distance_km"] = pd.to_numeric(m["distance_km_haversine"], errors="coerce")
    m["distance_km_source"] = ""
    m.loc[m["distance_km"].notna(), "distance_km_source"] = "haversine"

    ext_df = _load_external_pair_distances()
    if ext_df is not None:
        m = m.merge(ext_df, on=["origin_fips5", "dest_fips5"], how="left")
        use = m["distance_km_external"].notna()
        m.loc[use, "distance_km"] = pd.to_numeric(m.loc[use, "distance_km_external"], errors="coerce")
        m.loc[use, "distance_km_source"] = "external"
        print(f"Merged external pair distances ({int(use.sum())} / {len(m)} rows matched).")
        m = m.drop(columns=["distance_km_external"])
    else:
        print("Distances: great-circle km from gazetteer (no external pair-distance CSV).")

    # Political / demographic dissimilarity: |dest − origin| measures "how different" the two counties are.
    for col in [
        "dem_two_party_share",
        "pct_nh_white",
        "median_home_value",
        "median_gross_rent",
        "log_pop_density",
        "affordability_index",
        "pct_owner_occupied",
        "owner_to_renter_ratio",
    ]:
        oc = f"o_{col}"
        dc = f"d_{col}"
        if oc in m.columns and dc in m.columns:
            m[f"abs_diff_{col}"] = (m[dc] - m[oc]).abs()

    # Directional: +1 if destination more Democratic than origin
    if "o_dem_two_party_share" in m.columns and "d_dem_two_party_share" in m.columns:
        m["dest_more_dem"] = (
            m["d_dem_two_party_share"] > m["o_dem_two_party_share"]
        ).astype(float)
        m.loc[m["o_dem_two_party_share"].isna() | m["d_dem_two_party_share"].isna(), "dest_more_dem"] = np.nan

    # Directional: destination lower median rent than origin (housing "cheaper")
    if "o_median_gross_rent" in m.columns and "d_median_gross_rent" in m.columns:
        m["dest_lower_rent_than_origin"] = (
            m["d_median_gross_rent"] < m["o_median_gross_rent"]
        ).astype(float)
        m.loc[
            m["o_median_gross_rent"].isna() | m["d_median_gross_rent"].isna(),
            "dest_lower_rent_than_origin",
        ] = np.nan

    # rent/income pressure: lower index = relatively more affordable
    if "o_affordability_index" in m.columns and "d_affordability_index" in m.columns:
        m["dest_lower_rent_pressure_than_origin"] = (
            m["d_affordability_index"] < m["o_affordability_index"]
        ).astype(float)
        m.loc[
            m["o_affordability_index"].isna() | m["d_affordability_index"].isna(),
            "dest_lower_rent_pressure_than_origin",
        ] = np.nan

    # Layer 3: county-level GBDT predictions from script 04; dyadic gap = "predicted dest minus predicted origin" (same model).
    if HOUSING_PRED_CSV.exists():
        hp = pd.read_csv(HOUSING_PRED_CSV, dtype={"county_fips5": str})
        hp["county_fips5"] = hp["county_fips5"].astype(str).str.zfill(5)
        if "housing_pred" not in hp.columns and "pred_median_home_value" in hp.columns:
            hp["housing_pred"] = hp["pred_median_home_value"]
        if "housing_pred" in hp.columns:
            # Attach the same predicted median home value twice: once keyed by origin FIPS, once by dest FIPS.
            ho = hp[["county_fips5", "housing_pred"]].rename(
                columns={"county_fips5": "origin_fips5", "housing_pred": "o_housing_pred"}
            )
            hd = hp[["county_fips5", "housing_pred"]].rename(
                columns={"county_fips5": "dest_fips5", "housing_pred": "d_housing_pred"}
            )
            m = m.merge(ho, on="origin_fips5", how="left").merge(hd, on="dest_fips5", how="left")
            # Positive gap ⇒ model predicts higher typical home values at destination than source (for that dyad).
            m["dyadic_housing_gap"] = m["d_housing_pred"] - m["o_housing_pred"]
        if "affordability_pred" in hp.columns:
            ao = hp[["county_fips5", "affordability_pred"]].rename(
                columns={"county_fips5": "origin_fips5", "affordability_pred": "o_affordability_pred"}
            )
            ad = hp[["county_fips5", "affordability_pred"]].rename(
                columns={"county_fips5": "dest_fips5", "affordability_pred": "d_affordability_pred"}
            )
            m = m.merge(ao, on="origin_fips5", how="left").merge(ad, on="dest_fips5", how="left")
            m["dyadic_affordability_gap"] = m["d_affordability_pred"] - m["o_affordability_pred"]
    else:
        print(f"No {HOUSING_PRED_CSV} yet — run script 04 before 03 for dyadic gaps (or run full pipeline).")

    # log1p compresses heavy tails (distance, populations) so regression coefficients are interpretable elasticities.
    if "distance_km" in m.columns:
        m["log_distance_km"] = np.log1p(m["distance_km"].clip(lower=0))
    for c in ["o_population", "d_population"]:
        if c in m.columns:
            m[f"log_{c}"] = np.log1p(m[c].clip(lower=0))

    # Human-readable county labels for every row (ACS NAME → "County, ST"); FIPS retained for merges.
    try:
        lab = load_fips_label_map(COV_CSV)
        if lab:
            m["origin_county_label"] = m["origin_fips5"].astype(str).str.zfill(5).map(lambda f: label_for_fips(f, lab))
            m["dest_county_label"] = m["dest_fips5"].astype(str).str.zfill(5).map(lambda f: label_for_fips(f, lab))
    except Exception as e:
        print(f"County labels on dyadic skipped: {e}")

    out = DATA_PROCESSED / "ny_fl_dyadic_2122.csv"
    m.to_csv(out, index=False)
    print(f"Wrote {len(m)} dyadic rows to {out}")


if __name__ == "__main__":
    main()
