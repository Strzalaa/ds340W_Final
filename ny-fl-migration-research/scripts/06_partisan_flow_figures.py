#!/usr/bin/env python3
"""
Script 06 — Descriptive figures: flows vs partisanship, top FL destinations.

Reads the dyads + covariates; saves a couple PNGs under outputs/figures — eyeball stuff before the regressions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nyfl.county_labels import label_for_fips, load_fips_label_map
from nyfl.paths import DATA_PROCESSED, OUTPUTS

DYADIC = DATA_PROCESSED / "ny_fl_dyadic_2122.csv"
COVARIATES = DATA_PROCESSED / "county_covariates_ny_fl.csv"


def _dest_county_labels(dest_fips5: pd.Series, dyad: pd.DataFrame) -> pd.Series:
    """Prefer dyadic dest_county_label; else ACS-derived 'County, ST'."""
    fips = dest_fips5.astype(str).str.zfill(5)
    if "dest_county_label" in dyad.columns and "dest_fips5" in dyad.columns:
        d2 = dyad[["dest_fips5", "dest_county_label"]].copy()
        d2["dest_fips5"] = d2["dest_fips5"].astype(str).str.zfill(5)
        mp = d2.drop_duplicates("dest_fips5").set_index("dest_fips5")["dest_county_label"]
        lab = fips.map(mp)
        if lab.notna().any():
            fill = load_fips_label_map(Path(COVARIATES) if Path(COVARIATES).is_file() else None)
            return lab.fillna(fips.map(lambda x: label_for_fips(x, fill)))
    mp2 = load_fips_label_map(Path(COVARIATES) if Path(COVARIATES).is_file() else None)
    return fips.map(lambda x: label_for_fips(x, mp2))


def main() -> None:
    """Bar chart of where people land + scatter of flow vs how blue the destination is."""
    m = pd.read_csv(DYADIC, dtype={"dest_fips5": str, "origin_fips5": str})
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Figure 1: aggregate all NY origins into each FL destination — who receives the most movers?
    if "dest_fips5" not in m.columns or "n_returns" not in m.columns:
        print("Missing columns; skip 06.")
        return

    if "d_dem_two_party_share" in m.columns:
        agg = m.groupby("dest_fips5", as_index=False).agg(
            total_returns=("n_returns", "sum"),
            d_dem=("d_dem_two_party_share", "first"),
        )
    else:
        agg = m.groupby("dest_fips5", as_index=False).agg(total_returns=("n_returns", "sum"))
        agg["d_dem"] = np.nan

    agg = agg.sort_values("total_returns", ascending=False).head(20)
    agg["dest_fips5"] = agg["dest_fips5"].astype(str).str.zfill(5)
    agg["dest_label"] = _dest_county_labels(agg["dest_fips5"], m)

    fig, ax = plt.subplots(figsize=(10, 7))
    plot = agg.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot))
    if plot["d_dem"].notna().any():
        norm = plt.Normalize(0, 1)
        bar_colors = plt.cm.RdBu_r(norm(plot["d_dem"].fillna(0.5).values))
    else:
        bar_colors = "steelblue"
    ax.barh(y_pos, plot["total_returns"], color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot["dest_label"])
    ax.set_xlabel("Total IRS returns (NY→FL into county)")
    ax.set_ylabel("Destination (Florida county)")
    ax.set_title("Top FL destination counties by NY→FL flow volume (IRS 2021–2022)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "top_fl_destinations_by_flow.png", dpi=150)
    plt.close()
    print(f"Saved {OUTPUTS / 'top_fl_destinations_by_flow.png'}")

    # Figure 2: dyad-level scatter — flow intensity vs destination county Biden/Trump balance (ecological).
    if "d_dem_two_party_share" in m.columns:
        m["log_flow"] = np.log1p(m["n_returns"])
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = m.dropna(subset=["d_dem_two_party_share"])
        ax.scatter(sub["d_dem_two_party_share"], sub["log_flow"], alpha=0.35, s=12, c="darkblue")
        ax.set_xlabel("Destination county Dem two-party share")
        ax.set_ylabel("log(1 + returns)")
        ax.set_title("Ecological sorting: NY→FL dyads vs FL destination partisanship")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "gimpel_style_flow_vs_dest_dem.png", dpi=150)
        plt.close()
        print(f"Saved {OUTPUTS / 'gimpel_style_flow_vs_dest_dem.png'}")


if __name__ == "__main__":
    main()
