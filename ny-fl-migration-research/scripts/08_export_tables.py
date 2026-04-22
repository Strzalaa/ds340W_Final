#!/usr/bin/env python3
"""
Script 08 — Publication-style tables and extra figures.

Exports CSVs (top destinations, income bins, NY-origin summaries) and matplotlib figures
to outputs/tables/ and outputs/figures/ with readable county names — slide fodder, doesn’t touch the regressions.
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

from nyfl.county_labels import fips_with_label, label_for_fips, load_fips_label_map
from nyfl.paths import DATA_PROCESSED, OUTPUTS

OUT_DIR = OUTPUTS.parent / "tables"


def _attach_labels(df: pd.DataFrame, fips_col: str, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out[fips_col] = out[fips_col].astype(str).str.zfill(5)
    out["county_name"] = out[fips_col].map(lambda f: label_for_fips(f, mapping))
    out["county_name_with_fips"] = out[fips_col].map(lambda f: fips_with_label(f, mapping))
    return out


def main() -> None:
    """Roll up dyads/covariates into summary tables and a few extra plots."""
    # Sums / bins / top-k for slides or the appendix.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    flows = pd.read_csv(DATA_PROCESSED / "ny_fl_flows_2122.csv", dtype={"origin_fips5": str, "dest_fips5": str})
    dyad = pd.read_csv(DATA_PROCESSED / "ny_fl_dyadic_2122.csv", dtype={"origin_fips5": str, "dest_fips5": str})
    cov_path = DATA_PROCESSED / "county_covariates_ny_fl.csv"
    mapping = load_fips_label_map(cov_path if Path(cov_path).is_file() else None)
    cov = pd.read_csv(cov_path, dtype={"county_fips5": str}) if Path(cov_path).is_file() else pd.DataFrame()

    total_ret = float(flows["n_returns"].sum())
    total_ex = float(flows["n_individuals"].sum()) if "n_individuals" in flows.columns else float("nan")
    w_agi = (
        float((flows["mean_agi"] * flows["n_returns"]).sum() / flows["n_returns"].sum())
        if "mean_agi" in flows.columns
        else float("nan")
    )
    hh = total_ex / total_ret if total_ret and not np.isnan(total_ex) else float("nan")

    summary = pd.DataFrame(
        [
            {"metric": "NY→FL county-pair flows (rows)", "value": len(flows)},
            {"metric": "Total returns (sum)", "value": total_ret},
            {"metric": "Mean returns per pair", "value": flows["n_returns"].mean()},
            {"metric": "Dyadic rows", "value": len(dyad)},
            {"metric": "Flow-weighted mean AGI ($)", "value": w_agi},
            {"metric": "Total exemptions (individuals, sum)", "value": total_ex},
            {"metric": "Average household size (exemptions / returns)", "value": hh},
        ]
    )
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    top = (
        flows.groupby("dest_fips5", as_index=False)["n_returns"]
        .sum()
        .sort_values("n_returns", ascending=False)
        .head(15)
    )
    top = _attach_labels(top, "dest_fips5", mapping)
    top.to_csv(OUT_DIR / "top15_fl_destinations.csv", index=False)

    if {"mean_agi", "n_returns", "n_individuals", "dest_fips5"}.issubset(flows.columns):
        f2 = flows.copy()
        f2["_wx"] = f2["mean_agi"] * f2["n_returns"]
        g = f2.groupby("dest_fips5", as_index=False).agg(
            total_returns=("n_returns", "sum"),
            total_exemptions=("n_individuals", "sum"),
            sum_wx=("_wx", "sum"),
        )
        g["flow_weighted_mean_agi"] = g["sum_wx"] / g["total_returns"].replace(0, np.nan)
        g = g.drop(columns=["sum_wx"]).sort_values("flow_weighted_mean_agi", ascending=False).head(15)
        g = _attach_labels(g, "dest_fips5", mapping)
        g["avg_household_size"] = g["total_exemptions"] / g["total_returns"].replace(0, np.nan)
        g.to_csv(OUT_DIR / "top15_fl_by_income.csv", index=False)

        plot_df = g.sort_values("flow_weighted_mean_agi", ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        y = np.arange(len(plot_df))
        norm = plt.Normalize(plot_df["flow_weighted_mean_agi"].min(), plot_df["flow_weighted_mean_agi"].max())
        colors = plt.cm.magma(norm(plot_df["flow_weighted_mean_agi"].values))
        ax.barh(y, plot_df["total_returns"], color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["county_name"])
        ax.set_xlabel("Total IRS returns (NY→FL into county)")
        ax.set_title("Top Florida destinations by flow-weighted mean AGI (length = volume, color = income)")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "top15_fl_income_profile.png", dpi=200)
        plt.close(fig)

        f3 = flows.copy()
        f3["_wx"] = f3["mean_agi"] * f3["n_returns"]
        dest_agi = f3.groupby("dest_fips5", as_index=False).agg(
            total_returns=("n_returns", "sum"),
            sum_wx=("_wx", "sum"),
        )
        dest_agi["fw_mean_agi"] = dest_agi["sum_wx"] / dest_agi["total_returns"].replace(0, np.nan)
        dest_agi = dest_agi.drop(columns=["sum_wx"])
        dest_agi["dest_fips5"] = dest_agi["dest_fips5"].astype(str).str.zfill(5)
        if not cov.empty and "dem_two_party_share" in cov.columns:
            cov2 = cov[["county_fips5", "dem_two_party_share"]].copy()
            cov2 = cov2.rename(columns={"county_fips5": "dest_fips5", "dem_two_party_share": "d_dem_two_party_share"})
            cov2["dest_fips5"] = cov2["dest_fips5"].astype(str).str.zfill(5)
            dest_agi = dest_agi.merge(cov2, on="dest_fips5", how="left")
        elif "d_dem_two_party_share" in dyad.columns:
            ddy = dyad[["dest_fips5", "d_dem_two_party_share"]].drop_duplicates(subset=["dest_fips5"])
            ddy["dest_fips5"] = ddy["dest_fips5"].astype(str).str.zfill(5)
            dest_agi = dest_agi.merge(ddy, on="dest_fips5", how="left")
        dest_agi["county_name"] = dest_agi["dest_fips5"].map(lambda f: label_for_fips(f, mapping))
        sub = dest_agi.dropna(subset=["d_dem_two_party_share", "fw_mean_agi"])
        fig, ax = plt.subplots(figsize=(8, 6))
        if not sub.empty:
            mx = float(sub["total_returns"].max())
            sz = np.clip(sub["total_returns"].values / mx * 800.0, 30.0, 800.0)
            ax.scatter(
                sub["d_dem_two_party_share"],
                sub["fw_mean_agi"],
                s=sz,
                alpha=0.55,
                c="#1f77b4",
                edgecolors="k",
                linewidths=0.3,
            )
            thr = sub["total_returns"].quantile(0.85)
            for _, r in sub.iterrows():
                if r["total_returns"] >= thr:
                    ax.annotate(
                        r["county_name"],
                        (r["d_dem_two_party_share"], r["fw_mean_agi"]),
                        fontsize=7,
                        xytext=(4, 4),
                        textcoords="offset points",
                    )
        ax.set_xlabel("2020 Democratic two-party vote share (destination county)")
        ax.set_ylabel("Flow-weighted mean AGI ($)")
        ax.set_title("High-income movers vs destination partisanship (dot size ∝ returns)")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "agi_vs_dest_partisan.png", dpi=200)
        plt.close(fig)

    if "origin_fips5" in flows.columns and "mean_agi" in flows.columns:
        f4 = flows.copy()
        f4["_wx"] = f4["mean_agi"] * f4["n_returns"]
        if "n_individuals" in flows.columns:
            ny = f4.groupby("origin_fips5", as_index=False).agg(
                total_returns=("n_returns", "sum"),
                total_exemptions=("n_individuals", "sum"),
                sum_wx=("_wx", "sum"),
            )
        else:
            ny = f4.groupby("origin_fips5", as_index=False).agg(
                total_returns=("n_returns", "sum"),
                sum_wx=("_wx", "sum"),
            )
            ny["total_exemptions"] = np.nan
        ny["flow_weighted_mean_agi"] = ny["sum_wx"] / ny["total_returns"].replace(0, np.nan)
        ny = ny.drop(columns=["sum_wx"]).sort_values("total_returns", ascending=False).head(10)
        ny["share_of_corridor_returns"] = ny["total_returns"] / total_ret
        ny = _attach_labels(ny, "origin_fips5", mapping)
        ny.to_csv(OUT_DIR / "top10_ny_origins.csv", index=False)

        plot_ny = ny.sort_values("total_returns", ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        yn = np.arange(len(plot_ny))
        norm = plt.Normalize(plot_ny["flow_weighted_mean_agi"].min(), plot_ny["flow_weighted_mean_agi"].max())
        ccolors = plt.cm.magma(norm(plot_ny["flow_weighted_mean_agi"].values))
        ax.barh(yn, plot_ny["total_returns"], color=ccolors)
        ax.set_yticks(yn)
        ax.set_yticklabels(plot_ny["county_name"])
        ax.set_xlabel("Total IRS returns sent to Florida")
        ax.set_title("Top New York origin counties (color = flow-weighted mean AGI)")
        fig.tight_layout()
        fig.savefig(OUTPUTS / "top10_ny_origins.png", dpi=200)
        plt.close(fig)

    print(f"Wrote {OUT_DIR / 'summary_metrics.csv'}, destination / income / NY-origin exports.")


if __name__ == "__main__":
    main()
