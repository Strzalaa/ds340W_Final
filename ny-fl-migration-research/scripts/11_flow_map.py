#!/usr/bin/env python3
"""
Script 11 — Simple Florida map: each county colored by total NY→FL inflow (IRS 2021–22).

Just the one state so we’re not fighting matplotlib across half the East Coast. Needs geopandas.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patheffects as pe
from urllib.request import urlretrieve

from nyfl.county_labels import label_for_fips, load_fips_label_map
from nyfl.paths import DATA_PROCESSED, DATA_RAW, OUTPUTS

DYADIC = DATA_PROCESSED / "ny_fl_dyadic_2122.csv"
COV_CSV = DATA_PROCESSED / "county_covariates_ny_fl.csv"
COUNTY_ZIP_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
COUNTY_ZIP_CACHE = DATA_RAW / "cb_2022_us_county_20m.zip"


def _ensure_county_shapes() -> Path:
    COUNTY_ZIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not COUNTY_ZIP_CACHE.is_file():
        print(f"Downloading county boundaries:\n  {COUNTY_ZIP_URL}")
        urlretrieve(COUNTY_ZIP_URL, COUNTY_ZIP_CACHE)
    return COUNTY_ZIP_CACHE


def _read_counties_gdf(zpath: Path):
    import geopandas as gpd

    with zipfile.ZipFile(zpath, "r") as zf:
        shps = [n for n in zf.namelist() if n.endswith(".shp")]
        if not shps:
            raise RuntimeError("No .shp inside county zip")
        member = shps[0]
    return gpd.read_file(f"zip://{zpath}!{member}")


def main() -> None:
    """Florida-only choropleth: total NY→FL inflow per destination county."""
    if not DYADIC.is_file():
        print(f"Missing {DYADIC}; run pipeline first.")
        sys.exit(1)

    dy = pd.read_csv(DYADIC, dtype={"dest_fips5": str})
    if "n_returns" not in dy.columns or "dest_fips5" not in dy.columns:
        print("Dyadic file needs dest_fips5 and n_returns.")
        sys.exit(1)

    zpath = _ensure_county_shapes()
    counties = _read_counties_gdf(zpath)
    counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)

    mapping = load_fips_label_map(COV_CSV if Path(COV_CSV).is_file() else None)

    fl = counties[counties["STATEFP"] == "12"].copy()

    dest_tot = dy.groupby("dest_fips5", as_index=False)["n_returns"].sum()
    dest_tot["dest_fips5"] = dest_tot["dest_fips5"].astype(str).str.zfill(5)
    dest_tot = dest_tot.rename(columns={"n_returns": "dest_inflow_returns"})
    fl = fl.merge(dest_tot, left_on="GEOID", right_on="dest_fips5", how="left").drop(columns=["dest_fips5"], errors="ignore")
    fl["total_returns"] = fl["dest_inflow_returns"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    fl.plot(
        ax=ax,
        column="total_returns",
        cmap="YlOrRd",
        legend=True,
        legend_kwds={"label": "NY→FL returns (destination county)", "shrink": 0.7},
        edgecolor="#333333",
        linewidth=0.35,
        missing_kwds={"color": "#f0f0f0", "edgecolor": "#cccccc"},
    )

    bfl = fl.total_bounds
    pad_lon, pad_lat = 0.5, 0.5
    ax.set_xlim(bfl[0] - pad_lon, bfl[2] + pad_lon)
    ax.set_ylim(bfl[1] - pad_lat, bfl[3] + pad_lat)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Top destinations by inflow — label at centroid (small offsets for SE FL cluster)
    top5 = dest_tot.sort_values("dest_inflow_returns", ascending=False).head(5)
    fl_idx = fl.set_index("GEOID")
    fl_offsets = {
        "12011": (0, -0.12),
        "12086": (0.06, 0.04),
    }
    for _, r in top5.iterrows():
        f5 = str(r["dest_fips5"]).zfill(5)
        if f5 not in fl_idx.index:
            continue
        c = fl_idx.loc[f5, "geometry"].centroid
        name = label_for_fips(f5, mapping)
        ox, oy = fl_offsets.get(f5, (0, 0))
        ax.annotate(
            name,
            (c.x + ox, c.y + oy),
            fontsize=9,
            fontweight="bold",
            ha="center",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )

    ax.set_title(
        "New York → Florida migration (2021–2022 IRS flows)\n"
        "Florida counties shaded by total inflow from New York (returns)",
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUTS / "ny_fl_flow_map.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
