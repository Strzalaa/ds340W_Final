"""Fetch county-level ACS 5-year estimates via api.census.gov (no key required for many endpoints)."""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

# 2022 ACS 5-year (aligns with migration discussion; adjust if needed)
ACS_YEAR = 2022
ACS_DATASET = "acs/acs5"

# B25077 median home value; B25064 median gross rent; B01003 population;
# B19013 median HH income; B03002 race; B25003 tenure (owner / renter counts)
ACS_VARS = (
    "NAME,B25077_001E,B25064_001E,B01003_001E,B19013_001E,B03002_003E,B03002_001E,"
    "B25003_001E,B25003_002E,B25003_003E"
)


def fetch_acs_counties_state(state_fips: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Hit the Census API for every county in one state (NY = 36, FL = 12). Year lives in ACS_YEAR up top."""
    sess = session or requests.Session()
    key = os.environ.get("CENSUS_API_KEY", "")
    base = f"https://api.census.gov/data/{ACS_YEAR}/{ACS_DATASET}"
    url = f"{base}?get={ACS_VARS}&for=county:*&in=state:{state_fips}"
    if key:
        url += f"&key={key}"
    r = sess.get(url, timeout=120)
    r.raise_for_status()
    rows = r.json()
    header, *body = rows
    df = pd.DataFrame(body, columns=header)
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["county_fips5"] = df["state"] + df["county"]
    rename = {
        "B25077_001E": "median_home_value",
        "B25064_001E": "median_gross_rent",
        "B01003_001E": "population",
        "B19013_001E": "median_hh_income",
        "B03002_003E": "nh_white",
        "B03002_001E": "total_race_eth",
        "B25003_001E": "housing_units_occupied",
        "B25003_002E": "owner_occupied_units",
        "B25003_003E": "renter_occupied_units",
    }
    df = df.rename(columns=rename)
    for c in rename.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "nh_white" in df.columns and "total_race_eth" in df.columns:
        df["pct_nh_white"] = df["nh_white"] / df["total_race_eth"].replace(0, float("nan"))
    if all(c in df.columns for c in ("owner_occupied_units", "housing_units_occupied")):
        ho = pd.to_numeric(df["housing_units_occupied"], errors="coerce").replace(0, float("nan"))
        oo = pd.to_numeric(df["owner_occupied_units"], errors="coerce")
        df["pct_owner_occupied"] = oo / ho
    if all(c in df.columns for c in ("owner_occupied_units", "renter_occupied_units")):
        ro = pd.to_numeric(df["renter_occupied_units"], errors="coerce").replace(0, float("nan"))
        oo = pd.to_numeric(df["owner_occupied_units"], errors="coerce")
        df["owner_to_renter_ratio"] = oo / ro
    if "median_gross_rent" in df.columns and "median_hh_income" in df.columns:
        inc = pd.to_numeric(df["median_hh_income"], errors="coerce").replace(0, float("nan"))
        rent = pd.to_numeric(df["median_gross_rent"], errors="coerce")
        # rent burden–style index (higher = more rent pressure relative to income)
        df["affordability_index"] = rent / inc
    time.sleep(0.3)  # be polite
    return df


def load_gazetteer(path: str) -> pd.DataFrame:
    """Census 2020 Gazetteer counties national file (tab-separated)."""
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1")
    cols = {c.upper(): c for c in df.columns}
    # GEOID is often 5-char county FIPS
    if "GEOID" in df.columns:
        df["county_fips5"] = df["GEOID"].astype(str).str.zfill(5)
    elif "USPS" in df.columns:
        # alternate layout
        pass
    latcol = "INTPTLAT" if "INTPTLAT" in df.columns else None
    loncol = "INTPTLONG" if "INTPTLONG" in df.columns else None
    if latcol:
        df["lat"] = pd.to_numeric(df[latcol], errors="coerce")
        df["lon"] = pd.to_numeric(df[loncol], errors="coerce")
    return df


def fetch_gazetteer(dest_path: str, session: requests.Session | None = None) -> None:
    """Download 2020 county gazetteer. Census publishes a .zip (the old .txt URL 404s)."""
    import io
    import zipfile
    from pathlib import Path

    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip"
    sess = session or requests.Session()
    r = sess.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        txts = [n for n in z.namelist() if n.lower().endswith(".txt")]
        if not txts:
            raise RuntimeError(f"No .txt in gazetteer zip: {z.namelist()}")
        data = z.read(txts[0])
    p = Path(dest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
