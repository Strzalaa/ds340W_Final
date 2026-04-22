#!/usr/bin/env python3
"""
Script 02 — Build one row per NY/FL county: ACS housing/demographics, gazetteer lat/lon, election if you have it.

Either hit the Census API or, in sample mode, read the bundled CSV. Writes `county_covariates_ny_fl.csv`; 03 joins
this twice (origin / destination) when it builds dyads. Election file is optional but nice to have for politics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from nyfl.census_acs import fetch_acs_counties_state, fetch_gazetteer
from nyfl.paths import DATA_PROCESSED, DATA_RAW, DATA_SAMPLE

GAZ = DATA_RAW / "2020_Gaz_counties_national.txt"
ELECTION_PREFERRED = DATA_RAW / "county_election_2020.csv"
ELECTION_SAMPLE = DATA_SAMPLE / "county_election_2020_ny_fl.csv"
ACS_SAMPLE = DATA_SAMPLE / "county_acs_2022_ny_fl.csv"


def _discover_election_csv(raw: Path) -> Path | None:
    """Resolve election file: env override, county_election_2020.csv, or county / countypres* names in raw/."""
    env = (os.environ.get("COUNTY_ELECTION_CSV") or os.environ.get("ELECTION_CSV") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
        print(f"COUNTY_ELECTION_CSV set but not found: {p}")
    if ELECTION_PREFERRED.is_file():
        return ELECTION_PREFERRED
    matches: list[Path] = []
    for p in raw.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.startswith("countyoutflow") or low.startswith("countyinflow"):
            continue
        suf = p.suffix.lower()
        stem = Path(p.name).stem.lower().replace(" ", "_").replace("-", "_")
        # e.g. countypres_2000-2024.csv → stem countypres_2000_2024; also allow extensionless countypres…
        looks_election = (
            stem.startswith("countypres")
            or ("county" in stem and "pres" in stem)
            or (not suf and low.startswith("countypres"))
        )
        if not looks_election:
            continue
        if suf not in (".csv", ".tsv") and not (not suf and low.startswith("countypres")):
            continue
        matches.append(p)
    matches = sorted(set(matches))
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        print(f"Multiple election CSVs in {raw} ({names}). Using {matches[0].name}. Prefer county_election_2020.csv.")
    return matches[0]


def _fips_series_to_5(s: pd.Series) -> pd.Series:
    out: list[str | float] = []
    for v in s:
        if pd.isna(v):
            out.append(np.nan)
            continue
        st = str(v).strip()
        if st.endswith(".0"):
            st = st[:-2]
        if st.isdigit() and len(st) > 5:
            st = st[-5:]
        try:
            iv = int(round(float(st)))
            out.append(str(iv).zfill(5))
        except ValueError:
            out.append(np.nan)
    return pd.Series(out, index=s.index, dtype="object")


def _read_election_table(path: Path) -> pd.DataFrame:
    """Load CSV/TSV; tolerate common encodings."""
    kwargs: dict = {"low_memory": False}
    if path.suffix.lower() == ".tsv":
        kwargs["sep"] = "\t"
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding_errors="replace", **kwargs)


def _prepare_election_df(el: pd.DataFrame) -> pd.DataFrame:
    """Return columns county_fips5, dem_two_party_share from common MIT / wide / long layouts."""
    el = el.copy()
    lower = {c.lower().strip(): c for c in el.columns}
    if "year" in lower:
        ye = pd.to_numeric(el[lower["year"]], errors="coerce")
        el_y = el.loc[ye == 2020].copy()
        if el_y.empty:
            raise ValueError(
                "Election file has a year column but no rows for 2020. "
                "This project merges 2020 presidential two-party share; filter the source to 2020 or fix the year column."
            )
        el = el_y
        lower = {c.lower().strip(): c for c in el.columns}

    def col(*names: str) -> pd.Series | None:
        for n in names:
            if n.lower() in lower:
                return el[lower[n.lower()]]
        return None

    # --- Long format (party + candidatevotes), e.g. MIT county returns ---
    cand_col = col("candidatevotes", "votes")
    party_col = col("party")
    if cand_col is not None and party_col is not None:
        fips_raw = col("county_fips5", "fips", "county_fips", "geoid", "countyfips")
        if fips_raw is not None:
            t = pd.DataFrame(
                {
                    "_f": _fips_series_to_5(fips_raw),
                    "_v": pd.to_numeric(cand_col, errors="coerce").fillna(0),
                    "_p": party_col.astype(str).str.upper(),
                }
            )
            if "year" in lower:
                y = pd.to_numeric(el[lower["year"]], errors="coerce")
                t_y = t.loc[y == 2020].copy()
                if not t_y.empty:
                    t = t_y
            dem_mask = t["_p"].str.contains("DEM", na=False) & ~t["_p"].str.contains("REP", na=False)
            rep_mask = t["_p"].str.contains("REP", na=False)
            dem = t.loc[dem_mask].groupby("_f")["_v"].sum()
            rep = t.loc[rep_mask].groupby("_f")["_v"].sum()
            rep = rep.reindex(dem.index).fillna(0)
            share = dem / (dem + rep).replace(0, np.nan)
            out = pd.DataFrame(
                {"county_fips5": share.index.map(lambda x: str(x).split(".")[0]), "dem_two_party_share": share.values}
            )
            out = out.dropna(subset=["dem_two_party_share"])
            if not out.empty and out["dem_two_party_share"].notna().any():
                return out

    # --- Wide format: FIPS + share or vote columns ---
    fips_raw = col("county_fips5", "fips", "county_fips", "geoid", "countyfips")
    if fips_raw is None:
        for c in el.columns:
            if c.upper() == "GEOID":
                fips_raw = el[c]
                break
    if fips_raw is None:
        raise ValueError(
            "Election CSV needs a FIPS column (county_fips5, FIPS, GEOID, …). "
            "See README.md (data/raw election CSV formats)."
        )

    share_s = col("dem_two_party_share", "d_share", "pct_dem", "per_dem", "dem_share", "p_dem")
    if share_s is None:
        vd = col("votes_dem", "dem_votes", "dem2020", "g20dem")
        vr = col("votes_rep", "rep_votes", "rep2020", "g20rep")
        if vd is not None and vr is not None:
            vd = pd.to_numeric(vd, errors="coerce")
            vr = pd.to_numeric(vr, errors="coerce")
            share_s = vd / (vd + vr).replace(0, np.nan)
    if share_s is None:
        raise ValueError(
            "Election CSV needs dem_two_party_share (or d_share), or votes_dem + votes_rep, "
            "or party + candidatevotes columns. See README.md."
        )

    out = pd.DataFrame(
        {
            "county_fips5": _fips_series_to_5(fips_raw),
            "dem_two_party_share": pd.to_numeric(share_s, errors="coerce"),
        }
    )
    out = out.dropna(subset=["county_fips5", "dem_two_party_share"])
    out = out.drop_duplicates("county_fips5", keep="last")
    return out


def _merge_gazetteer_land_lat_lon(acs: pd.DataFrame, gaz_path: Path) -> pd.DataFrame:
    """Join gazetteer centroids + land area so we get distance in 03 and a rough density measure."""
    # Gazetteer gives centroids + land area; we use density as a rough "rural vs urban" handle.
    if not gaz_path.exists():
        return acs
    gaz = pd.read_csv(gaz_path, sep="\t", encoding="latin-1", dtype=str)
    if "GEOID" not in gaz.columns:
        return acs
    gaz["county_fips5"] = gaz["GEOID"].astype(str).str.zfill(5)
    gaz["lat"] = pd.to_numeric(gaz.get("INTPTLAT"), errors="coerce")
    gaz["lon"] = pd.to_numeric(gaz.get("INTPTLONG"), errors="coerce")
    if "ALAND" in gaz.columns:
        aland = pd.to_numeric(gaz["ALAND"], errors="coerce")
        # ALAND is square meters; convert to square miles
        gaz["land_sqmi"] = aland / 2.589988110336e6
    keep = ["county_fips5", "lat", "lon"] + (["land_sqmi"] if "land_sqmi" in gaz.columns else [])
    acs = acs.merge(gaz[keep].drop_duplicates("county_fips5"), on="county_fips5", how="left")
    if "land_sqmi" in acs.columns and "population" in acs.columns:
        acs["pop_density"] = acs["population"] / acs["land_sqmi"].replace(0, np.nan)
        acs["log_pop_density"] = np.log1p(acs["pop_density"].clip(lower=0))
    return acs


def main() -> None:
    """Rip ACS (or sample), tack on gazetteer + election when possible, save the covariate CSV."""
    # Bundled tiny extracts: fast CI / no Census key; full pipeline should unset USE_SAMPLE_COVARIATES.
    use_sample = os.environ.get("USE_SAMPLE_COVARIATES", "").lower() in ("1", "true", "yes")
    if use_sample:
        df = pd.read_csv(ACS_SAMPLE, dtype={"county_fips5": str})
        df["county_fips5"] = df["county_fips5"].str.zfill(5)
        el = pd.read_csv(ELECTION_SAMPLE, dtype={"county_fips5": str})
        el["county_fips5"] = el["county_fips5"].str.zfill(5)
        out = df.merge(el[["county_fips5", "dem_two_party_share"]], on="county_fips5", how="left")
        out = _merge_gazetteer_land_lat_lon(out, GAZ)
        if "land_sqmi" not in out.columns or out["land_sqmi"].isna().all():
            out["pop_density"] = out["population"] / 500.0
            out["log_pop_density"] = np.log1p(out["pop_density"].clip(lower=0))
            print("Sample mode: no gazetteer land — using rough pop_density = population/500.")
        path = DATA_PROCESSED / "county_covariates_ny_fl.csv"
        out.to_csv(path, index=False)
        print(f"Wrote sample covariates ({len(out)} counties) to {path}")
        return

    session = __import__("requests").Session()
    parts = []
    # Census state FIPS: 36 = New York, 12 = Florida — all counties in each state in one API pull per state.
    for st in ("36", "12"):
        print(f"Fetching ACS for state {st} ...")
        parts.append(fetch_acs_counties_state(st, session=session))
    acs = pd.concat(parts, ignore_index=True)

    if not GAZ.exists():
        try:
            fetch_gazetteer(str(GAZ), session=session)
        except Exception as e:
            print(f"Gazetteer download failed: {e}. Lat/lon will be missing.")

    if GAZ.exists():
        acs = _merge_gazetteer_land_lat_lon(acs, GAZ)

    elec_path = _discover_election_csv(DATA_RAW)
    if elec_path is not None:
        try:
            el_raw = _read_election_table(elec_path)
            el = _prepare_election_df(el_raw)
            el["county_fips5"] = el["county_fips5"].astype(str).str.strip().str.zfill(5)
            if "dem_two_party_share" in acs.columns:
                acs = acs.drop(columns=["dem_two_party_share"])
            acs = acs.merge(el[["county_fips5", "dem_two_party_share"]], on="county_fips5", how="left")
            nmiss = int(acs["dem_two_party_share"].isna().sum())
            print(f"Merged election from {elec_path.name} — dem_two_party_share missing for {nmiss} / {len(acs)} counties.")
        except Exception as e:
            print(f"Could not parse election file {elec_path}: {e}")
            print("Falling back to sample election subset. Add a full county presidential CSV under data/raw/ (README).")
            el = pd.read_csv(ELECTION_SAMPLE, dtype={"county_fips5": str})
            el["county_fips5"] = el["county_fips5"].str.zfill(5)
            if "dem_two_party_share" in acs.columns:
                acs = acs.drop(columns=["dem_two_party_share"])
            acs = acs.merge(el[["county_fips5", "dem_two_party_share"]], on="county_fips5", how="left")
    else:
        print(
            f"No election CSV in {DATA_RAW} (expected county_election_2020.csv or a file with "
            f"'county' and 'pres' in the name, e.g. County Presidential.csv). "
            "See README.md (election CSV in data/raw/). Using sample election for NY/FL subset only."
        )
        el = pd.read_csv(ELECTION_SAMPLE, dtype={"county_fips5": str})
        el["county_fips5"] = el["county_fips5"].str.zfill(5)
        if "dem_two_party_share" in acs.columns:
            acs = acs.drop(columns=["dem_two_party_share"])
        acs = acs.merge(el[["county_fips5", "dem_two_party_share"]], on="county_fips5", how="left")

    path = DATA_PROCESSED / "county_covariates_ny_fl.csv"
    acs.to_csv(path, index=False)
    print(f"Wrote {len(acs)} county rows to {path}")


if __name__ == "__main__":
    main()
