"""IRS county-to-county migration helpers (2021–2022 filing-year data, file countyoutflow2122.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NY_STATEFIPS = "36"
FL_STATEFIPS = "12"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def load_county_outflow(path: str | Path) -> pd.DataFrame:
    """Read countyoutflow and upper-case the headers. We try a few encodings because IRS files aren’t always UTF-8."""
    p = Path(path)
    last_err: Exception | None = None
    df = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False, encoding=encoding)
            break
        except UnicodeDecodeError as e:
            last_err = e
    if df is None:
        assert last_err is not None
        raise last_err
    df = normalize_columns(df)
    required = {
        "Y1_STATEFIPS",
        "Y1_COUNTYFIPS",
        "Y2_STATEFIPS",
        "Y2_COUNTYFIPS",
        "N1",
        "N2",
        "AGI",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {p}")
    return df


def filter_ny_to_fl(df: pd.DataFrame) -> pd.DataFrame:
    """Keep NY→FL rows with real county codes (skip 000 state buckets) and positive flows."""
    d = df.copy()
    for col in ("Y1_STATEFIPS", "Y1_COUNTYFIPS", "Y2_STATEFIPS", "Y2_COUNTYFIPS"):
        d[col] = d[col].astype(str).str.strip().str.zfill(3 if "COUNTY" in col else 2)
    # State totals and non-county destinations use Y*_COUNTYFIPS == 000 — exclude those rows.
    mask = (
        (d["Y1_STATEFIPS"] == NY_STATEFIPS)
        & (d["Y2_STATEFIPS"] == FL_STATEFIPS)
        & (d["Y1_COUNTYFIPS"] != "000")
        & (d["Y2_COUNTYFIPS"] != "000")
    )
    out = d.loc[mask].copy()
    # N1 = number of returns in the flow; N2 = exemptions (people); AGI = total adjusted gross income, thousands of $.
    out["n_returns"] = pd.to_numeric(out["N1"], errors="coerce")
    out["n_individuals"] = pd.to_numeric(out["N2"], errors="coerce")
    out["agi_thousands"] = pd.to_numeric(out["AGI"], errors="coerce")
    # Suppressed cells are -1
    out = out[out["n_returns"] > 0]
    # 5-digit FIPS: 2-digit state + 3-digit county (already zero-padded above).
    out["origin_fips5"] = out["Y1_STATEFIPS"] + out["Y1_COUNTYFIPS"]
    out["dest_fips5"] = out["Y2_STATEFIPS"] + out["Y2_COUNTYFIPS"]
    # Mean AGI per return ($); AGI field is total group AGI in thousands of dollars (IRS guide).
    out["mean_agi"] = (out["agi_thousands"] * 1000.0) / out["n_returns"]
    return out
