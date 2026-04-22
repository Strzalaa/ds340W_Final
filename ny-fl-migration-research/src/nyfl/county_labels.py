"""Turn ACS NAME + FIPS into short strings like 'Palm Beach, FL' for tables and plots.

Purely cosmetic — doesn’t touch coefficients. We also rewrite FE lines in saved summaries so you’re not staring at raw FIPS.
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from nyfl.paths import DATA_PROCESSED

# Full state names as returned by Census ACS NAME suffix (lowercase keys)
_STATE_FULL_TO_ABBR: dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}


def _short_from_acs_name(name: str) -> str:
    """'Palm Beach County, Florida' -> 'Palm Beach, FL'."""
    if not isinstance(name, str) or not name.strip():
        return ""
    parts = [p.strip() for p in name.rsplit(",", 1)]
    if len(parts) != 2:
        return name.strip()
    county_part, state_full = parts
    key = state_full.lower().strip()
    abbr = _STATE_FULL_TO_ABBR.get(key)
    if abbr is None and len(state_full) == 2:
        abbr = state_full.upper()
    if abbr is None:
        abbr = state_full[:2].upper()
    base = county_part
    for suf in (" County", " Borough", " Census Area", " Municipality", " Parish"):
        if len(base) > len(suf) and base.lower().endswith(suf.lower()):
            base = base[: -len(suf)].strip()
            break
    return f"{base}, {abbr}"


def load_fips_label_map(
    covariates_csv: Path | None = None,
) -> dict[str, str]:
    """FIPS → 'County, ST' from the processed covariate CSV (same file 03 uses)."""
    path = covariates_csv or (DATA_PROCESSED / "county_covariates_ny_fl.csv")
    if not path.is_file():
        return {}
    df = pd.read_csv(path, dtype={"county_fips5": str})
    if "county_fips5" not in df.columns:
        return {}
    df["county_fips5"] = df["county_fips5"].astype(str).str.zfill(5)
    if "NAME" not in df.columns:
        return {r["county_fips5"]: r["county_fips5"] for _, r in df.iterrows()}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        f5 = str(row["county_fips5"]).zfill(5)
        lbl = _short_from_acs_name(str(row["NAME"]))
        out[f5] = lbl if lbl else f5
    return out


def label_for_fips(fips5: str | float, mapping: dict[str, str]) -> str:
    s = str(fips5).split(".")[0].zfill(5) if pd.notna(fips5) else ""
    if not s or s == "nan":
        return ""
    return mapping.get(s, f"(unknown county {s})")


def fips_with_label(fips5: str | float, mapping: dict[str, str]) -> str:
    """Always show FIPS next to name: 'Palm Beach, FL (FIPS 12099)'."""
    s = str(fips5).split(".")[0].zfill(5) if pd.notna(fips5) else ""
    lbl = label_for_fips(s, mapping)
    if not s:
        return lbl
    return f"{lbl} (FIPS {s})"


def decorate_statsmodels_summary_fips(text: str, mapping: dict[str, str]) -> str:
    """Replace `C(origin_fips5)[T.xxxxx]` / `C(dest_fips5)[T.xxxxx]` in summary text with county names.

    String hack on the saved text — numbers underneath don’t move.
    """

    def _orig_sub(m: re.Match[str]) -> str:
        f = m.group(1)
        return f"C(origin FE: {label_for_fips(f, mapping)} [{f}])"

    def _dest_sub(m: re.Match[str]) -> str:
        f = m.group(1)
        return f"C(dest FE: {label_for_fips(f, mapping)} [{f}])"

    text = re.sub(r"C\(origin_fips5\)\[T\.(\d{5})\]", _orig_sub, text)
    text = re.sub(r"C\(dest_fips5\)\[T\.(\d{5})\]", _dest_sub, text)
    return text
