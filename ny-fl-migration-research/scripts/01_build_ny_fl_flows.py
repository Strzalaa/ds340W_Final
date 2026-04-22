#!/usr/bin/env python3
"""
Script 01 — Pull IRS county outflows and keep NY → FL county pairs only.

Loads countyoutflow (download, env path, or the tiny sample). Drops state-level “000” county codes so we’re
left with real origin/dest counties. Writes `ny_fl_flows_2122.csv`; later scripts attach covariates and merge.

`n_returns` is the flow size we use downstream — one row per directed county pair.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import urllib.request

from nyfl.irs import filter_ny_to_fl, load_county_outflow
from nyfl.paths import DATA_PROCESSED, DATA_RAW, DATA_SAMPLE

IRS_URL = "https://www.irs.gov/pub/irs-soi/countyoutflow2122.csv"
DEFAULT_RAW = DATA_RAW / "countyoutflow2122.csv"
SAMPLE = DATA_SAMPLE / "countyoutflow2122_sample.csv"


def download_file(url: str, dest: Path) -> None:
    """Grab the IRS CSV into data/raw/ if you don’t already have it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    # Some servers block default Python user-agents; a custom UA avoids a 403 on the IRS host.
    req = urllib.request.Request(url, headers={"User-Agent": "ny-fl-migration-research/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        dest.write_bytes(resp.read())


def main() -> None:
    """Wire up the raw file, filter to NY→FL, save the processed flows CSV."""
    # Sample mode: tiny bundled CSV for offline grading; production runs must use full IRS file.
    use_sample = os.environ.get("USE_SAMPLE_IRS", "").lower() in ("1", "true", "yes")
    raw_path = Path(os.environ.get("IRS_COUNTY_OUTFLOW_CSV", DEFAULT_RAW))

    if use_sample or os.environ.get("FORCE_SAMPLE"):
        raw_path = SAMPLE
        print(f"Using sample IRS extract: {raw_path}")
    elif not raw_path.exists():
        try:
            download_file(IRS_URL, DEFAULT_RAW)
            raw_path = DEFAULT_RAW
        except Exception as e:
            print(f"Download failed ({e}). Falling back to bundled sample.")
            raw_path = SAMPLE

    # Normalize column names to uppercase and parse IRS fields (see nyfl.irs).
    df = load_county_outflow(raw_path)
    # One row per origin county → destination county pair, NY→FL only, positive flows.
    ny_fl = filter_ny_to_fl(df)
    out = DATA_PROCESSED / "ny_fl_flows_2122.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ny_fl.to_csv(out, index=False)
    print(f"Wrote {len(ny_fl)} rows to {out}")
    print(ny_fl.head())


if __name__ == "__main__":
    main()
