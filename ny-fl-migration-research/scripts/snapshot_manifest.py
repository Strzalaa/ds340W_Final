#!/usr/bin/env python3
"""
Reproducibility manifest: SHA-256 hashes of key processed and raw inputs.

Run at end of the pipeline (run_all.sh). Writes data/SUBMISSION_MANIFEST.txt so reviewers can
verify which files a given results bundle was built from.
Hashes + env vars won’t fix your model — they just prove what you ran against.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "SUBMISSION_MANIFEST.txt"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.lower() in ("1", "true", "yes")


def _discover_election_for_manifest(raw: Path) -> Path | None:
    """Same resolution order as scripts/02_fetch_county_covariates._discover_election_csv."""
    env = (os.environ.get("COUNTY_ELECTION_CSV") or os.environ.get("ELECTION_CSV") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
    preferred = raw / "county_election_2020.csv"
    if preferred.is_file():
        return preferred
    matches: list[Path] = []
    for p in raw.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.startswith("countyoutflow") or low.startswith("countyinflow"):
            continue
        suf = p.suffix.lower()
        stem = Path(p.name).stem.lower().replace(" ", "_").replace("-", "_")
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
    return matches[0] if matches else None


def main() -> None:
    """Dump timestamps, env, and file hashes into SUBMISSION_MANIFEST.txt."""
    # If two people’s hashes don’t match, check sample flags and backends before arguing about results.
    use_irs = os.environ.get("USE_SAMPLE_IRS", "")
    use_cov = os.environ.get("USE_SAMPLE_COVARIATES", "")
    housing_b = os.environ.get("HOUSING_BACKEND", "")
    sample_mode = _truthy(use_irs) or _truthy(use_cov)
    data_mode = "sample" if sample_mode else "production"
    lines = [
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "=== pipeline environment (at manifest time) ===",
        f"USE_SAMPLE_IRS={use_irs!r}",
        f"USE_SAMPLE_COVARIATES={use_cov!r}",
        f"HOUSING_BACKEND={housing_b!r}",
        f"data_mode={data_mode}",
        "",
        "=== data/processed ===",
    ]
    # Every processed artifact gets a content hash — if you change code and re-run, hashes update.
    if PROCESSED.exists():
        for p in sorted(PROCESSED.glob("*.csv")) + sorted(PROCESSED.glob("*.txt")):
            if p.is_file():
                stat = p.stat()
                lines.append(f"{p.name}\tbytes={stat.st_size}\tsha256={sha256_file(p)}")
    lines.append("")
    lines.append("=== data/raw (if present) ===")
    for name in ("countyoutflow2122.csv", "2020_Gaz_counties_national.txt"):
        p = RAW / name
        if p.is_file():
            lines.append(f"{name}\tbytes={p.stat().st_size}\tsha256={sha256_file(p)}")
        else:
            lines.append(f"{name}\t(missing)")
    elec = _discover_election_for_manifest(RAW)
    if elec is not None:
        st = elec.stat()
        lines.append(
            f"{elec.name}\tbytes={st.st_size}\tsha256={sha256_file(elec)}\t"
            f"(county presidential merge source for script 02; canonical name is county_election_2020.csv)"
        )
    else:
        lines.append(
            "county_presidential_election\t(missing — add data/raw/county_election_2020.csv "
            "or countypres*.csv / *county*pres*.csv; see README.md)"
        )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
