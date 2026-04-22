#!/usr/bin/env python3
"""
Script 10 — Tie ML housing backend choice to Layer 2 Poisson fit (downstream deviance).

For each requested backend: subprocess runs 04 with that backend, then 03 to rebuild dyads,
then fits the full Layer 2 Poisson design (same as script 05 via nyfl.layer2_specs) and records deviance.

Outputs:
  - housing_downstream_glm_comparison.csv
  - Merges with housing_ml_backend_comparison.csv (from 04 --compare) when present → housing_novelty_full_comparison.csv
  - ml_backend_tradeoff.png (holdout error vs downstream deviance) when both exist.

run_all.sh puts HOUSING_BACKEND back and re-runs 04+03 afterward so your processed folder isn’t stuck on a comparison backend.

Idea: see if a county model with lower holdout MAE also gives you better Poisson deviance on flows — quick sanity check, not proof of anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson

from nyfl.layer2_specs import build_layer2_design
from nyfl.paths import DATA_PROCESSED, OUTPUTS

DYADIC = DATA_PROCESSED / "ny_fl_dyadic_2122.csv"
COMPARE_ML = OUTPUTS.parent / "tables" / "housing_ml_backend_comparison.csv"
OUT_DOWN = OUTPUTS.parent / "tables" / "housing_downstream_glm_comparison.csv"
OUT_FULL = OUTPUTS.parent / "tables" / "housing_novelty_full_comparison.csv"

DEFAULT_BACKENDS = ("lgbm", "sklearn_gbr", "hist_gbrt")


def _prep(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    if "distance_km" in m.columns:
        m["log_distance_km"] = np.log1p(m["distance_km"].clip(lower=0))
    for c in ["o_population", "d_population"]:
        if c in m.columns:
            m[f"log_{c}"] = np.log1p(m[c].clip(lower=0))
    return m


def glm_poisson_metrics(m: pd.DataFrame) -> dict[str, float | int | str]:
    """Grab deviance / AIC / N from a Poisson fit on whatever dyadic CSV is sitting on disk."""
    # Identical mean structure as script 05 Layer-2 Poisson (offset + z-scored X; no intercept column here).
    design = build_layer2_design(m, full=True)
    if design is None:
        return {"error": "could not build Layer-2 design (need o_population, covariates)"}
    Xc = design.X.astype(float)
    off = np.asarray(design.offset, dtype=float)
    try:
        glm = sm.GLM(design.y, Xc, family=Poisson(), offset=off)
        res = glm.fit()
        return {
            "null_deviance": float(res.null_deviance),
            "deviance": float(res.deviance),
            "aic": float(res.aic),
            "nobs": int(res.nobs),
            "error": "",
        }
    except Exception as e:
        return {"error": str(e)[:500]}


def run_pipeline(backend: str) -> tuple[bool, str]:
    """Call 04 then 03 for one backend so gaps match that model."""
    py = sys.executable
    # Fresh housing predictions for this backend, then rebuild dyads so dyadic_housing_gap matches.
    r04 = subprocess.run(
        [py, str(ROOT / "scripts" / "04_housing_gbdt_county.py"), "--backend", backend],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r04.returncode != 0:
        return False, (r04.stderr or r04.stdout or "04 failed")[-2000:]
    r03 = subprocess.run(
        [py, str(ROOT / "scripts" / "03_merge_and_dyadic.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r03.returncode != 0:
        return False, (r03.stderr or r03.stdout or "03 failed")[-2000:]
    return True, ""


def _backend_pretty(backend: str) -> str:
    b = backend.lower().strip()
    if b == "lgbm":
        return "LightGBM"
    if b == "sklearn_gbr":
        return "GradientBoosting"
    if b == "hist_gbrt":
        return "HistGradientBoosting"
    return backend


def plot_ml_tradeoff(ml_home: pd.DataFrame, down: pd.DataFrame) -> None:
    """Side-by-side bars: prediction error vs downstream deviance if we have both tables."""
    import matplotlib.pyplot as plt

    if ml_home.empty or down.empty:
        return
    d = down.dropna(subset=["deviance", "backend"]).copy()
    if d.empty:
        return
    m = ml_home.dropna(subset=["backend", "mae"]).copy()
    if m.empty:
        return
    merged = d.merge(m[["backend", "mae"]], on="backend", how="inner")
    merged = merged.dropna(subset=["deviance", "mae"])
    if merged.empty:
        return
    merged["backend_label"] = merged["backend"].map(_backend_pretty)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(merged))
    ax0.bar(x, merged["mae"], color=["#4c72b0", "#55a868", "#c44e52"][: len(merged)])
    ax0.set_xticks(x)
    ax0.set_xticklabels(merged["backend_label"], rotation=15, ha="right")
    ax0.set_ylabel("Holdout MAE ($)")
    ax0.set_title("Median home value — which backend predicts best?")

    ax1.bar(x, merged["deviance"], color=["#8172b2", "#ccb974", "#64b5cd"][: len(merged)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged["backend_label"], rotation=15, ha="right")
    ax1.set_ylabel("GLM Poisson deviance (lower is better)")
    ax1.set_title("Downstream migration fit after each backend’s housing predictions")
    fig.suptitle("Layer 3 tradeoff: county housing accuracy vs gravity-model deviance", fontsize=12)
    fig.tight_layout()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUTS / "ml_backend_tradeoff.png", dpi=220)
    plt.close(fig)
    print(f"Saved {OUTPUTS / 'ml_backend_tradeoff.png'}")


def main() -> None:
    """Loop backends (or --dry-run), write CSVs, maybe the tradeoff PNG."""
    p = argparse.ArgumentParser(
        description="Layer 2 GLM Poisson fit (full spec) after each Layer 3 backend; writes comparison CSVs."
    )
    p.add_argument(
        "--backends",
        nargs="*",
        default=list(DEFAULT_BACKENDS),
        help=f"Backends to try (default: {DEFAULT_BACKENDS})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fit GLM on current ny_fl_dyadic_2122.csv (no 04/03 re-run). Uses backend column 'current_file'.",
    )
    args = p.parse_args()

    OUTPUTS.parent.mkdir(parents=True, exist_ok=True)
    (OUTPUTS.parent / "tables").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    if args.dry_run:
        if not DYADIC.is_file():
            print(f"Missing {DYADIC}; run pipeline first.")
            sys.exit(1)
        m = _prep(pd.read_csv(DYADIC, dtype={"origin_fips5": str, "dest_fips5": str}))
        met = glm_poisson_metrics(m)
        met["backend"] = "current_file"
        met["pipeline_ok"] = True
        rows.append(met)
    else:
        # Each backend retrains Layer 3 and refreshes dyadic CSV — downstream deviance is comparable across rows.
        for backend in args.backends:
            ok, err = run_pipeline(backend)
            rec: dict = {"backend": backend, "pipeline_ok": ok}
            if not ok:
                rec["error"] = err
                rows.append(rec)
                print(f"[{backend}] pipeline FAILED\n{err[:500]}")
                continue
            if not DYADIC.is_file():
                rec["error"] = "dyadic file missing after 03"
                rows.append(rec)
                continue
            m = _prep(pd.read_csv(DYADIC, dtype={"origin_fips5": str, "dest_fips5": str}))
            met = glm_poisson_metrics(m)
            if met.get("error"):
                rec["error"] = met["error"]
                rows.append(rec)
                continue
            met["backend"] = backend
            met["pipeline_ok"] = True
            rows.append(met)
            print(f"[{backend}] null_dev={met.get('null_deviance')} deviance={met.get('deviance')} n={met.get('nobs')}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DOWN, index=False)
    print(f"\nWrote {OUT_DOWN}")

    # Join with 04 --compare holdout MAE so one table shows "prediction error vs migration-model fit".
    ml_home = pd.DataFrame()
    if COMPARE_ML.is_file() and not out.empty:
        ml = pd.read_csv(COMPARE_ML)
        if "target" in ml.columns:
            ml_home = ml[ml["target"] == "median_home_value"].drop_duplicates(subset=["backend"])
        else:
            ml_home = ml
        if not ml_home.empty:
            agg = ml_home.groupby("backend", as_index=False).agg(
                {"mae": "first", "rmse": "first", "split": "first", "error": "first"}
            )
            merged = out.merge(agg, on="backend", how="outer", suffixes=("", "_ml"))
        else:
            merged = out
        merged.to_csv(OUT_FULL, index=False)
        print(f"Wrote {OUT_FULL} (merged with {COMPARE_ML.name})")
    else:
        merged = out
        out.to_csv(OUT_FULL, index=False)
        print(f"Wrote {OUT_FULL} (no ML compare file at {COMPARE_ML})")

    try:
        if COMPARE_ML.is_file():
            ml = pd.read_csv(COMPARE_ML)
            if "target" in ml.columns:
                ml_home = ml[ml["target"] == "median_home_value"].drop_duplicates(subset=["backend"])
            else:
                ml_home = ml
        plot_ml_tradeoff(ml_home, out)
    except Exception as e:
        print(f"ML tradeoff figure skipped: {e}")


if __name__ == "__main__":
    main()
