#!/usr/bin/env python3
"""
Script 09 — Layer 2 Poisson block knockouts (which predictor groups matter for deviance?).

Refits GLM Poisson with the same offset and design conventions as script 05 (nyfl.layer2_specs).
Special case: when distance is dropped, an intercept is included so the model is identified.
Writes the text dump + glm_knockout_deviance.csv. Same offset as 05; we only rip out groups of predictors and stare at deviance.
"""

from __future__ import annotations

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
OUT = DATA_PROCESSED / "ablation_glm_poisson_results.txt"
OUT_KNOCK_CSV = OUTPUTS.parent / "tables" / "glm_knockout_deviance.csv"


def _prep(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    if "distance_km" in m.columns:
        m["log_distance_km"] = np.log1p(m["distance_km"].clip(lower=0))
    for c in ["o_population", "d_population"]:
        if c in m.columns:
            m[f"log_{c}"] = np.log1p(m[c].clip(lower=0))
    return m


BLOCKS: dict[str, list[str] | None] = {
    "full": None,
    "no_political": [
        "abs_diff_dem_two_party_share",
        "dest_more_dem",
    ],
    "no_housing": [
        "dyadic_housing_gap",
        "dyadic_affordability_gap",
        "abs_diff_affordability_index",
        "dest_lower_rent_than_origin",
        "dest_lower_rent_pressure_than_origin",
    ],
    "no_distance": ["log_distance_km"],
}


def fit_glm_poisson_with_offset(
    y: pd.Series,
    X: pd.DataFrame,
    offset: pd.Series,
) -> tuple[str, float, float, int]:
    """Same Poisson setup as 05. If we drop distance we add a constant back — otherwise the fit goes haywire."""
    Xn = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = y.notna() & Xn.notna().all(axis=1) & offset.notna()
    y2 = y.loc[ok].astype(float)
    X2 = Xn.loc[ok]
    off2 = np.asarray(offset.loc[ok], dtype=float)
    if len(y2) < 5 or X2.shape[1] < 1:
        return "SKIPPED — insufficient rows.\n", float("nan"), float("nan"), 0
    # Match scripts/05: no intercept when distance is in the model (avoids singular cluster cov).
    # When log_distance_km is dropped (no_distance block), re-introduce an intercept — without it,
    # the remaining z-scored covariates are not identified relative to the offset and IRLS can blow up.
    if "log_distance_km" in X2.columns:
        Xc = X2.astype(float)
    else:
        Xc = sm.add_constant(X2.astype(float), has_constant="add")
    try:
        glm = sm.GLM(y2, Xc, family=Poisson(), offset=off2)
        res = glm.fit()
        txt = (
            res.summary().as_text()
            + f"\nNull deviance: {res.null_deviance:.4f}  Deviance: {res.deviance:.4f}  N: {int(res.nobs)}\n"
        )
        return txt, float(res.null_deviance), float(res.deviance), int(res.nobs)
    except Exception as e:
        return f"ERROR: {e}\n", float("nan"), float("nan"), 0


def main() -> None:
    """Run each knockout, save the summaries and CSV, try to plot bars."""
    m = _prep(pd.read_csv(DYADIC, dtype={"origin_fips5": str, "dest_fips5": str}))
    design = build_layer2_design(m, full=True)
    if design is None:
        msg = "Layer 2 GLM knockouts: SKIPPED — could not build Layer-2 design (check o_population, covariates).\n"
        OUT.write_text(msg)
        print(msg)
        return

    full_cols = list(design.X.columns)
    if len(full_cols) < 2:
        msg = "Layer 2 GLM knockouts: SKIPPED — need at least two covariates.\n"
        OUT.write_text(msg)
        print(msg)
        return

    lines: list[str] = []
    lines.append(
        "GLM Poisson block knockouts (Layer 2 — offset=log(origin population); same predictors as scripts/05)\n\n"
    )

    knock_rows: list[dict[str, object]] = []

    for label, drop in BLOCKS.items():
        use_cols = [c for c in full_cols if not drop or c not in drop]
        if len(use_cols) < 1:
            lines.append(f"=== {label} === SKIPPED (no columns left)\n\n")
            continue
        Xsub = design.X[use_cols]
        txt, ndev, dev, nobs = fit_glm_poisson_with_offset(design.y, Xsub, design.offset)
        lines.append(f"=== {label} ===  (predictors: {', '.join(use_cols)})\n")
        lines.append(txt)
        lines.append("\n")
        knock_rows.append(
            {
                "block": label,
                "null_deviance": ndev,
                "deviance": dev,
                "nobs": nobs,
            }
        )

    text = "".join(lines)
    OUT.write_text(text)
    print(text)
    print(f"Wrote {OUT}")

    OUT_KNOCK_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(knock_rows).to_csv(OUT_KNOCK_CSV, index=False)
    print(f"Wrote {OUT_KNOCK_CSV}")

    # Bar chart of deviance by knockout block
    try:
        import matplotlib.pyplot as plt

        dfk = pd.DataFrame(knock_rows).dropna(subset=["deviance"])
        if not dfk.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            order = ["full", "no_political", "no_housing", "no_distance"]
            dfk["_ord"] = dfk["block"].map({b: i for i, b in enumerate(order)})
            dfk = dfk.sort_values("_ord")
            colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"][: len(dfk)]
            bars = ax.bar(dfk["block"], dfk["deviance"], color=colors)
            ax.set_ylabel("GLM Poisson deviance (lower is better)")
            ax.set_title("Which covariate block matters for explaining flows?")
            for rect, val in zip(bars, dfk["deviance"].tolist(), strict=False):
                h = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            fig.tight_layout()
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            fig.savefig(OUTPUTS / "glm_knockout_deviance.png", dpi=200)
            plt.close(fig)
            print(f"Saved {OUTPUTS / 'glm_knockout_deviance.png'}")
    except Exception as e:
        print(f"Knockout deviance figure skipped: {e}")


if __name__ == "__main__":
    main()
