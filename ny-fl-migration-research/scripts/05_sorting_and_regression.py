#!/usr/bin/env python3
"""
Script 05 — Main regression outputs (Layers 1 and 2).

Layer 1 (log scale):
  - Log(1 + n_returns) ~ partisan terms + interaction (o_dem * d_dem) + gravity + optional ML gaps.
  - Cluster-robust SE by origin county; reduced specs (gravity-only, politics without ML gaps).
  - Two-way county fixed effects variant (large FE table; FIPS in output decorated with county names).

Layer 2 (count scale):
  - Poisson / NB / GEE with offset = log(origin population); see nyfl.layer2_specs for design details.
  - Writes regression_layer*_*.txt under data/processed/.
  - Exports model_comparison_across_specs.csv/.txt and coefficient_comparison_specs.png.

Housing gap columns appear only if 04 ran before 03 (run_all.sh order).

Big picture:
  Layer 1 is log OLS (gravity + politics + optional ML gaps), with SEs clustered by origin because lots of
  dyads share the same NY county. Layer 2 keeps raw counts and uses an origin-population offset, then runs
  Poisson/GEE/NB variants so we’re in line with how migration papers usually do it and we get deviance for knockouts.

--- Layer 2 Poisson / GEE / NB fix (why we changed the spec) ---
# The original Layer 2 GLM kept log_o_population inside the linear predictor while also leaning on
# distance and several housing/affordability terms. On this NY→FL dyad (389 positive flows) the
# resulting Poisson design was effectively saturated: statsmodels reported pseudo R² = 1 and
# cluster / GEE standard errors for the intercept and log_distance_km exploded to NaN — classic
# symptoms of a near rank-deficient information matrix together with Poisson overfitting on counts.
# We now follow a standard gravity/count formulation: remove log_o_population from the mean
# structure and supply offset = log(origin county population) as exposure. When both
# abs_diff_affordability_index and dyadic_affordability_gap are present and |corr| > 0.95, we drop
# abs_diff_affordability_index to avoid redundant scaling that was hurting conditioning.
# Continuous non-binary predictors are additionally z-scored inside `nyfl.layer2_specs` so the
# Poisson information matrix stays well conditioned (eliminating aliasing that produced identical
# Wald z-stats for the intercept and log-distance and NaN cluster/GEE standard errors). Coefficients
# on z-scored variables are in “per 1 SD” units; binary indicators remain on the 0/1 scale.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import NegativeBinomial, Poisson
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.linear_model import OLS

from nyfl.county_labels import decorate_statsmodels_summary_fips, load_fips_label_map
from nyfl.layer2_specs import build_layer2_design
from nyfl.paths import DATA_PROCESSED, OUTPUTS

DYADIC = DATA_PROCESSED / "ny_fl_dyadic_2122.csv"
OUT_LAYER1 = DATA_PROCESSED / "regression_layer1_ols_summary.txt"
OUT_LAYER1_FE = DATA_PROCESSED / "regression_layer1_twoway_fe_summary.txt"
OUT_LAYER2 = DATA_PROCESSED / "regression_layer2_count_glm_summary.txt"
OUT_TABLES = OUTPUTS.parent / "tables"

# ---------------------------------------------------------------------------
# Data prep: log_flow, partisan interaction, log distance and populations
# ---------------------------------------------------------------------------


def _prep(m: pd.DataFrame) -> pd.DataFrame:
    """Log flow, Dem×Dem interaction, log distance / pops — the usual gravity setup for the regressions below."""
    m = m.copy()
    m["log_flow"] = np.log1p(m["n_returns"])
    # Interaction: both counties more Democratic ⇒ larger product — captures "both blue" vs mixed pairs.
    if {"o_dem_two_party_share", "d_dem_two_party_share"}.issubset(m.columns):
        m["dem_origin_x_dest"] = m["o_dem_two_party_share"] * m["d_dem_two_party_share"]
    if "distance_km" in m.columns:
        m["log_distance_km"] = np.log1p(m["distance_km"].clip(lower=0))
    for c in ["o_population", "d_population"]:
        if c in m.columns:
            m[f"log_{c}"] = np.log1p(m[c].clip(lower=0))
    return m


# ---------------------------------------------------------------------------
# Layer 1 — OLS on log(1 + returns)
# ---------------------------------------------------------------------------


def fit_layer1_interaction(m: pd.DataFrame) -> tuple[str, Any]:
    """Main Layer 1 spec — log flows on politics (incl. interaction), gravity, ML gaps if present.

    We cluster by origin FIPS because the same county sends migrants to many FL destinations.
    """
    # Main spec: log flows ~ politics (incl. interaction) + logs + housing gap, cluster by origin.
    cols = [
        "o_dem_two_party_share",
        "d_dem_two_party_share",
        "dem_origin_x_dest",
        "log_distance_km",
        "log_o_population",
        "log_d_population",
    ]
    if "dyadic_housing_gap" in m.columns:
        cols.append("dyadic_housing_gap")
    if "dyadic_affordability_gap" in m.columns:
        cols.append("dyadic_affordability_gap")
    cols = [c for c in cols if c in m.columns]
    # Dependent variable: log1p(n_returns) prepared in _prep — stabilizes variance vs raw counts.
    y = m["log_flow"]
    X = m[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    # Listwise deletion: any NaN in y or any regressor drops the dyad for this spec.
    ok = y.notna() & X.notna().all(axis=1)
    if "origin_fips5" in m.columns:
        ok = ok & m["origin_fips5"].notna()
    y = y.loc[ok]
    X = X.loc[ok]
    # Same origin county appears in many rows — cluster SEs at origin to relax independence.
    groups = m.loc[ok, "origin_fips5"].astype(str) if "origin_fips5" in m.columns else None

    if len(y) < 10 or X.shape[1] < 2:
        return "Layer 1 (interaction OLS): SKIPPED — insufficient complete rows.\n", None

    X = sm.add_constant(X, has_constant="add")
    model = OLS(y, X)
    try:
        if groups is not None and groups.nunique() > 1:
            # cov_type="cluster": robust to arbitrary correlation within each origin FIPS group.
            res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            header = "Layer 1 — log(1+returns) OLS, clustered SE by origin\n"
        else:
            res = model.fit(cov_type="HC1")
            header = "Layer 1 — log(1+returns) OLS, HC1 (not enough clusters for clustering)\n"
    except Exception as e:
        res = model.fit(cov_type="HC1")
        header = f"Layer 1 — log(1+returns) OLS, HC1 (clustering blew up: {e})\n"

    return header + res.summary().as_text() + "\n", res


def fit_layer1_gravity_pops_only(m: pd.DataFrame) -> str:
    """Stripped-down spec: distance + populations only — baseline before you add politics or housing."""
    cols = ["log_distance_km", "log_o_population", "log_d_population"]
    cols = [c for c in cols if c in m.columns]
    y = m["log_flow"]
    X = m[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    ok = y.notna() & X.notna().all(axis=1)
    if "origin_fips5" in m.columns:
        ok = ok & m["origin_fips5"].notna()
    y = y.loc[ok]
    X = X.loc[ok]
    groups = m.loc[ok, "origin_fips5"].astype(str) if "origin_fips5" in m.columns else None
    if len(y) < 10 or X.shape[1] < 1:
        return "Layer 1 — REDUCED (gravity + pops only): SKIPPED — insufficient complete rows.\n"
    X = sm.add_constant(X, has_constant="add")
    model = OLS(y, X)
    try:
        if groups is not None and groups.nunique() > 1:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            head = "Layer 1 — REDUCED log(1+returns) OLS (gravity + pops only), clustered SE by origin\n"
        else:
            res = model.fit(cov_type="HC1")
            head = "Layer 1 — REDUCED log(1+returns) OLS (gravity + pops only), HC1\n"
    except Exception as e:
        res = model.fit(cov_type="HC1")
        head = f"Layer 1 — REDUCED OLS (gravity + pops), HC1 (cluster fallback: {e})\n"
    return head + res.summary().as_text() + "\n"


def fit_layer1_politics_no_ml_gaps(m: pd.DataFrame) -> str:
    """Same politics + gravity as the main spec but drops the GBDT dyadic gap columns if they exist."""
    cols = [
        "o_dem_two_party_share",
        "d_dem_two_party_share",
        "dem_origin_x_dest",
        "log_distance_km",
        "log_o_population",
        "log_d_population",
    ]
    cols = [c for c in cols if c in m.columns]
    y = m["log_flow"]
    X = m[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    ok = y.notna() & X.notna().all(axis=1)
    if "origin_fips5" in m.columns:
        ok = ok & m["origin_fips5"].notna()
    y = y.loc[ok]
    X = X.loc[ok]
    groups = m.loc[ok, "origin_fips5"].astype(str) if "origin_fips5" in m.columns else None
    if len(y) < 10 or X.shape[1] < 2:
        return "Layer 1 — REDUCED (politics + gravity + pops, no ML gaps): SKIPPED — insufficient complete rows.\n"
    X = sm.add_constant(X, has_constant="add")
    model = OLS(y, X)
    try:
        if groups is not None and groups.nunique() > 1:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            head = (
                "Layer 1 — REDUCED log(1+returns) OLS (politics + gravity + pops, no ML housing gaps), "
                "clustered SE by origin\n"
            )
        else:
            res = model.fit(cov_type="HC1")
            head = "Layer 1 — REDUCED log(1+returns) OLS (politics + gravity + pops, no ML gaps), HC1\n"
    except Exception as e:
        res = model.fit(cov_type="HC1")
        head = f"Layer 1 — REDUCED OLS (politics, no ML gaps), HC1 (cluster fallback: {e})\n"
    return head + res.summary().as_text() + "\n"


def fit_layer1_twoway_fe(m: pd.DataFrame) -> tuple[str, Any]:
    """Origin and destination county FE — soaks up anything fixed at either end of the dyad.

    Output is huge; we truncate the printed table so the file stays usable.
    """
    try:
        import patsy
    except ImportError:
        return "Layer 1 (two-way FE): SKIPPED — install patsy for FE formulas.\n", None

    need = {"origin_fips5", "dest_fips5", "log_flow", "log_distance_km"}
    if not need.issubset(m.columns):
        return "Layer 1 (two-way FE): SKIPPED — missing FIPS or log_distance_km.\n", None

    parts = ["log_flow ~ 1 + log_distance_km"]
    for c in [
        "abs_diff_dem_two_party_share",
        "dest_more_dem",
        "dyadic_housing_gap",
        "dyadic_affordability_gap",
    ]:
        if c in m.columns:
            parts[0] += f" + {c}"
    formula = parts[0] + " + C(origin_fips5) + C(dest_fips5)"
    try:
        y, X = patsy.dmatrices(formula, data=m, return_type="dataframe")
    except Exception as e:
        return f"Layer 1 (two-way FE): formula error: {e}\n", None

    y = y.iloc[:, 0]
    ok = y.notna() & X.notna().all(axis=1)
    y, X = y.loc[ok], X.loc[ok]
    if len(y) < 10:
        return "Layer 1 (two-way FE): SKIPPED — insufficient rows after FE design.\n", None

    if "origin_fips5" in m.columns:
        origin = m.loc[y.index, "origin_fips5"].astype(str)
    else:
        origin = None
    model = OLS(y, X)
    try:
        if origin is not None and origin.nunique() > 1:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": origin})
            head = "Layer 1 — two-way county FE, SE clustered by origin\n"
        else:
            res = model.fit(cov_type="HC1")
            head = "Layer 1 — two-way county FE, HC1\n"
    except Exception:
        res = model.fit(cov_type="HC1")
        head = "Layer 1 — two-way county FE, HC1 (cluster fallback)\n"
    txt = res.summary().as_text()
    if len(txt) > 12000:
        txt = txt[:12000] + "\n... [truncated; full FE table very long] ...\n"
    return head + txt + "\n", res


# ---------------------------------------------------------------------------
# Layer 2 — Poisson / GEE / Negative Binomial (count of returns)
# ---------------------------------------------------------------------------


def _fit_layer2_from_design(
    design: Any,
    *,
    header_prefix: str,
) -> str:
    """Fit Poisson (IID + cluster), GEE, and NB on the same X/offset so results line up row-for-row."""
    lines: list[str] = []
    y_count = design.y
    X = design.X
    offset = design.offset
    groups = design.groups
    if design.dropped_for_collinearity:
        lines.append(
            f"{header_prefix}Design note — dropped collinear predictors: "
            f"{'; '.join(design.dropped_for_collinearity)}\n"
        )
    if design.standardized:
        lines.append(
            f"{header_prefix}Z-scored continuous predictors (mean 0, SD 1): "
            f"{', '.join(design.standardized)}\n"
        )
    lines.append(
        f"{header_prefix}Offset: log(origin population); log_o_population removed from mean structure.\n"
    )

    # No explicit intercept: with a nonnegative offset, a separate constant is not identified in
    # cluster-robust GLM (SEs for const and log_distance were NaN). The offset pins the baseline scale.
    Xc = X.astype(float)
    off = np.asarray(offset, dtype=float)

    try:
        gee = GEE(
            y_count,
            Xc,
            groups=groups,
            family=Poisson(),
            cov_struct=Independence(),
            offset=off,
        )
        gee_res = gee.fit()
        lines.append(f"{header_prefix}=== Layer 2 — GEE Poisson (groups = origin county, independence) ===\n")
        lines.append(gee_res.summary().as_text() + "\n")
    except Exception as e:
        lines.append(f"{header_prefix}=== GEE Poisson failed: {e} ===\n")

    try:
        glm_p = sm.GLM(y_count, Xc, family=Poisson(), offset=off)
        gp_iid = glm_p.fit()
        lines.append(f"{header_prefix}=== Layer 2 — GLM Poisson (IID SE), with offset ===\n")
        lines.append(gp_iid.summary().as_text() + "\n")
        try:
            X0 = np.ones((len(y_count), 1))
            null_p = sm.GLM(y_count, X0, family=Poisson(), offset=off).fit()
            mcf = 1.0 - (gp_iid.llf / null_p.llf) if null_p.llf != 0 else float("nan")
            lines.append(
                f"{header_prefix}McFadden pseudo R² (vs offset + scalar intercept): {mcf:.4f}  "
                f"(Cox–Snell pseudo R² in the header can stay at 1.0 for Poisson; prefer McFadden or deviance ratio.)\n"
            )
        except Exception as _e_null:
            lines.append(f"{header_prefix}(McFadden pseudo R² skipped: {_e_null})\n")
        try:
            gp_clu = glm_p.fit(cov_type="cluster", cov_kwds={"groups": groups})
            lines.append(f"{header_prefix}=== GLM Poisson — cluster-robust SE by origin, with offset ===\n")
            lines.append(gp_clu.summary().as_text() + "\n")
        except Exception as e2:
            lines.append(f"{header_prefix}(GLM Poisson cluster SE failed: {e2})\n")
    except Exception as e:
        lines.append(f"{header_prefix}=== GLM Poisson failed: {e} ===\n")

    try:
        glm_nb = sm.GLM(y_count, Xc, family=NegativeBinomial(alpha=1.0), offset=off)
        nb = glm_nb.fit()
        lines.append(f"{header_prefix}=== Layer 2 — GLM Negative Binomial (alpha=1.0), with offset ===\n")
        lines.append(nb.summary().as_text() + "\n")
    except Exception as e:
        lines.append(f"{header_prefix}=== GLM Negative Binomial failed: {e} ===\n")

    return "".join(lines)


def fit_layer2_glm(m: pd.DataFrame) -> str:
    """Full Layer 2 — includes ML housing/affordability stuff when those columns exist."""
    design = build_layer2_design(m, full=True)
    if design is None:
        return "Layer 2: SKIPPED — need abs_diff and gravity columns / o_population for offset.\n"
    return _fit_layer2_from_design(design, header_prefix="")


def fit_layer2_glm_reduced(m: pd.DataFrame) -> str:
    """Layer 2 without the ML housing block — matches the spirit of the no_housing knockout in 09."""
    design = build_layer2_design(m, full=False)
    if design is None:
        return "Layer 2 — REDUCED (core dyadic + gravity): SKIPPED — need abs_diff and gravity columns.\n"
    lines = [
        "=== Layer 2 — REDUCED spec (core politics/density/race + gravity + dest pop; "
        "origin pop as offset; no rent / affordability / ML gap terms) ===\n",
    ]
    lines.append(_fit_layer2_from_design(design, header_prefix=""))
    return "".join(lines)


def _coef_safe(res: Any, name: str) -> tuple[float, float, float, float]:
    if res is None or not hasattr(res, "params") or name not in res.params.index:
        return (float("nan"),) * 4
    p = float(res.params[name])
    try:
        se = float(res.bse[name])
        lo, hi = res.conf_int().loc[name]
        return p, se, float(lo), float(hi)
    except Exception:
        return p, float("nan"), float("nan"), float("nan")


def export_spec_comparison(
    m: pd.DataFrame,
    res_pooled: Any,
    res_fe: Any,
    glm_poisson_iid: Any,
) -> None:
    """One table that lines up coefs across the three big specs (not every variable shows up everywhere)."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def add_row(var: str, label: str, pooled_name: str | None, fe_name: str | None, glm_name: str | None):
        pp = _coef_safe(res_pooled, pooled_name) if pooled_name else (float("nan"),) * 4
        pf = _coef_safe(res_fe, fe_name) if fe_name else (float("nan"),) * 4
        pg = _coef_safe(glm_poisson_iid, glm_name) if glm_name else (float("nan"),) * 4
        rows.append(
            {
                "variable_key": var,
                "variable_label": label,
                "pooled_ols_coef": pp[0],
                "pooled_ols_se": pp[1],
                "pooled_ols_ci_low": pp[2],
                "pooled_ols_ci_high": pp[3],
                "twoway_fe_coef": pf[0],
                "twoway_fe_se": pf[1],
                "twoway_fe_ci_low": pf[2],
                "twoway_fe_ci_high": pf[3],
                "glm_poisson_coef": pg[0],
                "glm_poisson_se": pg[1],
                "glm_poisson_ci_low": pg[2],
                "glm_poisson_ci_high": pg[3],
            }
        )

    add_row(
        "dem_origin_x_dest",
        "Origin × destination Dem share (pooled OLS interaction)",
        "dem_origin_x_dest",
        None,
        None,
    )
    add_row(
        "abs_diff_dem",
        "Absolute difference in Dem two-party share (FE / GLM)",
        None,
        "abs_diff_dem_two_party_share",
        "abs_diff_dem_two_party_share",
    )
    add_row("dest_more_dem", "Indicator: destination more Democratic than origin", None, "dest_more_dem", "dest_more_dem")
    add_row("dyadic_housing_gap", "ML dyadic housing gap (dest pred − origin pred)", "dyadic_housing_gap", "dyadic_housing_gap", "dyadic_housing_gap")
    add_row(
        "dyadic_affordability_gap",
        "ML dyadic affordability gap",
        "dyadic_affordability_gap",
        "dyadic_affordability_gap",
        "dyadic_affordability_gap",
    )
    add_row("log_distance_km", "log(1 + distance km)", "log_distance_km", "log_distance_km", "log_distance_km")
    add_row("log_o_population", "log(1 + origin population)", "log_o_population", None, "(in offset, not in GLM mean)")
    add_row("log_d_population", "log(1 + destination population)", "log_d_population", None, "log_d_population")

    df = pd.DataFrame(rows)
    csv_path = OUT_TABLES / "model_comparison_across_specs.csv"
    txt_path = OUT_TABLES / "model_comparison_across_specs.txt"
    df.to_csv(csv_path, index=False)
    txt_path.write_text(df.to_string(index=False) + "\n")

    # Coefficient comparison plot: one row per (variable, specification).
    try:
        import matplotlib.pyplot as plt

        rows: list[dict[str, Any]] = []

        def push_row(y_label: str, spec: str, pname: str, res: Any, color: str) -> None:
            if res is None or pname not in getattr(res, "params", pd.Index([])):
                return
            p, se, lo, hi = _coef_safe(res, pname)
            if np.isnan(p):
                return
            rows.append({"y_label": y_label, "spec": spec, "p": p, "lo": lo, "hi": hi, "color": color})

        if res_pooled is not None:
            push_row("Dem × Dem (interaction)", "Pooled OLS", "dem_origin_x_dest", res_pooled, "#1f77b4")
            push_row("log distance", "Pooled OLS", "log_distance_km", res_pooled, "#1f77b4")
            push_row("ML housing gap", "Pooled OLS", "dyadic_housing_gap", res_pooled, "#1f77b4")
        if res_fe is not None:
            push_row("Dest more Democratic", "Two-way FE", "dest_more_dem", res_fe, "#ff7f0e")
            push_row("log distance", "Two-way FE", "log_distance_km", res_fe, "#ff7f0e")
            push_row("ML housing gap", "Two-way FE", "dyadic_housing_gap", res_fe, "#ff7f0e")
        if glm_poisson_iid is not None:
            push_row("Dest more Democratic", "GLM Poisson", "dest_more_dem", glm_poisson_iid, "#2ca02c")
            push_row("log distance", "GLM Poisson", "log_distance_km", glm_poisson_iid, "#2ca02c")
            push_row("ML housing gap", "GLM Poisson", "dyadic_housing_gap", glm_poisson_iid, "#2ca02c")

        if not rows:
            raise RuntimeError("no comparable coefficients to plot")

        fig_h = max(4.5, 0.42 * len(rows))
        fig, ax = plt.subplots(figsize=(9, fig_h))
        y = np.arange(len(rows))
        for i, r in enumerate(rows):
            p, lo, hi = r["p"], r["lo"], r["hi"]
            xerr = None
            if not (np.isnan(lo) or np.isnan(hi)):
                xerr = np.array([[p - lo], [hi - p]])
            ax.errorbar(p, i, xerr=xerr, fmt="o", color=r["color"], ecolor=r["color"], capsize=3)
        ax.axvline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['y_label']} — {r['spec']}" for r in rows], fontsize=8)
        ax.set_xlabel("Coefficient (95% CI when available)")
        ax.set_title("Partisan & housing coefficients across specifications")
        fig.tight_layout()
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUTS / "coefficient_comparison_specs.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Coefficient comparison plot skipped: {e}")

    print(f"Wrote {csv_path}, {txt_path}")


def _refit_glm_poisson_iid_for_export(m: pd.DataFrame) -> Any:
    """Pull Poisson coefs for the cross-spec table — same Layer-2 design as everywhere else."""
    design = build_layer2_design(m, full=True)
    if design is None:
        return None
    Xc = design.X.astype(float)
    off = np.asarray(design.offset, dtype=float)
    try:
        return sm.GLM(design.y, Xc, family=Poisson(), offset=off).fit()
    except Exception:
        return None


def main() -> None:
    """Run the stack of regressions and dump summaries + the coef comparison files."""
    fips_map = load_fips_label_map()
    m = pd.read_csv(DYADIC, dtype={"origin_fips5": str, "dest_fips5": str})
    m = _prep(m)

    buf1, res_pooled = fit_layer1_interaction(m)
    buf1 += "\n" + fit_layer1_gravity_pops_only(m)
    buf1 += "\n" + fit_layer1_politics_no_ml_gaps(m)
    buf1_fe, res_fe = fit_layer1_twoway_fe(m)
    buf2 = fit_layer2_glm(m)
    buf2 += "\n" + fit_layer2_glm_reduced(m)

    buf1 = decorate_statsmodels_summary_fips(buf1, fips_map)
    buf1_fe = decorate_statsmodels_summary_fips(buf1_fe, fips_map)
    buf2 = decorate_statsmodels_summary_fips(buf2, fips_map)

    OUT_LAYER1.write_text(buf1)
    OUT_LAYER1_FE.write_text(buf1_fe)
    OUT_LAYER2.write_text(buf2)
    print(buf1)
    print(buf1_fe)
    print(buf2)
    print(f"Wrote {OUT_LAYER1}, {OUT_LAYER1_FE}, {OUT_LAYER2}")

    legacy = DATA_PROCESSED / "regression_ols_log_flow_summary.txt"
    legacy.write_text(buf1)
    print(f"Wrote {legacy}")

    glm_iid = _refit_glm_poisson_iid_for_export(m)
    try:
        export_spec_comparison(m, res_pooled, res_fe, glm_iid)
    except Exception as e:
        print(f"Spec comparison export skipped: {e}")

    try:
        import matplotlib.pyplot as plt

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        if "d_dem_two_party_share" in m.columns:
            plot_df = m.dropna(subset=["d_dem_two_party_share"])
            if plot_df.empty:
                print("Scatter skipped: no non-missing d_dem_two_party_share (add county election CSV).")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(plot_df["d_dem_two_party_share"], plot_df["log_flow"], alpha=0.5, s=12)
                ax.set_xlabel("Destination county Dem two-party share (2020)")
                ax.set_ylabel("log(1 + IRS returns), NY→FL dyad")
                ax.set_title("NY→FL flows vs destination partisanship (dyad level)")
                fig.tight_layout()
                fig.savefig(OUTPUTS / "sorting_flow_vs_dest_dem.png", dpi=150)
                print(f"Saved {OUTPUTS / 'sorting_flow_vs_dest_dem.png'}")
                plt.close()
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
