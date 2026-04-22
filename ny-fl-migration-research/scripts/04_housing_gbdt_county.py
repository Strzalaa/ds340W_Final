#!/usr/bin/env python3
"""
Script 04 — Layer 3: tabular ML on county rows (not on dyads).

Trains gradient boosting to predict:
  - median_home_value from ACS features (rent, income, race, tenure, density, ...).
  - affordability_index when available.

Writes county_housing_gbdt_predictions.csv with predicted values and residuals.
Script 03 merges these as origin/dest predictions and builds dyadic_housing_gap = d_pred - o_pred.

Flags:
  --backend auto|lgbm|sklearn_gbr|hist_gbrt — which library trains the production predictions.
  --compare — only evaluates holdout MAE/RMSE for all three backends; writes housing_ml_backend_comparison.csv; does not overwrite main predictions.

We fit on counties, not dyads — then 03 subtracts predicted origin from predicted dest to get a gap per pair.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from nyfl.paths import DATA_PROCESSED, OUTPUTS

COV_CSV = DATA_PROCESSED / "county_covariates_ny_fl.csv"
OUT_CSV = DATA_PROCESSED / "county_housing_gbdt_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25

COMPARE_BACKENDS = ("lgbm", "sklearn_gbr", "hist_gbrt")


def _fit_lgbm(X: np.ndarray, y: np.ndarray) -> tuple[object, str]:
    """LightGBM regressor for median home value (or affordability if we use it)."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X, y)
    return model, "LightGBM"


def _fit_sklearn_gbr(X: np.ndarray, y: np.ndarray) -> tuple[object, str]:
    """Plain sklearn boosting — backup when LightGBM won’t load (e.g. libomp on Mac)."""
    model = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        max_depth=3,
        n_estimators=300,
        learning_rate=0.05,
    )
    model.fit(X, y)
    return model, "sklearn GradientBoostingRegressor"


def _fit_hist_gbrt(X: np.ndarray, y: np.ndarray) -> tuple[object, str]:
    """HistGradientBoosting — third option when you run `--compare`."""
    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model, "sklearn HistGradientBoostingRegressor"


def fit_backend(backend: str, X: np.ndarray, y: np.ndarray) -> tuple[object, str]:
    """Pick LightGBM / sklearn GBR / hist GBRT."""
    b = backend.lower().strip()
    if b == "lgbm":
        return _fit_lgbm(X, y)
    if b == "sklearn_gbr":
        return _fit_sklearn_gbr(X, y)
    if b == "hist_gbrt":
        return _fit_hist_gbrt(X, y)
    raise ValueError(f"Unknown backend: {backend}")


def resolve_auto_backend() -> str:
    """Use LightGBM if the import works; fall back to sklearn."""
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):
        return "sklearn_gbr"
    return "lgbm"


def _eval_one_split(
    X: np.ndarray,
    y: np.ndarray,
    backend: str,
) -> dict:
    """MAE/RMSE for `--compare` (in-sample if we have almost no counties)."""
    label = backend
    try:
        if len(X) < 8:
            model, pretty = fit_backend(backend, X, y)
            pred = model.predict(X)
            mae = mean_absolute_error(y, pred)
            rmse = mean_squared_error(y, pred) ** 0.5
            return {
                "backend": label,
                "backend_label": pretty,
                "split": "in_sample",
                "n_rows": len(y),
                "mae": mae,
                "rmse": rmse,
                "error": "",
            }
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        model, pretty = fit_backend(backend, X_train, y_train)
        pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred_test)
        rmse = mean_squared_error(y_test, pred_test) ** 0.5
        return {
            "backend": label,
            "backend_label": pretty,
            "split": f"holdout_{TEST_SIZE}",
            "n_rows": len(y),
            "mae": mae,
            "rmse": rmse,
            "error": "",
        }
    except Exception as e:
        return {
            "backend": label,
            "backend_label": label,
            "split": "",
            "n_rows": len(y) if y is not None else 0,
            "mae": np.nan,
            "rmse": np.nan,
            "error": (str(e)[:300] + "…") if len(str(e)) > 300 else str(e),
        }


def run_compare(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Run every backend through the same train/test split for the comparison CSV."""
    rows: list[dict] = []
    if target not in df.columns or len(feature_cols) < 2:
        return pd.DataFrame(rows)
    work = df.dropna(subset=[target] + feature_cols).copy()
    if work.empty:
        return pd.DataFrame(rows)
    X = work[feature_cols].values.astype(float)
    y = work[target].values.astype(float)

    for backend in COMPARE_BACKENDS:
        m = _eval_one_split(X, y, backend)
        m["target"] = target
        rows.append(m)
    return pd.DataFrame(rows)


def _train_predict(
    d: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    backend: str,
    label_prefix: str,
) -> tuple[pd.DataFrame, object | None, str | None]:
    """Train, print holdout error, then retrain on everyone so every county has a prediction for 03."""
    if target not in d.columns or len(feature_cols) < 2:
        return d, None, None
    # Complete cases only: rows with missing target or any feature are dropped for training.
    work = d.dropna(subset=[target] + feature_cols).copy()
    if work.empty:
        return d, None, None
    X = work[feature_cols].values
    y = work[target].values
    pred_col = "housing_pred" if target == "median_home_value" else "affordability_pred"
    resid_col = "resid_home_value" if target == "median_home_value" else "resid_affordability_index"

    if len(work) < 8:
        print(f"Only {len(work)} counties for {target} — fitting on all data for demo.")
        model, pretty = fit_backend(backend, X, y)
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = mean_squared_error(y, pred) ** 0.5
        print(f"  [{label_prefix}] in-sample MAE: {mae:.6f}  RMSE: {rmse:.6f}  ({pretty})")
        work[pred_col] = pred
        work[resid_col] = work[target] - work[pred_col]
        out = d.merge(work[["county_fips5", pred_col, resid_col]], on="county_fips5", how="left")
        return out, model, pretty

    # Holdout split: print out-of-sample error so we don't over-trust in-sample fit on tiny N.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model, pretty = fit_backend(backend, X_train, y_train)
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = mean_squared_error(y_test, pred_test) ** 0.5
    print(f"  [{label_prefix}] holdout MAE: {mae:.6f}  RMSE: {rmse:.6f}  ({pretty})")
    # Retrain on all counties so every FIPS gets a prediction for script 03 (merge back to full `d`).
    model, pretty = fit_backend(backend, X, y)
    work[pred_col] = model.predict(X)
    work[resid_col] = work[target] - work[pred_col]
    out = d.merge(work[["county_fips5", pred_col, resid_col]], on="county_fips5", how="left")
    return out, model, pretty


def main() -> None:
    """`--compare` writes the backend table; otherwise train and save predictions for 03."""
    p = argparse.ArgumentParser(description="County housing / affordability tabular ML (Layer 3).")
    p.add_argument(
        "--backend",
        choices=("auto", "lgbm", "sklearn_gbr", "hist_gbrt"),
        default="auto",
        help="Training backend for written predictions (default: auto = LightGBM if import works).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Evaluate all backends (holdout MAE/RMSE), write comparison CSV, exit without overwriting main predictions.",
    )
    args = p.parse_args()

    df = pd.read_csv(COV_CSV, dtype={"county_fips5": str})
    df["county_fips5"] = df["county_fips5"].str.zfill(5)

    # Tabular predictors: all ACS / engineered county-level features (no dyadic terms — this is Layer 3 on counties).
    base_features = [
        c
        for c in [
            "median_gross_rent",
            "population",
            "median_hh_income",
            "pct_nh_white",
            "pct_owner_occupied",
            "owner_to_renter_ratio",
            "log_pop_density",
        ]
        if c in df.columns
    ]
    if len(base_features) < 2:
        print("Not enough feature columns for GBDT; skipping.")
        return

    if args.compare:
        out_dir = OUTPUTS.parent / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmp_path = out_dir / "housing_ml_backend_comparison.csv"

        frames: list[pd.DataFrame] = []
        for tgt, prefix in [("median_home_value", "home_value"), ("affordability_index", "affordability")]:
            if tgt == "affordability_index" and (
                "affordability_index" not in df.columns or df["affordability_index"].notna().sum() < 3
            ):
                continue
            feat = base_features
            if tgt == "affordability_index":
                feat = [c for c in base_features if c not in ("median_gross_rent", "median_hh_income")]
                if len(feat) < 2:
                    feat = [c for c in base_features if c != "median_gross_rent"]
            fr = run_compare(df, tgt, feat)
            if not fr.empty:
                fr["feature_set"] = prefix
                frames.append(fr)

        if frames:
            comp = pd.concat(frames, ignore_index=True)
            comp.to_csv(cmp_path, index=False)
            print(f"Wrote {cmp_path}\n")
            print(comp.to_string(index=False))
        else:
            print("Compare: no rows (check covariates and affordability_index).")
        return

    resolved = resolve_auto_backend() if args.backend == "auto" else args.backend
    if resolved == "lgbm":
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError) as e:
            print(
                f"LightGBM unavailable ({type(e).__name__}); falling back to sklearn GradientBoostingRegressor. "
                f"To use LightGBM on Mac, install OpenMP: brew install libomp"
            )
            resolved = "sklearn_gbr"

    print(f"Using backend: {resolved}")

    d = df.copy()
    d, model_h, _ = _train_predict(d, "median_home_value", base_features, resolved, "home value")
    if "housing_pred" not in d.columns and "pred_median_home_value" in d.columns:
        d["housing_pred"] = d["pred_median_home_value"]
    if "housing_pred" in d.columns:
        d["pred_median_home_value"] = d["housing_pred"]
    if "resid_home_value" not in d.columns:
        d["resid_home_value"] = np.nan

    if "affordability_index" in d.columns and d["affordability_index"].notna().sum() >= 3:
        aff_features = [c for c in base_features if c not in ("median_gross_rent", "median_hh_income")]
        if len(aff_features) < 2:
            aff_features = [c for c in base_features if c != "median_gross_rent"]
        d, model_a, _ = _train_predict(d, "affordability_index", aff_features, resolved, "affordability index")
        if "affordability_pred" not in d.columns:
            d["affordability_pred"] = np.nan
            d["resid_affordability_index"] = np.nan
    else:
        print("No usable affordability_index column — skipping affordability GBDT.")
        d["affordability_pred"] = np.nan
        d["resid_affordability_index"] = np.nan

    out_cols = ["county_fips5", "median_home_value", "housing_pred", "pred_median_home_value", "resid_home_value"]
    if "affordability_index" in d.columns:
        out_cols.extend(["affordability_index", "affordability_pred", "resid_affordability_index"])
    else:
        out_cols.extend(["affordability_pred", "resid_affordability_index"])
    out_pred = d[[c for c in out_cols if c in d.columns]].copy()
    out_pred.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    if model_h is not None and hasattr(model_h, "feature_importances_"):
        imp = pd.DataFrame({"feature": base_features, "importance": getattr(model_h, "feature_importances_", np.zeros(len(base_features)))})
        imp = imp.sort_values("importance", ascending=False)
        print("Home value model — feature importance:\n" + imp.to_string(index=False))

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        if model_h is not None and hasattr(model_h, "feature_importances_"):
            fig, ax = plt.subplots(figsize=(6, 4))
            imp = pd.DataFrame({"feature": base_features, "importance": model_h.feature_importances_}).sort_values("importance", ascending=True)
            ax.barh(imp["feature"], imp["importance"])
            ax.set_title("Housing model — median home value (tabular boosting)")
            fig.tight_layout()
            fig.savefig(OUTPUTS / "housing_gbdt_importance.png", dpi=150)
            print(f"Saved {OUTPUTS / 'housing_gbdt_importance.png'}")
            plt.close(fig)
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
