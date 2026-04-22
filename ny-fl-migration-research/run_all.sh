#!/usr/bin/env bash
# =============================================================================
# Run the whole NY → FL pipeline from repo root (venv on, requirements installed — see README).
# Windows: Git Bash or `bash run_all.sh`; PowerShell won’t run this as-is.
#
# Order actually matters: train housing (04) before building dyads (03) so ML gaps exist;
# later we compare backends and then put HOUSING_BACKEND back the way it was.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export MPLBACKEND=Agg

: "${USE_SAMPLE_IRS:=}"
: "${USE_SAMPLE_COVARIATES:=}"
: "${ALLOW_SAMPLE_PIPELINE:=}"
# Layer 3 housing model: auto tries LightGBM, else sklearn (see scripts/04_housing_gbdt_county.py)
: "${HOUSING_BACKEND:=auto}"

_sample_truthy() {
  case "${1:-}" in 1|true|TRUE|yes|Yes|YES) return 0 ;; *) return 1 ;; esac
}
if _sample_truthy "${USE_SAMPLE_IRS}" || _sample_truthy "${USE_SAMPLE_COVARIATES}"; then
  if ! _sample_truthy "${ALLOW_SAMPLE_PIPELINE}"; then
    echo "ERROR: USE_SAMPLE_IRS or USE_SAMPLE_COVARIATES is set." >&2
    echo "  Submission / production runs must use full datasets: unset both variables." >&2
    echo "  Offline demo only: ALLOW_SAMPLE_PIPELINE=1 USE_SAMPLE_IRS=1 USE_SAMPLE_COVARIATES=1 ./run_all.sh" >&2
    exit 1
  fi
  echo "WARNING: sample pipeline mode (bundled extracts). Do not cite these outputs as production results." >&2
fi

echo "Starting pipeline (04 before 03 so housing predictions exist for dyadic merge)..."

# --- Data: IRS flows + county covariates (ACS, election, gazetteer) ---
python scripts/01_build_ny_fl_flows.py
python scripts/02_fetch_county_covariates.py

# --- Layer 3: train housing/affordability models on counties; writes predictions CSV ---
python scripts/04_housing_gbdt_county.py --backend "${HOUSING_BACKEND}"

# --- Build dyadic table: merge flows x covariates, distance, diffs, ML dyadic gaps ---
python scripts/03_merge_and_dyadic.py
python scripts/diagnose_dyadic_missingness.py

# --- Models: Layer 1 OLS, Layer 2 GLM/GEE/NB; figures; ablations; GLM knockouts ---
python scripts/05_sorting_and_regression.py
python scripts/06_partisan_flow_figures.py
python scripts/07_sensitivity_ablation.py
python scripts/09_glm_block_knockout.py

# --- Compare ML backends (holdout error) + downstream Poisson deviance; restore primary backend ---
python scripts/04_housing_gbdt_county.py --compare
python scripts/10_housing_downstream_compare.py --backends lgbm sklearn_gbr hist_gbrt
python scripts/04_housing_gbdt_county.py --backend "${HOUSING_BACKEND}"
python scripts/03_merge_and_dyadic.py

# --- Export tables, map, Hasan ML vs raw ACS comparison, reproducibility manifest ---
python scripts/08_export_tables.py
python scripts/11_flow_map.py
python scripts/12_hasan_housing_comparison.py
python scripts/snapshot_manifest.py

echo "All done. Outputs: data/processed/, outputs/figures/, outputs/tables/, data/SUBMISSION_MANIFEST.txt"
