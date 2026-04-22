# NY → Florida migration (county-level research pipeline)

Reproducible analysis of NY→FL county flows (IRS 2021–2022): OLS sorting models, gravity-style GLMs, optional boosting-based housing signals. **Run everything with `./run_all.sh`** after installing dependencies.

**Requirements:** Python 3.10+ (3.11+ recommended), `pip`, and `git`. On Windows, **Git for Windows** (includes Git Bash) is needed to run the Bash driver script, or use WSL.

---

## 1. Clone the repository

```bash
git clone <repository-url>.git
cd ny-fl-migration-research
```

---

## 2. Create a virtual environment and install dependencies

### macOS / Linux (Terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows — Git Bash (recommended)

`run_all.sh` is a Bash script; Git Bash matches the commands below.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Windows — PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 3. Run the full pipeline

Unset any sample/demo flags so the run uses full inputs (when available):

### macOS / Linux / Git Bash

```bash
chmod +x run_all.sh
unset USE_SAMPLE_IRS USE_SAMPLE_COVARIATES ALLOW_SAMPLE_PIPELINE
export PYTHONPATH="$(pwd)/src"
export PATH="$(pwd)/.venv/bin:$PATH"
./run_all.sh
```

### Windows — PowerShell

From the repo root, with `bash` available (Git for Windows or WSL):

```powershell
$env:USE_SAMPLE_IRS=""
$env:USE_SAMPLE_COVARIATES=""
$env:ALLOW_SAMPLE_PIPELINE=""
$env:PYTHONPATH = "$PWD\src"
bash ./run_all.sh
```

If `bash` is not found, install **Git for Windows** and run the **macOS / Git Bash** block from **Git Bash** in the project folder.

**Success:** the script should end with:

`All done. Outputs: data/processed/, outputs/figures/, outputs/tables/, data/SUBMISSION_MANIFEST.txt`

---

## 4. Data inputs (`data/raw/`)

Scripts can fetch some inputs automatically. You may also place files manually:

| File | Role |
|------|------|
| `countyoutflow2122.csv` | IRS county outflow |
| `2020_Gaz_counties_national.txt` | Census gazetteer (lat/lon, land area) |
| `county_election_2020.csv` or `countypres*.csv` | County two-party presidential share (2020) |

---

## 5. Outputs

After a successful run:

| Location | Contents |
|----------|----------|
| `data/processed/` | Merged CSVs, regression summary `.txt` files |
| `outputs/tables/` | Summary and comparison CSV/TXT tables |
| `outputs/figures/` | Figures (PNG) |
| `data/SUBMISSION_MANIFEST.txt` | File hashes for reproducibility |

---

## 6. Troubleshooting

**LightGBM on macOS fails (OpenMP):**

```bash
brew install libomp
```

**Use scikit-learn boosting only (no LightGBM):**

```bash
HOUSING_BACKEND=sklearn_gbr ./run_all.sh
```

**Offline / bundled sample data (for testing only — not for publication numbers):**

```bash
ALLOW_SAMPLE_PIPELINE=1 USE_SAMPLE_IRS=1 USE_SAMPLE_COVARIATES=1 ./run_all.sh
```

**Why Git Bash is suggested on Windows:** `run_all.sh` uses Bash (`unset`, `export`, `./`). PowerShell can run it via `bash ./run_all.sh` once Git Bash or WSL provides `bash`.

---

## 7. Repository layout

| Path | Purpose |
|------|---------|
| `scripts/` | Pipeline scripts (`01`–`12`) and helpers |
| `src/nyfl/` | Python package (IRS/ACS helpers, labels, model specs) |
| `data/raw/` | Inputs you supply or scripts download |
| `data/processed/` | Built datasets and model text outputs |
| `outputs/` | Final tables and figures |

---

License: MIT (`LICENSE`). Third-party data and boundaries follow their respective terms.
