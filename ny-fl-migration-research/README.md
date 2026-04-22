# NY → Florida migration (county-level research pipeline)

Reproducible analysis of NY→FL county flows (IRS 2021–2022): OLS sorting models, gravity-style GLMs, optional boosting-based housing signals.

**Run everything with `./run_all.sh` after installing dependencies.**

**Requirements:** Python 3.10+ (3.11+ recommended), `pip`, and `git`.  
On Windows, **Git for Windows** (includes Git Bash) is recommended, or use WSL.

---

## 1. Clone the repository

```bash
git clone https://github.com/Strzalaa/ds340W_Final.git
cd ds340W_Final/ny-fl-migration-research
```

⚠️ **Important:** All commands below must be run inside `ny-fl-migration-research/`, where `requirements.txt` is located.

---

## 2. Create a virtual environment and install dependencies

### macOS / Linux (Terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows — Git Bash (recommended)

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows — PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
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

From the same folder:

```powershell
$env:USE_SAMPLE_IRS=""
$env:USE_SAMPLE_COVARIATES=""
$env:ALLOW_SAMPLE_PIPELINE=""
$env:PYTHONPATH = "$PWD\src"
bash ./run_all.sh
```

If `bash` is not found, install Git for Windows and run from Git Bash instead.

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

**Offline / sample data (testing only):**

```bash
ALLOW_SAMPLE_PIPELINE=1 USE_SAMPLE_IRS=1 USE_SAMPLE_COVARIATES=1 ./run_all.sh
```

---

## 7. Repository layout

| Path | Purpose |
|------|---------|
| `scripts/` | Pipeline scripts (01–12) and helpers |
| `src/nyfl/` | Python package |
| `data/raw/` | Inputs |
| `data/processed/` | Built datasets and outputs |
| `outputs/` | Final tables and figures |

---

License: MIT (`LICENSE`). Third-party data and boundaries follow their respective terms.
