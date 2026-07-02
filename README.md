# Cred_Offline

Offline macroprudential and regional credit-analysis workspace. Models the effects of monetary policy shocks on new lending across three segments — **consumer credit** (ConsCred), **retail/personal lending** (FL), and **mortgages** (Mort) — across Russian regions. Primary outputs are formatted Excel result tables and Word analytical reports in `Отчеты/`.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core analysis and helper scripts |
| Jupyter Notebook | Primary research and modeling workflow |
| pandas / numpy | Data transformation and numerical work |
| statsmodels / scipy | Panel regression, econometric tests, seasonal diagnostics |
| matplotlib / seaborn | Diagnostics and exploratory visualization |
| Excel `.xlsx` | Input datasets and exported analysis artifacts |
| X-13ARIMA-SEATS (`x13as.exe`) | Seasonal adjustment (Windows-local binary) |

---

## Commands

```bash
# Start notebooks
python -m jupyter lab

# Run tests
python -m pytest Modules/test_fun.py

# Syntax check
python -m py_compile Modules/*.py

# Compile package
python -m compileall Modules
```

---

## Structure

```
Cred_Offline/
├── Modules/                  # Canonical Python package — source of truth
│   ├── panel_utils.py        # Panel regression and econometrics core
│   ├── model_results_export.py  # ModelResultsAggregator → Excel exports
│   ├── seasonal_adjustment.py   # X-13 + STL seasonal helpers
│   ├── fun_taylor.py         # Taylor-rule macro calculations
│   └── test_fun.py           # Model-spec helpers and lightweight tests
│
├── *.ipynb                   # Research notebooks
│   ├── Macropru_{ConsCred,FL,Mort}.ipynb   # Macroprudential analysis
│   ├── Model_d_Int_Rate_{ConsCred,FL,Mort}.ipynb  # Interest rate models
│   ├── Clustering_new_*.ipynb  # Region clustering
│   ├── Taylor.ipynb
│   ├── Seasonal_Adjustment.ipynb
│   └── Macropru_Old/         # Archived notebook versions
│
├── Operations/               # Source Excel workbooks (regression inputs)
├── Results/                  # Exported model tables (timestamped .xlsx)
│   ├── MP_clust/             # Macropru cluster-spec results
│   └── MP_dk/                # Macropru dk-spec results
│
├── Отчеты/                   # Word reports (.docx) and changelogs (.md)
│   └── Отдельно по рынкам/   # Per-segment reports
│
├── База данных_рег и фед показатели*.xlsx  # Main regional + federal dataset
├── *.pkl                     # Cached panel data and cluster filters
├── x13as.exe / x13.json      # X-13ARIMA-SEATS binary and config
└── .agent/                   # Agent templates, skills, and references
```

---

## Architecture

**Notebook-first.** Notebooks drive all analysis; `Modules/` provides reusable helpers. The standard import pattern is `from Modules.xxx import ...` — always run notebooks from the repository root.

**Data flow:** Excel inputs (`База данных_рег и фед показатели.xlsx`, `Operations/`) → notebook preprocessing → panel dataset → `Modules/panel_utils.py` estimation → `Modules/model_results_export.py` → `Results/*.xlsx` → Word reports in `Отчеты/`.

**Model spec management:** `Modules/test_fun.py` provides `build_shock_variants` to construct exogenous variable lists per shock type, and `save_model_spec`/`load_models` to persist specs to `Models.pkl`.

**Estimator selection:** Hausman test p-value decides FE vs RE; defaults conservatively to FE on error or missing value.

**Region outlier filtering:** IQR-based region exclusion runs before panel estimation via `panel_utils.detect_outlier_regions_iqr`.

---

## Key Files

| File | Purpose |
|------|---------|
| [`Modules/panel_utils.py`](Modules/panel_utils.py) | Core panel regression, diagnostics, and outlier detection |
| [`Modules/model_results_export.py`](Modules/model_results_export.py) | Aggregates POOL/FE/RE results and exports formatted Excel tables |
| [`Modules/seasonal_adjustment.py`](Modules/seasonal_adjustment.py) | Seasonal diagnostics and X-13-based adjustment |
| [`Modules/fun_taylor.py`](Modules/fun_taylor.py) | Taylor-rule and macro helper calculations |
| [`Modules/test_fun.py`](Modules/test_fun.py) | Model-spec helpers and test suite |
| [`x13.json`](x13.json) | X-13ARIMA-SEATS configuration |

---

## Conventions

- **Canonical package:** always edit `Modules/` — root-level `panel_utils.py`, `model_results_export.py`, and `fun_taylor.py` are duplicates and may diverge.
- **Shock variables:** prefixes `d_Mon_Shock`, `d_ROISFIX`, `d_MIACR`; lag suffix `_lagN` is stripped only for shock vars in export output.
- **Naming:** preserve Russian domain names and transliterated variable names as-is.
- **Type hints:** required on all new or modified `.py` functions.
- **Simplicity:** no speculative helpers, wrappers, or extra modules for single-path changes confined to one file.

---

## Notes

- `x13as.exe` is a Windows-local binary; seasonal adjustment will not work in non-Windows environments.
- `Models.pkl` (model spec store used by `test_fun.py`) is not tracked — it is created at runtime on first `save_model_spec` call.
- `~$` prefixed files in `Отчеты/` are Word lock files; do not edit the corresponding `.docx` while they are present.
- `tmp/` and `.agent/tmp/` directories contain unpacked OOXML fragments from prior document editing sessions and are not active working surfaces.
