# PRD: Free Testosterone Estimation Model Development

## Introduction

Develop a clinically-validated free testosterone (FT) estimation algorithm from total testosterone (TT), SHBG, and albumin. The project implements a hybrid approach: mechanistic solvers (Vermeulen, Södergård, Zakharov) as baseline, enhanced by machine learning models trained on public datasets. Deliverables include a Python package and Jupyter notebooks for reproducible research.

**Target Users:** Academic researchers, clinicians in rural healthcare, global health organizations  
**Timeline:** 6-month publication-grade development cycle  
**Tech Stack:** Python (NumPy, SciPy, Scikit-Learn, pandas, LightGBM)

## Goals

- Build NHANES data pipeline to acquire and harmonize ~4,000 training samples
- Implement three mechanistic FT solvers with unit tests
- Train ML models using Vermeulen as baseline feature (hybrid approach)
- Validate on ED-measured reference dataset (EMAS)
- Package as reproducible Python library + Jupyter notebooks
- Target: Mean bias < ±0.5 nmol/L, Lin's CCC ≥ 0.90

---

## User Stories

### Phase 1: Data Sourcing & Harmonization

---

#### US-001: Create project structure and dependencies
**Description:** As a developer, I want a clean Python project structure so that code is organized and reproducible.

**Acceptance Criteria:**
- [x] Create `freeT/` package directory with `__init__.py`
- [x] Create `freeT/models.py`, `freeT/utils.py`, `freeT/data.py` (empty modules)
- [x] Create `tests/` directory with `test_models.py` (empty)
- [x] Create `notebooks/` directory
- [x] Create `requirements.txt` with: numpy, scipy, pandas, scikit-learn, lightgbm, matplotlib, pytest
- [x] Typecheck passes (no syntax errors)

---

#### US-002: Implement unit conversion utilities
**Description:** As a developer, I want reliable unit conversion functions so that data from different sources can be harmonized.

**Acceptance Criteria:**
- [x] Add `ng_dl_to_nmol_l(value)` function in `freeT/utils.py`
- [x] Add `nmol_l_to_ng_dl(value)` function
- [x] Add `mg_dl_to_g_l(value)` and `g_l_to_mg_dl(value)` functions
- [x] Add unit tests in `tests/test_utils.py` verifying conversions
- [x] Tests pass: `pytest tests/test_utils.py`

---

#### US-003: Create NHANES download module
**Description:** As a researcher, I want to programmatically download NHANES testosterone data so that I have a reproducible data pipeline.

**Acceptance Criteria:**
- [x] Add `download_nhanes(output_dir, cycles)` function in `freeT/data.py`
- [x] Function downloads TST, SHBG, ALB XPT files for cycles 2011-2016
- [x] Creates `data/raw/` directory if not exists
- [x] Handles download errors gracefully with informative messages
- [x] Typecheck passes

---

#### US-004: Implement XPT file parser
**Description:** As a developer, I want to parse NHANES XPT files into pandas DataFrames so that data is usable.

**Acceptance Criteria:**
- [x] Add `read_xpt(filepath)` function in `freeT/data.py`
- [x] Function reads SAS transport format and returns DataFrame
- [x] Handles missing file with informative error
- [x] Add unit test with mock data
- [x] Tests pass

---

#### US-005: Create NHANES data cleaning pipeline
**Description:** As a researcher, I want automated data cleaning so that outliers and missing values are handled consistently.

**Acceptance Criteria:**
- [x] Add `clean_nhanes_data(tst_df, shbg_df, alb_df)` function in `freeT/data.py`
- [x] Merges datasets on SEQN (sample ID)
- [x] Applies unit conversions (TT: ng/dL → nmol/L, Albumin: g/dL → g/L)
- [x] Removes physiologic outliers (TT < 0.5, SHBG > 250, Alb < 30)
- [x] Returns cleaned DataFrame with standardized column names
- [x] Typecheck passes


---

#### US-006: Generate data quality report
**Description:** As a researcher, I want a quality report so that I can verify data before training.

**Acceptance Criteria:**
- [x] Add `generate_quality_report(df, output_path)` function in `freeT/data.py`
- [x] Report includes: record count, mean/SD for TT/SHBG/Albumin, missing value counts
- [x] Saves report to specified path as text file
- [x] Typecheck passes


---

#### US-007: Create data sourcing notebook
**Description:** As a researcher, I want a notebook documenting the data pipeline so that the process is reproducible.

**Acceptance Criteria:**
- [x] Create `notebooks/01_data_sourcing.ipynb`
- [x] Notebook demonstrates: downloading, parsing, cleaning, quality report
- [x] Includes markdown documentation of each step
- [x] Notebook executes without errors

---

### Phase 2: Mechanistic Solvers

---

#### US-008: Implement Vermeulen cubic solver
**Description:** As a developer, I want the Vermeulen (1999) FT calculation so that I have a validated baseline.

**Acceptance Criteria:**
- [x] Add `calc_ft_vermeulen(tt_nmoll, shbg_nmoll, alb_gl, K_shbg=1e9, K_alb=3.6e4)` in `freeT/models.py`
- [x] Uses scipy.optimize.brentq for root-finding
- [x] Returns FT in nmol/L
- [x] Raises ValueError for invalid inputs (negative, NaN)
- [x] Typecheck passes

---

#### US-009: Test Vermeulen solver against published values
**Description:** As a developer, I want unit tests validating Vermeulen against published examples.

**Acceptance Criteria:**
- [x] Add test in `tests/test_models.py` for Vermeulen
- [x] Test case: TT=15, SHBG=40, Alb=45 → FT ≈ 0.30 nmol/L
- [x] Test case: TT=10, SHBG=20, Alb=42 → verify against ISSAM calculator
- [x] Tests pass: `pytest tests/test_models.py`

---

#### US-010: Implement Södergård solver variant
**Description:** As a developer, I want the Södergård solver so that I can compare equation variants.

**Acceptance Criteria:**
- [x] Add `calc_ft_sodergard(tt_nmoll, shbg_nmoll, alb_gl)` in `freeT/models.py`
- [x] Uses K_shbg=1.2e9, K_alb=2.4e4
- [x] Internally calls Vermeulen with modified constants
- [x] Add unit test comparing to Vermeulen (should differ slightly)
- [x] Tests pass

---

#### US-011: Implement Zakharov allosteric solver (simplified)
**Description:** As a developer, I want the Zakharov solver so that I have all three reference methods.

**Acceptance Criteria:**
- [ ] Add `calc_ft_zakharov(tt_nmoll, shbg_nmoll, alb_gl, cooperativity=0.5)` in `freeT/models.py`
- [ ] Uses scipy.optimize.fsolve for nonlinear system
- [ ] Add unit test verifying output in valid range (0 < FT < TT)
- [ ] Tests pass

---

#### US-012: Create bioavailable testosterone function
**Description:** As a developer, I want to calculate bioavailable testosterone so the package is complete.

**Acceptance Criteria:**
- [ ] Add `calc_bioavailable_t(tt_nmoll, shbg_nmoll, alb_gl)` in `freeT/models.py`
- [ ] Bioavailable = FT + albumin-bound fraction
- [ ] Add unit test verifying Bioavailable > FT
- [ ] Tests pass

---

#### US-013: Create solver comparison notebook
**Description:** As a researcher, I want a notebook comparing the three solvers visually.

**Acceptance Criteria:**
- [ ] Create `notebooks/02_solver_comparison.ipynb`
- [ ] Compare Vermeulen, Södergård, Zakharov on synthetic grid (TT: 5-30, SHBG: 10-80)
- [ ] Generate scatter/line plots showing differences
- [ ] Include markdown interpretation of results
- [ ] Notebook executes without errors

---

### Phase 3: ML Model Development

---

#### US-014: Create feature engineering module
**Description:** As a developer, I want feature engineering functions for ML training.

**Acceptance Criteria:**
- [ ] Add `create_features(df)` function in `freeT/train.py`
- [ ] Creates features: tt_nmoll, shbg_nmoll, alb_gl, shbg_tt_ratio
- [ ] Adds ft_vermeulen baseline feature
- [ ] Returns feature matrix X and column names
- [ ] Typecheck passes

---

#### US-015: Implement train/test split with stratification
**Description:** As a developer, I want stratified splitting so that subgroups are balanced.

**Acceptance Criteria:**
- [ ] Add `stratified_split(df, test_size=0.3, random_state=42)` in `freeT/train.py`
- [ ] Stratifies by SHBG tertiles
- [ ] Returns X_train, X_test, y_train, y_test
- [ ] Typecheck passes

---

#### US-016: Train Ridge regression baseline
**Description:** As a researcher, I want a Ridge regression model as simple baseline.

**Acceptance Criteria:**
- [ ] Add `train_ridge(X_train, y_train, alpha=1.0)` in `freeT/train.py`
- [ ] Returns fitted Ridge model
- [ ] Add function `save_model(model, filepath)` using joblib
- [ ] Typecheck passes

---

#### US-017: Train Random Forest model
**Description:** As a researcher, I want a Random Forest model for nonlinear patterns.

**Acceptance Criteria:**
- [ ] Add `train_random_forest(X_train, y_train, n_estimators=200)` in `freeT/train.py`
- [ ] Returns fitted RandomForestRegressor
- [ ] Typecheck passes

---

#### US-018: Train LightGBM model
**Description:** As a researcher, I want a LightGBM model for best performance.

**Acceptance Criteria:**
- [ ] Add `train_lightgbm(X_train, y_train, X_val, y_val)` in `freeT/train.py`
- [ ] Uses early stopping with 20 rounds
- [ ] Returns fitted LGBMRegressor
- [ ] Typecheck passes

---

#### US-019: Implement cross-validation wrapper
**Description:** As a developer, I want 10-fold CV to evaluate models consistently.

**Acceptance Criteria:**
- [ ] Add `cross_validate_model(model, X, y, n_splits=10)` in `freeT/train.py`
- [ ] Returns dict with RMSE_mean, RMSE_std, MAE_mean, MAE_std
- [ ] Typecheck passes

---

#### US-020: Create model training notebook
**Description:** As a researcher, I want a notebook documenting model training workflow.

**Acceptance Criteria:**
- [ ] Create `notebooks/03_model_training.ipynb`
- [ ] Trains Ridge, RF, LightGBM on NHANES data
- [ ] Shows CV results comparison table
- [ ] Saves best model to `models/` directory
- [ ] Notebook executes without errors (may need data)

---

### Phase 4: Evaluation & Validation

---

#### US-021: Implement Bland-Altman analysis
**Description:** As a researcher, I want Bland-Altman statistics for agreement analysis.

**Acceptance Criteria:**
- [ ] Add `bland_altman_stats(y_true, y_pred)` in `freeT/evaluate.py`
- [ ] Returns dict: mean_bias, std_diff, loa_lower, loa_upper
- [ ] Add unit test with known values
- [ ] Tests pass

---

#### US-022: Implement Lin's CCC
**Description:** As a researcher, I want Lin's Concordance Correlation Coefficient for validation.

**Acceptance Criteria:**
- [ ] Add `lins_ccc(y_true, y_pred)` in `freeT/evaluate.py`
- [ ] Returns CCC value between -1 and 1
- [ ] Add unit test: identical arrays → CCC = 1.0
- [ ] Tests pass

---

#### US-023: Create comprehensive evaluation function
**Description:** As a researcher, I want a single function returning all metrics.

**Acceptance Criteria:**
- [ ] Add `evaluate_model(y_true, y_pred, model_name)` in `freeT/evaluate.py`
- [ ] Returns dict: rmse, mae, bias, r_pearson, lin_ccc, ba_stats
- [ ] Typecheck passes

---

#### US-024: Implement bootstrap confidence intervals
**Description:** As a researcher, I want bootstrap CIs for reporting uncertainty.

**Acceptance Criteria:**
- [ ] Add `bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=2000)` in `freeT/evaluate.py`
- [ ] Returns (lower, upper, mean) tuple
- [ ] Typecheck passes

---

#### US-025: Create evaluation notebook
**Description:** As a researcher, I want a notebook with full model evaluation and plots.

**Acceptance Criteria:**
- [ ] Create `notebooks/04_evaluation.ipynb`
- [ ] Evaluates all models on internal test set
- [ ] Generates Bland-Altman plots
- [ ] Shows subgroup analysis by SHBG tertile
- [ ] Notebook executes without errors

---

### Phase 5: Package & Documentation

---

#### US-026: Create prediction API
**Description:** As a developer, I want a simple API for making predictions.

**Acceptance Criteria:**
- [ ] Add `predict_ft(tt, shbg, alb, method='hybrid')` in `freeT/predict.py`
- [ ] method='vermeulen' uses mechanistic solver
- [ ] method='hybrid' uses best ML model
- [ ] Returns dict: ft_pred, ci_lower, ci_upper, method
- [ ] Typecheck passes

---

#### US-027: Create package setup.py
**Description:** As a developer, I want the package to be pip-installable.

**Acceptance Criteria:**
- [ ] Create `setup.py` with name='freeT', version='0.1.0'
- [ ] Includes dependencies from requirements.txt
- [ ] Package installs with `pip install -e .`
- [ ] Import works: `from freeT import models`

---

#### US-028: Write README with usage examples
**Description:** As a user, I want clear documentation to use the package.

**Acceptance Criteria:**
- [ ] Update README.md with: Installation, Quick Start, API Reference
- [ ] Include code examples for each solver
- [ ] Document data pipeline usage
- [ ] README renders correctly on GitHub

---

#### US-029: Run full test suite
**Description:** As a developer, I want all tests passing before release.

**Acceptance Criteria:**
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No test failures or errors
- [ ] Coverage > 70% on core modules

---

---

## Non-Goals

- **Not building a web application** – Python package and notebooks only
- **Not pursuing regulatory approval** – Research/educational use only
- **Not training on EMAS data** – Reserved strictly for external validation
- **Not implementing deep learning** – Neural networks out of scope for v1
- **Not supporting real-time predictions** – Batch processing is acceptable

## Technical Considerations

- **Existing documentation:** The whitepaper (`FT_Model_Whitepaper.md`) contains detailed derivations
- **Data access:** NHANES is public; EMAS requires author contact
- **Validation standard:** ED-measured FT from Fiers et al. (2018)
- **Unit conventions:** All internal calculations in SI units (nmol/L, g/L)

## Verification Plan

### Automated Tests
- Run `pytest tests/ -v` after each phase
- All solvers verified against published reference values (ISSAM calculator)
- ML models verified via cross-validation metrics

### Manual Verification
- Notebooks should execute end-to-end without errors
- Compare Vermeulen output to online ISSAM calculator: https://www.issam.ch/freetesto.htm
- Final package should be pip-installable: `pip install -e .`
