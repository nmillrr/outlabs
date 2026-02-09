# PRD: HbA1c Estimation from Routine Blood Markers

## Introduction

Develop a clinically-validated HbA1c estimation library that predicts glycated hemoglobin from fasting plasma glucose, lipid panels, and demographic factors—reducing dependence on specialized HbA1c assays in resource-limited settings. The project implements a hybrid approach: mechanistic estimators (ADAG inversion, glycation kinetics) as baselines, enhanced by machine learning models trained on NHANES data with HPLC-measured HbA1c as ground truth.

**Target Users:** Academic researchers, clinicians in resource-limited settings, global health organizations, diabetes screening programs  
**Timeline:** 6-month publication-grade development cycle  
**Tech Stack:** Python (NumPy, SciPy, Scikit-Learn, pandas, LightGBM)

## Background

### The Clinical Problem

Direct HbA1c measurement requires expensive, specialized equipment:
- **HPLC** (High-Performance Liquid Chromatography) — Gold standard, expensive equipment
- **Immunoassay** — Requires specialized reagents
- **Boronate affinity** — Needs dedicated analyzer

Many rural clinics, low-resource hospitals, and global health settings lack access to HbA1c testing but **do have** routine glucose and basic chemistry panels.

### What is HbA1c?

Hemoglobin A1c measures the percentage of hemoglobin proteins with glucose attached. Because red blood cells live ~120 days, HbA1c reflects average blood glucose over 2-3 months—the gold standard for diabetes diagnosis and monitoring.

### Our Approach

Unlike Free Testosterone or LDL-C, there is no universally established mechanistic equation for HbA1c. This project uses:

1. **Empirical Regression Models** — ADAG inversion and multi-linear regression
2. **Glycation Kinetics** — First-order model adjusted for hemoglobin and RBC lifespan
3. **Machine Learning** — Ensemble models trained on NHANES glycemic data
4. **Multi-marker Integration** — Combining glucose with lipids, hemoglobin, age for improved accuracy

---

## Goals

- Build NHANES data pipeline to acquire glycemic panels with HPLC-measured HbA1c (~10,000+ samples)
- Implement two mechanistic HbA1c estimators with unit tests (ADAG, glycation kinetics)
- Train unified ML model using multi-marker features (hybrid approach)
- Validate against HPLC reference standard from NHANES
- Seek external validation dataset if available (fallback: NHANES-only validation)
- Perform full subgroup analysis (anemia, pregnancy, hemoglobinopathies, CKD, age extremes)
- Package as reproducible Python library + Jupyter notebooks
- Target: RMSE < 0.5% HbA1c, Mean bias < ±0.2%, Lin's CCC ≥ 0.85

---

## User Stories

### Phase 1: Data Sourcing & Harmonization

---

#### US-001: Create project structure and dependencies
**Description:** As a developer, I want a clean Python project structure so that code is organized and reproducible.

**Acceptance Criteria:**
- [x] Create `hba1cE/` package directory with `__init__.py`
- [x] Create `hba1cE/models.py`, `hba1cE/utils.py`, `hba1cE/data.py` (empty modules)
- [x] Create `tests/` directory with `test_models.py` (empty)
- [x] Create `notebooks/` directory
- [x] Create `requirements.txt` with: numpy, scipy, pandas, scikit-learn, lightgbm, matplotlib, pytest
- [x] Typecheck passes (no syntax errors)

---

#### US-002: Implement unit conversion utilities
**Description:** As a developer, I want reliable unit conversion functions so that data from different sources can be harmonized.

**Acceptance Criteria:**
- [x] Add `mg_dl_to_mmol_l(glucose_mgdl)` function in `hba1cE/utils.py` (glucose: ÷18.018)
- [x] Add `mmol_l_to_mg_dl(glucose_mmol)` function
- [x] Add `percent_to_mmol_mol(hba1c_percent)` for HbA1c NGSP→IFCC conversion
- [x] Add `mmol_mol_to_percent(hba1c_mmol)` for IFCC→NGSP conversion
- [x] Add unit tests in `tests/test_utils.py` verifying conversions
- [x] Tests pass: `pytest tests/test_utils.py`

---

#### US-003: Create NHANES glycemic download module
**Description:** As a researcher, I want to programmatically download NHANES glycemic data so that I have a reproducible data pipeline.

**Acceptance Criteria:**
- [x] Add `download_nhanes_glycemic(output_dir, cycles)` function in `hba1cE/data.py`
- [x] Function downloads GHB (HbA1c), GLU (fasting glucose), TRIGLY, HDL, CBC XPT files for cycles 2011-2018
- [x] Downloads DEMO files for age/sex demographics
- [x] Creates `data/raw/` directory if not exists
- [x] Handles download errors gracefully with informative messages
- [x] Typecheck passes

---

#### US-004: Implement XPT file parser
**Description:** As a developer, I want to parse NHANES XPT files into pandas DataFrames so that data is usable.

**Acceptance Criteria:**
- [x] Add `read_xpt(filepath)` function in `hba1cE/data.py`
- [x] Function reads SAS transport format and returns DataFrame
- [x] Handles missing file with informative error
- [x] Add unit test with mock data
- [x] Tests pass

---

#### US-005: Create NHANES glycemic data cleaning pipeline
**Description:** As a researcher, I want automated data cleaning so that outliers and missing values are handled consistently.

**Acceptance Criteria:**
- [x] Add `clean_glycemic_data(ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df)` function in `hba1cE/data.py`
- [x] Merges datasets on SEQN (sample ID)
- [x] Renames columns to: hba1c_percent (LBXGH), fpg_mgdl (LBXGLU), tg_mgdl (LBXTR), hdl_mgdl (LBDHDD), hgb_gdl (LBXHGB), mcv_fl (LBXMCVSI), age_years (RIDAGEYR), sex (RIAGENDR)
- [x] Removes physiologic outliers (HbA1c < 3% or > 20%, FPG < 40 or > 600 mg/dL)
- [x] Returns cleaned DataFrame with complete cases only
- [x] Typecheck passes

---

#### US-006: Generate data quality report
**Description:** As a researcher, I want a quality report so that I can verify data before training.

**Acceptance Criteria:**
- [x] Add `generate_quality_report(df, output_path)` function in `hba1cE/data.py`
- [x] Report includes: record count, mean/SD for FPG/HbA1c/TG/HDL/Hgb/MCV
- [x] Includes HbA1c distribution breakdown (<5.7%, 5.7-6.4%, ≥6.5%)
- [x] Includes FPG distribution breakdown (<100, 100-125, ≥126 mg/dL)
- [x] Saves report to specified path as text file
- [x] Typecheck passes

---

#### US-007: Create data sourcing notebook
**Description:** As a researcher, I want a notebook documenting the data pipeline so that the process is reproducible.

**Acceptance Criteria:**
- [x] Create `notebooks/01_data_sourcing.ipynb`
- [x] Notebook demonstrates: downloading, parsing, cleaning, quality report
- [x] Includes markdown documentation of each step
- [x] Visualizes HbA1c vs FPG scatter plot and distributions
- [x] Notebook executes without errors

---

### Phase 2: Mechanistic Estimators

---

#### US-008: Implement ADAG equation (inverted)
**Description:** As a developer, I want the ADAG-derived HbA1c estimation as the simplest baseline.

**Acceptance Criteria:**
- [x] Add `calc_hba1c_adag(fpg_mgdl)` in `hba1cE/models.py`
- [x] Formula: eHbA1c = (FPG + 46.7) / 28.7
- [x] Returns estimated HbA1c in percent
- [x] Raises ValueError for invalid inputs (negative, NaN, FPG < 40)
- [x] Add docstring with reference to Nathan et al. (2008)
- [x] Typecheck passes

---

#### US-009: Test ADAG against expected values
**Description:** As a developer, I want unit tests validating ADAG against known examples.

**Acceptance Criteria:**
- [x] Add test in `tests/test_models.py` for ADAG
- [x] Test case: FPG=126 mg/dL → eHbA1c ≈ 6.0%
- [x] Test case: FPG=154 mg/dL → eHbA1c ≈ 7.0%
- [x] Test invalid input handling (negative, NaN)
- [x] Tests pass: `pytest tests/test_models.py`

---

#### US-010: Implement glycation kinetics model
**Description:** As a developer, I want a first-order glycation kinetics model adjusted for hemoglobin.

**Acceptance Criteria:**
- [x] Add `calc_hba1c_kinetic(fpg_mgdl, hgb_gdl=14.0, rbc_lifespan_days=120)` in `hba1cE/models.py`
- [x] Formula: HbA1c = k × [Glucose_avg] × RBC_lifespan / Hemoglobin_factor
- [x] Default k = 4.5 × 10⁻⁵ per mg/dL per day (adjustable parameter)
- [x] Adjusts for hemoglobin level (anemia correction)
- [x] Add unit test verifying output in valid range (3% < HbA1c < 20%)
- [x] Typecheck passes


---

#### US-011: Implement multi-linear regression estimator
**Description:** As a developer, I want a multi-linear regression model using published NHANES coefficients.

**Acceptance Criteria:**
- [x] Add `calc_hba1c_regression(fpg_mgdl, age_years, tg_mgdl, hdl_mgdl, hgb_gdl)` in `hba1cE/models.py`
- [x] Formula: eHbA1c = β₀ + β₁×FPG + β₂×Age + β₃×TG + β₄×HDL + β₅×Hgb
- [x] Use placeholder coefficients initially (will be fitted from data)
- [x] Add `fit_regression_coefficients(df)` to derive coefficients from NHANES
- [x] Typecheck passes

---

#### US-012: Create estimator comparison notebook
**Description:** As a researcher, I want a notebook comparing mechanistic estimators visually.

**Acceptance Criteria:**
- [x] Create `notebooks/02_estimator_comparison.ipynb`
- [x] Compare ADAG vs Kinetic vs Regression on NHANES data
- [x] Generate scatter plots: estimated vs measured HbA1c for each method
- [x] Include markdown interpretation of each estimator's limitations
- [x] Notebook executes without errors

---

### Phase 3: ML Model Development

---

#### US-013: Create feature engineering module
**Description:** As a developer, I want feature engineering functions for ML training.

**Acceptance Criteria:**
- [x] Add `create_features(df)` function in `hba1cE/train.py`
- [x] Creates features: fpg_mgdl, tg_mgdl, hdl_mgdl, age_years, hgb_gdl, mcv_fl
- [x] Adds ratio features: tg_hdl_ratio, fpg_age_interaction
- [x] Adds all mechanistic estimator predictions as features (ADAG, kinetic, regression)
- [x] Returns feature matrix X and column names
- [x] Typecheck passes

---

#### US-014: Implement train/test split with stratification
**Description:** As a developer, I want stratified splitting so that HbA1c subgroups are balanced.

**Acceptance Criteria:**
- [ ] Add `stratified_split(df, test_size=0.3, random_state=42)` in `hba1cE/train.py`
- [ ] Stratifies by HbA1c ranges (<5.7%, 5.7-6.4%, 6.5-8%, 8-10%, >10%)
- [ ] Returns X_train, X_test, y_train, y_test
- [ ] Typecheck passes

---

#### US-015: Train Ridge regression baseline
**Description:** As a researcher, I want a Ridge regression model as simple ML baseline.

**Acceptance Criteria:**
- [ ] Add `train_ridge(X_train, y_train, alpha=1.0)` in `hba1cE/train.py`
- [ ] Returns fitted Ridge model
- [ ] Add function `save_model(model, filepath)` using joblib
- [ ] Typecheck passes

---

#### US-016: Train Random Forest model
**Description:** As a researcher, I want a Random Forest model for nonlinear patterns.

**Acceptance Criteria:**
- [ ] Add `train_random_forest(X_train, y_train, n_estimators=200)` in `hba1cE/train.py`
- [ ] Returns fitted RandomForestRegressor
- [ ] Typecheck passes

---

#### US-017: Train LightGBM model
**Description:** As a researcher, I want a LightGBM model for best performance.

**Acceptance Criteria:**
- [ ] Add `train_lightgbm(X_train, y_train, X_val, y_val)` in `hba1cE/train.py`
- [ ] Uses early stopping with 20 rounds
- [ ] Returns fitted LGBMRegressor
- [ ] Typecheck passes

---

#### US-018: Implement cross-validation wrapper
**Description:** As a developer, I want 10-fold CV to evaluate models consistently.

**Acceptance Criteria:**
- [ ] Add `cross_validate_model(model, X, y, n_splits=10)` in `hba1cE/train.py`
- [ ] Returns dict with RMSE_mean, RMSE_std, MAE_mean, MAE_std
- [ ] Typecheck passes

---

#### US-019: Create model training notebook
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

#### US-020: Implement Bland-Altman analysis
**Description:** As a researcher, I want Bland-Altman statistics for agreement analysis.

**Acceptance Criteria:**
- [ ] Add `bland_altman_stats(y_true, y_pred)` in `hba1cE/evaluate.py`
- [ ] Returns dict: mean_bias, std_diff, loa_lower, loa_upper
- [ ] Add unit test with known values
- [ ] Tests pass

---

#### US-021: Implement Lin's CCC
**Description:** As a researcher, I want Lin's Concordance Correlation Coefficient for validation.

**Acceptance Criteria:**
- [ ] Add `lins_ccc(y_true, y_pred)` in `hba1cE/evaluate.py`
- [ ] Returns CCC value between -1 and 1
- [ ] Add unit test: identical arrays → CCC = 1.0
- [ ] Tests pass

---

#### US-022: Create comprehensive evaluation function
**Description:** As a researcher, I want a single function returning all metrics.

**Acceptance Criteria:**
- [ ] Add `evaluate_model(y_true, y_pred, model_name)` in `hba1cE/evaluate.py`
- [ ] Returns dict: rmse, mae, bias, r_pearson, lin_ccc, ba_stats, pct_within_0_5
- [ ] pct_within_0_5 = percentage of predictions within ±0.5% of measured
- [ ] Typecheck passes

---

#### US-023: Implement HbA1c-stratified evaluation
**Description:** As a researcher, I want to evaluate models by HbA1c clinical ranges.

**Acceptance Criteria:**
- [ ] Add `evaluate_by_hba1c_strata(y_true, y_pred, hba1c_values)` in `hba1cE/evaluate.py`
- [ ] Stratifies by: <5.7% (normal), 5.7-6.4% (prediabetes), ≥6.5% (diabetes)
- [ ] Returns metrics dict for each stratum
- [ ] Typecheck passes

---

#### US-024: Implement subgroup evaluation (anemia, age)
**Description:** As a researcher, I want to evaluate models in clinically relevant subgroups.

**Acceptance Criteria:**
- [ ] Add `evaluate_by_subgroup(y_true, y_pred, df, subgroup_col, subgroup_values)` in `hba1cE/evaluate.py`
- [ ] Add `define_subgroups(df)` function that creates subgroup columns:
  - anemia: Hgb < 12 g/dL (female) or < 13 g/dL (male)
  - age_group: <40, 40-60, >60 years
  - mcv_group: low (<80), normal (80-100), high (>100) fL
- [ ] Returns metrics dict for each subgroup
- [ ] Typecheck passes

---

#### US-025: Implement bootstrap confidence intervals
**Description:** As a researcher, I want bootstrap CIs for reporting uncertainty.

**Acceptance Criteria:**
- [ ] Add `bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=2000)` in `hba1cE/evaluate.py`
- [ ] Returns (lower, upper, mean) tuple
- [ ] Typecheck passes

---

#### US-026: Search for external validation datasets
**Description:** As a researcher, I want to identify potential external validation datasets.

**Acceptance Criteria:**
- [ ] Research and document potential external datasets in `notebooks/external_data_sources.md`
- [ ] Consider: UK Biobank, ARIC, Framingham, MESA (if accessible)
- [ ] Document access requirements and data variables available
- [ ] If no accessible dataset, document NHANES-only validation plan
- [ ] Typecheck passes (no code changes, documentation only)

---

#### US-027: Create evaluation notebook
**Description:** As a researcher, I want a notebook with full model evaluation and plots.

**Acceptance Criteria:**
- [ ] Create `notebooks/04_evaluation.ipynb`
- [ ] Evaluates all models and estimators on internal test set
- [ ] Generates Bland-Altman plots
- [ ] Shows subgroup analysis (anemia, age, HbA1c strata)
- [ ] Compares hybrid ML vs individual estimators
- [ ] Reports % within ±0.5% threshold
- [ ] Notebook executes without errors

---

### Phase 5: Package & Documentation

---

#### US-028: Create prediction API
**Description:** As a developer, I want a simple API for making predictions.

**Acceptance Criteria:**
- [ ] Add `predict_hba1c(fpg, tg=None, hdl=None, age=None, hgb=None, mcv=None, method='hybrid')` in `hba1cE/predict.py`
- [ ] method='adag', 'kinetic', 'regression' uses specific estimator
- [ ] method='hybrid' uses best ML model (accepts partial inputs gracefully)
- [ ] Returns dict: hba1c_pred, ci_lower, ci_upper, method, warning (if inputs incomplete)
- [ ] Typecheck passes

---

#### US-029: Create package setup.py
**Description:** As a developer, I want the package to be pip-installable.

**Acceptance Criteria:**
- [ ] Create `setup.py` with name='hba1cE', version='0.1.0'
- [ ] Includes dependencies from requirements.txt
- [ ] Package installs with `pip install -e .`
- [ ] Import works: `from hba1cE import models`

---

#### US-030: Write README with usage examples
**Description:** As a user, I want clear documentation to use the package.

**Acceptance Criteria:**
- [ ] Update README.md with: Installation, Quick Start, API Reference
- [ ] Include code examples for each estimator
- [ ] Document clinical limitations and when direct HbA1c is required
- [ ] Explain subgroup considerations (anemia, hemoglobinopathies)
- [ ] README renders correctly on GitHub

---

#### US-031: Run full test suite
**Description:** As a developer, I want all tests passing before release.

**Acceptance Criteria:**
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No test failures or errors
- [ ] Coverage > 70% on core modules

---

## Non-Goals

- **Not building a web application** — Python package and notebooks only
- **Not pursuing regulatory approval** — Research/educational use only
- **Not replacing direct HbA1c measurement** — Estimation for screening only
- **Not supporting hemoglobin variants** — HbS, HbC, HbE require special handling
- **Not implementing CGM-based estimation** — Only using routine bloodwork
- **Not building stacked ensemble initially** — Single best ML model for MVP

## Technical Considerations

- **Data access:** NHANES glycemic panels with HPLC-measured HbA1c are public
- **Validation standard:** HPLC-measured HbA1c (LBXGH) from NHANES laboratory data
- **Unit conventions:** All internal calculations in mg/dL and %; can convert to mmol/L and mmol/mol
- **Clinical thresholds:** HbA1c 5.7% (prediabetes), 6.5% (diabetes) — errors near thresholds have high clinical impact
- **Subgroup handling:** Model should flag high-uncertainty cases (anemia, extreme age)
- **External validation:** Prioritize NHANES internal validation; pursue external if datasets accessible

## Verification Plan

### Automated Tests
- Run `pytest tests/ -v` after each phase
- All estimators verified against published reference values
- ML models verified via cross-validation metrics
- Target: RMSE < 0.5%, % within ±0.5% > 80%

### Manual Verification
- Notebooks should execute end-to-end without errors
- Compare estimator outputs to published ADAG values
- Final package should be pip-installable: `pip install -e .`
- Research comparison: Generate publication-ready figures comparing all methods
