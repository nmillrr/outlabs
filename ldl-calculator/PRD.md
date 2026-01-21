# PRD: LDL Cholesterol Estimation Model Development

## Introduction

Develop a clinically-validated LDL cholesterol (LDL-C) estimation algorithm that improves upon the traditional Friedewald equation, particularly for patients with elevated triglyceride levels. The project implements a hybrid approach: mechanistic equations (Friedewald, Martin-Hopkins, Sampson) as baselines, enhanced by a unified machine learning model trained on NHANES data with direct LDL measurements (beta-quantification). The goal is a single hybrid model that performs well across all triglyceride ranges without requiring equation switching.

**Target Users:** Academic researchers, clinicians, laboratory professionals, cardiovascular risk assessment tools  
**Timeline:** 6-month publication-grade development cycle  
**Tech Stack:** Python (NumPy, SciPy, Scikit-Learn, pandas, LightGBM)

## Background

### The Problem with Friedewald

The Friedewald equation (1972) is the traditional standard for calculating LDL-C:

$$\text{LDL-C} = \text{TC} - \text{HDL-C} - \frac{\text{TG}}{5}$$

This formula assumes a fixed 5:1 ratio of triglycerides to VLDL-C, which breaks down when:
- Triglyceride levels exceed 400 mg/dL
- LDL-C is very low (< 70 mg/dL)
- Patients have diabetes, metabolic syndrome, or kidney disease

### Modern Alternatives

**Martin-Hopkins Equation:** Uses a 180-cell lookup table for an adjustable TG:VLDL-C factor based on individual TG and non-HDL-C levels. More accurate at low LDL-C and elevated TG.

**Sampson (NIH Equation 2):** Developed using beta-quantification in a population including high-TG individuals:

$$\text{LDL-C} = \frac{\text{TC}}{0.948} - \frac{\text{HDL-C}}{0.971} - \left(\frac{\text{TG}}{8.56} + \frac{\text{TG} \times \text{non-HDL-C}}{2140} - \frac{\text{TG}^2}{16100}\right) - 9.44$$

### Our Approach

Rather than switching between equations based on TG thresholds, we will train a unified ML model that:
1. Uses all three equations as baseline features
2. Learns corrections from direct LDL measurements (beta-quantification)
3. Provides robust estimates across the full TG range (0-800 mg/dL)

---

## Goals

- Build NHANES data pipeline to acquire lipid panels with direct LDL measurements from ultracentrifugation (~5,000+ samples)
- Implement three mechanistic LDL-C equations with unit tests
- Train unified ML model using equation outputs as features (hybrid approach)
- Validate against beta-quantification reference standard
- Systematically compare all approaches (research focus)
- Package as reproducible Python library + Jupyter notebooks
- Target: Mean bias < ±3 mg/dL, Lin's CCC ≥ 0.95 across all TG ranges

---

## User Stories

### Phase 1: Data Sourcing & Harmonization

---

#### US-001: Create project structure and dependencies
**Description:** As a developer, I want a clean Python project structure so that code is organized and reproducible.

**Acceptance Criteria:**
- [x] Create `ldlC/` package directory with `__init__.py`
- [x] Create `ldlC/models.py`, `ldlC/utils.py`, `ldlC/data.py` (empty modules)
- [x] Create `tests/` directory with `test_models.py` (empty)
- [x] Create `notebooks/` directory
- [x] Create `requirements.txt` with: numpy, scipy, pandas, scikit-learn, lightgbm, matplotlib, pytest
- [x] Typecheck passes (no syntax errors)

---

#### US-002: Implement unit conversion utilities
**Description:** As a developer, I want reliable unit conversion functions so that data from different sources can be harmonized.

**Acceptance Criteria:**
- [x] Add `mg_dl_to_mmol_l(value, molecule='cholesterol')` function in `ldlC/utils.py`
- [x] Add `mmol_l_to_mg_dl(value, molecule='cholesterol')` function
- [x] Support conversions for cholesterol (÷38.67) and triglycerides (÷88.57)
- [x] Add unit tests in `tests/test_utils.py` verifying conversions
- [x] Tests pass: `pytest tests/test_utils.py`

---

#### US-003: Create NHANES lipid download module
**Description:** As a researcher, I want to programmatically download NHANES lipid panel data so that I have a reproducible data pipeline.

**Acceptance Criteria:**
- [x] Add `download_nhanes_lipids(output_dir, cycles)` function in `ldlC/data.py`
- [x] Function downloads TRIGLY, HDL, TCHOL, and LDL (direct) XPT files for cycles 2005-2018
- [x] Creates `data/raw/` directory if not exists
- [x] Handles download errors gracefully with informative messages
- [x] Typecheck passes

---

#### US-004: Implement XPT file parser
**Description:** As a developer, I want to parse NHANES XPT files into pandas DataFrames so that data is usable.

**Acceptance Criteria:**
- [x] Add `read_xpt(filepath)` function in `ldlC/data.py`
- [x] Function reads SAS transport format and returns DataFrame
- [x] Handles missing file with informative error
- [x] Add unit test with mock data
- [x] Tests pass

---

#### US-005: Create NHANES lipid data cleaning pipeline
**Description:** As a researcher, I want automated data cleaning so that outliers and missing values are handled consistently.

**Acceptance Criteria:**
- [x] Add `clean_lipid_data(tc_df, hdl_df, tg_df, ldl_direct_df)` function in `ldlC/data.py`
- [x] Merges datasets on SEQN (sample ID)
- [x] Renames columns to standardized names: tc_mgdl, hdl_mgdl, tg_mgdl, ldl_direct_mgdl
- [x] Removes physiologic outliers (TC < 50, TG > 2000, HDL < 10)
- [x] Calculates non_hdl_mgdl (TC - HDL)
- [x] Returns cleaned DataFrame
- [x] Typecheck passes

---

#### US-006: Generate data quality report
**Description:** As a researcher, I want a quality report so that I can verify data before training.

**Acceptance Criteria:**
- [x] Add `generate_quality_report(df, output_path)` function in `ldlC/data.py`
- [x] Report includes: record count, mean/SD for TC/HDL/TG/LDL-direct, missing value counts
- [x] Includes TG distribution breakdown (<150, 150-400, 400-800, >800 mg/dL)
- [x] Saves report to specified path as text file
- [x] Typecheck passes

---

#### US-007: Create data sourcing notebook
**Description:** As a researcher, I want a notebook documenting the data pipeline so that the process is reproducible.

**Acceptance Criteria:**
- [x] Create `notebooks/01_data_sourcing.ipynb`
- [x] Notebook demonstrates: downloading, parsing, cleaning, quality report
- [x] Includes markdown documentation of each step
- [x] Visualizes TG distribution and LDL ranges
- [x] Notebook executes without errors

---

### Phase 2: Mechanistic Equations

---

#### US-008: Implement Friedewald equation
**Description:** As a developer, I want the Friedewald (1972) LDL-C calculation as the traditional baseline.

**Acceptance Criteria:**
- [x] Add `calc_ldl_friedewald(tc_mgdl, hdl_mgdl, tg_mgdl)` in `ldlC/models.py`
- [x] Formula: LDL = TC - HDL - (TG / 5)
- [x] Returns LDL-C in mg/dL
- [x] Returns NaN for TG > 400 mg/dL (with warning)
- [x] Raises ValueError for invalid inputs (negative, NaN)
- [x] Typecheck passes

---

#### US-009: Test Friedewald against published values
**Description:** As a developer, I want unit tests validating Friedewald against known examples.

**Acceptance Criteria:**
- [x] Add test in `tests/test_models.py` for Friedewald
- [x] Test case: TC=200, HDL=50, TG=150 → LDL = 120 mg/dL
- [x] Test case: TC=180, HDL=45, TG=100 → LDL = 115 mg/dL
- [x] Test high TG warning/NaN behavior
- [x] Tests pass: `pytest tests/test_models.py`

---

#### US-010: Implement Martin-Hopkins equation
**Description:** As a developer, I want the Martin-Hopkins equation with adjustable TG:VLDL factor.

**Acceptance Criteria:**
- [x] Add `calc_ldl_martin_hopkins(tc_mgdl, hdl_mgdl, tg_mgdl)` in `ldlC/models.py`
- [x] Implement 180-cell lookup table for TG:VLDL adjustment factor
- [x] Formula: LDL = TC - HDL - (TG / adjustable_factor)
- [x] Works for TG up to 800 mg/dL
- [x] Add unit test comparing to Friedewald (should differ at extreme values)
- [x] Tests pass

---

#### US-011: Implement Sampson (NIH Equation 2)
**Description:** As a developer, I want the Sampson equation for high-TG accuracy.

**Acceptance Criteria:**
- [x] Add `calc_ldl_sampson(tc_mgdl, hdl_mgdl, tg_mgdl)` in `ldlC/models.py`
- [x] Full formula with quadratic TG term
- [x] Works for TG up to 800 mg/dL
- [x] Add unit test verifying output in valid range (0 < LDL < TC)
- [x] Tests pass

---

#### US-012: Implement extended Martin-Hopkins for very high TG
**Description:** As a developer, I want the extended Martin-Hopkins variant for TG 400-800 mg/dL.

**Acceptance Criteria:**
- [ ] Add `calc_ldl_martin_hopkins_extended(tc_mgdl, hdl_mgdl, tg_mgdl)` in `ldlC/models.py`
- [ ] Uses extended coefficient table for TG > 400 mg/dL
- [ ] Add unit test for high-TG cases
- [ ] Tests pass

---

#### US-013: Create equation comparison notebook
**Description:** As a researcher, I want a notebook comparing all four equations visually.

**Acceptance Criteria:**
- [ ] Create `notebooks/02_equation_comparison.ipynb`
- [ ] Compare Friedewald, Martin-Hopkins, Extended M-H, Sampson on synthetic grid
- [ ] Generate heatmaps showing differences across TG and TC ranges
- [ ] Include markdown interpretation of when each equation excels
- [ ] Notebook executes without errors

---

### Phase 3: ML Model Development

---

#### US-014: Create feature engineering module
**Description:** As a developer, I want feature engineering functions for ML training.

**Acceptance Criteria:**
- [ ] Add `create_features(df)` function in `ldlC/train.py`
- [ ] Creates features: tc_mgdl, hdl_mgdl, tg_mgdl, non_hdl_mgdl
- [ ] Adds ratio features: tg_hdl_ratio, tc_hdl_ratio
- [ ] Adds all equation baseline predictions as features
- [ ] Returns feature matrix X and column names
- [ ] Typecheck passes

---

#### US-015: Implement train/test split with stratification
**Description:** As a developer, I want stratified splitting so that TG subgroups are balanced.

**Acceptance Criteria:**
- [ ] Add `stratified_split(df, test_size=0.3, random_state=42)` in `ldlC/train.py`
- [ ] Stratifies by TG quartiles (< 100, 100-150, 150-200, 200-400, > 400 mg/dL)
- [ ] Returns X_train, X_test, y_train, y_test
- [ ] Typecheck passes

---

#### US-016: Train Ridge regression baseline
**Description:** As a researcher, I want a Ridge regression model as simple baseline.

**Acceptance Criteria:**
- [ ] Add `train_ridge(X_train, y_train, alpha=1.0)` in `ldlC/train.py`
- [ ] Returns fitted Ridge model
- [ ] Add function `save_model(model, filepath)` using joblib
- [ ] Typecheck passes

---

#### US-017: Train Random Forest model
**Description:** As a researcher, I want a Random Forest model for nonlinear patterns.

**Acceptance Criteria:**
- [ ] Add `train_random_forest(X_train, y_train, n_estimators=200)` in `ldlC/train.py`
- [ ] Returns fitted RandomForestRegressor
- [ ] Typecheck passes

---

#### US-018: Train LightGBM model
**Description:** As a researcher, I want a LightGBM model for best performance.

**Acceptance Criteria:**
- [ ] Add `train_lightgbm(X_train, y_train, X_val, y_val)` in `ldlC/train.py`
- [ ] Uses early stopping with 20 rounds
- [ ] Returns fitted LGBMRegressor
- [ ] Typecheck passes

---

#### US-019: Implement cross-validation wrapper
**Description:** As a developer, I want 10-fold CV to evaluate models consistently.

**Acceptance Criteria:**
- [ ] Add `cross_validate_model(model, X, y, n_splits=10)` in `ldlC/train.py`
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
- [ ] Add `bland_altman_stats(y_true, y_pred)` in `ldlC/evaluate.py`
- [ ] Returns dict: mean_bias, std_diff, loa_lower, loa_upper
- [ ] Add unit test with known values
- [ ] Tests pass

---

#### US-022: Implement Lin's CCC
**Description:** As a researcher, I want Lin's Concordance Correlation Coefficient for validation.

**Acceptance Criteria:**
- [ ] Add `lins_ccc(y_true, y_pred)` in `ldlC/evaluate.py`
- [ ] Returns CCC value between -1 and 1
- [ ] Add unit test: identical arrays → CCC = 1.0
- [ ] Tests pass

---

#### US-023: Create comprehensive evaluation function
**Description:** As a researcher, I want a single function returning all metrics.

**Acceptance Criteria:**
- [ ] Add `evaluate_model(y_true, y_pred, model_name)` in `ldlC/evaluate.py`
- [ ] Returns dict: rmse, mae, bias, r_pearson, lin_ccc, ba_stats
- [ ] Typecheck passes

---

#### US-024: Implement TG-stratified evaluation
**Description:** As a researcher, I want to evaluate models by TG subgroups.

**Acceptance Criteria:**
- [ ] Add `evaluate_by_tg_strata(y_true, y_pred, tg_values)` in `ldlC/evaluate.py`
- [ ] Stratifies by: < 150, 150-400, 400-800 mg/dL
- [ ] Returns metrics dict for each stratum
- [ ] Typecheck passes

---

#### US-025: Implement bootstrap confidence intervals
**Description:** As a researcher, I want bootstrap CIs for reporting uncertainty.

**Acceptance Criteria:**
- [ ] Add `bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=2000)` in `ldlC/evaluate.py`
- [ ] Returns (lower, upper, mean) tuple
- [ ] Typecheck passes

---

#### US-026: Create evaluation notebook
**Description:** As a researcher, I want a notebook with full model evaluation and plots.

**Acceptance Criteria:**
- [ ] Create `notebooks/04_evaluation.ipynb`
- [ ] Evaluates all models and equations on internal test set
- [ ] Generates Bland-Altman plots
- [ ] Shows subgroup analysis by TG strata
- [ ] Compares hybrid ML vs individual equations
- [ ] Notebook executes without errors

---

### Phase 5: Package & Documentation

---

#### US-027: Create prediction API
**Description:** As a developer, I want a simple API for making predictions.

**Acceptance Criteria:**
- [ ] Add `predict_ldl(tc, hdl, tg, method='hybrid')` in `ldlC/predict.py`
- [ ] method='friedewald', 'martin_hopkins', 'sampson' uses specific equation
- [ ] method='hybrid' uses best ML model
- [ ] Returns dict: ldl_pred, ci_lower, ci_upper, method, warning (if TG high)
- [ ] Typecheck passes

---

#### US-028: Create package setup.py
**Description:** As a developer, I want the package to be pip-installable.

**Acceptance Criteria:**
- [ ] Create `setup.py` with name='ldlC', version='0.1.0'
- [ ] Includes dependencies from requirements.txt
- [ ] Package installs with `pip install -e .`
- [ ] Import works: `from ldlC import models`

---

#### US-029: Write README with usage examples
**Description:** As a user, I want clear documentation to use the package.

**Acceptance Criteria:**
- [ ] Update README.md with: Installation, Quick Start, API Reference
- [ ] Include code examples for each equation
- [ ] Document when to use hybrid vs specific equations
- [ ] Explain TG threshold considerations
- [ ] README renders correctly on GitHub

---

#### US-030: Run full test suite
**Description:** As a developer, I want all tests passing before release.

**Acceptance Criteria:**
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No test failures or errors
- [ ] Coverage > 70% on core modules

---

## Non-Goals

- **Not building a web application** – Python package and notebooks only
- **Not pursuing regulatory approval** – Research/educational use only
- **Not implementing real-time predictions** – Batch processing is acceptable
- **Not supporting TG > 800 mg/dL** – Direct LDL measurement required for extreme hypertriglyceridemia
- **Not implementing multiple equation switching** – Single unified hybrid model is the goal

## Technical Considerations

- **Data access:** NHANES lipid panels with direct LDL measurements are public
- **Validation standard:** Beta-quantification (direct LDL) from NHANES laboratory data
- **Unit conventions:** All internal calculations in mg/dL (can convert to mmol/L)
- **Martin-Hopkins table:** 180-cell lookup table published in supplementary materials
- **High TG handling:** Model should gracefully handle edge cases, return confidence indicators

## Verification Plan

### Automated Tests
- Run `pytest tests/ -v` after each phase
- All equations verified against published reference values and online calculators
- ML models verified via cross-validation metrics

### Manual Verification
- Notebooks should execute end-to-end without errors
- Compare equation outputs to online LDL calculators
- Final package should be pip-installable: `pip install -e .`
- Research comparison: Generate publication-ready figures comparing all methods
