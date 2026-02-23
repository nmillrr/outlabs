# PRD: eGFR Estimation from Routine Blood Markers

## Introduction

Develop a clinically-validated estimated Glomerular Filtration Rate (eGFR) library that calculates kidney function from serum creatinine, cystatin C, and demographic factors—enabling kidney disease screening in clinics without access to nuclear medicine GFR testing. The project implements multiple established equations (CKD-EPI 2021, MDRD, Cockcroft-Gault) as mechanistic baselines, enhanced by a hybrid ML model trained on NHANES data with measured GFR from external CKD cohorts as validation reference.

**Target Users:** Rural clinics, low-resource hospitals, global health organizations, CKD screening programs  
**Timeline:** 6-month publication-grade development cycle  
**Tech Stack:** Python (NumPy, SciPy, Scikit-Learn, pandas, LightGBM)

## Background

### The Clinical Problem

Direct GFR measurement requires expensive, invasive procedures:
- **Iothalamate clearance** — Nuclear medicine, IV injection + timed urine collection
- **Iohexol clearance** — Contrast agent injection + serial blood draws
- **Inulin clearance** — Gold standard, extremely labor-intensive

Most rural clinics and low-resource hospitals lack access to measured GFR but **do have** routine chemistry panels with serum creatinine, and many can run cystatin C assays.

### What is eGFR?

Glomerular Filtration Rate measures how much blood the kidneys filter per minute (mL/min/1.73 m²). It is the single best indicator of kidney function and is used for:
- **CKD staging** (G1–G5)
- **Drug dosing** (renal-cleared medications)
- **Transplant evaluation**
- **Progression monitoring**

eGFR equations estimate this from serum biomarkers, avoiding the need for direct measurement.

### Our Approach

This project implements the three most clinically relevant eGFR equations for global rural use:

1. **CKD-EPI 2021** — Current KDIGO-recommended, race-free creatinine equation
2. **MDRD** — Legacy 4-variable equation, still widely used in many health systems
3. **Cockcroft-Gault** — Weight-based CrCl estimation, essential for drug dosing

Plus a hybrid ML model combining all equations with additional biomarkers for improved accuracy.

### Why These Three?

| Equation | Why Rural Clinics Need It |
|----------|--------------------------|
| **CKD-EPI 2021** | Current international standard, race-free, best accuracy for GFR > 60 |
| **MDRD** | Many lab systems still report MDRD; needed for backward compatibility |
| **Cockcroft-Gault** | Required for drug dosing (FDA labels reference CrCl, not eGFR) |

---

## Goals

- Build NHANES data pipeline to acquire kidney function panels (creatinine, cystatin C, demographics, ~15,000+ samples)
- Implement three mechanistic eGFR equations with unit tests (CKD-EPI 2021, MDRD, Cockcroft-Gault)
- Train unified ML model using multi-marker features (hybrid approach)
- Validate against measured GFR from external CKD cohort if available (fallback: cross-equation concordance on NHANES)
- Perform CKD-stage classification analysis (G1–G5 concordance)
- Package as reproducible Python library + Jupyter notebooks
- Target: P30 accuracy ≥ 85% (% within ±30% of measured GFR), mean bias < ±5 mL/min/1.73 m²

---

## User Stories

### Phase 1: Data Sourcing & Harmonization

---

#### US-001: Create project structure and dependencies
**Description:** As a developer, I want a clean Python project structure so that code is organized and reproducible.

**Acceptance Criteria:**
- [x] Create `eGFR/` package directory with `__init__.py`
- [x] Create `eGFR/models.py`, `eGFR/utils.py`, `eGFR/data.py` (empty modules with docstrings)
- [x] Create `eGFR/train.py`, `eGFR/evaluate.py`, `eGFR/predict.py` (empty modules with docstrings)
- [x] Create `tests/` directory with `test_models.py`, `test_utils.py` (empty)
- [x] Create `notebooks/` directory
- [x] Create `requirements.txt` with: numpy, scipy, pandas, scikit-learn, lightgbm, matplotlib, pytest
- [x] Typecheck passes (no syntax errors)

---

#### US-002: Implement unit conversion utilities
**Description:** As a developer, I want reliable unit conversion functions so that data from different lab systems can be harmonized.

**Acceptance Criteria:**
- [x] Add `creatinine_mgdl_to_umoll(cr_mgdl)` (×88.4) in `eGFR/utils.py`
- [x] Add `creatinine_umoll_to_mgdl(cr_umoll)` (÷88.4)
- [x] Add `egfr_to_ckd_stage(egfr)` returning G1–G5 string
- [x] Add `lbs_to_kg(weight_lbs)` and `kg_to_lbs(weight_kg)`
- [x] Add `inches_to_cm(height_in)` and `cm_to_inches(height_cm)`
- [x] Add unit tests in `tests/test_utils.py` verifying all conversions
- [x] Tests pass: `pytest tests/test_utils.py`

---

#### US-003: Create NHANES kidney data download module
**Description:** As a researcher, I want to programmatically download NHANES kidney function data so that I have a reproducible data pipeline.

**Acceptance Criteria:**
- [x] Add `download_nhanes_kidney(output_dir, cycles)` function in `eGFR/data.py`
- [x] Function downloads BIOPRO (biochemistry profile with creatinine), DEMO (demographics) XPT files for cycles 2005-2018
- [x] Downloads SSPRT (cystatin C) XPT files for cycles where available (1999-2002)
- [x] Downloads BMX (body measures — weight, height) for Cockcroft-Gault
- [x] Creates `data/raw/` directory if not exists
- [x] Handles download errors gracefully with informative messages
- [x] Typecheck passes

---

#### US-004: Implement XPT file parser
**Description:** As a developer, I want to parse NHANES XPT files into pandas DataFrames so that data is usable.

**Acceptance Criteria:**
- [x] Add `read_xpt(filepath)` function in `eGFR/data.py`
- [x] Function reads SAS transport format and returns DataFrame
- [x] Handles missing file with informative error
- [x] Add unit test with mock data
- [x] Tests pass

---

#### US-005: Create NHANES kidney data cleaning pipeline
**Description:** As a researcher, I want automated data cleaning so that outliers and missing values are handled consistently.

**Acceptance Criteria:**
- [x] Add `clean_kidney_data(biopro_df, demo_df, bmx_df, cystatin_df=None)` function in `eGFR/data.py`
- [x] Merges datasets on SEQN (sample ID)
- [x] Renames columns to: cr_mgdl (LBXSCR), age_years (RIDAGEYR), sex (RIAGENDR), weight_kg (BMXWT), height_cm (BMXHT), cystatin_c_mgL (SSPRT if available)
- [x] Applies IDMS creatinine standardization correction for pre-2007 NHANES cycles
- [x] Removes physiologic outliers (creatinine < 0.2 or > 15 mg/dL, age < 18)
- [x] Returns cleaned DataFrame with complete cases only
- [x] Typecheck passes

---

#### US-006: Generate data quality report
**Description:** As a researcher, I want a quality report so that I can verify data before analysis.

**Acceptance Criteria:**
- [x] Add `generate_quality_report(df, output_path)` function in `eGFR/data.py`
- [x] Report includes: record count, mean/SD for creatinine/age/weight/height/cystatin C
- [x] Includes CKD stage distribution based on CKD-EPI 2021 eGFR
- [x] Includes sex distribution breakdown
- [x] Saves report to specified path as text file
- [x] Typecheck passes

---

#### US-007: Create data sourcing notebook
**Description:** As a researcher, I want a notebook documenting the data pipeline so that the process is reproducible.

**Acceptance Criteria:**
- [x] Create `notebooks/01_data_sourcing.ipynb`
- [x] Notebook demonstrates: downloading, parsing, cleaning, quality report
- [x] Includes markdown documentation of each step
- [x] Visualizes creatinine distribution and age/sex demographics
- [x] Notebook executes without errors

---

### Phase 2: Mechanistic Estimators

---

#### US-008: Implement CKD-EPI 2021 equation (creatinine-based)
**Description:** As a developer, I want the current KDIGO-recommended eGFR equation as the primary estimator.

**Acceptance Criteria:**
- [x] Add `calc_egfr_ckd_epi_2021(cr_mgdl, age_years, sex)` in `eGFR/models.py`
- [x] Formula: eGFR = 142 × min(SCr/κ, 1)^α × max(SCr/κ, 1)^-1.200 × 0.9938^Age × 1.012 [if female]
- [x] κ = 0.7 (F) / 0.9 (M); α = -0.241 (F) / -0.302 (M)
- [x] sex parameter accepts 'M'/'F' or 1/2 (NHANES coding)
- [x] Raises ValueError for invalid inputs (negative, NaN, age < 18)
- [x] Add docstring with reference to Inker et al. (2021) NEJM
- [x] Typecheck passes

---

#### US-009: Test CKD-EPI 2021 against known values
**Description:** As a developer, I want unit tests validating CKD-EPI 2021 against published reference values.

**Acceptance Criteria:**
- [ ] Add test in `tests/test_models.py` for CKD-EPI 2021
- [ ] Test case: 50-year-old male, SCr=1.0 mg/dL → eGFR ≈ 92 mL/min/1.73m²
- [ ] Test case: 50-year-old female, SCr=0.8 mg/dL → eGFR ≈ 99 mL/min/1.73m²
- [ ] Test case: 70-year-old male, SCr=1.5 mg/dL → eGFR ≈ 45 mL/min/1.73m²
- [ ] Test invalid input handling (negative, NaN, age < 18)
- [ ] Tests pass: `pytest tests/test_models.py`

---

#### US-010: Implement MDRD equation
**Description:** As a developer, I want the MDRD equation for backward compatibility with older lab systems.

**Acceptance Criteria:**
- [ ] Add `calc_egfr_mdrd(cr_mgdl, age_years, sex, is_black=False)` in `eGFR/models.py`
- [ ] Formula: eGFR = 175 × SCr^-1.154 × Age^-0.203 × 0.742 [if female] × 1.212 [if Black]
- [ ] Uses IDMS-traceable creatinine (175 coefficient, not 186)
- [ ] Warns when eGFR > 60 (MDRD less accurate at higher GFR)
- [ ] Add docstring with reference to Levey et al. (2006)
- [ ] Typecheck passes

---

#### US-011: Test MDRD against known values
**Description:** As a developer, I want unit tests validating MDRD against published examples.

**Acceptance Criteria:**
- [ ] Add test in `tests/test_models.py` for MDRD
- [ ] Test known reference values from NKF calculator
- [ ] Test warning is issued when eGFR > 60
- [ ] Test `is_black` race coefficient application
- [ ] Tests pass

---

#### US-012: Implement Cockcroft-Gault equation
**Description:** As a developer, I want CrCl estimation for drug dosing applications.

**Acceptance Criteria:**
- [ ] Add `calc_crcl_cockcroft_gault(cr_mgdl, age_years, weight_kg, sex)` in `eGFR/models.py`
- [ ] Formula: CrCl = [(140 - Age) × Weight / (72 × SCr)] × 0.85 [if female]
- [ ] Returns creatinine clearance in mL/min (NOT mL/min/1.73m²)
- [ ] Add optional BSA-adjusted variant: `calc_crcl_cockcroft_gault_bsa(... , height_cm)`
- [ ] Add docstring with reference to Cockcroft & Gault (1976)
- [ ] Typecheck passes

---

#### US-013: Test Cockcroft-Gault against known values
**Description:** As a developer, I want unit tests validating Cockcroft-Gault.

**Acceptance Criteria:**
- [ ] Add test in `tests/test_models.py` for Cockcroft-Gault
- [ ] Test 70-year-old 70 kg male, SCr=1.0 → CrCl ≈ 68 mL/min
- [ ] Test BSA-adjusted variant
- [ ] Test invalid input handling
- [ ] Tests pass

---

#### US-014: Create estimator comparison notebook
**Description:** As a researcher, I want a notebook comparing eGFR equations visually.

**Acceptance Criteria:**
- [ ] Create `notebooks/02_estimator_comparison.ipynb`
- [ ] Compare CKD-EPI 2021 vs MDRD vs Cockcroft-Gault on NHANES data
- [ ] Generate scatter plots: CKD-EPI vs MDRD, CKD-EPI vs Cockcroft-Gault
- [ ] Show CKD stage reclassification analysis between equations
- [ ] Include markdown interpretation of disagreements
- [ ] Notebook executes without errors

---

### Phase 3: ML Model Development

---

#### US-015: Create feature engineering module
**Description:** As a developer, I want feature engineering functions for ML training.

**Acceptance Criteria:**
- [ ] Add `create_features(df)` function in `eGFR/train.py`
- [ ] Creates features: cr_mgdl, age_years, sex_numeric, weight_kg, height_cm, bmi
- [ ] Adds cystatin C features if available: cystatin_c_mgL, cr_cys_ratio
- [ ] Adds all mechanistic estimator predictions as features (CKD-EPI, MDRD, Cockcroft-Gault)
- [ ] Adds derived features: 1/creatinine, log(creatinine), age×creatinine interaction
- [ ] Returns feature matrix X and column names
- [ ] Typecheck passes

---

#### US-016: Implement train/test split with stratification
**Description:** As a developer, I want stratified splitting so that CKD stages are balanced.

**Acceptance Criteria:**
- [ ] Add `stratified_split(df, test_size=0.3, random_state=42)` in `eGFR/train.py`
- [ ] Stratifies by eGFR ranges (>90, 60-89, 45-59, 30-44, 15-29, <15 mL/min/1.73m²)
- [ ] Returns X_train, X_test, y_train, y_test
- [ ] Typecheck passes

---

#### US-017: Train Ridge regression baseline
**Description:** As a researcher, I want a Ridge regression model as simple ML baseline.

**Acceptance Criteria:**
- [ ] Add `train_ridge(X_train, y_train, alpha=1.0)` in `eGFR/train.py`
- [ ] Returns fitted Ridge model
- [ ] Add function `save_model(model, filepath)` using joblib
- [ ] Typecheck passes

---

#### US-018: Train Random Forest model
**Description:** As a researcher, I want a Random Forest model for nonlinear patterns.

**Acceptance Criteria:**
- [ ] Add `train_random_forest(X_train, y_train, n_estimators=200)` in `eGFR/train.py`
- [ ] Returns fitted RandomForestRegressor
- [ ] Typecheck passes

---

#### US-019: Train LightGBM model
**Description:** As a researcher, I want a LightGBM model for best performance.

**Acceptance Criteria:**
- [ ] Add `train_lightgbm(X_train, y_train, X_val, y_val)` in `eGFR/train.py`
- [ ] Uses early stopping with 20 rounds
- [ ] Returns fitted LGBMRegressor
- [ ] Typecheck passes

---

#### US-020: Implement cross-validation wrapper
**Description:** As a developer, I want 10-fold CV to evaluate models consistently.

**Acceptance Criteria:**
- [ ] Add `cross_validate_model(model, X, y, n_splits=10)` in `eGFR/train.py`
- [ ] Returns dict with RMSE_mean, RMSE_std, MAE_mean, MAE_std
- [ ] Typecheck passes

---

#### US-021: Create model training notebook
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

#### US-022: Implement Bland-Altman analysis
**Description:** As a researcher, I want Bland-Altman statistics for agreement analysis.

**Acceptance Criteria:**
- [ ] Add `bland_altman_stats(y_true, y_pred)` in `eGFR/evaluate.py`
- [ ] Returns dict: mean_bias, std_diff, loa_lower, loa_upper
- [ ] Add unit test with known values
- [ ] Tests pass

---

#### US-023: Implement P30 accuracy metric
**Description:** As a researcher, I want P30 (% within ±30% of measured GFR) as the standard eGFR accuracy metric.

**Acceptance Criteria:**
- [ ] Add `p30_accuracy(y_true, y_pred)` in `eGFR/evaluate.py`
- [ ] Returns percentage of predictions within ±30% of reference
- [ ] Add `p10_accuracy(y_true, y_pred)` for stricter ±10% threshold
- [ ] Add unit tests verifying both metrics
- [ ] Tests pass

---

#### US-024: Create comprehensive evaluation function
**Description:** As a researcher, I want a single function returning all eGFR-relevant metrics.

**Acceptance Criteria:**
- [ ] Add `evaluate_model(y_true, y_pred, model_name)` in `eGFR/evaluate.py`
- [ ] Returns dict: rmse, mae, bias, r_pearson, p30, p10, ba_stats, ckd_stage_agreement
- [ ] ckd_stage_agreement = percentage of concordant CKD stage classifications
- [ ] Typecheck passes

---

#### US-025: Implement CKD-stage stratified evaluation
**Description:** As a researcher, I want to evaluate models by CKD stage.

**Acceptance Criteria:**
- [ ] Add `evaluate_by_ckd_stage(y_true, y_pred, egfr_values)` in `eGFR/evaluate.py`
- [ ] Stratifies by: G1 (≥90), G2 (60-89), G3a (45-59), G3b (30-44), G4 (15-29), G5 (<15)
- [ ] Returns metrics dict for each stage
- [ ] Typecheck passes

---

#### US-026: Implement bootstrap confidence intervals
**Description:** As a researcher, I want bootstrap CIs for reporting uncertainty.

**Acceptance Criteria:**
- [ ] Add `bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=2000)` in `eGFR/evaluate.py`
- [ ] Returns (lower, upper, mean) tuple
- [ ] Typecheck passes

---

#### US-027: Search for external validation datasets with measured GFR
**Description:** As a researcher, I want to identify external CKD cohort datasets with measured GFR.

**Acceptance Criteria:**
- [ ] Research and document potential external datasets in `notebooks/external_data_sources.md`
- [ ] Consider: CKD-EPI development cohort, CRIC Study, MDRD Study data, AASK, iGFR datasets on Physionet/Figshare
- [ ] Document access requirements, measurement method (iothalamate vs iohexol), and sample sizes
- [ ] If no freely accessible dataset, document NHANES cross-equation concordance validation plan
- [ ] Typecheck passes (no code changes, documentation only)

---

#### US-028: External validation with measured GFR dataset
**Description:** As a researcher, I want to validate the models against an independent dataset with measured GFR so that I can demonstrate clinical accuracy.

**Acceptance Criteria:**
- [ ] Identify and download a suitable open-access CKD dataset with measured GFR
- [ ] Write a loader/cleaning function in `eGFR/data.py` for the chosen dataset
- [ ] Run all mechanistic equations and best ML model on external dataset
- [ ] Report P30, P10, RMSE, MAE, bias, and CKD stage concordance
- [ ] Create `notebooks/05_external_validation.ipynb` documenting results
- [ ] Document any dataset limitations
- [ ] All existing tests still pass

---

#### US-029: Create evaluation notebook
**Description:** As a researcher, I want a notebook with full model evaluation and plots.

**Acceptance Criteria:**
- [ ] Create `notebooks/04_evaluation.ipynb`
- [ ] Evaluates all models and estimators on internal test set
- [ ] Generates Bland-Altman plots
- [ ] Shows CKD-stage reclassification analysis
- [ ] Compares hybrid ML vs individual equations
- [ ] Reports P30 accuracy
- [ ] Notebook executes without errors

---

### Phase 5: Package & Documentation

---

#### US-030: Create prediction API
**Description:** As a developer, I want a simple API for making eGFR predictions.

**Acceptance Criteria:**
- [ ] Add `predict_egfr(cr_mgdl, age, sex, weight_kg=None, height_cm=None, cystatin_c=None, method='ckd_epi_2021')` in `eGFR/predict.py`
- [ ] method='ckd_epi_2021', 'mdrd', 'cockcroft_gault' uses specific equation
- [ ] method='hybrid' uses best ML model (accepts partial inputs gracefully)
- [ ] Returns dict: egfr_pred, ckd_stage, method, warning (if inputs incomplete or CrCl vs eGFR mismatch)
- [ ] Typecheck passes

---

#### US-031: Create package setup.py
**Description:** As a developer, I want the package to be pip-installable.

**Acceptance Criteria:**
- [ ] Create `setup.py` with name='eGFR', version='0.1.0'
- [ ] Includes dependencies from requirements.txt
- [ ] Package installs with `pip install -e .`
- [ ] Import works: `from eGFR import models`

---

#### US-032: Write README with usage examples
**Description:** As a user, I want clear documentation to use the package.

**Acceptance Criteria:**
- [ ] Update README.md with: Installation, Quick Start, API Reference
- [ ] Include code examples for each equation
- [ ] Document clinical limitations and when measured GFR is required
- [ ] Explain drug dosing considerations (CrCl vs eGFR)
- [ ] README renders correctly on GitHub

---

#### US-033: Run full test suite
**Description:** As a developer, I want all tests passing before release.

**Acceptance Criteria:**
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No test failures or errors
- [ ] Coverage > 70% on core modules

---

## Non-Goals

- **Not building a web application** — Python package and notebooks only
- **Not pursuing regulatory approval** — Research/educational use only
- **Not replacing measured GFR** — Estimation for screening and monitoring only
- **Not implementing pediatric equations** — Adult patients (≥18) only
- **Not implementing cystatin C-only equations** — Creatinine-based primary, cystatin C as optional ML feature
- **Not building stacked ensemble initially** — Single best ML model for MVP
- **Not estimating proteinuria/UACR** — Only GFR estimation

## Technical Considerations

- **Data access:** NHANES biochemistry profiles with creatinine are public (2005-2018); cystatin C limited to older cycles (1999-2002)
- **Creatinine standardization:** NHANES creatinine was re-calibrated to IDMS in 2007; pre-2007 data needs correction factor
- **Validation challenge:** NHANES does not include measured GFR — external CKD cohort dataset required for true validation
- **Clinical thresholds:** eGFR 60 (CKD stage G3+), eGFR 15 (kidney failure) — errors near thresholds have high clinical impact
- **CrCl vs eGFR:** Cockcroft-Gault returns CrCl (mL/min), not eGFR (mL/min/1.73m²) — API must clearly distinguish
- **Drug dosing:** Many FDA drug labels reference CrCl, not eGFR; Cockcroft-Gault is critical for pharmacy use cases
- **Race variable:** CKD-EPI 2021 is race-free; MDRD retains race coefficient with deprecation warning

## Verification Plan

### Automated Tests
- Run `pytest tests/ -v` after each phase
- All equations verified against published reference values (NKF calculator, MDCalc)
- ML models verified via cross-validation metrics
- Target: P30 ≥ 85%, RMSE < 15 mL/min/1.73m²

### Manual Verification
- Notebooks should execute end-to-end without errors
- Compare equation outputs to NKF online GFR calculator
- Final package should be pip-installable: `pip install -e .`
- Cross-reference CKD-EPI 2021 outputs with MDCalc GFR calculator
