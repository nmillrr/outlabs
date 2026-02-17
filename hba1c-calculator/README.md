# hba1cE: HbA1c Estimation from Routine Blood Markers

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clinically-validated HbA1c estimation library that predicts glycated hemoglobin from fasting glucose, lipid panels, and demographic factors—reducing dependence on specialized HbA1c assays in resource-limited settings.

## Installation

### From source (recommended)

```bash
git clone https://github.com/outlabs/hba1c-calculator.git
cd hba1c-calculator
pip install -e .
```

### Dependencies

Core dependencies are installed automatically:

- **numpy**, **scipy** — Numerical computation
- **pandas** — Data manipulation
- **scikit-learn** — ML models (Ridge, Random Forest)
- **lightgbm** — Gradient boosting model
- **matplotlib** — Visualization

> **macOS note:** LightGBM requires OpenMP. Install with `brew install libomp` before use.

### Development install

```bash
pip install -e ".[dev]"   # includes pytest
```

---

## Quick Start

### Simple glucose-only estimation (ADAG)

```python
from hba1cE.models import calc_hba1c_adag

# Estimate HbA1c from fasting plasma glucose
hba1c = calc_hba1c_adag(fpg_mgdl=126.0)
print(f"Estimated HbA1c: {hba1c:.1f}%")  # → 6.0%
```

### Kinetic model with hemoglobin adjustment

```python
from hba1cE.models import calc_hba1c_kinetic

# Adjust for anemia (low hemoglobin)
hba1c = calc_hba1c_kinetic(fpg_mgdl=126.0, hgb_gdl=10.5)
print(f"Estimated HbA1c: {hba1c:.2f}%")
```

### Multi-marker regression

```python
from hba1cE.models import calc_hba1c_regression

hba1c = calc_hba1c_regression(
    fpg_mgdl=126.0,
    age_years=55,
    tg_mgdl=150.0,
    hdl_mgdl=45.0,
    hgb_gdl=14.0,
)
print(f"Estimated HbA1c: {hba1c:.2f}%")
```

### Unified prediction API (recommended)

```python
from hba1cE.predict import predict_hba1c

# Full multi-marker hybrid prediction
result = predict_hba1c(
    fpg=126,       # Fasting glucose (mg/dL) — required
    tg=150,        # Triglycerides (mg/dL)
    hdl=45,        # HDL cholesterol (mg/dL)
    age=55,        # Age (years)
    hgb=14.0,      # Hemoglobin (g/dL)
    mcv=90.0,      # Mean corpuscular volume (fL)
    method='hybrid'
)
print(f"Estimated HbA1c: {result['hba1c_pred']:.1f}%")
print(f"95% CI: [{result['ci_lower']:.1f}, {result['ci_upper']:.1f}]%")

# Works with partial inputs — missing values use population medians
result = predict_hba1c(fpg=126, method='hybrid')
if result.get('warning'):
    print(f"Warning: {result['warning']}")
```

---

## API Reference

### Mechanistic Estimators (`hba1cE.models`)

| Function | Description | Required Inputs |
|----------|-------------|-----------------|
| `calc_hba1c_adag(fpg_mgdl)` | ADAG inversion (Nathan et al., 2008) | FPG |
| `calc_hba1c_kinetic(fpg_mgdl, hgb_gdl, rbc_lifespan_days, k)` | First-order glycation kinetics | FPG |
| `calc_hba1c_regression(fpg_mgdl, age_years, tg_mgdl, hdl_mgdl, hgb_gdl)` | Multi-linear regression | FPG, Age, TG, HDL, Hgb |
| `fit_regression_coefficients(df)` | Fit regression coefficients from data | DataFrame |

### Prediction API (`hba1cE.predict`)

```python
predict_hba1c(
    fpg,                    # float — Fasting plasma glucose (mg/dL), required
    tg=None,                # float — Triglycerides (mg/dL)
    hdl=None,               # float — HDL cholesterol (mg/dL)
    age=None,               # float — Age (years)
    hgb=None,               # float — Hemoglobin (g/dL)
    mcv=None,               # float — Mean corpuscular volume (fL)
    method='hybrid',        # str  — 'adag', 'kinetic', 'regression', or 'hybrid'
    model_dir=None,         # str  — Custom path to saved models
) -> dict
```

**Returns** a dict with: `hba1c_pred`, `ci_lower`, `ci_upper`, `method`, `warning`

### Evaluation Metrics (`hba1cE.evaluate`)

| Function | Description |
|----------|-------------|
| `evaluate_model(y_true, y_pred, model_name)` | RMSE, MAE, bias, Pearson r, Lin's CCC, Bland-Altman, % within ±0.5% |
| `bland_altman_stats(y_true, y_pred)` | Mean bias, std of differences, limits of agreement |
| `lins_ccc(y_true, y_pred)` | Lin's Concordance Correlation Coefficient |
| `evaluate_by_hba1c_strata(y_true, y_pred, hba1c_values)` | Metrics stratified by clinical ranges |
| `evaluate_by_subgroup(y_true, y_pred, df, subgroup_col, subgroup_values)` | Metrics by clinical subgroup |
| `bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap)` | Bootstrap confidence intervals |

### Unit Conversions (`hba1cE.utils`)

```python
from hba1cE.utils import mg_dl_to_mmol_l, percent_to_mmol_mol

glucose_mmol = mg_dl_to_mmol_l(126.0)       # → 7.0 mmol/L
hba1c_ifcc = percent_to_mmol_mol(6.5)       # → 48 mmol/mol
```

### Data Pipeline (`hba1cE.data`)

| Function | Description |
|----------|-------------|
| `download_nhanes_glycemic(output_dir, cycles)` | Download NHANES XPT files (2011–2018) |
| `read_xpt(filepath)` | Parse SAS transport files into DataFrames |
| `clean_glycemic_data(ghb_df, glu_df, ...)` | Merge, rename, and clean NHANES data |
| `generate_quality_report(df, output_path)` | Summary statistics and clinical distributions |

---

## Clinical Limitations

> **⚠️ This tool is for research and screening purposes only. It does not replace direct HbA1c measurement for clinical diagnosis.**

### When direct HbA1c measurement is required

- **Diagnosis confirmation** — HbA1c estimates should not be used to confirm a diabetes diagnosis
- **Treatment decisions** — Medication adjustments require laboratory HbA1c
- **Near clinical thresholds** — Small errors at 5.7% (prediabetes) or 6.5% (diabetes) can change classification
- **Post-treatment monitoring** — Tracking response to therapy requires measured values

### Accuracy considerations

- **FPG ≠ average glucose** — Fasting glucose is a single time-point; HbA1c reflects a 3-month average
- **Glucose variability** — Patients with high glycemic variability may have poor FPG→HbA1c correlation
- **Non-glycemic factors** — RBC lifespan, hemoglobin variants, assay interference affect true HbA1c
- **Racial/ethnic differences** — HbA1c may differ by ~0.4% between populations at the same average glucose

### Subgroup considerations

Estimates may be less accurate or require careful interpretation in:

| Subgroup | Impact on HbA1c | Recommendation |
|----------|-----------------|----------------|
| **Anemia** (Hgb < 12/13 g/dL) | Altered RBC turnover affects glycation time | Use kinetic model with hemoglobin adjustment |
| **Hemoglobinopathies** (HbS, HbC, HbE) | Interferes with many HbA1c assays | Do not use estimation; require HPLC |
| **Chronic kidney disease** | Uremia reduces RBC survival | Interpret with caution; may overestimate |
| **Pregnancy** | Physiological hemodilution | Not validated for this population |
| **Age extremes** (< 18 or > 80) | Different glycation kinetics | Use age-adjusted models, interpret cautiously |
| **Recent transfusion** | Dilutes glycated hemoglobin | Do not estimate; wait 2–3 months |

---

## Project Structure

```
hba1c-calculator/
├── hba1cE/                    # Python package
│   ├── __init__.py
│   ├── models.py              # ADAG, kinetic, regression estimators
│   ├── predict.py             # Unified prediction API
│   ├── train.py               # ML model training (Ridge, RF, LightGBM)
│   ├── evaluate.py            # Validation metrics & subgroup analysis
│   ├── data.py                # NHANES data pipeline
│   └── utils.py               # Unit conversions (mg/dL ↔ mmol/L)
├── tests/                     # Unit tests (196+ tests)
├── notebooks/
│   ├── 01_data_sourcing.ipynb
│   ├── 02_estimator_comparison.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_external_validation.ipynb
├── models/                    # Trained model artifacts (.joblib)
├── data/                      # NHANES raw + processed data
├── requirements.txt
└── setup.py
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_sourcing` | NHANES download, parsing, cleaning, quality report |
| `02_estimator_comparison` | ADAG vs Kinetic vs Regression scatter plots and Bland-Altman |
| `03_model_training` | Ridge, RF, LightGBM training with cross-validation |
| `04_evaluation` | Full metrics, subgroup analysis, bootstrap CIs, clinical agreement |
| `05_external_validation` | External dataset validation and limitations |

---

## Validation Summary

### Performance targets

| Metric | Target | Description |
|--------|--------|-------------|
| RMSE | < 0.5% | Average prediction error |
| Mean Bias | < ±0.2% | Systematic over/under-estimation |
| Lin's CCC | ≥ 0.85 | Agreement with HPLC-measured values |
| % within ±0.5% | > 80% | Clinical decision accuracy |

### Data source

- **NHANES 2011–2018** glycemic panels (~10,000+ samples)
- **Gold standard:** HPLC-measured HbA1c (LBXGH), NGSP-certified

---

## References

1. **Nathan DM, et al.** (2008). Translating the A1C Assay Into Estimated Average Glucose Values. *Diabetes Care*. [DOI: 10.2337/dc08-0545](https://doi.org/10.2337/dc08-0545)
2. **Sacks DB, et al.** (2011). Guidelines and Recommendations for Laboratory Analysis in the Diagnosis and Management of Diabetes Mellitus. *Diabetes Care*. [DOI: 10.2337/dc11-9998](https://doi.org/10.2337/dc11-9998)
3. **Bergenstal RM, et al.** (2018). Racial Differences in the Relationship of Glucose Concentrations and Hemoglobin A1c Levels. *Ann Intern Med*. [DOI: 10.7326/M17-2865](https://doi.org/10.7326/M17-2865)
4. **NGSP** (National Glycohemoglobin Standardization Program): http://www.ngsp.org/

---

## License

MIT License — See [LICENSE](LICENSE) for details.
