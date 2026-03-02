# eGFR — Estimated Glomerular Filtration Rate Calculator

A clinically-validated Python library for estimating kidney function from routine blood markers. Implements the three most widely used eGFR equations — **CKD-EPI 2021**, **MDRD**, and **Cockcroft-Gault** — plus a hybrid ML model trained on NHANES data.

Designed for rural clinics, low-resource hospitals, and CKD screening programs that lack access to nuclear medicine GFR testing.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/outlabs/egfr-calculator.git
cd egfr-calculator

# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"
```

### Requirements

- Python ≥ 3.9
- NumPy, SciPy, pandas, scikit-learn, LightGBM, matplotlib

All dependencies are listed in `requirements.txt` and installed automatically via `pip install -e .`.

---

## Quick Start

```python
from eGFR.predict import predict_egfr

# CKD-EPI 2021 (recommended, race-free)
result = predict_egfr(cr_mgdl=1.2, age=55, sex="M")
print(f"eGFR: {result['egfr_pred']:.1f} mL/min/1.73m²")
print(f"CKD Stage: {result['ckd_stage']}")
# eGFR: 72.9 mL/min/1.73m²
# CKD Stage: G2

# Cockcroft-Gault for drug dosing
result = predict_egfr(cr_mgdl=1.0, age=70, sex="M",
                      weight_kg=70, method="cockcroft_gault")
print(f"CrCl: {result['egfr_pred']:.1f} mL/min")
print(f"Warning: {result['warning']}")
```

---

## API Reference

### Unified Prediction API

The simplest way to use the package:

```python
from eGFR.predict import predict_egfr

result = predict_egfr(
    cr_mgdl=1.0,          # Serum creatinine (mg/dL) — required
    age=50,               # Age in years (≥ 18) — required
    sex="M",              # "M"/"F" or 1/2 (NHANES coding) — required
    weight_kg=75.0,       # Body weight (kg) — required for cockcroft_gault
    height_cm=175.0,      # Height (cm) — optional, used by hybrid
    cystatin_c=0.9,       # Cystatin C (mg/L) — optional, used by hybrid
    method="ckd_epi_2021" # "ckd_epi_2021" | "mdrd" | "cockcroft_gault" | "hybrid"
)

# Returns:
# {
#     "egfr_pred": 92.1,        # Estimated value
#     "ckd_stage": "G1",        # CKD stage (G1–G5)
#     "method": "ckd_epi_2021", # Method used
#     "warning": None           # Clinical caveats (if any)
# }
```

### Individual Equations

For direct access to each equation:

#### CKD-EPI 2021 (Current Standard)

```python
from eGFR.models import calc_egfr_ckd_epi_2021

# 50-year-old male, SCr = 1.0 mg/dL
egfr = calc_egfr_ckd_epi_2021(cr_mgdl=1.0, age_years=50, sex="M")
# Returns: ~92 mL/min/1.73m²

# NHANES numeric sex coding also accepted (1=male, 2=female)
egfr = calc_egfr_ckd_epi_2021(cr_mgdl=0.8, age_years=50, sex=2)
# Returns: ~90 mL/min/1.73m²
```

Race-free equation recommended by KDIGO. Reference: Inker et al. (2021) *N Engl J Med*. 385(19):1737-1749.

#### MDRD (Legacy)

```python
from eGFR.models import calc_egfr_mdrd

# 60-year-old female, SCr = 1.2 mg/dL
egfr = calc_egfr_mdrd(cr_mgdl=1.2, age_years=60, sex="F")
# Issues UserWarning when eGFR > 60 (MDRD less accurate above this)

# Race coefficient (deprecated, retained for backward compatibility)
egfr = calc_egfr_mdrd(cr_mgdl=1.2, age_years=60, sex="F", is_black=True)
```

Uses IDMS-traceable creatinine (175 coefficient). Reference: Levey et al. (2006) *Ann Intern Med*. 145(4):247-254.

#### Cockcroft-Gault (Drug Dosing)

```python
from eGFR.models import calc_crcl_cockcroft_gault, calc_crcl_cockcroft_gault_bsa

# 70-year-old male, 70 kg, SCr = 1.0 mg/dL
crcl = calc_crcl_cockcroft_gault(
    cr_mgdl=1.0, age_years=70, weight_kg=70, sex="M"
)
# Returns: ~68 mL/min (NOT mL/min/1.73m²)

# BSA-adjusted variant (normalised to 1.73 m²)
crcl_bsa = calc_crcl_cockcroft_gault_bsa(
    cr_mgdl=1.0, age_years=70, weight_kg=70, sex="M", height_cm=175
)
```

Reference: Cockcroft & Gault (1976) *Nephron*. 16(1):31-41.

### Utility Functions

```python
from eGFR.utils import (
    creatinine_mgdl_to_umoll,   # mg/dL → µmol/L (×88.4)
    creatinine_umoll_to_mgdl,   # µmol/L → mg/dL (÷88.4)
    egfr_to_ckd_stage,          # eGFR → "G1"–"G5"
    lbs_to_kg, kg_to_lbs,       # Weight conversion
    inches_to_cm, cm_to_inches, # Height conversion
)

# CKD staging
stage = egfr_to_ckd_stage(45.0)  # Returns "G3a"
```

### Evaluation Metrics

```python
from eGFR.evaluate import (
    evaluate_model,       # Comprehensive metrics dict
    bland_altman_stats,   # Mean bias, LOA
    p30_accuracy,         # % within ±30% of reference
    p10_accuracy,         # % within ±10% of reference
    bootstrap_ci,         # Bootstrap confidence intervals
    evaluate_by_ckd_stage # Stratified evaluation by CKD stage
)

import numpy as np

y_true = np.array([90, 60, 30, 15])
y_pred = np.array([88, 65, 28, 14])

metrics = evaluate_model(y_true, y_pred, model_name="CKD-EPI 2021")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"P30:  {metrics['p30']:.1f}%")
print(f"Bias: {metrics['bias']:.2f}")
```

---

## CKD Stage Classification

The library classifies eGFR values into CKD stages per KDIGO 2012:

| Stage | eGFR (mL/min/1.73 m²) | Description |
|-------|------------------------|-------------|
| G1 | ≥ 90 | Normal or high |
| G2 | 60–89 | Mildly decreased |
| G3a | 45–59 | Mildly to moderately decreased |
| G3b | 30–44 | Moderately to severely decreased |
| G4 | 15–29 | Severely decreased |
| G5 | < 15 | Kidney failure |

---

## Drug Dosing: CrCl vs eGFR

> **⚠️ Important for pharmacy and prescribing use cases**

Many FDA drug labels specify dose adjustments based on **creatinine clearance (CrCl)**, not eGFR. These are **not interchangeable**:

| Metric | Equation | Units | Use Case |
|--------|----------|-------|----------|
| **eGFR** | CKD-EPI 2021, MDRD | mL/min/1.73 m² | CKD staging, monitoring |
| **CrCl** | Cockcroft-Gault | mL/min | Drug dosing (FDA labels) |

**Key differences:**
- CrCl is weight-dependent and **not** normalised to body surface area
- eGFR is normalised to 1.73 m² BSA — appropriate for population-level staging
- For drug dosing, always use `method="cockcroft_gault"` or `calc_crcl_cockcroft_gault()`
- The `predict_egfr()` API automatically warns when using Cockcroft-Gault

---

## Clinical Limitations

> **This library is for research and educational use only. It is not a substitute for clinical judgement.**

- **eGFR is an estimate** — direct GFR measurement (iothalamate, iohexol, or inulin clearance) remains the gold standard
- **Not validated for clinical deployment** — no regulatory approval has been sought
- **Adult patients only** — all equations require age ≥ 18; pediatric equations (Schwartz, Bedside CKD-EPI) are not included
- **Creatinine limitations** — eGFR accuracy decreases in patients with:
  - Extremes of muscle mass (bodybuilders, amputees, sarcopenia)
  - Rapidly changing kidney function (AKI)
  - High protein diet or creatine supplementation
  - Certain medications that inhibit tubular creatinine secretion (cimetidine, trimethoprim)
- **Cystatin C** — not used by any mechanistic equation in this library; only available as an ML model feature
- **MDRD race coefficient** — retained for backward compatibility but **deprecated** per KDIGO 2021 guidance; use CKD-EPI 2021 as the primary equation
- **Measured GFR validation** — the hybrid ML model was trained on NHANES data (no measured GFR); external validation used synthetic data. For publication-grade results, validate against a CKD cohort with measured GFR (e.g., CRIC Study via NIDDK-CR)

---

## Notebooks

The `notebooks/` directory contains reproducible Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `01_data_sourcing.ipynb` | NHANES download, parsing, cleaning, quality report |
| `02_estimator_comparison.ipynb` | Visual comparison of all 3 equations |
| `03_model_training.ipynb` | Ridge, Random Forest, LightGBM training + CV |
| `04_evaluation.ipynb` | Full evaluation: Bland-Altman plots, P30 CIs, CKD reclassification |
| `05_external_validation.ipynb` | Validation against measured GFR (synthetic dataset) |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_models.py -v
pytest tests/test_utils.py -v
pytest tests/test_evaluate.py -v
pytest tests/test_train.py -v
pytest tests/test_predict.py -v
```

---

## Project Structure

```
egfr-calculator/
├── eGFR/                  # Main package
│   ├── __init__.py
│   ├── models.py          # CKD-EPI 2021, MDRD, Cockcroft-Gault equations
│   ├── predict.py         # Unified prediction API
│   ├── evaluate.py        # Evaluation metrics (Bland-Altman, P30, bootstrap CI)
│   ├── train.py           # Feature engineering, model training, cross-validation
│   ├── data.py            # NHANES data pipeline (download, parse, clean)
│   └── utils.py           # Unit conversions, CKD staging
├── tests/                 # pytest test suite
├── notebooks/             # Jupyter notebooks (5 notebooks)
├── models/                # Saved ML models (generated by training notebook)
├── setup.py               # Package configuration
└── requirements.txt       # Dependencies
```

---

## License

MIT

---

## References

1. Inker LA, et al. New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race. *N Engl J Med*. 2021;385(19):1737-1749.
2. Levey AS, et al. Using Standardized Serum Creatinine Values in the Modification of Diet in Renal Disease Study Equation for Estimating Glomerular Filtration Rate. *Ann Intern Med*. 2006;145(4):247-254.
3. Cockcroft DW, Gault MH. Prediction of Creatinine Clearance from Serum Creatinine. *Nephron*. 1976;16(1):31-41.
4. KDIGO 2012 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease.
