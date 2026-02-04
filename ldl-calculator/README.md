# ldlC: LDL Cholesterol Estimation Package

A clinically-validated LDL cholesterol (LDL-C) estimation library implementing multiple mechanistic equations enhanced by a unified machine learning model. Improves upon the traditional Friedewald equation, particularly for patients with elevated triglyceride levels.

## Features

- **Four mechanistic equations**: Friedewald, Martin-Hopkins, Extended Martin-Hopkins, Sampson
- **Hybrid ML model**: Combines all equations for optimal accuracy across TG ranges
- **Clinical validation**: Validated against beta-quantification reference standard
- **NHANES data pipeline**: Tools for downloading and processing NHANES lipid data
- **Comprehensive evaluation**: Bland-Altman analysis, Lin's CCC, bootstrap confidence intervals

## Installation

```bash
# Clone the repository
git clone https://github.com/outlabs/ldl-calculator.git
cd ldl-calculator

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Quick Start

```python
from ldlC import models
from ldlC.predict import predict_ldl

# Example patient: TC=200, HDL=50, TG=150 mg/dL

# Using individual equations
ldl_friedewald = models.calc_ldl_friedewald(200, 50, 150)
print(f"Friedewald: {ldl_friedewald:.1f} mg/dL")  # 120.0 mg/dL

# Using the prediction API (recommended)
result = predict_ldl(200, 50, 150, method='sampson')
print(f"Sampson: {result['ldl_pred']} mg/dL")
print(f"95% CI: [{result['ci_lower']}, {result['ci_upper']}] mg/dL")
```

## API Reference

### Prediction API

The `predict_ldl()` function is the recommended interface for making LDL-C predictions:

```python
from ldlC.predict import predict_ldl

result = predict_ldl(
    tc=200,       # Total cholesterol (mg/dL)
    hdl=50,       # HDL cholesterol (mg/dL)
    tg=150,       # Triglycerides (mg/dL)
    method='hybrid'  # Prediction method (default: 'hybrid')
)

# Returns dict with:
# - ldl_pred: Predicted LDL-C (mg/dL)
# - ci_lower: Lower 95% confidence bound
# - ci_upper: Upper 95% confidence bound
# - method: Method used
# - warning: Warning message (if any)
```

**Available methods:**
- `'friedewald'` - Traditional Friedewald equation (TG < 400)
- `'martin_hopkins'` - Martin-Hopkins with adjustable TG:VLDL factor
- `'martin_hopkins_extended'` - Extended M-H for high TG (400-800)
- `'sampson'` - Sampson/NIH Equation 2
- `'hybrid'` - ML model combining all equations (default)

### Individual Equations

#### Friedewald Equation (1972)

```python
from ldlC.models import calc_ldl_friedewald

# Formula: LDL-C = TC - HDL - (TG / 5)
ldl = calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=150)
# Returns: 120.0 mg/dL

# Note: Returns NaN with warning for TG > 400 mg/dL
```

#### Martin-Hopkins Equation

Uses a 180-cell lookup table for an adjustable TG:VLDL factor:

```python
from ldlC.models import calc_ldl_martin_hopkins

# Formula: LDL-C = TC - HDL - (TG / adjustable_factor)
ldl = calc_ldl_martin_hopkins(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=300)
# Works for TG up to 800 mg/dL
```

#### Extended Martin-Hopkins

For patients with TG 400-800 mg/dL, uses finer granularity in the high-TG range:

```python
from ldlC.models import calc_ldl_martin_hopkins_extended

ldl = calc_ldl_martin_hopkins_extended(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=500)
```

#### Sampson Equation (NIH Equation 2)

Best accuracy for high-TG patients, includes quadratic TG term:

```python
from ldlC.models import calc_ldl_sampson

# Formula: LDL = TC/0.948 - HDL/0.971 - (TG/8.56 + TG*nonHDL/2140 - TG²/16100) - 9.44
ldl = calc_ldl_sampson(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=500)
# Works for TG up to 800 mg/dL
```

### Utility Functions

```python
from ldlC.utils import mg_dl_to_mmol_l, mmol_l_to_mg_dl

# Unit conversions
tc_mmol = mg_dl_to_mmol_l(200, molecule='cholesterol')  # 5.17 mmol/L
tg_mmol = mg_dl_to_mmol_l(150, molecule='triglycerides')  # 1.69 mmol/L

# Convert back
tc_mgdl = mmol_l_to_mg_dl(5.17, molecule='cholesterol')  # ~200 mg/dL
```

## Triglyceride Threshold Considerations

| TG Level (mg/dL) | Recommended Method | Notes |
|-----------------|-------------------|-------|
| < 150 | Any equation | All methods accurate |
| 150-400 | Martin-Hopkins, Sampson, or Hybrid | Friedewald less reliable |
| 400-800 | Extended M-H, Sampson, or Hybrid | Friedewald returns NaN |
| > 800 | Direct LDL measurement | Calculation methods not recommended |

### When to Use Each Method

- **Friedewald**: Traditional baseline, good for TG < 150 mg/dL
- **Martin-Hopkins**: Better for low LDL-C or moderately elevated TG
- **Extended M-H**: Specifically calibrated for TG 400-800 mg/dL
- **Sampson**: Best single equation across all TG ranges
- **Hybrid ML** (default): Combines strengths of all equations; best overall accuracy

## Evaluation Metrics

```python
from ldlC.evaluate import bland_altman_stats, lins_ccc, evaluate_model

# Bland-Altman analysis
ba = bland_altman_stats(y_true, y_pred)
# Returns: mean_bias, std_diff, loa_lower, loa_upper

# Lin's Concordance Correlation Coefficient
ccc = lins_ccc(y_true, y_pred)  # Range: -1 to 1, 1 = perfect agreement

# Comprehensive evaluation
metrics = evaluate_model(y_true, y_pred, model_name="Sampson")
# Returns: rmse, mae, bias, r_pearson, lin_ccc, ba_stats
```

## Data Pipeline

Download and process NHANES lipid panel data:

```python
from ldlC.data import download_nhanes_lipids, read_xpt, clean_lipid_data

# Download NHANES data
download_nhanes_lipids(output_dir='data/raw', cycles=['2015-2016', '2017-2018'])

# Read XPT files
tc_df = read_xpt('data/raw/TCHOL_I.XPT')

# Clean and merge datasets
cleaned_df = clean_lipid_data(tc_df, hdl_df, tg_df, ldl_direct_df)
```

## Project Structure

```
ldl-calculator/
├── ldlC/                    # Main package
│   ├── models.py            # Mechanistic equations
│   ├── predict.py           # Prediction API
│   ├── train.py             # ML training functions
│   ├── evaluate.py          # Evaluation metrics
│   ├── data.py              # NHANES data pipeline
│   └── utils.py             # Unit conversions
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_sourcing.ipynb
│   ├── 02_equation_comparison.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── tests/                   # Unit tests
├── models/                  # Trained models
└── requirements.txt
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=ldlC
```

## References

1. Friedewald WT, et al. *Estimation of the concentration of low-density lipoprotein cholesterol in plasma, without use of the preparative ultracentrifuge.* Clin Chem. 1972.

2. Martin SS, et al. *Comparison of a novel method vs the Friedewald equation for estimating low-density lipoprotein cholesterol levels from the standard lipid profile.* JAMA. 2013.

3. Sampson M, et al. *A New Equation for Calculation of Low-Density Lipoprotein Cholesterol in Patients With Normolipidemia and/or Hypertriglyceridemia.* JAMA Cardiology. 2020.

## License

MIT License - See LICENSE file for details.

## Authors

OutLabs Research Team
