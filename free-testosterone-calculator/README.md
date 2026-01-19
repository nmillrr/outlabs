# Free Testosterone Model

A Python package for estimating free testosterone (FT) from total testosterone (TT), SHBG, and albumin. Implements clinically-validated mechanistic solvers (Vermeulen, Södergård, Zakharov) with optional hybrid ML enhancement.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Installation

```bash
# Clone the repository
git clone https://github.com/nmillrr/outlabs.git
cd free-testosterone-calculator

# Install the package (editable mode)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Quick Start

```python
from freeT.models import calc_ft_vermeulen

# Calculate free testosterone
# Inputs: TT (nmol/L), SHBG (nmol/L), Albumin (g/L)
ft = calc_ft_vermeulen(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=45.0)
print(f"Free testosterone: {ft:.3f} nmol/L")
# Output: Free testosterone: 0.269 nmol/L
```

### Using the Prediction API

```python
from freeT.predict import predict_ft

# Simple prediction (uses hybrid ML model if available)
result = predict_ft(tt=15.0, shbg=40.0, alb=45.0)
print(f"FT: {result['ft_pred']:.3f} nmol/L (method: {result['method']})")

# Force mechanistic solver only
result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
```

---

## API Reference

### Mechanistic Solvers (`freeT.models`)

#### `calc_ft_vermeulen(tt_nmoll, shbg_nmoll, alb_gl, K_shbg=1e9, K_alb=3.6e4)`

Calculate free testosterone using the Vermeulen (1999) equation.

```python
from freeT.models import calc_ft_vermeulen

# Standard usage
ft = calc_ft_vermeulen(15.0, 40.0, 45.0)

# Custom binding constants
ft = calc_ft_vermeulen(15.0, 40.0, 45.0, K_shbg=1.2e9, K_alb=2.4e4)
```

**Parameters:**
- `tt_nmoll`: Total testosterone (nmol/L)
- `shbg_nmoll`: SHBG (nmol/L)
- `alb_gl`: Albumin (g/L)
- `K_shbg`: SHBG binding constant (default: 1e9 L/mol)
- `K_alb`: Albumin binding constant (default: 3.6e4 L/mol)

**Returns:** Free testosterone in nmol/L

---

#### `calc_ft_sodergard(tt_nmoll, shbg_nmoll, alb_gl)`

Södergård variant with different binding constants (K_shbg=1.2e9, K_alb=2.4e4).

```python
from freeT.models import calc_ft_sodergard

ft = calc_ft_sodergard(15.0, 40.0, 45.0)
```

---

#### `calc_ft_zakharov(tt_nmoll, shbg_nmoll, alb_gl, cooperativity=0.5)`

Zakharov allosteric model accounting for cooperative SHBG binding.

```python
from freeT.models import calc_ft_zakharov

# Standard usage
ft = calc_ft_zakharov(15.0, 40.0, 45.0)

# Adjust cooperativity parameter
ft = calc_ft_zakharov(15.0, 40.0, 45.0, cooperativity=0.3)
```

---

#### `calc_bioavailable_t(tt_nmoll, shbg_nmoll, alb_gl)`

Calculate bioavailable testosterone (free + albumin-bound).

```python
from freeT.models import calc_bioavailable_t

bio_t = calc_bioavailable_t(15.0, 40.0, 45.0)
```

---

### Prediction API (`freeT.predict`)

#### `predict_ft(tt, shbg, alb, method='hybrid', model_path=None)`

Unified prediction API supporting mechanistic and hybrid ML methods.

```python
from freeT.predict import predict_ft

# Hybrid method (ML with Vermeulen fallback)
result = predict_ft(15.0, 40.0, 45.0, method='hybrid')

# Returns dict with:
# - ft_pred: Predicted FT (nmol/L)
# - ci_lower: Lower 95% CI (if available)
# - ci_upper: Upper 95% CI (if available)
# - method: Method actually used
```

---

### Data Pipeline (`freeT.data`)

#### Download NHANES Data

```python
from freeT.data import download_nhanes

# Download testosterone, SHBG, and albumin data (2011-2016)
result = download_nhanes(output_dir="data/raw", cycles=["2015-2016"])
print(f"Downloaded {len(result['downloaded'])} files")
```

#### Parse XPT Files

```python
from freeT.data import read_xpt

df = read_xpt("data/raw/2015_2016/TST_I.XPT")
```

#### Clean and Merge Data

```python
from freeT.data import read_xpt, clean_nhanes_data

tst = read_xpt("data/raw/2015_2016/TST_I.XPT")
shbg = read_xpt("data/raw/2015_2016/SHBG_I.XPT")
alb = read_xpt("data/raw/2015_2016/BIOPRO_I.XPT")

# Clean and merge datasets
clean_df = clean_nhanes_data(tst, shbg, alb)
# Returns: seqn, tt_nmoll, shbg_nmoll, alb_gl columns
```

#### Generate Quality Report

```python
from freeT.data import generate_quality_report

report = generate_quality_report(clean_df, "reports/quality.txt")
print(f"Total records: {report['record_count']}")
```

---

### Unit Conversions (`freeT.utils`)

```python
from freeT.utils import ng_dl_to_nmol_l, nmol_l_to_ng_dl
from freeT.utils import mg_dl_to_g_l, g_l_to_mg_dl

# Testosterone: ng/dL ↔ nmol/L
tt_nmol = ng_dl_to_nmol_l(400)  # 400 ng/dL → ~13.9 nmol/L

# Albumin: mg/dL ↔ g/L
alb_gl = mg_dl_to_g_l(4500)  # 4500 mg/dL → 45 g/L
```

---

## Background

### Testosterone in Blood

Testosterone exists in three forms:
- **Free testosterone**: Unbound and biologically active
- **SHBG-bound**: Tightly bound, unavailable to tissues
- **Albumin-bound**: Weakly bound, partially bioavailable

Together these equal **total testosterone**.

### Why Calculate Free Testosterone?

Direct measurement of free testosterone (via equilibrium dialysis or ultrafiltration) is expensive and inaccessible to many clinics. This package provides validated mathematical models to estimate FT from routine lab measurements.

The Vermeulen model offers the [most robust](https://doi.org/10.1210/jc.2017-02360) method with best correlation to measured values.

### Mathematical Model

The mass balance equation:

$$TT = FT + \text{SHBG-bound} + \text{Albumin-bound}$$

With binding equilibria:

$$\text{SHBG-bound}=\frac{[\text{SHBG}] \cdot K_{\text{SHBG}} \cdot FT}{1 + K_{\text{SHBG}} \cdot FT}$$

$$\text{Albumin-bound}=K_{\text{ALB}} \cdot [\text{ALB}] \cdot FT$$

### Binding Constants

From Vermeulen et al. (1999):
- $K_{\text{SHBG}} = 1.0 \times 10^9$ L/mol
- $K_{\text{ALB}} = 3.6 \times 10^4$ L/mol

### Units

| Measurement | Common Units | SI Units (Internal) |
|------------|--------------|---------------------|
| Testosterone | ng/dL | nmol/L (÷ 28.84) |
| SHBG | nmol/L | nmol/L |
| Albumin | g/dL | g/L (× 10) |

---

## Notebooks

Interactive Jupyter notebooks in `notebooks/`:

1. **01_data_sourcing.ipynb** - NHANES data pipeline
2. **02_solver_comparison.ipynb** - Compare Vermeulen/Södergård/Zakharov
3. **03_model_training.ipynb** - ML model training workflow
4. **04_evaluation.ipynb** - Model evaluation and validation

---

## References

- Vermeulen A, et al. (1999). A Critical Evaluation of Simple Methods for the Estimation of Free Testosterone in Serum. *J Clin Endocrinol Metab*. [DOI:10.1210/jcem.84.10.6079](https://doi.org/10.1210/jcem.84.10.6079)
- Södergård R, et al. (1982). Sex hormone-binding globulin. *Ann Clin Res*.
- Zakharov MN, et al. (2015). Allosteric effects on androgen binding. *Mol Cell Endocrinol*. [DOI:10.1016/j.mce.2014.09.001](https://doi.org/10.1016/j.mce.2014.09.001)
- ISSAM Free Testosterone Calculator: https://www.issam.ch/freetesto.htm

---

## License

Research and educational use only.
