<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Active%20Development-orange" alt="Status">
</p>

# Outlabs: Accessible Clinical Biomarker Models

**Bringing publication-grade diagnostic algorithms to clinics that need them most.**

---

## 🎯 Mission

Healthcare shouldn't depend on expensive lab equipment. **Outlabs** develops open-source, clinically-validated models that estimate biomarkers from routine blood tests—making advanced diagnostics accessible to rural clinics, low-resource hospitals, and global health organizations worldwide.

Outlabs combines:
- **Mechanistic solvers** derived from peer-reviewed biochemistry literature
- **Machine learning models** trained on large population datasets (NHANES, UK Biobank)
- **Rigorous validation** against gold-standard measurements (equilibrium dialysis, LC-MS/MS, beta-quantification)

All wrapped in clean Python APIs that any clinician or researcher can use.

---

## 🔬 Current Models

### Free Testosterone Estimation ([free-testosterone-calculator/](free-testosterone-calculator/))

| Method | Description | Status |
|--------|-------------|--------|
| **Vermeulen (1999)** | Cubic mass-action equilibrium solver | ✅ Complete |
| **Södergård (1982)** | Alternative binding constants | ✅ Complete |
| **Zakharov (2015)** | Allosteric model with cooperativity | ✅ Complete |
| **Hybrid ML** | LightGBM trained on NHANES data | 🔄 In Progress |

**Input requirements:** Total testosterone, SHBG, Albumin (all from routine bloodwork)

**Validation target:** Mean bias < ±0.5 nmol/L, Lin's CCC ≥ 0.90 vs. equilibrium dialysis

---

### LDL Cholesterol Estimation ([ldl-calculator/](ldl-calculator/))

| Method | Description | Status |
|--------|-------------|--------|
| **Friedewald (1972)** | Classic TG/5 formula | ✅ Complete |
| **Martin-Hopkins** | Adjustable TG:VLDL factor with 180-cell lookup | ✅ Complete |
| **Extended Martin-Hopkins** | High-TG variant (400-800 mg/dL) | ✅ Complete |
| **Sampson/NIH (2020)** | Quadratic TG correction | ✅ Complete |
| **Hybrid ML** | Ensemble combining all equations | 🔄 In Progress |

**Input requirements:** Total cholesterol, HDL, Triglycerides (standard lipid panel)

**Validation target:** Mean bias < ±5 mg/dL, Lin's CCC ≥ 0.95 vs. beta-quantification

---

## 🚀 Quick Start

### Free Testosterone Calculator

```bash
cd outlabs/free-testosterone-calculator
pip install -e .
```

```python
from freeT.models import calc_ft_vermeulen, calc_ft_sodergard, calc_ft_zakharov
from freeT.predict import predict_ft

# Patient lab values (SI units)
tt = 15.0      # Total testosterone (nmol/L)
shbg = 40.0    # SHBG (nmol/L)
albumin = 45.0 # Albumin (g/L)

# Mechanistic solvers
ft_vermeulen = calc_ft_vermeulen(tt, shbg, albumin)
ft_sodergard = calc_ft_sodergard(tt, shbg, albumin)
ft_zakharov = calc_ft_zakharov(tt, shbg, albumin)

# Or use the unified prediction API
result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
print(f"Free T: {result['ft_pred']:.3f} nmol/L")
```

### LDL Cholesterol Calculator

```bash
cd outlabs/ldl-calculator
pip install -e .
```

```python
from ldlC.models import calc_ldl_friedewald, calc_ldl_sampson
from ldlC.predict import predict_ldl

# Patient lab values (mg/dL)
tc = 200       # Total cholesterol
hdl = 50       # HDL cholesterol
tg = 150       # Triglycerides

# Mechanistic equations
ldl_friedewald = calc_ldl_friedewald(tc, hdl, tg)  # Classic formula
ldl_sampson = calc_ldl_sampson(tc, hdl, tg)        # Better for high TG

# Or use the unified prediction API
result = predict_ldl(tc=200, hdl=50, tg=150, method='sampson')
print(f"LDL-C: {result['ldl_pred']:.1f} mg/dL")
print(f"95% CI: [{result['ci_lower']:.1f}, {result['ci_upper']:.1f}]")
```

---

## 📊 Why This Matters

### The Clinical Problem

Many important biomarkers are **expensive or inaccessible** to measure directly but can be **calculated** from routine lab tests. Outlabs focuses on:

1. **Free Testosterone** — Requires equilibrium dialysis; calculated from TT, SHBG, Albumin
2. **LDL Cholesterol** — Gold standard requires ultracentrifugation; calculated from lipid panel

### The Accuracy Problem

Existing calculators often have **systematic biases** in specific populations:
- Friedewald LDL underestimates in high-TG patients
- Free testosterone calculators show 20-30% bias in PCOS and TRT patients

### Our Solution

Implement **multiple validated equations** with **hybrid ML enhancement**, trained on large diverse datasets (NHANES). Validate against gold-standard measurements with honest performance metrics.

---

## 🏗️ Project Structure

```
outlabs/
├── free-testosterone-calculator/
│   ├── freeT/                    # Python package
│   │   ├── models.py            # Vermeulen, Södergård, Zakharov solvers
│   │   ├── predict.py           # Unified prediction API
│   │   ├── train.py             # ML model training
│   │   ├── evaluate.py          # Validation metrics
│   │   ├── data.py              # NHANES data pipeline
│   │   └── utils.py             # Unit conversions
│   ├── tests/                    # Comprehensive test suite
│   ├── notebooks/                # Reproducible analysis
│   ├── FT_Model_Whitepaper.md   # Technical methodology
│   └── setup.py                  # pip-installable
│
└── ldl-calculator/
    ├── ldlC/                     # Python package
    │   ├── models.py            # Friedewald, Martin-Hopkins, Sampson
    │   ├── predict.py           # Unified prediction API
    │   ├── train.py             # ML model training
    │   ├── evaluate.py          # Bland-Altman, Lin's CCC
    │   ├── data.py              # NHANES lipid data pipeline
    │   └── utils.py             # Unit conversions
    ├── tests/                    # Comprehensive test suite
    ├── notebooks/                # Reproducible analysis
    └── setup.py                  # pip-installable
```

---

## 📈 Roadmap

### Free Testosterone Calculator

| Phase | Description | Status |
|-------|-------------|--------|
| Data Infrastructure | NHANES download, XPT parsing, cleaning | ✅ Complete |
| Mechanistic Solvers | Vermeulen, Södergård, Zakharov | ✅ Complete |
| Prediction API | Unified interface with CI | ✅ Complete |
| ML Models | LightGBM hybrid enhancement | 🔄 In Progress |
| Validation | Bland-Altman, Lin's CCC, EMAS cohort | ⏳ Planned |

### LDL Cholesterol Calculator

| Phase | Description | Status |
|-------|-------------|--------|
| Data Infrastructure | NHANES lipid panel pipeline | ✅ Complete |
| Mechanistic Equations | Friedewald, Martin-Hopkins, Sampson | ✅ Complete |
| Prediction API | Unified interface with CI | ✅ Complete |
| ML Models | Ensemble hybrid model | 🔄 In Progress |
| Validation | Beta-quantification comparison | ⏳ Planned |

---

## 🎯 Future Models

| Biomarker | Status | Clinical Use |
|-----------|--------|--------------|
| Free Testosterone | 🔄 Active | Hypogonadism, PCOS |
| Bioavailable Testosterone | ✅ Complete | Androgen status |
| LDL Cholesterol | 🔄 Active | Cardiovascular risk |
| Free T3/T4 | 📋 Planned | Thyroid function |
| eGFR variants | 📋 Planned | Kidney function |
| Free PSA ratio | 📋 Planned | Prostate screening |

---

## 🏥 Clinical Disclaimer

> **⚠️ Research Use Only**
> 
> These models are intended for academic research and educational purposes. They are **not** FDA-cleared or CE-marked for clinical diagnosis. Always consult qualified healthcare providers for medical decisions.

---

## 📚 References

### Free Testosterone
1. **Vermeulen A, et al.** (1999). A critical evaluation of simple methods for the estimation of free testosterone in serum. *J Clin Endocrinol Metab*. [DOI: 10.1210/jcem.84.10.6079](https://doi.org/10.1210/jcem.84.10.6079)
2. **Zakharov MN, et al.** (2015). Role of the homodimeric and heterodimeric SHBG in testosterone equilibrium dialysis. *Mol Cell Endocrinol*. [DOI: 10.1016/j.mce.2014.09.001](https://doi.org/10.1016/j.mce.2014.09.001)

### LDL Cholesterol
3. **Friedewald WT, et al.** (1972). Estimation of LDL cholesterol without use of the preparative ultracentrifuge. *Clin Chem*.
4. **Martin SS, et al.** (2013). Comparison of a novel method vs the Friedewald equation for estimating LDL-C. *JAMA*. [DOI: 10.1001/jama.2013.280532](https://doi.org/10.1001/jama.2013.280532)
5. **Sampson M, et al.** (2020). A new equation for LDL-C in patients with hypertriglyceridemia. *JAMA Cardiology*. [DOI: 10.1001/jamacardio.2020.0013](https://doi.org/10.1001/jamacardio.2020.0013)

---

## 🤝 Contributing

All contributions welcome! Areas where help is needed:

- **Validation data**: Access to gold-standard measured datasets
- **Clinical feedback**: Real-world usability testing
- **Additional biomarkers**: Extending to new calculators
- **Translations**: Documentation in other languages

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built for the clinics that need it most.</strong><br>
  <em>Outlabs • 2025</em>
</p>
