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

We combine:
- **Mechanistic solvers** derived from peer-reviewed biochemistry literature
- **Machine learning models** trained on large population datasets (NHANES, UK Biobank)
- **Rigorous validation** against gold-standard measurements (equilibrium dialysis, LC-MS/MS)

All wrapped in a clean Python API that any clinician or researcher can use.

---

## 🔬 Current Models

### Free Testosterone Estimation

| Method | Description | Status |
|--------|-------------|--------|
| **Vermeulen (1999)** | Cubic mass-action equilibrium solver | ✅ Complete |
| **Södergård (1982)** | Alternative binding constants | ✅ Complete |
| **Zakharov (2015)** | Allosteric model with cooperativity | ✅ Complete |
| **Hybrid ML** | LightGBM trained on NHANES data | 🔄 In Progress |

**Input requirements:** Total testosterone, SHBG, Albumin (all from routine bloodwork)

**Validation target:** Mean bias < ±0.5 nmol/L, Lin's CCC ≥ 0.90 vs. equilibrium dialysis

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/nmillrr/outlabs.git
cd outlabs/free-testosterone-calculator

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from freeT.models import calc_ft_vermeulen, calc_ft_sodergard, calc_ft_zakharov

# Patient lab values (all in SI units)
tt = 15.0      # Total testosterone (nmol/L)
shbg = 40.0    # SHBG (nmol/L)
albumin = 45.0 # Albumin (g/L)

# Calculate free testosterone using different methods
ft_vermeulen = calc_ft_vermeulen(tt, shbg, albumin)
ft_sodergard = calc_ft_sodergard(tt, shbg, albumin)
ft_zakharov = calc_ft_zakharov(tt, shbg, albumin)

print(f"Free T (Vermeulen): {ft_vermeulen:.3f} nmol/L")
print(f"Free T (Södergård): {ft_sodergard:.3f} nmol/L")
print(f"Free T (Zakharov):  {ft_zakharov:.3f} nmol/L")
```

### Unit Conversions

```python
from freeT.utils import ng_dl_to_nmol_l, nmol_l_to_ng_dl

# Convert between common units
tt_ng_dl = 432  # US lab format
tt_nmol = ng_dl_to_nmol_l(tt_ng_dl)  # → ~14.98 nmol/L
```

---

## 📊 Why This Matters

### The Clinical Problem

Free testosterone is the **biologically active** fraction that drives androgen-dependent processes. But measuring it directly requires **equilibrium dialysis**—expensive, slow, and unavailable in most clinical settings.

Instead, labs calculate free testosterone from:
- **Total testosterone** (easily measured via immunoassay or LC-MS/MS)
- **SHBG** (sex hormone-binding globulin)
- **Albumin** (routine chemistry panel)

### The Accuracy Problem

Current calculators show **20-30% systematic bias** in certain populations:
- Women with PCOS
- Men on testosterone replacement
- Patients with altered SHBG (liver disease, thyroid disorders, obesity)

### Our Solution

Train hybrid models on large, diverse datasets while preserving mechanistic interpretability. Validate against **gold-standard equilibrium dialysis measurements** and report honest performance metrics with bootstrap confidence intervals.

---

## 🏗️ Project Structure

```
outlabs/free-testosterone-calculator/
├── freeT/                    # Main Python package
│   ├── models.py            # Mechanistic solvers (Vermeulen, Södergård, Zakharov)
│   ├── data.py              # NHANES download & cleaning pipeline
│   ├── utils.py             # Unit conversions
│   └── train.py             # ML model training
├── tests/                    # Comprehensive test suite
├── notebooks/                # Reproducible analysis
│   ├── 01_data_sourcing.ipynb
│   └── 02_solver_comparison.ipynb
├── FT_Model_Whitepaper.md   # Technical derivations & methodology
├── PRD.md                   # Development roadmap
└── requirements.txt
```

---

## 📈 Roadmap

### ✅ Phase 1: Data Infrastructure (Complete)
- NHANES 2011-2016 download pipeline
- XPT parsing and data cleaning
- Unit conversion utilities

### ✅ Phase 2: Mechanistic Solvers (Complete)
- Vermeulen cubic solver with Brent's method
- Södergård variant
- Zakharov allosteric model
- Bioavailable testosterone calculation

### 🔄 Phase 3: ML Models (In Progress)
- Feature engineering with mechanistic baseline
- Ridge, Random Forest, LightGBM training
- Cross-validation pipeline

### ⏳ Phase 4: Validation
- Bland-Altman analysis
- Lin's CCC metrics
- External validation on EMAS cohort

### ⏳ Phase 5: Package & Publication
- pip-installable package
- Clinical guidance documentation
- Peer-reviewed publication

---

## 🎯 Future Models

Outlabs aims to expand beyond free testosterone to other **inaccessible-but-calculable** biomarkers:

| Biomarker | Status | Clinical Use |
|-----------|--------|--------------|
| Free Testosterone | 🔄 Active | Hypogonadism, PCOS |
| Bioavailable Testosterone | ✅ Done | Androgen status |
| Free T3/T4 | 📋 Planned | Thyroid function |
| Calculated LDL | 📋 Planned | Cardiovascular risk |
| eGFR variants | 📋 Planned | Kidney function |
| Free PSA ratio | 📋 Planned | Prostate screening |

---

## 🏥 Clinical Disclaimer

> **⚠️ Research Use Only**
> 
> These models are intended for academic research and educational purposes. They are **not** FDA-cleared or CE-marked for clinical diagnosis. Always consult qualified healthcare providers for medical decisions.

---

## 📚 References

1. **Vermeulen A, et al.** (1999). A critical evaluation of simple methods for the estimation of free testosterone in serum. *J Clin Endocrinol Metab*. [DOI: 10.1210/jcem.84.10.6079](https://doi.org/10.1210/jcem.84.10.6079)

2. **Fiers T, et al.** (2018). Reassessing Free-Testosterone Calculation by Liquid Chromatography–Tandem Mass Spectrometry Direct Equilibrium Dialysis. *J Clin Endocrinol Metab*. [DOI: 10.1210/jc.2017-02360](https://doi.org/10.1210/jc.2017-02360)

3. **Zakharov MN, et al.** (2015). Role of the homodimeric and heterodimeric human sex hormone-binding globulin in testosterone equilibrium dialysis. *Mol Cell Endocrinol*. [DOI: 10.1016/j.mce.2014.09.001](https://doi.org/10.1016/j.mce.2014.09.001)

---

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **Validation data**: Access to ED-measured datasets
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
