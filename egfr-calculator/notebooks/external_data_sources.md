# External Validation Datasets with Measured GFR

## Overview

This document surveys potential external datasets containing **measured GFR (mGFR)** for validating our eGFR models against a gold-standard reference. Measured GFR is obtained via exogenous filtration markers (iothalamate, iohexol, inulin, or ⁵¹Cr-EDTA clearance) and provides the ground truth that estimated GFR equations approximate.

**Key requirement:** The dataset must include serum creatinine (and ideally cystatin C) alongside measured GFR to enable direct comparison of our mechanistic equations and hybrid ML model.

---

## 1. NIDDK Central Repository Datasets

### 1a. Chronic Renal Insufficiency Cohort (CRIC) Study

| Attribute | Detail |
|-----------|--------|
| **Source** | NIDDK Central Repository (NIDDK-CR) |
| **URL** | https://repository.niddk.nih.gov/studies/cric/ |
| **Sample Size** | ~3,900 participants at enrollment |
| **Population** | Adults with CKD (eGFR 20–70 mL/min/1.73m²), racially/ethnically diverse |
| **mGFR Method** | ¹²⁵I-iothalamate clearance |
| **Key Variables** | Serum creatinine, cystatin C, age, sex, race, BMI, measured GFR |
| **Access** | Requires data use agreement (DUA) submitted to NIDDK-CR; free for academic researchers |
| **Strengths** | Large sample, iothalamate gold standard, includes cystatin C, longitudinal data |
| **Limitations** | Restricted to CKD population (no healthy controls), DUA process takes 2–4 weeks |

### 1b. African American Study of Kidney Disease (AASK)

| Attribute | Detail |
|-----------|--------|
| **Source** | NIDDK Central Repository (NIDDK-CR) |
| **URL** | https://repository.niddk.nih.gov/studies/aask/ |
| **Sample Size** | ~1,094 (Trial) + ~691 (Cohort) |
| **Population** | African American adults with hypertension-related CKD |
| **mGFR Method** | ¹²⁵I-iothalamate clearance (primary outcome) |
| **Key Variables** | Serum creatinine, age, sex, measured GFR, blood pressure, proteinuria |
| **Access** | DUA via NIDDK-CR |
| **Strengths** | Iothalamate GFR as primary endpoint, well-characterized CKD cohort |
| **Limitations** | African American only (limited generalizability), primarily CKD G3–G4, cystatin C not routinely collected |

### 1c. MDRD Study

| Attribute | Detail |
|-----------|--------|
| **Source** | NIDDK Central Repository |
| **URL** | https://repository.niddk.nih.gov/studies/mdrd/ |
| **Sample Size** | ~1,628 participants |
| **Population** | Adults with CKD (GFR 13–55 mL/min/1.73m²) |
| **mGFR Method** | ¹²⁵I-iothalamate clearance |
| **Key Variables** | Serum creatinine, age, sex, race, measured GFR |
| **Access** | DUA via NIDDK-CR |
| **Strengths** | Original development dataset for MDRD equation, well-validated mGFR |
| **Limitations** | Older dataset (1989–1993), creatinine not IDMS-standardized (requires correction), narrow GFR range |

---

## 2. CKD-EPI Development Cohort

| Attribute | Detail |
|-----------|--------|
| **Source** | Multi-study pooled dataset used to develop CKD-EPI equations |
| **Sample Size** | ~8,254 (development) + ~3,896 (validation) |
| **Population** | Diverse CKD and healthy populations pooled from 10+ studies |
| **mGFR Method** | Various (iothalamate, iohexol, ⁵¹Cr-EDTA, depending on source study) |
| **Key Variables** | Serum creatinine, cystatin C, age, sex, race, measured GFR |
| **Access** | **Not publicly available** — individual-level data held by CKD-EPI consortium; published summary statistics available in Inker et al. (2021) NEJM |
| **Strengths** | Largest mGFR reference dataset, diverse populations, multiple GFR markers |
| **Limitations** | Data not directly accessible; must contact CKD-EPI investigators |

---

## 3. Open-Access / Low-Barrier Datasets

### 3a. PhysioNet / Figshare

**Current status:** No freely downloadable dataset on PhysioNet or Figshare contains measured GFR alongside serum creatinine and demographics in a format suitable for eGFR validation. Some Figshare entries contain GFR equation comparison tables but not raw patient-level data.

### 3b. kidney.epi R Package (Synthetic Data)

| Attribute | Detail |
|-----------|--------|
| **Source** | CRAN `kidney.epi` R package |
| **Sample Size** | Synthetic dataset (~500 samples) |
| **Key Variables** | Serum creatinine, cystatin C, age, sex, race, "measured" GFR (simulated) |
| **Access** | Freely available via `install.packages("kidney.epi")` |
| **Strengths** | Immediate access, structured for eGFR validation, no DUA |
| **Limitations** | **Synthetic data** — not from real patients; insufficient for clinical validation |

---

## 4. NHANES Cross-Equation Concordance (Fallback Plan)

If no freely accessible external dataset with measured GFR can be obtained, we will validate using **cross-equation concordance on NHANES data**:

### Approach

1. **Compute eGFR using all three equations** (CKD-EPI 2021, MDRD, Cockcroft-Gault) on the NHANES cohort
2. **Assess concordance** between equation pairs (CKD-EPI vs MDRD, CKD-EPI vs CG)
3. **CKD stage reclassification analysis** — quantify how often equations disagree on CKD staging
4. **Hybrid ML model vs mechanistic** — compare ML predictions to each equation (treat CKD-EPI 2021 as reference standard since KDIGO-recommended)
5. **Bland-Altman analysis** for inter-equation agreement
6. **Stratified analysis** by age, sex, creatinine range, and CKD stage

### Justification

While cross-equation concordance does not substitute for mGFR validation, it demonstrates:
- Internal consistency of the hybrid ML model
- Agreement with the current clinical standard (CKD-EPI 2021)
- Clinical relevance of reclassification patterns
- That the ML model does not introduce systematic bias vs established equations

### Limitations

- No ground truth (no mGFR) — cannot compute true P30/P10 accuracy
- Circular validation risk if ML model is trained on CKD-EPI outputs
- Cannot assess absolute accuracy, only relative agreement

---

## Recommendation

| Priority | Dataset | Feasibility | Recommendation |
|----------|---------|-------------|----------------|
| **1** | CRIC Study (NIDDK-CR) | High (free, DUA required) | **Best option** — large sample, iothalamate mGFR, cystatin C available |
| **2** | AASK (NIDDK-CR) | High (free, DUA required) | Good supplement, but limited to African American population |
| **3** | MDRD Study (NIDDK-CR) | High (free, DUA required) | Historical value, but older non-IDMS creatinine |
| **4** | CKD-EPI Consortium | Low (requires collaboration) | Ideal but access typically limited to consortium members |
| **5** | NHANES Concordance | Immediate (no mGFR) | **Fallback** — use if no external mGFR dataset obtained within timeline |

### Recommended Path

1. **Submit NIDDK-CR DUA for CRIC Study** — most comprehensive dataset with iothalamate mGFR + creatinine + cystatin C
2. **Proceed with NHANES cross-equation concordance validation** while DUA is processed
3. **If CRIC data obtained**, run full external validation (US-028) with P30, P10, RMSE, bias, CKD concordance
4. **If no external data by deadline**, publish with NHANES concordance analysis + note external validation as future work

---

## References

- Inker LA, et al. New creatinine- and cystatin C-based equations to estimate GFR without race. *N Engl J Med*. 2021;385(19):1737-1749.
- Levey AS, et al. A new equation to estimate glomerular filtration rate. *Ann Intern Med*. 2009;150(9):604-612.
- Lash JP, et al. Chronic Renal Insufficiency Cohort (CRIC) Study. *J Am Soc Nephrol*. 2009;20(suppl):S218-S224.
- Agodoa LY, et al. Effect of ramipril vs amlodipine on renal outcomes in hypertensive nephrosclerosis (AASK). *JAMA*. 2001;285(21):2719-2728.
- Levey AS, et al. A more accurate method to estimate glomerular filtration rate from serum creatinine. *Ann Intern Med*. 1999;130(6):461-470.
