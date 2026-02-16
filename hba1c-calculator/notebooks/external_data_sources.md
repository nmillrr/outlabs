# External Validation Datasets for HbA1c Estimation

## Overview

This document surveys potential external validation datasets for our HbA1c estimation models. External validation — testing on data not used during development — is critical for demonstrating generalizability. Our models were developed on NHANES 2011–2018 data with HPLC-measured HbA1c as the reference standard.

**Required variables for validation:**
- HbA1c (measured, ideally HPLC) — ground truth
- Fasting plasma glucose (FPG)
- Ideally also: triglycerides, HDL-C, hemoglobin, MCV, age, sex

---

## Tier 1: Large Cohort Studies (Gold Standard, Restricted Access)

### UK Biobank
- **Population:** ~500,000 UK adults aged 40–69 at recruitment (2006–2010)
- **Variables available:** HbA1c, fasting glucose, lipid panel (TG, HDL), CBC, age, sex, ethnicity
- **HbA1c method:** Immunoassay (Bio-Rad Variant II Turbo)
- **Access:** Application via UK Biobank Access Management System. Requires approved health-related research proposal. Open to global academic and commercial researchers.
- **Cost:** Free for academic research; annual maintenance fee may apply
- **Strengths:** Very large sample, multi-ethnic (though predominantly White British), extensive phenotyping
- **Limitations:** Not HPLC-measured HbA1c; UK population may differ from US NHANES demographics
- **Link:** https://www.ukbiobank.ac.uk/

### Atherosclerosis Risk in Communities (ARIC) Study
- **Population:** ~15,792 adults aged 45–64 from 4 US communities (1987–ongoing)
- **Variables available:** HbA1c, fasting glucose, lipid panel, CBC, demographics
- **HbA1c method:** HPLC (Tosoh analyzers at later visits)
- **Access:** Proposal submission via ARIC Collaborative Studies Coordinating Center. Also available through NHLBI BioLINCC and dbGaP.
- **Cost:** Free through BioLINCC/dbGaP; requires IRB approval and data use agreement
- **Strengths:** US population, longitudinal data, HPLC-measured HbA1c, includes African-American participants (~27%)
- **Limitations:** Older cohort; application review timeline 2–6 months
- **Link:** https://sites.cscc.unc.edu/aric/

### Multi-Ethnic Study of Atherosclerosis (MESA)
- **Population:** ~6,814 adults aged 45–84 from 6 US sites (2000–ongoing)
- **Variables available:** Fasting glucose (all visits), HbA1c (visit 2+), lipids, demographics
- **Ethnicity:** White (38%), African-American (28%), Hispanic (22%), Chinese-American (12%)
- **Access:** Application via MESA Coordinating Center. Available through NHLBI BioLINCC.
- **Cost:** Free through BioLINCC; requires IRB approval
- **Strengths:** Truly multi-ethnic US cohort; excellent for subgroup validation by race/ethnicity
- **Limitations:** Smaller sample; HbA1c not measured at all visits
- **Link:** https://www.mesa-nhlbi.org/

### Framingham Heart Study (FHS)
- **Population:** ~15,000 participants across 3 generations, Framingham MA (1948–ongoing)
- **Variables available:** Fasting glucose, HbA1c (later exams), lipids, CBC, demographics
- **Access:** dbGaP (genotype-phenotype studies), BioLINCC (phenotype-only), or direct proposal to FHS
- **Cost:** Free via dbGaP/BioLINCC; service fees may apply for internal repository
- **Strengths:** Longest-running cardiovascular cohort; excellent longitudinal data
- **Limitations:** Predominantly White population; limited ethnic diversity; primarily genotype-phenotype via dbGaP
- **Link:** https://www.framinghamheartstudy.org/

---

## Tier 2: Openly Available Datasets (Immediate Access)

### Figshare: HbA1c vs FPG Validity Data
- **Description:** Dataset specifically designed for HbA1c–FPG validity analysis
- **Variables:** HbA1c, fasting plasma glucose
- **Access:** Open download (Figshare), no application required
- **Strengths:** Directly relevant; immediate access
- **Limitations:** Unknown sample size, population, and HbA1c measurement method; may lack additional biomarkers (TG, HDL, Hgb, MCV)
- **Link:** https://figshare.com (search "hba1c vs fpg validity data")

### Kaggle: Diabetes Prediction Dataset
- **Description:** Dataset with HbA1c level and blood glucose level plus demographics
- **Variables:** HbA1c_level, blood_glucose_level, age, BMI, gender
- **Access:** Open download (Kaggle account required)
- **Strengths:** Easy to access; includes HbA1c and glucose
- **Limitations:** Not a clinical research dataset; may be synthetic or aggregated; glucose may not be fasting; lacks lipid panel, CBC, and hemoglobin data
- **Link:** https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

### Mendeley Data: Iraqi Diabetes Dataset
- **Description:** Medical data with HbA1c and blood sugar levels from Iraqi patients
- **Variables:** HBA1C, Sugar Level Blood, additional clinical variables
- **Access:** Open download (Mendeley Data)
- **Strengths:** Different geographic population; immediate access
- **Limitations:** Iraqi population may have different HbA1c–glucose relationships; unclear measurement methods; potentially different ethnic background from NHANES

---

## Tier 3: Clinical Trial / Hospital Datasets

### UCI / OpenML: Diabetes 130-Hospitals Dataset
- **Description:** 10 years of clinical care data from 130 US hospitals (1999–2008)
- **Variables:** HbA1c test result (categorical: >8, >7, normal, none), medications, demographics
- **Access:** Open download via UCI ML Repository or OpenML
- **Strengths:** Large (100k+ records); US hospital population
- **Limitations:** HbA1c is **categorical** (not continuous) — cannot be used for regression validation; more suited for classification tasks
- **Link:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

---

## Recommended Validation Strategy

### If External Data is Accessible (Preferred)

1. **Primary external validation:** Apply to ARIC or MESA via BioLINCC (both have HPLC-measured HbA1c and multi-ethnic US populations)
2. **Quick feasibility check:** Download Figshare HbA1c–FPG dataset for immediate preliminary validation
3. **Report metrics:** RMSE, MAE, bias, Lin's CCC, % within ±0.5%, Bland-Altman plots on external data

### NHANES-Only Validation Plan (Fallback)

Given that external dataset access requires applications and IRB approval (timeline: 2–6 months), we will proceed with robust NHANES-only validation:

1. **Temporal split:** Train on 2011–2016 cycles, test on 2017–2018 cycle (simulates prospective validation)
2. **Stratified holdout:** 70/30 stratified split with evaluation by:
   - HbA1c clinical strata (normal / prediabetes / diabetes)
   - Subgroups (anemia, age groups, MCV groups)
3. **10-fold cross-validation:** For all models with bootstrap confidence intervals
4. **Clinical concordance:** Report % within ±0.5% HbA1c, Lin's CCC ≥ 0.85 target
5. **Limitation disclosure:** Clearly state that external validation was not performed and recommend it for future work

---

## Summary Table

| Dataset | HbA1c Method | FPG | Lipids | CBC | Access | Timeline |
|---------|-------------|-----|--------|-----|--------|----------|
| UK Biobank | Immunoassay | ✓ | ✓ | ✓ | Application | 2–4 mo |
| ARIC | HPLC | ✓ | ✓ | ✓ | BioLINCC/dbGaP | 2–6 mo |
| MESA | Varies | ✓ | ✓ | ✓ | BioLINCC | 2–6 mo |
| Framingham | Varies | ✓ | ✓ | ✓ | BioLINCC/dbGaP | 2–6 mo |
| Figshare | Unknown | ✓ | ✗ | ✗ | Open | Immediate |
| Kaggle | Unknown | ✓ | ✗ | ✗ | Open | Immediate |
| UCI 130-Hosp | Categorical | ✗ | ✗ | ✗ | Open | Immediate |

**Recommendation:** Proceed with NHANES-only validation (fallback plan) for the current project timeline. Apply to ARIC via BioLINCC for future external validation, as it offers HPLC-measured HbA1c in a diverse US population.
