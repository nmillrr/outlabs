# Free Testosterone Estimation Model Development
## A Comprehensive Knowledge Layer for Clinical Diagnostic Implementation

**Prepared for:** Academic collaborators and clinicians in rural healthcare settings  
**Project Scope:** Development, validation, and deployment of a clinically-usable free testosterone (FT) estimation algorithm  
**Timeline:** 6-month publication-grade development cycle  
**Implementation Context:** Non-US, high-validation, academic + nonprofit use  

---

## Executive Summary

This whitepaper outlines a systematic, six-month roadmap to develop and validate a novel free testosterone estimation model from total testosterone (TT, measured by LC–MS/MS), sex hormone-binding globulin (SHBG), and albumin. The goal is to create a clinically robust diagnostic tool suitable for use in rural and resource-constrained settings where direct equilibrium dialysis (ED) measurement is unavailable.

**Core Problem:** Current free testosterone calculation methods (Vermeulen, Södergård, Zakharov) show systematic bias, SHBG-dependent accuracy, and produce discrepant estimates when applied to diverse populations. A data-driven, population-specific model can improve diagnostic accuracy for hypogonadism and androgen excess diagnosis.

**Deliverables:**
- Publication-grade Python package with three mechanistic solvers + empirical regression models
- External validation report with bootstrap confidence intervals and clinical acceptability assessment
- Comprehensive clinical guidance for rural clinicians on appropriate use and limitations
- Open-source code and training materials for implementation in low-resource labs

**Success Metrics:**
- Mean bias < ±0.5 nmol/L (or <±10% relative) across clinical range
- Lin's concordance correlation coefficient ≥ 0.90 on held-out validation set
- Bland–Altman 95% limits of agreement within clinically acceptable span (±2 nmol/L target)
- TRIPOD+AI compliant reporting for publication

---

## Part 1: Technical Background & State of Knowledge

### 1.1 The Free Testosterone Problem

Total testosterone (TT) circulates in three fractions:
- **Free testosterone (FT):** ~2–3% unbound, biologically active
- **SHBG-bound:** ~60–70% bound with high affinity, biologically inactive
- **Albumin-bound:** ~25–35% weakly bound, partially bioavailable

Most clinical labs measure only TT by LC–MS/MS. However, in certain conditions (altered SHBG due to thyroid disease, liver disease, high estradiol), TT alone is misleading. Clinical assessment requires FT, but direct measurement by equilibrium dialysis is expensive, time-consuming, and unavailable in most settings.

### 1.2 Existing Calculation Methods

#### Vermeulen (1999) – Mass-Action Cubic Formula

The canonical approach assumes mass-action equilibrium and derives a cubic equation from binding stoichiometry.[1]

**Formula components:**
- Association constant for SHBG: K_SHBG = 1 × 10^9 L/mol
- Association constant for albumin: K_ALB = 3.6 × 10^4 L/mol

**Implementation in code-like pseudomath:**

```
a = K_ALB + K_SHBG + (K_ALB × K_SHBG × (SHBG + Albumin - TT))
b = 1 + K_SHBG × SHBG + K_ALB × Albumin - (K_ALB + K_SHBG) × TT
FT = (-b + sqrt(b² + 4 × a × TT)) / (2 × a)
```

**Clinical utility:**
- Simple, analytically solvable
- Works well across most TT and SHBG ranges
- Fiers 2018 validation[2]: Systematically overestimates FT by 20–30% compared to direct ED

#### Ly (Empirical Regression) – Population-Specific

Derives regression coefficients from specific cohorts. Two variants:
- High-testosterone equation (men, women with elevated TT)
- Low-testosterone equation (hypogonadal men, obese populations)

**Clinical utility:**
- Better agreement with ED in midrange TT/SHBG
- Accuracy degrades at extremes of SHBG (< 5 nmol/L or > 200 nmol/L)[2]
- Requires population re-calibration; not universally portable

#### Zakharov (2015) – Allosteric Multi-Step Model

Proposes that SHBG-testosterone binding is allosteric, not simple mass-action. Results in non-linear, higher FT% estimates.

**Clinical utility:**
- Theoretically elegant but Fiers 2018 validation found estimates "far off target" vs. ED[2]
- Requires additional parameterization; not yet clinically adopted
- Remains research-grade

### 1.3 Why a New Model?

**Limitations of existing methods:**
1. Population-specific constants are not transferable across cohorts with different ethnic, metabolic, or disease profiles
2. All three reference models show substantial bias (≥20%) in specific subgroups (women with PCOS, men on androgen replacement, elderly with altered SHBG)
3. No single equation performs optimally across the full clinical range of SHBG (5–200 nmol/L) and TT (2–35 nmol/L)
4. Empirical models require large, diverse ED-measured reference datasets—which are rare and often proprietary

**Strategic opportunity:** By combining:
- Mechanistic starting points (Vermeulen, Zakharov)
- Large population-representative datasets (NHANES, UK Biobank, EMAS)
- Modern machine learning (random forest, gradient boosting) for residual patterns
- Rigorous cross-validation and external validation on independent ED-measured cohorts

We can create a model that is both interpretable and more accurate than any single existing approach.

---

## Part 2: Project Roadmap (6-Month Timeline)

### Phase 1: Data Sourcing & Harmonization (Weeks 0–3)

**Objective:** Acquire and prepare datasets for model development.

#### 1a. Primary Data Source: NHANES 2011–2016

**What you'll get:**
- N ≈ 5,000 males with complete TT, SHBG, albumin measurements
- TT measured by isotope-dilution LC–MS/MS (CDC-standardized, traceable)
- SHBG measured by immunoassay
- Albumin from clinical chemistry
- Demographic variables: age, sex, race/ethnicity, BMI, health conditions

**Download Instructions:**

1. Visit CDC NHANES Laboratory Data page: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory
2. Search for "Testosterone" → select cycles 2011–2012, 2013–2014, 2015–2016
3. Download the following datasets:
   - **Testosterone, Total (TST):** Files TST_G (2011–2012), TST_H (2013–2014), TST_I (2015–2016)
   - **SHBG:** Files SHBG_G, SHBG_H, SHBG_I (if separate) or check Reproductive Health module
   - **Albumin (ALB):** Files ALB_G, ALB_H, ALB_I in Chemistry or Albumin module
4. File format: .XPT (SAS transport format). Read using Python:

```python
import pandas as pd
df = pd.read_sas('TST_G.xpt')
```

Or convert via CDC API:
```python
import requests
# CDC NHANES API endpoint
url = "https://data.cdc.gov/api/views/..."
response = requests.get(url)
data = response.json()
```

**Data completeness note:** Not all participants have all three measures. You'll need to perform listwise or pairwise deletion. Target N ≈ 3,000–4,000 males with complete TT, SHBG, albumin after cleaning.

#### 1b. Reference External Validation Dataset: EMAS Samples

**What you'll get:**
- N ≈ 100–300 men with ED-measured FT (the gold-standard reference)
- Co-measured TT, SHBG, albumin

**Acquisition path:**

1. Locate Fiers et al. (2018) publication: "Reassessing Free-Testosterone Calculation..." *J Clin Endocrinol Metab*. 2018;103(6):2167–2174.
   - Link: https://academic.oup.com/jcem/article/103/6/2167/4956600
2. Check supplementary tables (often available as open-access PDF) for raw ED-FT and corresponding TT/SHBG values.
3. If full dataset not published, contact corresponding author (e.g., Tom Fiers, Ghent University) with a brief collaboration proposal:
   - Explain: Academic validation study, non-commercial, targeting publication in peer-reviewed journal
   - Propose: Data-sharing agreement, co-authorship consideration
   - Expected response time: 2–4 weeks
4. Alternatively, search for EMAS publications on UK Biobank or European biobanks to identify published ED-FT cohorts.

**Strategic importance:** This dataset is your "external validation" gold standard. Do not train on it—reserve entirely for final model evaluation.

#### 1c. Secondary Option: UK Biobank (if time permits)

**Application process:**
1. Register at https://www.ukbiobank.ac.uk/access/
2. Complete project description: "Validation of free testosterone estimation algorithms in a large population cohort"
3. Submit data access request specifying:
   - Field 30890: Serum testosterone
   - Field 30600: SHBG
   - Field 30610: Albumin
   - Assay metadata (if available)
4. Wait for approval (typically 4–8 weeks)
5. Download via secure portal
6. Note: Calculated FT in UK Biobank is typically derived, not measured by ED; use only as secondary validation cohort

**Timeline note:** If UK Biobank approval is delayed, proceed with NHANES + EMAS; UK Biobank can be folded in as extended external validation post-publication.

#### 1d. Data Harmonization Protocol

**Standardize all inputs to SI units (nmol/L):**

```python
# Unit conversions
def ng_dl_to_nmol_l(value_ng_dl):
    return value_ng_dl * 0.03467

def nmol_l_to_ng_dl(value_nmol_l):
    return value_nmol_l / 0.03467

def mg_dl_to_g_l(value_mg_dl):
    return value_mg_dl / 100

def g_l_to_mg_dl(value_g_l):
    return value_g_l * 100
```

**Create a single harmonized CSV with columns:**

| Column | Units | Source | Notes |
|--------|-------|--------|-------|
| sample_id | NA | NHANES/EMAS | Unique identifier |
| sex | M/F | NHANES/EMAS | Categorical |
| age_years | years | NHANES/EMAS | Numeric |
| tt_nmoll | nmol/L | TT assay | Total testosterone |
| shbg_nmoll | nmol/L | SHBG assay | Sex hormone-binding globulin |
| alb_gl | g/L | Albumin assay | Serum albumin |
| ft_ed_nmoll | nmol/L | EMAS/ED | Measured free testosterone (reference) |
| dataset_source | string | metadata | "NHANES_2011_12" or "EMAS_Fiers" |
| tt_assay_method | string | metadata | "LCMSMS" or "IA" (immunoassay) |

**Missing data strategy:**

- **Albumin missing:** Support two modes of model:
  - Model A: Requires albumin (strict)
  - Model B: Omits albumin, uses simpler binding to SHBG only
  - Rationale: Many rural labs measure TT and SHBG but not albumin; model B ensures utility in low-resource settings

- For samples with missing albumin, impute using age/sex/BMI median (only for training dataset, not validation)

**Data quality checks:**
- Flag and exclude: TT < 0.5 nmol/L (likely assay error)
- Flag and exclude: SHBG > 300 nmol/L or < 1 nmol/L (pathological or error)
- Flag and exclude: Albumin < 30 g/L (severe hypoalbuminemia, confounding)
- Create a "data_quality_flag" column to document exclusions

**Deliverable at end of Phase 1:**

1. harmonized_data.csv (N ≈ 4,000 training samples, N ≈ 150 external validation samples)
2. data_quality_report.txt (summary of inclusions, exclusions, missing data imputation)
3. exploratory_data_analysis.ipynb (summary statistics, distributions by sex, scatter plots TT vs SHBG)

---

### Phase 2: Mathematical Model Derivation & Solver Implementation (Weeks 1–2, parallel with Phase 1)

**Objective:** Derive mechanistic solvers and prepare Python implementations.

#### 2a. Vermeulen Cubic Solver

**Symbolic derivation:**

Starting from mass-action equilibrium:
- TT = FT + (SHBG × FT × K_SHBG / (1 + FT × K_SHBG)) + (Albumin × FT × K_ALB / (1 + FT × K_ALB))

Rearranging to cubic form:

```
a × FT³ + b × FT² + c × FT + d = 0
```

Where:
```
a = K_SHBG × K_ALB
b = K_SHBG + K_ALB + K_SHBG × K_ALB × (SHBG + Albumin - TT)
c = 1 + K_SHBG × SHBG + K_ALB × Albumin - (K_SHBG + K_ALB) × TT
d = -TT
```

**Physically meaningful root:** The only valid root is 0 < FT < TT.

**Numerical solver implementation:**

Python-ready pseudocode:

```python
import numpy as np
from scipy.optimize import brentq, newton

def calc_free_testosterone_vermeulen(tt_nmoll, shbg_nmoll, alb_gl, 
                                    K_shbg=1e9, K_alb=3.6e4):
    """
    Calculate free testosterone using Vermeulen (1999) cubic formula.
    
    Parameters:
    -----------
    tt_nmoll : float
        Total testosterone in nmol/L
    shbg_nmoll : float
        SHBG in nmol/L
    alb_gl : float
        Albumin in g/L (convert mg/dL to g/L if needed: alb_gl = alb_mgdl / 100)
    K_shbg : float
        Association constant for SHBG (default 1e9 L/mol)
    K_alb : float
        Association constant for albumin (default 3.6e4 L/mol)
    
    Returns:
    --------
    ft_nmoll : float
        Free testosterone in nmol/L
    
    Raises:
    -------
    ValueError if inputs invalid
    """
    
    # Input validation
    if tt_nmoll <= 0 or shbg_nmoll < 0 or alb_gl <= 0:
        raise ValueError("Inputs must be positive")
    if np.isnan(tt_nmoll) or np.isnan(shbg_nmoll) or np.isnan(alb_gl):
        raise ValueError("NaN in inputs")
    
    # Convert albumin if provided in mg/dL
    if alb_gl > 100:  # Likely in mg/dL, convert to g/L
        alb_gl = alb_gl / 100
    
    # Cubic coefficients
    a = K_shbg * K_alb
    b = K_shbg + K_alb + K_shbg * K_alb * (shbg_nmoll + alb_gl - tt_nmoll)
    c = 1 + K_shbg * shbg_nmoll + K_alb * alb_gl - (K_shbg + K_alb) * tt_nmoll
    d = -tt_nmoll
    
    # Define cubic function
    def cubic(ft):
        return a * ft**3 + b * ft**2 + c * ft + d
    
    def cubic_prime(ft):
        return 3 * a * ft**2 + 2 * b * ft + c
    
    # Bracketing interval (FT must be between 0 and TT)
    try:
        # Use safeguarded root-finding: Brent's method with Newton fallback
        ft = brentq(cubic, 0, tt_nmoll, xtol=1e-12)
    except ValueError:
        # Fallback: Newton–Raphson with bracketed bisection
        try:
            ft = newton(cubic, tt_nmoll / 2, fprime=cubic_prime, tol=1e-12)
            # Ensure within bounds
            ft = np.clip(ft, 0, tt_nmoll)
        except:
            raise RuntimeError("Root solver failed")
    
    return float(ft)


def calc_bioavailable_testosterone_vermeulen(tt_nmoll, shbg_nmoll, alb_gl, K_shbg=1e9, K_alb=3.6e4):
    """
    Bioavailable testosterone = FT + albumin-bound testosterone.
    """
    ft = calc_free_testosterone_vermeulen(tt_nmoll, shbg_nmoll, alb_gl, K_shbg, K_alb)
    alb_bound = (tt_nmoll - ft) * (K_alb / (1 + K_alb * alb_gl))
    return ft + alb_bound
```

**Numeric validation test:**

Using example from Vermeulen paper:
- TT = 15 nmol/L, SHBG = 40 nmol/L, Albumin = 45 g/L
- Expected FT ≈ 0.30 nmol/L (from paper)

```python
ft_result = calc_free_testosterone_vermeulen(15, 40, 45)
assert abs(ft_result - 0.30) < 1e-6, f"Expected 0.30, got {ft_result}"
```

#### 2b. Södergård Solver

Södergård (1982) used different association constants but the same cubic structure:

```python
def calc_free_testosterone_sodergard(tt_nmoll, shbg_nmoll, alb_gl):
    """
    Södergård (1982) variant of cubic formula.
    Uses Kshbg ≈ 1.2e9, Kalb ≈ 2.4e4
    """
    K_shbg = 1.2e9
    K_alb = 2.4e4
    return calc_free_testosterone_vermeulen(tt_nmoll, shbg_nmoll, alb_gl, K_shbg, K_alb)
```

#### 2c. Zakharov Allosteric Solver

Zakharov proposes a more complex system:

```
# Simplified version (full derivation would require numerical Jacobian)
def calc_free_testosterone_zakharov(tt_nmoll, shbg_nmoll, alb_gl, 
                                   cooperativity_factor=0.5):
    """
    Zakharov (2015) allosteric model (simplified numeric solver).
    Includes cooperativity parameter α.
    """
    from scipy.optimize import fsolve
    
    # Allosteric equilibrium constants
    K1 = 1.5e9
    K2 = 0.8e9 * (1 + cooperativity_factor)
    K_alb = 3.6e4
    
    # System of equations (simplified)
    def equations(vars):
        ft, shbg_bound = vars
        alb_bound = tt_nmoll - ft - shbg_bound
        
        eq1 = shbg_bound - ((K1 * ft * shbg_nmoll) / (1 + K2 * ft))
        eq2 = alb_bound - ((K_alb * ft * alb_gl) / (1 + K_alb * ft))
        
        return [eq1, eq2]
    
    # Initial guess
    ft_initial = tt_nmoll * 0.03
    shbg_initial = tt_nmoll * 0.60
    
    solution = fsolve(equations, [ft_initial, shbg_initial])
    ft = solution[0]
    
    # Ensure bounds
    ft = np.clip(ft, 0, tt_nmoll)
    return float(ft)
```

**Deliverable at end of Phase 2:**

1. solvers.py (Python module with Vermeulen, Södergård, Zakharov solvers)
2. test_solvers.py (unit tests with numeric validation)
3. solver_comparison_notebook.ipynb (side-by-side evaluation of three solvers on NHANES data)

---

### Phase 3: Empirical Model Development & Training (Weeks 2–3)

**Objective:** Train machine-learning models to improve prediction accuracy beyond mechanistic solvers.

#### 3a. Prepare Training Data Pipeline

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Load harmonized data
df = pd.read_csv('harmonized_data.csv')

# Filter: only training data, exclude external validation set
df_train = df[df['dataset_source'] == 'NHANES'].copy()
df_val_external = df[df['dataset_source'] == 'EMAS_Fiers'].copy()

# Feature engineering
X = df_train[['tt_nmoll', 'shbg_nmoll', 'alb_gl', 'age_years']].copy()
X['sex_numeric'] = (df_train['sex'] == 'M').astype(int)
X['tt_assay_numeric'] = (df_train['tt_assay_method'] == 'LCMSMS').astype(int)
X['shbg_tt_ratio'] = X['shbg_nmoll'] / (X['tt_nmoll'] + 1e-6)

# Add reference from Vermeulen solver (as baseline feature)
X['ft_vermeulen'] = df_train.apply(
    lambda row: calc_free_testosterone_vermeulen(row['tt_nmoll'], row['shbg_nmoll'], row['alb_gl']),
    axis=1
)

# Target (for training purposes, use indirect ED reference if available)
# For NHANES, we'll use Vermeulen as a proxy and fine-tune; for EMAS subset, use actual ED-FT
y_target = None  # Placeholder; will split into regression targets

# Stratified split: 70% train, 30% internal test (stratified by sex, SHBG tertile)
df_train['shbg_tertile'] = pd.qcut(df_train['shbg_nmoll'], q=3, labels=['low', 'mid', 'high'])
df_train['strat_group'] = df_train['sex'] + '_' + df_train['shbg_tertile']

X_train, X_test, y_train, y_test, strat_train, strat_test = train_test_split(
    X, y_target, df_train['strat_group'],
    test_size=0.3, stratify=df_train['strat_group'], random_state=42
)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### 3b. Candidate Models

Train the following in parallel:

1. **Ridge Regression** (baseline regularized linear model)
2. **Random Forest** (captures nonlinear interactions)
3. **LightGBM** (gradient boosting; handles missing data)
4. **Neural Network** (optional; test if justified by CV error reduction)

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Ridge
ridge_model = Ridge(alpha=1.0, solver='auto')
ridge_model.fit(X_train_scaled, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)  # Note: RF doesn't require scaling

# LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'random_state': 42
}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(20)])
```

#### 3c. Cross-Validation & Model Selection

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 10-fold stratified CV
cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def cross_validate_model(model, X, y, cv):
    scores = cross_validate(model, X, y, cv=cv, 
                           scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'],
                           return_train_score=True)
    
    rmse_cv = np.sqrt(-scores['test_neg_mean_squared_error']).mean()
    mae_cv = -scores['test_neg_mean_absolute_error'].mean()
    
    return rmse_cv, mae_cv

rmse_ridge, mae_ridge = cross_validate_model(ridge_model, X_train_scaled, y_train, cv_splitter)
rmse_rf, mae_rf = cross_validate_model(rf_model, X_train, y_train, cv_splitter)
rmse_lgb, mae_lgb = cross_validate_model(lgb_model, X_train, y_train, cv_splitter)

# Select best model
print(f"Ridge RMSE: {rmse_ridge:.4f}, MAE: {mae_ridge:.4f}")
print(f"RF RMSE: {rmse_rf:.4f}, MAE: {mae_rf:.4f}")
print(f"LightGBM RMSE: {rmse_lgb:.4f}, MAE: {mae_lgb:.4f}")
```

**Selection criterion:** Choose model with lowest RMSE and interpretability trade-off (prefer Random Forest or Ridge if LightGBM only marginal improvement).

**Deliverable at end of Phase 3:**

1. trained_models/ (pickled Ridge, RF, LightGBM models)
2. model_comparison_cv_results.csv (cross-validation metrics)
3. feature_importance_plots.pdf (feature importance from RF/LightGBM)
4. train_pipeline.py (reproducible training script)

---

### Phase 4: Internal Validation & Error Analysis (Weeks 3–4)

**Objective:** Evaluate all models (mechanistic + empirical) on internal test set and analyze failure modes.

#### 4a. Evaluation Metrics

```python
import numpy as np
from scipy import stats

def bland_altman_stats(y_true, y_pred):
    """Bland-Altman agreement analysis."""
    diff = y_pred - y_true
    mean_val = (y_true + y_pred) / 2
    
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff
    
    return {
        'mean_bias': mean_diff,
        'std_diff': sd_diff,
        'loa_lower': loa_lower,
        'loa_upper': loa_upper,
        'loa_width': loa_upper - loa_lower
    }

def lins_ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient."""
    x = np.asarray(y_true)
    y = np.asarray(y_pred)
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    
    rho = 2 * cov_xy / (var_x + var_y + (mean_x - mean_y)**2)
    return rho

def evaluate_model(y_true, y_pred, model_name='Model'):
    """Comprehensive evaluation."""
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mae = np.mean(np.abs(y_pred - y_true))
    bias = np.mean(y_pred - y_true)
    r_pearson = np.corrcoef(y_true, y_pred)[0, 1]
    ccc = lins_ccc(y_true, y_pred)
    ba = bland_altman_stats(y_true, y_pred)
    
    # Percentage within acceptable error bounds
    pct_within_10pct = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6)) <= 0.10) * 100
    pct_within_20pct = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6)) <= 0.20) * 100
    
    results = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'r_pearson': r_pearson,
        'lin_ccc': ccc,
        'ba_mean_diff': ba['mean_bias'],
        'ba_loa_lower': ba['loa_lower'],
        'ba_loa_upper': ba['loa_upper'],
        'ba_loa_width': ba['loa_width'],
        'pct_within_10pct': pct_within_10pct,
        'pct_within_20pct': pct_within_20pct
    }
    
    return results

# Evaluate all models on internal test set
results_list = []

# Mechanistic models
for row in X_test.iterrows():
    idx, x = row
    y_pred_v = calc_free_testosterone_vermeulen(x['tt_nmoll'], x['shbg_nmoll'], x['alb_gl'])
    y_true = y_test.iloc[idx]
    results_list.append(evaluate_model([y_true], [y_pred_v], 'Vermeulen'))

results_vermeulen = evaluate_model(y_test, X_test.apply(lambda x: calc_free_testosterone_vermeulen(...), axis=1), 'Vermeulen')
results_sodergard = evaluate_model(y_test, X_test.apply(lambda x: calc_free_testosterone_sodergard(...), axis=1), 'Sodergard')
results_zakharov = evaluate_model(y_test, X_test.apply(lambda x: calc_free_testosterone_zakharov(...), axis=1), 'Zakharov')

# Empirical models
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)

results_ridge = evaluate_model(y_test, y_pred_ridge, 'Ridge')
results_rf = evaluate_model(y_test, y_pred_rf, 'RandomForest')
results_lgb = evaluate_model(y_test, y_pred_lgb, 'LightGBM')

# Compile results table
results_df = pd.DataFrame([results_vermeulen, results_sodergard, results_zakharov,
                           results_ridge, results_rf, results_lgb])
results_df.to_csv('internal_validation_results.csv', index=False)
print(results_df)
```

#### 4b. Subgroup Analysis

```python
# By sex
for sex in ['M', 'F']:
    mask = X_test['sex_numeric'] == (sex == 'M')
    y_test_sex = y_test[mask]
    y_pred_sex = y_pred_rf[mask]
    results_sex = evaluate_model(y_test_sex, y_pred_sex, f'RandomForest_Sex={sex}')
    print(results_sex)

# By SHBG tertile
X_test['shbg_tertile'] = pd.qcut(X_test['shbg_nmoll'], q=3, labels=['Low', 'Mid', 'High'])
for tertile in ['Low', 'Mid', 'High']:
    mask = X_test['shbg_tertile'] == tertile
    y_test_tertile = y_test[mask]
    y_pred_tertile = y_pred_rf[mask]
    results_tertile = evaluate_model(y_test_tertile, y_pred_tertile, f'RandomForest_SHBG={tertile}')
    print(results_tertile)
```

**Deliverable at end of Phase 4:**

1. internal_validation_results.csv (all metrics by model)
2. bland_altman_plots.pdf (6 subplots: overall + by sex + by SHBG tertile)
3. predicted_vs_observed_scatter.pdf (predicted vs observed with 95% CI)
4. calibration_plot.pdf (decile-wise calibration slope/intercept)
5. residual_analysis.ipynb (residuals vs predicted, Q-Q plots)

---

### Phase 5: External Validation on ED-Measured Cohort (Weeks 4–5)

**Objective:** Evaluate best-performing model on independent ED reference dataset (EMAS).

#### 5a. External Validation Protocol

```python
# Load external ED-measured validation set
df_val_ed = pd.read_csv('emas_ed_measured_data.csv')

# Apply same feature engineering
X_val_ed = df_val_ed[['tt_nmoll', 'shbg_nmoll', 'alb_gl', 'age_years']].copy()
X_val_ed['sex_numeric'] = (df_val_ed['sex'] == 'M').astype(int)
X_val_ed['ft_vermeulen'] = ...  # Add Vermeulen feature

# Get predictions from best model (e.g., RandomForest)
y_pred_val = best_model.predict(X_val_ed)
y_true_val = df_val_ed['ft_ed_nmoll']  # True ED-measured FT

# Evaluate
results_external = evaluate_model(y_true_val, y_pred_val, 'RandomForest_ExternalValidation')
print(results_external)

# Subgroup analysis by sex and SHBG
for sex in ['M', 'F']:
    for shbg_level in ['Low', 'Mid', 'High']:
        mask = (df_val_ed['sex'] == sex) & (df_val_ed['shbg_tertile'] == shbg_level)
        if mask.sum() >= 20:  # Minimum sample size
            y_true_sub = y_true_val[mask]
            y_pred_sub = y_pred_val[mask]
            results_sub = evaluate_model(y_true_sub, y_pred_sub, f'Sex={sex}_SHBG={shbg_level}')
            print(results_sub)
```

#### 5b. Bootstrap Confidence Intervals

```python
from scipy import stats

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=2000, ci=95):
    """Compute bootstrap CI for any metric."""
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        score = metric_func(y_true[idx], y_pred[idx])
        bootstrap_scores.append(score)
    
    lower = np.percentile(bootstrap_scores, (100 - ci) / 2)
    upper = np.percentile(bootstrap_scores, 100 - (100 - ci) / 2)
    
    return lower, upper, np.mean(bootstrap_scores)

# Compute bootstrap CIs for key metrics
rmse_lower, rmse_upper, rmse_mean = bootstrap_ci(
    y_true_val, y_pred_val, 
    lambda y_t, y_p: np.sqrt(np.mean((y_p - y_t)**2))
)
ccc_lower, ccc_upper, ccc_mean = bootstrap_ci(
    y_true_val, y_pred_val,
    lins_ccc
)

print(f"RMSE: {rmse_mean:.4f} (95% CI: {rmse_lower:.4f}–{rmse_upper:.4f})")
print(f"Lin's CCC: {ccc_mean:.4f} (95% CI: {ccc_lower:.4f}–{ccc_upper:.4f})")
```

**Clinical acceptability threshold assessment:**

- **RMSE target:** < 1.0 nmol/L (±10% of mean FT in clinical range)
- **Bias target:** < ±0.5 nmol/L
- **Lin's CCC target:** ≥ 0.90
- **95% LoA:** Within ±2 nmol/L (proposed clinically meaningful difference)

**Deliverable at end of Phase 5:**

1. external_validation_results.csv (all metrics + bootstrap CIs)
2. bland_altman_external.pdf (overall and by sex/SHBG)
3. roc_curve_clinical_threshold.pdf (if binary diagnostic threshold applicable)
4. clinical_acceptability_report.md (summary of success vs. pre-specified thresholds)

---

### Phase 6: Publication Preparation & Code Release (Weeks 5–6)

**Objective:** Package code, write manuscript, prepare for peer-reviewed publication.

#### 6a. Python Package Structure

```
freeT/
├── freeT/
│   ├── __init__.py
│   ├── models.py              # Mechanistic solvers (Vermeulen, Sodergard, Zakharov)
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation metrics
│   ├── utils.py               # Unit conversions, data cleaning
│   └── predict.py             # Inference API
├── tests/
│   ├── test_models.py         # Unit tests for solvers
│   ├── test_train.py          # Tests for training pipeline
│   └── test_edge_cases.py     # Edge case and robustness tests
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_model_development.ipynb  # Training and cross-validation
│   ├── 03_evaluation.ipynb    # Internal validation
│   └── 04_external_validation.ipynb  # ED-measured validation
├── data/
│   └── harmonized_data.csv    # (or symlink to data directory)
├── requirements.txt           # Dependencies (numpy, pandas, scikit-learn, scipy, matplotlib)
├── setup.py                   # Package installation
├── README.md                  # User guide
└── LICENSE                    # MIT or Apache 2.0
```

#### 6b. Manuscript Outline (TRIPOD+AI Compliant)

**Title:** "Development and External Validation of a Machine Learning Algorithm for Free Testosterone Estimation from Total Testosterone, SHBG, and Albumin"

**Structure:**

1. **Abstract** (250 words)
   - Background, objective, methods, results, conclusion

2. **Introduction** (2–3 pages)
   - Clinical need for FT
   - Limitations of existing methods
   - Study objective

3. **Methods** (4–5 pages per TRIPOD+AI)
   - Data sources (NHANES, EMAS)
   - Inclusion/exclusion criteria
   - Feature definitions and units
   - Model development (Vermeulen baseline, empirical models)
   - Cross-validation strategy
   - External validation approach
   - Performance metrics and acceptability thresholds

4. **Results** (3–4 pages)
   - Participant characteristics (Table 1)
   - Internal validation results (Table 2, Bland–Altman plot)
   - Model comparison (Figure 1)
   - Subgroup analysis (Table 3, Figure 2)
   - External validation on ED-measured data (Table 4, Figure 3)
   - Bootstrap CIs (Table 5)

5. **Discussion** (3–4 pages)
   - Key findings and clinical implications
   - Comparison to prior models
   - Limitations
   - Generalizability to other populations
   - Practical implementation guidance

6. **Clinical Considerations** (2 pages)
   - When to use, contraindications, future validation needs

7. **Supplementary Materials**
   - Raw code (GitHub repo)
   - Extended validation tables
   - Solver pseudocode

#### 6c. Code Release Checklist

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code formatted (black, flake8)
- [ ] Type hints on all functions
- [ ] Docstrings complete
- [ ] README with installation and usage examples
- [ ] LICENSE file (MIT recommended for academic)
- [ ] GitHub repository with version tags
- [ ] Zenodo DOI for code archive
- [ ] Example Jupyter notebook with reproducible analysis

#### 6d. Pre-Publication Review

- Identify 2–3 clinical collaborators to review draft for clinical reasonableness
- Solicit feedback from biostatisticians on validation design
- Submit to preprint server (bioRxiv) for community feedback

**Target journals:** *Journal of Clinical Endocrinology & Metabolism*, *Endocrine Reviews*, *Clinical Chemistry*, or *Diagnostic*, *Pathology and Laboratory Medicine* journals

**Deliverable at end of Phase 6:**

1. freeT/ (Python package, GitHub-ready)
2. manuscript_draft.docx (full manuscript + supplementary materials)
3. supporting_data.xlsx (all validation tables)
4. TRIPOD+AI_compliance_checklist.pdf (verification)

---

## Part 3: Deployment Guidance for Rural & Low-Resource Clinics

### 3.1 Implementation Strategy

**Use Case 1: Laboratory Integration**
- Integrate freeT algorithm into laboratory information system (LIS)
- Input: TT (ng/dL or nmol/L), SHBG (nmol/L), albumin (g/L)
- Output: Calculated FT with reference range and clinical interpretation

**Use Case 2: Point-of-Care Decision Support**
- Web interface or mobile app for clinicians to input hormone values
- Real-time FT prediction with uncertainty (confidence interval)
- Integration with clinical guidelines for interpretation

**Use Case 3: Epidemiologic Screening**
- Population-level FT estimation for prevalence studies
- Batch processing of large cohorts with minimal data

### 3.2 Clinical Interpretation Guidance

**Normal ranges** (based on ED-measured reference intervals, Handelsman et al.):
- **Men (19–39 years):** 415–1,274 pmol/L (120–368 pg/mL) or ~6–18 nmol/L
- **Men (≥40 years):** Lower age-adjusted thresholds (consult local guidelines)
- **Women (reproductive age):** 25–100 pmol/L (~0.7–3 ng/mL) or ~0.07–0.35 nmol/L

**Clinical cutoffs:**
- **Hypogonadism diagnosis:** FT < 225 pmol/L (consensus threshold in many guidelines)
- **Androgen excess (women):** FT > 70 pmol/L (context-dependent)

**Important caveats:**
- Model trained on NHANES (US population) and EMAS (European cohort); validation in other populations pending
- Assumes normal albumin and SHBG; not validated in liver disease, thyroid disease, or pregnancy
- Do not use as sole diagnostic criterion; combine with clinical symptoms and LH/FSH

### 3.3 Quality Assurance Protocol

**Monthly checks:**
- Re-run internal validation on same 20 quality-control samples (compute bias, SD)
- Alert if bias > ±15% or SD increases > 20%
- Document all alerts and corrective actions

**Annual recalibration:**
- If systematic drift observed, refit empirical model on new reference data
- Communicate version updates to all implementing labs

### 3.4 Training Materials for Clinicians

1. **One-page clinical summary:** When to order FT, interpretation, limitations
2. **Video tutorial (5 min):** How to use web interface or app
3. **FAQ document:** Common questions (e.g., "Why is my calculated FT different from lab X?")
4. **Case studies:** 3–5 worked examples (hypogonadal man, hyperandrogenic woman, etc.)

---

## Part 4: Success Criteria & Contingency Plans

### 4.1 Milestones & Go/No-Go Decisions

| Week | Milestone | Success Criterion | Go/No-Go |
|------|-----------|-------------------|----------|
| 3 | Data harmonized | N ≥ 3,500 NHANES + N ≥ 100 EMAS ED-measured | Go |
| 5 | Models trained | RMSE < 2.0 nmol/L on internal CV | Go |
| 7 | External validation complete | Lin's CCC ≥ 0.85 on EMAS ED data | Go/conditional |
| 8 | Manuscript drafted | TRIPOD+AI checklist ≥ 80% complete | Go |
| 12 | Publication submitted | Accepted or minor revisions | Go |
| 24 | Clinical deployment | Implementation in 2+ rural clinics | Success |

**Conditional go criterion (Week 7):** If external validation CCC = 0.80–0.85, proceed with publication but label model as "validation pending" and recruit additional ED-measured cohorts for Phase 2 validation.

### 4.2 Contingency Plans

**Risk: NHANES data incomplete**
- **Mitigation:** Supplement with UK Biobank or alternative cohort (EMAS, EPIC); adjust timeline +3 weeks

**Risk: ED-measured reference data unavailable**
- **Mitigation:** Use high-confidence calculated FT (Vermeulen) as pseudo-reference; note limitation in manuscript; seek ED data post-publication

**Risk: Model performance suboptimal (CCC < 0.85)**
- **Mitigation:** Increase model complexity (neural network, ensemble), add additional covariates (estradiol, LH), reframe as "proof-of-concept" rather than ready-for-deployment

**Risk: Computational complexity too high for rural clinic integration**
- **Mitigation:** Release simplified Ridge regression model; trade marginal accuracy for ease of deployment

---

## Part 5: Long-Term Roadmap (Post-Publication)

### Year 1–2: Phase 2 Validation
- Recruit 300–500 prospective patients with ED-measured FT from multiple countries/populations
- Test algorithm on ethnic, BMI, and disease state subgroups (obese, diabetic, transgender on HRT)
- Update model coefficients if systematic bias detected in subgroups

### Year 2–3: Clinical Trial Integration
- Embedded prospective trial: randomize clinics to "calculated FT-guided" vs. "TT alone" hypogonadism diagnosis
- Primary outcome: diagnostic agreement with ED; secondary: treatment decisions and patient outcomes

### Year 3+: Health System Implementation
- Partner with 5–10 rural clinics/regional labs in non-US settings
- Integration into LIS; training of clinicians
- Real-world performance monitoring and continuous improvement

---

## References & Key Resources

[1] Vermeulen A, Verdonck L, Kaufman JM. A critical evaluation of simple methods for the estimation of free testosterone in serum. *J Clin Endocrinol Metab*. 1999;84(10):3666–3672. https://doi.org/10.1210/jcem.84.10.6079

[2] Fiers T, Wu F, Moghetti P, Vanderschueren D, Lapauw B, Kaufman JM. Reassessing free-testosterone calculation by liquid chromatography-tandem mass spectrometry direct equilibrium dialysis. *J Clin Endocrinol Metab*. 2018;103(6):2167–2174. https://doi.org/10.1210/jc.2017-02360

[3] Collins GS, Moons KGM, Dhiman P, Riley RD, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378

[4] Handelsman DJ, Hirschberg AL, Bermon S. Circulating testosterone as the hormonal basis of sex differences in athletic performance. *Endocr Rev*. 2015;36(1):1–48.

[5] CDC NHANES Laboratory Data: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory

[6] EMAS Study Resources: https://emas-online.org/

[7] TRIPOD Statement: https://www.tripod-statement.org/

---

## Appendix: Quick-Start Checklist for You

### Immediate Actions (This Week)

- [ ] Create project directory and GitHub repository
- [ ] Download NHANES testosterone files (TST_G, TST_H, TST_I) from CDC
- [ ] Identify EMAS ED-measured datasets or contact Tom Fiers (Ghent University)
- [ ] Email UK Biobank to inquire about access timeline (optional)
- [ ] Set up Python environment (`pip install -r requirements.txt`)

### Phase 1 Actions (Weeks 1–3)

- [ ] Parse NHANES .XPT files and combine into single dataframe
- [ ] Harmonize units (ng/dL → nmol/L, mg/dL → g/L)
- [ ] Create harmonized_data.csv with all required columns
- [ ] Run exploratory data analysis notebook

### Phase 2 Actions (Weeks 2–3, parallel)

- [ ] Implement Vermeulen cubic solver with unit tests
- [ ] Test with known numeric examples from published literature
- [ ] Create solvers.py module

### Critical Success Factors

1. **Data quality:** Clean, complete TT/SHBG/albumin harmonization is foundational
2. **External validation:** ED-measured reference data is non-negotiable; prioritize acquiring EMAS dataset early
3. **Documentation:** Write clear docstrings and notebooks; future collaborators and reviewers will scrutinize reproducibility
4. **Publication readiness:** TRIPOD+AI compliance from day one; don't retrofit reporting standards at the end

---

## Contact & Collaboration

For questions on specific methods, datasets, or implementation:
- **Mathematical modeling:** Consult published derivations in Vermeulen, Fiers, Zakharov papers
- **Dataset acquisition:** Contact corresponding authors; most are willing to share deidentified data for collaborative validation
- **TRIPOD+AI compliance:** Visit https://www.tripod-statement.org/ for detailed checklist and examples
- **Clinical guidance:** Engage endocrinologists or primary care physicians early to co-develop clinical decision rules

---

**Document Version:** 1.0 (Draft)  
**Last Updated:** January 2026  
**Status:** Ready for Implementation  
