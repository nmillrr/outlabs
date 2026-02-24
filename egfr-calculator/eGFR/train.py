"""
eGFR/train.py — ML Model Training

Provides functions for:
  - Feature engineering from clinical data
  - Stratified train/test splitting by CKD stage
  - Training Ridge, Random Forest, and LightGBM models
  - Cross-validation evaluation
  - Model persistence (save/load)
"""

from __future__ import annotations

import math
import warnings
from typing import Union

import numpy as np
import pandas as pd

from eGFR.models import (
    calc_egfr_ckd_epi_2021,
    calc_egfr_mdrd,
    calc_crcl_cockcroft_gault,
)


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create feature matrix from a cleaned clinical DataFrame.

    Constructs the full set of ML features used by the hybrid eGFR model.
    The input DataFrame should come from ``clean_kidney_data()`` and must
    contain at minimum: ``cr_mgdl``, ``age_years``, ``sex``, ``weight_kg``,
    ``height_cm``.

    Feature groups
    --------------
    1. **Base clinical** — cr_mgdl, age_years, sex_numeric, weight_kg,
       height_cm, bmi
    2. **Cystatin C** (if column present) — cystatin_c_mgL, cr_cys_ratio
    3. **Mechanistic estimators** — egfr_ckd_epi_2021, egfr_mdrd,
       crcl_cockcroft_gault
    4. **Derived** — inv_creatinine (1/cr), log_creatinine (ln(cr)),
       age_cr_interaction (age × cr)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned clinical data.  Required columns: ``cr_mgdl``,
        ``age_years``, ``sex`` (1/2 or 'M'/'F'), ``weight_kg``,
        ``height_cm``.  Optional: ``cystatin_c_mgL``.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(X, feature_names)`` where *X* is a DataFrame of numeric features
        and *feature_names* is the ordered list of column names.

    Raises
    ------
    ValueError
        If any required column is missing from *df*.
    """
    _REQUIRED = ["cr_mgdl", "age_years", "sex", "weight_kg", "height_cm"]
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = pd.DataFrame(index=df.index)

    # --- 1. Base clinical features ---
    X["cr_mgdl"] = df["cr_mgdl"].astype(float)
    X["age_years"] = df["age_years"].astype(float)
    X["sex_numeric"] = df["sex"].apply(_sex_to_numeric).astype(float)
    X["weight_kg"] = df["weight_kg"].astype(float)
    X["height_cm"] = df["height_cm"].astype(float)

    # BMI = weight (kg) / height (m)^2
    height_m = X["height_cm"] / 100.0
    X["bmi"] = X["weight_kg"] / (height_m ** 2)

    # --- 2. Cystatin C features (optional) ---
    has_cystatin = "cystatin_c_mgL" in df.columns
    if has_cystatin:
        X["cystatin_c_mgL"] = df["cystatin_c_mgL"].astype(float)
        X["cr_cys_ratio"] = X["cr_mgdl"] / X["cystatin_c_mgL"]

    # --- 3. Mechanistic estimator predictions ---
    X["egfr_ckd_epi_2021"] = df.apply(
        lambda r: _safe_ckd_epi(r["cr_mgdl"], r["age_years"], r["sex"]),
        axis=1,
    )
    X["egfr_mdrd"] = df.apply(
        lambda r: _safe_mdrd(r["cr_mgdl"], r["age_years"], r["sex"]),
        axis=1,
    )
    X["crcl_cockcroft_gault"] = df.apply(
        lambda r: _safe_cg(
            r["cr_mgdl"], r["age_years"], r["weight_kg"], r["sex"]
        ),
        axis=1,
    )

    # --- 4. Derived features ---
    X["inv_creatinine"] = 1.0 / X["cr_mgdl"]
    X["log_creatinine"] = np.log(X["cr_mgdl"])
    X["age_cr_interaction"] = X["age_years"] * X["cr_mgdl"]

    feature_names = list(X.columns)
    return X, feature_names


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sex_to_numeric(sex) -> float:
    """Convert sex value to numeric (0 = male, 1 = female)."""
    if isinstance(sex, str):
        s = sex.strip().upper()
        if s in ("M", "MALE"):
            return 0.0
        if s in ("F", "FEMALE"):
            return 1.0
    if isinstance(sex, (int, float)):
        if sex == 1:   # NHANES male
            return 0.0
        if sex == 2:   # NHANES female
            return 1.0
    return float("nan")


def _safe_ckd_epi(cr: float, age: float, sex) -> float:
    """Compute CKD-EPI 2021 eGFR, returning NaN on invalid inputs."""
    try:
        return calc_egfr_ckd_epi_2021(float(cr), float(age), sex)
    except (ValueError, TypeError):
        return float("nan")


def _safe_mdrd(cr: float, age: float, sex) -> float:
    """Compute MDRD eGFR, returning NaN on invalid inputs."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return calc_egfr_mdrd(float(cr), float(age), sex)
    except (ValueError, TypeError):
        return float("nan")


def _safe_cg(cr: float, age: float, weight: float, sex) -> float:
    """Compute Cockcroft-Gault CrCl, returning NaN on invalid inputs."""
    try:
        return calc_crcl_cockcroft_gault(
            float(cr), float(age), float(weight), sex
        )
    except (ValueError, TypeError):
        return float("nan")
