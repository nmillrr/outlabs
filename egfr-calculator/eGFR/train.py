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
from sklearn.model_selection import train_test_split

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


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets, stratified by eGFR range.

    Computes CKD-EPI 2021 eGFR for each row (used as the target *y*) and
    bins the values into six clinically meaningful ranges for stratification:

    - ≥ 90  mL/min/1.73 m²
    - 60–89
    - 45–59
    - 30–44
    - 15–29
    - < 15

    When a bin contains fewer than 2 samples (making stratified splitting
    impossible), those samples are merged into the nearest occupied bin so
    that ``train_test_split`` can proceed.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned clinical data produced by ``clean_kidney_data()``.
        Required columns: ``cr_mgdl``, ``age_years``, ``sex``,
        ``weight_kg``, ``height_cm``.
    test_size : float, default 0.3
        Fraction of the dataset to use as the test set.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``(X_train, X_test, y_train, y_test)`` where *X* is the feature
        matrix from ``create_features()`` and *y* is the CKD-EPI 2021 eGFR.

    Raises
    ------
    ValueError
        If required columns are missing from *df* or the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Build features and compute target (CKD-EPI 2021 eGFR)
    X, _ = create_features(df)
    y = X["egfr_ckd_epi_2021"].copy()

    # Drop rows where eGFR could not be computed
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if len(y) < 2:
        raise ValueError(
            "Not enough valid samples for splitting "
            f"(got {len(y)}, need ≥ 2)."
        )

    # Bin eGFR into 6 clinical ranges for stratification
    bins = [0, 15, 30, 45, 60, 90, float("inf")]
    labels = ["<15", "15-29", "30-44", "45-59", "60-89", ">=90"]
    strata = pd.cut(y, bins=bins, labels=labels, right=False).astype(str)

    # Merge tiny bins (< 2 samples) into nearest neighbour so that
    # sklearn's stratified split never fails on single-member groups.
    counts = strata.value_counts()
    for lbl in labels:
        if counts.get(lbl, 0) < 2:
            idx = labels.index(lbl)
            neighbours = (
                [labels[idx - 1]] if idx > 0 else []
            ) + (
                [labels[idx + 1]] if idx < len(labels) - 1 else []
            )
            replacement = next(
                (n for n in neighbours if counts.get(n, 0) >= 2),
                labels[-1],
            )
            strata = strata.replace(lbl, replacement)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_ridge(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    alpha: float = 1.0,
) -> "Ridge":
    """Train a Ridge regression model on the provided data.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target values (eGFR).
    alpha : float, default 1.0
        Regularisation strength (L2 penalty).

    Returns
    -------
    sklearn.linear_model.Ridge
        Fitted Ridge regression model.

    Raises
    ------
    ValueError
        If *X_train* and *y_train* have incompatible shapes or are empty.
    """
    from sklearn.linear_model import Ridge

    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=float).ravel()

    if X_arr.size == 0 or y_arr.size == 0:
        raise ValueError("Training data must not be empty.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X_train and y_train row counts differ "
            f"({X_arr.shape[0]} vs {y_arr.shape[0]})."
        )

    model = Ridge(alpha=alpha)
    model.fit(X_arr, y_arr)
    return model


def train_random_forest(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    n_estimators: int = 200,
) -> "RandomForestRegressor":
    """Train a Random Forest regression model on the provided data.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target values (eGFR).
    n_estimators : int, default 200
        Number of trees in the forest.

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Fitted Random Forest regression model.

    Raises
    ------
    ValueError
        If *X_train* and *y_train* have incompatible shapes or are empty.
    """
    from sklearn.ensemble import RandomForestRegressor

    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=float).ravel()

    if X_arr.size == 0 or y_arr.size == 0:
        raise ValueError("Training data must not be empty.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X_train and y_train row counts differ "
            f"({X_arr.shape[0]} vs {y_arr.shape[0]})."
        )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_arr, y_arr)
    return model


def save_model(model: object, filepath: str) -> None:
    """Persist a trained model to disk using joblib.

    Creates parent directories if they do not already exist.

    Parameters
    ----------
    model : object
        Any fitted scikit-learn–compatible estimator.
    filepath : str
        Destination path (e.g. ``"models/ridge_v1.joblib"``).

    Raises
    ------
    ValueError
        If *filepath* is empty.
    """
    import os
    import joblib

    if not filepath:
        raise ValueError("filepath must not be empty.")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    joblib.dump(model, filepath)


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
