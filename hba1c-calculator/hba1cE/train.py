"""
ML training utilities for HbA1c estimation.

This module contains functions for feature engineering and model training.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hba1cE.models import (
    calc_hba1c_adag,
    calc_hba1c_kinetic,
    calc_hba1c_regression,
)


def create_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature matrix for ML training from glycemic data.

    This function generates a comprehensive feature set including:
    - Raw biomarkers: FPG, TG, HDL, age, hemoglobin, MCV
    - Ratio features: TG/HDL ratio, FPG-age interaction
    - Mechanistic estimator predictions: ADAG, kinetic, regression

    Args:
        df: pandas DataFrame with columns: fpg_mgdl, tg_mgdl, hdl_mgdl,
            age_years, hgb_gdl, mcv_fl

    Returns:
        Tuple of (X, feature_names) where:
        - X: numpy array of shape (n_samples, n_features)
        - feature_names: list of feature column names

    Raises:
        ValueError: If required columns are missing from DataFrame.

    Example:
        >>> from hba1cE.data import load_cleaned_data
        >>> df = load_cleaned_data()
        >>> X, feature_names = create_features(df)
        >>> print(f"Created {len(feature_names)} features for {X.shape[0]} samples")
    """
    required_columns = ["fpg_mgdl", "tg_mgdl", "hdl_mgdl", "age_years", "hgb_gdl", "mcv_fl"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Base features (raw biomarkers)
    features = {
        "fpg_mgdl": df["fpg_mgdl"].values,
        "tg_mgdl": df["tg_mgdl"].values,
        "hdl_mgdl": df["hdl_mgdl"].values,
        "age_years": df["age_years"].values,
        "hgb_gdl": df["hgb_gdl"].values,
        "mcv_fl": df["mcv_fl"].values,
    }

    # Ratio features
    features["tg_hdl_ratio"] = df["tg_mgdl"].values / df["hdl_mgdl"].values
    features["fpg_age_interaction"] = df["fpg_mgdl"].values * df["age_years"].values

    # Mechanistic estimator predictions as features
    fpg_arr = df["fpg_mgdl"].values
    hgb_arr = df["hgb_gdl"].values

    # ADAG estimator
    features["adag_estimate"] = calc_hba1c_adag(fpg_arr)

    # Kinetic estimator
    features["kinetic_estimate"] = calc_hba1c_kinetic(fpg_arr, hgb_gdl=hgb_arr)

    # Regression estimator (using default coefficients)
    features["regression_estimate"] = calc_hba1c_regression(
        fpg_mgdl=fpg_arr,
        age_years=df["age_years"].values,
        tg_mgdl=df["tg_mgdl"].values,
        hdl_mgdl=df["hdl_mgdl"].values,
        hgb_gdl=hgb_arr,
    )

    # Build feature matrix
    feature_names = list(features.keys())
    X = np.column_stack([features[name] for name in feature_names])

    return X, feature_names


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets with stratification by HbA1c ranges.

    Stratifies by clinical HbA1c ranges to ensure balanced representation
    of normal, prediabetes, and various diabetes severity levels:
    - <5.7% (normal)
    - 5.7-6.4% (prediabetes)
    - 6.5-8% (controlled diabetes)
    - 8-10% (uncontrolled diabetes)
    - >10% (severely uncontrolled)

    Args:
        df: pandas DataFrame with required columns for create_features()
            and 'hba1c_percent' target column.
        test_size: Proportion of data to use for test set (default 0.3).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where:
        - X_train: Training feature matrix
        - X_test: Test feature matrix  
        - y_train: Training target values (HbA1c %)
        - y_test: Test target values (HbA1c %)

    Raises:
        ValueError: If 'hba1c_percent' column is missing from DataFrame.
        ValueError: If any stratum has fewer than 2 samples.

    Example:
        >>> from hba1cE.data import load_cleaned_data
        >>> df = load_cleaned_data()
        >>> X_train, X_test, y_train, y_test = stratified_split(df)
        >>> print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    """
    if "hba1c_percent" not in df.columns:
        raise ValueError("Missing required column: 'hba1c_percent'")

    # Create features
    X, _ = create_features(df)
    y = df["hba1c_percent"].values

    # Create stratification bins based on HbA1c clinical ranges
    # <5.7%, 5.7-6.4%, 6.5-8%, 8-10%, >10%
    bins = [0, 5.7, 6.5, 8.0, 10.0, float("inf")]
    labels = [0, 1, 2, 3, 4]  # Numeric labels for strata
    strata = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

    # Check that all strata have at least 2 samples for valid split
    strata_counts = pd.Series(strata).value_counts()
    for label in labels:
        if strata_counts.get(label, 0) < 2:
            raise ValueError(
                f"Stratum {label} has fewer than 2 samples. "
                "Need at least 2 samples per stratum for stratified split."
            )

    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strata
    )

    return X_train, X_test, y_train, y_test
