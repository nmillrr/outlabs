"""
ML training utilities for HbA1c estimation.

This module contains functions for feature engineering and model training.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

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
