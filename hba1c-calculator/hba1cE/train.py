"""
ML training utilities for HbA1c estimation.

This module contains functions for feature engineering and model training.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
import joblib

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


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> Ridge:
    """
    Train a Ridge regression model for HbA1c prediction.

    Ridge regression adds L2 regularization to linear regression,
    which helps prevent overfitting and handles collinear features.

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Training target values of shape (n_samples,).
        alpha: Regularization strength (default 1.0). Larger values
            specify stronger regularization.

    Returns:
        Fitted Ridge regression model.

    Raises:
        ValueError: If X_train and y_train have incompatible shapes.

    Example:
        >>> from hba1cE.train import stratified_split, train_ridge
        >>> X_train, X_test, y_train, y_test = stratified_split(df)
        >>> model = train_ridge(X_train, y_train, alpha=1.0)
        >>> y_pred = model.predict(X_test)
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train has {X_train.shape[0]} samples but y_train has {y_train.shape[0]} samples"
        )

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    return model


def save_model(model: object, filepath: str) -> None:
    """
    Save a trained model to disk using joblib.

    Uses joblib for efficient serialization of scikit-learn models,
    which handles numpy arrays better than pickle.

    Args:
        model: Trained scikit-learn model object.
        filepath: Path where the model should be saved.

    Raises:
        OSError: If the file cannot be written.

    Example:
        >>> model = train_ridge(X_train, y_train)
        >>> save_model(model, "models/ridge_model.joblib")
    """
    from pathlib import Path

    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
) -> RandomForestRegressor:
    """
    Train a Random Forest regression model for HbA1c prediction.

    Random Forest is an ensemble method that fits multiple decision trees
    and averages their predictions, which helps capture nonlinear patterns
    and reduce overfitting.

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Training target values of shape (n_samples,).
        n_estimators: Number of trees in the forest (default 200).

    Returns:
        Fitted RandomForestRegressor model.

    Raises:
        ValueError: If X_train and y_train have incompatible shapes.

    Example:
        >>> from hba1cE.train import stratified_split, train_random_forest
        >>> X_train, X_test, y_train, y_test = stratified_split(df)
        >>> model = train_random_forest(X_train, y_train, n_estimators=200)
        >>> y_pred = model.predict(X_test)
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train has {X_train.shape[0]} samples but y_train has {y_train.shape[0]} samples"
        )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 20,
) -> "LGBMRegressor":
    """
    Train a LightGBM model for HbA1c prediction with early stopping.

    LightGBM is a gradient boosting framework that uses tree-based learning.
    It is efficient and often achieves best performance on tabular data.
    Early stopping prevents overfitting by halting training when validation
    performance stops improving.

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Training target values of shape (n_samples,).
        X_val: Validation feature matrix for early stopping.
        y_val: Validation target values for early stopping.
        n_estimators: Maximum number of boosting iterations (default 1000).
        early_stopping_rounds: Stop training if validation score doesn't
            improve for this many rounds (default 20).

    Returns:
        Fitted LGBMRegressor model.

    Raises:
        ValueError: If X_train and y_train have incompatible shapes.
        ValueError: If X_val and y_val have incompatible shapes.

    Example:
        >>> from hba1cE.train import stratified_split, train_lightgbm
        >>> X_train, X_test, y_train, y_test = stratified_split(df, test_size=0.3)
        >>> # Use part of test as validation for early stopping
        >>> X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
        >>> model = train_lightgbm(X_train, y_train, X_val, y_val)
        >>> y_pred = model.predict(X_test)
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train has {X_train.shape[0]} samples but y_train has {y_train.shape[0]} samples"
        )
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError(
            f"X_val has {X_val.shape[0]} samples but y_val has {y_val.shape[0]} samples"
        )

    from lightgbm import LGBMRegressor
    import lightgbm

    model = LGBMRegressor(
        n_estimators=n_estimators,
        random_state=42,
        verbosity=-1,  # Suppress warnings
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lightgbm.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        ],
    )

    return model


def cross_validate_model(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
) -> dict:
    """
    Perform k-fold cross-validation to evaluate model performance.

    Uses k-fold cross-validation to compute RMSE and MAE metrics,
    providing both mean and standard deviation across folds for
    robust performance estimation.

    Args:
        model: A scikit-learn compatible estimator (must have fit/predict methods).
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        n_splits: Number of cross-validation folds (default 10).

    Returns:
        Dictionary with keys:
        - RMSE_mean: Mean RMSE across all folds
        - RMSE_std: Standard deviation of RMSE across folds
        - MAE_mean: Mean MAE across all folds
        - MAE_std: Standard deviation of MAE across folds

    Raises:
        ValueError: If X and y have incompatible shapes.
        ValueError: If n_splits < 2.

    Example:
        >>> from hba1cE.train import train_ridge, cross_validate_model
        >>> model = Ridge(alpha=1.0)
        >>> results = cross_validate_model(model, X, y, n_splits=10)
        >>> print(f"RMSE: {results['RMSE_mean']:.3f} ± {results['RMSE_std']:.3f}")
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} samples but y has {y.shape[0]} samples"
        )
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmse_scores = []
    mae_scores = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone model to get a fresh instance for each fold
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        y_pred = model_clone.predict(X_val)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        rmse_scores.append(rmse)

        # Calculate MAE
        mae = np.mean(np.abs(y_val - y_pred))
        mae_scores.append(mae)

    return {
        "RMSE_mean": float(np.mean(rmse_scores)),
        "RMSE_std": float(np.std(rmse_scores)),
        "MAE_mean": float(np.mean(mae_scores)),
        "MAE_std": float(np.std(mae_scores)),
    }

