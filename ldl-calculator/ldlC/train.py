"""
LDL-C ML model training module.

This module contains functions for feature engineering and model training
for the hybrid LDL-C estimation model.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, List, Any

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from ldlC.models import (
    calc_ldl_friedewald,
    calc_ldl_martin_hopkins,
    calc_ldl_martin_hopkins_extended,
    calc_ldl_sampson
)


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create feature matrix for ML model training.
    
    This function generates features from lipid panel data including:
    - Raw lipid values: tc_mgdl, hdl_mgdl, tg_mgdl, non_hdl_mgdl
    - Ratio features: tg_hdl_ratio, tc_hdl_ratio
    - Baseline equation predictions: friedewald, martin_hopkins, martin_hopkins_extended, sampson
    
    Args:
        df: DataFrame containing at minimum columns:
            - tc_mgdl: Total cholesterol in mg/dL
            - hdl_mgdl: HDL cholesterol in mg/dL
            - tg_mgdl: Triglycerides in mg/dL
            May optionally contain:
            - non_hdl_mgdl: Non-HDL cholesterol (calculated if missing)
    
    Returns:
        Tuple of (X, feature_names) where:
            - X: DataFrame with all features
            - feature_names: List of column names in the feature matrix
    
    Raises:
        ValueError: If required columns are missing from input DataFrame.
    """
    required_cols = ['tc_mgdl', 'hdl_mgdl', 'tg_mgdl']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create working copy
    X = pd.DataFrame()
    
    # Raw lipid features
    X['tc_mgdl'] = df['tc_mgdl'].astype(float)
    X['hdl_mgdl'] = df['hdl_mgdl'].astype(float)
    X['tg_mgdl'] = df['tg_mgdl'].astype(float)
    
    # Non-HDL-C (calculate if not present)
    if 'non_hdl_mgdl' in df.columns:
        X['non_hdl_mgdl'] = df['non_hdl_mgdl'].astype(float)
    else:
        X['non_hdl_mgdl'] = X['tc_mgdl'] - X['hdl_mgdl']
    
    # Ratio features
    # Use np.where to handle division by zero gracefully
    X['tg_hdl_ratio'] = np.where(
        X['hdl_mgdl'] > 0,
        X['tg_mgdl'] / X['hdl_mgdl'],
        np.nan
    )
    X['tc_hdl_ratio'] = np.where(
        X['hdl_mgdl'] > 0,
        X['tc_mgdl'] / X['hdl_mgdl'],
        np.nan
    )
    
    # Baseline equation predictions
    # Apply each equation to get predicted LDL values
    def safe_friedewald(row):
        """Apply Friedewald with NaN for invalid inputs."""
        try:
            return calc_ldl_friedewald(row['tc_mgdl'], row['hdl_mgdl'], row['tg_mgdl'])
        except (ValueError, TypeError):
            return np.nan
    
    def safe_martin_hopkins(row):
        """Apply Martin-Hopkins with NaN for invalid inputs."""
        try:
            return calc_ldl_martin_hopkins(row['tc_mgdl'], row['hdl_mgdl'], row['tg_mgdl'])
        except (ValueError, TypeError):
            return np.nan
    
    def safe_martin_hopkins_extended(row):
        """Apply Extended Martin-Hopkins with NaN for invalid inputs."""
        try:
            return calc_ldl_martin_hopkins_extended(row['tc_mgdl'], row['hdl_mgdl'], row['tg_mgdl'])
        except (ValueError, TypeError):
            return np.nan
    
    def safe_sampson(row):
        """Apply Sampson with NaN for invalid inputs."""
        try:
            return calc_ldl_sampson(row['tc_mgdl'], row['hdl_mgdl'], row['tg_mgdl'])
        except (ValueError, TypeError):
            return np.nan
    
    # Apply equation predictions
    X['ldl_friedewald'] = df.apply(safe_friedewald, axis=1)
    X['ldl_martin_hopkins'] = df.apply(safe_martin_hopkins, axis=1)
    X['ldl_martin_hopkins_extended'] = df.apply(safe_martin_hopkins_extended, axis=1)
    X['ldl_sampson'] = df.apply(safe_sampson, axis=1)
    
    feature_names = list(X.columns)
    
    return X, feature_names


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets, stratified by TG quartiles.
    
    This ensures that each TG subgroup is represented proportionally in both
    train and test sets, which is critical for evaluating model performance
    across the full TG range.
    
    TG strata:
    - < 100 mg/dL (normal)
    - 100-150 mg/dL (borderline)
    - 150-200 mg/dL (borderline high)
    - 200-400 mg/dL (high)
    - > 400 mg/dL (very high)
    
    Args:
        df: DataFrame containing lipid panel data with at minimum:
            - tc_mgdl, hdl_mgdl, tg_mgdl columns (for features)
            - ldl_direct_mgdl column (target variable)
        test_size: Fraction of data to use for testing (default 0.3).
        random_state: Random seed for reproducibility (default 42).
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where:
            - X_train: Training feature DataFrame
            - X_test: Test feature DataFrame  
            - y_train: Training target Series
            - y_test: Test target Series
    
    Raises:
        ValueError: If required columns are missing or data is insufficient.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Validate required columns
    required_cols = ['tc_mgdl', 'hdl_mgdl', 'tg_mgdl', 'ldl_direct_mgdl']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create TG strata bins for stratification
    # Bins: < 100, 100-150, 150-200, 200-400, > 400 mg/dL
    tg_bins = [0, 100, 150, 200, 400, float('inf')]
    tg_labels = ['<100', '100-150', '150-200', '200-400', '>400']
    
    df_clean = df.dropna(subset=required_cols).copy()
    
    if len(df_clean) < 10:
        raise ValueError("Insufficient data for stratified split (need at least 10 samples)")
    
    # Create TG stratum labels
    tg_strata = pd.cut(
        df_clean['tg_mgdl'],
        bins=tg_bins,
        labels=tg_labels,
        include_lowest=True
    )
    
    # Create features using create_features function
    X, feature_names = create_features(df_clean)
    y = df_clean['ldl_direct_mgdl'].reset_index(drop=True)
    
    # Reset index for alignment
    X = X.reset_index(drop=True)
    tg_strata = tg_strata.reset_index(drop=True)
    
    # Use StratifiedShuffleSplit for stratified splitting
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    # Get train/test indices
    train_idx, test_idx = next(splitter.split(X, tg_strata))
    
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0
) -> Ridge:
    """
    Train a Ridge regression model for LDL-C prediction.
    
    Ridge regression is a simple linear model with L2 regularization,
    serving as a baseline for comparison with more complex models.
    
    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series (LDL-direct values).
        alpha: Regularization strength (default 1.0). Higher values 
               increase regularization.
    
    Returns:
        Fitted Ridge regression model.
    
    Raises:
        ValueError: If input data shapes are incompatible.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have same length. "
            f"Got {len(X_train)} and {len(y_train)}"
        )
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200
) -> RandomForestRegressor:
    """
    Train a Random Forest regression model for LDL-C prediction.
    
    Random Forest is an ensemble learning method that builds multiple
    decision trees and averages their predictions. It captures nonlinear
    relationships and interactions between features.
    
    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series (LDL-direct values).
        n_estimators: Number of trees in the forest (default 200).
                      Higher values improve performance but increase
                      training time.
    
    Returns:
        Fitted RandomForestRegressor model.
    
    Raises:
        ValueError: If input data shapes are incompatible.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have same length. "
            f"Got {len(X_train)} and {len(y_train)}"
        )
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,  # For reproducibility
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 20
) -> LGBMRegressor:
    """
    Train a LightGBM regression model for LDL-C prediction.
    
    LightGBM is a gradient boosting framework that uses tree-based learning
    algorithms. It is designed to be distributed and efficient with faster
    training speed and higher efficiency compared to other boosting methods.
    
    Uses early stopping to prevent overfitting by monitoring validation loss
    and stopping training when no improvement is seen for a specified number
    of rounds.
    
    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series (LDL-direct values).
        X_val: Validation feature DataFrame for early stopping.
        y_val: Validation target Series for early stopping.
        n_estimators: Maximum number of boosting iterations (default 1000).
                      Early stopping typically terminates before this.
        early_stopping_rounds: Number of rounds without improvement before
                               stopping (default 20).
    
    Returns:
        Fitted LGBMRegressor model.
    
    Raises:
        ValueError: If input data shapes are incompatible.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have same length. "
            f"Got {len(X_train)} and {len(y_train)}"
        )
    
    if len(X_val) != len(y_val):
        raise ValueError(
            f"X_val and y_val must have same length. "
            f"Got {len(X_val)} and {len(y_val)}"
        )
    
    model = LGBMRegressor(
        n_estimators=n_estimators,
        random_state=42,  # For reproducibility
        n_jobs=-1,  # Use all available cores
        verbosity=-1  # Suppress warnings during training
    )
    
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            # Early stopping callback
            __import__('lightgbm').early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        ]
    )
    
    return model


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk using joblib.
    
    Creates parent directories if they don't exist.
    
    Args:
        model: Trained model object to save (any scikit-learn compatible model).
        filepath: Path to save the model (typically with .joblib extension).
    
    Returns:
        None
    
    Raises:
        IOError: If the file cannot be written.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
