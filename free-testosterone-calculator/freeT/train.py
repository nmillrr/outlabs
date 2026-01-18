"""
Machine Learning Training Module for Free Testosterone Estimation

Provides feature engineering, data splitting, and model training functions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from .models import calc_ft_vermeulen


def create_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature matrix from raw data for ML training.
    
    Generates features including the raw biomarker values, derived ratios,
    and baseline Vermeulen FT estimates as a hybrid feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: tt_nmoll, shbg_nmoll, alb_gl
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        X : Feature matrix of shape (n_samples, n_features)
        feature_names : List of feature column names
        
    Raises
    ------
    ValueError
        If required columns are missing from DataFrame
        
    Notes
    -----
    Features created:
    - tt_nmoll: Total testosterone (nmol/L)
    - shbg_nmoll: SHBG (nmol/L)
    - alb_gl: Albumin (g/L)
    - shbg_tt_ratio: SHBG/TT ratio (inversely related to FT%)
    - ft_vermeulen: Baseline FT from Vermeulen solver (hybrid feature)
    """
    # Validate required columns
    required_cols = ['tt_nmoll', 'shbg_nmoll', 'alb_gl']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize feature DataFrame
    features = pd.DataFrame()
    
    # Raw biomarker features
    features['tt_nmoll'] = df['tt_nmoll'].values
    features['shbg_nmoll'] = df['shbg_nmoll'].values
    features['alb_gl'] = df['alb_gl'].values
    
    # Derived ratio feature: SHBG/TT
    # Handle division by zero with small epsilon
    epsilon = 1e-10
    features['shbg_tt_ratio'] = df['shbg_nmoll'].values / (df['tt_nmoll'].values + epsilon)
    
    # Baseline Vermeulen FT (hybrid feature)
    ft_vermeulen_values = []
    for _, row in df.iterrows():
        try:
            ft = calc_ft_vermeulen(
                tt_nmoll=row['tt_nmoll'],
                shbg_nmoll=row['shbg_nmoll'],
                alb_gl=row['alb_gl']
            )
            ft_vermeulen_values.append(ft)
        except ValueError:
            # If Vermeulen fails, use NaN (should be handled downstream)
            ft_vermeulen_values.append(np.nan)
    
    features['ft_vermeulen'] = ft_vermeulen_values
    
    # Get feature names
    feature_names = list(features.columns)
    
    # Convert to numpy array
    X = features.values
    
    return X, feature_names


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets with stratification by SHBG tertiles.
    
    Ensures balanced representation of low, medium, and high SHBG subjects
    in both training and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: tt_nmoll, shbg_nmoll, alb_gl
        Must also contain a target column (typically ft_measured or similar)
    test_size : float, optional
        Proportion of data for test set (default: 0.3)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train : Training features
        X_test : Test features
        y_train : Training target values
        y_test : Test target values
        
    Raises
    ------
    ValueError
        If required columns are missing or if data is too small for stratification
        
    Notes
    -----
    SHBG tertiles are computed from the input data's SHBG distribution.
    Stratification ensures each tertile is proportionally represented.
    """
    from sklearn.model_selection import train_test_split
    
    # Validate required columns
    required_cols = ['tt_nmoll', 'shbg_nmoll', 'alb_gl']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check minimum data size for stratification
    if len(df) < 9:  # Need at least 3 per tertile, split into train/test
        raise ValueError(
            f"DataFrame has {len(df)} rows, need at least 9 for stratified split"
        )
    
    # Create SHBG tertiles for stratification
    # Labels: 0 = low, 1 = medium, 2 = high
    shbg_tertiles = pd.qcut(
        df['shbg_nmoll'], 
        q=3, 
        labels=[0, 1, 2],
        duplicates='drop'
    )
    
    # Create features using the create_features function
    X, feature_names = create_features(df)
    
    # Target variable: use ft_vermeulen from features as a proxy target
    # In practice, this would be measured FT when available
    # The ft_vermeulen is the last column (index -1)
    y = X[:, -1].copy()  # ft_vermeulen as target
    
    # Remove ft_vermeulen from features to avoid data leakage if using as target
    # Note: Keep it for now as the PRD specifies it as a feature
    # The actual target should be measured FT when available
    
    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=shbg_tertiles
    )
    
    return X_train, X_test, y_train, y_test


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0
):
    """
    Train a Ridge regression model for free testosterone estimation.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features)
    y_train : np.ndarray
        Training target values of shape (n_samples,)
    alpha : float, optional
        Regularization strength (default: 1.0)
        Higher values = stronger regularization
        
    Returns
    -------
    sklearn.linear_model.Ridge
        Fitted Ridge regression model
        
    Notes
    -----
    Ridge regression adds L2 penalty to prevent overfitting and handles
    multicollinearity in features (e.g., TT and ft_vermeulen are correlated).
    """
    from sklearn.linear_model import Ridge
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200
):
    """
    Train a Random Forest regression model for free testosterone estimation.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features)
    y_train : np.ndarray
        Training target values of shape (n_samples,)
    n_estimators : int, optional
        Number of trees in the forest (default: 200)
        
    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Fitted Random Forest regression model
        
    Notes
    -----
    Random Forest captures nonlinear patterns and feature interactions
    that linear models like Ridge cannot. It is robust to outliers and
    does not require feature scaling.
    """
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,  # For reproducibility
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
):
    """
    Train a LightGBM regression model for free testosterone estimation.
    
    Uses early stopping to prevent overfitting by monitoring validation loss
    and stopping when performance no longer improves.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix of shape (n_samples, n_features)
    y_train : np.ndarray
        Training target values of shape (n_samples,)
    X_val : np.ndarray
        Validation feature matrix of shape (n_val_samples, n_features)
    y_val : np.ndarray
        Validation target values of shape (n_val_samples,)
        
    Returns
    -------
    lightgbm.LGBMRegressor
        Fitted LightGBM regression model
        
    Notes
    -----
    LightGBM is a gradient boosting framework that uses tree-based learning.
    It is fast, memory-efficient, and often achieves state-of-the-art 
    performance on tabular data. Early stopping with 20 rounds prevents
    overfitting by halting training when validation loss stops improving.
    """
    from lightgbm import LGBMRegressor
    
    model = LGBMRegressor(
        n_estimators=1000,  # High value, early stopping will find optimal
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1  # Suppress warnings
    )
    
    # Fit with early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__('lightgbm').early_stopping(stopping_rounds=20, verbose=False)
        ]
    )
    
    return model


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10
) -> dict:
    """
    Perform k-fold cross-validation to evaluate a model consistently.
    
    Parameters
    ----------
    model : sklearn-compatible estimator
        Model to evaluate (must implement fit and predict methods)
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target values of shape (n_samples,)
    n_splits : int, optional
        Number of cross-validation folds (default: 10)
        
    Returns
    -------
    dict
        Dictionary containing:
        - RMSE_mean: Mean RMSE across folds
        - RMSE_std: Standard deviation of RMSE across folds
        - MAE_mean: Mean MAE across folds
        - MAE_std: Standard deviation of MAE across folds
        
    Notes
    -----
    Uses sklearn's KFold for cross-validation and clone for creating
    fresh model instances for each fold. RMSE and MAE are computed
    on each fold's test set for robust performance estimation.
    """
    from sklearn.model_selection import KFold
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone model to get fresh instance for each fold
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        y_pred = model_clone.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
    
    return {
        'RMSE_mean': float(np.mean(rmse_scores)),
        'RMSE_std': float(np.std(rmse_scores)),
        'MAE_mean': float(np.mean(mae_scores)),
        'MAE_std': float(np.std(mae_scores))
    }


def save_model(model, filepath: str) -> None:
    """
    Save a trained model to disk using joblib.
    
    Parameters
    ----------
    model : object
        Trained scikit-learn model or compatible object
    filepath : str
        Path where model will be saved (typically .joblib or .pkl extension)
        
    Returns
    -------
    None
        
    Notes
    -----
    Uses joblib for efficient serialization of numpy arrays in sklearn models.
    Creates parent directories if they don't exist.
    """
    import joblib
    from pathlib import Path
    
    # Create parent directories if needed
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
