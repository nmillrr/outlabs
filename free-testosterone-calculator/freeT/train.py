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
