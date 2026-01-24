"""
LDL-C ML model training module.

This module contains functions for feature engineering and model training
for the hybrid LDL-C estimation model.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

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
