"""
Evaluation metrics for LDL-C estimation model validation.

This module provides statistical methods for agreement analysis and model evaluation,
including Bland-Altman analysis, Lin's Concordance Correlation Coefficient, and
stratified performance metrics.
"""

import numpy as np
from typing import Dict, Union


def bland_altman_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Bland-Altman statistics for agreement analysis.
    
    Bland-Altman analysis measures the agreement between two quantitative methods.
    It calculates the mean difference (bias) and the limits of agreement (LOA),
    which are defined as mean ± 1.96 * standard deviation of differences.
    
    Parameters
    ----------
    y_true : np.ndarray
        True/reference values (e.g., direct LDL measurements from beta-quantification)
    y_pred : np.ndarray
        Predicted/estimated values (e.g., LDL from equation or ML model)
        
    Returns
    -------
    dict
        Dictionary containing:
        - mean_bias: Mean difference (y_pred - y_true)
        - std_diff: Standard deviation of differences
        - loa_lower: Lower limit of agreement (mean_bias - 1.96 * std_diff)
        - loa_upper: Upper limit of agreement (mean_bias + 1.96 * std_diff)
        
    Raises
    ------
    ValueError
        If arrays are empty, have different lengths, or contain only NaN values
        
    Examples
    --------
    >>> y_true = np.array([100, 120, 110, 130, 105])
    >>> y_pred = np.array([102, 118, 112, 128, 107])
    >>> stats = bland_altman_stats(y_true, y_pred)
    >>> print(f"Bias: {stats['mean_bias']:.2f} mg/dL")
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Validate inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Arrays must have same length. Got y_true: {len(y_true)}, y_pred: {len(y_pred)}"
        )
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        raise ValueError("No valid (non-NaN) pairs found in input arrays")
    
    # Calculate differences (predicted - true)
    differences = y_pred_clean - y_true_clean
    
    # Calculate Bland-Altman statistics
    mean_bias = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))  # Use sample std (ddof=1)
    
    # Limits of agreement (95% CI)
    loa_lower = mean_bias - 1.96 * std_diff
    loa_upper = mean_bias + 1.96 * std_diff
    
    return {
        'mean_bias': mean_bias,
        'std_diff': std_diff,
        'loa_lower': loa_lower,
        'loa_upper': loa_upper
    }
