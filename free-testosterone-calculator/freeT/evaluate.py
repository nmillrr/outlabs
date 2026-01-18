"""
Evaluation metrics for free testosterone estimation models.

This module provides statistical functions for model validation, including
Bland-Altman analysis for agreement assessment.
"""

from typing import Dict
import numpy as np


def bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate Bland-Altman statistics for agreement analysis.
    
    The Bland-Altman method is used to assess the agreement between two
    quantitative measurements. It calculates the mean bias (average difference)
    and limits of agreement (±1.96 standard deviations of the differences).
    
    Parameters
    ----------
    y_true : array-like
        True/reference values (e.g., measured free testosterone)
    y_pred : array-like
        Predicted values (e.g., estimated free testosterone)
    
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
        If inputs have different lengths or contain fewer than 2 observations
    
    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    >>> stats = bland_altman_stats(y_true, y_pred)
    >>> print(f"Mean bias: {stats['mean_bias']:.3f}")
    Mean bias: 0.020
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Input validation
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} and {y_pred.shape}"
        )
    
    if len(y_true) < 2:
        raise ValueError(
            f"At least 2 observations required for Bland-Altman analysis. "
            f"Got {len(y_true)}"
        )
    
    # Check for NaN values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values")
    
    # Calculate differences (predicted - true)
    differences = y_pred - y_true
    
    # Calculate statistics
    mean_bias = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))  # Use sample std (ddof=1)
    
    # Limits of agreement (±1.96 SD)
    loa_lower = mean_bias - 1.96 * std_diff
    loa_upper = mean_bias + 1.96 * std_diff
    
    return {
        'mean_bias': mean_bias,
        'std_diff': std_diff,
        'loa_lower': float(loa_lower),
        'loa_upper': float(loa_upper),
    }
