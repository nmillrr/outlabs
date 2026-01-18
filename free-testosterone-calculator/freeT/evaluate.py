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


def lins_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Lin's Concordance Correlation Coefficient (CCC).
    
    Lin's CCC measures the agreement between two variables, combining
    precision (Pearson correlation) and accuracy (closeness to the 45° line).
    It is commonly used in method comparison studies.
    
    CCC = (2 * r * sd_true * sd_pred) / (sd_true² + sd_pred² + (mean_true - mean_pred)²)
    
    where r is the Pearson correlation coefficient.
    
    Parameters
    ----------
    y_true : array-like
        True/reference values
    y_pred : array-like
        Predicted values
    
    Returns
    -------
    float
        Lin's CCC value between -1 and 1, where:
        - 1 indicates perfect agreement
        - 0 indicates no agreement
        - -1 indicates perfect disagreement
    
    Raises
    ------
    ValueError
        If inputs have different lengths, contain fewer than 2 observations,
        or contain NaN values
    
    References
    ----------
    Lin, L. I. (1989). A concordance correlation coefficient to evaluate
    reproducibility. Biometrics, 45(1), 255-268.
    
    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect agreement
    >>> lins_ccc(y_true, y_pred)
    1.0
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
            f"At least 2 observations required for Lin's CCC. "
            f"Got {len(y_true)}"
        )
    
    # Check for NaN values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values")
    
    # Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Calculate variances (using population variance, n in denominator)
    var_true = np.var(y_true, ddof=0)
    var_pred = np.var(y_pred, ddof=0)
    
    # Calculate covariance (using population covariance)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # Calculate CCC
    # CCC = 2 * covariance / (var_true + var_pred + (mean_true - mean_pred)^2)
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    # Handle edge case where denominator is zero (constant arrays)
    if denominator == 0:
        # If both arrays are constant and equal, perfect agreement
        if mean_true == mean_pred:
            return 1.0
        else:
            return 0.0
    
    ccc = numerator / denominator
    
    return float(ccc)
