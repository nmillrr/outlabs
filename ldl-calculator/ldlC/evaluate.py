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


def lins_ccc(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate Lin's Concordance Correlation Coefficient (CCC).
    
    Lin's CCC measures the agreement between two quantitative measurements,
    accounting for both precision (how close the measurements are to each other)
    and accuracy (how far each measurement is from the true value).
    
    Formula:
        CCC = (2 * r * s_x * s_y) / (s_x^2 + s_y^2 + (mean_x - mean_y)^2)
    
    where r is Pearson's correlation, s_x and s_y are standard deviations,
    and mean_x, mean_y are the means of the two measurements.
    
    Parameters
    ----------
    y_true : np.ndarray
        True/reference values (e.g., direct LDL measurements from beta-quantification)
    y_pred : np.ndarray
        Predicted/estimated values (e.g., LDL from equation or ML model)
        
    Returns
    -------
    float
        Lin's CCC value between -1 and 1, where 1 indicates perfect agreement
        
    Raises
    ------
    ValueError
        If arrays are empty, have different lengths, or contain only NaN values
        
    Examples
    --------
    >>> y_true = np.array([100, 120, 110, 130, 105])
    >>> y_pred = np.array([100, 120, 110, 130, 105])  # Perfect agreement
    >>> ccc = lins_ccc(y_true, y_pred)
    >>> print(f"CCC: {ccc:.4f}")  # Should print 1.0000
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
    
    if len(y_true_clean) < 2:
        raise ValueError("At least 2 valid data points are required to calculate CCC")
    
    # Calculate means
    mean_true = np.mean(y_true_clean)
    mean_pred = np.mean(y_pred_clean)
    
    # Calculate variances (sample variance with n-1 denominator)
    var_true = np.var(y_true_clean, ddof=1)
    var_pred = np.var(y_pred_clean, ddof=1)
    
    # Calculate covariance (sample covariance with n-1 denominator)
    covariance = np.cov(y_true_clean, y_pred_clean, ddof=1)[0, 1]
    
    # Calculate CCC using the formula:
    # CCC = 2 * cov(x,y) / (var_x + var_y + (mean_x - mean_y)^2)
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    if denominator == 0:
        # Both arrays are constant and equal - perfect agreement
        return 1.0
    
    ccc = numerator / denominator
    
    return float(ccc)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model"
) -> Dict[str, Union[float, str, Dict[str, float]]]:
    """
    Compute comprehensive evaluation metrics for an LDL-C estimation model.
    
    This function calculates all standard metrics for model validation in a single call,
    including error metrics (RMSE, MAE), agreement metrics (Lin's CCC, Bland-Altman),
    and correlation (Pearson's r).
    
    Parameters
    ----------
    y_true : np.ndarray
        True/reference values (e.g., direct LDL measurements from beta-quantification)
    y_pred : np.ndarray
        Predicted/estimated values (e.g., LDL from equation or ML model)
    model_name : str, optional
        Name of the model being evaluated (default: "model")
        
    Returns
    -------
    dict
        Dictionary containing:
        - model_name: Name of the evaluated model
        - n_samples: Number of valid samples used in evaluation
        - rmse: Root Mean Square Error
        - mae: Mean Absolute Error
        - bias: Mean difference (y_pred - y_true)
        - r_pearson: Pearson correlation coefficient
        - lin_ccc: Lin's Concordance Correlation Coefficient
        - ba_stats: Dict with Bland-Altman statistics (mean_bias, std_diff, loa_lower, loa_upper)
        
    Raises
    ------
    ValueError
        If arrays are empty, have different lengths, or contain only NaN values
        
    Examples
    --------
    >>> y_true = np.array([100, 120, 110, 130, 105])
    >>> y_pred = np.array([102, 118, 112, 128, 107])
    >>> metrics = evaluate_model(y_true, y_pred, model_name="Friedewald")
    >>> print(f"RMSE: {metrics['rmse']:.2f}, CCC: {metrics['lin_ccc']:.4f}")
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
    
    n_samples = len(y_true_clean)
    
    if n_samples == 0:
        raise ValueError("No valid (non-NaN) pairs found in input arrays")
    
    if n_samples < 2:
        raise ValueError("At least 2 valid data points are required for evaluation")
    
    # Calculate error metrics
    errors = y_pred_clean - y_true_clean
    squared_errors = errors ** 2
    
    rmse = float(np.sqrt(np.mean(squared_errors)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    
    # Calculate Pearson correlation coefficient
    r_pearson = float(np.corrcoef(y_true_clean, y_pred_clean)[0, 1])
    
    # Calculate Lin's CCC (reuse existing function)
    lin_ccc_value = lins_ccc(y_true_clean, y_pred_clean)
    
    # Calculate Bland-Altman statistics (reuse existing function)
    ba_stats = bland_altman_stats(y_true_clean, y_pred_clean)
    
    return {
        'model_name': model_name,
        'n_samples': n_samples,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'r_pearson': r_pearson,
        'lin_ccc': lin_ccc_value,
        'ba_stats': ba_stats
    }


def evaluate_by_tg_strata(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tg_values: np.ndarray
) -> Dict[str, Dict[str, Union[float, str, Dict[str, float]]]]:
    """
    Evaluate model performance stratified by triglyceride (TG) levels.
    
    Stratifies evaluation by clinically relevant TG thresholds to assess
    model performance across different patient populations. This is critical
    for validating LDL-C estimation methods since many equations perform
    differently at high TG levels.
    
    TG Strata:
        - low_tg: TG < 150 mg/dL (normal)
        - medium_tg: 150 ≤ TG < 400 mg/dL (borderline to high)
        - high_tg: 400 ≤ TG ≤ 800 mg/dL (very high)
    
    Parameters
    ----------
    y_true : np.ndarray
        True/reference values (e.g., direct LDL measurements from beta-quantification)
    y_pred : np.ndarray
        Predicted/estimated values (e.g., LDL from equation or ML model)
    tg_values : np.ndarray
        Triglyceride values in mg/dL for each sample
        
    Returns
    -------
    dict
        Dictionary with keys 'low_tg', 'medium_tg', 'high_tg', 'overall'.
        Each contains evaluation metrics from evaluate_model(), or None if
        insufficient samples in that stratum.
        
    Raises
    ------
    ValueError
        If arrays are empty or have different lengths
        
    Examples
    --------
    >>> y_true = np.array([100, 120, 110, 130, 105])
    >>> y_pred = np.array([102, 118, 112, 128, 107])
    >>> tg = np.array([100, 200, 350, 500, 80])
    >>> results = evaluate_by_tg_strata(y_true, y_pred, tg)
    >>> print(f"Low TG RMSE: {results['low_tg']['rmse']:.2f}")
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    tg_values = np.asarray(tg_values, dtype=float)
    
    # Validate inputs
    if len(y_true) == 0 or len(y_pred) == 0 or len(tg_values) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(y_true) != len(y_pred) or len(y_true) != len(tg_values):
        raise ValueError(
            f"All arrays must have same length. Got y_true: {len(y_true)}, "
            f"y_pred: {len(y_pred)}, tg_values: {len(tg_values)}"
        )
    
    # Remove rows with any NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(tg_values))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    tg_clean = tg_values[valid_mask]
    
    if len(y_true_clean) == 0:
        raise ValueError("No valid (non-NaN) data points found in input arrays")
    
    # Define TG strata thresholds (mg/dL)
    strata = {
        'low_tg': (0, 150),       # Normal TG
        'medium_tg': (150, 400),  # Borderline to high TG
        'high_tg': (400, 800)     # Very high TG
    }
    
    results = {}
    
    # Evaluate each stratum
    for stratum_name, (lower, upper) in strata.items():
        # Create mask for this stratum
        if stratum_name == 'low_tg':
            mask = tg_clean < upper
        elif stratum_name == 'high_tg':
            mask = (tg_clean >= lower) & (tg_clean <= upper)
        else:
            mask = (tg_clean >= lower) & (tg_clean < upper)
        
        y_true_stratum = y_true_clean[mask]
        y_pred_stratum = y_pred_clean[mask]
        
        # Need at least 2 samples for meaningful evaluation
        if len(y_true_stratum) >= 2:
            try:
                metrics = evaluate_model(
                    y_true_stratum, 
                    y_pred_stratum, 
                    model_name=stratum_name
                )
                metrics['n_samples'] = len(y_true_stratum)
                metrics['tg_range'] = f"{lower}-{upper} mg/dL"
                results[stratum_name] = metrics
            except ValueError:
                # Not enough valid data in this stratum
                results[stratum_name] = None
        else:
            results[stratum_name] = None
    
    # Also include overall metrics
    if len(y_true_clean) >= 2:
        overall_metrics = evaluate_model(y_true_clean, y_pred_clean, model_name="overall")
        overall_metrics['tg_range'] = "all"
        results['overall'] = overall_metrics
    else:
        results['overall'] = None
    
    return results
