"""
eGFR/evaluate.py — Model Evaluation Metrics

Provides functions for:
  - Bland-Altman agreement analysis
  - P30/P10 accuracy metrics
  - Comprehensive model evaluation
  - CKD-stage stratified evaluation
  - Bootstrap confidence intervals
"""

import numpy as np


def bland_altman_stats(y_true, y_pred):
    """Compute Bland-Altman agreement statistics.

    The Bland-Altman method assesses agreement between two measurement
    methods by analysing the differences between paired observations.

    Parameters
    ----------
    y_true : array-like
        Reference (measured) values.
    y_pred : array-like
        Predicted (estimated) values.

    Returns
    -------
    dict
        mean_bias : float
            Mean difference (y_pred − y_true).  Positive ⇒ overestimation.
        std_diff : float
            Standard deviation of the differences.
        loa_lower : float
            Lower 95 % limit of agreement (mean_bias − 1.96 × std_diff).
        loa_upper : float
            Upper 95 % limit of agreement (mean_bias + 1.96 × std_diff).

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, or have mismatched lengths.

    References
    ----------
    Bland JM, Altman DG. Statistical methods for assessing agreement
    between two methods of clinical measurement. Lancet. 1986;1(8476):307-10.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")

    differences = y_pred - y_true
    mean_bias = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    loa_lower = mean_bias - 1.96 * std_diff
    loa_upper = mean_bias + 1.96 * std_diff

    return {
        "mean_bias": mean_bias,
        "std_diff": std_diff,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
    }
