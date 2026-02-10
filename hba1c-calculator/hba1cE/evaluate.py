"""Evaluation metrics for HbA1c estimation models.

Functions for computing agreement statistics, concordance metrics,
and clinical performance measures against HPLC-measured HbA1c reference values.
"""

from typing import Dict, Union

import numpy as np


def bland_altman_stats(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Compute Bland-Altman agreement statistics.

    Bland-Altman analysis assesses the agreement between two measurement methods
    by plotting differences against means and computing limits of agreement.

    Parameters
    ----------
    y_true : array-like
        True (reference) HbA1c values, e.g. HPLC-measured.
    y_pred : array-like
        Predicted (estimated) HbA1c values.

    Returns
    -------
    dict
        Dictionary with keys:
        - mean_bias: Mean difference (pred - true).
        - std_diff: Standard deviation of differences.
        - loa_lower: Lower limit of agreement (mean_bias - 1.96 * std_diff).
        - loa_upper: Upper limit of agreement (mean_bias + 1.96 * std_diff).

    Raises
    ------
    ValueError
        If inputs have different lengths, are empty, or contain NaN values.

    References
    ----------
    Bland JM, Altman DG. "Statistical methods for assessing agreement between
    two methods of clinical measurement." Lancet. 1986;1(8476):307-310.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")

    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        raise ValueError("Inputs must not be empty.")

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Inputs must have the same length. "
            f"Got y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}."
        )

    if np.any(np.isnan(y_true_arr)) or np.any(np.isnan(y_pred_arr)):
        raise ValueError("Inputs must not contain NaN values.")

    differences = y_pred_arr - y_true_arr
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


def lins_ccc(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute Lin's Concordance Correlation Coefficient (CCC).

    Lin's CCC measures the agreement between two continuous variables,
    combining both precision (Pearson correlation) and accuracy (bias
    correction factor) into a single index. CCC = 1 indicates perfect
    agreement; CCC = 0 indicates no agreement; CCC = -1 indicates
    perfect reversed agreement.

    Parameters
    ----------
    y_true : array-like
        True (reference) HbA1c values, e.g. HPLC-measured.
    y_pred : array-like
        Predicted (estimated) HbA1c values.

    Returns
    -------
    float
        Lin's CCC value in the range [-1, 1].

    Raises
    ------
    ValueError
        If inputs have different lengths, fewer than 2 elements,
        or contain NaN values.

    References
    ----------
    Lin LI. "A concordance correlation coefficient to evaluate
    reproducibility." Biometrics. 1989;45(1):255-268.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Inputs must have the same length. "
            f"Got y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}."
        )

    if len(y_true_arr) < 2 or len(y_pred_arr) < 2:
        raise ValueError("Inputs must have at least 2 elements.")

    if np.any(np.isnan(y_true_arr)) or np.any(np.isnan(y_pred_arr)):
        raise ValueError("Inputs must not contain NaN values.")

    mean_true = np.mean(y_true_arr)
    mean_pred = np.mean(y_pred_arr)
    var_true = np.var(y_true_arr, ddof=0)
    var_pred = np.var(y_pred_arr, ddof=0)
    covariance = np.mean((y_true_arr - mean_true) * (y_pred_arr - mean_pred))

    # CCC = 2 * cov(X,Y) / (var(X) + var(Y) + (mean_X - mean_Y)^2)
    numerator = 2.0 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    if denominator == 0.0:
        # All values are identical constants → perfect agreement
        return 1.0

    return float(numerator / denominator)
