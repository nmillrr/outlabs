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

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    The Bland-Altman method assesses agreement between two measurement
    methods by analysing the differences between paired observations.
=======
    Calculates the mean bias (mean of differences), standard deviation of
    differences, and 95% limits of agreement (LoA) between two sets of
    measurements.
>>>>>>> Stashed changes
=======
    Calculates the mean bias (mean of differences), standard deviation of
    differences, and 95% limits of agreement (LoA) between two sets of
    measurements.
>>>>>>> Stashed changes
=======
    Calculates the mean bias (mean of differences), standard deviation of
    differences, and 95% limits of agreement (LoA) between two sets of
    measurements.
>>>>>>> Stashed changes

    Parameters
    ----------
    y_true : array-like
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        Reference (measured) values.
    y_pred : array-like
        Predicted (estimated) values.
=======
        Reference / true values (e.g. measured GFR).
    y_pred : array-like
        Predicted / estimated values (e.g. eGFR).
>>>>>>> Stashed changes
=======
        Reference / true values (e.g. measured GFR).
    y_pred : array-like
        Predicted / estimated values (e.g. eGFR).
>>>>>>> Stashed changes
=======
        Reference / true values (e.g. measured GFR).
    y_pred : array-like
        Predicted / estimated values (e.g. eGFR).
>>>>>>> Stashed changes

    Returns
    -------
    dict
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        mean_bias : float
            Mean difference (y_pred − y_true).  Positive ⇒ overestimation.
        std_diff : float
            Standard deviation of the differences.
        loa_lower : float
            Lower 95 % limit of agreement (mean_bias − 1.96 × std_diff).
        loa_upper : float
            Upper 95 % limit of agreement (mean_bias + 1.96 × std_diff).
=======
        Keys: ``mean_bias``, ``std_diff``, ``loa_lower``, ``loa_upper``.
>>>>>>> Stashed changes
=======
        Keys: ``mean_bias``, ``std_diff``, ``loa_lower``, ``loa_upper``.
>>>>>>> Stashed changes
=======
        Keys: ``mean_bias``, ``std_diff``, ``loa_lower``, ``loa_upper``.
>>>>>>> Stashed changes

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, or have mismatched lengths.

    References
    ----------
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    Bland JM, Altman DG. Statistical methods for assessing agreement
    between two methods of clinical measurement. Lancet. 1986;1(8476):307-10.
=======
    Bland JM, Altman DG. Statistical methods for assessing agreement between
    two methods of clinical measurement. Lancet. 1986;1(8476):307-310.
>>>>>>> Stashed changes
=======
    Bland JM, Altman DG. Statistical methods for assessing agreement between
    two methods of clinical measurement. Lancet. 1986;1(8476):307-310.
>>>>>>> Stashed changes
=======
    Bland JM, Altman DG. Statistical methods for assessing agreement between
    two methods of clinical measurement. Lancet. 1986;1(8476):307-310.
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    differences = y_pred - y_true
    mean_bias = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
=======
    diff = y_true - y_pred
    mean_bias = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
>>>>>>> Stashed changes
=======
    diff = y_true - y_pred
    mean_bias = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
>>>>>>> Stashed changes
=======
    diff = y_true - y_pred
    mean_bias = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
>>>>>>> Stashed changes
    loa_lower = mean_bias - 1.96 * std_diff
    loa_upper = mean_bias + 1.96 * std_diff

    return {
        "mean_bias": mean_bias,
        "std_diff": std_diff,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
    }
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream


def _pn_accuracy(y_true, y_pred, threshold: float) -> float:
    """Return the percentage of predictions within ±*threshold*% of reference.

    Helper shared by :func:`p30_accuracy` and :func:`p10_accuracy`.
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

    # Avoid division by zero for y_true == 0
    if np.any(y_true == 0):
        raise ValueError(
            "Reference values must not be zero (division by zero in "
            "percentage error calculation)."
        )

    pct_error = np.abs(y_pred - y_true) / np.abs(y_true) * 100.0
    within = np.sum(pct_error <= threshold)
    return float(within / len(y_true) * 100.0)


def p30_accuracy(y_true, y_pred) -> float:
    """Percentage of predictions within ±30 % of reference values.

    P30 is the standard accuracy metric for eGFR equations recommended by
    KDIGO.  A clinically acceptable eGFR equation should achieve P30 ≥ 85 %.

    Parameters
    ----------
    y_true : array-like
        Reference (measured) GFR values.
    y_pred : array-like
        Predicted (estimated) GFR values.

    Returns
    -------
    float
        Percentage (0–100) of predictions within ±30 % of reference.

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, have mismatched lengths, or
        contain zero reference values.
    """
    return _pn_accuracy(y_true, y_pred, threshold=30.0)


def p10_accuracy(y_true, y_pred) -> float:
    """Percentage of predictions within ±10 % of reference values.

    P10 is a stricter accuracy metric used in eGFR research to evaluate
    precision at tighter tolerances.

    Parameters
    ----------
    y_true : array-like
        Reference (measured) GFR values.
    y_pred : array-like
        Predicted (estimated) GFR values.

    Returns
    -------
    float
        Percentage (0–100) of predictions within ±10 % of reference.

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, have mismatched lengths, or
        contain zero reference values.
    """
    return _pn_accuracy(y_true, y_pred, threshold=10.0)
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
