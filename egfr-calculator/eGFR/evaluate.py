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


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Compute a comprehensive set of eGFR evaluation metrics.

    Combines RMSE, MAE, mean bias, Pearson *r*, P30, P10, Bland-Altman
    statistics, and CKD-stage concordance into a single results dict.

    Parameters
    ----------
    y_true : array-like
        Reference (measured) GFR values in mL/min/1.73 m².
    y_pred : array-like
        Predicted (estimated) GFR values in mL/min/1.73 m².
    model_name : str
        Human-readable identifier for the model (stored in the returned dict).

    Returns
    -------
    dict
        model_name : str
        rmse : float
        mae : float
        bias : float  (mean of y_pred − y_true; positive ⇒ overestimation)
        r_pearson : float
        p30 : float
        p10 : float
        ba_stats : dict  (from :func:`bland_altman_stats`)
        ckd_stage_agreement : float  (percentage of concordant CKD stages)

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, have mismatched lengths, or
        contain zero reference values.
    """
    from eGFR.utils import egfr_to_ckd_stage

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # ── input validation (shared with sub-functions, but fail-fast here) ──
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")

    # ── core regression metrics ──────────────────────────────────────────
    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))

    # Pearson correlation coefficient
    if y_true.size >= 2:
        r_pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r_pearson = float("nan")

    # ── eGFR-specific metrics ────────────────────────────────────────────
    # P30 / P10 require non-zero reference values
    has_zeros = np.any(y_true == 0)
    if has_zeros:
        _p30 = float("nan")
        _p10 = float("nan")
    else:
        _p30 = p30_accuracy(y_true, y_pred)
        _p10 = p10_accuracy(y_true, y_pred)

    ba = bland_altman_stats(y_true, y_pred)

    # ── CKD-stage concordance ────────────────────────────────────────────
    stages_true = np.array([egfr_to_ckd_stage(v) for v in y_true])
    stages_pred = np.array([egfr_to_ckd_stage(v) for v in y_pred])
    ckd_stage_agreement = float(np.mean(stages_true == stages_pred) * 100.0)

    return {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r_pearson": r_pearson,
        "p30": _p30,
        "p10": _p10,
        "ba_stats": ba,
        "ckd_stage_agreement": ckd_stage_agreement,
    }


# ── CKD-stage boundaries (mL/min/1.73 m²) ─────────────────────────────
_CKD_STAGES = [
    ("G1", 90, float("inf")),
    ("G2", 60, 90),
    ("G3a", 45, 60),
    ("G3b", 30, 45),
    ("G4", 15, 30),
    ("G5", 0, 15),
]


def evaluate_by_ckd_stage(y_true, y_pred, egfr_values):
    """Evaluate model performance stratified by CKD stage.

    Patients are assigned to stages based on *egfr_values* (typically the
    reference eGFR), and per-stage metrics are computed for each non-empty
    stage.

    CKD stages follow KDIGO 2012 guidelines:
        G1  ≥ 90,  G2  60–89,  G3a  45–59,  G3b  30–44,
        G4  15–29,  G5  < 15 mL/min/1.73 m².

    Parameters
    ----------
    y_true : array-like
        Reference (measured) GFR values.
    y_pred : array-like
        Predicted (estimated) GFR values.
    egfr_values : array-like
        eGFR values used for stage assignment (may be the same as *y_true*).

    Returns
    -------
    dict[str, dict]
        Outer keys are stage names (e.g. ``"G1"``, ``"G3a"``).
        Each inner dict contains:
            n : int – sample count
            rmse : float
            mae : float
            bias : float (mean of y_pred − y_true)
            p30 : float or nan (percentage within ±30 %)

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, or have mismatched lengths.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    egfr_values = np.asarray(egfr_values, dtype=float)

    # ── input validation ─────────────────────────────────────────────
    if y_true.size == 0 or y_pred.size == 0 or egfr_values.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true.shape != y_pred.shape or y_true.shape != egfr_values.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, "
            f"egfr_values {egfr_values.shape}."
        )
    if (np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred))
            or np.any(np.isnan(egfr_values))):
        raise ValueError("Input arrays must not contain NaN values.")

    results: dict = {}

    for stage_name, lower, upper in _CKD_STAGES:
        mask = (egfr_values >= lower) & (egfr_values < upper)
        # G1 upper bound is inf, so the < inf check always passes for ≥ 90
        n = int(mask.sum())
        if n == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        residuals = yp - yt

        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))
        bias = float(np.mean(residuals))

        # P30 (skip if any reference value is zero)
        if np.any(yt == 0):
            _p30 = float("nan")
        else:
            pct_err = np.abs(residuals) / np.abs(yt) * 100.0
            _p30 = float(np.sum(pct_err <= 30.0) / n * 100.0)

        results[stage_name] = {
            "n": n,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "p30": _p30,
        }

    return results


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap: int = 2000,
                 ci: float = 95.0, random_state: int = 42):
    """Compute bootstrap confidence intervals for any evaluation metric.

    Resamples *y_true* and *y_pred* with replacement *n_bootstrap* times,
    computes *metric_func* on each resample, and returns the percentile-based
    confidence interval.

    Parameters
    ----------
    y_true : array-like
        Reference (measured) values.
    y_pred : array-like
        Predicted (estimated) values.
    metric_func : callable
        A function ``metric_func(y_true, y_pred) -> float``.
    n_bootstrap : int, optional
        Number of bootstrap resamples (default 2000).
    ci : float, optional
        Confidence level in percent (default 95.0).
    random_state : int, optional
        Seed for reproducibility (default 42).

    Returns
    -------
    tuple[float, float, float]
        ``(lower, upper, mean)`` where *lower* and *upper* are the CI bounds
        and *mean* is the mean of the bootstrap distribution.

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, have mismatched lengths,
        *n_bootstrap* < 1, or *ci* is not in (0, 100).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # ── input validation ─────────────────────────────────────────────
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")
    if not (0 < ci < 100):
        raise ValueError("ci must be between 0 and 100 (exclusive).")

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_func(y_true[idx], y_pred[idx])

    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha))
    upper = float(np.percentile(scores, 100.0 - alpha))
    mean = float(np.mean(scores))

    return (lower, upper, mean)
