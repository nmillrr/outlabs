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


def evaluate_model(y_true, y_pred, model_name="model"):
    """Compute a comprehensive suite of eGFR-relevant evaluation metrics.

    Consolidates RMSE, MAE, bias, Pearson correlation, P30/P10 accuracy,
    Bland-Altman statistics, and CKD-stage concordance into a single dict.

    Parameters
    ----------
    y_true : array-like
        Reference (measured) GFR values in mL/min/1.73 m².
    y_pred : array-like
        Predicted (estimated) GFR values in mL/min/1.73 m².
    model_name : str, optional
        Human-readable label for the model (default ``"model"``).

    Returns
    -------
    dict
        Keys returned:

        - **model_name** (*str*) — echo of *model_name* parameter
        - **rmse** (*float*) — root-mean-square error
        - **mae** (*float*) — mean absolute error
        - **bias** (*float*) — mean signed error (pred − true)
        - **r_pearson** (*float*) — Pearson correlation coefficient
        - **p30** (*float*) — % of predictions within ±30 % of reference
        - **p10** (*float*) — % of predictions within ±10 % of reference
        - **ba_stats** (*dict*) — Bland-Altman statistics
          (mean_bias, std_diff, loa_lower, loa_upper)
        - **ckd_stage_agreement** (*float*) — % of concordant CKD-stage
          classifications between true and predicted eGFR

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, have mismatched shapes, or
        contain zero reference values.
    """
    from eGFR.utils import egfr_to_ckd_stage

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # ── Input validation ────────────────────────────────────────────────
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Input arrays must not contain NaN values.")
    if np.any(y_true == 0):
        raise ValueError(
            "Reference values must not be zero (division by zero in "
            "percentage error calculation)."
        )

    # ── Core regression metrics ─────────────────────────────────────────
    errors = y_pred - y_true
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    # Pearson r
    if y_true.size >= 2:
        r_pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r_pearson = float("nan")

    # ── eGFR-specific metrics ───────────────────────────────────────────
    p30 = p30_accuracy(y_true, y_pred)
    p10 = p10_accuracy(y_true, y_pred)
    ba_stats = bland_altman_stats(y_true, y_pred)

    # ── CKD-stage concordance ───────────────────────────────────────────
    stages_true = [egfr_to_ckd_stage(v) for v in y_true]
    stages_pred = [egfr_to_ckd_stage(v) for v in y_pred]
    concordant = sum(1 for st, sp in zip(stages_true, stages_pred) if st == sp)
    ckd_stage_agreement = float(concordant / len(stages_true) * 100.0)

    return {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r_pearson": r_pearson,
        "p30": p30,
        "p10": p10,
        "ba_stats": ba_stats,
        "ckd_stage_agreement": ckd_stage_agreement,
    }


def evaluate_by_ckd_stage(y_true, y_pred, egfr_values=None):
    """Evaluate model performance stratified by CKD stage.

    Stratifies predictions by the CKD stage derived from *egfr_values*
    (or *y_true* if *egfr_values* is ``None``) and computes evaluation
    metrics for each stage.

    CKD stage ranges (KDIGO 2012):
        - G1: ≥ 90  mL/min/1.73 m²
        - G2: 60–89
        - G3a: 45–59
        - G3b: 30–44
        - G4: 15–29
        - G5: < 15

    Parameters
    ----------
    y_true : array-like
        Reference (measured) GFR values.
    y_pred : array-like
        Predicted (estimated) GFR values.
    egfr_values : array-like, optional
        eGFR values used for stage assignment. If ``None``, *y_true* is used.

    Returns
    -------
    dict
        Mapping of CKD stage string (e.g. ``"G1"``) to a metrics dict
        containing *n*, *rmse*, *mae*, *bias*, *p30*, and *p10*.

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, or have mismatched shapes.
    """
    from eGFR.utils import egfr_to_ckd_stage

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

    if egfr_values is None:
        egfr_values = y_true
    else:
        egfr_values = np.asarray(egfr_values, dtype=float)

    stages = np.array([egfr_to_ckd_stage(v) for v in egfr_values])

    stage_order = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
    result = {}
    for stage in stage_order:
        mask = stages == stage
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        errors = yp - yt
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))

        # P30/P10 — skip if any reference is zero
        if np.any(yt == 0):
            p30 = float("nan")
            p10 = float("nan")
        else:
            p30 = p30_accuracy(yt, yp)
            p10 = p10_accuracy(yt, yp)

        result[stage] = {
            "n": int(mask.sum()),
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "p30": p30,
            "p10": p10,
        }

    return result


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=2000,
                 ci=95.0, random_state=42):
    """Compute bootstrap confidence intervals for a metric function.

    Resamples *(y_true, y_pred)* pairs with replacement *n_bootstrap* times
    and computes the given *metric_func* on each resample to estimate a
    percentile-based confidence interval.

    Parameters
    ----------
    y_true : array-like
        Reference (measured) GFR values.
    y_pred : array-like
        Predicted (estimated) GFR values.
    metric_func : callable
        ``metric_func(y_true, y_pred) -> float``.
    n_bootstrap : int, default 2000
        Number of bootstrap resamples.
    ci : float, default 95.0
        Confidence level as a percentage in (0, 100).
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    tuple of (float, float, float)
        ``(lower, upper, mean)`` — the percentile-based CI bounds and the
        mean of the bootstrap distribution.

    Raises
    ------
    ValueError
        If inputs are empty, contain NaN, have mismatched shapes,
        *n_bootstrap* < 1, or *ci* is not in (0, 100).
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
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")
    if not (0 < ci < 100):
        raise ValueError(f"ci must be in (0, 100), got {ci}.")

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_func(y_true[idx], y_pred[idx])

    alpha = (100 - ci) / 2.0
    lower = float(np.percentile(scores, alpha))
    upper = float(np.percentile(scores, 100 - alpha))
    mean = float(np.mean(scores))

    return (lower, upper, mean)
