"""Evaluation metrics for HbA1c estimation models.

Functions for computing agreement statistics, concordance metrics,
and clinical performance measures against HPLC-measured HbA1c reference values.
"""

from typing import Dict, List, Optional, Union

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


def evaluate_model(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    model_name: str = "model",
) -> Dict[str, object]:
    """Compute comprehensive evaluation metrics for an HbA1c estimation model.

    Calculates RMSE, MAE, mean bias, Pearson correlation, Lin's CCC,
    Bland-Altman statistics, and the percentage of predictions within
    ±0.5% of measured HbA1c.

    Parameters
    ----------
    y_true : array-like
        True (reference) HbA1c values, e.g. HPLC-measured.
    y_pred : array-like
        Predicted (estimated) HbA1c values.
    model_name : str, optional
        Name of the model being evaluated (default ``"model"``).

    Returns
    -------
    dict
        Dictionary with keys:
        - model_name: Name of the evaluated model.
        - rmse: Root Mean Squared Error.
        - mae: Mean Absolute Error.
        - bias: Mean signed difference (pred − true).
        - r_pearson: Pearson correlation coefficient.
        - lin_ccc: Lin's Concordance Correlation Coefficient.
        - ba_stats: Bland-Altman statistics dict.
        - pct_within_0_5: Percentage of predictions within ±0.5% of true.

    Raises
    ------
    ValueError
        If inputs have different lengths, fewer than 2 elements,
        or contain NaN values.
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

    if len(y_true_arr) < 2:
        raise ValueError("Inputs must have at least 2 elements.")

    if np.any(np.isnan(y_true_arr)) or np.any(np.isnan(y_pred_arr)):
        raise ValueError("Inputs must not contain NaN values.")

    errors = y_pred_arr - y_true_arr
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    # Pearson correlation
    from scipy.stats import pearsonr

    r_pearson, _ = pearsonr(y_true_arr, y_pred_arr)

    # Lin's CCC (reuse existing function)
    ccc = lins_ccc(y_true_arr, y_pred_arr)

    # Bland-Altman statistics (reuse existing function)
    ba = bland_altman_stats(y_true_arr, y_pred_arr)

    # Percentage within ±0.5%
    within = np.abs(errors) <= 0.5
    pct_within_0_5 = float(np.mean(within) * 100.0)

    return {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r_pearson": float(r_pearson),
        "lin_ccc": ccc,
        "ba_stats": ba,
        "pct_within_0_5": pct_within_0_5,
    }


def evaluate_by_hba1c_strata(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    hba1c_values: Union[np.ndarray, list],
) -> Dict[str, Dict[str, object]]:
    """Evaluate model performance stratified by HbA1c clinical ranges.

    Splits predictions into three clinically meaningful HbA1c strata
    (normal, prediabetes, diabetes) and computes comprehensive metrics
    for each stratum using :func:`evaluate_model`.

    Parameters
    ----------
    y_true : array-like
        True (reference) HbA1c values, e.g. HPLC-measured.
    y_pred : array-like
        Predicted (estimated) HbA1c values.
    hba1c_values : array-like
        HbA1c values used for stratification (typically same as *y_true*).

    Returns
    -------
    dict
        Dictionary keyed by stratum name (``"normal"``, ``"prediabetes"``,
        ``"diabetes"``).  Each value is a metrics dict from
        :func:`evaluate_model`, or ``None`` if the stratum contains fewer
        than 2 samples.

    Raises
    ------
    ValueError
        If inputs have different lengths or contain NaN values.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    hba1c_arr = np.asarray(hba1c_values, dtype=float)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1 or hba1c_arr.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")

    if not (len(y_true_arr) == len(y_pred_arr) == len(hba1c_arr)):
        raise ValueError(
            f"All inputs must have the same length. "
            f"Got y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}, "
            f"hba1c_values={len(hba1c_arr)}."
        )

    if len(y_true_arr) == 0:
        raise ValueError("Inputs must not be empty.")

    if (
        np.any(np.isnan(y_true_arr))
        or np.any(np.isnan(y_pred_arr))
        or np.any(np.isnan(hba1c_arr))
    ):
        raise ValueError("Inputs must not contain NaN values.")

    strata = {
        "normal": hba1c_arr < 5.7,
        "prediabetes": (hba1c_arr >= 5.7) & (hba1c_arr < 6.5),
        "diabetes": hba1c_arr >= 6.5,
    }

    results: Dict[str, object] = {}
    for name, mask in strata.items():
        if np.sum(mask) < 2:
            results[name] = None
        else:
            results[name] = evaluate_model(
                y_true_arr[mask], y_pred_arr[mask], model_name=name
            )

    return results


def define_subgroups(df: "pd.DataFrame") -> "pd.DataFrame":
    """Create clinical subgroup columns on a DataFrame.

    Adds boolean or categorical columns for clinically relevant subgroups
    that may influence HbA1c estimation accuracy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``hgb_gdl``, ``sex``, ``age_years``,
        and ``mcv_fl`` columns.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with three new columns:

        - ``anemia`` (bool): ``True`` when Hgb < 12 g/dL (female, sex == 2)
          or Hgb < 13 g/dL (male, sex == 1).
        - ``age_group`` (str): ``"<40"``, ``"40-60"``, or ``">60"``.
        - ``mcv_group`` (str): ``"low"`` (< 80 fL), ``"normal"``
          (80–100 fL), or ``"high"`` (> 100 fL).

    Raises
    ------
    ValueError
        If any of the required columns are missing from *df*.
    """
    import pandas as pd

    required_cols = {"hgb_gdl", "sex", "age_years", "mcv_fl"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    # Anemia: sex-specific hemoglobin thresholds
    # NHANES coding: 1 = male, 2 = female
    out["anemia"] = False
    out.loc[(out["sex"] == 2) & (out["hgb_gdl"] < 12.0), "anemia"] = True
    out.loc[(out["sex"] == 1) & (out["hgb_gdl"] < 13.0), "anemia"] = True

    # Age groups
    conditions_age = [
        out["age_years"] < 40,
        (out["age_years"] >= 40) & (out["age_years"] <= 60),
        out["age_years"] > 60,
    ]
    out["age_group"] = np.select(conditions_age, ["<40", "40-60", ">60"], default="<40")

    # MCV groups
    conditions_mcv = [
        out["mcv_fl"] < 80,
        (out["mcv_fl"] >= 80) & (out["mcv_fl"] <= 100),
        out["mcv_fl"] > 100,
    ]
    out["mcv_group"] = np.select(conditions_mcv, ["low", "normal", "high"], default="normal")

    return out


def evaluate_by_subgroup(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    df: "pd.DataFrame",
    subgroup_col: str,
    subgroup_values: List[str],
) -> Dict[str, Optional[Dict[str, object]]]:
    """Evaluate model performance for each value of a subgroup column.

    Parameters
    ----------
    y_true : array-like
        True (reference) HbA1c values.
    y_pred : array-like
        Predicted (estimated) HbA1c values.
    df : pd.DataFrame
        DataFrame with the same number of rows as *y_true* / *y_pred*,
        containing the column named *subgroup_col*.
    subgroup_col : str
        Name of the column in *df* that defines subgroups.
    subgroup_values : list of str
        Specific values of *subgroup_col* to evaluate.

    Returns
    -------
    dict
        Dictionary keyed by subgroup value.  Each value is a metrics dict
        from :func:`evaluate_model`, or ``None`` if the subgroup contains
        fewer than 2 samples.

    Raises
    ------
    ValueError
        If *y_true* and *y_pred* differ in length, if lengths don't match
        *df*, if *subgroup_col* is not in *df*, or if inputs contain NaN.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}."
        )

    if len(y_true_arr) != len(df):
        raise ValueError(
            f"y_true/y_pred length ({len(y_true_arr)}) must match "
            f"DataFrame length ({len(df)})."
        )

    if subgroup_col not in df.columns:
        raise ValueError(f"Column '{subgroup_col}' not found in DataFrame.")

    if np.any(np.isnan(y_true_arr)) or np.any(np.isnan(y_pred_arr)):
        raise ValueError("Inputs must not contain NaN values.")

    results: Dict[str, Optional[Dict[str, object]]] = {}
    for value in subgroup_values:
        # Handle both boolean and string subgroup values
        if isinstance(value, bool):
            mask = df[subgroup_col].values == value
        else:
            mask = df[subgroup_col].values.astype(str) == str(value)

        n_samples = int(np.sum(mask))
        if n_samples < 2:
            results[str(value)] = None
        else:
            results[str(value)] = evaluate_model(
                y_true_arr[mask], y_pred_arr[mask], model_name=f"{subgroup_col}={value}"
            )

    return results
