"""
Prediction API for HbA1c estimation.

Provides a unified interface for predicting HbA1c from routine blood markers
using either mechanistic estimators or trained ML models.
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from hba1cE.models import (
    calc_hba1c_adag,
    calc_hba1c_kinetic,
    calc_hba1c_regression,
)

# Default model directory relative to project root
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Feature names expected by the ML models (must match create_features output)
_ML_FEATURE_NAMES = [
    "fpg_mgdl",
    "tg_mgdl",
    "hdl_mgdl",
    "age_years",
    "hgb_gdl",
    "mcv_fl",
    "tg_hdl_ratio",
    "fpg_age_interaction",
    "adag_estimate",
    "kinetic_estimate",
    "regression_estimate",
]

_VALID_METHODS = {"adag", "kinetic", "regression", "hybrid"}


def _load_best_model(model_dir: Optional[Path] = None):
    """Load the best saved ML model from disk.

    Looks for a joblib model file in the models directory.  Searches for
    ``best_model.joblib`` first, then falls back to the first ``.joblib``
    file found.

    Parameters
    ----------
    model_dir : Path, optional
        Directory to search for model files.  Defaults to the ``models/``
        directory at the project root.

    Returns
    -------
    object or None
        A fitted scikit-learn-compatible model, or ``None`` if no model
        file is found.
    """
    if model_dir is None:
        model_dir = _MODEL_DIR

    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None

    # Prefer a model explicitly named "best_model"
    best_path = model_dir / "best_model.joblib"
    if best_path.exists():
        import joblib
        return joblib.load(best_path)

    # Fallback: pick the first .joblib file
    joblib_files = sorted(model_dir.glob("*.joblib"))
    if joblib_files:
        import joblib
        return joblib.load(joblib_files[0])

    return None


def _build_feature_vector(
    fpg: float,
    tg: Optional[float],
    hdl: Optional[float],
    age: Optional[float],
    hgb: Optional[float],
    mcv: Optional[float],
) -> np.ndarray:
    """Build a single-row feature vector for ML prediction.

    Fills in default values for any missing (``None``) optional inputs
    and computes derived features (ratios and mechanistic estimates).

    Parameters
    ----------
    fpg : float
        Fasting plasma glucose in mg/dL (required).
    tg : float or None
        Triglycerides in mg/dL.
    hdl : float or None
        HDL cholesterol in mg/dL.
    age : float or None
        Age in years.
    hgb : float or None
        Hemoglobin in g/dL.
    mcv : float or None
        Mean corpuscular volume in fL.

    Returns
    -------
    np.ndarray
        Feature vector of shape ``(1, 11)``.
    """
    # Defaults are population medians from NHANES data
    tg_val = tg if tg is not None else 120.0
    hdl_val = hdl if hdl is not None else 52.0
    age_val = age if age is not None else 50.0
    hgb_val = hgb if hgb is not None else 14.0
    mcv_val = mcv if mcv is not None else 90.0

    # Ratio features
    tg_hdl_ratio = tg_val / hdl_val
    fpg_age_interaction = fpg * age_val

    # Mechanistic estimator predictions
    adag_est = calc_hba1c_adag(fpg)
    kinetic_est = calc_hba1c_kinetic(fpg, hgb_gdl=hgb_val)
    regression_est = calc_hba1c_regression(
        fpg_mgdl=fpg,
        age_years=age_val,
        tg_mgdl=tg_val,
        hdl_mgdl=hdl_val,
        hgb_gdl=hgb_val,
    )

    feature_values = [
        fpg,
        tg_val,
        hdl_val,
        age_val,
        hgb_val,
        mcv_val,
        tg_hdl_ratio,
        fpg_age_interaction,
        adag_est,
        kinetic_est,
        regression_est,
    ]

    return np.array(feature_values).reshape(1, -1)


def predict_hba1c(
    fpg: float,
    tg: Optional[float] = None,
    hdl: Optional[float] = None,
    age: Optional[float] = None,
    hgb: Optional[float] = None,
    mcv: Optional[float] = None,
    method: str = "hybrid",
    model_dir: Optional[str] = None,
) -> Dict[str, Union[float, str, None]]:
    """Predict HbA1c from routine blood markers.

    Provides a unified prediction interface supporting mechanistic
    estimators (``'adag'``, ``'kinetic'``, ``'regression'``) and a
    hybrid ML approach (``'hybrid'``) that leverages a trained model.

    Parameters
    ----------
    fpg : float
        Fasting plasma glucose in mg/dL (required for all methods).
    tg : float, optional
        Triglycerides in mg/dL.  Required for ``'regression'`` method;
        optional for ``'hybrid'`` (uses population median if missing).
    hdl : float, optional
        HDL cholesterol in mg/dL.  Required for ``'regression'`` method;
        optional for ``'hybrid'`` (uses population median if missing).
    age : float, optional
        Age in years.  Required for ``'regression'`` method; optional
        for ``'hybrid'`` (uses population median if missing).
    hgb : float, optional
        Hemoglobin in g/dL.  Optional for ``'kinetic'`` and
        ``'hybrid'`` methods (uses 14.0 if missing).
    mcv : float, optional
        Mean corpuscular volume in fL.  Optional for ``'hybrid'``
        (uses population median if missing).
    method : str, optional
        Estimation method: ``'adag'``, ``'kinetic'``, ``'regression'``,
        or ``'hybrid'`` (default ``'hybrid'``).
    model_dir : str, optional
        Path to directory containing saved ML model files.  Only used
        when ``method='hybrid'``.  Defaults to ``models/`` at the
        project root.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``hba1c_pred`` (float): Estimated HbA1c in percent.
        - ``ci_lower`` (float or None): Lower bound of 95 % prediction
          interval (``None`` for mechanistic methods).
        - ``ci_upper`` (float or None): Upper bound of 95 % prediction
          interval (``None`` for mechanistic methods).
        - ``method`` (str): The estimation method actually used.
        - ``warning`` (str or None): Warning message, e.g. if inputs
          were incomplete or the method was downgraded.

    Raises
    ------
    ValueError
        If ``fpg`` is invalid (negative, NaN, or < 40 mg/dL) or
        ``method`` is not one of the valid options.
    """
    # --- Input validation ---------------------------------------------------
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {sorted(_VALID_METHODS)}."
        )

    if not isinstance(fpg, (int, float)):
        raise ValueError("fpg must be a numeric value.")
    if math.isnan(fpg):
        raise ValueError("fpg must not be NaN.")
    if fpg < 0:
        raise ValueError("fpg must not be negative.")
    if fpg < 40:
        raise ValueError("fpg must be at least 40 mg/dL.")

    # --- Mechanistic methods ------------------------------------------------
    warnings: List[str] = []

    if method == "adag":
        pred = float(calc_hba1c_adag(fpg))
        return {
            "hba1c_pred": pred,
            "ci_lower": None,
            "ci_upper": None,
            "method": "adag",
            "warning": None,
        }

    if method == "kinetic":
        hgb_val = hgb if hgb is not None else 14.0
        if hgb is None:
            warnings.append("hgb not provided; using default 14.0 g/dL")
        pred = float(calc_hba1c_kinetic(fpg, hgb_gdl=hgb_val))
        return {
            "hba1c_pred": pred,
            "ci_lower": None,
            "ci_upper": None,
            "method": "kinetic",
            "warning": "; ".join(warnings) if warnings else None,
        }

    if method == "regression":
        missing_inputs: List[str] = []
        if tg is None:
            missing_inputs.append("tg")
        if hdl is None:
            missing_inputs.append("hdl")
        if age is None:
            missing_inputs.append("age")
        if hgb is None:
            missing_inputs.append("hgb")
        if missing_inputs:
            raise ValueError(
                f"Regression method requires: {', '.join(missing_inputs)}. "
                "Supply these inputs or use method='hybrid'."
            )
        pred = float(
            calc_hba1c_regression(
                fpg_mgdl=fpg,
                age_years=age,  # type: ignore[arg-type]
                tg_mgdl=tg,  # type: ignore[arg-type]
                hdl_mgdl=hdl,  # type: ignore[arg-type]
                hgb_gdl=hgb,  # type: ignore[arg-type]
            )
        )
        return {
            "hba1c_pred": pred,
            "ci_lower": None,
            "ci_upper": None,
            "method": "regression",
            "warning": None,
        }

    # --- Hybrid ML method ---------------------------------------------------
    assert method == "hybrid"

    # Track which optional inputs were missing
    missing_optional: List[str] = []
    if tg is None:
        missing_optional.append("tg")
    if hdl is None:
        missing_optional.append("hdl")
    if age is None:
        missing_optional.append("age")
    if hgb is None:
        missing_optional.append("hgb")
    if mcv is None:
        missing_optional.append("mcv")
    if missing_optional:
        warnings.append(
            f"Missing inputs ({', '.join(missing_optional)}) replaced with "
            f"population median defaults"
        )

    # Try to load the ML model
    dir_path = Path(model_dir) if model_dir else None
    ml_model = _load_best_model(dir_path)

    if ml_model is None:
        # Fallback to ADAG when no trained model is available
        warnings.append(
            "No trained ML model found; falling back to ADAG estimator"
        )
        pred = float(calc_hba1c_adag(fpg))
        return {
            "hba1c_pred": pred,
            "ci_lower": None,
            "ci_upper": None,
            "method": "adag",
            "warning": "; ".join(warnings) if warnings else None,
        }

    # Build feature vector and predict
    X = _build_feature_vector(fpg, tg, hdl, age, hgb, mcv)
    pred = float(ml_model.predict(X)[0])

    # Simple empirical CI: ±0.5 % (typical RMSE for trained models)
    ci_lower = pred - 0.5
    ci_upper = pred + 0.5

    return {
        "hba1c_pred": pred,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "method": "hybrid",
        "warning": "; ".join(warnings) if warnings else None,
    }
