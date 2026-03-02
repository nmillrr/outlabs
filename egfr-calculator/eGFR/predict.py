"""
eGFR/predict.py — Prediction API

Provides a unified prediction interface for:
  - CKD-EPI 2021 eGFR estimation
  - MDRD eGFR estimation
  - Cockcroft-Gault CrCl estimation
  - Hybrid ML-based eGFR prediction
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Optional, Union

from eGFR.models import (
    calc_crcl_cockcroft_gault,
    calc_egfr_ckd_epi_2021,
    calc_egfr_mdrd,
)
from eGFR.utils import egfr_to_ckd_stage


_VALID_METHODS = {"ckd_epi_2021", "mdrd", "cockcroft_gault", "hybrid"}

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def predict_egfr(
    cr_mgdl: float,
    age: float,
    sex: Union[str, int],
    weight_kg: Optional[float] = None,
    height_cm: Optional[float] = None,
    cystatin_c: Optional[float] = None,
    method: str = "ckd_epi_2021",
) -> dict[str, Any]:
    """Compute an eGFR (or CrCl) prediction using the specified method.

    Parameters
    ----------
    cr_mgdl : float
        Serum creatinine in mg/dL.  Must be > 0 and finite.
    age : float
        Patient age in years.  Must be >= 18 and finite.
    sex : str or int
        Patient sex.  Accepts ``'M'``/``'F'`` strings or 1/2 (NHANES coding).
    weight_kg : float, optional
        Body weight in kg.  Required for ``'cockcroft_gault'`` and recommended
        for ``'hybrid'``.
    height_cm : float, optional
        Height in cm.  Recommended for ``'hybrid'``.
    cystatin_c : float, optional
        Serum cystatin C in mg/L.  Optional additional feature for ``'hybrid'``.
    method : str
        Estimation method.  One of ``'ckd_epi_2021'`` (default), ``'mdrd'``,
        ``'cockcroft_gault'``, or ``'hybrid'``.

    Returns
    -------
    dict
        Keys:

        - ``egfr_pred`` (float) — Estimated value (eGFR or CrCl).
        - ``ckd_stage`` (str) — CKD stage derived from the prediction.
        - ``method`` (str) — The method that was used.
        - ``warning`` (str | None) — Any clinical caveat or input warning.

    Raises
    ------
    ValueError
        If ``method`` is unrecognised, or required inputs are missing/invalid.
    """
    method = method.strip().lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {sorted(_VALID_METHODS)}"
        )

    warning_msg: str | None = None

    if method == "ckd_epi_2021":
        egfr_pred = calc_egfr_ckd_epi_2021(cr_mgdl, age, sex)

    elif method == "mdrd":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            egfr_pred = calc_egfr_mdrd(cr_mgdl, age, sex)
        if caught:
            warning_msg = str(caught[-1].message)

    elif method == "cockcroft_gault":
        if weight_kg is None:
            raise ValueError("weight_kg is required for method='cockcroft_gault'")
        egfr_pred = calc_crcl_cockcroft_gault(cr_mgdl, age, weight_kg, sex)
        warning_msg = (
            "Cockcroft-Gault returns creatinine clearance (CrCl) in mL/min, "
            "not eGFR (mL/min/1.73 m²). Use CKD-EPI 2021 for eGFR-based "
            "CKD staging; use CrCl for FDA drug-dosing guidance."
        )

    elif method == "hybrid":
        egfr_pred, warning_msg = _predict_hybrid(
            cr_mgdl, age, sex, weight_kg, height_cm, cystatin_c
        )

    ckd_stage = egfr_to_ckd_stage(egfr_pred)

    return {
        "egfr_pred": egfr_pred,
        "ckd_stage": ckd_stage,
        "method": method,
        "warning": warning_msg,
    }


# ------------------------------------------------------------------
# Hybrid ML helper
# ------------------------------------------------------------------

def _predict_hybrid(
    cr_mgdl: float,
    age: float,
    sex: Union[str, int],
    weight_kg: Optional[float],
    height_cm: Optional[float],
    cystatin_c: Optional[float],
) -> tuple[float, str | None]:
    """Run the hybrid ML model, falling back to CKD-EPI 2021 if unavailable."""
    import numpy as np

    warning_msg: str | None = None

    # Try to load saved model
    model = _load_hybrid_model()
    if model is None:
        # Fallback: average of available mechanistic estimators
        egfr_pred, warning_msg = _hybrid_fallback(
            cr_mgdl, age, sex, weight_kg, height_cm, cystatin_c
        )
        return egfr_pred, warning_msg

    # Build feature vector in the same order as train.py create_features()
    from eGFR.models import _normalize_sex

    sex_norm = _normalize_sex(sex)
    sex_numeric = 1.0 if sex_norm == "F" else 0.0

    # Base features: cr_mgdl, age_years, sex_numeric, weight_kg, height_cm, bmi
    wt = weight_kg if weight_kg is not None else 80.0  # sensible default
    ht = height_cm if height_cm is not None else 170.0  # sensible default
    bmi = wt / ((ht / 100.0) ** 2) if ht > 0 else 25.0

    incomplete_inputs: list[str] = []
    if weight_kg is None:
        incomplete_inputs.append("weight_kg")
    if height_cm is None:
        incomplete_inputs.append("height_cm")

    # Mechanistic estimator features
    egfr_ckdepi = calc_egfr_ckd_epi_2021(cr_mgdl, age, sex)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        egfr_mdrd = calc_egfr_mdrd(cr_mgdl, age, sex)
    crcl_cg = calc_crcl_cockcroft_gault(cr_mgdl, age, wt, sex) if wt else 0.0

    # Derived features
    inv_cr = 1.0 / cr_mgdl
    log_cr = math.log(cr_mgdl)
    age_cr = age * cr_mgdl

    features = [
        cr_mgdl, age, sex_numeric, wt, ht, bmi,
        egfr_ckdepi, egfr_mdrd, crcl_cg,
        inv_cr, log_cr, age_cr,
    ]

    # Optional cystatin C features (must match training feature count)
    if cystatin_c is not None:
        cr_cys_ratio = cr_mgdl / cystatin_c if cystatin_c > 0 else 0.0
        features.insert(6, cystatin_c)
        features.insert(7, cr_cys_ratio)
    else:
        incomplete_inputs.append("cystatin_c")

    X = np.array(features).reshape(1, -1)

    try:
        egfr_pred = float(model.predict(X)[0])
    except Exception as exc:
        # Feature-count mismatch or model error — fall back gracefully
        egfr_pred, warning_msg = _hybrid_fallback(
            cr_mgdl, age, sex, weight_kg, height_cm, cystatin_c
        )
        warning_msg = (
            f"Hybrid model prediction failed ({exc}); "
            "fell back to mechanistic estimator average."
        )
        return egfr_pred, warning_msg

    if incomplete_inputs:
        warning_msg = (
            f"Inputs {incomplete_inputs} were not provided; defaults were used. "
            "Prediction may be less accurate."
        )

    return egfr_pred, warning_msg


def _hybrid_fallback(
    cr_mgdl: float,
    age: float,
    sex: Union[str, int],
    weight_kg: Optional[float],
    height_cm: Optional[float],
    cystatin_c: Optional[float],
) -> tuple[float, str]:
    """Average available mechanistic estimators as a hybrid fallback."""
    estimates: list[float] = []

    estimates.append(calc_egfr_ckd_epi_2021(cr_mgdl, age, sex))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimates.append(calc_egfr_mdrd(cr_mgdl, age, sex))

    # CG only if weight is available
    if weight_kg is not None:
        estimates.append(
            calc_crcl_cockcroft_gault(cr_mgdl, age, weight_kg, sex)
        )

    egfr_pred = sum(estimates) / len(estimates)
    warning_msg = (
        "No trained hybrid model found; returning average of mechanistic "
        f"estimators ({len(estimates)} equations)."
    )
    return egfr_pred, warning_msg


def _load_hybrid_model():
    """Attempt to load the best saved ML model from the models/ directory."""
    import joblib

    model_path = os.path.join(_DEFAULT_MODEL_DIR, "best_model.joblib")
    if not os.path.isfile(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None
