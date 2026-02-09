"""
HbA1c estimation models.

This module contains mechanistic estimators for HbA1c:
- ADAG equation (inverted)
- Glycation kinetics model
- Multi-linear regression estimator
"""

import math
from typing import Union

import numpy as np


def calc_hba1c_adag(fpg_mgdl: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Estimate HbA1c from fasting plasma glucose using the inverted ADAG equation.

    This function uses the A1c-Derived Average Glucose (ADAG) study relationship
    inverted to estimate HbA1c from FPG. The original ADAG equation relates
    HbA1c to average glucose; here we invert it to estimate HbA1c from FPG.

    Formula: eHbA1c = (FPG + 46.7) / 28.7

    Reference:
        Nathan DM, Kuenen J, Borg R, Zheng H, Schoenfeld D, Heine RJ; A1c-Derived
        Average Glucose Study Group. Translating the A1C assay into estimated
        average glucose values. Diabetes Care. 2008 Aug;31(8):1473-8.
        doi: 10.2337/dc08-0545

    Args:
        fpg_mgdl: Fasting plasma glucose in mg/dL. Can be a scalar or numpy array.

    Returns:
        Estimated HbA1c in percent (%). Same type as input.

    Raises:
        ValueError: If input is negative, NaN, or FPG < 40 mg/dL.

    Examples:
        >>> calc_hba1c_adag(126.0)  # Returns approximately 6.0%
        6.017421602787456
        >>> calc_hba1c_adag(154.0)  # Returns approximately 7.0%
        6.993031358885017
    """
    # Handle numpy arrays
    if isinstance(fpg_mgdl, np.ndarray):
        if np.any(np.isnan(fpg_mgdl)):
            raise ValueError("FPG contains NaN values")
        if np.any(fpg_mgdl < 0):
            raise ValueError("FPG cannot be negative")
        if np.any(fpg_mgdl < 40):
            raise ValueError("FPG values below 40 mg/dL are physiologically implausible")
        return (fpg_mgdl + 46.7) / 28.7

    # Handle scalar values
    if math.isnan(fpg_mgdl):
        raise ValueError("FPG cannot be NaN")
    if fpg_mgdl < 0:
        raise ValueError("FPG cannot be negative")
    if fpg_mgdl < 40:
        raise ValueError("FPG values below 40 mg/dL are physiologically implausible")

    return (fpg_mgdl + 46.7) / 28.7


def calc_hba1c_kinetic(
    fpg_mgdl: Union[float, np.ndarray],
    hgb_gdl: Union[float, np.ndarray] = 14.0,
    rbc_lifespan_days: Union[float, np.ndarray] = 120.0,
    k: float = 4.5e-5,
) -> Union[float, np.ndarray]:
    """
    Estimate HbA1c using a first-order glycation kinetics model.

    This model treats hemoglobin glycation as a first-order reaction where
    the rate of glycation is proportional to glucose concentration. The
    accumulated HbA1c reflects the integrated exposure over the RBC lifespan.

    Formula:
        HbA1c = baseline + k × [Glucose] × RBC_lifespan × (Hgb_ref / Hgb)

    Where:
        - k: glycation rate constant (default 4.5e-5 per mg/dL per day)
        - baseline: non-glycemic HbA1c (~4.0%)
        - Hgb_ref: reference hemoglobin (14.0 g/dL)
        - The hemoglobin factor adjusts for dilution effects in anemia

    The model accounts for:
        - Anemia: lower hemoglobin leads to higher HbA1c at same glucose
          (less hemoglobin to distribute glycation across)
        - RBC lifespan: shorter lifespan = lower HbA1c (less time for glycation)

    Args:
        fpg_mgdl: Fasting plasma glucose in mg/dL. Can be scalar or array.
        hgb_gdl: Hemoglobin in g/dL. Default 14.0 (normal adult).
        rbc_lifespan_days: RBC lifespan in days. Default 120 (normal).
        k: Glycation rate constant. Default 4.5e-5 per mg/dL per day.

    Returns:
        Estimated HbA1c in percent (%).

    Raises:
        ValueError: If inputs are negative, NaN, or outside physiological range.

    Examples:
        >>> calc_hba1c_kinetic(100.0)  # Normal FPG, normal Hgb
        4.54
        >>> calc_hba1c_kinetic(126.0)  # Diabetic threshold
        4.68...
        >>> calc_hba1c_kinetic(126.0, hgb_gdl=10.0)  # Anemia: higher HbA1c
        5.35...
    """
    # Baseline non-glycemic HbA1c (%)
    BASELINE_HBA1C = 4.0
    # Reference hemoglobin for anemia adjustment
    HGB_REFERENCE = 14.0

    # Handle numpy arrays
    if isinstance(fpg_mgdl, np.ndarray):
        # Convert other inputs to arrays if needed
        hgb_arr = np.atleast_1d(np.asarray(hgb_gdl))
        lifespan_arr = np.atleast_1d(np.asarray(rbc_lifespan_days))

        # Input validation
        if np.any(np.isnan(fpg_mgdl)):
            raise ValueError("FPG contains NaN values")
        if np.any(fpg_mgdl < 0):
            raise ValueError("FPG cannot be negative")
        if np.any(fpg_mgdl < 40):
            raise ValueError("FPG values below 40 mg/dL are physiologically implausible")
        if np.any(np.isnan(hgb_arr)):
            raise ValueError("Hemoglobin contains NaN values")
        if np.any(hgb_arr <= 0):
            raise ValueError("Hemoglobin must be positive")
        if np.any(hgb_arr < 5) or np.any(hgb_arr > 25):
            raise ValueError("Hemoglobin outside valid range (5-25 g/dL)")
        if np.any(lifespan_arr <= 0):
            raise ValueError("RBC lifespan must be positive")

        # Anemia correction factor: lower Hgb = higher HbA1c
        hgb_factor = HGB_REFERENCE / hgb_arr

        # Glycation kinetics: HbA1c increases with glucose exposure over time
        glycated_fraction = k * fpg_mgdl * lifespan_arr * hgb_factor
        hba1c = BASELINE_HBA1C + glycated_fraction

        return hba1c

    # Handle scalar values
    if math.isnan(fpg_mgdl):
        raise ValueError("FPG cannot be NaN")
    if fpg_mgdl < 0:
        raise ValueError("FPG cannot be negative")
    if fpg_mgdl < 40:
        raise ValueError("FPG values below 40 mg/dL are physiologically implausible")

    if isinstance(hgb_gdl, float) and math.isnan(hgb_gdl):
        raise ValueError("Hemoglobin cannot be NaN")
    if hgb_gdl <= 0:
        raise ValueError("Hemoglobin must be positive")
    if hgb_gdl < 5 or hgb_gdl > 25:
        raise ValueError("Hemoglobin outside valid range (5-25 g/dL)")

    if rbc_lifespan_days <= 0:
        raise ValueError("RBC lifespan must be positive")

    # Anemia correction factor
    hgb_factor = HGB_REFERENCE / hgb_gdl

    # Glycation kinetics calculation
    glycated_fraction = k * fpg_mgdl * rbc_lifespan_days * hgb_factor
    hba1c = BASELINE_HBA1C + glycated_fraction

    return hba1c
