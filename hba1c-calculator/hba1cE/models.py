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
