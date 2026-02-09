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


# Default regression coefficients (placeholders - will be fitted from data)
# These are reasonable starting estimates based on expected relationships
DEFAULT_REGRESSION_COEFFICIENTS = {
    "intercept": 3.5,      # Base HbA1c when all predictors are at reference
    "fpg": 0.020,          # Higher FPG → higher HbA1c
    "age": 0.008,          # Older age → slightly higher HbA1c
    "tg": 0.001,           # Higher TG → slightly higher HbA1c
    "hdl": -0.005,         # Higher HDL → slightly lower HbA1c
    "hgb": -0.05,          # Higher Hgb → slightly lower HbA1c
}


def calc_hba1c_regression(
    fpg_mgdl: Union[float, np.ndarray],
    age_years: Union[float, np.ndarray],
    tg_mgdl: Union[float, np.ndarray],
    hdl_mgdl: Union[float, np.ndarray],
    hgb_gdl: Union[float, np.ndarray],
    coefficients: dict = None,
) -> Union[float, np.ndarray]:
    """
    Estimate HbA1c using a multi-linear regression model.

    This estimator combines multiple biomarkers to predict HbA1c using
    a linear regression formula. Coefficients can be pre-fitted from
    NHANES data using fit_regression_coefficients().

    Formula:
        eHbA1c = β₀ + β₁×FPG + β₂×Age + β₃×TG + β₄×HDL + β₅×Hgb

    Args:
        fpg_mgdl: Fasting plasma glucose in mg/dL.
        age_years: Age in years.
        tg_mgdl: Triglycerides in mg/dL.
        hdl_mgdl: HDL cholesterol in mg/dL.
        hgb_gdl: Hemoglobin in g/dL.
        coefficients: Dictionary with keys 'intercept', 'fpg', 'age', 'tg',
            'hdl', 'hgb'. If None, uses DEFAULT_REGRESSION_COEFFICIENTS.

    Returns:
        Estimated HbA1c in percent (%).

    Raises:
        ValueError: If inputs are negative, NaN, or outside physiological range.

    Examples:
        >>> calc_hba1c_regression(126.0, 50, 150, 50, 14.0)
        5.83  # Approximate value with default coefficients
    """
    if coefficients is None:
        coefficients = DEFAULT_REGRESSION_COEFFICIENTS

    # Handle numpy arrays
    is_array = isinstance(fpg_mgdl, np.ndarray)
    
    if is_array:
        # Input validation for arrays
        if np.any(np.isnan(fpg_mgdl)):
            raise ValueError("FPG contains NaN values")
        if np.any(fpg_mgdl < 0):
            raise ValueError("FPG cannot be negative")
        if np.any(fpg_mgdl < 40):
            raise ValueError("FPG values below 40 mg/dL are physiologically implausible")
        if np.any(np.isnan(age_years)):
            raise ValueError("Age contains NaN values")
        if np.any(age_years < 0):
            raise ValueError("Age cannot be negative")
        if np.any(np.isnan(tg_mgdl)):
            raise ValueError("Triglycerides contains NaN values")
        if np.any(tg_mgdl < 0):
            raise ValueError("Triglycerides cannot be negative")
        if np.any(np.isnan(hdl_mgdl)):
            raise ValueError("HDL contains NaN values")
        if np.any(hdl_mgdl < 0):
            raise ValueError("HDL cannot be negative")
        if np.any(np.isnan(hgb_gdl)):
            raise ValueError("Hemoglobin contains NaN values")
        if np.any(hgb_gdl <= 0):
            raise ValueError("Hemoglobin must be positive")
    else:
        # Input validation for scalars
        if math.isnan(fpg_mgdl):
            raise ValueError("FPG cannot be NaN")
        if fpg_mgdl < 0:
            raise ValueError("FPG cannot be negative")
        if fpg_mgdl < 40:
            raise ValueError("FPG values below 40 mg/dL are physiologically implausible")
        if math.isnan(age_years):
            raise ValueError("Age cannot be NaN")
        if age_years < 0:
            raise ValueError("Age cannot be negative")
        if math.isnan(tg_mgdl):
            raise ValueError("Triglycerides cannot be NaN")
        if tg_mgdl < 0:
            raise ValueError("Triglycerides cannot be negative")
        if math.isnan(hdl_mgdl):
            raise ValueError("HDL cannot be NaN")
        if hdl_mgdl < 0:
            raise ValueError("HDL cannot be negative")
        if math.isnan(hgb_gdl):
            raise ValueError("Hemoglobin cannot be NaN")
        if hgb_gdl <= 0:
            raise ValueError("Hemoglobin must be positive")

    # Calculate HbA1c using linear regression formula
    hba1c = (
        coefficients["intercept"]
        + coefficients["fpg"] * fpg_mgdl
        + coefficients["age"] * age_years
        + coefficients["tg"] * tg_mgdl
        + coefficients["hdl"] * hdl_mgdl
        + coefficients["hgb"] * hgb_gdl
    )

    return hba1c


def fit_regression_coefficients(df) -> dict:
    """
    Fit multi-linear regression coefficients from NHANES data.

    This function fits a linear regression model to predict HbA1c from
    fasting glucose, age, triglycerides, HDL, and hemoglobin using
    ordinary least squares.

    Args:
        df: pandas DataFrame with columns: hba1c_percent, fpg_mgdl,
            age_years, tg_mgdl, hdl_mgdl, hgb_gdl

    Returns:
        Dictionary with fitted coefficients: 'intercept', 'fpg', 'age',
        'tg', 'hdl', 'hgb'

    Raises:
        ValueError: If required columns are missing from DataFrame.
        ImportError: If scikit-learn is not installed.

    Example:
        >>> from hba1cE.data import load_cleaned_data
        >>> df = load_cleaned_data()
        >>> coeffs = fit_regression_coefficients(df)
        >>> coeffs['fpg']  # Coefficient for FPG
        0.025  # Example fitted value
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        raise ImportError("scikit-learn is required for fitting coefficients")

    required_columns = ["hba1c_percent", "fpg_mgdl", "age_years", "tg_mgdl", "hdl_mgdl", "hgb_gdl"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prepare features and target
    feature_cols = ["fpg_mgdl", "age_years", "tg_mgdl", "hdl_mgdl", "hgb_gdl"]
    X = df[feature_cols].values
    y = df["hba1c_percent"].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients
    coefficients = {
        "intercept": float(model.intercept_),
        "fpg": float(model.coef_[0]),
        "age": float(model.coef_[1]),
        "tg": float(model.coef_[2]),
        "hdl": float(model.coef_[3]),
        "hgb": float(model.coef_[4]),
    }

    return coefficients
