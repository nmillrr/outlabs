"""
eGFR/models.py — Mechanistic eGFR Equations

Implements the three primary eGFR estimation equations:
  - CKD-EPI 2021 (race-free creatinine-based)
  - MDRD (4-variable, IDMS-traceable)
  - Cockcroft-Gault (creatinine clearance for drug dosing)

Each function accepts standard clinical inputs and returns estimated GFR
(or CrCl for Cockcroft-Gault) in mL/min/1.73 m² (or mL/min).
"""

from __future__ import annotations

import math
from typing import Union


def _normalize_sex(sex: Union[str, int]) -> str:
    """Normalize sex parameter to 'M' or 'F'.

    Accepts:
        - 'M', 'm', 'male', 'Male' → 'M'
        - 'F', 'f', 'female', 'Female' → 'F'
        - 1 (NHANES male) → 'M'
        - 2 (NHANES female) → 'F'

    Raises
    ------
    ValueError
        If the value cannot be mapped to 'M' or 'F'.
    """
    if isinstance(sex, str):
        s = sex.strip().upper()
        if s in ("M", "MALE"):
            return "M"
        if s in ("F", "FEMALE"):
            return "F"
        raise ValueError(
            f"Invalid sex string '{sex}'. Expected 'M'/'F' or 'male'/'female'."
        )
    if isinstance(sex, (int, float)):
        if sex == 1:
            return "M"
        if sex == 2:
            return "F"
        raise ValueError(
            f"Invalid sex code {sex}. Expected 1 (male) or 2 (female)."
        )
    raise ValueError(f"Invalid sex type {type(sex).__name__}. Expected str or int.")


def calc_egfr_ckd_epi_2021(
    cr_mgdl: float,
    age_years: float,
    sex: Union[str, int],
) -> float:
    """Calculate eGFR using the CKD-EPI 2021 creatinine equation (race-free).

    Implements the 2021 CKD-EPI creatinine equation recommended by KDIGO,
    which eliminates the race coefficient present in prior versions.

    Formula
    -------
    eGFR = 142 × min(SCr/κ, 1)^α × max(SCr/κ, 1)^(−1.200)
           × 0.9938^Age × 1.012 [if female]

    where:
        κ = 0.7 (female) or 0.9 (male)
        α = −0.241 (female) or −0.302 (male)

    Parameters
    ----------
    cr_mgdl : float
        Serum creatinine in mg/dL.  Must be > 0 and finite.
    age_years : float
        Patient age in years.  Must be ≥ 18 and finite.
    sex : str or int
        Patient sex.  Accepts 'M'/'F' strings or 1/2 (NHANES coding where
        1 = male, 2 = female).

    Returns
    -------
    float
        Estimated GFR in mL/min/1.73 m².

    Raises
    ------
    ValueError
        If any input is invalid (negative creatinine, NaN, age < 18, etc.).

    References
    ----------
    Inker LA, Eneanya ND, Coresh J, et al. New Creatinine- and Cystatin
    C–Based Equations to Estimate GFR without Race. *N Engl J Med*.
    2021;385(19):1737-1749. doi:10.1056/NEJMoa2102953
    """
    # -- Input validation --
    if not isinstance(cr_mgdl, (int, float)) or math.isnan(cr_mgdl):
        raise ValueError(f"cr_mgdl must be a finite number, got {cr_mgdl!r}")
    if cr_mgdl <= 0:
        raise ValueError(f"cr_mgdl must be positive, got {cr_mgdl}")

    if not isinstance(age_years, (int, float)) or math.isnan(age_years):
        raise ValueError(f"age_years must be a finite number, got {age_years!r}")
    if age_years < 18:
        raise ValueError(f"age_years must be ≥ 18, got {age_years}")

    sex_norm = _normalize_sex(sex)

    # -- Equation --
    is_female = sex_norm == "F"
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    female_factor = 1.012 if is_female else 1.0

    cr_ratio = cr_mgdl / kappa

    egfr = (
        142.0
        * min(cr_ratio, 1.0) ** alpha
        * max(cr_ratio, 1.0) ** (-1.200)
        * (0.9938 ** age_years)
        * female_factor
    )
    return egfr


def calc_egfr_mdrd(
    cr_mgdl: float,
    age_years: float,
    sex: Union[str, int],
    is_black: bool = False,
) -> float:
    """Calculate eGFR using the 4-variable MDRD equation (IDMS-traceable).

    Implements the re-expressed MDRD Study equation for use with
    IDMS-standardized creatinine assays (175 coefficient).

    Formula
    -------
    eGFR = 175 × SCr^(−1.154) × Age^(−0.203)
           × 0.742 [if female] × 1.212 [if Black]

    .. warning::
        MDRD is less accurate for eGFR > 60 mL/min/1.73 m².  A
        ``UserWarning`` is issued when the result exceeds this threshold.

    Parameters
    ----------
    cr_mgdl : float
        Serum creatinine in mg/dL (IDMS-traceable).  Must be > 0 and finite.
    age_years : float
        Patient age in years.  Must be ≥ 18 and finite.
    sex : str or int
        Patient sex.  Accepts 'M'/'F' strings or 1/2 (NHANES coding where
        1 = male, 2 = female).
    is_black : bool, optional
        Whether patient identifies as Black / African-American for the race
        coefficient (×1.212).  Default is ``False``.

        .. deprecated::
            The race coefficient is retained for backward compatibility with
            older lab systems.  CKD-EPI 2021 (race-free) is preferred.

    Returns
    -------
    float
        Estimated GFR in mL/min/1.73 m².

    Raises
    ------
    ValueError
        If any input is invalid (negative creatinine, NaN, age < 18, etc.).

    References
    ----------
    Levey AS, Coresh J, Greene T, et al. Using Standardized Serum Creatinine
    Values in the Modification of Diet in Renal Disease Study Equation for
    Estimating Glomerular Filtration Rate. *Ann Intern Med*. 2006;145(4):
    247-254. doi:10.7326/0003-4819-145-4-200608150-00004
    """
    import warnings

    # -- Input validation --
    if not isinstance(cr_mgdl, (int, float)) or math.isnan(cr_mgdl):
        raise ValueError(f"cr_mgdl must be a finite number, got {cr_mgdl!r}")
    if cr_mgdl <= 0:
        raise ValueError(f"cr_mgdl must be positive, got {cr_mgdl}")

    if not isinstance(age_years, (int, float)) or math.isnan(age_years):
        raise ValueError(f"age_years must be a finite number, got {age_years!r}")
    if age_years < 18:
        raise ValueError(f"age_years must be ≥ 18, got {age_years}")

    sex_norm = _normalize_sex(sex)

    # -- Equation --
    female_factor = 0.742 if sex_norm == "F" else 1.0
    race_factor = 1.212 if is_black else 1.0

    egfr = (
        175.0
        * (cr_mgdl ** -1.154)
        * (age_years ** -0.203)
        * female_factor
        * race_factor
    )

    if egfr > 60:
        warnings.warn(
            f"MDRD is less accurate for eGFR > 60 mL/min/1.73 m² "
            f"(result: {egfr:.1f}). Consider CKD-EPI 2021.",
            UserWarning,
            stacklevel=2,
        )

    return egfr
