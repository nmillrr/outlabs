"""
eGFR/utils.py — Unit Conversion and Clinical Utilities

Provides helper functions for:
  - Creatinine unit conversion (mg/dL ↔ µmol/L)
  - Weight/height unit conversion (lbs ↔ kg, inches ↔ cm)
  - CKD stage classification from eGFR values
"""

# ---------------------------------------------------------------------------
# Creatinine conversions
# ---------------------------------------------------------------------------

def creatinine_mgdl_to_umoll(cr_mgdl: float) -> float:
    """Convert serum creatinine from mg/dL to µmol/L.

    Conversion factor: 1 mg/dL = 88.4 µmol/L

    Parameters
    ----------
    cr_mgdl : float
        Serum creatinine in mg/dL.

    Returns
    -------
    float
        Serum creatinine in µmol/L.
    """
    return cr_mgdl * 88.4


def creatinine_umoll_to_mgdl(cr_umoll: float) -> float:
    """Convert serum creatinine from µmol/L to mg/dL.

    Conversion factor: 1 µmol/L = 1/88.4 mg/dL

    Parameters
    ----------
    cr_umoll : float
        Serum creatinine in µmol/L.

    Returns
    -------
    float
        Serum creatinine in mg/dL.
    """
    return cr_umoll / 88.4


# ---------------------------------------------------------------------------
# CKD stage classification
# ---------------------------------------------------------------------------

def egfr_to_ckd_stage(egfr: float) -> str:
    """Classify CKD stage from an eGFR value (mL/min/1.73 m²).

    Uses KDIGO 2012 thresholds:
        G1  — eGFR ≥ 90   (normal or high)
        G2  — eGFR 60–89   (mildly decreased)
        G3a — eGFR 45–59   (mildly to moderately decreased)
        G3b — eGFR 30–44   (moderately to severely decreased)
        G4  — eGFR 15–29   (severely decreased)
        G5  — eGFR < 15    (kidney failure)

    Parameters
    ----------
    egfr : float
        Estimated GFR in mL/min/1.73 m².

    Returns
    -------
    str
        CKD stage string, e.g. ``"G1"``, ``"G3a"``, ``"G5"``.
    """
    if egfr >= 90:
        return "G1"
    elif egfr >= 60:
        return "G2"
    elif egfr >= 45:
        return "G3a"
    elif egfr >= 30:
        return "G3b"
    elif egfr >= 15:
        return "G4"
    else:
        return "G5"


# ---------------------------------------------------------------------------
# Weight conversions
# ---------------------------------------------------------------------------

def lbs_to_kg(weight_lbs: float) -> float:
    """Convert weight from pounds to kilograms.

    Parameters
    ----------
    weight_lbs : float
        Weight in pounds.

    Returns
    -------
    float
        Weight in kilograms.
    """
    return weight_lbs * 0.453592


def kg_to_lbs(weight_kg: float) -> float:
    """Convert weight from kilograms to pounds.

    Parameters
    ----------
    weight_kg : float
        Weight in kilograms.

    Returns
    -------
    float
        Weight in pounds.
    """
    return weight_kg / 0.453592


# ---------------------------------------------------------------------------
# Height conversions
# ---------------------------------------------------------------------------

def inches_to_cm(height_in: float) -> float:
    """Convert height from inches to centimetres.

    Parameters
    ----------
    height_in : float
        Height in inches.

    Returns
    -------
    float
        Height in centimetres.
    """
    return height_in * 2.54


def cm_to_inches(height_cm: float) -> float:
    """Convert height from centimetres to inches.

    Parameters
    ----------
    height_cm : float
        Height in centimetres.

    Returns
    -------
    float
        Height in inches.
    """
    return height_cm / 2.54
