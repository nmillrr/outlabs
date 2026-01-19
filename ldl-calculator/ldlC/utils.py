"""
Utility functions for ldlC package.

This module contains:
- Unit conversion functions (mg/dL to mmol/L and vice versa)
- Validation helpers
"""

from typing import Union

# Conversion factors
# Cholesterol: molecular weight ~386.65 g/mol → conversion factor 38.67
# Triglycerides: molecular weight ~885.7 g/mol → conversion factor 88.57
CHOLESTEROL_CONVERSION_FACTOR = 38.67
TRIGLYCERIDE_CONVERSION_FACTOR = 88.57


def mg_dl_to_mmol_l(
    value: Union[float, int], molecule: str = "cholesterol"
) -> float:
    """
    Convert a lipid concentration from mg/dL to mmol/L.

    Args:
        value: Concentration in mg/dL.
        molecule: Type of molecule - 'cholesterol' or 'triglycerides'.

    Returns:
        Concentration in mmol/L.

    Raises:
        ValueError: If molecule type is not recognized.

    Examples:
        >>> mg_dl_to_mmol_l(200, 'cholesterol')
        5.173...
        >>> mg_dl_to_mmol_l(150, 'triglycerides')
        1.693...
    """
    molecule = molecule.lower()

    if molecule == "cholesterol":
        return value / CHOLESTEROL_CONVERSION_FACTOR
    elif molecule in ("triglycerides", "triglyceride", "tg"):
        return value / TRIGLYCERIDE_CONVERSION_FACTOR
    else:
        raise ValueError(
            f"Unknown molecule type: '{molecule}'. "
            "Supported types: 'cholesterol', 'triglycerides'"
        )


def mmol_l_to_mg_dl(
    value: Union[float, int], molecule: str = "cholesterol"
) -> float:
    """
    Convert a lipid concentration from mmol/L to mg/dL.

    Args:
        value: Concentration in mmol/L.
        molecule: Type of molecule - 'cholesterol' or 'triglycerides'.

    Returns:
        Concentration in mg/dL.

    Raises:
        ValueError: If molecule type is not recognized.

    Examples:
        >>> mmol_l_to_mg_dl(5.17, 'cholesterol')
        199.92...
        >>> mmol_l_to_mg_dl(1.69, 'triglycerides')
        149.68...
    """
    molecule = molecule.lower()

    if molecule == "cholesterol":
        return value * CHOLESTEROL_CONVERSION_FACTOR
    elif molecule in ("triglycerides", "triglyceride", "tg"):
        return value * TRIGLYCERIDE_CONVERSION_FACTOR
    else:
        raise ValueError(
            f"Unknown molecule type: '{molecule}'. "
            "Supported types: 'cholesterol', 'triglycerides'"
        )
