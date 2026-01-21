"""
LDL-C estimation models.

This module contains mechanistic equations for LDL-C calculation:
- Friedewald equation
- Martin-Hopkins equation
- Extended Martin-Hopkins (for high TG)
- Sampson (NIH Equation 2)
"""

import math
import warnings
from typing import Union


def calc_ldl_friedewald(
    tc_mgdl: Union[float, int],
    hdl_mgdl: Union[float, int],
    tg_mgdl: Union[float, int],
) -> float:
    """
    Calculate LDL-C using the Friedewald equation (1972).

    Formula: LDL-C = TC - HDL-C - (TG / 5)

    This assumes a fixed 5:1 ratio of triglycerides to VLDL-C, which
    may be inaccurate for TG > 400 mg/dL.

    Args:
        tc_mgdl: Total cholesterol in mg/dL
        hdl_mgdl: HDL cholesterol in mg/dL
        tg_mgdl: Triglycerides in mg/dL

    Returns:
        Estimated LDL-C in mg/dL. Returns NaN for TG > 400 mg/dL.

    Raises:
        ValueError: If any input is negative or NaN.
    """
    # Validate inputs
    for name, value in [("tc_mgdl", tc_mgdl), ("hdl_mgdl", hdl_mgdl), ("tg_mgdl", tg_mgdl)]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError(f"Invalid input: {name} cannot be NaN or None")
        if value < 0:
            raise ValueError(f"Invalid input: {name} cannot be negative (got {value})")

    # Check for high triglycerides
    if tg_mgdl > 400:
        warnings.warn(
            f"Friedewald equation is unreliable for TG > 400 mg/dL (got {tg_mgdl}). "
            "Consider using Martin-Hopkins or Sampson equation instead.",
            UserWarning,
        )
        return float("nan")

    # Calculate LDL-C using Friedewald formula
    ldl_c = tc_mgdl - hdl_mgdl - (tg_mgdl / 5)

    return float(ldl_c)


# Martin-Hopkins 180-cell lookup table
# Rows: Non-HDL-C ranges (30 ranges)
# Columns: TG ranges (<100, 100-129, 130-159, 160-189, 190-219, ≥220)
# For TG 400-800, we use the extended row in the table
_MARTIN_HOPKINS_NON_HDL_RANGES = [
    (7, 49), (50, 56), (57, 61), (62, 66), (67, 71),
    (72, 75), (76, 79), (80, 83), (84, 87), (88, 92),
    (93, 96), (97, 100), (101, 105), (106, 110), (111, 115),
    (116, 120), (121, 126), (127, 132), (133, 138), (139, 146),
    (147, 154), (155, 163), (164, 173), (174, 185), (186, 201),
    (202, 220), (221, 247), (248, 292), (293, 399), (400, 13975),
]

_MARTIN_HOPKINS_TG_RANGES = [
    (0, 99), (100, 129), (130, 159), (160, 189), (190, 219), (220, 800),
]

# Factor table indexed by [non_hdl_row][tg_col]
_MARTIN_HOPKINS_FACTORS = [
    # <100  100-129 130-159 160-189 190-219 >=220
    [3.5, 3.4, 3.3, 3.3, 3.2, 3.1],  # 7-49
    [4.0, 3.9, 3.7, 3.6, 3.6, 3.4],  # 50-56
    [4.3, 4.1, 4.0, 3.9, 3.8, 3.6],  # 57-61
    [4.5, 4.3, 4.1, 4.0, 3.9, 3.9],  # 62-66
    [4.7, 4.4, 4.3, 4.2, 4.1, 3.9],  # 67-71
    [4.8, 4.6, 4.4, 4.2, 4.2, 4.1],  # 72-75
    [4.9, 4.6, 4.5, 4.3, 4.3, 4.2],  # 76-79
    [5.0, 4.8, 4.6, 4.4, 4.3, 4.2],  # 80-83
    [5.1, 4.8, 4.6, 4.5, 4.4, 4.3],  # 84-87
    [5.2, 4.9, 4.7, 4.6, 4.4, 4.3],  # 88-92
    [5.3, 5.0, 4.8, 4.7, 4.5, 4.4],  # 93-96
    [5.4, 5.1, 4.8, 4.7, 4.5, 4.3],  # 97-100
    [5.5, 5.2, 5.0, 4.7, 4.6, 4.5],  # 101-105
    [5.6, 5.3, 5.0, 4.8, 4.6, 4.5],  # 106-110
    [5.7, 5.4, 5.1, 4.9, 4.7, 4.5],  # 111-115
    [5.8, 5.5, 5.2, 5.0, 4.8, 4.6],  # 116-120
    [6.0, 5.5, 5.3, 5.0, 4.8, 4.6],  # 121-126
    [6.1, 5.7, 5.3, 5.1, 4.9, 4.7],  # 127-132
    [6.2, 5.8, 5.4, 5.2, 5.0, 4.7],  # 133-138
    [6.3, 5.9, 5.6, 5.3, 5.0, 4.8],  # 139-146
    [6.5, 6.0, 5.7, 5.4, 5.1, 4.8],  # 147-154
    [6.7, 6.2, 5.8, 5.4, 5.2, 4.9],  # 155-163
    [6.8, 6.3, 5.9, 5.5, 5.3, 5.0],  # 164-173
    [7.0, 6.5, 6.0, 5.7, 5.4, 5.1],  # 174-185
    [7.3, 6.7, 6.2, 5.8, 5.5, 5.2],  # 186-201
    [7.6, 6.9, 6.4, 6.0, 5.6, 5.3],  # 202-220
    [8.0, 7.2, 6.6, 6.2, 5.9, 5.4],  # 221-247
    [8.5, 7.6, 7.0, 6.5, 6.1, 5.6],  # 248-292
    [9.5, 8.3, 7.5, 7.0, 6.5, 5.9],  # 293-399
    [11.9, 10.0, 8.8, 8.1, 7.5, 6.7],  # 400-13975 (extended for high non-HDL)
]


def _get_non_hdl_row(non_hdl_mgdl: float) -> int:
    """Get the row index in the Martin-Hopkins table for a given non-HDL-C value."""
    for i, (low, high) in enumerate(_MARTIN_HOPKINS_NON_HDL_RANGES):
        if low <= non_hdl_mgdl <= high:
            return i
    # If above highest range, use last row
    if non_hdl_mgdl > 13975:
        return len(_MARTIN_HOPKINS_NON_HDL_RANGES) - 1
    # If below lowest range, use first row
    return 0


def _get_tg_col(tg_mgdl: float) -> int:
    """Get the column index in the Martin-Hopkins table for a given TG value."""
    for i, (low, high) in enumerate(_MARTIN_HOPKINS_TG_RANGES):
        if low <= tg_mgdl <= high:
            return i
    # If above highest range (>800), use last column
    if tg_mgdl > 800:
        return len(_MARTIN_HOPKINS_TG_RANGES) - 1
    # If below 0, use first column
    return 0


def calc_ldl_martin_hopkins(
    tc_mgdl: Union[float, int],
    hdl_mgdl: Union[float, int],
    tg_mgdl: Union[float, int],
) -> float:
    """
    Calculate LDL-C using the Martin-Hopkins equation with adjustable TG:VLDL factor.

    This method uses a 180-cell lookup table to determine the appropriate TG:VLDL
    adjustment factor based on the patient's non-HDL-C and triglyceride levels.
    This provides more accurate LDL-C estimates than Friedewald, especially for
    patients with low LDL-C or elevated triglycerides.

    Formula: LDL-C = TC - HDL-C - (TG / adjustable_factor)

    Args:
        tc_mgdl: Total cholesterol in mg/dL
        hdl_mgdl: HDL cholesterol in mg/dL
        tg_mgdl: Triglycerides in mg/dL

    Returns:
        Estimated LDL-C in mg/dL. Works for TG up to 800 mg/dL.

    Raises:
        ValueError: If any input is negative or NaN.
    """
    # Validate inputs
    for name, value in [("tc_mgdl", tc_mgdl), ("hdl_mgdl", hdl_mgdl), ("tg_mgdl", tg_mgdl)]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError(f"Invalid input: {name} cannot be NaN or None")
        if value < 0:
            raise ValueError(f"Invalid input: {name} cannot be negative (got {value})")

    # Calculate non-HDL-C
    non_hdl_mgdl = tc_mgdl - hdl_mgdl

    # Look up the adjustment factor from the 180-cell table
    row = _get_non_hdl_row(non_hdl_mgdl)
    col = _get_tg_col(tg_mgdl)
    factor = _MARTIN_HOPKINS_FACTORS[row][col]

    # Calculate LDL-C using the adjustable factor
    ldl_c = tc_mgdl - hdl_mgdl - (tg_mgdl / factor)

    return float(ldl_c)
