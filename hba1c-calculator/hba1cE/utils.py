"""
Utility functions for hba1cE.

This module contains unit conversion functions and other utilities
for data harmonization across different measurement systems.
"""

from typing import Union
import numpy as np

# Glucose conversion factor: 1 mmol/L = 18.018 mg/dL
GLUCOSE_CONVERSION_FACTOR = 18.018

# HbA1c NGSP to IFCC conversion: IFCC (mmol/mol) = (NGSP (%) - 2.15) * 10.929
NGSP_IFCC_SLOPE = 10.929
NGSP_IFCC_INTERCEPT = 2.15


def mg_dl_to_mmol_l(glucose_mgdl: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert glucose from mg/dL to mmol/L.
    
    Parameters
    ----------
    glucose_mgdl : float or array-like
        Glucose concentration in mg/dL
        
    Returns
    -------
    float or array-like
        Glucose concentration in mmol/L
        
    Examples
    --------
    >>> mg_dl_to_mmol_l(180)
    9.99...
    >>> mg_dl_to_mmol_l(100)
    5.55...
    """
    return glucose_mgdl / GLUCOSE_CONVERSION_FACTOR


def mmol_l_to_mg_dl(glucose_mmol: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert glucose from mmol/L to mg/dL.
    
    Parameters
    ----------
    glucose_mmol : float or array-like
        Glucose concentration in mmol/L
        
    Returns
    -------
    float or array-like
        Glucose concentration in mg/dL
        
    Examples
    --------
    >>> mmol_l_to_mg_dl(5.5)
    99.099
    >>> mmol_l_to_mg_dl(10.0)
    180.18
    """
    return glucose_mmol * GLUCOSE_CONVERSION_FACTOR


def percent_to_mmol_mol(hba1c_percent: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert HbA1c from NGSP (%) to IFCC (mmol/mol).
    
    The IFCC (International Federation of Clinical Chemistry) reference method
    uses mmol/mol, while NGSP (National Glycohemoglobin Standardization Program)
    uses percent.
    
    Formula: IFCC (mmol/mol) = (NGSP (%) - 2.15) × 10.929
    
    Parameters
    ----------
    hba1c_percent : float or array-like
        HbA1c in NGSP percent (%)
        
    Returns
    -------
    float or array-like
        HbA1c in IFCC mmol/mol
        
    Examples
    --------
    >>> percent_to_mmol_mol(6.5)  # Diabetes threshold
    47.5...
    >>> percent_to_mmol_mol(5.7)  # Prediabetes threshold
    38.8...
    
    References
    ----------
    Little RR, Rohlfing CL. NGSP and IFCC reference method standardization.
    """
    return (hba1c_percent - NGSP_IFCC_INTERCEPT) * NGSP_IFCC_SLOPE


def mmol_mol_to_percent(hba1c_mmol: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert HbA1c from IFCC (mmol/mol) to NGSP (%).
    
    The IFCC (International Federation of Clinical Chemistry) reference method
    uses mmol/mol, while NGSP (National Glycohemoglobin Standardization Program)
    uses percent.
    
    Formula: NGSP (%) = (IFCC (mmol/mol) / 10.929) + 2.15
    
    Parameters
    ----------
    hba1c_mmol : float or array-like
        HbA1c in IFCC mmol/mol
        
    Returns
    -------
    float or array-like
        HbA1c in NGSP percent (%)
        
    Examples
    --------
    >>> mmol_mol_to_percent(48)  # Diabetes threshold (approx)
    6.54...
    >>> mmol_mol_to_percent(39)  # Prediabetes threshold (approx)
    5.72...
    
    References
    ----------
    Little RR, Rohlfing CL. NGSP and IFCC reference method standardization.
    """
    return (hba1c_mmol / NGSP_IFCC_SLOPE) + NGSP_IFCC_INTERCEPT
