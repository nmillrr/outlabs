"""
Utility functions for unit conversions and data processing.
"""

# Testosterone molecular weight: ~288.4 g/mol
# Conversion factor: ng/dL to nmol/L = 0.0347
_TT_CONVERSION_FACTOR = 0.0347


def ng_dl_to_nmol_l(value: float) -> float:
    """Convert testosterone from ng/dL to nmol/L.
    
    Args:
        value: Testosterone concentration in ng/dL
        
    Returns:
        Testosterone concentration in nmol/L
    """
    return value * _TT_CONVERSION_FACTOR


def nmol_l_to_ng_dl(value: float) -> float:
    """Convert testosterone from nmol/L to ng/dL.
    
    Args:
        value: Testosterone concentration in nmol/L
        
    Returns:
        Testosterone concentration in ng/dL
    """
    return value / _TT_CONVERSION_FACTOR


def mg_dl_to_g_l(value: float) -> float:
    """Convert albumin from mg/dL to g/L.
    
    Note: 1 mg/dL = 0.01 g/L (divide by 100 for mg->g, multiply by 10 for dL->L)
    
    Args:
        value: Albumin concentration in mg/dL
        
    Returns:
        Albumin concentration in g/L
    """
    return value * 0.01


def g_l_to_mg_dl(value: float) -> float:
    """Convert albumin from g/L to mg/dL.
    
    Args:
        value: Albumin concentration in g/L
        
    Returns:
        Albumin concentration in mg/dL
    """
    return value * 100
