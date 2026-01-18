"""
Mechanistic Free Testosterone Solvers

Implements Vermeulen, Södergård, and Zakharov FT calculation methods.
Reference: Vermeulen A et al. (1999) J Clin Endocrinol Metab
"""

import math
from scipy.optimize import brentq


def calc_ft_vermeulen(
    tt_nmoll: float,
    shbg_nmoll: float,
    alb_gl: float,
    K_shbg: float = 1e9,
    K_alb: float = 3.6e4
) -> float:
    """
    Calculate free testosterone using the Vermeulen (1999) equation.
    
    This method solves the mass balance equation for testosterone binding
    to SHBG and albumin using numerical root-finding.
    
    Parameters
    ----------
    tt_nmoll : float
        Total testosterone concentration in nmol/L
    shbg_nmoll : float
        SHBG concentration in nmol/L
    alb_gl : float
        Albumin concentration in g/L
    K_shbg : float, optional
        Association constant for SHBG-testosterone binding (L/mol)
        Default: 1e9 (Vermeulen 1999)
    K_alb : float, optional
        Association constant for albumin-testosterone binding (L/mol)
        Default: 3.6e4 (Vermeulen 1999)
    
    Returns
    -------
    float
        Free testosterone concentration in nmol/L
    
    Raises
    ------
    ValueError
        If any input is negative, NaN, or logically invalid
    
    Notes
    -----
    The mass balance equation is:
    TT = FT + [SHBG-T] + [Alb-T]
    
    Where:
    - [SHBG-T] = SHBG * K_shbg * FT / (1 + K_shbg * FT)
    - [Alb-T] = Alb_mol * K_alb * FT / (1 + K_alb * FT)
    
    Albumin molecular weight: 66430 g/mol
    """
    # Input validation
    if math.isnan(tt_nmoll) or math.isnan(shbg_nmoll) or math.isnan(alb_gl):
        raise ValueError("Input values cannot be NaN")
    
    if tt_nmoll < 0:
        raise ValueError(f"Total testosterone cannot be negative: {tt_nmoll}")
    
    if shbg_nmoll < 0:
        raise ValueError(f"SHBG cannot be negative: {shbg_nmoll}")
    
    if alb_gl < 0:
        raise ValueError(f"Albumin cannot be negative: {alb_gl}")
    
    if K_shbg <= 0 or K_alb <= 0:
        raise ValueError("Association constants must be positive")
    
    # Handle edge case of zero TT
    if tt_nmoll == 0:
        return 0.0
    
    # Convert albumin from g/L to mol/L (MW = 66430 g/mol)
    alb_mw = 66430.0
    alb_mol = alb_gl / alb_mw
    
    # Convert concentrations to mol/L for calculation
    tt_mol = tt_nmoll * 1e-9  # nmol/L to mol/L
    shbg_mol = shbg_nmoll * 1e-9  # nmol/L to mol/L
    
    def mass_balance(ft_mol: float) -> float:
        """
        Mass balance equation: TT - FT - SHBG-bound - Alb-bound = 0
        
        Uses standard binding equilibrium:
        [Protein-T] = [Protein] * K * [FT] / (1 + K * [FT])
        """
        if ft_mol <= 0:
            return -tt_mol  # Return negative to guide solver
        
        # SHBG-bound testosterone
        shbg_bound = shbg_mol * K_shbg * ft_mol / (1 + K_shbg * ft_mol)
        
        # Albumin-bound testosterone
        alb_bound = alb_mol * K_alb * ft_mol / (1 + K_alb * ft_mol)
        
        # Mass balance: TT = FT + SHBG-bound + Alb-bound
        return tt_mol - ft_mol - shbg_bound - alb_bound
    
    # Solve using Brent's method
    # Lower bound: very small positive (not zero to avoid division issues)
    # Upper bound: total testosterone (FT cannot exceed TT)
    ft_lower = 1e-15  # Essentially zero but positive
    ft_upper = tt_mol  # FT cannot exceed TT
    
    try:
        ft_mol = brentq(mass_balance, ft_lower, ft_upper, xtol=1e-15)
    except ValueError as e:
        # If brentq fails, it means no root in interval (shouldn't happen
        # with valid physiological inputs)
        raise ValueError(
            f"Could not solve for free testosterone. Check inputs: "
            f"TT={tt_nmoll}, SHBG={shbg_nmoll}, Alb={alb_gl}. Error: {e}"
        )
    
    # Convert back to nmol/L
    ft_nmoll = ft_mol * 1e9
    
    return ft_nmoll


def calc_ft_sodergard(
    tt_nmoll: float,
    shbg_nmoll: float,
    alb_gl: float
) -> float:
    """
    Calculate free testosterone using the Södergård equation variant.
    
    This method uses different binding constants from the Vermeulen original:
    - K_shbg = 1.2e9 L/mol (vs Vermeulen's 1e9)
    - K_alb = 2.4e4 L/mol (vs Vermeulen's 3.6e4)
    
    Reference: Södergård R et al. (1982) J Steroid Biochem
    
    Parameters
    ----------
    tt_nmoll : float
        Total testosterone concentration in nmol/L
    shbg_nmoll : float
        SHBG concentration in nmol/L
    alb_gl : float
        Albumin concentration in g/L
    
    Returns
    -------
    float
        Free testosterone concentration in nmol/L
    
    Raises
    ------
    ValueError
        If any input is negative, NaN, or logically invalid
    
    Notes
    -----
    Internally calls calc_ft_vermeulen with modified binding constants.
    The Södergård constants typically yield slightly different FT estimates
    compared to the Vermeulen constants.
    """
    # Södergård binding constants
    K_SHBG_SODERGARD = 1.2e9  # L/mol
    K_ALB_SODERGARD = 2.4e4   # L/mol
    
    return calc_ft_vermeulen(
        tt_nmoll=tt_nmoll,
        shbg_nmoll=shbg_nmoll,
        alb_gl=alb_gl,
        K_shbg=K_SHBG_SODERGARD,
        K_alb=K_ALB_SODERGARD
    )
