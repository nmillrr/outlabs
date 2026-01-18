"""
Mechanistic Free Testosterone Solvers

Implements Vermeulen, Södergård, and Zakharov FT calculation methods.
Reference: Vermeulen A et al. (1999) J Clin Endocrinol Metab
"""

import math
from scipy.optimize import brentq, fsolve


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


def calc_ft_zakharov(
    tt_nmoll: float,
    shbg_nmoll: float,
    alb_gl: float,
    cooperativity: float = 0.5
) -> float:
    """
    Calculate free testosterone using a simplified Zakharov allosteric model.
    
    This method accounts for cooperative binding of testosterone to SHBG,
    where binding at one site affects binding affinity at other sites.
    The allosteric model uses the Hill equation modification.
    
    Reference: Zakharov MN et al. (2015) J Clin Endocrinol Metab
    
    Parameters
    ----------
    tt_nmoll : float
        Total testosterone concentration in nmol/L
    shbg_nmoll : float
        SHBG concentration in nmol/L
    alb_gl : float
        Albumin concentration in g/L
    cooperativity : float, optional
        Hill coefficient for cooperative binding (0-1 range typical)
        Default: 0.5 (moderate negative cooperativity)
        - 0 = strong negative cooperativity
        - 1 = no cooperativity (reduces to Vermeulen)
        - >1 = positive cooperativity
    
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
    The allosteric model modifies SHBG binding:
    [SHBG-T] = SHBG * (K_shbg * FT)^n / (1 + (K_shbg * FT)^n)
    
    where n = 1 + cooperativity (Hill-like coefficient).
    
    Uses scipy.optimize.fsolve for solving the nonlinear system.
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
    
    if math.isnan(cooperativity):
        raise ValueError("Cooperativity cannot be NaN")
    
    # Handle edge case of zero TT
    if tt_nmoll == 0:
        return 0.0
    
    # Binding constants (using Vermeulen defaults as base)
    K_shbg = 1e9   # L/mol
    K_alb = 3.6e4  # L/mol
    
    # Convert albumin from g/L to mol/L (MW = 66430 g/mol)
    alb_mw = 66430.0
    alb_mol = alb_gl / alb_mw
    
    # Convert concentrations to mol/L for calculation
    tt_mol = tt_nmoll * 1e-9  # nmol/L to mol/L
    shbg_mol = shbg_nmoll * 1e-9  # nmol/L to mol/L
    
    # Hill-like coefficient for allosteric effect
    n = 1.0 + cooperativity
    
    def allosteric_mass_balance(ft_mol_arr):
        """
        Mass balance with allosteric SHBG binding.
        
        Uses Hill equation modification for SHBG:
        [SHBG-T] = SHBG * (K * FT)^n / (1 + (K * FT)^n)
        
        Albumin binding remains non-cooperative.
        """
        ft_mol = ft_mol_arr[0]
        
        # Prevent negative or zero FT
        if ft_mol <= 0:
            ft_mol = 1e-15
        
        # Allosteric SHBG binding (Hill-like)
        kf_n = (K_shbg * ft_mol) ** n
        shbg_bound = shbg_mol * kf_n / (1 + kf_n)
        
        # Standard albumin binding (non-cooperative)
        alb_bound = alb_mol * K_alb * ft_mol / (1 + K_alb * ft_mol)
        
        # Mass balance residual
        residual = tt_mol - ft_mol - shbg_bound - alb_bound
        return [residual]
    
    # Initial guess: use Vermeulen solution as starting point
    # This provides a good initial estimate for fsolve
    try:
        initial_guess = calc_ft_vermeulen(tt_nmoll, shbg_nmoll, alb_gl) * 1e-9
    except ValueError:
        # If Vermeulen fails, use a simple estimate (1% of TT)
        initial_guess = tt_mol * 0.01
    
    # Solve using fsolve
    solution, info, ier, mesg = fsolve(
        allosteric_mass_balance,
        [initial_guess],
        full_output=True
    )
    
    ft_mol = solution[0]
    
    # Validate solution
    if ft_mol <= 0:
        # Try again with different initial guess
        solution2, info2, ier2, mesg2 = fsolve(
            allosteric_mass_balance,
            [tt_mol * 0.02],  # 2% of TT
            full_output=True
        )
        ft_mol = solution2[0]
    
    # Final validation: ensure 0 < FT < TT
    if ft_mol <= 0 or ft_mol > tt_mol:
        raise ValueError(
            f"Could not find valid solution. FT={ft_mol*1e9:.4f} nmol/L "
            f"for TT={tt_nmoll}, SHBG={shbg_nmoll}, Alb={alb_gl}"
        )
    
    # Convert back to nmol/L
    ft_nmoll = ft_mol * 1e9
    
    return ft_nmoll


def calc_bioavailable_t(
    tt_nmoll: float,
    shbg_nmoll: float,
    alb_gl: float
) -> float:
    """
    Calculate bioavailable testosterone (free + albumin-bound).
    
    Bioavailable testosterone represents the fraction of total testosterone
    that is available to tissues: the unbound (free) fraction plus the
    weakly albumin-bound fraction. SHBG-bound testosterone is excluded
    as it is considered biologically inactive.
    
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
        Bioavailable testosterone concentration in nmol/L
    
    Raises
    ------
    ValueError
        If any input is negative, NaN, or logically invalid
    
    Notes
    -----
    Bioavailable T = Free T + Albumin-bound T
                   = TT - SHBG-bound T
    
    Uses the Vermeulen solver to calculate free testosterone and
    derives the albumin-bound fraction from mass balance.
    """
    # Input validation done in calc_ft_vermeulen
    # Handle edge case of zero TT
    if tt_nmoll == 0:
        return 0.0
    
    # Calculate free testosterone using Vermeulen
    ft_nmoll = calc_ft_vermeulen(tt_nmoll, shbg_nmoll, alb_gl)
    
    # Calculate SHBG-bound fraction using binding equilibrium
    # [SHBG-T] = SHBG * K_shbg * FT / (1 + K_shbg * FT)
    K_shbg = 1e9  # Vermeulen constant (L/mol)
    ft_mol = ft_nmoll * 1e-9  # Convert to mol/L
    shbg_mol = shbg_nmoll * 1e-9  # Convert to mol/L
    
    shbg_bound_mol = shbg_mol * K_shbg * ft_mol / (1 + K_shbg * ft_mol)
    shbg_bound_nmoll = shbg_bound_mol * 1e9  # Convert back to nmol/L
    
    # Bioavailable = Total - SHBG-bound = Free + Albumin-bound
    bioavailable_nmoll = tt_nmoll - shbg_bound_nmoll
    
    return bioavailable_nmoll

