"""
Tests for mechanistic FT solvers.

Validates Vermeulen solver against published reference values from
the ISSAM Free Testosterone Calculator (https://www.issam.ch/freetesto.htm).
"""

import math
import pytest

from freeT.models import calc_ft_vermeulen


class TestVermeulenSolver:
    """Tests for the Vermeulen (1999) free testosterone calculation."""
    
    def test_reference_case_1(self):
        """
        Test case from ISSAM calculator:
        TT=15 nmol/L, SHBG=40 nmol/L, Alb=45 g/L → FT ≈ 0.269 nmol/L
        
        Note: PRD specified FT ≈ 0.30, but ISSAM calculator yields 0.269.
        Using tolerance of ±0.02 nmol/L to account for implementation differences.
        """
        result = calc_ft_vermeulen(
            tt_nmoll=15.0,
            shbg_nmoll=40.0,
            alb_gl=45.0
        )
        # ISSAM calculator: 0.269 nmol/L (1.79%)
        assert 0.25 < result < 0.30, f"Expected FT ~0.269, got {result}"
        # More precise assertion with tolerance
        assert abs(result - 0.269) < 0.02, f"Expected FT ≈ 0.269, got {result}"
    
    def test_reference_case_2_issam(self):
        """
        Test case verified against ISSAM calculator:
        TT=10 nmol/L, SHBG=20 nmol/L, Alb=42 g/L → FT ≈ 0.258 nmol/L
        
        Using tolerance of ±0.02 nmol/L for slight implementation variations.
        """
        result = calc_ft_vermeulen(
            tt_nmoll=10.0,
            shbg_nmoll=20.0,
            alb_gl=42.0
        )
        # ISSAM calculator: 0.258 nmol/L (2.58%)
        assert 0.24 < result < 0.28, f"Expected FT ~0.258, got {result}"
        # More precise assertion with tolerance
        assert abs(result - 0.258) < 0.02, f"Expected FT ≈ 0.258, got {result}"
    
    def test_ft_less_than_tt(self):
        """Free testosterone must always be less than total testosterone."""
        test_cases = [
            (15.0, 40.0, 45.0),
            (10.0, 20.0, 42.0),
            (25.0, 60.0, 43.0),
            (5.0, 30.0, 40.0),
        ]
        for tt, shbg, alb in test_cases:
            result = calc_ft_vermeulen(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
            assert 0 < result < tt, f"FT {result} should be between 0 and TT {tt}"
    
    def test_ft_positive(self):
        """Free testosterone must always be positive for valid inputs."""
        result = calc_ft_vermeulen(tt_nmoll=10.0, shbg_nmoll=40.0, alb_gl=43.0)
        assert result > 0, "Free testosterone must be positive"
    
    def test_physiological_range(self):
        """FT should be in physiologically reasonable range (1-3% of TT typically)."""
        tt = 20.0  # nmol/L
        result = calc_ft_vermeulen(tt_nmoll=tt, shbg_nmoll=35.0, alb_gl=43.0)
        ft_percent = (result / tt) * 100
        # Normal range: ~1-4% of total testosterone
        assert 0.5 < ft_percent < 5.0, f"FT% {ft_percent} outside normal range"
    
    def test_higher_shbg_lower_ft(self):
        """Higher SHBG should result in lower free testosterone."""
        ft_low_shbg = calc_ft_vermeulen(tt_nmoll=15.0, shbg_nmoll=20.0, alb_gl=43.0)
        ft_high_shbg = calc_ft_vermeulen(tt_nmoll=15.0, shbg_nmoll=60.0, alb_gl=43.0)
        assert ft_low_shbg > ft_high_shbg, "Higher SHBG should result in lower FT"
    
    def test_higher_tt_higher_ft(self):
        """Higher total testosterone should result in higher free testosterone."""
        ft_low_tt = calc_ft_vermeulen(tt_nmoll=10.0, shbg_nmoll=40.0, alb_gl=43.0)
        ft_high_tt = calc_ft_vermeulen(tt_nmoll=25.0, shbg_nmoll=40.0, alb_gl=43.0)
        assert ft_high_tt > ft_low_tt, "Higher TT should result in higher FT"
    
    # Input validation tests
    def test_negative_tt_raises_error(self):
        """Negative total testosterone should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calc_ft_vermeulen(tt_nmoll=-5.0, shbg_nmoll=40.0, alb_gl=43.0)
    
    def test_negative_shbg_raises_error(self):
        """Negative SHBG should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calc_ft_vermeulen(tt_nmoll=15.0, shbg_nmoll=-10.0, alb_gl=43.0)
    
    def test_negative_albumin_raises_error(self):
        """Negative albumin should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calc_ft_vermeulen(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=-5.0)
    
    def test_nan_input_raises_error(self):
        """NaN inputs should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            calc_ft_vermeulen(tt_nmoll=float('nan'), shbg_nmoll=40.0, alb_gl=43.0)
        
        with pytest.raises(ValueError, match="NaN"):
            calc_ft_vermeulen(tt_nmoll=15.0, shbg_nmoll=float('nan'), alb_gl=43.0)
        
        with pytest.raises(ValueError, match="NaN"):
            calc_ft_vermeulen(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=float('nan'))
    
    def test_zero_tt_returns_zero(self):
        """Zero total testosterone should return zero free testosterone."""
        result = calc_ft_vermeulen(tt_nmoll=0.0, shbg_nmoll=40.0, alb_gl=43.0)
        assert result == 0.0, "Zero TT should yield zero FT"
    
    def test_custom_binding_constants(self):
        """Custom binding constants should affect the result."""
        # Default constants
        ft_default = calc_ft_vermeulen(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=43.0)
        
        # Higher SHBG binding affinity should give lower FT
        ft_high_k_shbg = calc_ft_vermeulen(
            tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=43.0, K_shbg=2e9
        )
        assert ft_high_k_shbg < ft_default, "Higher K_shbg should reduce FT"
        
        # Higher albumin binding affinity should give lower FT
        ft_high_k_alb = calc_ft_vermeulen(
            tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=43.0, K_alb=7.2e4
        )
        assert ft_high_k_alb < ft_default, "Higher K_alb should reduce FT"
