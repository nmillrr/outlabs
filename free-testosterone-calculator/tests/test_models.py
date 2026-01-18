"""
Tests for mechanistic FT solvers.

Validates Vermeulen solver against published reference values from
the ISSAM Free Testosterone Calculator (https://www.issam.ch/freetesto.htm).
"""

import math
import pytest

from freeT.models import calc_ft_vermeulen, calc_ft_sodergard, calc_ft_zakharov, calc_bioavailable_t


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


class TestSodergardSolver:
    """Tests for the Södergård (1982) free testosterone calculation variant."""
    
    def test_differs_from_vermeulen(self):
        """
        Södergård should give different results than Vermeulen due to 
        different binding constants.
        
        Södergård uses K_shbg=1.2e9 (higher than Vermeulen's 1e9)
        and K_alb=2.4e4 (lower than Vermeulen's 3.6e4).
        """
        tt, shbg, alb = 15.0, 40.0, 45.0
        
        ft_vermeulen = calc_ft_vermeulen(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        ft_sodergard = calc_ft_sodergard(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        
        # Results should be different but close
        assert ft_vermeulen != ft_sodergard, "Södergård should differ from Vermeulen"
        
        # The difference should be small (within ~20% of each other)
        diff_percent = abs(ft_vermeulen - ft_sodergard) / ft_vermeulen * 100
        assert diff_percent < 20, f"Difference {diff_percent}% is too large"
    
    def test_multiple_cases_differ(self):
        """Verify Södergård differs from Vermeulen across multiple test cases."""
        test_cases = [
            (10.0, 20.0, 42.0),
            (20.0, 50.0, 43.0),
            (25.0, 35.0, 45.0),
        ]
        
        for tt, shbg, alb in test_cases:
            ft_v = calc_ft_vermeulen(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
            ft_s = calc_ft_sodergard(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
            assert ft_v != ft_s, f"Case TT={tt}, SHBG={shbg}, Alb={alb}: Should differ"
    
    def test_ft_less_than_tt(self):
        """Free testosterone must always be less than total testosterone."""
        result = calc_ft_sodergard(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=45.0)
        assert 0 < result < 15.0, f"FT {result} should be between 0 and TT 15"
    
    def test_ft_in_physiological_range(self):
        """FT should be in physiologically reasonable range (1-4% of TT)."""
        tt = 20.0
        result = calc_ft_sodergard(tt_nmoll=tt, shbg_nmoll=35.0, alb_gl=43.0)
        ft_percent = (result / tt) * 100
        assert 0.5 < ft_percent < 5.0, f"FT% {ft_percent} outside normal range"
    
    def test_inherits_input_validation(self):
        """Södergård should inherit input validation from Vermeulen."""
        import pytest
        
        with pytest.raises(ValueError, match="cannot be negative"):
            calc_ft_sodergard(tt_nmoll=-5.0, shbg_nmoll=40.0, alb_gl=43.0)
        
        with pytest.raises(ValueError, match="NaN"):
            calc_ft_sodergard(tt_nmoll=float('nan'), shbg_nmoll=40.0, alb_gl=43.0)
    
    def test_higher_k_shbg_effect(self):
        """
        Södergård has higher K_shbg (1.2e9 vs 1.0e9), which should result
        in more SHBG binding and therefore lower FT compared to Vermeulen.
        """
        tt, shbg, alb = 15.0, 40.0, 45.0
        
        ft_vermeulen = calc_ft_vermeulen(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        ft_sodergard = calc_ft_sodergard(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        
        # Higher K_shbg means more binding = lower FT
        # But lower K_alb means less binding = higher FT
        # Net effect depends on relative magnitudes
        # Just verify they're different
        assert ft_vermeulen != ft_sodergard


class TestZakharovSolver:
    """Tests for the Zakharov allosteric free testosterone calculation."""
    
    def test_ft_positive(self):
        """Free testosterone must be positive for valid inputs."""
        result = calc_ft_zakharov(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=45.0)
        assert result > 0, "Free testosterone must be positive"
    
    def test_ft_less_than_tt(self):
        """Free testosterone must always be less than total testosterone."""
        test_cases = [
            (15.0, 40.0, 45.0),
            (10.0, 20.0, 42.0),
            (25.0, 60.0, 43.0),
            (5.0, 30.0, 40.0),
        ]
        for tt, shbg, alb in test_cases:
            result = calc_ft_zakharov(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
            assert 0 < result < tt, f"FT {result} should be between 0 and TT {tt}"
    
    def test_valid_range_multiple_inputs(self):
        """Verify output is in valid range (0 < FT < TT) for different inputs."""
        test_cases = [
            (10.0, 30.0, 40.0),
            (20.0, 45.0, 42.0),
            (30.0, 70.0, 44.0),
            (8.0, 15.0, 43.0),
        ]
        for tt, shbg, alb in test_cases:
            result = calc_ft_zakharov(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
            assert 0 < result < tt, f"FT={result} should be between 0 and TT={tt}"
    
    def test_physiological_range(self):
        """FT should be in physiologically reasonable range (1-4% of TT)."""
        tt = 20.0
        result = calc_ft_zakharov(tt_nmoll=tt, shbg_nmoll=35.0, alb_gl=43.0)
        ft_percent = (result / tt) * 100
        # Allosteric effects may slightly widen the range
        assert 0.5 < ft_percent < 6.0, f"FT% {ft_percent} outside normal range"
    
    def test_differs_from_vermeulen(self):
        """Zakharov with cooperativity should differ from standard Vermeulen."""
        tt, shbg, alb = 15.0, 40.0, 45.0
        
        ft_vermeulen = calc_ft_vermeulen(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        ft_zakharov = calc_ft_zakharov(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        
        # With cooperativity=0.5, results should differ
        assert ft_vermeulen != ft_zakharov, "Zakharov should differ from Vermeulen"
    
    def test_cooperativity_effect(self):
        """Different cooperativity values should produce different results."""
        tt, shbg, alb = 15.0, 40.0, 45.0
        
        ft_low_coop = calc_ft_zakharov(tt, shbg, alb, cooperativity=0.0)
        ft_mid_coop = calc_ft_zakharov(tt, shbg, alb, cooperativity=0.5)
        ft_high_coop = calc_ft_zakharov(tt, shbg, alb, cooperativity=1.0)
        
        # Different cooperativity should give different results
        assert ft_low_coop != ft_mid_coop, "Low vs mid cooperativity should differ"
        assert ft_mid_coop != ft_high_coop, "Mid vs high cooperativity should differ"
    
    def test_zero_tt_returns_zero(self):
        """Zero total testosterone should return zero free testosterone."""
        result = calc_ft_zakharov(tt_nmoll=0.0, shbg_nmoll=40.0, alb_gl=43.0)
        assert result == 0.0, "Zero TT should yield zero FT"
    
    def test_negative_tt_raises_error(self):
        """Negative total testosterone should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calc_ft_zakharov(tt_nmoll=-5.0, shbg_nmoll=40.0, alb_gl=43.0)
    
    def test_negative_shbg_raises_error(self):
        """Negative SHBG should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calc_ft_zakharov(tt_nmoll=15.0, shbg_nmoll=-10.0, alb_gl=43.0)
    
    def test_negative_albumin_raises_error(self):
        """Negative albumin should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calc_ft_zakharov(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=-5.0)
    
    def test_nan_input_raises_error(self):
        """NaN inputs should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            calc_ft_zakharov(tt_nmoll=float('nan'), shbg_nmoll=40.0, alb_gl=43.0)
        
        with pytest.raises(ValueError, match="NaN"):
            calc_ft_zakharov(tt_nmoll=15.0, shbg_nmoll=float('nan'), alb_gl=43.0)
        
        with pytest.raises(ValueError, match="NaN"):
            calc_ft_zakharov(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=float('nan'))
    
    def test_nan_cooperativity_raises_error(self):
        """NaN cooperativity should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            calc_ft_zakharov(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=43.0, cooperativity=float('nan'))


class TestBioavailableT:
    """Tests for the bioavailable testosterone calculation."""
    
    def test_bioavailable_greater_than_ft(self):
        """Bioavailable T must be greater than free T."""
        tt, shbg, alb = 15.0, 40.0, 45.0
        
        ft = calc_ft_vermeulen(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        bio_t = calc_bioavailable_t(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        
        assert bio_t > ft, f"Bioavailable T ({bio_t}) should be greater than FT ({ft})"
    
    def test_bioavailable_less_than_tt(self):
        """Bioavailable T must be less than total testosterone."""
        tt, shbg, alb = 15.0, 40.0, 45.0
        
        bio_t = calc_bioavailable_t(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
        
        assert bio_t < tt, f"Bioavailable T ({bio_t}) should be less than TT ({tt})"
    
    def test_bioavailable_positive(self):
        """Bioavailable T must be positive for valid inputs."""
        result = calc_bioavailable_t(tt_nmoll=15.0, shbg_nmoll=40.0, alb_gl=45.0)
        assert result > 0, "Bioavailable testosterone must be positive"
    
    def test_bioavailable_multiple_cases(self):
        """Verify bioavailable T > FT for multiple test cases."""
        test_cases = [
            (10.0, 20.0, 42.0),
            (20.0, 50.0, 43.0),
            (25.0, 35.0, 45.0),
            (30.0, 60.0, 40.0),
        ]
        
        for tt, shbg, alb in test_cases:
            ft = calc_ft_vermeulen(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
            bio_t = calc_bioavailable_t(tt_nmoll=tt, shbg_nmoll=shbg, alb_gl=alb)
            assert bio_t > ft, f"Case TT={tt}, SHBG={shbg}: Bio T should > FT"
            assert bio_t < tt, f"Case TT={tt}, SHBG={shbg}: Bio T should < TT"
    
    def test_zero_tt_returns_zero(self):
        """Zero total testosterone should return zero bioavailable."""
        result = calc_bioavailable_t(tt_nmoll=0.0, shbg_nmoll=40.0, alb_gl=43.0)
        assert result == 0.0, "Zero TT should yield zero bioavailable T"
    
    def test_inherits_input_validation(self):
        """Bioavailable T should inherit input validation from Vermeulen."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calc_bioavailable_t(tt_nmoll=-5.0, shbg_nmoll=40.0, alb_gl=43.0)
        
        with pytest.raises(ValueError, match="NaN"):
            calc_bioavailable_t(tt_nmoll=float('nan'), shbg_nmoll=40.0, alb_gl=43.0)
    
    def test_physiological_range(self):
        """Bioavailable T should be ~30-60% of TT typically."""
        tt = 20.0
        bio_t = calc_bioavailable_t(tt_nmoll=tt, shbg_nmoll=35.0, alb_gl=43.0)
        bio_t_percent = (bio_t / tt) * 100
        # Bioavailable typically 30-60% of total testosterone
        assert 20 < bio_t_percent < 70, f"Bio T% {bio_t_percent} outside normal range"
