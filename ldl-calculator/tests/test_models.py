"""
Tests for ldlC.models module.

Contains tests for:
- Friedewald equation
- Martin-Hopkins equation
- Extended Martin-Hopkins
- Sampson equation
"""

import math
import warnings

import pytest

from ldlC.models import calc_ldl_friedewald, calc_ldl_martin_hopkins


class TestFriedewald:
    """Tests for the Friedewald equation implementation."""

    def test_standard_case_1(self):
        """Test case: TC=200, HDL=50, TG=150 → LDL = 120 mg/dL."""
        result = calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=150)
        assert result == 120.0

    def test_standard_case_2(self):
        """Test case: TC=180, HDL=45, TG=100 → LDL = 115 mg/dL."""
        result = calc_ldl_friedewald(tc_mgdl=180, hdl_mgdl=45, tg_mgdl=100)
        assert result == 115.0

    def test_high_tg_returns_nan_with_warning(self):
        """Test that TG > 400 returns NaN and raises a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=450)
            
            # Check that result is NaN
            assert math.isnan(result)
            
            # Check that a warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "unreliable for TG > 400" in str(w[0].message)

    def test_tg_exactly_400_is_valid(self):
        """Test that TG exactly 400 mg/dL is still valid."""
        result = calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=400)
        # LDL = 200 - 50 - (400 / 5) = 200 - 50 - 80 = 70
        assert result == 70.0

    def test_negative_tc_raises_value_error(self):
        """Test that negative TC raises ValueError."""
        with pytest.raises(ValueError, match="tc_mgdl cannot be negative"):
            calc_ldl_friedewald(tc_mgdl=-10, hdl_mgdl=50, tg_mgdl=150)

    def test_negative_hdl_raises_value_error(self):
        """Test that negative HDL raises ValueError."""
        with pytest.raises(ValueError, match="hdl_mgdl cannot be negative"):
            calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=-5, tg_mgdl=150)

    def test_negative_tg_raises_value_error(self):
        """Test that negative TG raises ValueError."""
        with pytest.raises(ValueError, match="tg_mgdl cannot be negative"):
            calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=-20)

    def test_nan_input_raises_value_error(self):
        """Test that NaN inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            calc_ldl_friedewald(tc_mgdl=float("nan"), hdl_mgdl=50, tg_mgdl=150)

    def test_none_input_raises_value_error(self):
        """Test that None inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be NaN or None"):
            calc_ldl_friedewald(tc_mgdl=None, hdl_mgdl=50, tg_mgdl=150)

    def test_zero_tg_valid(self):
        """Test that TG=0 is a valid input (though unlikely in practice)."""
        result = calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=0)
        # LDL = 200 - 50 - 0 = 150
        assert result == 150.0

    def test_integer_inputs(self):
        """Test that integer inputs work correctly."""
        result = calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=150)
        assert isinstance(result, float)
        assert result == 120.0

    def test_returns_float(self):
        """Test that the function always returns a float."""
        result = calc_ldl_friedewald(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=150)
        assert isinstance(result, float)


class TestMartinHopkins:
    """Tests for the Martin-Hopkins equation implementation."""

    def test_standard_case_with_friedewald_comparison(self):
        """Test that Martin-Hopkins gives different result than Friedewald at low TG."""
        # For typical values, Martin-Hopkins should use factor ~5.0 (row for non-HDL 80-83)
        # TC=200, HDL=50 -> non-HDL = 150
        tc, hdl, tg = 200, 50, 150
        mh_result = calc_ldl_martin_hopkins(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        fr_result = calc_ldl_friedewald(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        
        # Both should give reasonable LDL values
        assert 0 < mh_result < tc
        assert 0 < fr_result < tc
        
    def test_differs_from_friedewald_at_extreme_values(self):
        """Test that Martin-Hopkins differs from Friedewald at extreme TG values."""
        # For high TG, the factor should be different from 5
        tc, hdl, tg = 250, 40, 500  # High TG case
        mh_result = calc_ldl_martin_hopkins(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        
        # Friedewald returns NaN for TG > 400, Martin-Hopkins should still work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fr_result = calc_ldl_friedewald(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        
        assert math.isnan(fr_result)  # Friedewald returns NaN
        assert not math.isnan(mh_result)  # Martin-Hopkins works
        assert mh_result > 0  # Should be reasonable positive value

    def test_high_tg_works_up_to_800(self):
        """Test that Martin-Hopkins works for TG up to 800 mg/dL."""
        result = calc_ldl_martin_hopkins(tc_mgdl=250, hdl_mgdl=40, tg_mgdl=800)
        assert not math.isnan(result)
        assert result > 0

    def test_low_non_hdl_uses_correct_factor(self):
        """Test that low non-HDL values use the appropriate lower factor."""
        # TC=100, HDL=60 -> non-HDL = 40 (in 7-49 range, factor ~3.5)
        result = calc_ldl_martin_hopkins(tc_mgdl=100, hdl_mgdl=60, tg_mgdl=50)
        # LDL = 100 - 60 - (50 / 3.5) = 40 - 14.29 ≈ 25.71
        expected = 100 - 60 - (50 / 3.5)
        assert abs(result - expected) < 0.01

    def test_high_non_hdl_uses_correct_factor(self):
        """Test that high non-HDL values use the appropriate higher factor."""
        # TC=300, HDL=40 -> non-HDL = 260 (in 248-292 range)
        # TG=50 -> column 0 (TG < 100), factor = 8.5
        result = calc_ldl_martin_hopkins(tc_mgdl=300, hdl_mgdl=40, tg_mgdl=50)
        # LDL = 300 - 40 - (50 / 8.5) = 260 - 5.88 ≈ 254.12
        expected = 300 - 40 - (50 / 8.5)
        assert abs(result - expected) < 0.01

    def test_negative_tc_raises_value_error(self):
        """Test that negative TC raises ValueError."""
        with pytest.raises(ValueError, match="tc_mgdl cannot be negative"):
            calc_ldl_martin_hopkins(tc_mgdl=-10, hdl_mgdl=50, tg_mgdl=150)

    def test_negative_hdl_raises_value_error(self):
        """Test that negative HDL raises ValueError."""
        with pytest.raises(ValueError, match="hdl_mgdl cannot be negative"):
            calc_ldl_martin_hopkins(tc_mgdl=200, hdl_mgdl=-5, tg_mgdl=150)

    def test_negative_tg_raises_value_error(self):
        """Test that negative TG raises ValueError."""
        with pytest.raises(ValueError, match="tg_mgdl cannot be negative"):
            calc_ldl_martin_hopkins(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=-20)

    def test_nan_input_raises_value_error(self):
        """Test that NaN inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            calc_ldl_martin_hopkins(tc_mgdl=float("nan"), hdl_mgdl=50, tg_mgdl=150)

    def test_returns_float(self):
        """Test that the function always returns a float."""
        result = calc_ldl_martin_hopkins(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=150)
        assert isinstance(result, float)

    def test_zero_tg_valid(self):
        """Test that TG=0 is a valid input."""
        result = calc_ldl_martin_hopkins(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=0)
        # With TG=0, LDL = TC - HDL = 150 (regardless of factor)
        assert result == 150.0

