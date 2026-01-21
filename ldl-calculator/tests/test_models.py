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

from ldlC.models import calc_ldl_friedewald


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
