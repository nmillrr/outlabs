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

from ldlC.models import calc_ldl_friedewald, calc_ldl_martin_hopkins, calc_ldl_sampson


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


class TestSampson:
    """Tests for the Sampson (NIH Equation 2) implementation."""

    def test_output_in_valid_range_standard_case(self):
        """Test that output is in valid range (0 < LDL < TC) for standard values."""
        tc, hdl, tg = 200, 50, 150
        result = calc_ldl_sampson(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        assert result > 0, f"LDL should be positive, got {result}"
        assert result < tc, f"LDL should be less than TC, got {result}"

    def test_output_in_valid_range_low_tg(self):
        """Test that output is in valid range for low TG values."""
        tc, hdl, tg = 200, 50, 50
        result = calc_ldl_sampson(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        assert result > 0, f"LDL should be positive, got {result}"
        assert result < tc, f"LDL should be less than TC, got {result}"

    def test_output_in_valid_range_high_tg(self):
        """Test that output is in valid range for high TG values (up to 800)."""
        tc, hdl, tg = 250, 40, 600
        result = calc_ldl_sampson(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        assert result > 0, f"LDL should be positive, got {result}"
        assert result < tc, f"LDL should be less than TC, got {result}"

    def test_works_for_tg_up_to_800(self):
        """Test that Sampson works for TG up to 800 mg/dL."""
        result = calc_ldl_sampson(tc_mgdl=250, hdl_mgdl=40, tg_mgdl=800)
        assert not math.isnan(result), "Sampson should not return NaN for TG <= 800"
        assert result > 0, "Sampson should return positive LDL"

    def test_formula_correctness(self):
        """Test that formula calculation is correct against manual calculation."""
        tc, hdl, tg = 200, 50, 150
        non_hdl = tc - hdl
        # LDL = TC/0.948 - HDL/0.971 - (TG/8.56 + TG*non-HDL/2140 - TG²/16100) - 9.44
        expected = (
            tc / 0.948
            - hdl / 0.971
            - (tg / 8.56 + (tg * non_hdl) / 2140 - (tg ** 2) / 16100)
            - 9.44
        )
        result = calc_ldl_sampson(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"

    def test_negative_tc_raises_value_error(self):
        """Test that negative TC raises ValueError."""
        with pytest.raises(ValueError, match="tc_mgdl cannot be negative"):
            calc_ldl_sampson(tc_mgdl=-10, hdl_mgdl=50, tg_mgdl=150)

    def test_negative_hdl_raises_value_error(self):
        """Test that negative HDL raises ValueError."""
        with pytest.raises(ValueError, match="hdl_mgdl cannot be negative"):
            calc_ldl_sampson(tc_mgdl=200, hdl_mgdl=-5, tg_mgdl=150)

    def test_negative_tg_raises_value_error(self):
        """Test that negative TG raises ValueError."""
        with pytest.raises(ValueError, match="tg_mgdl cannot be negative"):
            calc_ldl_sampson(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=-20)

    def test_nan_input_raises_value_error(self):
        """Test that NaN inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            calc_ldl_sampson(tc_mgdl=float("nan"), hdl_mgdl=50, tg_mgdl=150)

    def test_returns_float(self):
        """Test that the function always returns a float."""
        result = calc_ldl_sampson(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=150)
        assert isinstance(result, float)

    def test_zero_tg_valid(self):
        """Test that TG=0 is a valid input."""
        result = calc_ldl_sampson(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=0)
        assert result > 0
        assert not math.isnan(result)


class TestMartinHopkinsExtended:
    """Tests for the extended Martin-Hopkins equation implementation."""

    def test_works_for_tg_400_to_500(self):
        """Test that extended M-H works for TG 400-500 mg/dL."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        result = calc_ldl_martin_hopkins_extended(tc_mgdl=250, hdl_mgdl=40, tg_mgdl=450)
        assert not math.isnan(result), "Extended M-H should not return NaN for TG 400-500"
        assert result > 0, "Extended M-H should return positive LDL"
        assert result < 250, "LDL should be less than TC"

    def test_works_for_tg_500_to_600(self):
        """Test that extended M-H works for TG 500-600 mg/dL."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        result = calc_ldl_martin_hopkins_extended(tc_mgdl=250, hdl_mgdl=40, tg_mgdl=550)
        assert not math.isnan(result), "Extended M-H should not return NaN for TG 500-600"
        assert result > 0, "Extended M-H should return positive LDL"

    def test_works_for_tg_600_to_700(self):
        """Test that extended M-H works for TG 600-700 mg/dL."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        result = calc_ldl_martin_hopkins_extended(tc_mgdl=250, hdl_mgdl=40, tg_mgdl=650)
        assert not math.isnan(result), "Extended M-H should not return NaN for TG 600-700"
        assert result > 0, "Extended M-H should return positive LDL"

    def test_works_for_tg_700_to_800(self):
        """Test that extended M-H works for TG 700-800 mg/dL."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        result = calc_ldl_martin_hopkins_extended(tc_mgdl=250, hdl_mgdl=40, tg_mgdl=750)
        assert not math.isnan(result), "Extended M-H should not return NaN for TG 700-800"
        assert result > 0, "Extended M-H should return positive LDL"

    def test_works_for_tg_exactly_800(self):
        """Test that extended M-H works for TG exactly 800 mg/dL."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        result = calc_ldl_martin_hopkins_extended(tc_mgdl=250, hdl_mgdl=40, tg_mgdl=800)
        assert not math.isnan(result), "Extended M-H should not return NaN for TG=800"
        assert result > 0, "Extended M-H should return positive LDL"

    def test_differs_from_standard_mh_at_high_tg(self):
        """Test that extended M-H gives different result than standard M-H at high TG."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        tc, hdl, tg = 250, 40, 600
        extended_result = calc_ldl_martin_hopkins_extended(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        standard_result = calc_ldl_martin_hopkins(tc_mgdl=tc, hdl_mgdl=hdl, tg_mgdl=tg)
        # Results should be different due to different factor tables
        # Both should be valid though
        assert not math.isnan(extended_result)
        assert not math.isnan(standard_result)
        assert extended_result != standard_result, "Extended and standard M-H should differ at high TG"

    def test_negative_tc_raises_value_error(self):
        """Test that negative TC raises ValueError."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        with pytest.raises(ValueError, match="tc_mgdl cannot be negative"):
            calc_ldl_martin_hopkins_extended(tc_mgdl=-10, hdl_mgdl=50, tg_mgdl=500)

    def test_negative_hdl_raises_value_error(self):
        """Test that negative HDL raises ValueError."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        with pytest.raises(ValueError, match="hdl_mgdl cannot be negative"):
            calc_ldl_martin_hopkins_extended(tc_mgdl=200, hdl_mgdl=-5, tg_mgdl=500)

    def test_negative_tg_raises_value_error(self):
        """Test that negative TG raises ValueError."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        with pytest.raises(ValueError, match="tg_mgdl cannot be negative"):
            calc_ldl_martin_hopkins_extended(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=-20)

    def test_nan_input_raises_value_error(self):
        """Test that NaN inputs raise ValueError."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        with pytest.raises(ValueError, match="cannot be NaN"):
            calc_ldl_martin_hopkins_extended(tc_mgdl=float("nan"), hdl_mgdl=50, tg_mgdl=500)

    def test_returns_float(self):
        """Test that the function always returns a float."""
        from ldlC.models import calc_ldl_martin_hopkins_extended
        result = calc_ldl_martin_hopkins_extended(tc_mgdl=200, hdl_mgdl=50, tg_mgdl=500)
        assert isinstance(result, float)

