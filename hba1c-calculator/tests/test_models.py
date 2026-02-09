"""
Tests for hba1cE.models module.

Contains unit tests for HbA1c estimation models including:
- ADAG equation tests
- Glycation kinetics model tests
- Multi-linear regression tests
"""

import math
import pytest
import numpy as np

from hba1cE.models import (
    calc_hba1c_adag,
    calc_hba1c_kinetic,
    calc_hba1c_regression,
    fit_regression_coefficients,
    DEFAULT_REGRESSION_COEFFICIENTS,
)


class TestCalcHba1cAdag:
    """Tests for the ADAG-derived HbA1c estimation function."""

    def test_fpg_126_returns_approximately_6_percent(self):
        """FPG=126 mg/dL should give eHbA1c ≈ 6.0%."""
        result = calc_hba1c_adag(126.0)
        assert abs(result - 6.0) < 0.1, f"Expected ~6.0%, got {result}%"

    def test_fpg_154_returns_approximately_7_percent(self):
        """FPG=154 mg/dL should give eHbA1c ≈ 7.0%."""
        result = calc_hba1c_adag(154.0)
        assert abs(result - 7.0) < 0.1, f"Expected ~7.0%, got {result}%"

    def test_negative_input_raises_value_error(self):
        """Negative FPG should raise ValueError."""
        with pytest.raises(ValueError, match="negative"):
            calc_hba1c_adag(-50.0)

    def test_nan_input_raises_value_error(self):
        """NaN FPG should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            calc_hba1c_adag(float('nan'))

    def test_fpg_below_40_raises_value_error(self):
        """FPG below 40 mg/dL (physiologically implausible) should raise ValueError."""
        with pytest.raises(ValueError, match="below 40"):
            calc_hba1c_adag(35.0)

    def test_array_input_returns_array(self):
        """Numpy array input should return corresponding array of estimates."""
        fpg_values = np.array([126.0, 154.0, 100.0])
        result = calc_hba1c_adag(fpg_values)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert abs(result[0] - 6.0) < 0.1
        assert abs(result[1] - 7.0) < 0.1

    def test_array_with_nan_raises_value_error(self):
        """Numpy array containing NaN should raise ValueError."""
        fpg_values = np.array([126.0, np.nan, 154.0])
        with pytest.raises(ValueError, match="NaN"):
            calc_hba1c_adag(fpg_values)

    def test_array_with_negative_raises_value_error(self):
        """Numpy array containing negative values should raise ValueError."""
        fpg_values = np.array([126.0, -10.0, 154.0])
        with pytest.raises(ValueError, match="negative"):
            calc_hba1c_adag(fpg_values)


class TestCalcHba1cKinetic:
    """Tests for the glycation kinetics HbA1c estimation function."""

    def test_normal_fpg_returns_valid_hba1c(self):
        """Normal FPG=100 mg/dL with default params should return valid HbA1c."""
        result = calc_hba1c_kinetic(100.0)
        # HbA1c should be in physiological range (3-20%)
        assert 3.0 < result < 20.0, f"Expected 3-20%, got {result}%"
        # Normal FPG should give low normal HbA1c
        assert result < 6.0, f"Expected < 6.0% for normal FPG, got {result}%"

    def test_diabetic_fpg_returns_higher_hba1c(self):
        """Diabetic FPG=200 mg/dL should return higher HbA1c."""
        result = calc_hba1c_kinetic(200.0)
        assert 3.0 < result < 20.0, f"Expected 3-20%, got {result}%"
        # Higher FPG should give higher HbA1c
        assert result > calc_hba1c_kinetic(100.0), "Higher FPG should give higher HbA1c"

    def test_anemia_increases_hba1c(self):
        """Lower hemoglobin (anemia) should increase estimated HbA1c."""
        normal_hgb = calc_hba1c_kinetic(126.0, hgb_gdl=14.0)
        anemia_hgb = calc_hba1c_kinetic(126.0, hgb_gdl=10.0)
        assert anemia_hgb > normal_hgb, "Anemia should increase HbA1c estimate"

    def test_shorter_rbc_lifespan_decreases_hba1c(self):
        """Shorter RBC lifespan should decrease estimated HbA1c."""
        normal_lifespan = calc_hba1c_kinetic(200.0, rbc_lifespan_days=120)
        short_lifespan = calc_hba1c_kinetic(200.0, rbc_lifespan_days=60)
        assert short_lifespan < normal_lifespan, "Shorter lifespan should decrease HbA1c"

    def test_output_in_valid_range(self):
        """Output should always be in physiological range (3% < HbA1c < 20%)."""
        # Test various inputs
        test_cases = [
            (40.0, 14.0, 120),   # Minimum FPG
            (100.0, 14.0, 120),  # Normal FPG
            (200.0, 14.0, 120),  # High FPG
            (300.0, 14.0, 120),  # Very high FPG
            (100.0, 10.0, 120),  # Anemia
            (100.0, 14.0, 60),   # Short lifespan
        ]
        for fpg, hgb, lifespan in test_cases:
            result = calc_hba1c_kinetic(fpg, hgb, lifespan)
            assert 3.0 < result < 20.0, f"FPG={fpg}, Hgb={hgb}, lifespan={lifespan} gave {result}%"

    def test_negative_fpg_raises_value_error(self):
        """Negative FPG should raise ValueError."""
        with pytest.raises(ValueError, match="negative"):
            calc_hba1c_kinetic(-50.0)

    def test_nan_fpg_raises_value_error(self):
        """NaN FPG should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            calc_hba1c_kinetic(float('nan'))

    def test_fpg_below_40_raises_value_error(self):
        """FPG below 40 mg/dL should raise ValueError."""
        with pytest.raises(ValueError, match="below 40"):
            calc_hba1c_kinetic(35.0)

    def test_invalid_hemoglobin_raises_value_error(self):
        """Hemoglobin outside range (5-25 g/dL) should raise ValueError."""
        with pytest.raises(ValueError, match="valid range"):
            calc_hba1c_kinetic(100.0, hgb_gdl=3.0)
        with pytest.raises(ValueError, match="valid range"):
            calc_hba1c_kinetic(100.0, hgb_gdl=30.0)

    def test_negative_lifespan_raises_value_error(self):
        """Negative RBC lifespan should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calc_hba1c_kinetic(100.0, rbc_lifespan_days=-10)

    def test_array_input_returns_array(self):
        """Numpy array input should return corresponding array of estimates."""
        fpg_values = np.array([100.0, 150.0, 200.0])
        result = calc_hba1c_kinetic(fpg_values)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        # All values should be in valid range
        assert all(3.0 < r < 20.0 for r in result)
        # Values should increase with FPG
        assert result[0] < result[1] < result[2]


class TestCalcHba1cRegression:
    """Tests for the multi-linear regression HbA1c estimation function."""

    def test_returns_valid_hba1c_with_default_coefficients(self):
        """Function should return valid HbA1c with default coefficients."""
        result = calc_hba1c_regression(126.0, 50, 150, 50, 14.0)
        # HbA1c should be in physiological range
        assert 3.0 < result < 20.0, f"Expected 3-20%, got {result}%"

    def test_higher_fpg_gives_higher_hba1c(self):
        """Higher FPG should increase estimated HbA1c."""
        low_fpg = calc_hba1c_regression(100.0, 50, 150, 50, 14.0)
        high_fpg = calc_hba1c_regression(200.0, 50, 150, 50, 14.0)
        assert high_fpg > low_fpg, "Higher FPG should give higher HbA1c"

    def test_older_age_gives_higher_hba1c(self):
        """Older age should increase estimated HbA1c (with default coefficients)."""
        young = calc_hba1c_regression(126.0, 30, 150, 50, 14.0)
        old = calc_hba1c_regression(126.0, 70, 150, 50, 14.0)
        assert old > young, "Older age should give higher HbA1c"

    def test_custom_coefficients_work(self):
        """Custom coefficients should be used when provided."""
        custom_coeffs = {
            "intercept": 5.0,
            "fpg": 0.01,
            "age": 0.0,
            "tg": 0.0,
            "hdl": 0.0,
            "hgb": 0.0,
        }
        result = calc_hba1c_regression(100.0, 50, 150, 50, 14.0, coefficients=custom_coeffs)
        expected = 5.0 + 0.01 * 100.0  # 6.0
        assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"

    def test_negative_fpg_raises_value_error(self):
        """Negative FPG should raise ValueError."""
        with pytest.raises(ValueError, match="negative"):
            calc_hba1c_regression(-50.0, 50, 150, 50, 14.0)

    def test_nan_fpg_raises_value_error(self):
        """NaN FPG should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            calc_hba1c_regression(float('nan'), 50, 150, 50, 14.0)

    def test_fpg_below_40_raises_value_error(self):
        """FPG below 40 mg/dL should raise ValueError."""
        with pytest.raises(ValueError, match="below 40"):
            calc_hba1c_regression(35.0, 50, 150, 50, 14.0)

    def test_negative_age_raises_value_error(self):
        """Negative age should raise ValueError."""
        with pytest.raises(ValueError, match="negative"):
            calc_hba1c_regression(126.0, -5, 150, 50, 14.0)

    def test_nan_tg_raises_value_error(self):
        """NaN triglycerides should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            calc_hba1c_regression(126.0, 50, float('nan'), 50, 14.0)

    def test_negative_hemoglobin_raises_value_error(self):
        """Negative hemoglobin should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calc_hba1c_regression(126.0, 50, 150, 50, -1.0)

    def test_array_input_returns_array(self):
        """Numpy array input should return corresponding array of estimates."""
        fpg_values = np.array([100.0, 150.0, 200.0])
        age_values = np.array([40, 50, 60])
        tg_values = np.array([100.0, 150.0, 200.0])
        hdl_values = np.array([50.0, 50.0, 50.0])
        hgb_values = np.array([14.0, 14.0, 14.0])
        result = calc_hba1c_regression(fpg_values, age_values, tg_values, hdl_values, hgb_values)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        # Values should increase with FPG (all other factors also increase)
        assert result[0] < result[1] < result[2]


class TestFitRegressionCoefficients:
    """Tests for the fit_regression_coefficients function."""

    def test_returns_dict_with_all_keys(self):
        """Should return dictionary with all coefficient keys."""
        import pandas as pd
        # Create mock data
        df = pd.DataFrame({
            'hba1c_percent': [5.5, 6.0, 6.5, 7.0, 7.5],
            'fpg_mgdl': [100.0, 110.0, 120.0, 140.0, 160.0],
            'age_years': [40, 45, 50, 55, 60],
            'tg_mgdl': [100.0, 120.0, 140.0, 160.0, 180.0],
            'hdl_mgdl': [60.0, 55.0, 50.0, 45.0, 40.0],
            'hgb_gdl': [14.0, 13.5, 13.0, 12.5, 12.0],
        })
        result = fit_regression_coefficients(df)
        expected_keys = ['intercept', 'fpg', 'age', 'tg', 'hdl', 'hgb']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_raises_value_error_for_missing_columns(self):
        """Should raise ValueError if required columns are missing."""
        import pandas as pd
        df = pd.DataFrame({
            'fpg_mgdl': [100.0, 110.0],
            'age_years': [40, 45],
        })  # Missing hba1c_percent, tg_mgdl, hdl_mgdl, hgb_gdl
        with pytest.raises(ValueError, match="Missing required columns"):
            fit_regression_coefficients(df)

    def test_coefficients_are_floats(self):
        """All coefficients should be float values."""
        import pandas as pd
        df = pd.DataFrame({
            'hba1c_percent': [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
            'fpg_mgdl': [100.0, 110.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0],
            'age_years': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
            'tg_mgdl': [100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0],
            'hdl_mgdl': [60.0, 55.0, 50.0, 45.0, 40.0, 38.0, 36.0, 34.0, 32.0, 30.0],
            'hgb_gdl': [14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0, 10.5, 10.0, 9.5],
        })
        result = fit_regression_coefficients(df)
        for key, value in result.items():
            assert isinstance(value, float), f"Coefficient {key} should be float, got {type(value)}"

