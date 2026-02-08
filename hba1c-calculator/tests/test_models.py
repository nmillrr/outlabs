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

from hba1cE.models import calc_hba1c_adag


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
