"""
Unit tests for hba1cE.utils module.

Tests unit conversion functions for glucose and HbA1c.
"""

import pytest
import numpy as np
from hba1cE.utils import (
    mg_dl_to_mmol_l,
    mmol_l_to_mg_dl,
    percent_to_mmol_mol,
    mmol_mol_to_percent,
)


class TestGlucoseConversions:
    """Tests for glucose unit conversions between mg/dL and mmol/L."""

    def test_mg_dl_to_mmol_l_normal_value(self):
        """Test conversion of normal fasting glucose (100 mg/dL)."""
        result = mg_dl_to_mmol_l(100)
        expected = 100 / 18.018
        assert pytest.approx(result, rel=1e-3) == expected

    def test_mg_dl_to_mmol_l_diabetic_threshold(self):
        """Test conversion at diabetic threshold (126 mg/dL ≈ 7.0 mmol/L)."""
        result = mg_dl_to_mmol_l(126)
        assert pytest.approx(result, abs=0.1) == 7.0

    def test_mmol_l_to_mg_dl_normal_value(self):
        """Test conversion of normal fasting glucose (5.5 mmol/L)."""
        result = mmol_l_to_mg_dl(5.5)
        expected = 5.5 * 18.018
        assert pytest.approx(result, rel=1e-3) == expected

    def test_mmol_l_to_mg_dl_diabetic_threshold(self):
        """Test conversion at diabetic threshold (7.0 mmol/L ≈ 126 mg/dL)."""
        result = mmol_l_to_mg_dl(7.0)
        assert pytest.approx(result, abs=1.0) == 126

    def test_glucose_round_trip_conversion(self):
        """Test that converting mg/dL → mmol/L → mg/dL returns original value."""
        original = 150.0
        converted = mmol_l_to_mg_dl(mg_dl_to_mmol_l(original))
        assert pytest.approx(converted, rel=1e-9) == original

    def test_glucose_array_conversion(self):
        """Test that array inputs work correctly."""
        values = np.array([80, 100, 126, 200])
        result = mg_dl_to_mmol_l(values)
        expected = values / 18.018
        np.testing.assert_allclose(result, expected, rtol=1e-9)


class TestHbA1cConversions:
    """Tests for HbA1c unit conversions between NGSP (%) and IFCC (mmol/mol)."""

    def test_percent_to_mmol_mol_diabetes_threshold(self):
        """Test conversion at diabetes threshold (6.5% ≈ 48 mmol/mol)."""
        result = percent_to_mmol_mol(6.5)
        expected = (6.5 - 2.15) * 10.929
        assert pytest.approx(result, rel=1e-3) == expected
        # Should be close to 48 mmol/mol
        assert pytest.approx(result, abs=0.5) == 47.5

    def test_percent_to_mmol_mol_prediabetes_threshold(self):
        """Test conversion at prediabetes threshold (5.7% ≈ 39 mmol/mol)."""
        result = percent_to_mmol_mol(5.7)
        expected = (5.7 - 2.15) * 10.929
        assert pytest.approx(result, rel=1e-3) == expected
        # Should be close to 39 mmol/mol
        assert pytest.approx(result, abs=0.5) == 38.8

    def test_mmol_mol_to_percent_diabetes_threshold(self):
        """Test conversion at diabetes threshold (48 mmol/mol ≈ 6.5%)."""
        result = mmol_mol_to_percent(48)
        expected = (48 / 10.929) + 2.15
        assert pytest.approx(result, rel=1e-3) == expected
        # Should be close to 6.5%
        assert pytest.approx(result, abs=0.1) == 6.54

    def test_mmol_mol_to_percent_prediabetes_threshold(self):
        """Test conversion at prediabetes threshold (39 mmol/mol ≈ 5.7%)."""
        result = mmol_mol_to_percent(39)
        expected = (39 / 10.929) + 2.15
        assert pytest.approx(result, rel=1e-3) == expected
        # Should be close to 5.7%
        assert pytest.approx(result, abs=0.1) == 5.72

    def test_hba1c_round_trip_conversion(self):
        """Test that converting % → mmol/mol → % returns original value."""
        original = 7.5
        converted = mmol_mol_to_percent(percent_to_mmol_mol(original))
        assert pytest.approx(converted, rel=1e-9) == original

    def test_hba1c_array_conversion(self):
        """Test that array inputs work correctly."""
        values = np.array([5.0, 5.7, 6.5, 8.0, 10.0])
        result = percent_to_mmol_mol(values)
        expected = (values - 2.15) * 10.929
        np.testing.assert_allclose(result, expected, rtol=1e-9)


class TestEdgeCases:
    """Tests for edge cases and special values."""

    def test_glucose_zero(self):
        """Test that zero glucose converts to zero."""
        assert mg_dl_to_mmol_l(0) == 0
        assert mmol_l_to_mg_dl(0) == 0

    def test_hba1c_very_low(self):
        """Test conversion of very low HbA1c (3%)."""
        result = percent_to_mmol_mol(3.0)
        assert result > 0  # Should still be positive

    def test_hba1c_very_high(self):
        """Test conversion of very high HbA1c (15%)."""
        result = percent_to_mmol_mol(15.0)
        expected = (15.0 - 2.15) * 10.929
        assert pytest.approx(result, rel=1e-3) == expected
