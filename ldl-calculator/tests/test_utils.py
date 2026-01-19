"""
Unit tests for ldlC.utils module.
"""

import pytest
from ldlC.utils import (
    mg_dl_to_mmol_l,
    mmol_l_to_mg_dl,
    CHOLESTEROL_CONVERSION_FACTOR,
    TRIGLYCERIDE_CONVERSION_FACTOR,
)


class TestMgDlToMmolL:
    """Tests for mg_dl_to_mmol_l function."""

    def test_cholesterol_basic_conversion(self):
        """Test basic cholesterol conversion from mg/dL to mmol/L."""
        # 200 mg/dL cholesterol ≈ 5.17 mmol/L
        result = mg_dl_to_mmol_l(200, "cholesterol")
        expected = 200 / CHOLESTEROL_CONVERSION_FACTOR
        assert abs(result - expected) < 0.001

    def test_cholesterol_default_molecule(self):
        """Test that cholesterol is the default molecule type."""
        result_explicit = mg_dl_to_mmol_l(200, "cholesterol")
        result_default = mg_dl_to_mmol_l(200)
        assert result_explicit == result_default

    def test_triglycerides_basic_conversion(self):
        """Test basic triglycerides conversion from mg/dL to mmol/L."""
        # 150 mg/dL triglycerides ≈ 1.69 mmol/L
        result = mg_dl_to_mmol_l(150, "triglycerides")
        expected = 150 / TRIGLYCERIDE_CONVERSION_FACTOR
        assert abs(result - expected) < 0.001

    def test_triglycerides_aliases(self):
        """Test that different triglyceride aliases work."""
        value = 150
        result_full = mg_dl_to_mmol_l(value, "triglycerides")
        result_singular = mg_dl_to_mmol_l(value, "triglyceride")
        result_abbrev = mg_dl_to_mmol_l(value, "tg")

        assert result_full == result_singular == result_abbrev

    def test_case_insensitive(self):
        """Test that molecule type is case-insensitive."""
        value = 200
        result_lower = mg_dl_to_mmol_l(value, "cholesterol")
        result_upper = mg_dl_to_mmol_l(value, "CHOLESTEROL")
        result_mixed = mg_dl_to_mmol_l(value, "Cholesterol")

        assert result_lower == result_upper == result_mixed

    def test_zero_value(self):
        """Test conversion of zero value."""
        assert mg_dl_to_mmol_l(0, "cholesterol") == 0
        assert mg_dl_to_mmol_l(0, "triglycerides") == 0

    def test_unknown_molecule_raises_error(self):
        """Test that unknown molecule type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown molecule type"):
            mg_dl_to_mmol_l(100, "protein")


class TestMmolLToMgDl:
    """Tests for mmol_l_to_mg_dl function."""

    def test_cholesterol_basic_conversion(self):
        """Test basic cholesterol conversion from mmol/L to mg/dL."""
        # 5.17 mmol/L cholesterol ≈ 200 mg/dL
        result = mmol_l_to_mg_dl(5.17, "cholesterol")
        expected = 5.17 * CHOLESTEROL_CONVERSION_FACTOR
        assert abs(result - expected) < 0.001

    def test_cholesterol_default_molecule(self):
        """Test that cholesterol is the default molecule type."""
        result_explicit = mmol_l_to_mg_dl(5.17, "cholesterol")
        result_default = mmol_l_to_mg_dl(5.17)
        assert result_explicit == result_default

    def test_triglycerides_basic_conversion(self):
        """Test basic triglycerides conversion from mmol/L to mg/dL."""
        # 1.69 mmol/L triglycerides ≈ 150 mg/dL
        result = mmol_l_to_mg_dl(1.69, "triglycerides")
        expected = 1.69 * TRIGLYCERIDE_CONVERSION_FACTOR
        assert abs(result - expected) < 0.001

    def test_triglycerides_aliases(self):
        """Test that different triglyceride aliases work."""
        value = 1.69
        result_full = mmol_l_to_mg_dl(value, "triglycerides")
        result_singular = mmol_l_to_mg_dl(value, "triglyceride")
        result_abbrev = mmol_l_to_mg_dl(value, "tg")

        assert result_full == result_singular == result_abbrev

    def test_case_insensitive(self):
        """Test that molecule type is case-insensitive."""
        value = 5.17
        result_lower = mmol_l_to_mg_dl(value, "cholesterol")
        result_upper = mmol_l_to_mg_dl(value, "CHOLESTEROL")
        result_mixed = mmol_l_to_mg_dl(value, "Cholesterol")

        assert result_lower == result_upper == result_mixed

    def test_zero_value(self):
        """Test conversion of zero value."""
        assert mmol_l_to_mg_dl(0, "cholesterol") == 0
        assert mmol_l_to_mg_dl(0, "triglycerides") == 0

    def test_unknown_molecule_raises_error(self):
        """Test that unknown molecule type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown molecule type"):
            mmol_l_to_mg_dl(5.0, "protein")


class TestRoundTrip:
    """Tests for round-trip conversions."""

    def test_cholesterol_round_trip(self):
        """Test that mg/dL -> mmol/L -> mg/dL gives original value."""
        original = 200.0
        mmol = mg_dl_to_mmol_l(original, "cholesterol")
        back_to_mg = mmol_l_to_mg_dl(mmol, "cholesterol")
        assert abs(back_to_mg - original) < 0.001

    def test_triglycerides_round_trip(self):
        """Test that mg/dL -> mmol/L -> mg/dL gives original value."""
        original = 150.0
        mmol = mg_dl_to_mmol_l(original, "triglycerides")
        back_to_mg = mmol_l_to_mg_dl(mmol, "triglycerides")
        assert abs(back_to_mg - original) < 0.001


class TestKnownValues:
    """Tests with clinically relevant known values."""

    def test_cholesterol_known_values(self):
        """Test cholesterol conversions with known clinical values."""
        # Total cholesterol 200 mg/dL is a common reference
        # Should be approximately 5.17 mmol/L
        result = mg_dl_to_mmol_l(200, "cholesterol")
        assert 5.1 < result < 5.3  # Reasonable clinical range

    def test_triglycerides_known_values(self):
        """Test triglycerides conversions with known clinical values."""
        # TG 150 mg/dL is borderline high
        # Should be approximately 1.69 mmol/L
        result = mg_dl_to_mmol_l(150, "triglycerides")
        assert 1.6 < result < 1.8  # Reasonable clinical range
