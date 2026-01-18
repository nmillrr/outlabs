"""
Unit tests for freeT/utils.py conversion functions.
"""

import pytest
from freeT.utils import (
    ng_dl_to_nmol_l,
    nmol_l_to_ng_dl,
    mg_dl_to_g_l,
    g_l_to_mg_dl,
)


class TestTestosteroneConversions:
    """Tests for testosterone unit conversions."""

    def test_ng_dl_to_nmol_l_typical_value(self):
        """Test conversion of typical testosterone value (500 ng/dL)."""
        # 500 ng/dL should be approximately 17.35 nmol/L
        result = ng_dl_to_nmol_l(500)
        assert abs(result - 17.35) < 0.1

    def test_ng_dl_to_nmol_l_low_value(self):
        """Test conversion of low testosterone value."""
        result = ng_dl_to_nmol_l(200)
        assert abs(result - 6.94) < 0.1

    def test_ng_dl_to_nmol_l_zero(self):
        """Test conversion of zero."""
        assert ng_dl_to_nmol_l(0) == 0

    def test_nmol_l_to_ng_dl_typical_value(self):
        """Test reverse conversion (nmol/L to ng/dL)."""
        # 17.35 nmol/L should be approximately 500 ng/dL
        result = nmol_l_to_ng_dl(17.35)
        assert abs(result - 500) < 5

    def test_roundtrip_ng_dl(self):
        """Test roundtrip conversion preserves value."""
        original = 432.5
        converted = ng_dl_to_nmol_l(original)
        back = nmol_l_to_ng_dl(converted)
        assert abs(back - original) < 0.01


class TestAlbuminConversions:
    """Tests for albumin unit conversions."""

    def test_mg_dl_to_g_l_typical_value(self):
        """Test conversion of typical albumin value (4000 mg/dL = 40 g/L)."""
        result = mg_dl_to_g_l(4000)
        assert result == 40.0

    def test_g_l_to_mg_dl_typical_value(self):
        """Test reverse conversion (g/L to mg/dL)."""
        result = g_l_to_mg_dl(45)
        assert result == 4500

    def test_mg_dl_to_g_l_zero(self):
        """Test conversion of zero."""
        assert mg_dl_to_g_l(0) == 0

    def test_roundtrip_albumin(self):
        """Test roundtrip conversion preserves value."""
        original = 4250.5
        converted = mg_dl_to_g_l(original)
        back = g_l_to_mg_dl(converted)
        assert abs(back - original) < 0.01
