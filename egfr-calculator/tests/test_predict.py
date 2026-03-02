"""Tests for eGFR/predict.py — prediction API."""

from __future__ import annotations

import math
import warnings

import pytest

from eGFR.predict import predict_egfr


# ===================================================================
# CKD-EPI 2021 method
# ===================================================================

class TestPredictCKDEPI2021:
    """Tests for method='ckd_epi_2021'."""

    def test_returns_dict_with_expected_keys(self):
        result = predict_egfr(1.0, 50, "M")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"egfr_pred", "ckd_stage", "method", "warning"}

    def test_method_field(self):
        result = predict_egfr(1.0, 50, "M")
        assert result["method"] == "ckd_epi_2021"

    def test_known_value_50m_scr1(self):
        result = predict_egfr(1.0, 50, "M", method="ckd_epi_2021")
        assert result["egfr_pred"] == pytest.approx(92, abs=3)
        assert result["ckd_stage"] == "G1"

    def test_known_value_70m_scr1_5(self):
        result = predict_egfr(1.5, 70, "M", method="ckd_epi_2021")
        assert result["egfr_pred"] == pytest.approx(50, abs=5)

    def test_no_warning_for_ckdepi(self):
        result = predict_egfr(1.0, 50, "M")
        assert result["warning"] is None

    def test_accepts_nhanes_sex_coding(self):
        result = predict_egfr(0.8, 50, 2, method="ckd_epi_2021")
        assert result["egfr_pred"] > 0

    def test_ckd_stage_classification(self):
        # Low eGFR should be higher stage
        result = predict_egfr(3.0, 70, "M")
        assert result["ckd_stage"] in ("G4", "G5")


# ===================================================================
# MDRD method
# ===================================================================

class TestPredictMDRD:
    """Tests for method='mdrd'."""

    def test_returns_correct_method(self):
        result = predict_egfr(1.0, 50, "M", method="mdrd")
        assert result["method"] == "mdrd"

    def test_known_value(self):
        result = predict_egfr(1.0, 50, "M", method="mdrd")
        assert result["egfr_pred"] == pytest.approx(79.1, abs=3)

    def test_warning_when_egfr_above_60(self):
        result = predict_egfr(1.0, 50, "M", method="mdrd")
        assert result["warning"] is not None
        assert "60" in result["warning"] or "less accurate" in result["warning"].lower()

    def test_no_warning_when_egfr_below_60(self):
        result = predict_egfr(2.5, 70, "M", method="mdrd")
        # MDRD < 60 should not trigger warning
        assert result["warning"] is None


# ===================================================================
# Cockcroft-Gault method
# ===================================================================

class TestPredictCockcroftGault:
    """Tests for method='cockcroft_gault'."""

    def test_returns_correct_method(self):
        result = predict_egfr(1.0, 70, "M", weight_kg=70, method="cockcroft_gault")
        assert result["method"] == "cockcroft_gault"

    def test_known_value_70m_70kg(self):
        result = predict_egfr(1.0, 70, "M", weight_kg=70, method="cockcroft_gault")
        assert result["egfr_pred"] == pytest.approx(68, abs=2)

    def test_crcl_warning(self):
        result = predict_egfr(1.0, 50, "M", weight_kg=80, method="cockcroft_gault")
        assert result["warning"] is not None
        assert "CrCl" in result["warning"] or "creatinine clearance" in result["warning"].lower()

    def test_raises_without_weight(self):
        with pytest.raises(ValueError, match="weight_kg"):
            predict_egfr(1.0, 50, "M", method="cockcroft_gault")


# ===================================================================
# Hybrid method
# ===================================================================

class TestPredictHybrid:
    """Tests for method='hybrid'."""

    def test_returns_correct_method(self):
        result = predict_egfr(1.0, 50, "M", method="hybrid")
        assert result["method"] == "hybrid"

    def test_returns_prediction(self):
        """Hybrid should return a prediction (model or fallback)."""
        result = predict_egfr(1.0, 50, "M", method="hybrid")
        assert result["egfr_pred"] > 0

    def test_warning_present(self):
        """Hybrid should always include some warning (incomplete inputs or fallback)."""
        result = predict_egfr(1.0, 50, "M", method="hybrid")
        assert result["warning"] is not None

    def test_with_all_inputs(self):
        """With all inputs, hybrid should produce a reasonable prediction."""
        result = predict_egfr(
            1.0, 50, "M", weight_kg=80, height_cm=175,
            cystatin_c=0.9, method="hybrid"
        )
        assert result["egfr_pred"] > 0

    def test_fallback_produces_reasonable_egfr(self):
        result = predict_egfr(1.0, 50, "M", weight_kg=80, method="hybrid")
        assert 50 < result["egfr_pred"] < 130

    def test_ckd_stage_present(self):
        result = predict_egfr(1.0, 50, "M", method="hybrid")
        assert result["ckd_stage"] in ("G1", "G2", "G3a", "G3b", "G4", "G5")


# ===================================================================
# Input validation
# ===================================================================

class TestPredictValidation:
    """Tests for input validation."""

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            predict_egfr(1.0, 50, "M", method="invalid_method")

    def test_negative_creatinine(self):
        with pytest.raises(ValueError, match="cr_mgdl"):
            predict_egfr(-1.0, 50, "M")

    def test_nan_creatinine(self):
        with pytest.raises(ValueError):
            predict_egfr(float("nan"), 50, "M")

    def test_age_below_18(self):
        with pytest.raises(ValueError, match="age"):
            predict_egfr(1.0, 10, "M")

    def test_invalid_sex(self):
        with pytest.raises(ValueError, match="sex"):
            predict_egfr(1.0, 50, "X")

    def test_method_whitespace_stripped(self):
        """Ensure leading/trailing whitespace on method is handled."""
        result = predict_egfr(1.0, 50, "M", method="  ckd_epi_2021  ")
        assert result["method"] == "ckd_epi_2021"

    def test_method_case_insensitive(self):
        result = predict_egfr(1.0, 50, "M", method="CKD_EPI_2021")
        assert result["method"] == "ckd_epi_2021"


# ===================================================================
# Return structure
# ===================================================================

class TestPredictReturnStructure:
    """Tests verifying the return dict structure across all methods."""

    @pytest.mark.parametrize("method", ["ckd_epi_2021", "mdrd"])
    def test_egfr_pred_is_float(self, method):
        result = predict_egfr(1.0, 50, "M", method=method)
        assert isinstance(result["egfr_pred"], float)

    def test_egfr_pred_is_float_cg(self):
        result = predict_egfr(1.0, 50, "M", weight_kg=80, method="cockcroft_gault")
        assert isinstance(result["egfr_pred"], float)

    def test_ckd_stage_is_string(self):
        result = predict_egfr(1.0, 50, "M")
        assert isinstance(result["ckd_stage"], str)

    def test_warning_is_none_or_string(self):
        result = predict_egfr(1.0, 50, "M")
        assert result["warning"] is None or isinstance(result["warning"], str)
