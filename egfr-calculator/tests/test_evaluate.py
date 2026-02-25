"""Tests for eGFR/evaluate.py — Bland-Altman, P30/P10, evaluate_model."""

import math

import numpy as np
import pytest

from eGFR.evaluate import bland_altman_stats, p30_accuracy, p10_accuracy, evaluate_model


# ── Known-value tests ───────────────────────────────────────────────────

class TestBlandAltmanKnownValues:
    """Verify Bland-Altman statistics against hand-calculated values."""

    def test_perfect_agreement(self):
        """When y_pred == y_true, bias and std should be 0."""
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = bland_altman_stats(y, y)
        assert result["mean_bias"] == pytest.approx(0.0)
        assert result["std_diff"] == pytest.approx(0.0)
        assert result["loa_lower"] == pytest.approx(0.0)
        assert result["loa_upper"] == pytest.approx(0.0)

    def test_constant_positive_bias(self):
        """Constant +5 offset → bias = 5, std = 0."""
        y_true = [10.0, 20.0, 30.0, 40.0, 50.0]
        y_pred = [15.0, 25.0, 35.0, 45.0, 55.0]
        result = bland_altman_stats(y_true, y_pred)
        assert result["mean_bias"] == pytest.approx(5.0)
        assert result["std_diff"] == pytest.approx(0.0)
        assert result["loa_lower"] == pytest.approx(5.0)
        assert result["loa_upper"] == pytest.approx(5.0)

    def test_constant_negative_bias(self):
        """Constant -3 offset → bias = -3, std = 0."""
        y_true = [10.0, 20.0, 30.0]
        y_pred = [7.0, 17.0, 27.0]
        result = bland_altman_stats(y_true, y_pred)
        assert result["mean_bias"] == pytest.approx(-3.0)
        assert result["std_diff"] == pytest.approx(0.0)

    def test_hand_calculated_values(self):
        """Verify against manually computed statistics.

        y_true = [100, 90, 80, 70]
        y_pred = [102, 88, 83, 69]
        diffs  = [  2, -2,  3, -1]
        mean_bias = 0.5
        std_diff  = sqrt(sum((d - 0.5)^2 for d in diffs) / 3)
                  = sqrt((2.25 + 6.25 + 6.25 + 2.25) / 3)
                  = sqrt(17.0 / 3)  ≈ 2.3805
        loa_lower = 0.5 - 1.96 * 2.3805 ≈ -4.166
        loa_upper = 0.5 + 1.96 * 2.3805 ≈  5.166
        """
        y_true = [100.0, 90.0, 80.0, 70.0]
        y_pred = [102.0, 88.0, 83.0, 69.0]
        result = bland_altman_stats(y_true, y_pred)

        assert result["mean_bias"] == pytest.approx(0.5, abs=1e-6)

        expected_std = math.sqrt(17.0 / 3)
        assert result["std_diff"] == pytest.approx(expected_std, abs=1e-6)

        expected_loa_lower = 0.5 - 1.96 * expected_std
        expected_loa_upper = 0.5 + 1.96 * expected_std
        assert result["loa_lower"] == pytest.approx(expected_loa_lower, abs=1e-4)
        assert result["loa_upper"] == pytest.approx(expected_loa_upper, abs=1e-4)

    def test_loa_symmetry_around_bias(self):
        """LOA should be symmetric around mean_bias."""
        y_true = np.array([50, 60, 70, 80, 90, 100])
        y_pred = np.array([52, 58, 73, 79, 92, 98])
        result = bland_altman_stats(y_true, y_pred)

        midpoint = (result["loa_lower"] + result["loa_upper"]) / 2
        assert midpoint == pytest.approx(result["mean_bias"], abs=1e-10)


# ── Return-type tests ──────────────────────────────────────────────────

class TestBlandAltmanReturnType:
    """Verify return structure and types."""

    def test_returns_dict(self):
        result = bland_altman_stats([1, 2, 3], [1, 2, 3])
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = bland_altman_stats([1, 2, 3], [1, 2, 3])
        for key in ("mean_bias", "std_diff", "loa_lower", "loa_upper"):
            assert key in result

    def test_values_are_float(self):
        result = bland_altman_stats([1, 2, 3], [4, 5, 6])
        for val in result.values():
            assert isinstance(val, float)


# ── Input-handling tests ────────────────────────────────────────────────

class TestBlandAltmanInputs:
    """Verify accepts various input types."""

    def test_accepts_lists(self):
        result = bland_altman_stats([1, 2, 3], [1, 2, 3])
        assert result["mean_bias"] == pytest.approx(0.0)

    def test_accepts_numpy_arrays(self):
        result = bland_altman_stats(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
        )
        assert result["mean_bias"] == pytest.approx(1.0)

    def test_accepts_single_element(self):
        result = bland_altman_stats([10.0], [12.0])
        assert result["mean_bias"] == pytest.approx(2.0)


# ── Error-handling tests ───────────────────────────────────────────────

class TestBlandAltmanErrors:
    """Verify proper error raising."""

    def test_empty_arrays(self):
        with pytest.raises(ValueError, match="empty"):
            bland_altman_stats([], [])

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="[Ss]hape"):
            bland_altman_stats([1, 2, 3], [1, 2])

    def test_nan_in_y_true(self):
        with pytest.raises(ValueError, match="NaN"):
            bland_altman_stats([1, float("nan"), 3], [1, 2, 3])

    def test_nan_in_y_pred(self):
        with pytest.raises(ValueError, match="NaN"):
            bland_altman_stats([1, 2, 3], [1, float("nan"), 3])


# ═══════════════════════════════════════════════════════════════════════
# P30 / P10 Accuracy
# ═══════════════════════════════════════════════════════════════════════


class TestP30KnownValues:
    """Verify P30 accuracy against hand-calculated values."""

    def test_perfect_predictions(self):
        """All predictions exact → P30 = 100 %."""
        y_true = [60.0, 90.0, 120.0]
        assert p30_accuracy(y_true, y_true) == pytest.approx(100.0)

    def test_all_within_30pct(self):
        """All predictions within ±30 % → P30 = 100 %."""
        y_true = [100.0, 100.0, 100.0]
        y_pred = [80.0, 120.0, 130.0]  # -20%, +20%, +30%
        assert p30_accuracy(y_true, y_pred) == pytest.approx(100.0)

    def test_some_outside_30pct(self):
        """2 of 4 within ±30 % → P30 = 50 %."""
        y_true = [100.0, 100.0, 100.0, 100.0]
        y_pred = [110.0, 130.0, 140.0, 150.0]  # 10%, 30%, 40%, 50%
        # 110 (10% off) ✓, 130 (30% off) ✓, 140 (40% off) ✗, 150 (50% off) ✗
        assert p30_accuracy(y_true, y_pred) == pytest.approx(50.0)

    def test_boundary_exactly_30pct(self):
        """Exactly ±30 % should be counted as within."""
        y_true = [100.0]
        y_pred_over = [130.0]  # exactly +30%
        y_pred_under = [70.0]  # exactly -30%
        assert p30_accuracy(y_true, y_pred_over) == pytest.approx(100.0)
        assert p30_accuracy(y_true, y_pred_under) == pytest.approx(100.0)

    def test_none_within_30pct(self):
        """All predictions far off → P30 = 0 %."""
        y_true = [100.0, 100.0]
        y_pred = [200.0, 10.0]
        assert p30_accuracy(y_true, y_pred) == pytest.approx(0.0)


class TestP10KnownValues:
    """Verify P10 accuracy against hand-calculated values."""

    def test_perfect_predictions(self):
        assert p10_accuracy([60.0, 90.0], [60.0, 90.0]) == pytest.approx(100.0)

    def test_some_within_10pct(self):
        """1 of 3 within ±10 % → P10 ≈ 33.33 %."""
        y_true = [100.0, 100.0, 100.0]
        y_pred = [105.0, 115.0, 140.0]  # 5%, 15%, 40%
        assert p10_accuracy(y_true, y_pred) == pytest.approx(100.0 / 3, abs=0.1)

    def test_boundary_exactly_10pct(self):
        y_true = [100.0]
        y_pred = [110.0]  # exactly +10%
        assert p10_accuracy(y_true, y_pred) == pytest.approx(100.0)

    def test_p10_stricter_than_p30(self):
        """P10 should always be ≤ P30 for the same data."""
        y_true = [100.0, 80.0, 60.0, 120.0, 90.0]
        y_pred = [95.0, 85.0, 55.0, 150.0, 100.0]
        assert p10_accuracy(y_true, y_pred) <= p30_accuracy(y_true, y_pred)


class TestPnAccuracyErrors:
    """Verify error handling for P30/P10."""

    def test_empty_arrays_p30(self):
        with pytest.raises(ValueError, match="empty"):
            p30_accuracy([], [])

    def test_empty_arrays_p10(self):
        with pytest.raises(ValueError, match="empty"):
            p10_accuracy([], [])

    def test_shape_mismatch_p30(self):
        with pytest.raises(ValueError, match="[Ss]hape"):
            p30_accuracy([1, 2, 3], [1, 2])

    def test_nan_values_p30(self):
        with pytest.raises(ValueError, match="NaN"):
            p30_accuracy([1.0, float("nan")], [1.0, 2.0])

    def test_nan_values_p10(self):
        with pytest.raises(ValueError, match="NaN"):
            p10_accuracy([1.0, 2.0], [1.0, float("nan")])

    def test_zero_reference_p30(self):
        with pytest.raises(ValueError, match="zero"):
            p30_accuracy([0.0, 100.0], [10.0, 110.0])

    def test_zero_reference_p10(self):
        with pytest.raises(ValueError, match="zero"):
            p10_accuracy([0.0, 100.0], [10.0, 110.0])


class TestPnAccuracyInputTypes:
    """Verify accepts various input types."""

    def test_accepts_lists(self):
        assert isinstance(p30_accuracy([100.0], [100.0]), float)

    def test_accepts_numpy(self):
        result = p30_accuracy(np.array([100.0, 80.0]), np.array([100.0, 80.0]))
        assert result == pytest.approx(100.0)


# ═══════════════════════════════════════════════════════════════════════
# evaluate_model — Comprehensive Evaluation
# ═══════════════════════════════════════════════════════════════════════


class TestEvaluateModelReturnStructure:
    """Verify return dict has the expected keys and types."""

    def test_required_keys(self):
        y_true = [90.0, 60.0, 30.0, 15.0]
        y_pred = [88.0, 62.0, 28.0, 14.0]
        result = evaluate_model(y_true, y_pred, "test_model")
        expected_keys = {
            "model_name", "rmse", "mae", "bias", "r_pearson",
            "p30", "p10", "ba_stats", "ckd_stage_agreement",
        }
        assert set(result.keys()) == expected_keys

    def test_model_name_passthrough(self):
        result = evaluate_model([90.0, 60.0], [88.0, 62.0], "CKD-EPI 2021")
        assert result["model_name"] == "CKD-EPI 2021"

    def test_numeric_values_are_float(self):
        result = evaluate_model([90.0, 60.0, 30.0], [88.0, 62.0, 28.0], "m")
        for key in ("rmse", "mae", "bias", "r_pearson", "p30", "p10", "ckd_stage_agreement"):
            assert isinstance(result[key], float), f"{key} should be float"

    def test_ba_stats_is_dict(self):
        result = evaluate_model([90.0, 60.0, 30.0], [88.0, 62.0, 28.0], "m")
        assert isinstance(result["ba_stats"], dict)
        assert "mean_bias" in result["ba_stats"]


class TestEvaluateModelKnownValues:
    """Verify metric values against hand-calculated expectations."""

    def test_perfect_predictions(self):
        y = [90.0, 60.0, 30.0, 15.0]
        result = evaluate_model(y, y, "perfect")
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mae"] == pytest.approx(0.0)
        assert result["bias"] == pytest.approx(0.0)
        assert result["p30"] == pytest.approx(100.0)
        assert result["p10"] == pytest.approx(100.0)
        assert result["ckd_stage_agreement"] == pytest.approx(100.0)

    def test_constant_bias(self):
        """Constant +5 offset → RMSE = MAE = bias = 5."""
        y_true = [90.0, 80.0, 70.0, 60.0]
        y_pred = [95.0, 85.0, 75.0, 65.0]
        result = evaluate_model(y_true, y_pred, "bias_test")
        assert result["rmse"] == pytest.approx(5.0)
        assert result["mae"] == pytest.approx(5.0)
        assert result["bias"] == pytest.approx(5.0)

    def test_pearson_perfect_correlation(self):
        """Perfectly linearly related predictions → r ≈ 1.0."""
        y_true = [30.0, 60.0, 90.0, 120.0]
        y_pred = [35.0, 65.0, 95.0, 125.0]  # y_pred = y_true + 5
        result = evaluate_model(y_true, y_pred, "corr_test")
        assert result["r_pearson"] == pytest.approx(1.0, abs=1e-10)

    def test_pearson_nan_for_single_element(self):
        """Single observation → r is undefined (NaN)."""
        result = evaluate_model([90.0], [92.0], "single")
        assert math.isnan(result["r_pearson"])


class TestEvaluateModelCKDStageAgreement:
    """Verify CKD-stage concordance metric."""

    def test_all_same_stage(self):
        """Predictions within the same CKD stage → 100 % agreement."""
        y_true = [95.0, 100.0, 110.0]  # all G1 (≥90)
        y_pred = [92.0, 105.0, 115.0]  # all G1
        result = evaluate_model(y_true, y_pred, "same_stage")
        assert result["ckd_stage_agreement"] == pytest.approx(100.0)

    def test_stage_disagreement(self):
        """50 % concordance when half cross a stage boundary."""
        # 95→95 (G1→G1 ✓), 55→55 (G3a→G3a ✓), 62→58 (G2→G3a ✗), 28→28 (G3b→G3b ✓)
        # Wait, let me be precise about boundaries:
        # 95 is G1, 55 is G3a, 62 is G2, 58 is G3a, 28 is G3b
        y_true = [95.0, 55.0, 62.0, 28.0]
        y_pred = [95.0, 55.0, 58.0, 28.0]
        # stage_true: G1, G3a, G2, G3b
        # stage_pred: G1, G3a, G3a, G3b
        # concordant: G1, G3a, G3b → 3/4 = 75%
        result = evaluate_model(y_true, y_pred, "mixed")
        assert result["ckd_stage_agreement"] == pytest.approx(75.0)


class TestEvaluateModelErrors:
    """Verify error handling."""

    def test_empty_arrays(self):
        with pytest.raises(ValueError, match="empty"):
            evaluate_model([], [], "empty")

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="[Ss]hape"):
            evaluate_model([1, 2, 3], [1, 2], "mismatch")

    def test_nan_values(self):
        with pytest.raises(ValueError, match="NaN"):
            evaluate_model([90.0, float("nan")], [90.0, 88.0], "nan_test")
