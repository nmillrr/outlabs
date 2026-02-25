"""Tests for eGFR/evaluate.py — Bland-Altman analysis."""

import math

import numpy as np
import pytest

from eGFR.evaluate import bland_altman_stats


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
