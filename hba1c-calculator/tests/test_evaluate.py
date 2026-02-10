"""Unit tests for hba1cE.evaluate module."""

import numpy as np
import pytest

from hba1cE.evaluate import (
    bland_altman_stats,
    evaluate_by_hba1c_strata,
    evaluate_model,
    lins_ccc,
)


class TestBlandAltmanStats:
    """Tests for bland_altman_stats function."""

    def test_perfect_agreement(self):
        """When predicted equals true, bias should be zero."""
        y_true = [5.0, 6.0, 7.0, 8.0, 9.0]
        y_pred = [5.0, 6.0, 7.0, 8.0, 9.0]
        result = bland_altman_stats(y_true, y_pred)
        assert result["mean_bias"] == pytest.approx(0.0, abs=1e-10)
        assert result["std_diff"] == pytest.approx(0.0, abs=1e-10)
        assert result["loa_lower"] == pytest.approx(0.0, abs=1e-10)
        assert result["loa_upper"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_positive_bias(self):
        """When all predictions are +0.5 higher, bias should be 0.5."""
        y_true = [5.0, 6.0, 7.0, 8.0]
        y_pred = [5.5, 6.5, 7.5, 8.5]
        result = bland_altman_stats(y_true, y_pred)
        assert result["mean_bias"] == pytest.approx(0.5, abs=1e-10)
        assert result["std_diff"] == pytest.approx(0.0, abs=1e-10)
        assert result["loa_lower"] == pytest.approx(0.5, abs=1e-10)
        assert result["loa_upper"] == pytest.approx(0.5, abs=1e-10)

    def test_constant_negative_bias(self):
        """When all predictions are -0.3 lower, bias should be -0.3."""
        y_true = [5.0, 6.0, 7.0]
        y_pred = [4.7, 5.7, 6.7]
        result = bland_altman_stats(y_true, y_pred)
        assert result["mean_bias"] == pytest.approx(-0.3, abs=1e-10)

    def test_known_values(self):
        """Test with hand-calculated values."""
        y_true = np.array([5.0, 6.0, 7.0, 8.0])
        y_pred = np.array([5.2, 5.8, 7.3, 8.1])
        # differences: [0.2, -0.2, 0.3, 0.1]
        # mean_bias = 0.1
        # std_diff = std([0.2, -0.2, 0.3, 0.1], ddof=1) ≈ 0.21602
        result = bland_altman_stats(y_true, y_pred)
        assert result["mean_bias"] == pytest.approx(0.1, abs=1e-10)
        expected_std = np.std([0.2, -0.2, 0.3, 0.1], ddof=1)
        assert result["std_diff"] == pytest.approx(expected_std, abs=1e-10)
        assert result["loa_lower"] == pytest.approx(
            0.1 - 1.96 * expected_std, abs=1e-10
        )
        assert result["loa_upper"] == pytest.approx(
            0.1 + 1.96 * expected_std, abs=1e-10
        )

    def test_loa_symmetry(self):
        """LOA should be symmetric around mean_bias."""
        y_true = [5.0, 6.0, 7.0, 8.0, 9.0]
        y_pred = [5.1, 6.3, 6.8, 8.2, 9.1]
        result = bland_altman_stats(y_true, y_pred)
        mid = (result["loa_lower"] + result["loa_upper"]) / 2
        assert mid == pytest.approx(result["mean_bias"], abs=1e-10)

    def test_returns_dict_keys(self):
        """Result should contain exactly the expected keys."""
        result = bland_altman_stats([5.0, 6.0], [5.1, 6.1])
        assert set(result.keys()) == {
            "mean_bias",
            "std_diff",
            "loa_lower",
            "loa_upper",
        }

    def test_numpy_array_input(self):
        """Should accept numpy arrays."""
        y_true = np.array([5.0, 6.0, 7.0])
        y_pred = np.array([5.1, 6.1, 7.1])
        result = bland_altman_stats(y_true, y_pred)
        assert result["mean_bias"] == pytest.approx(0.1, abs=1e-10)

    def test_empty_input_raises(self):
        """Should raise ValueError for empty inputs."""
        with pytest.raises(ValueError, match="empty"):
            bland_altman_stats([], [])

    def test_mismatched_lengths_raises(self):
        """Should raise ValueError for inputs of different lengths."""
        with pytest.raises(ValueError, match="same length"):
            bland_altman_stats([5.0, 6.0], [5.0])

    def test_nan_input_raises(self):
        """Should raise ValueError for NaN values."""
        with pytest.raises(ValueError, match="NaN"):
            bland_altman_stats([5.0, float("nan")], [5.0, 6.0])
        with pytest.raises(ValueError, match="NaN"):
            bland_altman_stats([5.0, 6.0], [5.0, float("nan")])


class TestLinsCCC:
    """Tests for lins_ccc function."""

    def test_identical_arrays(self):
        """Identical arrays should return CCC = 1.0."""
        y = [5.0, 6.0, 7.0, 8.0, 9.0]
        assert lins_ccc(y, y) == pytest.approx(1.0, abs=1e-10)

    def test_near_perfect_agreement(self):
        """Nearly identical arrays should return CCC close to 1.0."""
        y_true = [5.0, 6.0, 7.0, 8.0, 9.0]
        y_pred = [5.01, 6.01, 7.01, 8.01, 9.01]
        result = lins_ccc(y_true, y_pred)
        assert result > 0.99

    def test_constant_offset_lowers_ccc(self):
        """Constant offset (bias) should lower CCC below Pearson r."""
        y_true = [5.0, 6.0, 7.0, 8.0, 9.0]
        y_pred = [6.0, 7.0, 8.0, 9.0, 10.0]  # +1.0 shift
        result = lins_ccc(y_true, y_pred)
        # Pearson r = 1.0 but CCC < 1.0 due to bias
        assert result < 1.0
        assert result > 0.0

    def test_known_values(self):
        """Test with hand-calculated CCC value."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Identical → CCC = 1.0
        assert lins_ccc(y_true, y_pred) == pytest.approx(1.0, abs=1e-10)

        # Manual calc for shifted data:
        # y_true = [1,2,3,4,5], y_pred = [2,3,4,5,6]
        # mean_true=3, mean_pred=4, var_true=2, var_pred=2, cov=2
        # CCC = 2*2 / (2 + 2 + 1) = 4/5 = 0.8
        result = lins_ccc([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        assert result == pytest.approx(0.8, abs=1e-10)

    def test_ccc_range(self):
        """CCC should be between -1 and 1."""
        y_true = [5.0, 6.5, 7.2, 8.1, 9.3]
        y_pred = [5.3, 6.1, 7.5, 7.9, 9.0]
        result = lins_ccc(y_true, y_pred)
        assert -1.0 <= result <= 1.0

    def test_symmetry(self):
        """CCC(x, y) should equal CCC(y, x)."""
        y_true = [5.0, 6.0, 7.0, 8.0, 9.0]
        y_pred = [5.2, 5.8, 7.3, 8.1, 8.9]
        assert lins_ccc(y_true, y_pred) == pytest.approx(
            lins_ccc(y_pred, y_true), abs=1e-10
        )

    def test_constant_arrays(self):
        """Constant identical arrays should return CCC = 1.0."""
        result = lins_ccc([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_too_few_elements_raises(self):
        """Should raise ValueError for fewer than 2 elements."""
        with pytest.raises(ValueError, match="at least 2"):
            lins_ccc([5.0], [5.0])

    def test_mismatched_lengths_raises(self):
        """Should raise ValueError for inputs of different lengths."""
        with pytest.raises(ValueError, match="same length"):
            lins_ccc([5.0, 6.0], [5.0])

    def test_nan_input_raises(self):
        """Should raise ValueError for NaN values."""
        with pytest.raises(ValueError, match="NaN"):
            lins_ccc([5.0, float("nan")], [5.0, 6.0])


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_all_keys(self):
        """Result should contain all expected keys."""
        y_true = [5.0, 6.0, 7.0, 8.0, 9.0]
        y_pred = [5.1, 6.2, 6.9, 8.1, 9.0]
        result = evaluate_model(y_true, y_pred)
        expected_keys = {
            "model_name",
            "rmse",
            "mae",
            "bias",
            "r_pearson",
            "lin_ccc",
            "ba_stats",
            "pct_within_0_5",
        }
        assert set(result.keys()) == expected_keys

    def test_perfect_predictions(self):
        """Identical predictions should yield RMSE=0, MAE=0, bias=0, CCC=1."""
        y = [5.0, 6.0, 7.0, 8.0, 9.0]
        result = evaluate_model(y, y)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["mae"] == pytest.approx(0.0, abs=1e-10)
        assert result["bias"] == pytest.approx(0.0, abs=1e-10)
        assert result["r_pearson"] == pytest.approx(1.0, abs=1e-10)
        assert result["lin_ccc"] == pytest.approx(1.0, abs=1e-10)
        assert result["pct_within_0_5"] == pytest.approx(100.0, abs=1e-10)

    def test_rmse_mae_known_values(self):
        """Test RMSE and MAE with hand-calculated values."""
        y_true = np.array([5.0, 6.0, 7.0, 8.0])
        y_pred = np.array([5.2, 5.8, 7.3, 8.1])
        # errors: [0.2, -0.2, 0.3, 0.1]
        # RMSE = sqrt(mean([0.04, 0.04, 0.09, 0.01])) = sqrt(0.045) ≈ 0.2121
        # MAE = mean([0.2, 0.2, 0.3, 0.1]) = 0.2
        result = evaluate_model(y_true, y_pred)
        assert result["rmse"] == pytest.approx(np.sqrt(0.045), abs=1e-10)
        assert result["mae"] == pytest.approx(0.2, abs=1e-10)

    def test_bias(self):
        """Constant positive offset should show correct bias."""
        y_true = [5.0, 6.0, 7.0, 8.0]
        y_pred = [5.5, 6.5, 7.5, 8.5]
        result = evaluate_model(y_true, y_pred)
        assert result["bias"] == pytest.approx(0.5, abs=1e-10)

    def test_pct_within_0_5(self):
        """Test percentage within ±0.5% threshold."""
        y_true = [5.0, 6.0, 7.0, 8.0]
        y_pred = [5.3, 6.8, 7.4, 9.0]  # within: 5.3, 7.4; outside: 6.8, 9.0
        result = evaluate_model(y_true, y_pred)
        assert result["pct_within_0_5"] == pytest.approx(50.0, abs=1e-10)

    def test_ba_stats_included(self):
        """Bland-Altman stats should be a nested dict with expected keys."""
        y_true = [5.0, 6.0, 7.0, 8.0, 9.0]
        y_pred = [5.1, 6.2, 6.9, 8.1, 9.0]
        result = evaluate_model(y_true, y_pred)
        ba = result["ba_stats"]
        assert isinstance(ba, dict)
        assert set(ba.keys()) == {"mean_bias", "std_diff", "loa_lower", "loa_upper"}

    def test_model_name(self):
        """Model name should be passed through."""
        result = evaluate_model([5.0, 6.0], [5.1, 6.1], model_name="Ridge")
        assert result["model_name"] == "Ridge"

    def test_mismatched_lengths_raises(self):
        """Should raise ValueError for inputs of different lengths."""
        with pytest.raises(ValueError, match="same length"):
            evaluate_model([5.0, 6.0], [5.0])

    def test_nan_raises(self):
        """Should raise ValueError for NaN values."""
        with pytest.raises(ValueError, match="NaN"):
            evaluate_model([5.0, float("nan")], [5.0, 6.0])

    def test_too_few_elements_raises(self):
        """Should raise ValueError for fewer than 2 elements."""
        with pytest.raises(ValueError, match="at least 2"):
            evaluate_model([5.0], [5.0])


class TestEvaluateByHba1cStrata:
    """Tests for evaluate_by_hba1c_strata function."""

    def test_returns_three_strata(self):
        """Result should contain normal, prediabetes, diabetes keys."""
        y_true = [5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
        y_pred = [5.1, 5.4, 6.1, 6.6, 7.1, 8.2]
        hba1c = [5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
        result = evaluate_by_hba1c_strata(y_true, y_pred, hba1c)
        assert set(result.keys()) == {"normal", "prediabetes", "diabetes"}

    def test_normal_strata_metrics(self):
        """Normal stratum should compute metrics for HbA1c < 5.7."""
        y_true = [5.0, 5.2, 5.4, 5.6, 6.0, 6.2, 7.0, 8.0]
        y_pred = [5.0, 5.2, 5.4, 5.6, 6.0, 6.2, 7.0, 8.0]
        hba1c = [5.0, 5.2, 5.4, 5.6, 6.0, 6.2, 7.0, 8.0]
        result = evaluate_by_hba1c_strata(y_true, y_pred, hba1c)
        assert result["normal"] is not None
        assert result["normal"]["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["normal"]["model_name"] == "normal"

    def test_diabetes_strata_metrics(self):
        """Diabetes stratum should compute metrics for HbA1c >= 6.5."""
        y_true = [5.0, 5.5, 7.0, 8.0, 9.0]
        y_pred = [5.1, 5.6, 7.2, 8.3, 9.1]
        hba1c = [5.0, 5.5, 7.0, 8.0, 9.0]
        result = evaluate_by_hba1c_strata(y_true, y_pred, hba1c)
        assert result["diabetes"] is not None
        assert result["diabetes"]["rmse"] > 0.0
        assert result["diabetes"]["model_name"] == "diabetes"

    def test_sparse_stratum_returns_none(self):
        """Strata with fewer than 2 samples should return None."""
        # All values are normal, prediabetes and diabetes have <2 samples
        y_true = [5.0, 5.2, 5.4]
        y_pred = [5.1, 5.3, 5.5]
        hba1c = [5.0, 5.2, 5.4]
        result = evaluate_by_hba1c_strata(y_true, y_pred, hba1c)
        assert result["normal"] is not None
        assert result["prediabetes"] is None
        assert result["diabetes"] is None

    def test_single_sample_stratum_returns_none(self):
        """A stratum with exactly 1 sample should return None."""
        y_true = [5.0, 5.2, 6.0]
        y_pred = [5.1, 5.3, 6.1]
        hba1c = [5.0, 5.2, 6.0]  # only 1 prediabetes sample
        result = evaluate_by_hba1c_strata(y_true, y_pred, hba1c)
        assert result["normal"] is not None
        assert result["prediabetes"] is None

    def test_mismatched_lengths_raises(self):
        """Should raise ValueError for inputs of different lengths."""
        with pytest.raises(ValueError, match="same length"):
            evaluate_by_hba1c_strata([5.0, 6.0], [5.1], [5.0, 6.0])

    def test_nan_raises(self):
        """Should raise ValueError for NaN values."""
        with pytest.raises(ValueError, match="NaN"):
            evaluate_by_hba1c_strata(
                [5.0, 6.0], [5.1, 6.1], [5.0, float("nan")]
            )

    def test_empty_raises(self):
        """Should raise ValueError for empty inputs."""
        with pytest.raises(ValueError, match="empty"):
            evaluate_by_hba1c_strata([], [], [])
