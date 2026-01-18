"""
Unit tests for freeT.evaluate module.
"""

import numpy as np
import pytest
from freeT.evaluate import bland_altman_stats, lins_ccc


class TestBlandAltmanStats:
    """Tests for bland_altman_stats function."""

    def test_perfect_agreement(self):
        """Test that identical arrays give zero bias and zero std."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == pytest.approx(0.0, abs=1e-10)
        assert stats['std_diff'] == pytest.approx(0.0, abs=1e-10)
        assert stats['loa_lower'] == pytest.approx(0.0, abs=1e-10)
        assert stats['loa_upper'] == pytest.approx(0.0, abs=1e-10)

    def test_constant_positive_bias(self):
        """Test with constant positive bias."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])  # All +0.5
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == pytest.approx(0.5, abs=1e-10)
        assert stats['std_diff'] == pytest.approx(0.0, abs=1e-10)
        assert stats['loa_lower'] == pytest.approx(0.5, abs=1e-10)
        assert stats['loa_upper'] == pytest.approx(0.5, abs=1e-10)

    def test_constant_negative_bias(self):
        """Test with constant negative bias (underestimation)."""
        y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        y_pred = np.array([1.5, 3.5, 5.5, 7.5, 9.5])  # All -0.5
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == pytest.approx(-0.5, abs=1e-10)
        assert stats['std_diff'] == pytest.approx(0.0, abs=1e-10)

    def test_known_values(self):
        """Test with known calculated values."""
        # Manually calculated example
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.0, 3.2])  # differences: 0.1, 0, 0.2
        
        stats = bland_altman_stats(y_true, y_pred)
        
        # Mean of [0.1, 0, 0.2] = 0.1
        assert stats['mean_bias'] == pytest.approx(0.1, abs=1e-10)
        
        # Sample std of [0.1, 0, 0.2] = 0.1 (with ddof=1)
        expected_std = np.std([0.1, 0, 0.2], ddof=1)
        assert stats['std_diff'] == pytest.approx(expected_std, abs=1e-10)
        
        # LOA
        assert stats['loa_lower'] == pytest.approx(0.1 - 1.96 * expected_std, abs=1e-10)
        assert stats['loa_upper'] == pytest.approx(0.1 + 1.96 * expected_std, abs=1e-10)

    def test_loa_symmetric_around_bias(self):
        """Test that limits of agreement are symmetric around mean bias."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([11.0, 19.0, 32.0, 38.0, 51.0])
        
        stats = bland_altman_stats(y_true, y_pred)
        
        # Check symmetry: distance from mean_bias to LOA bounds should be equal
        dist_lower = stats['mean_bias'] - stats['loa_lower']
        dist_upper = stats['loa_upper'] - stats['mean_bias']
        
        assert dist_lower == pytest.approx(dist_upper, abs=1e-10)
        assert dist_lower == pytest.approx(1.96 * stats['std_diff'], abs=1e-10)

    def test_returns_dict_with_correct_keys(self):
        """Test that function returns dict with expected keys."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert isinstance(stats, dict)
        assert 'mean_bias' in stats
        assert 'std_diff' in stats
        assert 'loa_lower' in stats
        assert 'loa_upper' in stats
        assert len(stats) == 4

    def test_accepts_lists(self):
        """Test that function accepts Python lists."""
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 2.1, 3.1, 4.1]
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == pytest.approx(0.1, abs=1e-10)

    def test_different_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="same shape"):
            bland_altman_stats(y_true, y_pred)

    def test_single_observation_raises_error(self):
        """Test that single observation raises ValueError."""
        y_true = np.array([1.0])
        y_pred = np.array([1.0])
        
        with pytest.raises(ValueError, match="At least 2 observations"):
            bland_altman_stats(y_true, y_pred)

    def test_empty_arrays_raises_error(self):
        """Test that empty arrays raise ValueError."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises(ValueError, match="At least 2 observations"):
            bland_altman_stats(y_true, y_pred)

    def test_nan_in_y_true_raises_error(self):
        """Test that NaN in y_true raises ValueError."""
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="NaN"):
            bland_altman_stats(y_true, y_pred)

    def test_nan_in_y_pred_raises_error(self):
        """Test that NaN in y_pred raises ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.nan, 3.0])
        
        with pytest.raises(ValueError, match="NaN"):
            bland_altman_stats(y_true, y_pred)

    def test_returns_float_values(self):
        """Test that all returned values are native Python floats."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.2, 2.1, 2.8, 4.3, 4.9])
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert isinstance(stats['mean_bias'], float)
        assert isinstance(stats['std_diff'], float)
        assert isinstance(stats['loa_lower'], float)
        assert isinstance(stats['loa_upper'], float)


class TestLinsCCC:
    """Tests for lins_ccc function."""

    def test_identical_arrays_return_one(self):
        """Test that identical arrays give CCC = 1.0 (perfect agreement)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert ccc == pytest.approx(1.0, abs=1e-10)

    def test_ccc_between_minus_one_and_one(self):
        """Test that CCC is always in range [-1, 1]."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.3, 2.8, 4.5, 4.9])
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert -1.0 <= ccc <= 1.0

    def test_negative_agreement(self):
        """Test that perfectly negatively correlated arrays give negative CCC."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reverse order
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert ccc < 0

    def test_constant_bias_reduces_ccc(self):
        """Test that constant bias reduces CCC below 1.0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # All +1.0 bias
        
        ccc = lins_ccc(y_true, y_pred)
        
        # Perfect correlation but shifted means, so CCC < 1
        assert ccc < 1.0
        assert ccc > 0.5  # But still positive and reasonable

    def test_different_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="same shape"):
            lins_ccc(y_true, y_pred)

    def test_single_observation_raises_error(self):
        """Test that single observation raises ValueError."""
        y_true = np.array([1.0])
        y_pred = np.array([1.0])
        
        with pytest.raises(ValueError, match="At least 2 observations"):
            lins_ccc(y_true, y_pred)

    def test_empty_arrays_raises_error(self):
        """Test that empty arrays raise ValueError."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises(ValueError, match="At least 2 observations"):
            lins_ccc(y_true, y_pred)

    def test_nan_in_y_true_raises_error(self):
        """Test that NaN in y_true raises ValueError."""
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="NaN"):
            lins_ccc(y_true, y_pred)

    def test_nan_in_y_pred_raises_error(self):
        """Test that NaN in y_pred raises ValueError."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.nan, 3.0])
        
        with pytest.raises(ValueError, match="NaN"):
            lins_ccc(y_true, y_pred)

    def test_accepts_lists(self):
        """Test that function accepts Python lists."""
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert ccc == pytest.approx(1.0, abs=1e-10)

    def test_returns_float(self):
        """Test that function returns a native Python float."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.2, 2.1, 2.8, 4.3, 4.9])
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert isinstance(ccc, float)

    def test_constant_arrays_same_value(self):
        """Test that constant arrays with same value return 1.0."""
        y_true = np.array([3.0, 3.0, 3.0, 3.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0])
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert ccc == pytest.approx(1.0, abs=1e-10)

    def test_high_agreement_gives_high_ccc(self):
        """Test that high agreement gives CCC close to 1."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([10.1, 19.9, 30.2, 39.8, 50.1])  # Small deviations
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert ccc > 0.99

