"""
Unit tests for ldlC/evaluate.py module.

Tests Bland-Altman analysis, Lin's CCC, and other evaluation metrics.
"""

import pytest
import numpy as np
from ldlC.evaluate import bland_altman_stats, lins_ccc


class TestBlandAltmanStats:
    """Test suite for bland_altman_stats function."""
    
    def test_perfect_agreement(self):
        """Test that identical arrays produce zero bias."""
        y_true = np.array([100, 120, 140, 160, 180])
        y_pred = np.array([100, 120, 140, 160, 180])
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == 0.0
        assert stats['std_diff'] == 0.0
        assert stats['loa_lower'] == 0.0
        assert stats['loa_upper'] == 0.0
    
    def test_constant_positive_bias(self):
        """Test constant positive bias detection."""
        y_true = np.array([100, 120, 140, 160, 180])
        y_pred = np.array([105, 125, 145, 165, 185])  # All +5 bias
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == 5.0
        assert stats['std_diff'] == 0.0
        assert stats['loa_lower'] == 5.0
        assert stats['loa_upper'] == 5.0
    
    def test_constant_negative_bias(self):
        """Test constant negative bias detection."""
        y_true = np.array([100, 120, 140, 160, 180])
        y_pred = np.array([97, 117, 137, 157, 177])  # All -3 bias
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == -3.0
        assert stats['std_diff'] == 0.0
    
    def test_known_values(self):
        """Test with known calculated values."""
        # Example: differences are [2, -2, 4, -4, 0]
        # Mean bias = 0, std_diff = sqrt(10) ≈ 3.162
        y_true = np.array([100, 102, 100, 104, 100])
        y_pred = np.array([102, 100, 104, 100, 100])
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert abs(stats['mean_bias']) < 1e-10  # Should be exactly 0
        assert abs(stats['std_diff'] - np.sqrt(10)) < 1e-10
        
        # LOA: 0 ± 1.96 * sqrt(10) ≈ ±6.198
        expected_loa = 1.96 * np.sqrt(10)
        assert abs(stats['loa_lower'] - (-expected_loa)) < 1e-10
        assert abs(stats['loa_upper'] - expected_loa) < 1e-10
    
    def test_limits_of_agreement_calculation(self):
        """Verify LOA = mean ± 1.96 * std."""
        y_true = np.array([100, 110, 120, 130, 140])
        y_pred = np.array([102, 108, 123, 127, 145])
        
        stats = bland_altman_stats(y_true, y_pred)
        
        # Verify LOA calculation
        expected_lower = stats['mean_bias'] - 1.96 * stats['std_diff']
        expected_upper = stats['mean_bias'] + 1.96 * stats['std_diff']
        
        assert abs(stats['loa_lower'] - expected_lower) < 1e-10
        assert abs(stats['loa_upper'] - expected_upper) < 1e-10
    
    def test_handles_nan_values(self):
        """Test that NaN values are properly excluded."""
        y_true = np.array([100, np.nan, 120, 130, np.nan])
        y_pred = np.array([102, 112, np.nan, 132, 142])
        
        # Only (100, 102) and (130, 132) are valid pairs
        # Bias = 2.0 for both pairs
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == 2.0
        assert stats['std_diff'] == 0.0
    
    def test_empty_array_raises_error(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            bland_altman_stats(np.array([]), np.array([]))
    
    def test_mismatched_lengths_raises_error(self):
        """Test that arrays of different lengths raise ValueError."""
        y_true = np.array([100, 120, 140])
        y_pred = np.array([102, 118])
        
        with pytest.raises(ValueError, match="same length"):
            bland_altman_stats(y_true, y_pred)
    
    def test_all_nan_raises_error(self):
        """Test that all-NaN arrays raise ValueError."""
        y_true = np.array([np.nan, np.nan, np.nan])
        y_pred = np.array([100, 110, 120])
        
        with pytest.raises(ValueError, match="No valid"):
            bland_altman_stats(y_true, y_pred)
    
    def test_list_input(self):
        """Test that lists are converted to arrays."""
        y_true = [100, 120, 140]
        y_pred = [102, 122, 142]
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == 2.0
    
    def test_single_value_pair(self):
        """Test with single value pair."""
        y_true = np.array([100])
        y_pred = np.array([105])
        
        stats = bland_altman_stats(y_true, y_pred)
        
        assert stats['mean_bias'] == 5.0
        # std with ddof=1 for single value is NaN - but that's expected behavior
        # We should get NaN for std and LOA with single value
        assert np.isnan(stats['std_diff']) or stats['std_diff'] == 0.0


class TestLinsCCC:
    """Test suite for lins_ccc function."""
    
    def test_identical_arrays_perfect_agreement(self):
        """Test that identical arrays → CCC = 1.0."""
        y_true = np.array([100, 120, 140, 160, 180])
        y_pred = np.array([100, 120, 140, 160, 180])
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert abs(ccc - 1.0) < 1e-10
    
    def test_perfect_negative_agreement(self):
        """Test that perfectly inversely related arrays → CCC = -1.0."""
        y_true = np.array([100, 120, 140, 160, 180])
        y_pred = np.array([180, 160, 140, 120, 100])  # Inverse
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert abs(ccc - (-1.0)) < 1e-10
    
    def test_ccc_range(self):
        """Test that CCC is always between -1 and 1."""
        np.random.seed(42)
        y_true = np.random.uniform(50, 200, 100)
        y_pred = y_true + np.random.normal(0, 10, 100)
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert -1.0 <= ccc <= 1.0
    
    def test_constant_bias_reduces_ccc(self):
        """Test that constant bias reduces CCC below 1.0."""
        y_true = np.array([100, 120, 140, 160, 180])
        y_pred = np.array([110, 130, 150, 170, 190])  # +10 bias
        
        ccc = lins_ccc(y_true, y_pred)
        
        # CCC should be high but less than 1.0 due to bias
        assert ccc < 1.0
        assert ccc > 0.9  # Should still be very high due to perfect correlation
    
    def test_handles_nan_values(self):
        """Test that NaN values are properly excluded."""
        y_true = np.array([100, np.nan, 140, 160, np.nan])
        y_pred = np.array([100, 120, np.nan, 160, 180])
        
        # Only (100, 100) and (160, 160) are valid pairs = perfect agreement on valid pairs
        ccc = lins_ccc(y_true, y_pred)
        
        assert abs(ccc - 1.0) < 1e-10
    
    def test_empty_array_raises_error(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            lins_ccc(np.array([]), np.array([]))
    
    def test_mismatched_lengths_raises_error(self):
        """Test that arrays of different lengths raise ValueError."""
        y_true = np.array([100, 120, 140])
        y_pred = np.array([102, 118])
        
        with pytest.raises(ValueError, match="same length"):
            lins_ccc(y_true, y_pred)
    
    def test_all_nan_raises_error(self):
        """Test that all-NaN arrays raise ValueError."""
        y_true = np.array([np.nan, np.nan, np.nan])
        y_pred = np.array([100, 110, 120])
        
        with pytest.raises(ValueError, match="No valid"):
            lins_ccc(y_true, y_pred)
    
    def test_single_value_raises_error(self):
        """Test that single value pair raises ValueError (need 2+ points for CCC)."""
        y_true = np.array([100])
        y_pred = np.array([105])
        
        with pytest.raises(ValueError, match="At least 2"):
            lins_ccc(y_true, y_pred)
    
    def test_list_input(self):
        """Test that lists are converted to arrays."""
        y_true = [100, 120, 140]
        y_pred = [100, 120, 140]
        
        ccc = lins_ccc(y_true, y_pred)
        
        assert abs(ccc - 1.0) < 1e-10
    
    def test_high_variance_difference(self):
        """Test CCC with different variances between arrays."""
        y_true = np.array([100, 110, 120, 130, 140])  # Low variance
        y_pred = np.array([50, 100, 120, 140, 190])   # High variance
        
        ccc = lins_ccc(y_true, y_pred)
        
        # CCC should be reduced due to variance mismatch
        assert ccc < 1.0
        assert ccc > 0  # Still positive correlation

