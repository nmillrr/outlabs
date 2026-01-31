"""
Unit tests for ldlC/evaluate.py module.

Tests Bland-Altman analysis, Lin's CCC, and other evaluation metrics.
"""

import pytest
import numpy as np
from ldlC.evaluate import bland_altman_stats


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
