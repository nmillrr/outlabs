"""
Unit tests for freeT/predict.py module.

Tests the prediction API for free testosterone estimation.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import math


class TestPredictFtVermeulen:
    """Tests for predict_ft with method='vermeulen'."""
    
    def test_returns_dict(self):
        """Test that predict_ft returns a dictionary."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
        
        assert isinstance(result, dict)
    
    def test_dict_keys(self):
        """Test that result dict has expected keys."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
        
        assert 'ft_pred' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'method' in result
    
    def test_vermeulen_method_set(self):
        """Test that method is set to 'vermeulen'."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
        
        assert result['method'] == 'vermeulen'
    
    def test_ci_none_for_vermeulen(self):
        """Test that CI values are None for vermeulen method."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
        
        assert result['ci_lower'] is None
        assert result['ci_upper'] is None
    
    def test_ft_pred_positive(self):
        """Test that ft_pred is positive for valid inputs."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
        
        assert result['ft_pred'] > 0
    
    def test_ft_pred_less_than_tt(self):
        """Test that ft_pred is less than total testosterone."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='vermeulen')
        
        assert result['ft_pred'] < 15.0
    
    def test_matches_vermeulen_solver(self):
        """Test that result matches direct Vermeulen solver call."""
        from freeT.predict import predict_ft
        from freeT.models import calc_ft_vermeulen
        
        tt, shbg, alb = 15.0, 40.0, 45.0
        
        result = predict_ft(tt=tt, shbg=shbg, alb=alb, method='vermeulen')
        expected = calc_ft_vermeulen(tt, shbg, alb)
        
        assert abs(result['ft_pred'] - expected) < 0.0001


class TestPredictFtHybrid:
    """Tests for predict_ft with method='hybrid'."""
    
    def test_hybrid_fallback_to_vermeulen(self):
        """Test that hybrid falls back to vermeulen when no model available."""
        from freeT.predict import predict_ft
        
        # Without a model file, should fallback
        result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='hybrid')
        
        # Method may be 'vermeulen' or 'hybrid' depending on model availability
        assert result['method'] in ('hybrid', 'vermeulen')
        assert result['ft_pred'] > 0
    
    def test_hybrid_returns_dict(self):
        """Test that hybrid method returns proper dict."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='hybrid')
        
        assert isinstance(result, dict)
        assert 'ft_pred' in result
        assert 'method' in result


class TestPredictFtInputValidation:
    """Tests for input validation in predict_ft."""
    
    def test_negative_tt_raises_error(self):
        """Test that negative TT raises ValueError."""
        from freeT.predict import predict_ft
        
        with pytest.raises(ValueError, match="non-negative"):
            predict_ft(tt=-5.0, shbg=40.0, alb=45.0)
    
    def test_negative_shbg_raises_error(self):
        """Test that negative SHBG raises ValueError."""
        from freeT.predict import predict_ft
        
        with pytest.raises(ValueError, match="non-negative"):
            predict_ft(tt=15.0, shbg=-10.0, alb=45.0)
    
    def test_zero_alb_raises_error(self):
        """Test that zero albumin raises ValueError."""
        from freeT.predict import predict_ft
        
        with pytest.raises(ValueError, match="positive"):
            predict_ft(tt=15.0, shbg=40.0, alb=0.0)
    
    def test_negative_alb_raises_error(self):
        """Test that negative albumin raises ValueError."""
        from freeT.predict import predict_ft
        
        with pytest.raises(ValueError, match="positive"):
            predict_ft(tt=15.0, shbg=40.0, alb=-5.0)
    
    def test_nan_tt_raises_error(self):
        """Test that NaN TT raises ValueError."""
        from freeT.predict import predict_ft
        
        with pytest.raises(ValueError, match="NaN"):
            predict_ft(tt=float('nan'), shbg=40.0, alb=45.0)
    
    def test_nan_shbg_raises_error(self):
        """Test that NaN SHBG raises ValueError."""
        from freeT.predict import predict_ft
        
        with pytest.raises(ValueError, match="NaN"):
            predict_ft(tt=15.0, shbg=float('nan'), alb=45.0)
    
    def test_nan_alb_raises_error(self):
        """Test that NaN albumin raises ValueError."""
        from freeT.predict import predict_ft
        
        with pytest.raises(ValueError, match="NaN"):
            predict_ft(tt=15.0, shbg=40.0, alb=float('nan'))
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        from freeT.predict import predict_ft
        
        with pytest.raises(ValueError, match="must be 'vermeulen' or 'hybrid'"):
            predict_ft(tt=15.0, shbg=40.0, alb=45.0, method='invalid')


class TestPredictFtEdgeCases:
    """Tests for edge cases in predict_ft."""
    
    def test_zero_tt_returns_zero(self):
        """Test that zero TT returns zero FT."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=0.0, shbg=40.0, alb=45.0, method='vermeulen')
        
        assert result['ft_pred'] == 0.0
    
    def test_zero_shbg(self):
        """Test that zero SHBG returns valid result."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=0.0, alb=45.0, method='vermeulen')
        
        assert result['ft_pred'] > 0
        assert result['ft_pred'] <= 15.0
    
    def test_high_shbg(self):
        """Test prediction with high SHBG."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=15.0, shbg=100.0, alb=45.0, method='vermeulen')
        
        assert result['ft_pred'] > 0
        assert result['ft_pred'] < 15.0
    
    def test_low_tt(self):
        """Test prediction with low TT."""
        from freeT.predict import predict_ft
        
        result = predict_ft(tt=5.0, shbg=40.0, alb=45.0, method='vermeulen')
        
        assert result['ft_pred'] > 0
        assert result['ft_pred'] < 5.0
    
    def test_nonexistent_model_path(self):
        """Test that nonexistent model path falls back to vermeulen."""
        from freeT.predict import predict_ft
        
        result = predict_ft(
            tt=15.0, shbg=40.0, alb=45.0,
            method='hybrid',
            model_path='/nonexistent/path/to/model.joblib'
        )
        
        # Should fall back to vermeulen
        assert result['method'] == 'vermeulen'
        assert result['ft_pred'] > 0


class TestPredictFtWithModel:
    """Tests for predict_ft with saved ML models."""
    
    def test_with_rf_model(self):
        """Test prediction with a saved Random Forest model."""
        from freeT.predict import predict_ft
        from freeT.train import train_random_forest, save_model
        import numpy as np
        
        # Create and train a simple model
        np.random.seed(42)
        X = np.random.uniform(5, 25, (100, 5))  # 5 features
        y = np.random.uniform(0.1, 0.5, 100)
        
        model = train_random_forest(X, y, n_estimators=10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_rf.joblib')
            save_model(model, model_path)
            
            result = predict_ft(
                tt=15.0, shbg=40.0, alb=45.0,
                method='hybrid',
                model_path=model_path
            )
            
            assert result['method'] == 'hybrid'
            assert result['ft_pred'] is not None
            # RF models should have confidence intervals
            assert result['ci_lower'] is not None
            assert result['ci_upper'] is not None
            assert result['ci_lower'] <= result['ci_upper']
    
    def test_with_ridge_model(self):
        """Test prediction with a saved Ridge model."""
        from freeT.predict import predict_ft
        from freeT.train import train_ridge, save_model
        import numpy as np
        
        # Create and train a simple model
        np.random.seed(42)
        X = np.random.uniform(5, 25, (100, 5))  # 5 features
        y = np.random.uniform(0.1, 0.5, 100)
        
        model = train_ridge(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_ridge.joblib')
            save_model(model, model_path)
            
            result = predict_ft(
                tt=15.0, shbg=40.0, alb=45.0,
                method='hybrid',
                model_path=model_path
            )
            
            assert result['method'] == 'hybrid'
            assert result['ft_pred'] is not None
            # Ridge models don't have estimators_, so no CI
            assert result['ci_lower'] is None
            assert result['ci_upper'] is None
