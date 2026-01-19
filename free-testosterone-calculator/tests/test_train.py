"""
Unit tests for freeT/train.py module.

Tests feature engineering, data splitting, and model training functions.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


class TestCreateFeatures:
    """Tests for create_features function."""
    
    def test_basic_feature_creation(self):
        """Test that create_features returns expected feature matrix."""
        from freeT.train import create_features
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0, 20.0, 10.0],
            'shbg_nmoll': [40.0, 30.0, 50.0],
            'alb_gl': [45.0, 42.0, 43.0],
        })
        
        X, feature_names = create_features(df)
        
        assert X.shape == (3, 5)  # 3 samples, 5 features
        assert len(feature_names) == 5
        assert 'tt_nmoll' in feature_names
        assert 'shbg_nmoll' in feature_names
        assert 'alb_gl' in feature_names
        assert 'shbg_tt_ratio' in feature_names
        assert 'ft_vermeulen' in feature_names
    
    def test_feature_values(self):
        """Test that feature values are computed correctly."""
        from freeT.train import create_features
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0],
            'shbg_nmoll': [40.0],
            'alb_gl': [45.0],
        })
        
        X, feature_names = create_features(df)
        
        # Check raw values are preserved
        tt_idx = feature_names.index('tt_nmoll')
        shbg_idx = feature_names.index('shbg_nmoll')
        alb_idx = feature_names.index('alb_gl')
        
        assert X[0, tt_idx] == 15.0
        assert X[0, shbg_idx] == 40.0
        assert X[0, alb_idx] == 45.0
        
        # Check ratio is computed correctly
        ratio_idx = feature_names.index('shbg_tt_ratio')
        expected_ratio = 40.0 / 15.0
        assert abs(X[0, ratio_idx] - expected_ratio) < 0.001
    
    def test_vermeulen_feature_computed(self):
        """Test that ft_vermeulen feature is computed."""
        from freeT.train import create_features
        from freeT.models import calc_ft_vermeulen
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0],
            'shbg_nmoll': [40.0],
            'alb_gl': [45.0],
        })
        
        X, feature_names = create_features(df)
        
        ft_idx = feature_names.index('ft_vermeulen')
        expected_ft = calc_ft_vermeulen(15.0, 40.0, 45.0)
        assert abs(X[0, ft_idx] - expected_ft) < 0.001
    
    def test_missing_column_error(self):
        """Test that missing columns raise ValueError."""
        from freeT.train import create_features
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0],
            # Missing shbg_nmoll and alb_gl
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            create_features(df)
    
    def test_empty_dataframe(self):
        """Test empty dataframe returns empty feature matrix."""
        from freeT.train import create_features
        
        df = pd.DataFrame({
            'tt_nmoll': [],
            'shbg_nmoll': [],
            'alb_gl': [],
        })
        
        X, feature_names = create_features(df)
        assert X.shape[0] == 0
        assert len(feature_names) == 5


class TestStratifiedSplit:
    """Tests for stratified_split function."""
    
    def test_split_returns_four_arrays(self):
        """Test that stratified_split returns X_train, X_test, y_train, y_test."""
        from freeT.train import stratified_split
        
        # Create sample data with enough samples for stratification
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            'tt_nmoll': np.random.uniform(10, 30, n),
            'shbg_nmoll': np.random.uniform(15, 80, n),
            'alb_gl': np.random.uniform(40, 50, n),
        })
        
        X_train, X_test, y_train, y_test = stratified_split(df)
        
        # Check all are numpy arrays
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
    
    def test_split_sizes(self):
        """Test that split has correct proportions."""
        from freeT.train import stratified_split
        
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'tt_nmoll': np.random.uniform(10, 30, n),
            'shbg_nmoll': np.random.uniform(15, 80, n),
            'alb_gl': np.random.uniform(40, 50, n),
        })
        
        X_train, X_test, y_train, y_test = stratified_split(df, test_size=0.3)
        
        # Test size should be ~30%
        test_ratio = len(X_test) / (len(X_train) + len(X_test))
        assert 0.25 < test_ratio < 0.35
    
    def test_random_state_reproducibility(self):
        """Test that random_state produces consistent splits."""
        from freeT.train import stratified_split
        
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            'tt_nmoll': np.random.uniform(10, 30, n),
            'shbg_nmoll': np.random.uniform(15, 80, n),
            'alb_gl': np.random.uniform(40, 50, n),
        })
        
        X_train1, X_test1, y_train1, y_test1 = stratified_split(df, random_state=123)
        X_train2, X_test2, y_train2, y_test2 = stratified_split(df, random_state=123)
        
        assert np.allclose(X_train1, X_train2)
        assert np.allclose(X_test1, X_test2)


class TestTrainRidge:
    """Tests for train_ridge function."""
    
    def test_returns_ridge_model(self):
        """Test that train_ridge returns fitted Ridge model."""
        from freeT.train import train_ridge
        from sklearn.linear_model import Ridge
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        
        model = train_ridge(X, y)
        
        assert isinstance(model, Ridge)
    
    def test_alpha_parameter(self):
        """Test that alpha parameter is applied."""
        from freeT.train import train_ridge
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        
        model = train_ridge(X, y, alpha=10.0)
        
        assert model.alpha == 10.0
    
    def test_model_can_predict(self):
        """Test that returned model can make predictions."""
        from freeT.train import train_ridge
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        
        model = train_ridge(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)


class TestTrainRandomForest:
    """Tests for train_random_forest function."""
    
    def test_returns_rf_model(self):
        """Test that train_random_forest returns fitted RandomForestRegressor."""
        from freeT.train import train_random_forest
        from sklearn.ensemble import RandomForestRegressor
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        
        model = train_random_forest(X, y, n_estimators=10)  # Fewer for speed
        
        assert isinstance(model, RandomForestRegressor)
    
    def test_n_estimators_parameter(self):
        """Test that n_estimators parameter is applied."""
        from freeT.train import train_random_forest
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        
        model = train_random_forest(X, y, n_estimators=25)
        
        assert model.n_estimators == 25
    
    def test_model_can_predict(self):
        """Test that returned model can make predictions."""
        from freeT.train import train_random_forest
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        
        model = train_random_forest(X, y, n_estimators=10)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)


class TestTrainLightGBM:
    """Tests for train_lightgbm function."""
    
    def test_returns_lightgbm_model(self):
        """Test that train_lightgbm returns fitted LGBMRegressor."""
        from freeT.train import train_lightgbm
        import lightgbm as lgb
        
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([1, 2, 3, 4])
        X_val = np.array([[2, 3], [4, 5]])
        y_val = np.array([1.5, 2.5])
        
        model = train_lightgbm(X_train, y_train, X_val, y_val)
        
        assert isinstance(model, lgb.LGBMRegressor)
    
    def test_model_can_predict(self):
        """Test that returned model can make predictions."""
        from freeT.train import train_lightgbm
        
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([1, 2, 3, 4])
        X_val = np.array([[2, 3], [4, 5]])
        y_val = np.array([1.5, 2.5])
        
        model = train_lightgbm(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_train)
        
        assert len(predictions) == len(y_train)


class TestCrossValidateModel:
    """Tests for cross_validate_model function."""
    
    def test_returns_dict_with_metrics(self):
        """Test that cross_validate_model returns dict with expected keys."""
        from freeT.train import cross_validate_model
        from sklearn.linear_model import Ridge
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                      [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        model = Ridge()
        results = cross_validate_model(model, X, y, n_splits=5)
        
        assert 'RMSE_mean' in results
        assert 'RMSE_std' in results
        assert 'MAE_mean' in results
        assert 'MAE_std' in results
    
    def test_metrics_are_floats(self):
        """Test that metric values are floats."""
        from freeT.train import cross_validate_model
        from sklearn.linear_model import Ridge
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                      [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        model = Ridge()
        results = cross_validate_model(model, X, y, n_splits=5)
        
        assert isinstance(results['RMSE_mean'], float)
        assert isinstance(results['RMSE_std'], float)
        assert isinstance(results['MAE_mean'], float)
        assert isinstance(results['MAE_std'], float)
    
    def test_metrics_non_negative(self):
        """Test that error metrics are non-negative."""
        from freeT.train import cross_validate_model
        from sklearn.linear_model import Ridge
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                      [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        model = Ridge()
        results = cross_validate_model(model, X, y, n_splits=5)
        
        assert results['RMSE_mean'] >= 0
        assert results['RMSE_std'] >= 0
        assert results['MAE_mean'] >= 0
        assert results['MAE_std'] >= 0


class TestSaveModel:
    """Tests for save_model function."""
    
    def test_saves_model_to_file(self):
        """Test that save_model creates file."""
        from freeT.train import save_model, train_ridge
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        model = train_ridge(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.joblib')
            save_model(model, filepath)
            
            assert os.path.exists(filepath)
    
    def test_saved_model_can_be_loaded(self):
        """Test that saved model can be loaded and used."""
        from freeT.train import save_model, train_ridge
        import joblib
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        model = train_ridge(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.joblib')
            save_model(model, filepath)
            
            loaded_model = joblib.load(filepath)
            predictions = loaded_model.predict(X)
            
            assert len(predictions) == len(y)
    
    def test_creates_parent_directories(self):
        """Test that save_model creates parent directories if needed."""
        from freeT.train import save_model, train_ridge
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        model = train_ridge(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'nested', 'dir', 'model.joblib')
            save_model(model, filepath)
            
            assert os.path.exists(filepath)
