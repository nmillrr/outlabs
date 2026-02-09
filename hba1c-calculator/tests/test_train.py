"""
Tests for hba1cE.train module.
"""

import numpy as np
import pandas as pd
import pytest

from hba1cE.train import create_features, stratified_split, train_ridge, save_model


class TestCreateFeatures:
    """Tests for create_features function."""

    def test_basic_functionality(self):
        """Test that create_features returns correct shape and feature names."""
        df = pd.DataFrame({
            "fpg_mgdl": [100.0, 126.0, 150.0],
            "tg_mgdl": [100.0, 150.0, 200.0],
            "hdl_mgdl": [50.0, 45.0, 40.0],
            "age_years": [30.0, 50.0, 65.0],
            "hgb_gdl": [14.0, 13.0, 12.0],
            "mcv_fl": [90.0, 88.0, 85.0],
        })

        X, feature_names = create_features(df)

        # Check shape: 3 samples, 11 features
        assert X.shape[0] == 3, "Should have 3 samples"
        assert X.shape[1] == 11, "Should have 11 features"
        assert len(feature_names) == 11, "Should have 11 feature names"

    def test_feature_names_correct(self):
        """Test that all expected feature names are present."""
        df = pd.DataFrame({
            "fpg_mgdl": [100.0],
            "tg_mgdl": [100.0],
            "hdl_mgdl": [50.0],
            "age_years": [30.0],
            "hgb_gdl": [14.0],
            "mcv_fl": [90.0],
        })

        X, feature_names = create_features(df)

        # Check raw biomarkers
        assert "fpg_mgdl" in feature_names
        assert "tg_mgdl" in feature_names
        assert "hdl_mgdl" in feature_names
        assert "age_years" in feature_names
        assert "hgb_gdl" in feature_names
        assert "mcv_fl" in feature_names

        # Check ratio features
        assert "tg_hdl_ratio" in feature_names
        assert "fpg_age_interaction" in feature_names

        # Check mechanistic estimator features
        assert "adag_estimate" in feature_names
        assert "kinetic_estimate" in feature_names
        assert "regression_estimate" in feature_names

    def test_ratio_features_calculated_correctly(self):
        """Test that ratio features are calculated correctly."""
        df = pd.DataFrame({
            "fpg_mgdl": [100.0],
            "tg_mgdl": [200.0],
            "hdl_mgdl": [50.0],
            "age_years": [40.0],
            "hgb_gdl": [14.0],
            "mcv_fl": [90.0],
        })

        X, feature_names = create_features(df)

        # TG/HDL ratio: 200/50 = 4.0
        tg_hdl_idx = feature_names.index("tg_hdl_ratio")
        assert X[0, tg_hdl_idx] == pytest.approx(4.0)

        # FPG-age interaction: 100 * 40 = 4000
        fpg_age_idx = feature_names.index("fpg_age_interaction")
        assert X[0, fpg_age_idx] == pytest.approx(4000.0)

    def test_mechanistic_estimates_in_valid_range(self):
        """Test that mechanistic estimator outputs are in valid HbA1c range."""
        df = pd.DataFrame({
            "fpg_mgdl": [100.0, 126.0, 200.0],
            "tg_mgdl": [100.0, 150.0, 250.0],
            "hdl_mgdl": [50.0, 45.0, 35.0],
            "age_years": [30.0, 50.0, 70.0],
            "hgb_gdl": [14.0, 13.0, 11.0],
            "mcv_fl": [90.0, 88.0, 82.0],
        })

        X, feature_names = create_features(df)

        # All mechanistic estimates should be in valid HbA1c range (3-20%)
        for est_name in ["adag_estimate", "kinetic_estimate", "regression_estimate"]:
            idx = feature_names.index(est_name)
            assert all(X[:, idx] > 3.0), f"{est_name} should be > 3%"
            assert all(X[:, idx] < 20.0), f"{est_name} should be < 20%"

    def test_missing_columns_raises_error(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            "fpg_mgdl": [100.0],
            "tg_mgdl": [100.0],
            # Missing hdl_mgdl, age_years, hgb_gdl, mcv_fl
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            create_features(df)

    def test_array_output_type(self):
        """Test that output is numpy array."""
        df = pd.DataFrame({
            "fpg_mgdl": [100.0, 150.0],
            "tg_mgdl": [100.0, 200.0],
            "hdl_mgdl": [50.0, 40.0],
            "age_years": [30.0, 60.0],
            "hgb_gdl": [14.0, 12.0],
            "mcv_fl": [90.0, 85.0],
        })

        X, feature_names = create_features(df)

        assert isinstance(X, np.ndarray)
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)


class TestStratifiedSplit:
    """Tests for stratified_split function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with data in all HbA1c strata."""
        # Create samples across all strata: <5.7, 5.7-6.4, 6.5-8, 8-10, >10
        np.random.seed(42)
        n_per_stratum = 10
        
        # Stratum 0: <5.7%
        hba1c_0 = np.random.uniform(4.5, 5.6, n_per_stratum)
        fpg_0 = np.random.uniform(70, 100, n_per_stratum)
        
        # Stratum 1: 5.7-6.4%
        hba1c_1 = np.random.uniform(5.7, 6.4, n_per_stratum)
        fpg_1 = np.random.uniform(100, 120, n_per_stratum)
        
        # Stratum 2: 6.5-8%
        hba1c_2 = np.random.uniform(6.5, 7.9, n_per_stratum)
        fpg_2 = np.random.uniform(120, 160, n_per_stratum)
        
        # Stratum 3: 8-10%
        hba1c_3 = np.random.uniform(8.0, 9.9, n_per_stratum)
        fpg_3 = np.random.uniform(160, 220, n_per_stratum)
        
        # Stratum 4: >10%
        hba1c_4 = np.random.uniform(10.1, 14.0, n_per_stratum)
        fpg_4 = np.random.uniform(220, 350, n_per_stratum)
        
        hba1c = np.concatenate([hba1c_0, hba1c_1, hba1c_2, hba1c_3, hba1c_4])
        fpg = np.concatenate([fpg_0, fpg_1, fpg_2, fpg_3, fpg_4])
        n = len(hba1c)
        
        return pd.DataFrame({
            "hba1c_percent": hba1c,
            "fpg_mgdl": fpg,
            "tg_mgdl": np.random.uniform(80, 200, n),
            "hdl_mgdl": np.random.uniform(35, 70, n),
            "age_years": np.random.uniform(25, 75, n),
            "hgb_gdl": np.random.uniform(11, 16, n),
            "mcv_fl": np.random.uniform(80, 100, n),
        })

    def test_basic_split(self, sample_df):
        """Test that stratified_split returns correct output types and shapes."""
        X_train, X_test, y_train, y_test = stratified_split(sample_df)
        
        # Check types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check shapes - should have 11 features
        assert X_train.shape[1] == 11
        assert X_test.shape[1] == 11
        
        # Check split ratio (default 0.3 test)
        total = len(sample_df)
        assert len(y_train) + len(y_test) == total
        assert len(y_test) == pytest.approx(total * 0.3, abs=2)

    def test_stratification_maintained(self, sample_df):
        """Test that all HbA1c strata are represented in both train and test."""
        X_train, X_test, y_train, y_test = stratified_split(sample_df)
        
        # Define strata boundaries
        def get_stratum(hba1c):
            if hba1c < 5.7:
                return 0
            elif hba1c < 6.5:
                return 1
            elif hba1c < 8.0:
                return 2
            elif hba1c < 10.0:
                return 3
            else:
                return 4
        
        train_strata = [get_stratum(y) for y in y_train]
        test_strata = [get_stratum(y) for y in y_test]
        
        # All strata should be present in both sets
        assert set(train_strata) == {0, 1, 2, 3, 4}
        assert set(test_strata) == {0, 1, 2, 3, 4}

    def test_custom_test_size(self, sample_df):
        """Test that custom test_size parameter works."""
        X_train, X_test, y_train, y_test = stratified_split(sample_df, test_size=0.2)
        
        total = len(sample_df)
        assert len(y_test) == pytest.approx(total * 0.2, abs=2)

    def test_reproducibility(self, sample_df):
        """Test that same random_state produces identical splits."""
        result1 = stratified_split(sample_df, random_state=123)
        result2 = stratified_split(sample_df, random_state=123)
        
        np.testing.assert_array_equal(result1[0], result2[0])  # X_train
        np.testing.assert_array_equal(result1[1], result2[1])  # X_test
        np.testing.assert_array_equal(result1[2], result2[2])  # y_train
        np.testing.assert_array_equal(result1[3], result2[3])  # y_test

    def test_different_random_states_differ(self, sample_df):
        """Test that different random_states produce different splits."""
        result1 = stratified_split(sample_df, random_state=1)
        result2 = stratified_split(sample_df, random_state=2)
        
        # y_train arrays should differ
        assert not np.array_equal(result1[2], result2[2])

    def test_missing_hba1c_column_raises_error(self):
        """Test that missing hba1c_percent column raises ValueError."""
        df = pd.DataFrame({
            "fpg_mgdl": [100.0, 126.0],
            "tg_mgdl": [100.0, 150.0],
            "hdl_mgdl": [50.0, 45.0],
            "age_years": [30.0, 50.0],
            "hgb_gdl": [14.0, 13.0],
            "mcv_fl": [90.0, 88.0],
        })
        
        with pytest.raises(ValueError, match="Missing required column"):
            stratified_split(df)


class TestTrainRidge:
    """Tests for train_ridge function."""

    def test_basic_training(self):
        """Test that train_ridge returns a fitted Ridge model."""
        np.random.seed(42)
        X_train = np.random.randn(100, 11)
        y_train = np.random.randn(100) * 2 + 6  # Simulated HbA1c values

        model = train_ridge(X_train, y_train)

        # Check it's a Ridge model
        from sklearn.linear_model import Ridge
        assert isinstance(model, Ridge)

        # Check model is fitted (has coef_)
        assert hasattr(model, "coef_")
        assert len(model.coef_) == 11
        assert hasattr(model, "intercept_")

    def test_prediction_works(self):
        """Test that trained model can make predictions."""
        np.random.seed(42)
        X_train = np.random.randn(100, 11)
        y_train = np.random.randn(100) * 2 + 6

        model = train_ridge(X_train, y_train)
        
        X_test = np.random.randn(20, 11)
        y_pred = model.predict(X_test)

        assert len(y_pred) == 20
        assert isinstance(y_pred, np.ndarray)

    def test_custom_alpha(self):
        """Test that custom alpha parameter is respected."""
        np.random.seed(42)
        X_train = np.random.randn(100, 11)
        y_train = np.random.randn(100) * 2 + 6

        model_alpha_1 = train_ridge(X_train, y_train, alpha=1.0)
        model_alpha_10 = train_ridge(X_train, y_train, alpha=10.0)

        # Higher alpha should shrink coefficients more
        coef_norm_1 = np.linalg.norm(model_alpha_1.coef_)
        coef_norm_10 = np.linalg.norm(model_alpha_10.coef_)
        assert coef_norm_10 < coef_norm_1

    def test_incompatible_shapes_raises_error(self):
        """Test that incompatible X and y shapes raise ValueError."""
        X_train = np.random.randn(100, 11)
        y_train = np.random.randn(50)  # Wrong number of samples

        with pytest.raises(ValueError, match="samples"):
            train_ridge(X_train, y_train)

    def test_single_sample(self):
        """Test training with minimal data still works."""
        X_train = np.array([[1.0, 2.0, 3.0]])
        y_train = np.array([6.0])

        model = train_ridge(X_train, y_train)
        assert hasattr(model, "coef_")


class TestSaveModel:
    """Tests for save_model function."""

    def test_save_model_creates_file(self, tmp_path):
        """Test that save_model creates a file at the specified path."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(np.random.randn(10, 3), np.random.randn(10))

        filepath = tmp_path / "model.joblib"
        save_model(model, str(filepath))

        assert filepath.exists()

    def test_saved_model_can_be_loaded(self, tmp_path):
        """Test that saved model can be loaded and used."""
        import joblib
        from sklearn.linear_model import Ridge

        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        filepath = tmp_path / "model.joblib"
        save_model(model, str(filepath))

        loaded_model = joblib.load(filepath)
        
        # Predictions should be identical
        X_test = np.random.randn(5, 3)
        np.testing.assert_array_almost_equal(
            model.predict(X_test),
            loaded_model.predict(X_test)
        )

    def test_creates_parent_directories(self, tmp_path):
        """Test that save_model creates parent directories if they don't exist."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
        model.fit(np.random.randn(10, 3), np.random.randn(10))

        filepath = tmp_path / "nested" / "dirs" / "model.joblib"
        save_model(model, str(filepath))

        assert filepath.exists()

