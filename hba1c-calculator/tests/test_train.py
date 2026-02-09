"""
Tests for hba1cE.train module.
"""

import numpy as np
import pandas as pd
import pytest

from hba1cE.train import create_features


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
