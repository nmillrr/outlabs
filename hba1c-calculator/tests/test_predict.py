"""Tests for hba1cE.predict module."""

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hba1cE.predict import (
    _build_feature_vector,
    _load_best_model,
    predict_hba1c,
)


# ---------------------------------------------------------------------------
# Tests for predict_hba1c – ADAG method
# ---------------------------------------------------------------------------
class TestPredictHba1cADAG:
    """Tests for predict_hba1c with method='adag'."""

    def test_adag_basic(self):
        result = predict_hba1c(126.0, method="adag")
        assert isinstance(result, dict)
        assert result["method"] == "adag"
        assert result["ci_lower"] is None
        assert result["ci_upper"] is None
        assert result["warning"] is None
        assert abs(result["hba1c_pred"] - 6.017) < 0.01

    def test_adag_diabetes_range(self):
        result = predict_hba1c(200.0, method="adag")
        assert result["hba1c_pred"] > 8.0

    def test_adag_ignores_optional_inputs(self):
        result = predict_hba1c(126.0, tg=150.0, hdl=50.0, method="adag")
        assert result["method"] == "adag"
        assert result["warning"] is None


# ---------------------------------------------------------------------------
# Tests for predict_hba1c – Kinetic method
# ---------------------------------------------------------------------------
class TestPredictHba1cKinetic:
    """Tests for predict_hba1c with method='kinetic'."""

    def test_kinetic_basic(self):
        result = predict_hba1c(126.0, method="kinetic")
        assert result["method"] == "kinetic"
        assert result["ci_lower"] is None
        assert result["ci_upper"] is None
        assert result["hba1c_pred"] > 3.0
        assert result["hba1c_pred"] < 20.0

    def test_kinetic_with_hgb(self):
        result = predict_hba1c(126.0, hgb=12.0, method="kinetic")
        assert result["warning"] is None

    def test_kinetic_without_hgb_warns(self):
        result = predict_hba1c(126.0, method="kinetic")
        assert result["warning"] is not None
        assert "hgb" in result["warning"]

    def test_kinetic_anemia_higher_estimate(self):
        """Lower hemoglobin (anemia) should give higher HbA1c estimate."""
        normal = predict_hba1c(126.0, hgb=14.0, method="kinetic")
        anemia = predict_hba1c(126.0, hgb=10.0, method="kinetic")
        assert anemia["hba1c_pred"] > normal["hba1c_pred"]


# ---------------------------------------------------------------------------
# Tests for predict_hba1c – Regression method
# ---------------------------------------------------------------------------
class TestPredictHba1cRegression:
    """Tests for predict_hba1c with method='regression'."""

    def test_regression_basic(self):
        result = predict_hba1c(
            126.0, tg=150.0, hdl=50.0, age=55.0, hgb=14.0, method="regression"
        )
        assert result["method"] == "regression"
        assert result["ci_lower"] is None
        assert result["ci_upper"] is None
        assert result["warning"] is None
        assert result["hba1c_pred"] > 3.0

    def test_regression_missing_tg_raises(self):
        with pytest.raises(ValueError, match="tg"):
            predict_hba1c(126.0, hdl=50.0, age=55.0, hgb=14.0, method="regression")

    def test_regression_missing_hdl_raises(self):
        with pytest.raises(ValueError, match="hdl"):
            predict_hba1c(126.0, tg=150.0, age=55.0, hgb=14.0, method="regression")

    def test_regression_missing_age_raises(self):
        with pytest.raises(ValueError, match="age"):
            predict_hba1c(126.0, tg=150.0, hdl=50.0, hgb=14.0, method="regression")

    def test_regression_missing_multiple_raises(self):
        with pytest.raises(ValueError, match="tg.*hdl.*age.*hgb"):
            predict_hba1c(126.0, method="regression")


# ---------------------------------------------------------------------------
# Tests for predict_hba1c – Hybrid method
# ---------------------------------------------------------------------------
class TestPredictHba1cHybrid:
    """Tests for predict_hba1c with method='hybrid'."""

    def test_hybrid_no_model_falls_back_to_adag(self):
        """When no model is found, hybrid falls back to ADAG."""
        result = predict_hba1c(126.0, method="hybrid")
        assert result["method"] == "adag"
        assert result["warning"] is not None
        assert "No trained ML model found" in result["warning"]

    @patch("hba1cE.predict._load_best_model")
    def test_hybrid_with_model(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([6.5])
        mock_load.return_value = mock_model

        result = predict_hba1c(
            126.0, tg=150.0, hdl=50.0, age=55.0, hgb=14.0, mcv=90.0,
            method="hybrid",
        )
        assert result["method"] == "hybrid"
        assert result["hba1c_pred"] == 6.5
        assert result["ci_lower"] == 6.0
        assert result["ci_upper"] == 7.0
        assert result["warning"] is None

    @patch("hba1cE.predict._load_best_model")
    def test_hybrid_partial_inputs_warns(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([6.5])
        mock_load.return_value = mock_model

        result = predict_hba1c(126.0, method="hybrid")
        assert result["method"] == "hybrid"
        assert result["warning"] is not None
        assert "Missing inputs" in result["warning"]

    @patch("hba1cE.predict._load_best_model")
    def test_hybrid_model_receives_correct_features(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([6.0])
        mock_load.return_value = mock_model

        predict_hba1c(
            126.0, tg=150.0, hdl=50.0, age=55.0, hgb=14.0, mcv=90.0,
            method="hybrid",
        )
        # Model should be called with a (1, 11) feature array
        call_args = mock_model.predict.call_args
        X = call_args[0][0]
        assert X.shape == (1, 11)


# ---------------------------------------------------------------------------
# Tests for input validation
# ---------------------------------------------------------------------------
class TestPredictInputValidation:
    """Tests for input validation in predict_hba1c."""

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Invalid method"):
            predict_hba1c(126.0, method="invalid")

    def test_negative_fpg(self):
        with pytest.raises(ValueError, match="negative"):
            predict_hba1c(-10.0, method="adag")

    def test_nan_fpg(self):
        with pytest.raises(ValueError, match="NaN"):
            predict_hba1c(float("nan"), method="adag")

    def test_low_fpg(self):
        with pytest.raises(ValueError, match="at least 40"):
            predict_hba1c(30.0, method="adag")

    def test_non_numeric_fpg(self):
        with pytest.raises(ValueError, match="numeric"):
            predict_hba1c("abc", method="adag")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests for _build_feature_vector
# ---------------------------------------------------------------------------
class TestBuildFeatureVector:
    """Tests for _build_feature_vector helper."""

    def test_all_inputs_provided(self):
        X = _build_feature_vector(126.0, 150.0, 50.0, 55.0, 14.0, 90.0)
        assert X.shape == (1, 11)
        assert X[0, 0] == 126.0  # fpg
        assert X[0, 1] == 150.0  # tg
        assert X[0, 2] == 50.0   # hdl

    def test_missing_inputs_use_defaults(self):
        X = _build_feature_vector(126.0, None, None, None, None, None)
        assert X.shape == (1, 11)
        assert X[0, 0] == 126.0
        assert X[0, 1] == 120.0   # default tg
        assert X[0, 2] == 52.0    # default hdl
        assert X[0, 3] == 50.0    # default age
        assert X[0, 4] == 14.0    # default hgb
        assert X[0, 5] == 90.0    # default mcv


# ---------------------------------------------------------------------------
# Tests for _load_best_model
# ---------------------------------------------------------------------------
class TestLoadBestModel:
    """Tests for _load_best_model helper."""

    def test_no_directory(self):
        result = _load_best_model(Path("/nonexistent/path"))
        assert result is None

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _load_best_model(Path(tmpdir))
            assert result is None

    def test_loads_best_model_file(self):
        from sklearn.linear_model import Ridge
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib
            model = Ridge(alpha=1.0)
            joblib.dump(model, Path(tmpdir) / "best_model.joblib")

            result = _load_best_model(Path(tmpdir))
            assert result is not None

    def test_loads_fallback_joblib(self):
        from sklearn.linear_model import Ridge
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib
            model = Ridge(alpha=1.0)
            joblib.dump(model, Path(tmpdir) / "ridge_model.joblib")

            result = _load_best_model(Path(tmpdir))
            assert result is not None
