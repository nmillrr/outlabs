"""Tests for eGFR/train.py — stratified_split function."""

import numpy as np
import pandas as pd
import pytest

from eGFR.train import stratified_split


# ---------------------------------------------------------------------------
# Helpers — synthetic clinical DataFrames
# ---------------------------------------------------------------------------

def _make_clinical_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic clinical DataFrame with realistic ranges.

    Generates samples spanning typical eGFR ranges so that stratification
    bins are populated.
    """
    rng = np.random.default_rng(seed)
    # Mix of creatinine values to cover wide eGFR range
    cr_values = np.concatenate([
        rng.uniform(0.5, 1.0, n // 2),   # normal → high eGFR
        rng.uniform(1.5, 5.0, n // 4),   # moderate CKD
        rng.uniform(5.0, 12.0, n // 4),  # severe CKD / low eGFR
    ])
    ages = rng.integers(18, 85, size=n)
    sexes = rng.choice([1, 2], size=n)
    weights = rng.uniform(50, 120, size=n)
    heights = rng.uniform(150, 195, size=n)

    return pd.DataFrame({
        "cr_mgdl": cr_values,
        "age_years": ages,
        "sex": sexes,
        "weight_kg": weights,
        "height_cm": heights,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStratifiedSplitBasic:
    """Core functionality tests."""

    def test_returns_four_outputs(self):
        df = _make_clinical_df(60)
        result = stratified_split(df)
        assert len(result) == 4

    def test_default_split_ratio(self):
        """Default test_size=0.3 → ~30% test, ~70% train."""
        df = _make_clinical_df(100)
        X_train, X_test, y_train, y_test = stratified_split(df)

        total = len(X_train) + len(X_test)
        assert total <= 100  # may lose some NaN rows
        assert total > 0

        test_frac = len(X_test) / total
        assert 0.2 <= test_frac <= 0.4  # close to 0.3

    def test_custom_test_size(self):
        df = _make_clinical_df(100)
        X_train, X_test, y_train, y_test = stratified_split(df, test_size=0.2)

        total = len(X_train) + len(X_test)
        test_frac = len(X_test) / total
        assert 0.1 <= test_frac <= 0.3  # close to 0.2

    def test_x_y_lengths_match(self):
        df = _make_clinical_df(80)
        X_train, X_test, y_train, y_test = stratified_split(df)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_no_index_overlap(self):
        """Train and test indices should not overlap."""
        df = _make_clinical_df(80)
        X_train, X_test, _, _ = stratified_split(df)
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0

    def test_feature_columns_present(self):
        """X_train/X_test should contain the expected feature columns."""
        df = _make_clinical_df(60)
        X_train, X_test, _, _ = stratified_split(df)
        expected_cols = {
            "cr_mgdl", "age_years", "sex_numeric", "weight_kg", "height_cm",
            "bmi", "egfr_ckd_epi_2021", "egfr_mdrd", "crcl_cockcroft_gault",
            "inv_creatinine", "log_creatinine", "age_cr_interaction",
        }
        assert expected_cols.issubset(set(X_train.columns))
        assert expected_cols.issubset(set(X_test.columns))


class TestReproducibility:
    """Determinism via random_state."""

    def test_same_seed_same_split(self):
        df = _make_clinical_df(80)
        a = stratified_split(df, random_state=123)
        b = stratified_split(df, random_state=123)
        pd.testing.assert_frame_equal(a[0], b[0])
        pd.testing.assert_frame_equal(a[1], b[1])

    def test_different_seed_different_split(self):
        df = _make_clinical_df(80)
        a = stratified_split(df, random_state=1)
        b = stratified_split(df, random_state=2)
        # Very unlikely to be identical
        assert not a[0].equals(b[0])


class TestStratification:
    """Verify that the split is stratified by eGFR bin."""

    @staticmethod
    def _bin_series(y: pd.Series) -> pd.Series:
        bins = [0, 15, 30, 45, 60, 90, float("inf")]
        labels = ["<15", "15-29", "30-44", "45-59", "60-89", ">=90"]
        return pd.cut(y, bins=bins, labels=labels, right=False)

    def test_bins_represented_in_both_splits(self):
        """Bins that have ≥2 samples should appear in both train and test."""
        df = _make_clinical_df(200, seed=0)
        _, _, y_train, y_test = stratified_split(df, random_state=0)

        train_bins = set(self._bin_series(y_train).dropna().unique())
        test_bins = set(self._bin_series(y_test).dropna().unique())
        # Every test bin should also appear in train
        assert test_bins.issubset(train_bins)


class TestEdgeCases:
    """Error handling and edge cases."""

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["cr_mgdl", "age_years", "sex", "weight_kg", "height_cm"])
        with pytest.raises(ValueError, match="empty"):
            stratified_split(df)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"cr_mgdl": [1.0], "age_years": [50]})
        with pytest.raises(ValueError, match="Missing required columns"):
            stratified_split(df)


# ---------------------------------------------------------------------------
# train_ridge tests
# ---------------------------------------------------------------------------

from eGFR.train import train_ridge


class TestTrainRidge:
    """Tests for train_ridge function."""

    def test_returns_ridge_model(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([10, 20, 30, 40])
        model = train_ridge(X, y)
        assert hasattr(model, "predict")
        assert hasattr(model, "coef_")

    def test_predictions_reasonable(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = train_ridge(X, y)
        pred = model.predict(np.array([[3]]))
        assert abs(pred[0] - 6) < 1.0

    def test_custom_alpha(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 2, 3, 4])
        model = train_ridge(X, y, alpha=10.0)
        assert hasattr(model, "predict")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            train_ridge(np.array([]).reshape(0, 1), np.array([]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            train_ridge(np.array([[1], [2], [3]]), np.array([1, 2]))


# ---------------------------------------------------------------------------
# train_random_forest tests
# ---------------------------------------------------------------------------

from eGFR.train import train_random_forest


class TestTrainRandomForest:
    """Tests for train_random_forest function."""

    def test_returns_rf_model(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([10, 20, 30, 40])
        model = train_random_forest(X, y, n_estimators=10)
        assert hasattr(model, "predict")
        assert hasattr(model, "feature_importances_")

    def test_predictions_work(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = train_random_forest(X, y, n_estimators=10)
        pred = model.predict(np.array([[3]]))
        assert pred[0] > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            train_random_forest(np.array([]).reshape(0, 1), np.array([]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            train_random_forest(np.array([[1], [2], [3]]), np.array([1, 2]))


# ---------------------------------------------------------------------------
# save_model tests
# ---------------------------------------------------------------------------

from eGFR.train import save_model


class TestSaveModel:
    """Tests for save_model function."""

    def test_saves_and_loads(self, tmp_path):
        import joblib
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        model = train_ridge(X, y)
        filepath = str(tmp_path / "model.joblib")
        save_model(model, filepath)

        loaded = joblib.load(filepath)
        pred_orig = model.predict(np.array([[2]]))
        pred_loaded = loaded.predict(np.array([[2]]))
        assert pred_orig[0] == pytest.approx(pred_loaded[0])

    def test_creates_parent_dirs(self, tmp_path):
        filepath = str(tmp_path / "nested" / "dir" / "model.joblib")
        save_model("dummy_model", filepath)
        import os
        assert os.path.isfile(filepath)

    def test_empty_filepath_raises(self):
        with pytest.raises(ValueError, match="empty"):
            save_model("model", "")

