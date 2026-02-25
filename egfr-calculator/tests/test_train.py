"""Tests for eGFR/train.py — stratified_split function."""

import numpy as np
import pandas as pd
import pytest

from eGFR.train import stratified_split, cross_validate_model


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
# cross_validate_model tests
# ---------------------------------------------------------------------------

class TestCrossValidateModel:
    """Tests for the cross_validate_model wrapper."""

    @staticmethod
    def _make_data(n=50, seed=42):
        """Simple synthetic regression data for CV testing."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.standard_normal(n) * 0.1
        return X, y

    def test_returns_expected_keys(self):
        from sklearn.linear_model import Ridge
        X, y = self._make_data()
        result = cross_validate_model(Ridge(), X, y, n_splits=3)
        expected_keys = {"RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std"}
        assert set(result.keys()) == expected_keys

    def test_values_are_nonnegative_floats(self):
        from sklearn.linear_model import Ridge
        X, y = self._make_data()
        result = cross_validate_model(Ridge(), X, y, n_splits=3)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float"
            assert val >= 0.0, f"{key} is negative"

    def test_rmse_ge_mae(self):
        """RMSE should always be >= MAE."""
        from sklearn.linear_model import Ridge
        X, y = self._make_data()
        result = cross_validate_model(Ridge(), X, y, n_splits=5)
        assert result["RMSE_mean"] >= result["MAE_mean"]

    def test_custom_n_splits(self):
        from sklearn.linear_model import Ridge
        X, y = self._make_data(n=20)
        result = cross_validate_model(Ridge(), X, y, n_splits=4)
        assert "RMSE_mean" in result

    def test_empty_data_raises(self):
        from sklearn.linear_model import Ridge
        with pytest.raises(ValueError, match="empty"):
            cross_validate_model(Ridge(), np.array([]).reshape(0, 1), np.array([]))

    def test_shape_mismatch_raises(self):
        from sklearn.linear_model import Ridge
        X = np.ones((10, 2))
        y = np.ones(5)
        with pytest.raises(ValueError, match="row counts differ"):
            cross_validate_model(Ridge(), X, y)

    def test_n_splits_exceeds_samples_raises(self):
        from sklearn.linear_model import Ridge
        X, y = self._make_data(n=5)
        with pytest.raises(ValueError, match="n_splits.*cannot exceed"):
            cross_validate_model(Ridge(), X, y, n_splits=10)

    def test_original_model_unchanged(self):
        """The original model should not be fitted after CV."""
        from sklearn.linear_model import Ridge
        model = Ridge()
        X, y = self._make_data()
        cross_validate_model(model, X, y, n_splits=3)
        # An unfitted Ridge has no coef_ attribute
        assert not hasattr(model, "coef_")

