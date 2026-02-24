"""Tests for eGFR/train.py — stratified_split and cross_validate_model."""

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
    """Tests for cross_validate_model."""

    @staticmethod
    def _make_ridge():
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    @staticmethod
    def _make_data(n=100, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.1, n)
        return X, y

    def test_returns_expected_keys(self):
        X, y = self._make_data()
        result = cross_validate_model(self._make_ridge(), X, y, n_splits=5)
        assert set(result.keys()) == {"RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std"}

    def test_values_are_float(self):
        X, y = self._make_data()
        result = cross_validate_model(self._make_ridge(), X, y, n_splits=5)
        for v in result.values():
            assert isinstance(v, float)

    def test_values_non_negative(self):
        X, y = self._make_data()
        result = cross_validate_model(self._make_ridge(), X, y, n_splits=5)
        for v in result.values():
            assert v >= 0.0

    def test_rmse_ge_mae(self):
        """RMSE is always >= MAE for the same predictions."""
        X, y = self._make_data()
        result = cross_validate_model(self._make_ridge(), X, y, n_splits=5)
        assert result["RMSE_mean"] >= result["MAE_mean"]

    def test_custom_n_splits(self):
        X, y = self._make_data(50)
        result = cross_validate_model(self._make_ridge(), X, y, n_splits=3)
        assert "RMSE_mean" in result

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cross_validate_model(self._make_ridge(), np.array([]), np.array([]))

    def test_shape_mismatch_raises(self):
        X = np.ones((10, 3))
        y = np.ones(5)
        with pytest.raises(ValueError, match="row counts differ"):
            cross_validate_model(self._make_ridge(), X, y)

    def test_n_splits_too_small_raises(self):
        X, y = self._make_data(20)
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            cross_validate_model(self._make_ridge(), X, y, n_splits=1)
