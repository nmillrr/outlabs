"""
Tests for eGFR/data.py — read_xpt and clean_kidney_data functions.

Uses unittest.mock to mock pd.read_sas for happy-path tests (avoids the
need to construct a spec-compliant SAS XPORT V5 binary), and uses real
files for error-path tests.
"""

import os
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from eGFR.data import read_xpt, clean_kidney_data, generate_quality_report


# ---------------------------------------------------------------------------
# Tests for read_xpt
# ---------------------------------------------------------------------------

class TestReadXpt:
    """Tests for the read_xpt function."""

    # ── Happy-path tests (mock pd.read_sas) ─────────────────────────────

    def test_reads_valid_xpt(self, tmp_path):
        """read_xpt should return a DataFrame with the expected columns."""
        # Create a dummy file so the os.path.isfile check passes
        xpt_path = str(tmp_path / "mock.xpt")
        with open(xpt_path, "wb") as f:
            f.write(b"\x00" * 80)  # dummy bytes

        expected_df = pd.DataFrame({
            "SEQN": [1.0, 2.0, 3.0],
            "LBXSCR": [0.9, 1.2, 1.5],
            "RIDAGEYR": [45.0, 60.0, 75.0],
        })

        with patch("eGFR.data.pd.read_sas", return_value=expected_df) as mock_read:
            df = read_xpt(xpt_path)
            mock_read.assert_called_once_with(xpt_path, format="xport", encoding="utf-8")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "SEQN" in df.columns
        assert "LBXSCR" in df.columns
        assert "RIDAGEYR" in df.columns

    def test_values_match(self, tmp_path):
        """Column values should match those returned by the reader."""
        xpt_path = str(tmp_path / "mock.xpt")
        with open(xpt_path, "wb") as f:
            f.write(b"\x00" * 80)

        expected_df = pd.DataFrame({
            "SEQN": [1.0, 2.0, 3.0],
            "LBXSCR": [0.9, 1.2, 1.5],
            "RIDAGEYR": [45.0, 60.0, 75.0],
        })

        with patch("eGFR.data.pd.read_sas", return_value=expected_df):
            df = read_xpt(xpt_path)

        np.testing.assert_array_almost_equal(df["SEQN"].values, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(df["LBXSCR"].values, [0.9, 1.2, 1.5])
        np.testing.assert_array_almost_equal(df["RIDAGEYR"].values, [45.0, 60.0, 75.0])

    def test_returns_dataframe_type(self, tmp_path):
        """Return type should be pd.DataFrame."""
        xpt_path = str(tmp_path / "mock.xpt")
        with open(xpt_path, "wb") as f:
            f.write(b"\x00" * 80)

        expected_df = pd.DataFrame({"A": [1.0]})

        with patch("eGFR.data.pd.read_sas", return_value=expected_df):
            result = read_xpt(xpt_path)

        assert type(result) is pd.DataFrame

    # ── Error-path tests (no mocking — real filesystem) ─────────────────

    def test_missing_file_raises_file_not_found(self):
        """read_xpt should raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="XPT file not found"):
            read_xpt("/tmp/does_not_exist_abc123.xpt")

    def test_corrupt_file_raises_value_error(self, tmp_path):
        """read_xpt should raise ValueError for a file that isn't valid XPT."""
        bad_path = str(tmp_path / "bad.xpt")
        with open(bad_path, "w") as f:
            f.write("this is not a valid XPT file at all")

        with pytest.raises(ValueError, match="Failed to parse"):
            read_xpt(bad_path)

    def test_read_sas_exception_wrapped(self, tmp_path):
        """Any exception from pd.read_sas should be wrapped in ValueError."""
        xpt_path = str(tmp_path / "mock.xpt")
        with open(xpt_path, "wb") as f:
            f.write(b"\x00" * 80)

        with patch("eGFR.data.pd.read_sas", side_effect=RuntimeError("boom")):
            with pytest.raises(ValueError, match="Failed to parse"):
                read_xpt(xpt_path)


# ---------------------------------------------------------------------------
# Helper fixtures for clean_kidney_data tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_biopro():
    """Minimal BIOPRO DataFrame with valid creatinine values."""
    return pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5],
        "LBXSCR": [0.9, 1.2, 1.5, 0.7, 2.0],
    })


@pytest.fixture()
def sample_demo():
    """Minimal DEMO DataFrame with age and sex."""
    return pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5],
        "RIDAGEYR": [45, 60, 75, 30, 50],
        "RIAGENDR": [1, 2, 1, 2, 1],  # 1=male, 2=female
    })


@pytest.fixture()
def sample_bmx():
    """Minimal BMX DataFrame with weight and height."""
    return pd.DataFrame({
        "SEQN": [1, 2, 3, 4, 5],
        "BMXWT": [80.0, 65.0, 90.0, 55.0, 75.0],
        "BMXHT": [175.0, 160.0, 180.0, 165.0, 170.0],
    })


# ---------------------------------------------------------------------------
# Tests for clean_kidney_data
# ---------------------------------------------------------------------------

class TestCleanKidneyData:
    """Tests for the clean_kidney_data function."""

    # ── Basic merging and column rename ────────────────────────────────

    def test_returns_dataframe(self, sample_biopro, sample_demo, sample_bmx):
        """Should return a pandas DataFrame."""
        result = clean_kidney_data(sample_biopro, sample_demo, sample_bmx)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, sample_biopro, sample_demo, sample_bmx):
        """Output should have standardized column names."""
        result = clean_kidney_data(sample_biopro, sample_demo, sample_bmx)
        expected_cols = {"seqn", "cr_mgdl", "age_years", "sex", "weight_kg", "height_cm"}
        assert expected_cols == set(result.columns)

    def test_merges_on_seqn(self, sample_biopro, sample_demo, sample_bmx):
        """All rows with matching SEQN should be merged."""
        result = clean_kidney_data(sample_biopro, sample_demo, sample_bmx)
        # All 5 subjects share the same SEQNs and all are valid adults
        assert len(result) == 5

    def test_inner_join_drops_non_matching(self):
        """Subjects only in one dataset should be dropped (inner join)."""
        bio = pd.DataFrame({"SEQN": [1, 2], "LBXSCR": [1.0, 1.2]})
        dem = pd.DataFrame({"SEQN": [1, 3], "RIDAGEYR": [50, 60], "RIAGENDR": [1, 2]})
        bmx = pd.DataFrame({"SEQN": [1, 2, 3], "BMXWT": [80, 70, 60], "BMXHT": [175, 170, 165]})

        result = clean_kidney_data(bio, dem, bmx)
        # Only SEQN=1 is in all three
        assert len(result) == 1
        assert result["seqn"].iloc[0] == 1

    def test_column_values_correct(self, sample_biopro, sample_demo, sample_bmx):
        """Values should be correctly mapped after rename."""
        result = clean_kidney_data(sample_biopro, sample_demo, sample_bmx)
        row = result.loc[result["seqn"] == 1].iloc[0]
        assert row["cr_mgdl"] == pytest.approx(0.9)
        assert row["age_years"] == 45
        assert row["sex"] == 1
        assert row["weight_kg"] == pytest.approx(80.0)
        assert row["height_cm"] == pytest.approx(175.0)

    # ── Outlier removal ───────────────────────────────────────────────

    def test_removes_low_creatinine(self):
        """Creatinine < 0.2 mg/dL should be removed."""
        bio = pd.DataFrame({"SEQN": [1, 2], "LBXSCR": [0.1, 1.0]})
        dem = pd.DataFrame({"SEQN": [1, 2], "RIDAGEYR": [50, 50], "RIAGENDR": [1, 1]})
        bmx = pd.DataFrame({"SEQN": [1, 2], "BMXWT": [80, 80], "BMXHT": [175, 175]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = clean_kidney_data(bio, dem, bmx)

        assert len(result) == 1
        assert result["seqn"].iloc[0] == 2

    def test_removes_high_creatinine(self):
        """Creatinine > 15 mg/dL should be removed."""
        bio = pd.DataFrame({"SEQN": [1, 2], "LBXSCR": [1.0, 16.0]})
        dem = pd.DataFrame({"SEQN": [1, 2], "RIDAGEYR": [50, 50], "RIAGENDR": [1, 1]})
        bmx = pd.DataFrame({"SEQN": [1, 2], "BMXWT": [80, 80], "BMXHT": [175, 175]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = clean_kidney_data(bio, dem, bmx)

        assert len(result) == 1
        assert result["seqn"].iloc[0] == 1

    def test_removes_minors(self):
        """Age < 18 should be removed."""
        bio = pd.DataFrame({"SEQN": [1, 2], "LBXSCR": [1.0, 1.0]})
        dem = pd.DataFrame({"SEQN": [1, 2], "RIDAGEYR": [17, 25], "RIAGENDR": [1, 1]})
        bmx = pd.DataFrame({"SEQN": [1, 2], "BMXWT": [70, 80], "BMXHT": [170, 175]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = clean_kidney_data(bio, dem, bmx)

        assert len(result) == 1
        assert result["seqn"].iloc[0] == 2

    def test_outlier_warning_issued(self):
        """A warning should be issued when outliers are removed."""
        bio = pd.DataFrame({"SEQN": [1, 2], "LBXSCR": [0.05, 1.0]})
        dem = pd.DataFrame({"SEQN": [1, 2], "RIDAGEYR": [50, 50], "RIAGENDR": [1, 1]})
        bmx = pd.DataFrame({"SEQN": [1, 2], "BMXWT": [80, 80], "BMXHT": [175, 175]})

        with pytest.warns(UserWarning, match="Removed 1 rows as physiologic outliers"):
            clean_kidney_data(bio, dem, bmx)

    def test_keeps_boundary_creatinine(self):
        """Creatinine exactly 0.2 and 15.0 should be kept (inclusive)."""
        bio = pd.DataFrame({"SEQN": [1, 2], "LBXSCR": [0.2, 15.0]})
        dem = pd.DataFrame({"SEQN": [1, 2], "RIDAGEYR": [50, 50], "RIAGENDR": [1, 1]})
        bmx = pd.DataFrame({"SEQN": [1, 2], "BMXWT": [80, 80], "BMXHT": [175, 175]})

        result = clean_kidney_data(bio, dem, bmx)
        assert len(result) == 2

    # ── IDMS correction ───────────────────────────────────────────────

    def test_idms_correction_applied(self):
        """When apply_idms_correction=True, creatinine should be multiplied by 0.95."""
        bio = pd.DataFrame({"SEQN": [1], "LBXSCR": [1.0]})
        dem = pd.DataFrame({"SEQN": [1], "RIDAGEYR": [50], "RIAGENDR": [1]})
        bmx = pd.DataFrame({"SEQN": [1], "BMXWT": [80], "BMXHT": [175]})

        result = clean_kidney_data(bio, dem, bmx, apply_idms_correction=True)
        assert result["cr_mgdl"].iloc[0] == pytest.approx(0.95)

    def test_idms_correction_not_applied_by_default(self):
        """By default, creatinine should NOT be corrected."""
        bio = pd.DataFrame({"SEQN": [1], "LBXSCR": [1.0]})
        dem = pd.DataFrame({"SEQN": [1], "RIDAGEYR": [50], "RIAGENDR": [1]})
        bmx = pd.DataFrame({"SEQN": [1], "BMXWT": [80], "BMXHT": [175]})

        result = clean_kidney_data(bio, dem, bmx)
        assert result["cr_mgdl"].iloc[0] == pytest.approx(1.0)

    # ── NaN / incomplete cases ────────────────────────────────────────

    def test_drops_nan_in_core_columns(self):
        """Rows with NaN in core columns should be dropped."""
        bio = pd.DataFrame({"SEQN": [1, 2], "LBXSCR": [1.0, np.nan]})
        dem = pd.DataFrame({"SEQN": [1, 2], "RIDAGEYR": [50, 50], "RIAGENDR": [1, 1]})
        bmx = pd.DataFrame({"SEQN": [1, 2], "BMXWT": [80, 80], "BMXHT": [175, 175]})

        result = clean_kidney_data(bio, dem, bmx)
        assert len(result) == 1

    def test_index_reset_after_cleaning(self):
        """Index should be reset to 0..n-1 after cleaning."""
        bio = pd.DataFrame({"SEQN": [1, 2, 3], "LBXSCR": [0.05, 1.0, 1.5]})
        dem = pd.DataFrame({"SEQN": [1, 2, 3], "RIDAGEYR": [50, 50, 50], "RIAGENDR": [1, 1, 1]})
        bmx = pd.DataFrame({"SEQN": [1, 2, 3], "BMXWT": [80, 80, 80], "BMXHT": [175, 175, 175]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = clean_kidney_data(bio, dem, bmx)

        assert list(result.index) == list(range(len(result)))

    # ── Cystatin C optional merge ────────────────────────────────────

    def test_cystatin_merge(self):
        """Cystatin C column should appear when cystatin_df is provided."""
        bio = pd.DataFrame({"SEQN": [1, 2], "LBXSCR": [1.0, 1.2]})
        dem = pd.DataFrame({"SEQN": [1, 2], "RIDAGEYR": [50, 60], "RIAGENDR": [1, 2]})
        bmx = pd.DataFrame({"SEQN": [1, 2], "BMXWT": [80, 65], "BMXHT": [175, 160]})
        cys = pd.DataFrame({"SEQN": [1, 2], "SSPRT": [0.8, 1.1]})

        result = clean_kidney_data(bio, dem, bmx, cystatin_df=cys)
        assert "cystatin_c_mgL" in result.columns
        assert result["cystatin_c_mgL"].iloc[0] == pytest.approx(0.8)

    def test_no_cystatin_column_without_df(self, sample_biopro, sample_demo, sample_bmx):
        """Without cystatin_df, the cystatin column should be absent."""
        result = clean_kidney_data(sample_biopro, sample_demo, sample_bmx)
        assert "cystatin_c_mgL" not in result.columns

    # ── Validation errors ─────────────────────────────────────────────

    def test_missing_biopro_column_raises(self, sample_demo, sample_bmx):
        """Missing LBXSCR in biopro_df should raise ValueError."""
        bad_bio = pd.DataFrame({"SEQN": [1]})
        with pytest.raises(ValueError, match="biopro_df is missing"):
            clean_kidney_data(bad_bio, sample_demo, sample_bmx)

    def test_missing_demo_column_raises(self, sample_biopro, sample_bmx):
        """Missing RIDAGEYR in demo_df should raise ValueError."""
        bad_dem = pd.DataFrame({"SEQN": [1], "RIAGENDR": [1]})
        with pytest.raises(ValueError, match="demo_df is missing"):
            clean_kidney_data(sample_biopro, bad_dem, sample_bmx)

    def test_missing_bmx_column_raises(self, sample_biopro, sample_demo):
        """Missing BMXWT in bmx_df should raise ValueError."""
        bad_bmx = pd.DataFrame({"SEQN": [1], "BMXHT": [175]})
        with pytest.raises(ValueError, match="bmx_df is missing"):
            clean_kidney_data(sample_biopro, sample_demo, bad_bmx)

    def test_missing_cystatin_column_raises(self, sample_biopro, sample_demo, sample_bmx):
        """Missing SSPRT in cystatin_df should raise ValueError."""
        bad_cys = pd.DataFrame({"SEQN": [1]})
        with pytest.raises(ValueError, match="cystatin_df is missing"):
            clean_kidney_data(sample_biopro, sample_demo, sample_bmx, cystatin_df=bad_cys)


# ---------------------------------------------------------------------------
# Helper: build a cleaned DataFrame for quality-report tests
# ---------------------------------------------------------------------------

def _make_cleaned_df(n=5, include_cystatin=False):
    """Return a DataFrame that looks like output of clean_kidney_data."""
    data = {
        "seqn": list(range(1, n + 1)),
        "cr_mgdl": [0.9, 1.2, 1.5, 0.7, 2.0][:n],
        "age_years": [45, 60, 75, 30, 50][:n],
        "sex": [1, 2, 1, 2, 1][:n],
        "weight_kg": [80.0, 65.0, 90.0, 55.0, 75.0][:n],
        "height_cm": [175.0, 160.0, 180.0, 165.0, 170.0][:n],
    }
    if include_cystatin:
        data["cystatin_c_mgL"] = [0.8, 1.1, 0.9, 1.0, 1.2][:n]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests for generate_quality_report
# ---------------------------------------------------------------------------

class TestGenerateQualityReport:
    """Tests for the generate_quality_report function."""

    def test_returns_string(self, tmp_path):
        """Should return the report text as a string."""
        df = _make_cleaned_df()
        report = generate_quality_report(df, str(tmp_path / "report.txt"))
        assert isinstance(report, str)

    def test_saves_file(self, tmp_path):
        """Report should be saved to the specified path."""
        out = str(tmp_path / "report.txt")
        generate_quality_report(_make_cleaned_df(), out)
        assert os.path.isfile(out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "DATA QUALITY REPORT" in content

    def test_creates_parent_dirs(self, tmp_path):
        """Should create parent directories if they don't exist."""
        out = str(tmp_path / "sub" / "dir" / "report.txt")
        generate_quality_report(_make_cleaned_df(), out)
        assert os.path.isfile(out)

    def test_record_count(self, tmp_path):
        """Report should include the correct total record count."""
        df = _make_cleaned_df(n=5)
        report = generate_quality_report(df, str(tmp_path / "r.txt"))
        assert "Total records: 5" in report

    def test_descriptive_stats_present(self, tmp_path):
        """Report should include mean/SD for creatinine, age, weight, height."""
        report = generate_quality_report(_make_cleaned_df(), str(tmp_path / "r.txt"))
        assert "Creatinine (mg/dL): mean=" in report
        assert "Age (years): mean=" in report
        assert "Weight (kg): mean=" in report
        assert "Height (cm): mean=" in report

    def test_descriptive_stats_values(self, tmp_path):
        """Mean and SD should be numerically correct."""
        df = _make_cleaned_df(n=5)
        report = generate_quality_report(df, str(tmp_path / "r.txt"))
        # cr_mgdl = [0.9, 1.2, 1.5, 0.7, 2.0] → mean ≈ 1.26
        expected_cr_mean = np.mean([0.9, 1.2, 1.5, 0.7, 2.0])
        assert f"mean={expected_cr_mean:.2f}" in report

    def test_ckd_stage_distribution_present(self, tmp_path):
        """Report should have CKD stage distribution section."""
        report = generate_quality_report(_make_cleaned_df(), str(tmp_path / "r.txt"))
        assert "CKD Stage Distribution (CKD-EPI 2021)" in report
        for stage in ["G1", "G2", "G3a", "G3b", "G4", "G5"]:
            assert stage in report

    def test_sex_distribution_present(self, tmp_path):
        """Report should include sex distribution."""
        report = generate_quality_report(_make_cleaned_df(), str(tmp_path / "r.txt"))
        assert "Sex Distribution" in report
        assert "Male:" in report
        assert "Female:" in report

    def test_sex_counts_correct(self, tmp_path):
        """Sex counts should match the input data."""
        df = _make_cleaned_df(n=5)  # sex = [1, 2, 1, 2, 1] → 3 male, 2 female
        report = generate_quality_report(df, str(tmp_path / "r.txt"))
        assert "Male:   3 (60.0%)" in report
        assert "Female: 2 (40.0%)" in report

    def test_cystatin_c_included_when_present(self, tmp_path):
        """Cystatin C stats should appear when column is present."""
        df = _make_cleaned_df(include_cystatin=True)
        report = generate_quality_report(df, str(tmp_path / "r.txt"))
        assert "Cystatin C (mg/L): mean=" in report

    def test_cystatin_c_absent_when_not_present(self, tmp_path):
        """Cystatin C stats should NOT appear when column is absent."""
        df = _make_cleaned_df(include_cystatin=False)
        report = generate_quality_report(df, str(tmp_path / "r.txt"))
        assert "Cystatin C" not in report

    def test_missing_column_raises(self, tmp_path):
        """Missing required columns should raise ValueError."""
        bad_df = pd.DataFrame({"cr_mgdl": [1.0]})
        with pytest.raises(ValueError, match="df is missing"):
            generate_quality_report(bad_df, str(tmp_path / "r.txt"))

    def test_report_saved_matches_returned(self, tmp_path):
        """Saved file content should match the returned string."""
        out = str(tmp_path / "r.txt")
        returned = generate_quality_report(_make_cleaned_df(), out)
        with open(out, encoding="utf-8") as f:
            saved = f.read()
        assert returned == saved

