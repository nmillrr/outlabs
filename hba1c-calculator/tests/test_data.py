"""
Unit tests for hba1cE.data module.

Tests XPT file parsing functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from hba1cE.data import (
    read_xpt,
    clean_glycemic_data,
    generate_quality_report,
    load_external_kaggle_diabetes,
)


class TestReadXpt:
    """Tests for the read_xpt function."""

    def test_read_xpt_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            read_xpt("nonexistent_file.XPT")
        
        assert "XPT file not found" in str(exc_info.value)
        assert "nonexistent_file.XPT" in str(exc_info.value)

    def test_read_xpt_invalid_file(self, tmp_path):
        """Test that ValueError is raised for invalid XPT file."""
        # Create a file with invalid content
        invalid_file = tmp_path / "invalid.XPT"
        invalid_file.write_text("This is not a valid XPT file")
        
        with pytest.raises(ValueError) as exc_info:
            read_xpt(str(invalid_file))
        
        assert "Failed to parse XPT file" in str(exc_info.value)

    def test_read_xpt_with_mocked_pandas(self, tmp_path):
        """Test reading a valid XPT file using mock."""
        # Create a temporary file (empty but exists)
        xpt_path = tmp_path / "mock_ghb.xpt"
        xpt_path.write_bytes(b"dummy content")
        
        # Create expected DataFrame
        expected_df = pd.DataFrame({
            "SEQN": [1, 2, 3, 4, 5],
            "LBXGH": [5.4, 5.7, 6.5, 7.2, 8.1],
        })
        
        # Mock pandas.read_sas to return our expected DataFrame
        with patch("hba1cE.data.pd.read_sas", return_value=expected_df) as mock_read:
            result_df = read_xpt(str(xpt_path))
            
            # Verify pandas.read_sas was called with correct arguments
            mock_read.assert_called_once_with(str(xpt_path), format="xport")
            
            # Verify result
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 5
            assert "SEQN" in result_df.columns
            assert "LBXGH" in result_df.columns

    def test_read_xpt_returns_dataframe(self, tmp_path):
        """Test that read_xpt returns a pandas DataFrame."""
        # Create a simple mock file
        xpt_path = tmp_path / "simple.xpt"
        xpt_path.write_bytes(b"dummy content")
        
        expected_df = pd.DataFrame({
            "COL1": [1.0, 2.0, 3.0],
            "COL2": [4.0, 5.0, 6.0],
        })
        
        with patch("hba1cE.data.pd.read_sas", return_value=expected_df):
            result = read_xpt(str(xpt_path))
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

    def test_read_xpt_propagates_parse_error(self, tmp_path):
        """Test that parse errors from pandas are wrapped in ValueError."""
        # Create a dummy file
        xpt_path = tmp_path / "broken.xpt"
        xpt_path.write_bytes(b"broken content")
        
        # Mock pandas.read_sas to raise an exception
        with patch("hba1cE.data.pd.read_sas", side_effect=Exception("Parse error")):
            with pytest.raises(ValueError) as exc_info:
                read_xpt(str(xpt_path))
            
            assert "Failed to parse XPT file" in str(exc_info.value)
            assert "Parse error" in str(exc_info.value)


class TestCleanGlycemicData:
    """Tests for the clean_glycemic_data function."""

    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for testing."""
        # GHB - Glycohemoglobin (HbA1c)
        ghb_df = pd.DataFrame({
            "SEQN": [1, 2, 3, 4, 5],
            "LBXGH": [5.4, 5.7, 6.5, 7.2, 8.1],
        })

        # GLU - Fasting Plasma Glucose
        glu_df = pd.DataFrame({
            "SEQN": [1, 2, 3, 4, 5],
            "LBXGLU": [95.0, 105.0, 126.0, 150.0, 180.0],
        })

        # TRIGLY - Triglycerides
        trigly_df = pd.DataFrame({
            "SEQN": [1, 2, 3, 4, 5],
            "LBXTR": [100.0, 150.0, 200.0, 250.0, 300.0],
        })

        # HDL - HDL Cholesterol
        hdl_df = pd.DataFrame({
            "SEQN": [1, 2, 3, 4, 5],
            "LBDHDD": [50.0, 55.0, 45.0, 40.0, 35.0],
        })

        # CBC - Complete Blood Count
        cbc_df = pd.DataFrame({
            "SEQN": [1, 2, 3, 4, 5],
            "LBXHGB": [14.0, 13.5, 12.0, 15.0, 11.0],
            "LBXMCVSI": [90.0, 85.0, 92.0, 88.0, 95.0],
        })

        # DEMO - Demographics
        demo_df = pd.DataFrame({
            "SEQN": [1, 2, 3, 4, 5],
            "RIDAGEYR": [45, 60, 35, 70, 55],
            "RIAGENDR": [1, 2, 1, 2, 1],  # 1=male, 2=female
        })

        return ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df

    def test_clean_glycemic_data_basic_merge(self, sample_dataframes):
        """Test that datasets are properly merged on SEQN."""
        ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df = sample_dataframes
        result = clean_glycemic_data(ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_clean_glycemic_data_column_names(self, sample_dataframes):
        """Test that columns are renamed correctly."""
        ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df = sample_dataframes
        result = clean_glycemic_data(ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df)

        expected_columns = [
            "hba1c_percent", "fpg_mgdl", "tg_mgdl", "hdl_mgdl",
            "hgb_gdl", "mcv_fl", "age_years", "sex"
        ]
        assert set(result.columns) == set(expected_columns)
        assert "SEQN" not in result.columns

    def test_clean_glycemic_data_removes_hba1c_outliers(self, sample_dataframes):
        """Test that HbA1c outliers are removed."""
        ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df = sample_dataframes

        # Add outlier values
        ghb_df = pd.concat([ghb_df, pd.DataFrame({
            "SEQN": [6, 7],
            "LBXGH": [2.5, 22.0],  # Both outside 3-20% range
        })], ignore_index=True)

        # Add matching rows to other dataframes
        for df_name in ["glu_df", "trigly_df", "hdl_df", "cbc_df", "demo_df"]:
            df = locals()[df_name]
            if df_name == "glu_df":
                glu_df = pd.concat([df, pd.DataFrame({"SEQN": [6, 7], "LBXGLU": [100.0, 100.0]})], ignore_index=True)
            elif df_name == "trigly_df":
                trigly_df = pd.concat([df, pd.DataFrame({"SEQN": [6, 7], "LBXTR": [150.0, 150.0]})], ignore_index=True)
            elif df_name == "hdl_df":
                hdl_df = pd.concat([df, pd.DataFrame({"SEQN": [6, 7], "LBDHDD": [50.0, 50.0]})], ignore_index=True)
            elif df_name == "cbc_df":
                cbc_df = pd.concat([df, pd.DataFrame({"SEQN": [6, 7], "LBXHGB": [14.0, 14.0], "LBXMCVSI": [90.0, 90.0]})], ignore_index=True)
            elif df_name == "demo_df":
                demo_df = pd.concat([df, pd.DataFrame({"SEQN": [6, 7], "RIDAGEYR": [50, 50], "RIAGENDR": [1, 1]})], ignore_index=True)

        result = clean_glycemic_data(ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df)

        # Should still have 5 rows (outliers removed)
        assert len(result) == 5
        assert all(result["hba1c_percent"] >= 3)
        assert all(result["hba1c_percent"] <= 20)

    def test_clean_glycemic_data_removes_fpg_outliers(self, sample_dataframes):
        """Test that FPG outliers are removed."""
        ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df = sample_dataframes

        # Add outlier values
        glu_df = pd.concat([glu_df, pd.DataFrame({
            "SEQN": [6, 7],
            "LBXGLU": [30.0, 650.0],  # Both outside 40-600 range
        })], ignore_index=True)

        # Add matching rows to other dataframes
        ghb_df = pd.concat([ghb_df, pd.DataFrame({"SEQN": [6, 7], "LBXGH": [5.5, 5.5]})], ignore_index=True)
        trigly_df = pd.concat([trigly_df, pd.DataFrame({"SEQN": [6, 7], "LBXTR": [150.0, 150.0]})], ignore_index=True)
        hdl_df = pd.concat([hdl_df, pd.DataFrame({"SEQN": [6, 7], "LBDHDD": [50.0, 50.0]})], ignore_index=True)
        cbc_df = pd.concat([cbc_df, pd.DataFrame({"SEQN": [6, 7], "LBXHGB": [14.0, 14.0], "LBXMCVSI": [90.0, 90.0]})], ignore_index=True)
        demo_df = pd.concat([demo_df, pd.DataFrame({"SEQN": [6, 7], "RIDAGEYR": [50, 50], "RIAGENDR": [1, 1]})], ignore_index=True)

        result = clean_glycemic_data(ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df)

        # Should still have 5 rows (outliers removed)
        assert len(result) == 5
        assert all(result["fpg_mgdl"] >= 40)
        assert all(result["fpg_mgdl"] <= 600)

    def test_clean_glycemic_data_removes_missing_values(self, sample_dataframes):
        """Test that rows with missing values are removed."""
        ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df = sample_dataframes

        # Introduce missing values
        ghb_df.loc[0, "LBXGH"] = np.nan

        result = clean_glycemic_data(ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df)

        # Should have 4 rows (one removed due to NaN)
        assert len(result) == 4

    def test_clean_glycemic_data_inner_merge(self, sample_dataframes):
        """Test that only rows present in all datasets are included."""
        ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df = sample_dataframes

        # Remove one SEQN from ghb_df
        ghb_df = ghb_df[ghb_df["SEQN"] != 3]

        result = clean_glycemic_data(ghb_df, glu_df, trigly_df, hdl_df, cbc_df, demo_df)

        # Should have 4 rows (SEQN 3 missing from ghb_df)
        assert len(result) == 4


class TestGenerateQualityReport:
    """Tests for the generate_quality_report function."""

    @pytest.fixture
    def sample_cleaned_df(self):
        """Create a sample cleaned dataframe for testing."""
        return pd.DataFrame({
            "hba1c_percent": [5.0, 5.5, 5.8, 6.2, 6.5, 7.0, 8.0, 9.0, 10.0, 5.2],
            "fpg_mgdl": [90.0, 95.0, 105.0, 115.0, 130.0, 150.0, 180.0, 200.0, 220.0, 92.0],
            "tg_mgdl": [100.0, 120.0, 150.0, 180.0, 200.0, 220.0, 250.0, 280.0, 300.0, 110.0],
            "hdl_mgdl": [55.0, 52.0, 48.0, 45.0, 42.0, 40.0, 38.0, 35.0, 32.0, 53.0],
            "hgb_gdl": [14.0, 13.5, 14.2, 13.0, 15.0, 14.5, 13.8, 12.5, 11.5, 14.0],
            "mcv_fl": [88.0, 90.0, 92.0, 85.0, 95.0, 91.0, 89.0, 87.0, 86.0, 90.0],
            "age_years": [45, 50, 55, 60, 65, 70, 75, 80, 85, 40],
            "sex": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        })

    def test_generate_quality_report_returns_dict(self, sample_cleaned_df, tmp_path):
        """Test that generate_quality_report returns a dictionary."""
        output_path = tmp_path / "report.txt"
        result = generate_quality_report(sample_cleaned_df, str(output_path))

        assert isinstance(result, dict)

    def test_generate_quality_report_record_count(self, sample_cleaned_df, tmp_path):
        """Test that record count is correct."""
        output_path = tmp_path / "report.txt"
        result = generate_quality_report(sample_cleaned_df, str(output_path))

        assert result["record_count"] == 10

    def test_generate_quality_report_stats(self, sample_cleaned_df, tmp_path):
        """Test that mean/SD statistics are included."""
        output_path = tmp_path / "report.txt"
        result = generate_quality_report(sample_cleaned_df, str(output_path))

        assert "stats" in result
        stats = result["stats"]

        # Check all variables are included
        expected_vars = ["fpg_mgdl", "hba1c_percent", "tg_mgdl", "hdl_mgdl", "hgb_gdl", "mcv_fl"]
        for var in expected_vars:
            assert var in stats
            assert "mean" in stats[var]
            assert "std" in stats[var]

    def test_generate_quality_report_hba1c_distribution(self, sample_cleaned_df, tmp_path):
        """Test HbA1c clinical distribution breakdown."""
        output_path = tmp_path / "report.txt"
        result = generate_quality_report(sample_cleaned_df, str(output_path))

        assert "hba1c_distribution" in result
        hba1c_dist = result["hba1c_distribution"]

        # Verify counts: <5.7% (normal), 5.7-6.4% (prediabetes), >=6.5% (diabetes)
        # Values: 5.0, 5.5, 5.8, 6.2, 6.5, 7.0, 8.0, 9.0, 10.0, 5.2
        # Normal (<5.7): 5.0, 5.5, 5.2 = 3
        # Prediabetes (5.7-6.4): 5.8, 6.2 = 2
        # Diabetes (>=6.5): 6.5, 7.0, 8.0, 9.0, 10.0 = 5
        assert hba1c_dist["normal_lt_5.7"] == 3
        assert hba1c_dist["prediabetes_5.7_to_6.4"] == 2
        assert hba1c_dist["diabetes_gte_6.5"] == 5

    def test_generate_quality_report_fpg_distribution(self, sample_cleaned_df, tmp_path):
        """Test FPG clinical distribution breakdown."""
        output_path = tmp_path / "report.txt"
        result = generate_quality_report(sample_cleaned_df, str(output_path))

        assert "fpg_distribution" in result
        fpg_dist = result["fpg_distribution"]

        # Verify counts: <100 (normal), 100-125 (prediabetes), >=126 (diabetes)
        # Values: 90.0, 95.0, 105.0, 115.0, 130.0, 150.0, 180.0, 200.0, 220.0, 92.0
        # Normal (<100): 90.0, 95.0, 92.0 = 3
        # Prediabetes (100-125): 105.0, 115.0 = 2
        # Diabetes (>=126): 130.0, 150.0, 180.0, 200.0, 220.0 = 5
        assert fpg_dist["normal_lt_100"] == 3
        assert fpg_dist["prediabetes_100_to_125"] == 2
        assert fpg_dist["diabetes_gte_126"] == 5

    def test_generate_quality_report_saves_file(self, sample_cleaned_df, tmp_path):
        """Test that report is saved to specified path."""
        output_path = tmp_path / "subdir" / "report.txt"
        generate_quality_report(sample_cleaned_df, str(output_path))

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        # Check key sections are present
        assert "NHANES GLYCEMIC DATA QUALITY REPORT" in content
        assert "Total Records: 10" in content
        assert "DESCRIPTIVE STATISTICS" in content
        assert "HbA1c DISTRIBUTION" in content
        assert "FPG DISTRIBUTION" in content


class TestLoadExternalKaggleDiabetes:
    """Tests for the load_external_kaggle_diabetes function."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a small CSV file mimicking the Kaggle dataset."""
        csv_path = tmp_path / "diabetes_prediction_dataset.csv"
        df = pd.DataFrame({
            "gender": ["Male", "Female", "Other", "Male", "Female"],
            "age": [45.0, 60.0, 30.0, 55.0, 70.0],
            "hypertension": [0, 1, 0, 0, 1],
            "heart_disease": [0, 0, 0, 1, 0],
            "smoking_history": ["never", "former", "never", "current", "never"],
            "bmi": [25.0, 30.0, 22.0, 28.0, 33.0],
            "HbA1c_level": [5.0, 6.5, 4.8, 7.0, 8.0],
            "blood_glucose_level": [100, 140, 90, 200, 260],
            "diabetes": [0, 1, 0, 1, 1],
        })
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_load_basic(self, sample_csv):
        """Test basic loading returns a DataFrame with expected rows."""
        df = load_external_kaggle_diabetes(str(sample_csv))
        assert isinstance(df, pd.DataFrame)
        # "Other" gender rows are dropped → 4 rows remain
        assert len(df) == 4

    def test_load_column_names(self, sample_csv):
        """Test that columns are renamed to NHANES-compatible names."""
        df = load_external_kaggle_diabetes(str(sample_csv))
        assert "hba1c_percent" in df.columns
        assert "fpg_mgdl" in df.columns
        assert "age_years" in df.columns
        assert "sex" in df.columns
        # Original names should not remain
        assert "HbA1c_level" not in df.columns
        assert "blood_glucose_level" not in df.columns
        assert "gender" not in df.columns

    def test_load_sex_mapping(self, sample_csv):
        """Test that gender strings are mapped to NHANES numeric codes."""
        df = load_external_kaggle_diabetes(str(sample_csv))
        assert set(df["sex"].unique()) <= {1, 2}

    def test_load_outlier_filtering(self, tmp_path):
        """Test that physiologic outliers are removed."""
        csv_path = tmp_path / "outliers.csv"
        df = pd.DataFrame({
            "gender": ["Male", "Male", "Male"],
            "age": [50.0, 50.0, 50.0],
            "hypertension": [0, 0, 0],
            "heart_disease": [0, 0, 0],
            "smoking_history": ["never", "never", "never"],
            "bmi": [25.0, 25.0, 25.0],
            "HbA1c_level": [5.5, 2.0, 22.0],   # row 2 & 3 are outliers
            "blood_glucose_level": [100, 100, 100],
            "diabetes": [0, 0, 0],
        })
        df.to_csv(csv_path, index=False)

        result = load_external_kaggle_diabetes(str(csv_path))
        assert len(result) == 1
        assert result.iloc[0]["hba1c_percent"] == 5.5

    def test_load_missing_columns(self, tmp_path):
        """Test that ValueError is raised when required columns are absent."""
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"col_a": [1, 2]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_external_kaggle_diabetes(str(csv_path))

    def test_load_file_not_found(self, tmp_path):
        """Test FileNotFoundError when file missing and download disabled."""
        bad_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_external_kaggle_diabetes(
                str(bad_path), download_url="http://invalid.invalid/x.csv"
            )
