"""
Unit tests for freeT/data.py module.

Tests XPT file parsing and data pipeline functions.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from freeT.data import read_xpt


class TestReadXpt:
    """Tests for the read_xpt function."""
    
    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            read_xpt("nonexistent_file.XPT")
        
        assert "XPT file not found" in str(exc_info.value)
        assert "download_nhanes()" in str(exc_info.value)
    
    def test_wrong_extension_error(self):
        """Test that ValueError is raised for wrong file extension."""
        # Create a temporary file with wrong extension
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                read_xpt(temp_path)
            
            assert "Expected XPT file" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_accepts_path_object(self):
        """Test that function accepts Path objects."""
        with pytest.raises(FileNotFoundError):
            read_xpt(Path("nonexistent.XPT"))
    
    def test_accepts_string_path(self):
        """Test that function accepts string paths."""
        with pytest.raises(FileNotFoundError):
            read_xpt("nonexistent.XPT")
    
    def test_invalid_xpt_content_error(self):
        """Test that ValueError is raised for invalid XPT content."""
        # Create a temporary file with XPT extension but invalid content
        with tempfile.NamedTemporaryFile(suffix=".XPT", delete=False) as f:
            f.write(b"This is not valid XPT data")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                read_xpt(temp_path)
            
            assert "Failed to parse XPT file" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_lowercase_extension_accepted(self):
        """Test that lowercase .xpt extension is accepted."""
        # Create a temp file with lowercase extension
        with tempfile.NamedTemporaryFile(suffix=".xpt", delete=False) as f:
            f.write(b"invalid")
            temp_path = f.name
        
        try:
            # Should raise ValueError for invalid content, not for extension
            with pytest.raises(ValueError) as exc_info:
                read_xpt(temp_path)
            
            # Verify it failed on content parsing, not extension
            assert "Failed to parse XPT file" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_successful_read_with_mock(self):
        """Test successful XPT reading using mock."""
        expected_df = pd.DataFrame({
            'SEQN': [1001, 1002, 1003],
            'LBXTST': [450.5, 380.2, 520.1]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".XPT", delete=False) as f:
            temp_path = f.name
        
        try:
            with mock.patch('pandas.read_sas', return_value=expected_df):
                result = read_xpt(temp_path)
            
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ['SEQN', 'LBXTST']
            assert len(result) == 3
        finally:
            os.unlink(temp_path)
    
    def test_returns_dataframe(self):
        """Test that function returns a pandas DataFrame when mocked."""
        mock_df = pd.DataFrame({'A': [1, 2, 3]})
        
        with tempfile.NamedTemporaryFile(suffix=".XPT", delete=False) as f:
            temp_path = f.name
        
        try:
            with mock.patch('pandas.read_sas', return_value=mock_df):
                result = read_xpt(temp_path)
            
            pd.testing.assert_frame_equal(result, mock_df)
        finally:
            os.unlink(temp_path)


class TestCleanNhanesData:
    """Tests for clean_nhanes_data function."""
    
    def test_basic_merge(self):
        """Test that data is merged on SEQN."""
        from freeT.data import clean_nhanes_data
        
        tst_df = pd.DataFrame({
            'SEQN': [1, 2, 3],
            'LBXTST': [450, 380, 520],  # ng/dL
        })
        shbg_df = pd.DataFrame({
            'SEQN': [1, 2, 3],
            'LBXSHBG': [40, 30, 50],  # nmol/L
        })
        alb_df = pd.DataFrame({
            'SEQN': [1, 2, 3],
            'LBXSAL': [4.5, 4.2, 4.3],  # g/dL
        })
        
        result = clean_nhanes_data(tst_df, shbg_df, alb_df, verbose=False)
        
        assert 'seqn' in result.columns
        assert 'tt_nmoll' in result.columns
        assert 'shbg_nmoll' in result.columns
        assert 'alb_gl' in result.columns
    
    def test_unit_conversion_tt(self):
        """Test that TT is converted from ng/dL to nmol/L."""
        from freeT.data import clean_nhanes_data
        
        tst_df = pd.DataFrame({'SEQN': [1], 'LBXTST': [288.4]})  # ≈ 10 nmol/L
        shbg_df = pd.DataFrame({'SEQN': [1], 'LBXSHBG': [40]})
        alb_df = pd.DataFrame({'SEQN': [1], 'LBXSAL': [4.5]})
        
        result = clean_nhanes_data(tst_df, shbg_df, alb_df, verbose=False)
        
        # 288.4 ng/dL * 0.0347 ≈ 10 nmol/L
        assert 9 < result['tt_nmoll'].iloc[0] < 11
    
    def test_unit_conversion_albumin(self):
        """Test that albumin is converted from g/dL to g/L."""
        from freeT.data import clean_nhanes_data
        
        tst_df = pd.DataFrame({'SEQN': [1], 'LBXTST': [450]})
        shbg_df = pd.DataFrame({'SEQN': [1], 'LBXSHBG': [40]})
        alb_df = pd.DataFrame({'SEQN': [1], 'LBXSAL': [4.5]})  # g/dL → 45 g/L
        
        result = clean_nhanes_data(tst_df, shbg_df, alb_df, verbose=False)
        
        assert result['alb_gl'].iloc[0] == 45.0
    
    def test_outlier_removal_low_tt(self):
        """Test that low TT values are removed."""
        from freeT.data import clean_nhanes_data
        
        # Create data with low TT (should be removed after conversion)
        tst_df = pd.DataFrame({
            'SEQN': [1, 2],
            'LBXTST': [10, 450],  # 10 ng/dL → 0.35 nmol/L (< 0.5, removed)
        })
        shbg_df = pd.DataFrame({'SEQN': [1, 2], 'LBXSHBG': [40, 40]})
        alb_df = pd.DataFrame({'SEQN': [1, 2], 'LBXSAL': [4.5, 4.5]})
        
        result = clean_nhanes_data(tst_df, shbg_df, alb_df, verbose=False)
        
        assert len(result) == 1
        assert result['seqn'].iloc[0] == 2
    
    def test_outlier_removal_high_shbg(self):
        """Test that high SHBG values are removed."""
        from freeT.data import clean_nhanes_data
        
        tst_df = pd.DataFrame({'SEQN': [1, 2], 'LBXTST': [450, 400]})
        shbg_df = pd.DataFrame({
            'SEQN': [1, 2],
            'LBXSHBG': [40, 300],  # 300 > 250, removed
        })
        alb_df = pd.DataFrame({'SEQN': [1, 2], 'LBXSAL': [4.5, 4.2]})
        
        result = clean_nhanes_data(tst_df, shbg_df, alb_df, verbose=False)
        
        assert len(result) == 1
        assert result['seqn'].iloc[0] == 1
    
    def test_outlier_removal_low_albumin(self):
        """Test that low albumin values are removed."""
        from freeT.data import clean_nhanes_data
        
        tst_df = pd.DataFrame({'SEQN': [1, 2], 'LBXTST': [450, 400]})
        shbg_df = pd.DataFrame({'SEQN': [1, 2], 'LBXSHBG': [40, 40]})
        alb_df = pd.DataFrame({
            'SEQN': [1, 2],
            'LBXSAL': [4.5, 2.5],  # 2.5 g/dL → 25 g/L (< 30, removed)
        })
        
        result = clean_nhanes_data(tst_df, shbg_df, alb_df, verbose=False)
        
        assert len(result) == 1
        assert result['seqn'].iloc[0] == 1
    
    def test_inner_join_only_common_seqn(self):
        """Test that only records with common SEQN are kept."""
        from freeT.data import clean_nhanes_data
        
        tst_df = pd.DataFrame({'SEQN': [1, 2], 'LBXTST': [450, 400]})
        shbg_df = pd.DataFrame({'SEQN': [1, 3], 'LBXSHBG': [40, 50]})  # No SEQN 2
        alb_df = pd.DataFrame({'SEQN': [1, 2], 'LBXSAL': [4.5, 4.2]})
        
        result = clean_nhanes_data(tst_df, shbg_df, alb_df, verbose=False)
        
        # Only SEQN 1 is in all three datasets
        assert len(result) == 1
        assert result['seqn'].iloc[0] == 1


class TestGenerateQualityReport:
    """Tests for generate_quality_report function."""
    
    def test_returns_dict(self):
        """Test that generate_quality_report returns a dict."""
        from freeT.data import generate_quality_report
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0, 20.0, 10.0],
            'shbg_nmoll': [40.0, 30.0, 50.0],
            'alb_gl': [45.0, 42.0, 43.0],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.txt')
            result = generate_quality_report(df, output_path)
            
            assert isinstance(result, dict)
    
    def test_dict_has_record_count(self):
        """Test that result dict has record count."""
        from freeT.data import generate_quality_report
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0, 20.0, 10.0],
            'shbg_nmoll': [40.0, 30.0, 50.0],
            'alb_gl': [45.0, 42.0, 43.0],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.txt')
            result = generate_quality_report(df, output_path)
            
            assert 'record_count' in result
            assert result['record_count'] == 3
    
    def test_creates_output_file(self):
        """Test that output file is created."""
        from freeT.data import generate_quality_report
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0, 20.0, 10.0],
            'shbg_nmoll': [40.0, 30.0, 50.0],
            'alb_gl': [45.0, 42.0, 43.0],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.txt')
            generate_quality_report(df, output_path)
            
            assert os.path.exists(output_path)
    
    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        from freeT.data import generate_quality_report
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0],
            'shbg_nmoll': [40.0],
            'alb_gl': [45.0],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'nested', 'dir', 'report.txt')
            generate_quality_report(df, output_path)
            
            assert os.path.exists(output_path)
    
    def test_statistics_in_result(self):
        """Test that statistics are included in result."""
        from freeT.data import generate_quality_report
        
        df = pd.DataFrame({
            'tt_nmoll': [15.0, 20.0, 10.0],
            'shbg_nmoll': [40.0, 30.0, 50.0],
            'alb_gl': [45.0, 42.0, 43.0],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'report.txt')
            result = generate_quality_report(df, output_path)
            
            # Check statistics keys exist
            assert 'statistics' in result
            assert 'tt_nmoll' in result['statistics']
