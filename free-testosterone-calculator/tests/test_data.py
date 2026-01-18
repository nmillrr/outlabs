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
