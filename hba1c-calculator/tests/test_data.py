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

from hba1cE.data import read_xpt


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
