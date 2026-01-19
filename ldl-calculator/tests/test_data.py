"""
Tests for ldlC/data.py module.
"""

import os
import tempfile
import pytest
import pandas as pd

from ldlC.data import read_xpt


class TestReadXpt:
    """Tests for the read_xpt function."""

    def test_read_xpt_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            read_xpt("non_existent_file.XPT")
        
        assert "XPT file not found" in str(exc_info.value)
        assert "non_existent_file.XPT" in str(exc_info.value)

    def test_read_xpt_informative_error_message(self):
        """Test that error message includes helpful guidance."""
        with pytest.raises(FileNotFoundError) as exc_info:
            read_xpt("missing/path/file.XPT")
        
        error_message = str(exc_info.value)
        assert "download_nhanes_lipids()" in error_message

    def test_read_xpt_invalid_file_format(self):
        """Test that ValueError is raised for invalid XPT file."""
        # Create a temporary file that is not a valid XPT
        with tempfile.NamedTemporaryFile(mode='w', suffix='.XPT', delete=False) as f:
            f.write("This is not a valid XPT file content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                read_xpt(temp_path)
            
            error_message = str(exc_info.value)
            assert "Unable to parse XPT file" in error_message
            assert "corrupted or not in valid SAS transport format" in error_message
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    def test_read_xpt_with_mock_data(self):
        """Test that read_xpt returns a DataFrame with mock XPT data.
        
        Uses pyreadstat to create a valid XPT file for testing.
        Falls back to skip if pyreadstat not available.
        """
        try:
            import pyreadstat
        except ImportError:
            pytest.skip("pyreadstat not available for creating mock XPT files")
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'SEQN': [1, 2, 3],
            'VALUE': [100.0, 150.0, 200.0]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, 'test.xpt')
            
            # Write using pyreadstat
            pyreadstat.write_xport(test_df, temp_path)
            
            # Read it back using our function
            result = read_xpt(temp_path)
            
            # Verify result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
