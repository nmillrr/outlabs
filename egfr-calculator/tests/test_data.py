"""
Tests for eGFR/data.py — read_xpt function.

Uses unittest.mock to mock pd.read_sas for happy-path tests (avoids the
need to construct a spec-compliant SAS XPORT V5 binary), and uses real
files for error-path tests.
"""

import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from eGFR.data import read_xpt


# ---------------------------------------------------------------------------
# Tests
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
