"""
Data acquisition and processing functions.

This module contains functions for:
- Downloading NHANES glycemic data
- Parsing XPT files
- Data cleaning and harmonization
- Quality report generation
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd


# NHANES data file mappings by cycle
# Format: cycle_code -> {data_type: filename}
NHANES_FILE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "2011-2012": {
        "GHB": "GHB_G.XPT",      # Glycohemoglobin (HbA1c)
        "GLU": "GLU_G.XPT",      # Fasting Plasma Glucose
        "TRIGLY": "TRIGLY_G.XPT", # Triglycerides
        "HDL": "HDL_G.XPT",      # HDL Cholesterol
        "CBC": "CBC_G.XPT",      # Complete Blood Count
        "DEMO": "DEMO_G.XPT",    # Demographics
    },
    "2013-2014": {
        "GHB": "GHB_H.XPT",
        "GLU": "GLU_H.XPT",
        "TRIGLY": "TRIGLY_H.XPT",
        "HDL": "HDL_H.XPT",
        "CBC": "CBC_H.XPT",
        "DEMO": "DEMO_H.XPT",
    },
    "2015-2016": {
        "GHB": "GHB_I.XPT",
        "GLU": "GLU_I.XPT",
        "TRIGLY": "TRIGLY_I.XPT",
        "HDL": "HDL_I.XPT",
        "CBC": "CBC_I.XPT",
        "DEMO": "DEMO_I.XPT",
    },
    "2017-2018": {
        "GHB": "GHB_J.XPT",
        "GLU": "GLU_J.XPT",
        "TRIGLY": "TRIGLY_J.XPT",
        "HDL": "HDL_J.XPT",
        "CBC": "CBC_J.XPT",
        "DEMO": "DEMO_J.XPT",
    },
}

# NHANES base URL
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"


def download_nhanes_glycemic(
    output_dir: str,
    cycles: Optional[List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Download NHANES glycemic data files.

    Downloads GHB (HbA1c), GLU (fasting glucose), TRIGLY (triglycerides),
    HDL, CBC (complete blood count), and DEMO (demographics) XPT files
    for the specified NHANES cycles.

    Args:
        output_dir: Directory to save downloaded files. Will be created if
            it doesn't exist. Files are saved to output_dir/raw/.
        cycles: List of NHANES cycles to download. Valid options:
            "2011-2012", "2013-2014", "2015-2016", "2017-2018".
            Defaults to all available cycles if None.

    Returns:
        Dict mapping cycle -> data_type -> filepath of downloaded files.

    Raises:
        ValueError: If an invalid cycle is specified.

    Example:
        >>> download_nhanes_glycemic("data", cycles=["2017-2018"])
        {'2017-2018': {'GHB': 'data/raw/GHB_J.XPT', ...}}
    """
    # Default to all available cycles
    if cycles is None:
        cycles = list(NHANES_FILE_MAPPINGS.keys())

    # Validate cycles
    invalid_cycles = [c for c in cycles if c not in NHANES_FILE_MAPPINGS]
    if invalid_cycles:
        raise ValueError(
            f"Invalid cycles: {invalid_cycles}. "
            f"Valid options: {list(NHANES_FILE_MAPPINGS.keys())}"
        )

    # Create output directory structure
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: Dict[str, Dict[str, str]] = {}

    for cycle in cycles:
        downloaded_files[cycle] = {}
        file_mappings = NHANES_FILE_MAPPINGS[cycle]

        for data_type, filename in file_mappings.items():
            output_path = raw_dir / filename
            url = f"{NHANES_BASE_URL}/{cycle}/{filename}"

            # Skip if file already exists
            if output_path.exists():
                print(f"[SKIP] {filename} already exists")
                downloaded_files[cycle][data_type] = str(output_path)
                continue

            # Download file
            try:
                print(f"[DOWNLOADING] {filename} from {url}")
                urllib.request.urlretrieve(url, output_path)
                print(f"[SUCCESS] Downloaded {filename}")
                downloaded_files[cycle][data_type] = str(output_path)
            except urllib.error.URLError as e:
                print(f"[ERROR] Failed to download {filename}: {e.reason}")
                downloaded_files[cycle][data_type] = ""
            except urllib.error.HTTPError as e:
                print(f"[ERROR] HTTP error downloading {filename}: {e.code} {e.reason}")
                downloaded_files[cycle][data_type] = ""
            except OSError as e:
                print(f"[ERROR] OS error saving {filename}: {e}")
                downloaded_files[cycle][data_type] = ""

    return downloaded_files


def read_xpt(filepath: str) -> pd.DataFrame:
    """
    Read a NHANES XPT (SAS transport) file into a pandas DataFrame.

    Args:
        filepath: Path to the XPT file to read.

    Returns:
        DataFrame containing the data from the XPT file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be parsed as a valid XPT file.

    Example:
        >>> df = read_xpt("data/raw/GHB_J.XPT")
        >>> df.head()
           SEQN    LBXGH
        0  93703     5.4
        ...
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"XPT file not found: {filepath}. "
            f"Please run download_nhanes_glycemic() first to download the data."
        )

    try:
        df = pd.read_sas(filepath, format="xport")
        return df
    except Exception as e:
        raise ValueError(
            f"Failed to parse XPT file '{filepath}': {e}. "
            f"The file may be corrupted or not in valid SAS transport format."
        ) from e
