"""
Data pipeline for NHANES testosterone data acquisition and processing.
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def read_xpt(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Read a SAS transport (XPT) file and return a pandas DataFrame.
    
    This function parses NHANES XPT files, which use the SAS transport format
    (SAS XPORT version 5/8). It's commonly used for CDC data distribution.
    
    Args:
        filepath: Path to the XPT file. Can be a string or Path object.
    
    Returns:
        pd.DataFrame: Contents of the XPT file as a DataFrame.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be parsed as XPT format.
    
    Example:
        >>> df = read_xpt("data/raw/2015_2016/TST_I.XPT")
        >>> print(df.columns.tolist())
        ['SEQN', 'LBXTST', ...]
    """
    filepath = Path(filepath)
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"XPT file not found: {filepath}. "
            f"Please ensure the file exists or download it using download_nhanes()."
        )
    
    # Check if file has correct extension (case-insensitive)
    if filepath.suffix.upper() != ".XPT":
        raise ValueError(
            f"Expected XPT file, got: {filepath.suffix}. "
            f"File must have .xpt or .XPT extension."
        )
    
    try:
        # Read the XPT file using pandas
        df = pd.read_sas(filepath, format='xport')
        return df
    except Exception as e:
        raise ValueError(
            f"Failed to parse XPT file '{filepath}': {str(e)}. "
            f"The file may be corrupted or not in valid XPT format."
        )


# NHANES data file URL templates
# CDC NHANES data is organized by cycle (e.g., 2011-2012, 2013-2014, 2015-2016)
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"

# File naming conventions for each cycle
# TST = Testosterone, SHBG = Sex Hormone Binding Globulin, ALB = Albumin
NHANES_FILES = {
    "2011-2012": {
        "TST": "TST_G.XPT",
        "SHBG": "SHBG_G.XPT", 
        "ALB": "BIOPRO_G.XPT",  # Albumin is in biochemistry profile
    },
    "2013-2014": {
        "TST": "TST_H.XPT",
        "SHBG": "SHBG_H.XPT",
        "ALB": "BIOPRO_H.XPT",
    },
    "2015-2016": {
        "TST": "TST_I.XPT",
        "SHBG": "SHBG_I.XPT",
        "ALB": "BIOPRO_I.XPT",
    },
}

# Default cycles to download
DEFAULT_CYCLES = ["2011-2012", "2013-2014", "2015-2016"]


def download_nhanes(
    output_dir: str = "data/raw",
    cycles: Optional[List[str]] = None,
    verbose: bool = True
) -> dict:
    """
    Download NHANES testosterone, SHBG, and albumin data files.
    
    Downloads XPT (SAS transport) files from CDC NHANES website for the
    specified survey cycles. Files are organized by cycle in the output
    directory.
    
    Args:
        output_dir: Directory to save downloaded files. Created if not exists.
                   Default is "data/raw".
        cycles: List of NHANES cycles to download (e.g., ["2011-2012", "2013-2014"]).
               Default downloads 2011-2016 cycles.
        verbose: If True, print progress messages.
    
    Returns:
        dict: Summary of downloads with structure:
              {
                  "downloaded": [(cycle, file_type, path), ...],
                  "failed": [(cycle, file_type, error_message), ...],
                  "skipped": [(cycle, file_type, path), ...]  # already existed
              }
    
    Raises:
        ValueError: If an invalid cycle is specified.
    
    Example:
        >>> result = download_nhanes("data/raw", cycles=["2015-2016"])
        >>> print(f"Downloaded {len(result['downloaded'])} files")
    """
    if cycles is None:
        cycles = DEFAULT_CYCLES.copy()
    
    # Validate cycles
    invalid_cycles = [c for c in cycles if c not in NHANES_FILES]
    if invalid_cycles:
        raise ValueError(
            f"Invalid cycle(s): {invalid_cycles}. "
            f"Valid cycles are: {list(NHANES_FILES.keys())}"
        )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result = {
        "downloaded": [],
        "failed": [],
        "skipped": [],
    }
    
    for cycle in cycles:
        cycle_dir = output_path / cycle.replace("-", "_")
        cycle_dir.mkdir(exist_ok=True)
        
        files = NHANES_FILES[cycle]
        
        for file_type, filename in files.items():
            file_path = cycle_dir / filename
            
            # Skip if file already exists
            if file_path.exists():
                if verbose:
                    print(f"[SKIP] {cycle}/{filename} already exists")
                result["skipped"].append((cycle, file_type, str(file_path)))
                continue
            
            # Build download URL
            url = f"{NHANES_BASE_URL}/{cycle}/{filename}"
            
            if verbose:
                print(f"[DOWNLOAD] {cycle}/{filename}...")
            
            try:
                # Download file with timeout
                urllib.request.urlretrieve(url, file_path)
                result["downloaded"].append((cycle, file_type, str(file_path)))
                if verbose:
                    print(f"  -> Saved to {file_path}")
                    
            except urllib.error.HTTPError as e:
                error_msg = f"HTTP Error {e.code}: {e.reason}"
                result["failed"].append((cycle, file_type, error_msg))
                if verbose:
                    print(f"  -> FAILED: {error_msg}")
                    
            except urllib.error.URLError as e:
                error_msg = f"URL Error: {e.reason}"
                result["failed"].append((cycle, file_type, error_msg))
                if verbose:
                    print(f"  -> FAILED: {error_msg}")
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                result["failed"].append((cycle, file_type, error_msg))
                if verbose:
                    print(f"  -> FAILED: {error_msg}")
    
    # Print summary
    if verbose:
        print(f"\n--- Download Summary ---")
        print(f"Downloaded: {len(result['downloaded'])} files")
        print(f"Skipped (existing): {len(result['skipped'])} files")
        print(f"Failed: {len(result['failed'])} files")
    
    return result
