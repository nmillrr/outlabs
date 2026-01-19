"""
Data sourcing and preprocessing functions.

This module contains:
- NHANES lipid data download functions
- XPT file parsing
- Data cleaning and harmonization pipelines
- Quality report generation
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Dict

# NHANES cycle suffixes for file naming
# Each cycle uses a letter suffix in file names
CYCLE_SUFFIXES: Dict[str, str] = {
    "2005-2006": "_D",
    "2007-2008": "_E",
    "2009-2010": "_F",
    "2011-2012": "_G",
    "2013-2014": "_H",
    "2015-2016": "_I",
    "2017-2018": "_J",
}

# File prefixes for different lipid measurements
# TRIGLY = Triglycerides, HDL = HDL cholesterol, TCHOL = Total cholesterol
# LDL data uses different file names depending on cycle
LIPID_FILE_PREFIXES: Dict[str, str] = {
    "triglycerides": "TRIGLY",
    "hdl": "HDL",
    "total_cholesterol": "TCHOL",
}

# Base URL for NHANES laboratory data
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"


def download_nhanes_lipids(
    output_dir: str,
    cycles: Optional[List[str]] = None,
    include_direct_ldl: bool = True,
) -> Dict[str, List[str]]:
    """
    Download NHANES lipid panel XPT files for specified cycles.

    Downloads TRIGLY, HDL, TCHOL, and optionally direct LDL measurement files
    from the CDC NHANES website for the specified survey cycles.

    Parameters
    ----------
    output_dir : str
        Base output directory. Files will be saved to output_dir/raw/
    cycles : list of str, optional
        List of NHANES cycles to download (e.g., ["2005-2006", "2007-2008"]).
        Defaults to all cycles from 2005-2018.
    include_direct_ldl : bool, optional
        Whether to download direct LDL measurement files. Default True.

    Returns
    -------
    dict
        Dictionary mapping cycle names to lists of successfully downloaded file paths.

    Raises
    ------
    ValueError
        If an invalid cycle is specified.

    Examples
    --------
    >>> download_nhanes_lipids("./data", cycles=["2015-2016"])
    {'2015-2016': ['./data/raw/TRIGLY_I.XPT', './data/raw/HDL_I.XPT', ...]}
    """
    # Default to all available cycles
    if cycles is None:
        cycles = list(CYCLE_SUFFIXES.keys())

    # Validate cycles
    for cycle in cycles:
        if cycle not in CYCLE_SUFFIXES:
            raise ValueError(
                f"Invalid cycle '{cycle}'. Available cycles: {list(CYCLE_SUFFIXES.keys())}"
            )

    # Create output directory
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: Dict[str, List[str]] = {}

    for cycle in cycles:
        suffix = CYCLE_SUFFIXES[cycle]
        cycle_files: List[str] = []

        # Extract years for URL construction
        start_year = cycle.split("-")[0]

        # Download standard lipid files (TRIGLY, HDL, TCHOL)
        for lipid_type, prefix in LIPID_FILE_PREFIXES.items():
            filename = f"{prefix}{suffix}.XPT"
            url = f"{NHANES_BASE_URL}/{cycle}/{filename}"
            output_path = raw_dir / filename

            success = _download_file(url, str(output_path), lipid_type, cycle)
            if success:
                cycle_files.append(str(output_path))

        # Download direct LDL file if requested
        # Note: LDL direct measurement files have different naming conventions
        if include_direct_ldl:
            ldl_filename = f"BIOPRO{suffix}.XPT"  # Biochemistry profile contains LDL
            ldl_url = f"{NHANES_BASE_URL}/{cycle}/{ldl_filename}"
            ldl_output_path = raw_dir / ldl_filename

            success = _download_file(ldl_url, str(ldl_output_path), "ldl_direct", cycle)
            if success:
                cycle_files.append(str(ldl_output_path))

        downloaded_files[cycle] = cycle_files

    return downloaded_files


def _download_file(url: str, output_path: str, file_type: str, cycle: str) -> bool:
    """
    Download a single file from URL to output path.

    Parameters
    ----------
    url : str
        URL to download from.
    output_path : str
        Local path to save the file.
    file_type : str
        Type of lipid file (for logging purposes).
    cycle : str
        NHANES cycle (for logging purposes).

    Returns
    -------
    bool
        True if download succeeded, False otherwise.
    """
    try:
        print(f"Downloading {file_type} for {cycle} from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  Successfully saved to {output_path}")
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code} downloading {file_type} for {cycle}: {e.reason}")
        print(f"  URL: {url}")
        return False
    except urllib.error.URLError as e:
        print(f"  URL Error downloading {file_type} for {cycle}: {e.reason}")
        print(f"  URL: {url}")
        return False
    except Exception as e:
        print(f"  Unexpected error downloading {file_type} for {cycle}: {str(e)}")
        return False
