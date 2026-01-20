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

import pandas as pd

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


def read_xpt(filepath: str) -> pd.DataFrame:
    """
    Read a SAS transport format (XPT) file into a pandas DataFrame.

    NHANES distributes data in SAS XPORT format (.XPT files). This function
    parses these files into pandas DataFrames for analysis.

    Parameters
    ----------
    filepath : str
        Path to the XPT file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the XPT file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file cannot be parsed as a valid XPT file.

    Examples
    --------
    >>> df = read_xpt("data/raw/TRIGLY_I.XPT")
    >>> df.head()
       SEQN  LBXSTR
    0  83732   145.0
    1  83733    89.0
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"XPT file not found: '{filepath}'. "
            "Please ensure the file path is correct or download the file first "
            "using download_nhanes_lipids()."
        )

    try:
        # pandas can read SAS transport format directly
        df = pd.read_sas(filepath, format='xport')
        return df
    except Exception as e:
        raise ValueError(
            f"Unable to parse XPT file '{filepath}': {str(e)}. "
            "The file may be corrupted or not in valid SAS transport format."
        )


def clean_lipid_data(
    tc_df: pd.DataFrame,
    hdl_df: pd.DataFrame,
    tg_df: pd.DataFrame,
    ldl_direct_df: pd.DataFrame,
    tc_column: str = "LBXTC",
    hdl_column: str = "LBDHDD",
    tg_column: str = "LBXSTR",
    ldl_column: str = "LBDLDL",
) -> pd.DataFrame:
    """
    Clean and merge NHANES lipid panel DataFrames.

    Merges individual lipid measurement DataFrames on SEQN (sample ID),
    renames columns to standardized names, removes physiologic outliers,
    and calculates derived values.

    Parameters
    ----------
    tc_df : pd.DataFrame
        DataFrame containing total cholesterol data with SEQN and TC measurement.
    hdl_df : pd.DataFrame
        DataFrame containing HDL cholesterol data with SEQN and HDL measurement.
    tg_df : pd.DataFrame
        DataFrame containing triglyceride data with SEQN and TG measurement.
    ldl_direct_df : pd.DataFrame
        DataFrame containing direct LDL measurement data with SEQN and LDL.
    tc_column : str, optional
        Column name for total cholesterol in tc_df. Default "LBXTC".
    hdl_column : str, optional
        Column name for HDL cholesterol in hdl_df. Default "LBDHDD".
    tg_column : str, optional
        Column name for triglycerides in tg_df. Default "LBXSTR".
    ldl_column : str, optional
        Column name for direct LDL in ldl_direct_df. Default "LBDLDL".

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        - SEQN: Sample identifier
        - tc_mgdl: Total cholesterol (mg/dL)
        - hdl_mgdl: HDL cholesterol (mg/dL)
        - tg_mgdl: Triglycerides (mg/dL)
        - ldl_direct_mgdl: Direct LDL measurement (mg/dL)
        - non_hdl_mgdl: Non-HDL cholesterol (TC - HDL, mg/dL)

    Notes
    -----
    Outlier removal criteria (physiologically implausible values):
    - TC < 50 mg/dL (removed)
    - TG > 2000 mg/dL (removed)
    - HDL < 10 mg/dL (removed)

    Examples
    --------
    >>> tc_df = read_xpt("data/raw/TCHOL_I.XPT")
    >>> hdl_df = read_xpt("data/raw/HDL_I.XPT")
    >>> tg_df = read_xpt("data/raw/TRIGLY_I.XPT")
    >>> ldl_df = read_xpt("data/raw/BIOPRO_I.XPT")
    >>> cleaned = clean_lipid_data(tc_df, hdl_df, tg_df, ldl_df)
    >>> cleaned.columns.tolist()
    ['SEQN', 'tc_mgdl', 'hdl_mgdl', 'tg_mgdl', 'ldl_direct_mgdl', 'non_hdl_mgdl']
    """
    # Extract relevant columns from each DataFrame
    tc_subset = tc_df[["SEQN", tc_column]].copy()
    tc_subset = tc_subset.rename(columns={tc_column: "tc_mgdl"})

    hdl_subset = hdl_df[["SEQN", hdl_column]].copy()
    hdl_subset = hdl_subset.rename(columns={hdl_column: "hdl_mgdl"})

    tg_subset = tg_df[["SEQN", tg_column]].copy()
    tg_subset = tg_subset.rename(columns={tg_column: "tg_mgdl"})

    ldl_subset = ldl_direct_df[["SEQN", ldl_column]].copy()
    ldl_subset = ldl_subset.rename(columns={ldl_column: "ldl_direct_mgdl"})

    # Merge all datasets on SEQN (inner join to keep only complete records)
    merged = tc_subset.merge(hdl_subset, on="SEQN", how="inner")
    merged = merged.merge(tg_subset, on="SEQN", how="inner")
    merged = merged.merge(ldl_subset, on="SEQN", how="inner")

    # Remove rows with any missing values in lipid columns
    lipid_columns = ["tc_mgdl", "hdl_mgdl", "tg_mgdl", "ldl_direct_mgdl"]
    merged = merged.dropna(subset=lipid_columns)

    # Remove physiologic outliers
    # TC < 50 mg/dL is physiologically implausible
    merged = merged[merged["tc_mgdl"] >= 50]
    # TG > 2000 mg/dL is extreme hypertriglyceridemia (requires direct LDL)
    merged = merged[merged["tg_mgdl"] <= 2000]
    # HDL < 10 mg/dL is physiologically implausible
    merged = merged[merged["hdl_mgdl"] >= 10]

    # Calculate non-HDL cholesterol (TC - HDL)
    merged["non_hdl_mgdl"] = merged["tc_mgdl"] - merged["hdl_mgdl"]

    # Reset index for clean output
    merged = merged.reset_index(drop=True)

    return merged
