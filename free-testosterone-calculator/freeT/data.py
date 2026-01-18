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


def clean_nhanes_data(
    tst_df: pd.DataFrame,
    shbg_df: pd.DataFrame,
    alb_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Clean and merge NHANES testosterone, SHBG, and albumin data.
    
    This function merges the three datasets on SEQN (participant ID),
    applies unit conversions to standardize measurements, and removes
    physiologically implausible outliers.
    
    Args:
        tst_df: DataFrame from TST XPT file containing testosterone data.
                Expected column: LBXTST (TT in ng/dL)
        shbg_df: DataFrame from SHBG XPT file containing SHBG data.
                 Expected column: LBXSHBG (SHBG in nmol/L)
        alb_df: DataFrame from BIOPRO XPT file containing albumin data.
                Expected column: LBXSAL (Albumin in g/dL)
        verbose: If True, print cleaning statistics.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized columns:
            - seqn: Participant ID
            - tt_nmoll: Total testosterone in nmol/L
            - shbg_nmoll: SHBG in nmol/L
            - alb_gl: Albumin in g/L
    
    Notes:
        Unit conversions applied:
        - TT: ng/dL → nmol/L (multiply by 0.0347)
        - Albumin: g/dL → g/L (multiply by 10)
        - SHBG: Already in nmol/L (no conversion needed)
        
        Outlier removal criteria (physiologically implausible):
        - TT < 0.5 nmol/L (after conversion)
        - SHBG > 250 nmol/L
        - Albumin < 30 g/L (after conversion)
    
    Example:
        >>> tst = read_xpt("data/raw/2015_2016/TST_I.XPT")
        >>> shbg = read_xpt("data/raw/2015_2016/SHBG_I.XPT")
        >>> alb = read_xpt("data/raw/2015_2016/BIOPRO_I.XPT")
        >>> clean_df = clean_nhanes_data(tst, shbg, alb)
    """
    from .utils import ng_dl_to_nmol_l
    
    # Store initial counts for stats
    initial_tst = len(tst_df)
    initial_shbg = len(shbg_df)
    initial_alb = len(alb_df)
    
    # Step 1: Select relevant columns and rename for merge
    # TST data - testosterone in ng/dL
    tst_clean = tst_df[['SEQN', 'LBXTST']].copy()
    tst_clean = tst_clean.rename(columns={'LBXTST': 'tt_ngdl'})
    
    # SHBG data - already in nmol/L
    shbg_clean = shbg_df[['SEQN', 'LBXSHBG']].copy()
    shbg_clean = shbg_clean.rename(columns={'LBXSHBG': 'shbg_nmoll'})
    
    # Albumin data from biochemistry profile - in g/dL
    alb_clean = alb_df[['SEQN', 'LBXSAL']].copy()
    alb_clean = alb_clean.rename(columns={'LBXSAL': 'alb_gdl'})
    
    # Step 2: Merge datasets on SEQN (inner join to keep only complete cases)
    merged = tst_clean.merge(shbg_clean, on='SEQN', how='inner')
    merged = merged.merge(alb_clean, on='SEQN', how='inner')
    
    after_merge = len(merged)
    
    # Step 3: Remove rows with missing values in key columns
    merged = merged.dropna(subset=['tt_ngdl', 'shbg_nmoll', 'alb_gdl'])
    after_dropna = len(merged)
    
    # Step 4: Apply unit conversions
    # TT: ng/dL → nmol/L
    merged['tt_nmoll'] = merged['tt_ngdl'].apply(ng_dl_to_nmol_l)
    
    # Albumin: g/dL → g/L (multiply by 10)
    merged['alb_gl'] = merged['alb_gdl'] * 10
    
    # Step 5: Remove physiologic outliers
    before_outliers = len(merged)
    
    # TT < 0.5 nmol/L is physiologically implausible
    merged = merged[merged['tt_nmoll'] >= 0.5]
    
    # SHBG > 250 nmol/L is very high, likely erroneous
    merged = merged[merged['shbg_nmoll'] <= 250]
    
    # Albumin < 30 g/L indicates severe hypoalbuminemia
    merged = merged[merged['alb_gl'] >= 30]
    
    after_outliers = len(merged)
    
    # Step 6: Select and rename final columns
    result = merged[['SEQN', 'tt_nmoll', 'shbg_nmoll', 'alb_gl']].copy()
    result = result.rename(columns={'SEQN': 'seqn'})
    
    # Reset index
    result = result.reset_index(drop=True)
    
    if verbose:
        print("--- NHANES Data Cleaning Summary ---")
        print(f"Input records: TST={initial_tst}, SHBG={initial_shbg}, ALB={initial_alb}")
        print(f"After merge (inner join): {after_merge}")
        print(f"After dropping NA: {after_dropna}")
        print(f"After outlier removal: {after_outliers}")
        print(f"Final cleaned records: {len(result)}")
        print(f"Removed as outliers: {before_outliers - after_outliers}")
    
    return result


def generate_quality_report(df: pd.DataFrame, output_path: str) -> dict:
    """
    Generate a data quality report for cleaned NHANES data.
    
    Creates a comprehensive quality report including record counts,
    descriptive statistics (mean, SD) for key variables, and missing
    value analysis. The report is saved to a text file and returned
    as a dictionary.
    
    Args:
        df: Cleaned DataFrame with columns: tt_nmoll, shbg_nmoll, alb_gl
            (from clean_nhanes_data output)
        output_path: File path where the report will be saved.
    
    Returns:
        dict: Quality report with structure:
            {
                "record_count": int,
                "statistics": {
                    "tt_nmoll": {"mean": float, "sd": float, "min": float, "max": float},
                    "shbg_nmoll": {"mean": float, "sd": float, "min": float, "max": float},
                    "alb_gl": {"mean": float, "sd": float, "min": float, "max": float}
                },
                "missing_values": {
                    "tt_nmoll": int,
                    "shbg_nmoll": int,
                    "alb_gl": int
                }
            }
    
    Example:
        >>> clean_df = clean_nhanes_data(tst, shbg, alb)
        >>> report = generate_quality_report(clean_df, "reports/quality.txt")
        >>> print(f"Total records: {report['record_count']}")
    """
    # Define the key columns for analysis
    key_columns = ['tt_nmoll', 'shbg_nmoll', 'alb_gl']
    column_labels = {
        'tt_nmoll': 'Total Testosterone (nmol/L)',
        'shbg_nmoll': 'SHBG (nmol/L)',
        'alb_gl': 'Albumin (g/L)'
    }
    
    # Build the report dictionary
    report = {
        "record_count": len(df),
        "statistics": {},
        "missing_values": {}
    }
    
    # Calculate statistics for each key column
    for col in key_columns:
        if col in df.columns:
            report["statistics"][col] = {
                "mean": float(df[col].mean()),
                "sd": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
            report["missing_values"][col] = int(df[col].isna().sum())
        else:
            report["statistics"][col] = {"mean": None, "sd": None, "min": None, "max": None}
            report["missing_values"][col] = len(df)  # All missing if column doesn't exist
    
    # Generate text report
    lines = []
    lines.append("=" * 60)
    lines.append("NHANES Data Quality Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total Records: {report['record_count']}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("Descriptive Statistics")
    lines.append("-" * 60)
    
    for col in key_columns:
        label = column_labels.get(col, col)
        stats = report["statistics"][col]
        lines.append(f"\n{label}:")
        if stats["mean"] is not None:
            lines.append(f"  Mean: {stats['mean']:.2f}")
            lines.append(f"  SD:   {stats['sd']:.2f}")
            lines.append(f"  Min:  {stats['min']:.2f}")
            lines.append(f"  Max:  {stats['max']:.2f}")
        else:
            lines.append("  (Column not found in data)")
    
    lines.append("")
    lines.append("-" * 60)
    lines.append("Missing Values")
    lines.append("-" * 60)
    
    total_missing = 0
    for col in key_columns:
        label = column_labels.get(col, col)
        missing = report["missing_values"][col]
        total_missing += missing
        lines.append(f"  {label}: {missing}")
    
    lines.append(f"\n  Total missing values: {total_missing}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("End of Report")
    lines.append("=" * 60)
    
    # Write report to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        f.write('\n'.join(lines))
    
    return report
