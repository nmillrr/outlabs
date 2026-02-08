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


def clean_glycemic_data(
    ghb_df: pd.DataFrame,
    glu_df: pd.DataFrame,
    trigly_df: pd.DataFrame,
    hdl_df: pd.DataFrame,
    cbc_df: pd.DataFrame,
    demo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and merge NHANES glycemic datasets.

    Merges datasets on SEQN (sample ID), renames columns to standardized names,
    removes physiologic outliers, and returns only complete cases.

    Args:
        ghb_df: DataFrame from GHB (glycohemoglobin/HbA1c) XPT file.
        glu_df: DataFrame from GLU (fasting plasma glucose) XPT file.
        trigly_df: DataFrame from TRIGLY (triglycerides) XPT file.
        hdl_df: DataFrame from HDL (HDL cholesterol) XPT file.
        cbc_df: DataFrame from CBC (complete blood count) XPT file.
        demo_df: DataFrame from DEMO (demographics) XPT file.

    Returns:
        Cleaned DataFrame with columns:
        - hba1c_percent: HbA1c in percent (from LBXGH)
        - fpg_mgdl: Fasting plasma glucose in mg/dL (from LBXGLU)
        - tg_mgdl: Triglycerides in mg/dL (from LBXTR)
        - hdl_mgdl: HDL cholesterol in mg/dL (from LBDHDD)
        - hgb_gdl: Hemoglobin in g/dL (from LBXHGB)
        - mcv_fl: Mean corpuscular volume in fL (from LBXMCVSI)
        - age_years: Age in years (from RIDAGEYR)
        - sex: Sex (1=male, 2=female, from RIAGENDR)

    Note:
        Outliers are removed based on physiologic plausibility:
        - HbA1c < 3% or > 20%
        - FPG < 40 or > 600 mg/dL
    """
    # Select relevant columns from each dataset
    ghb = ghb_df[["SEQN", "LBXGH"]].copy()
    glu = glu_df[["SEQN", "LBXGLU"]].copy()
    trigly = trigly_df[["SEQN", "LBXTR"]].copy()
    hdl = hdl_df[["SEQN", "LBDHDD"]].copy()
    cbc = cbc_df[["SEQN", "LBXHGB", "LBXMCVSI"]].copy()
    demo = demo_df[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()

    # Merge all datasets on SEQN
    merged = ghb.merge(glu, on="SEQN", how="inner")
    merged = merged.merge(trigly, on="SEQN", how="inner")
    merged = merged.merge(hdl, on="SEQN", how="inner")
    merged = merged.merge(cbc, on="SEQN", how="inner")
    merged = merged.merge(demo, on="SEQN", how="inner")

    # Rename columns to standardized names
    column_mapping = {
        "LBXGH": "hba1c_percent",
        "LBXGLU": "fpg_mgdl",
        "LBXTR": "tg_mgdl",
        "LBDHDD": "hdl_mgdl",
        "LBXHGB": "hgb_gdl",
        "LBXMCVSI": "mcv_fl",
        "RIDAGEYR": "age_years",
        "RIAGENDR": "sex",
    }
    merged = merged.rename(columns=column_mapping)

    # Remove SEQN column (no longer needed after merging)
    merged = merged.drop(columns=["SEQN"])

    # Drop rows with any missing values (complete cases only)
    cleaned = merged.dropna()

    # Remove physiologic outliers
    # HbA1c: < 3% or > 20% are not physiologically plausible
    cleaned = cleaned[(cleaned["hba1c_percent"] >= 3) & (cleaned["hba1c_percent"] <= 20)]

    # FPG: < 40 or > 600 mg/dL are not physiologically plausible
    cleaned = cleaned[(cleaned["fpg_mgdl"] >= 40) & (cleaned["fpg_mgdl"] <= 600)]

    # Reset index after filtering
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def generate_quality_report(df: pd.DataFrame, output_path: str) -> Dict[str, Any]:
    """
    Generate a data quality report for cleaned glycemic data.

    Creates a comprehensive quality report including record counts,
    descriptive statistics, and clinical distribution breakdowns for
    HbA1c and fasting plasma glucose.

    Args:
        df: Cleaned DataFrame from clean_glycemic_data().
        output_path: Path where the report will be saved as a text file.

    Returns:
        Dict containing all report metrics:
        - record_count: Total number of records
        - stats: Dict of mean/SD for each variable
        - hba1c_distribution: Dict with counts for clinical categories
        - fpg_distribution: Dict with counts for clinical categories

    Example:
        >>> report = generate_quality_report(cleaned_df, "data/quality_report.txt")
        >>> print(report['record_count'])
        8542
    """
    report: Dict[str, Any] = {}

    # Record count
    report["record_count"] = len(df)

    # Descriptive statistics (mean/SD) for key variables
    variables = ["fpg_mgdl", "hba1c_percent", "tg_mgdl", "hdl_mgdl", "hgb_gdl", "mcv_fl"]
    stats: Dict[str, Dict[str, float]] = {}
    for var in variables:
        if var in df.columns:
            stats[var] = {
                "mean": float(df[var].mean()),
                "std": float(df[var].std()),
            }
    report["stats"] = stats

    # HbA1c distribution breakdown (clinical categories)
    # <5.7% (normal), 5.7-6.4% (prediabetes), ≥6.5% (diabetes)
    hba1c = df["hba1c_percent"]
    hba1c_dist = {
        "normal_lt_5.7": int((hba1c < 5.7).sum()),
        "prediabetes_5.7_to_6.4": int(((hba1c >= 5.7) & (hba1c < 6.5)).sum()),
        "diabetes_gte_6.5": int((hba1c >= 6.5).sum()),
    }
    report["hba1c_distribution"] = hba1c_dist

    # FPG distribution breakdown (clinical categories)
    # <100 mg/dL (normal), 100-125 mg/dL (prediabetes), ≥126 mg/dL (diabetes)
    fpg = df["fpg_mgdl"]
    fpg_dist = {
        "normal_lt_100": int((fpg < 100).sum()),
        "prediabetes_100_to_125": int(((fpg >= 100) & (fpg < 126)).sum()),
        "diabetes_gte_126": int((fpg >= 126).sum()),
    }
    report["fpg_distribution"] = fpg_dist

    # Generate text report
    report_lines = [
        "=" * 60,
        "NHANES GLYCEMIC DATA QUALITY REPORT",
        "=" * 60,
        "",
        f"Total Records: {report['record_count']}",
        "",
        "-" * 40,
        "DESCRIPTIVE STATISTICS (Mean ± SD)",
        "-" * 40,
    ]

    var_labels = {
        "fpg_mgdl": "Fasting Plasma Glucose (mg/dL)",
        "hba1c_percent": "HbA1c (%)",
        "tg_mgdl": "Triglycerides (mg/dL)",
        "hdl_mgdl": "HDL Cholesterol (mg/dL)",
        "hgb_gdl": "Hemoglobin (g/dL)",
        "mcv_fl": "Mean Corpuscular Volume (fL)",
    }

    for var, label in var_labels.items():
        if var in stats:
            mean = stats[var]["mean"]
            std = stats[var]["std"]
            report_lines.append(f"{label}: {mean:.2f} ± {std:.2f}")

    report_lines.extend([
        "",
        "-" * 40,
        "HbA1c DISTRIBUTION (Clinical Categories)",
        "-" * 40,
        f"Normal (<5.7%):      {hba1c_dist['normal_lt_5.7']:,} ({100*hba1c_dist['normal_lt_5.7']/len(df):.1f}%)",
        f"Prediabetes (5.7-6.4%): {hba1c_dist['prediabetes_5.7_to_6.4']:,} ({100*hba1c_dist['prediabetes_5.7_to_6.4']/len(df):.1f}%)",
        f"Diabetes (≥6.5%):    {hba1c_dist['diabetes_gte_6.5']:,} ({100*hba1c_dist['diabetes_gte_6.5']/len(df):.1f}%)",
        "",
        "-" * 40,
        "FPG DISTRIBUTION (Clinical Categories)",
        "-" * 40,
        f"Normal (<100 mg/dL):     {fpg_dist['normal_lt_100']:,} ({100*fpg_dist['normal_lt_100']/len(df):.1f}%)",
        f"Prediabetes (100-125):   {fpg_dist['prediabetes_100_to_125']:,} ({100*fpg_dist['prediabetes_100_to_125']/len(df):.1f}%)",
        f"Diabetes (≥126 mg/dL):   {fpg_dist['diabetes_gte_126']:,} ({100*fpg_dist['diabetes_gte_126']/len(df):.1f}%)",
        "",
        "=" * 60,
    ])

    # Write report to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
