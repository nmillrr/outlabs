"""
eGFR/data.py — NHANES Data Pipeline

Provides functions for:
  - Downloading NHANES kidney function data (BIOPRO, DEMO, BMX, cystatin C)
  - Parsing SAS transport (XPT) files into pandas DataFrames
  - Cleaning and harmonizing kidney function datasets
  - Generating data quality reports
"""

import os
import urllib.request
import urllib.error
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# XPT (SAS transport) file reader
# ---------------------------------------------------------------------------

def read_xpt(filepath: str) -> pd.DataFrame:
    """Read a SAS transport (XPT) file and return a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ``.XPT`` file to read.

    Returns
    -------
    pd.DataFrame
        The contents of the XPT file as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be parsed as a valid SAS transport file.

    Examples
    --------
    >>> from eGFR.data import read_xpt
    >>> df = read_xpt("data/raw/BIOPRO_J.XPT")
    >>> df.head()
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"XPT file not found: '{filepath}'. "
            "Please check the path or download the data first using "
            "download_nhanes_kidney()."
        )

    try:
        df = pd.read_sas(filepath, format="xport", encoding="utf-8")
    except Exception as exc:
        raise ValueError(
            f"Failed to parse '{filepath}' as a SAS transport (XPT) file: {exc}"
        ) from exc

    return df


# ---------------------------------------------------------------------------
# NHANES cycle name suffixes used in CDC URLs
# e.g. 2005-2006 → "_D", 2007-2008 → "_E", etc.
# ---------------------------------------------------------------------------
_CYCLE_SUFFIX = {
    "1999-2000": "",
    "2001-2002": "_B",
    "2003-2004": "_C",
    "2005-2006": "_D",
    "2007-2008": "_E",
    "2009-2010": "_F",
    "2011-2012": "_G",
    "2013-2014": "_H",
    "2015-2016": "_I",
    "2017-2018": "_J",
}

# NHANES base URL for XPT files
_NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Nhanes"

# Default cycles for the main download (BIOPRO, DEMO, BMX)
DEFAULT_CYCLES: List[str] = [
    "2005-2006",
    "2007-2008",
    "2009-2010",
    "2011-2012",
    "2013-2014",
    "2015-2016",
    "2017-2018",
]

# Cystatin C (special survey) cycles — only available 1999-2002
CYSTATIN_CYCLES: List[str] = [
    "1999-2000",
    "2001-2002",
]


def _build_url(cycle: str, dataset: str) -> str:
    """Build the CDC NHANES download URL for a given cycle and dataset.

    Parameters
    ----------
    cycle : str
        NHANES cycle string, e.g. "2005-2006".
    dataset : str
        Dataset prefix, e.g. "BIOPRO", "DEMO", "BMX", "SSPRT".

    Returns
    -------
    str
        Full URL to the XPT file on the CDC server.
    """
    suffix = _CYCLE_SUFFIX.get(cycle, "")
    filename = f"{dataset}{suffix}.XPT"
    return f"{_NHANES_BASE}/{cycle}/{filename}"


def _download_file(url: str, dest_path: str) -> bool:
    """Download a single file from *url* to *dest_path*.

    Returns True on success, False on failure (with a printed message).
    """
    try:
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✓ Saved to {dest_path}")
        return True
    except urllib.error.HTTPError as exc:
        print(f"  ✗ HTTP {exc.code} for {url} — skipping.")
        return False
    except urllib.error.URLError as exc:
        print(f"  ✗ Network error for {url}: {exc.reason} — skipping.")
        return False
    except OSError as exc:
        print(f"  ✗ File error saving {dest_path}: {exc} — skipping.")
        return False


def download_nhanes_kidney(
    output_dir: str = "data/raw",
    cycles: Optional[List[str]] = None,
) -> dict:
    """Download NHANES kidney-function XPT files for the requested cycles.

    Downloads:
      - **BIOPRO** (biochemistry profile — contains serum creatinine LBXSCR)
      - **DEMO** (demographics — age, sex)
      - **BMX** (body measures — weight, height for Cockcroft-Gault)
    for each cycle in *cycles* (default 2005-2018).

    Additionally downloads:
      - **SSPRT** (cystatin C) for cycles 1999-2000 and 2001-2002 where
        the special survey data is available.

    Parameters
    ----------
    output_dir : str, default "data/raw"
        Directory where XPT files will be saved.  Created if it does not
        exist.
    cycles : list of str, optional
        NHANES cycles to download.  Each entry should be a string like
        ``"2005-2006"``.  Defaults to ``DEFAULT_CYCLES`` (2005-2018).

    Returns
    -------
    dict
        A summary dict with keys ``"downloaded"`` (list of file paths) and
        ``"failed"`` (list of URLs that could not be downloaded).

    Examples
    --------
    >>> from eGFR.data import download_nhanes_kidney
    >>> summary = download_nhanes_kidney("data/raw", ["2017-2018"])
    """
    if cycles is None:
        cycles = DEFAULT_CYCLES

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    downloaded: List[str] = []
    failed: List[str] = []

    # --- Main datasets (BIOPRO, DEMO, BMX) for each requested cycle --------
    datasets = ["BIOPRO", "DEMO", "BMX"]

    for cycle in cycles:
        if cycle not in _CYCLE_SUFFIX:
            print(f"WARNING: Unknown NHANES cycle '{cycle}' — skipping.")
            continue

        print(f"\n=== Cycle {cycle} ===")
        for ds in datasets:
            url = _build_url(cycle, ds)
            suffix = _CYCLE_SUFFIX[cycle]
            filename = f"{ds}{suffix}.XPT"
            dest = os.path.join(output_dir, filename)

            if os.path.exists(dest):
                print(f"  • {filename} already exists — skipping download.")
                downloaded.append(dest)
                continue

            if _download_file(url, dest):
                downloaded.append(dest)
            else:
                failed.append(url)

    # --- Cystatin C (SSPRT) — only for 1999-2002 --------------------------
    print("\n=== Cystatin C (SSPRT) special survey ===")
    for cycle in CYSTATIN_CYCLES:
        url = _build_url(cycle, "SSPRT")
        suffix = _CYCLE_SUFFIX[cycle]
        filename = f"SSPRT{suffix}.XPT"
        dest = os.path.join(output_dir, filename)

        if os.path.exists(dest):
            print(f"  • {filename} already exists — skipping download.")
            downloaded.append(dest)
            continue

        if _download_file(url, dest):
            downloaded.append(dest)
        else:
            failed.append(url)

    # --- Summary -----------------------------------------------------------
    print(f"\nDone. {len(downloaded)} files saved, {len(failed)} failed.")
    return {"downloaded": downloaded, "failed": failed}


# ---------------------------------------------------------------------------
# IDMS creatinine standardization correction factor
# Pre-2007 NHANES creatinine was NOT calibrated to IDMS standards.
# Correction: adjusted_Cr = 0.95 × original_Cr
# Reference: Selvin et al., Clin Chem 2007
# ---------------------------------------------------------------------------
_IDMS_CORRECTION_FACTOR = 0.95


# ---------------------------------------------------------------------------
# Data cleaning / harmonization
# ---------------------------------------------------------------------------

# Columns we expect from each NHANES dataset and their standardized names
_BIOPRO_RENAME: Dict[str, str] = {
    "SEQN": "seqn",
    "LBXSCR": "cr_mgdl",
}

_DEMO_RENAME: Dict[str, str] = {
    "SEQN": "seqn",
    "RIDAGEYR": "age_years",
    "RIAGENDR": "sex",
}

_BMX_RENAME: Dict[str, str] = {
    "SEQN": "seqn",
    "BMXWT": "weight_kg",
    "BMXHT": "height_cm",
}

_CYSTATIN_RENAME: Dict[str, str] = {
    "SEQN": "seqn",
    "SSPRT": "cystatin_c_mgL",
}


def clean_kidney_data(
    biopro_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    bmx_df: pd.DataFrame,
    cystatin_df: Optional[pd.DataFrame] = None,
    *,
    apply_idms_correction: bool = False,
) -> pd.DataFrame:
    """Clean and merge NHANES kidney-function datasets.

    Merges BIOPRO (creatinine), DEMO (demographics), BMX (body measures),
    and optionally cystatin C data on SEQN.  Renames columns to a
    standardised schema, applies IDMS creatinine correction for pre-2007
    cycles when requested, removes physiologic outliers, and returns
    complete cases.

    Parameters
    ----------
    biopro_df : pd.DataFrame
        Biochemistry profile data containing at least ``SEQN`` and ``LBXSCR``.
    demo_df : pd.DataFrame
        Demographics data containing at least ``SEQN``, ``RIDAGEYR``, and
        ``RIAGENDR``.
    bmx_df : pd.DataFrame
        Body measures data containing at least ``SEQN``, ``BMXWT``, and
        ``BMXHT``.
    cystatin_df : pd.DataFrame, optional
        Cystatin C data containing at least ``SEQN`` and ``SSPRT``.
    apply_idms_correction : bool, default False
        If ``True``, multiply creatinine values by 0.95 to correct for
        non-IDMS-standardised assays (pre-2007 NHANES data).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: ``seqn``, ``cr_mgdl``,
        ``age_years``, ``sex``, ``weight_kg``, ``height_cm``, and
        optionally ``cystatin_c_mgL``.  Only complete (non-NaN) rows are
        returned.

    Raises
    ------
    ValueError
        If any required column is missing from the input DataFrames.
    """

    # ── Validate required columns ──────────────────────────────────────
    _validate_columns(biopro_df, {"SEQN", "LBXSCR"}, "biopro_df")
    _validate_columns(demo_df, {"SEQN", "RIDAGEYR", "RIAGENDR"}, "demo_df")
    _validate_columns(bmx_df, {"SEQN", "BMXWT", "BMXHT"}, "bmx_df")
    if cystatin_df is not None:
        _validate_columns(cystatin_df, {"SEQN", "SSPRT"}, "cystatin_df")

    # ── Select & rename ────────────────────────────────────────────────
    bio = biopro_df[list(_BIOPRO_RENAME.keys())].rename(columns=_BIOPRO_RENAME)
    dem = demo_df[list(_DEMO_RENAME.keys())].rename(columns=_DEMO_RENAME)
    bmx = bmx_df[list(_BMX_RENAME.keys())].rename(columns=_BMX_RENAME)

    # ── Merge on SEQN (inner join) ─────────────────────────────────────
    merged = bio.merge(dem, on="seqn", how="inner").merge(
        bmx, on="seqn", how="inner"
    )

    # Optionally merge cystatin C
    if cystatin_df is not None:
        cys = cystatin_df[list(_CYSTATIN_RENAME.keys())].rename(
            columns=_CYSTATIN_RENAME
        )
        merged = merged.merge(cys, on="seqn", how="left")

    # ── IDMS creatinine correction (pre-2007 data) ─────────────────────
    if apply_idms_correction:
        merged["cr_mgdl"] = merged["cr_mgdl"] * _IDMS_CORRECTION_FACTOR

    # ── Remove physiologic outliers ────────────────────────────────────
    n_before = len(merged)

    # Creatinine outside 0.2–15 mg/dL
    cr_mask = (merged["cr_mgdl"] >= 0.2) & (merged["cr_mgdl"] <= 15.0)
    # Adults only (age >= 18)
    age_mask = merged["age_years"] >= 18

    merged = merged.loc[cr_mask & age_mask].copy()

    n_outliers = n_before - len(merged)
    if n_outliers > 0:
        warnings.warn(
            f"Removed {n_outliers} rows as physiologic outliers "
            f"(creatinine outside 0.2–15 mg/dL or age < 18)."
        )

    # ── Drop rows with any remaining NaN in core columns ───────────────
    core_cols = ["seqn", "cr_mgdl", "age_years", "sex", "weight_kg", "height_cm"]
    merged = merged.dropna(subset=core_cols).reset_index(drop=True)

    return merged


def _validate_columns(
    df: pd.DataFrame, required: set, df_name: str
) -> None:
    """Raise ``ValueError`` if *df* is missing any *required* columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{df_name} is missing required column(s): {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Private CKD-EPI 2021 helper (used by quality report before models.py
# implements the full public API)
# ---------------------------------------------------------------------------

def _calc_ckd_epi_2021(cr_mgdl: float, age_years: float, sex: int) -> float:
    """Compute CKD-EPI 2021 eGFR for a single observation.

    Parameters
    ----------
    cr_mgdl : float
        Serum creatinine in mg/dL.
    age_years : float
        Age in years (≥ 18).
    sex : int
        1 = male, 2 = female (NHANES coding).

    Returns
    -------
    float
        Estimated GFR in mL/min/1.73 m².
    """
    is_female = sex == 2
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    female_factor = 1.012 if is_female else 1.0

    cr_ratio = cr_mgdl / kappa
    return (
        142.0
        * min(cr_ratio, 1.0) ** alpha
        * max(cr_ratio, 1.0) ** (-1.200)
        * (0.9938 ** age_years)
        * female_factor
    )


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------

def generate_quality_report(df: pd.DataFrame, output_path: str) -> str:
    """Generate a data quality report for a cleaned kidney-function DataFrame.

    The report includes:
      - Total record count
      - Mean and standard deviation for creatinine, age, weight, height,
        and cystatin C (if present)
      - CKD stage distribution computed via CKD-EPI 2021 eGFR
      - Sex distribution breakdown

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame produced by :func:`clean_kidney_data`.  Must
        contain at least ``cr_mgdl``, ``age_years``, ``sex``, ``weight_kg``,
        and ``height_cm``.
    output_path : str
        File path where the text report will be saved.  Parent directories
        are created automatically if they do not exist.

    Returns
    -------
    str
        The full report text (also saved to *output_path*).

    Raises
    ------
    ValueError
        If *df* is missing required columns.
    """
    from eGFR.utils import egfr_to_ckd_stage

    # ── Validate required columns ──────────────────────────────────────
    required = {"cr_mgdl", "age_years", "sex", "weight_kg", "height_cm"}
    _validate_columns(df, required, "df")

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  DATA QUALITY REPORT")
    lines.append("=" * 60)
    lines.append("")

    # ── Record count ───────────────────────────────────────────────────
    lines.append(f"Total records: {len(df)}")
    lines.append("")

    # ── Descriptive statistics ─────────────────────────────────────────
    lines.append("-" * 40)
    lines.append("Descriptive Statistics")
    lines.append("-" * 40)

    stat_columns = [
        ("cr_mgdl", "Creatinine (mg/dL)"),
        ("age_years", "Age (years)"),
        ("weight_kg", "Weight (kg)"),
        ("height_cm", "Height (cm)"),
    ]
    if "cystatin_c_mgL" in df.columns:
        stat_columns.append(("cystatin_c_mgL", "Cystatin C (mg/L)"))

    for col, label in stat_columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        lines.append(f"  {label}: mean={mean_val:.2f}, SD={std_val:.2f}")

    lines.append("")

    # ── CKD stage distribution (CKD-EPI 2021) ─────────────────────────
    lines.append("-" * 40)
    lines.append("CKD Stage Distribution (CKD-EPI 2021)")
    lines.append("-" * 40)

    egfr_values = df.apply(
        lambda row: _calc_ckd_epi_2021(row["cr_mgdl"], row["age_years"], row["sex"]),
        axis=1,
    )
    stages = egfr_values.apply(egfr_to_ckd_stage)
    stage_counts = stages.value_counts()

    for stage in ["G1", "G2", "G3a", "G3b", "G4", "G5"]:
        count = stage_counts.get(stage, 0)
        pct = 100.0 * count / len(df) if len(df) > 0 else 0.0
        lines.append(f"  {stage}: {count} ({pct:.1f}%)")

    lines.append("")

    # ── Sex distribution ───────────────────────────────────────────────
    lines.append("-" * 40)
    lines.append("Sex Distribution")
    lines.append("-" * 40)

    sex_counts = df["sex"].value_counts()
    n_male = sex_counts.get(1, 0)
    n_female = sex_counts.get(2, 0)
    pct_male = 100.0 * n_male / len(df) if len(df) > 0 else 0.0
    pct_female = 100.0 * n_female / len(df) if len(df) > 0 else 0.0
    lines.append(f"  Male:   {n_male} ({pct_male:.1f}%)")
    lines.append(f"  Female: {n_female} ({pct_female:.1f}%)")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)

    # ── Save to file ───────────────────────────────────────────────────
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report
