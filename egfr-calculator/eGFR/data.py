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
from typing import List, Optional

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
