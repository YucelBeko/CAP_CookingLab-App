"""
data_merger_core_review_v1.py — Datalogger Dashboard Pro  (v4.0)
==========================================
Pure data / business-logic layer.
No GUI imports — this module must stay framework-agnostic so it can be
unit-tested without a display, used from a CLI, or bundled headlessly.

Public surface
--------------
read_udaq(path)          → pd.DataFrame
read_gm10(path)          → tuple[pd.DataFrame, pd.DataFrame]   (raw, clean)
read_vts(path)           → pd.DataFrame
build_report(...)        → pd.DataFrame
save_excel(...)          → None
AlarmLine                dataclass
VeriOkumaHatasi          exception
VeriBirlestirmeHatasi    exception
resource_path(rel)       → str
setup_logger()           → logging.Logger
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# App metadata (single source of truth — imported by app.py)
# ---------------------------------------------------------------------------

APP_TITLE   = "Datalogger Dashboard Pro"
APP_VERSION = "v4.0"
COMPANY     = "Beko PCI"
CHART_FOOTER = ""          # Optional watermark text shown on every chart

# ---------------------------------------------------------------------------
# Brand palette constants (also imported by app.py for consistent styling)
# ---------------------------------------------------------------------------

C_PRIMARY  = "#001986"
C_SUCCESS  = "#00864e"
C_DANGER   = "#e63757"
C_WARNING  = "#f6c343"
C_DARK     = "#12263f"
C_LIGHT_BG = "#f9fbfd"
C_PANEL_BG = "#edf2f9"

ALARM_COLOR_PRESETS = ["#e63757", "#f6c343", "#2c7be5", "#00864e", "#9b59b6"]


# ---------------------------------------------------------------------------
# PyInstaller-compatible resource resolver
# ---------------------------------------------------------------------------

def resource_path(relative: str) -> str:
    """Return absolute path to a bundled resource.

    Works both in development (uses __file__ directory) and when frozen
    by PyInstaller (_MEIPASS is injected at runtime).
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


# ---------------------------------------------------------------------------
# Logger — writes to hata_log.txt next to the executable
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    """Configure and return the application logger.

    Log file is created alongside the executable (or script directory during
    development) so end-users can find it easily.
    """
    log_dir  = os.path.dirname(os.path.abspath(getattr(sys, "executable", __file__)))
    log_file = os.path.join(log_dir, "hata_log.txt")

    logger = logging.getLogger("DataloggerDashboard")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        try:
            handler = logging.FileHandler(log_file, encoding="utf-8")
        except OSError:
            # Streamlit Cloud veya salt-okunur ortamlarda modül importu
            # log dosyası yüzünden başarısız olmasın.
            handler = logging.StreamHandler()

        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


log = setup_logger()


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class VeriOkumaHatasi(Exception):
    """Raised when a data file cannot be read or parsed."""

class VeriBirlestirmeHatasi(Exception):
    """Raised when report assembly fails (e.g. no valid DataFrames provided)."""


# ---------------------------------------------------------------------------
# AlarmLine — value model, no GUI state
# ---------------------------------------------------------------------------

@dataclass
class AlarmLine:
    """Represents a single reference / alarm annotation drawn on the chart.

    `id` is auto-assigned from a class-level counter so each instance is
    uniquely addressable without requiring the caller to track IDs.

    Attributes
    ----------
    value     : numeric position on the axis
    label     : text shown in the legend
    color     : CSS hex string, e.g. '#e63757'
    style     : matplotlib linestyle string ('--', '-', '-.', ':')
    thickness : line width in points
    active    : when False the line is hidden but kept in the session
    direction : 'horizontal' → axhline (y value),
                'vertical'   → axvline (x / minute value)
    """

    value    : float
    label    : str
    color    : str
    style    : str  = "--"
    thickness: float = 1.5
    active   : bool  = True
    direction: str   = "horizontal"   # 'horizontal' | 'vertical'

    # Auto-increment ID — shared across all instances
    _counter: int = field(default=0, init=False, repr=False, compare=False)
    id      : int = field(default=0, init=False)

    # Class-level counter kept outside the dataclass machinery
    _id_counter: int = 0

    def __post_init__(self) -> None:
        AlarmLine._id_counter += 1
        object.__setattr__(self, "id", AlarmLine._id_counter)

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the ID counter — call before restoring a saved session."""
        cls._id_counter = 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for session persistence."""
        return {
            "value"    : self.value,
            "label"    : self.label,
            "color"    : self.color,
            "style"    : self.style,
            "thickness": self.thickness,
            "active"   : self.active,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AlarmLine":
        """Deserialize from a session JSON dict."""
        obj = cls(
            value    =data["value"],
            label    =data["label"],
            color    =data["color"],
            style    =data.get("style",     "--"),
            thickness=data.get("thickness", 1.5),
            active   =data.get("active",    True),
            direction=data.get("direction", "horizontal"),
        )
        return obj


# ---------------------------------------------------------------------------
# UDAQ reader  (.log / .txt, tab-separated, UTF-8 / CP1254 / Latin-1)
# ---------------------------------------------------------------------------

_UDAQ_ENCODINGS = ("utf-8", "cp1254", "latin-1", "iso-8859-9")


def read_udaq(path: str) -> pd.DataFrame:
    """Read a UDAQ datalogger export file.

    Tries multiple encodings in order.  Returns a clean DataFrame with
    all-NaN columns dropped.

    Raises
    ------
    VeriOkumaHatasi
        If the file cannot be opened or produces an empty result.
    """
    last_error: Optional[Exception] = None

    for enc in _UDAQ_ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                encoding=enc,
                skiprows=1,
                on_bad_lines="skip",
                low_memory=False,
            )
            df = df.dropna(axis=1, how="all")

            if df.empty:
                raise VeriOkumaHatasi("UDAQ file was read but contains no data.")

            log.debug(f"read_udaq: loaded {path!r} with encoding={enc}, shape={df.shape}")
            return df

        except UnicodeDecodeError:
            continue   # try next encoding
        except pd.errors.EmptyDataError:
            raise VeriOkumaHatasi("Selected UDAQ file is empty.")
        except VeriOkumaHatasi:
            raise
        except Exception as exc:
            last_error = exc
            if enc == _UDAQ_ENCODINGS[-1]:
                raise VeriOkumaHatasi(f"Could not read UDAQ file: {exc}") from exc

    raise VeriOkumaHatasi("UDAQ file could not be opened with any supported encoding.")


# ---------------------------------------------------------------------------
# GM10 reader  (.xlsx / .xls)
# ---------------------------------------------------------------------------

def _clean_gm10(raw: pd.DataFrame) -> pd.DataFrame:
    """Locate the real header row in a raw GM10 Excel export and tidy up.

    GM10 files embed several metadata rows before the actual data header.
    The strategy is to scan the first 50 rows for a line that contains
    both 'date' and 'time' (case-insensitive), then merge the two rows
    immediately above it to reconstruct meaningful column names.

    Raises
    ------
    VeriOkumaHatasi
        If the expected header structure is not found.
    """
    header_idx = -1

    for i in range(min(50, len(raw))):
        row_text = " ".join(raw.iloc[i].dropna().astype(str)).lower()
        if "date" in row_text and "time" in row_text:
            header_idx = i
            break

    if header_idx == -1:
        raise VeriOkumaHatasi(
            "GM10 file: could not find 'Date' and 'Time' header row.\n"
            "Make sure the file is an unmodified GM10 export."
        )

    if header_idx < 2:
        raise VeriOkumaHatasi(
            "GM10 file: header row is unexpectedly close to the top (row < 2). "
            "File may be truncated or in an unsupported format."
        )

    # Row two lines above header contains the descriptive channel names;
    # the header row itself contains unit / sub-labels.  Merge: prefer the
    # descriptive name when present, fall back to the sub-label.
    descriptive = raw.iloc[header_idx - 2].fillna("").astype(str)
    sublabels   = raw.iloc[header_idx    ].fillna("").astype(str)

    columns = [
        d.strip() if (d := desc.strip()) and d.lower() != "nan" else sub.strip()
        for desc, sub in zip(descriptive, sublabels)
    ]

    data = raw.iloc[header_idx + 1:].copy()
    data.columns = columns

    return (
        data
        .reset_index(drop=True)
        .dropna(how="all", axis=1)
        .dropna(how="all", axis=0)
    )


def read_gm10(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a GM10 Excel file.

    Returns
    -------
    (raw_df, clean_df)
        raw_df  : unmodified DataFrame straight from the Excel parser
        clean_df: header-corrected, NaN-stripped DataFrame ready for merging

    Raises
    ------
    VeriOkumaHatasi
    """
    # Prefer calamine (faster, no Java) if installed; fall back to openpyxl.
    engines: list[str] = []
    try:
        import python_calamine  # noqa: F401
        engines.append("calamine")
    except ImportError:
        pass
    engines.append("openpyxl")

    last_error: Optional[Exception] = None

    for engine in engines:
        try:
            raw   = pd.read_excel(path, header=None, engine=engine)
            clean = _clean_gm10(raw)
            log.debug(f"read_gm10: loaded {path!r} via {engine}, shape={clean.shape}")
            return raw, clean
        except VeriOkumaHatasi:
            raise   # propagate our own errors immediately
        except Exception as exc:
            last_error = exc

    raise VeriOkumaHatasi(
        f"GM10 file could not be opened: {last_error}\n"
        "Check that the file is not corrupted and is not open in another program."
    )


# ---------------------------------------------------------------------------
# VTS reader  (.xlsx / .xls / .csv)
# ---------------------------------------------------------------------------

def read_vts(path: str) -> pd.DataFrame:
    """Read a VTS file (Excel or CSV).

    Raises
    ------
    VeriOkumaHatasi
    """
    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding="utf-8-sig",
                on_bad_lines="skip",
            )
        else:
            df = pd.read_excel(path)

        df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
        log.debug(f"read_vts: loaded {path!r}, shape={df.shape}")
        return df

    except Exception as exc:
        raise VeriOkumaHatasi(
            f"Could not read VTS file: {exc}\n"
            "Make sure the file is not open in another program."
        ) from exc


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

# Source exports use several names for redundant time/index columns. We rebuild
# a clean Step/Minute index after merging, so these columns are removed per
# source before concatenation.
_TIME_COLUMN_TOKENS = {
    "date", "time", "datetime", "timestamp", "zaman", "ms", "minute", "step"
}


def _is_redundant_time_column(column_name: object) -> bool:
    """Return True when a column name represents a time/index field."""
    text = str(column_name).strip().lower()
    if not text:
        return False

    normalized = text
    for char in "_-/().[]":
        normalized = normalized.replace(char, " ")

    tokens = set(normalized.split())
    return bool(tokens & _TIME_COLUMN_TOKENS)


def _make_unique_columns(
    columns: list[object],
    source_name: str,
    used_names: set[str],
) -> list[str]:
    """Create stable unique report column names without changing unique names."""
    result: list[str] = []
    local_counts: dict[str, int] = {}

    for index, raw_name in enumerate(columns, start=1):
        base = str(raw_name).strip()
        if not base or base.lower() == "nan":
            base = f"Unnamed_{index}"

        local_counts[base] = local_counts.get(base, 0) + 1
        candidate = base

        # Only duplicate names receive a source suffix; existing unique names
        # remain exactly as users already see them.
        if candidate in used_names or local_counts[base] > 1:
            candidate = f"{base} [{source_name}]"

        suffix = 2
        while candidate in used_names:
            candidate = f"{base} [{source_name} {suffix}]"
            suffix += 1

        used_names.add(candidate)
        result.append(candidate)

    return result


def build_report(
    gm10_clean : Optional[pd.DataFrame],
    vts_clean  : Optional[pd.DataFrame],
    udaq_clean : Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge available sources into one report with unique channel names.

    Each source is cleaned independently before concatenation. Redundant
    time/index columns are removed from every source, not just from the first
    columns of the merged table. Duplicate channel names receive a source
    suffix so downstream plotting always receives a Series rather than a
    duplicate-column DataFrame.
    """
    parts: list[pd.DataFrame] = []
    used_names: set[str] = set()

    for source_name, df in (
        ("GM10", gm10_clean),
        ("VTS", vts_clean),
        ("UDAQ", udaq_clean),
    ):
        if df is None:
            continue

        copy = df.copy().reset_index(drop=True)
        copy.columns = copy.columns.astype(str)

        keep_columns = [
            col for col in copy.columns
            if not _is_redundant_time_column(col)
        ]
        copy = copy.loc[:, keep_columns]

        copy.columns = _make_unique_columns(
            list(copy.columns),
            source_name,
            used_names,
        )
        parts.append(copy)

    if not parts:
        raise VeriBirlestirmeHatasi("No data to merge — all sources are None.")

    report = pd.concat(parts, axis=1)

    n = len(report)
    report.insert(0, "Step", range(1, n + 1))
    report.insert(1, "Minute", np.arange(1, n + 1) / 60.0)

    log.debug(f"build_report: final shape={report.shape}")
    return report


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def save_excel(
    path      : str,
    report    : pd.DataFrame,
    gm10_raw  : Optional[pd.DataFrame] = None,
    vts_raw   : Optional[pd.DataFrame] = None,
    udaq_raw  : Optional[pd.DataFrame] = None,
) -> None:
    """Write the merged report and optional raw sources to an .xlsx file.

    Sheet layout
    ------------
    Ozet_Rapor  : merged, cleaned report (always present)
    GM10_Ham    : raw GM10 export          (if gm10_raw is not None)
    VTS_Ham     : raw VTS export           (if vts_raw  is not None)
    UDAQ_Ham    : raw UDAQ export          (if udaq_raw is not None)
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="Ozet_Rapor", index=False)

        if gm10_raw is not None:
            gm10_raw.to_excel(writer, sheet_name="GM10_Ham", index=False, header=False)

        if vts_raw is not None:
            vts_raw.to_excel(writer, sheet_name="VTS_Ham", index=False)

        if udaq_raw is not None:
            udaq_raw.to_excel(writer, sheet_name="UDAQ_Ham", index=False)

    log.info(f"save_excel: written to {path!r}")


# ---------------------------------------------------------------------------
# Column-type classifier (shared between core and app layers)
# ---------------------------------------------------------------------------

def classify_columns(
    df      : pd.DataFrame,
    columns : list[str],
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Classify numeric channels as analog or discrete/digital.

    A channel is treated as digital only when it has a small number of
    integer-like states within 0..20. This keeps relay/state channels on the
    right axis while avoiding the old false-positive case where any analog
    signal below 20 was classified as digital.
    """
    col_types : dict[str, str] = {}
    col_arrays: dict[str, np.ndarray] = {}

    for col in columns:
        series = df[col]
        if isinstance(series, pd.DataFrame):
            raise ValueError(
                f"Duplicate report column detected: {col!r}. "
                "Rebuild the report with build_report()."
            )

        numeric = pd.to_numeric(series, errors="coerce")
        col_arrays[col] = numeric.to_numpy(dtype=float)

        valid = numeric.dropna()
        if valid.empty:
            col_types[col] = "analog"
            continue

        unique_values = np.unique(valid.to_numpy(dtype=float))
        integer_like = np.allclose(
            unique_values,
            np.round(unique_values),
            rtol=0.0,
            atol=1e-8,
        )
        limited_states = len(unique_values) <= 20
        bounded_state_range = (
            float(unique_values.min()) >= 0.0
            and float(unique_values.max()) <= 20.0
        )

        col_types[col] = (
            "digital"
            if integer_like and limited_states and bounded_state_range
            else "analog"
        )

    return col_types, col_arrays

