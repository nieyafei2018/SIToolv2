# -*- coding: utf-8 -*-
"""Filesystem validators for SIToolv2 raw reference data.

Each public ``validate_*`` function checks that a dataset's raw data directory
exists and contains the expected sub-directory / file layout.  It returns a
:class:`ValidationResult` (a named tuple) with three fields:

- ``status``  : one of ``'OK'``, ``'MISSING'``, ``'STRUCTURE_ERROR'``
- ``dataset`` : human-readable dataset label
- ``message`` : single-line description of the result

The dict :data:`VALIDATOR_REGISTRY` maps each ``prep_*`` method name to its
validator callable.  Pass the same ``kwargs`` as the corresponding prep task
and the function returns a :class:`ValidationResult`.
"""

from __future__ import annotations

import glob as _glob
import os
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ValidationResult(NamedTuple):
    """Outcome of a single dataset validation check."""

    status: str   # 'OK', 'MISSING', or 'STRUCTURE_ERROR'
    dataset: str  # human-readable label
    message: str  # explanation (path, what was missing, etc.)

    @property
    def ok(self) -> bool:
        return self.status == "OK"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _missing(dataset: str, path: Path) -> ValidationResult:
    return ValidationResult("MISSING", dataset, f"Root directory not found: {path}")


def _structure_error(dataset: str, path: Path, detail: str) -> ValidationResult:
    return ValidationResult("STRUCTURE_ERROR", dataset, f"{path} — {detail}")


def _ok(dataset: str, path: Path) -> ValidationResult:
    return ValidationResult("OK", dataset, str(path))


def _dir_exists(path: Path) -> bool:
    return path.is_dir()


def _has_files(path: Path, pattern: str = "*") -> bool:
    """Return True if *path* contains at least one file matching *pattern*."""
    return bool(list(path.glob(pattern))[:1])


# ---------------------------------------------------------------------------
# Per-dataset validators
# ---------------------------------------------------------------------------

def validate_NSIDC_SIconc(data_path: str, **_) -> ValidationResult:
    """Validate NSIDC CDR v5 sea-ice concentration (G02202_V5).

    Expected layout::

        <data_path>/
            north/
                aggregate/
                    *.nc
            south/
                aggregate/
                    *.nc
    """
    label = "NSIDC CDR v5 (SIconc)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    missing_parts: List[str] = []
    for hemi in ("north", "south"):
        agg = root / hemi / "aggregate"
        if not _dir_exists(agg):
            missing_parts.append(f"{hemi}/aggregate/")
        elif not _has_files(agg, "*.nc"):
            missing_parts.append(f"{hemi}/aggregate/ (no *.nc files)")

    if missing_parts:
        return _structure_error(label, root, "missing: " + ", ".join(missing_parts))
    return _ok(label, root)


def validate_OSI_SIconc(data_path: str, **_) -> ValidationResult:
    """Validate OSI-450 v3p1 sea-ice concentration.

    Expected layout::

        <data_path>/
            <YYYY>/
                <MM>/
                    *nh* or *sh* files
    """
    label = "OSI-450 v3p1 (SIconc)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    # At least one year sub-directory
    year_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
    if not year_dirs:
        return _structure_error(label, root, "no YYYY/ subdirectories found")

    # Sample: find at least one month dir with data files
    sample_year = sorted(year_dirs)[0]
    month_dirs = [d for d in sample_year.iterdir() if d.is_dir()]
    if not month_dirs:
        return _structure_error(label, root, f"no month subdirectories in {sample_year.name}/")

    sample_month = sorted(month_dirs)[0]
    if not _has_files(sample_month, "*.nc"):
        return _structure_error(label, root, f"no *.nc files in {sample_year.name}/{sample_month.name}/")

    return _ok(label, root)


def validate_NSIDC_SIdrift(data_path: str, **_) -> ValidationResult:
    """Validate NSIDC Polar Pathfinder sea-ice motion (drift).

    Expected layout::

        <data_path>/
            north/
                daily/
                    *.nc
            south/
                daily/
                    *.nc
    """
    label = "NSIDC Polar Pathfinder (SIdrift)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    missing_parts: List[str] = []
    for hemi in ("north", "south"):
        daily = root / hemi / "daily"
        if not _dir_exists(daily):
            missing_parts.append(f"{hemi}/daily/")
        elif not _has_files(daily, "*.nc"):
            missing_parts.append(f"{hemi}/daily/ (no *.nc files)")

    if missing_parts:
        return _structure_error(label, root, "missing: " + ", ".join(missing_parts))
    return _ok(label, root)


def validate_OSI_SIdrift(data_path: str, **_) -> ValidationResult:
    """Validate OSI-455 sea-ice drift.

    Expected layout::

        <data_path>/
            v1/
                merged/
                    <YYYY>/
    """
    label = "OSI-455 v1 (SIdrift)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    merged = root / "v1" / "merged"
    if not _dir_exists(merged):
        return _structure_error(label, root, "missing v1/merged/")

    year_dirs = [d for d in merged.iterdir() if d.is_dir() and d.name.isdigit()]
    if not year_dirs:
        return _structure_error(label, root, "v1/merged/ contains no YYYY/ subdirectories")

    return _ok(label, root)


def validate_MPMR_SNdepth(data_path: str, **_) -> ValidationResult:
    """Validate MPMR passive microwave snow depth (Antarctic).

    Expected layout::

        <data_path>/
            MPMR/
                Antarctic_SnowDepth/
                    <YYYY>/
                        *_SD.tif
                Antarctic_SnowDepth_uncertainty/
    """
    label = "MPMR (SNdepth)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    sd_dir = root / "MPMR" / "Antarctic_SnowDepth"
    if not _dir_exists(sd_dir):
        return _structure_error(label, root, "missing MPMR/Antarctic_SnowDepth/")

    year_dirs = [d for d in sd_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not year_dirs:
        return _structure_error(label, root, "MPMR/Antarctic_SnowDepth/ has no YYYY/ subdirs")

    unc_dir = root / "MPMR" / "Antarctic_SnowDepth_uncertainty"
    if not _dir_exists(unc_dir):
        return _structure_error(label, root, "missing MPMR/Antarctic_SnowDepth_uncertainty/")

    return _ok(label, root)


def validate_CTOHv2_SNdepth(data_path: str, **_) -> ValidationResult:
    """Validate CTOH v2 snow depth (Arctic + Antarctic).

    Expected layout::

        <data_path>/
            v2.0/
                Arctic/
                    SIT_nh_*.nc
                Antarctic/
                    SIT_sh_*.nc
    """
    label = "CTOH v2.0 (SNdepth)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    v2 = root / "v2.0"
    if not _dir_exists(v2):
        return _structure_error(label, root, "missing v2.0/")

    missing_parts: List[str] = []
    for hemi_dir, pattern in (("Arctic", "SIT_nh_*.nc"), ("Antarctic", "SIT_sh_*.nc")):
        hd = v2 / hemi_dir
        if not _dir_exists(hd):
            missing_parts.append(f"v2.0/{hemi_dir}/")
        elif not _has_files(hd, pattern):
            missing_parts.append(f"v2.0/{hemi_dir}/ (no {pattern})")

    if missing_parts:
        return _structure_error(label, root, "missing: " + ", ".join(missing_parts))
    return _ok(label, root)


def validate_SnowLG_SNdepth(data_path: str, **_) -> ValidationResult:
    """Validate SnowModel-LG snow depth (Arctic, NSIDC-0758).

    Expected layout::

        <data_path>/
            SM_snod_ERA5_*.nc
            SM_snod_MERRA2_*.nc
    """
    label = "SnowModel-LG NSIDC-0758 (SNdepth)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    missing_files: List[str] = []
    for pattern in ("SM_snod_ERA5_*.nc", "SM_snod_MERRA2_*.nc"):
        if not _has_files(root, pattern):
            missing_files.append(pattern)

    if missing_files:
        return _structure_error(label, root, "missing files: " + ", ".join(missing_files))
    return _ok(label, root)


def validate_CSBD_SIthick(data_path: str, **_) -> ValidationResult:
    """Validate CSBD CryoSat-2 sea-ice thickness (Bristol).

    Expected layout::

        <data_path>/
            Sea_Ice_Thickness/
                ubristol_cryosat2_seaicethickness_nh_80km_v1p7.nc  (or similar)
    """
    label = "CSBD CryoSat-2 (SIthick)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    sit_dir = root / "Sea_Ice_Thickness"
    if not _dir_exists(sit_dir):
        return _structure_error(label, root, "missing Sea_Ice_Thickness/")

    if not _has_files(sit_dir, "*.nc") and not _has_files(sit_dir, "*.tif"):
        return _structure_error(label, root, "Sea_Ice_Thickness/ contains no data files")

    return _ok(label, root)


def validate_CTOHv2_SIthick(data_path: str, **_) -> ValidationResult:
    """Validate CTOH v2 sea-ice thickness — same directory as CTOHv2 snow depth.

    Expected layout::

        <data_path>/
            v2.0/
                Arctic/
                    SIT_nh_*.nc
                Antarctic/
                    SIT_sh_*.nc
    """
    label = "CTOH v2.0 (SIthick)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    v2 = root / "v2.0"
    if not _dir_exists(v2):
        return _structure_error(label, root, "missing v2.0/")

    missing_parts: List[str] = []
    for hemi_dir in ("Arctic", "Antarctic"):
        hd = v2 / hemi_dir
        if not _dir_exists(hd):
            missing_parts.append(f"v2.0/{hemi_dir}/")

    if missing_parts:
        return _structure_error(label, root, "missing: " + ", ".join(missing_parts))
    return _ok(label, root)


def validate_GIOMAS_SIthick(data_path: str, **_) -> ValidationResult:
    """Validate GIOMAS sea-ice thickness.

    Expected layout::

        <data_path>/
            Monthly/
                heff.H*
            grids/
                grid.dat
    """
    label = "GIOMAS (SIthick)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    monthly = root / "Monthly"
    if not _dir_exists(monthly):
        return _structure_error(label, root, "missing Monthly/")

    if not _has_files(monthly, "heff.H*"):
        return _structure_error(label, root, "Monthly/ contains no heff.H* binary files")

    grid_file = root / "grids" / "grid.dat"
    if not grid_file.is_file():
        return _structure_error(label, root, "missing grids/grid.dat")

    return _ok(label, root)


def validate_TOPAZ_SIthick(data_path: str, **_) -> ValidationResult:
    """Validate TOPAZ4 sea-ice thickness.

    Expected layout::

        <data_path>/
            cmems_mod_arc_phy_my_topaz4_P1M_*.nc
    """
    label = "TOPAZ4 (SIthick)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    if not _has_files(root, "cmems_mod_arc_phy_my_topaz4_P1M_*.nc"):
        return _structure_error(label, root, "no cmems_mod_arc_phy_my_topaz4_P1M_*.nc files found")

    return _ok(label, root)


def validate_PIOMAS_SIthick(data_path: str, **_) -> ValidationResult:
    """Validate PIOMAS sea-ice thickness.

    Expected layout::

        <data_path>/
            Monthly/
                heff.H*
            grids/
                longrid.dat
                latgrid.dat
    """
    label = "PIOMAS (SIthick)"
    root = Path(data_path)
    if not _dir_exists(root):
        return _missing(label, root)

    monthly = root / "Monthly"
    if not _dir_exists(monthly):
        return _structure_error(label, root, "missing Monthly/")

    if not _has_files(monthly, "heff.H*"):
        return _structure_error(label, root, "Monthly/ contains no heff.H* binary files")

    grids = root / "grids"
    missing_grids: List[str] = []
    for gf in ("longrid.dat", "latgrid.dat"):
        if not (grids / gf).is_file():
            missing_grids.append(f"grids/{gf}")

    if missing_grids:
        return _structure_error(label, root, "missing: " + ", ".join(missing_grids))

    return _ok(label, root)


# ---------------------------------------------------------------------------
# Registry: maps prep method name → validator callable
# ---------------------------------------------------------------------------

VALIDATOR_REGISTRY: Dict[str, Callable[..., ValidationResult]] = {
    "prep_NSIDC_SIconc":       validate_NSIDC_SIconc,
    "prep_OSI_SIconc":         validate_OSI_SIconc,
    "prep_NSIDC_SIdrift":      validate_NSIDC_SIdrift,
    "prep_OSI_SIdrift":        validate_OSI_SIdrift,
    "prep_MPMR_SNdepth":       validate_MPMR_SNdepth,
    "prep_CTOHv2_SNdepth_cont": validate_CTOHv2_SNdepth,
    "prep_SnowLG_SNdepth":     validate_SnowLG_SNdepth,
    "prep_CSBD_SIthick":       validate_CSBD_SIthick,
    "prep_CTOHv2_SIthick_cont": validate_CTOHv2_SIthick,
    "prep_GIOMAS_SIthick":     validate_GIOMAS_SIthick,
    "prep_TOPAZ_SIthick":      validate_TOPAZ_SIthick,
    "prep_PIOMAS_SIthick":     validate_PIOMAS_SIthick,
}


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def validate_tasks(tasks: List[dict]) -> List[ValidationResult]:
    """Validate filesystem paths for a list of prep tasks.

    Each task dict must have a ``method`` key and a ``kwargs`` dict.  Tasks
    whose method has no registered validator are silently skipped (OK assumed).

    Returns one :class:`ValidationResult` per *unique* (method, data_path)
    combination — duplicate calls with different hemisphere args produce a
    single shared result.

    Parameters
    ----------
    tasks:
        List of task dicts as loaded from the YAML ``prep_tasks`` section.

    Returns
    -------
    list of ValidationResult
    """
    seen: Dict[tuple, ValidationResult] = {}
    results: List[ValidationResult] = []

    for task in tasks:
        method = str(task.get("method") or "").strip()
        kwargs = task.get("kwargs") or {}
        data_path = kwargs.get("data_path", "")
        key = (method, data_path)

        if key in seen:
            continue

        validator = VALIDATOR_REGISTRY.get(method)
        if validator is None:
            # No validator registered; assume OK
            continue

        result = validator(**kwargs)
        seen[key] = result
        results.append(result)

    return results


def print_validation_report(results: List[ValidationResult]) -> None:
    """Write a formatted validation summary to stderr.

    Writing to stderr keeps the report in the same output stream as log
    messages, so it appears in the correct chronological position when
    stdout and stderr are merged (e.g. ``python main.py --prep-ref 2>&1``).
    """
    import sys

    def _print(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    if not results:
        _print("No datasets to validate.")
        return

    col_w = max(len(r.dataset) for r in results) + 2
    _print("\nValidating raw reference data paths...")
    _print("-" * (col_w + 80))

    for r in results:
        tag = {
            "OK":             "  OK  ",
            "MISSING":        " MISS ",
            "STRUCTURE_ERROR": " ERR  ",
        }.get(r.status, r.status)
        label = r.dataset.ljust(col_w)
        _print(f"  [{tag}]  {label}  {r.message}")

    n_ok = sum(1 for r in results if r.status == "OK")
    n_miss = sum(1 for r in results if r.status == "MISSING")
    n_err = sum(1 for r in results if r.status == "STRUCTURE_ERROR")
    _print("-" * (col_w + 80))
    _print(f"  Results: {n_ok} OK, {n_miss} MISSING, {n_err} STRUCTURE_ERROR\n")
