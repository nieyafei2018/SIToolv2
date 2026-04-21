#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================
SIToolv2 — Sea Ice Evaluation Tool
====================================
Unified entry point for all sea ice evaluation modules.

Usage
-----
    python main.py <case_name> [--modules MODULE ...] [--log-level LEVEL] [--recalculate] [--jobs N] [--keep-staging]

Examples
--------
    python main.py highres
    python main.py highres --modules SIconc SIdrift
    python main.py highres --modules SIconc --log-level DEBUG
    python main.py highres --jobs 8
"""

import argparse
import datetime
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait
)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr
from cdo import Cdo
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from tqdm import tqdm

from scripts import plot_figs as pf
from scripts import preprocess as PP
from scripts.pipeline import recipe_reader as RR
from scripts import report
from scripts.utils import runtime_efficiency as rte
from scripts import sea_ice_metrics as SIM
from scripts import utils
from scripts.config import DAYS_PER_MONTH, setup_logging

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# Matplotlib state is not thread-safe. All plotting calls are serialized
# through one re-entrant lock so hemisphere-level worker threads can still
# run preprocessing and metric calculations concurrently without corrupting
# figure rendering internals.
_PLOT_LOCK = threading.RLock()

ALL_MODULES = [
    'SIconc', 'SIdrift', 'SIthick', 'SNdepth', 'SICB', 'SItrans',
    'SIMbudget', 'SNMbudget',
]

# Maps each module name to its enable/disable flag in the recipe common section.
# Used to skip modules whose flag is explicitly set to False in the recipe.
_MODULE_FLAG = {
    'SIconc':  'eval_sic',
    'SIdrift': 'eval_sid',
    'SIthick': 'eval_sit',
    'SNdepth': 'eval_snd',
    'SICB':    'eval_sicb',
    'SItrans': 'eval_sitrans',
    'SIMbudget': 'eval_simb',
    'SNMbudget': 'eval_snmb',
}

CACHE_SCHEMA_VERSION = 'metrics-cache-v4'
UNIFIED_CACHE_SCHEMA_VERSION = 'metrics-cache-unified-v1'
_CACHE_ENGINES = ('netcdf4', 'h5netcdf', None)
_CACHE_META_GROUP = '/__metadata__'
_NC_IO_RETRIES = 4
_NC_IO_BASE_DELAY_SEC = 0.25
_DEFAULT_MAX_CPU_FRACTION = 0.60
_GROUP_MEAN_LABEL_PREFIX = 'GroupMean['
_GROUP_MEAN_LABEL_SUFFIX = ']'


def _is_retryable_nc_error(exc: Exception) -> bool:
    """Return True when a NetCDF read/write failure is likely transient/lock-related."""
    text = str(exc).lower()
    if isinstance(exc, (PermissionError, BlockingIOError)):
        return True
    retry_tokens = (
        'permission denied',
        'access is denied',
        'resource temporarily unavailable',
        'file is already open',
        'unable to open file',
        'hdf error',
        'locking',
    )
    return any(tok in text for tok in retry_tokens)


def _safe_unlink(path: Path) -> None:
    """Delete a file with bounded retries for transient lock contention."""
    for attempt in range(_NC_IO_RETRIES):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except Exception as exc:
            if attempt >= _NC_IO_RETRIES - 1 or not _is_retryable_nc_error(exc):
                raise
            time.sleep(_NC_IO_BASE_DELAY_SEC * (attempt + 1))


def _safe_replace(src: Path, dst: Path) -> None:
    """Atomically replace one file with retries for transient lock contention."""
    for attempt in range(_NC_IO_RETRIES):
        try:
            os.replace(src, dst)
            return
        except Exception as exc:
            if attempt >= _NC_IO_RETRIES - 1 or not _is_retryable_nc_error(exc):
                raise
            time.sleep(_NC_IO_BASE_DELAY_SEC * (attempt + 1))


def _ensure_accessible_directory(path: Path) -> Tuple[bool, Optional[str]]:
    """Create a directory if needed and verify it is traversable/writable."""
    try:
        if path.exists() and not path.is_dir():
            return False, f'Path exists but is not a directory: {path}'
        path.mkdir(parents=True, exist_ok=True)
        # Probe directory traversal to catch filesystem ghost entries on some drvfs mounts.
        os.listdir(path)
        probe = path / '.sitool_probe'
        if not probe.exists():
            probe.mkdir()
            probe.rmdir()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _resolve_case_run_dir(case_name: str, log_warning: bool = True) -> Path:
    """Resolve a writable case directory, with fallback for broken/ghost paths."""
    cases_root = Path('cases')
    preferred = cases_root / case_name
    ok, reason = _ensure_accessible_directory(preferred)
    if ok:
        return preferred

    fallback = cases_root / '_runs' / case_name
    ok_fb, reason_fb = _ensure_accessible_directory(fallback)
    if not ok_fb:
        raise OSError(
            f"Unable to initialize case directory. Preferred='{preferred}' ({reason}); "
            f"fallback='{fallback}' ({reason_fb})."
        )

    if log_warning:
        logger.warning(
            "Case directory '%s' is not usable (%s). Using fallback run directory: '%s'.",
            preferred, reason, fallback
        )
    return fallback


def _configure_worker_logging(case_name: str, hemisphere: str, module: str) -> Path:
    """Configure one isolated log file for a worker process.

    Process-based hemisphere workers should not write into the main run log
    file, because independent file offsets can corrupt the stream under
    concurrent writes. Each worker writes to its own append-only log file.
    """
    level_name = os.environ.get('SITOOL_LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    case_dir = _resolve_case_run_dir(case_name, log_warning=False)
    worker_dir = case_dir / 'logs' / 'workers'
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_log = worker_dir / f'{module}_{hemisphere}_pid{os.getpid()}.log'

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    root.handlers.clear()
    root.setLevel(level)

    fh = logging.FileHandler(worker_log, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    logging.captureWarnings(True)
    return worker_log


def _resolve_worker_count(requested_jobs: int, max_cpu_fraction: float = _DEFAULT_MAX_CPU_FRACTION) -> int:
    """Resolve an effective worker count and enforce the interactive >60% confirmation rule.

    The function intentionally performs the confirmation check at runtime because
    the final worker count depends on the current machine capacity.
    """
    available = max(1, int(os.cpu_count() or 1))
    requested = max(1, int(requested_jobs))
    if requested > available:
        logger.warning(
            "Requested jobs (%d) exceed available CPU threads (%d). Capping to %d.",
            requested, available, available,
        )
        requested = available

    limit = max(1, int(math.floor(available * float(max_cpu_fraction))))
    if requested <= limit:
        return requested

    prompt = (
        f"Requested jobs ({requested}) exceed {int(max_cpu_fraction * 100)}% of available threads "
        f"({available}; threshold={limit}). Continue? [y/N]: "
    )

    # If the process is not attached to a TTY we cannot ask for confirmation.
    # In that case we fail-safe by reducing to the threshold.
    if not getattr(sys.stdin, 'isatty', lambda: False)():
        logger.warning(
            "No interactive TTY detected. Reducing jobs from %d to safe threshold %d.",
            requested, limit,
        )
        return limit

    answer = input(prompt).strip().lower()
    if answer in {'y', 'yes'}:
        return requested

    logger.warning(
        "User declined high-concurrency run. Reducing jobs from %d to %d.",
        requested, limit,
    )
    return limit


def _resolve_default_prep_parallel_cap(requested_jobs: int, module: str, case_name: str = '') -> int:
    """Resolve default Python-side preprocessing worker cap for one module task.

    Safety-first policy:
      - keep low default fan-out to reduce NetCDF/HDF lock/contention risks;
      - allow slightly higher cap for long-tail preprocess-heavy modules;
      - keep explicit environment override support.
    """
    safe_jobs = max(1, int(requested_jobs))
    module_name = str(module or '').strip()
    case_name_norm = str(case_name or '').strip().lower()
    aggressive_parallel = str(os.environ.get('SITOOL_AGGRESSIVE_PARALLEL', '')).strip().lower() in {
        '1', 'true', 'yes', 'on',
    }
    highres_like_case = case_name_norm.startswith('highres')

    override_raw = str(os.environ.get('SITOOL_PREP_DEFAULT_MAX_PARALLEL', '')).strip()
    if override_raw:
        try:
            override_val = max(1, int(override_raw))
            return max(1, min(safe_jobs, override_val))
        except Exception:
            logger.warning(
                "Ignoring invalid SITOOL_PREP_DEFAULT_MAX_PARALLEL='%s'.",
                override_raw,
            )

    # Aggressive mode (enabled for highres by default, or globally via
    # SITOOL_AGGRESSIVE_PARALLEL=1):
    # allow broader Python-side fan-out for preprocess-heavy modules so long-tail
    # tasks can consume freed CPU budget instead of idling.
    if aggressive_parallel or highres_like_case:
        if module_name in {'SIMbudget', 'SNMbudget'}:
            cap = max(4, min(12, int(math.ceil(float(safe_jobs) * 0.7))))
        elif module_name in {'SICB'}:
            cap = max(8, min(30, int(math.ceil(float(safe_jobs) * 0.9))))
        elif module_name in {'SIdrift', 'SItrans'}:
            cap = max(4, min(20, int(math.ceil(float(safe_jobs) * 0.8))))
        else:
            cap = max(3, min(12, int(math.ceil(float(safe_jobs) * 0.7))))
        return max(1, min(safe_jobs, cap))

    # Conservative defaults for stability (non-highres/non-aggressive path):
    # - <=2 module jobs: keep serial preprocessing;
    # - otherwise allow light Python fan-out;
    # - SIdrift/SICB can benefit from one extra worker when thread budget is high.
    if safe_jobs <= 2:
        cap = 1
    else:
        cap = 2
    if module_name in {'SIdrift', 'SICB'} and safe_jobs >= 16:
        cap = 3

    return max(1, min(safe_jobs, cap))


def _install_thread_safe_plot_wrappers() -> None:
    """Wrap plot_figs.plot_* entry points with a global plotting lock.

    The lock is re-entrant so nested plot helpers can call each other safely.
    """
    for name in dir(pf):
        if not name.startswith('plot_'):
            continue
        func = getattr(pf, name, None)
        if not callable(func):
            continue
        if getattr(func, '_sitool_locked_plot', False):
            continue

        def _make_locked(f):
            def _locked(*args, **kwargs):
                with _PLOT_LOCK:
                    return f(*args, **kwargs)
            _locked._sitool_locked_plot = True
            _locked.__name__ = getattr(f, '__name__', name)
            _locked.__doc__ = getattr(f, '__doc__', None)
            return _locked

        setattr(pf, name, _make_locked(func))


def _model_count_for_module(module: str, module_vars: Dict[str, Any]) -> int:
    """Return the declared model count for one module from the recipe."""
    if module == 'SIdrift':
        return len(module_vars.get('model_file_u') or module_vars.get('model_file') or [])
    if module == 'SICB':
        return len(module_vars.get('model_file_sic') or module_vars.get('model_file') or [])
    if module == 'SIMbudget':
        return len(
            module_vars.get('model_file_sidmassdyn')
            or module_vars.get('model_file_sidmasssi')
            or module_vars.get('model_file')
            or []
        )
    if module == 'SNMbudget':
        return len(
            module_vars.get('model_file_sndmassdyn')
            or module_vars.get('model_file_sndmasssi')
            or module_vars.get('model_file')
            or []
        )
    return len(module_vars.get('model_file') or [])


def _obs_count_for_module(module: str, module_vars: Dict[str, Any], hemisphere: str) -> int:
    """Return the observation count for one module/hemisphere from the recipe."""
    hms = str(hemisphere or '').lower()
    if module in {'SIMbudget', 'SNMbudget'}:
        return 0
    if module == 'SICB':
        obs_sic = len(module_vars.get(f'ref_{hms}_sic') or [])
        obs_sid = len(module_vars.get(f'ref_{hms}_sidrift') or [])
        return max(obs_sic, obs_sid)
    return len(module_vars.get(f'ref_{hms}') or [])


def _estimate_per_hemisphere_parallel_cap(
    recipe: RR.RecipeReader,
    modules_to_run: List[str],
    eval_hms: List[str],
) -> int:
    """Estimate safe per-hemisphere worker cap from recipe dataset counts.

    Heuristic:
      per-hemisphere cap = max over modules/hemispheres of (model_count + min(obs_count, 2))
    where "2" represents at most two observation products (obs1/obs2).
    """
    cap = 1
    for module in modules_to_run:
        module_vars = recipe.variables.get(module)
        if not isinstance(module_vars, dict):
            continue
        model_count = _model_count_for_module(module, module_vars)
        for hms in eval_hms:
            obs_count = _obs_count_for_module(module, module_vars, hms)
            cap = max(cap, int(model_count) + min(int(obs_count), 2))
    return max(1, cap)


def _resolve_safe_job_plan(
    requested_jobs: int,
    recipe: RR.RecipeReader,
    modules_to_run: List[str],
    eval_hms: List[str],
) -> Tuple[int, List[int], int]:
    """Resolve safe job options and pick one concrete job count.

    Returns:
        selected_jobs: Effective safe worker count used by this run.
        safe_options: Recommended safe choices for -j.
        per_hemi_cap: Estimated maximum safe workers per hemisphere.
    """
    per_hemi_cap = _estimate_per_hemisphere_parallel_cap(recipe, modules_to_run, eval_hms)
    hms_workers = 2 if len(eval_hms) >= 2 else 1
    requested = max(1, int(requested_jobs))

    # Legacy cap (= hemisphere-count * estimated per-hemisphere dataset fan-out)
    # is robust but often too conservative for modern multi-core runs.
    # Default behavior now allows using the user-requested thread budget, while
    # retaining an opt-in strict mode for conservative deployments.
    legacy_cap = max(1, hms_workers * per_hemi_cap)
    strict_mode = str(os.environ.get('SITOOL_STRICT_SAFE_JOBS', '0')).strip().lower() in {
        '1', 'true', 'yes', 'on',
    }
    safe_total_cap = legacy_cap if strict_mode else max(legacy_cap, requested)

    if hms_workers == 2:
        max_even = safe_total_cap - (safe_total_cap % 2)
        safe_options = [1]
        if max_even >= 2:
            safe_options.extend(list(range(2, max_even + 1, 2)))
    else:
        safe_options = list(range(1, safe_total_cap + 1))

    candidate = min(requested, safe_total_cap)
    if candidate in safe_options:
        selected = candidate
    else:
        lower = [opt for opt in safe_options if opt <= candidate]
        selected = lower[-1] if lower else safe_options[0]

    return selected, safe_options, per_hemi_cap



# ---------------------------------------------------------------------------
# Cache/staging helpers (imported from scripts.pipeline.cache_utils)
# ---------------------------------------------------------------------------
from scripts.pipeline import cache_utils as _cache_utils

_parallel_map_ordered = _cache_utils._parallel_map_ordered
_case_dir_from_output_dir = _cache_utils._case_dir_from_output_dir
_get_stage_dir = _cache_utils._get_stage_dir
_save_pickle_atomic = _cache_utils._save_pickle_atomic
_load_pickle = _cache_utils._load_pickle
_cleanup_stage_dir = _cache_utils._cleanup_stage_dir
_get_metrics_cache_file = _cache_utils._get_metrics_cache_file
_open_dataset_with_engines = _cache_utils._open_dataset_with_engines
_write_dataset_with_engines = _cache_utils._write_dataset_with_engines
_read_cache_root_attrs = _cache_utils._read_cache_root_attrs
_update_cache_root_attrs = _cache_utils._update_cache_root_attrs
_sanitize_group_name = _cache_utils._sanitize_group_name
_derive_label_from_ref_name = _cache_utils._derive_label_from_ref_name
_get_reference_labels = _cache_utils._get_reference_labels
_derive_label_from_model_pattern = _cache_utils._derive_label_from_model_pattern
_get_recipe_model_labels = _cache_utils._get_recipe_model_labels
_unique_entity_name = _cache_utils._unique_entity_name
_infer_units = _cache_utils._infer_units
_build_units_map = _cache_utils._build_units_map
_load_grid_coords = _cache_utils._load_grid_coords
_write_metric_record = _cache_utils._write_metric_record
_read_metric_record = _cache_utils._read_metric_record
_write_module_metadata = _cache_utils._write_module_metadata
_read_module_metadata = _cache_utils._read_module_metadata
_save_module_cache = _cache_utils._save_module_cache
_load_module_cache = _cache_utils._load_module_cache
_write_json_payload_group = _cache_utils._write_json_payload_group
_build_unified_metrics_cache = _cache_utils._build_unified_metrics_cache



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_module_header(module: str) -> None:
    """Print a separator header marking the start of a module evaluation, for easy log navigation."""
    logger.info("=" * 40)
    logger.info("Starting evaluation of module: %s", module)
    logger.info("=" * 40)


def _get_year_range(module_vars: dict) -> Tuple[int, int]:
    """Extract and validate (start_year, end_year) from module config."""
    year_range = module_vars.get('year_range')
    if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
        raise ValueError("Each module requires 'year_range: [start_year, end_year]'.")

    try:
        year_sta = int(year_range[0])
        year_end = int(year_range[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("'year_range' must contain integer years.") from exc

    if year_sta > year_end:
        raise ValueError(f"Invalid year_range: start year {year_sta} is after end year {year_end}.")

    return year_sta, year_end


def _get_metric(module_vars: dict) -> str:
    """Read scalar metric name from module config with backward-compatible fallback."""
    metric = module_vars.get('stats_metric', module_vars.get('metric', 'RMSE'))
    return str(metric)


def _validate_sidrift_direction_config(module_vars: dict, n_models: int) -> List[str]:
    """Validate and normalize SIdrift per-model direction tags.

    Returns a list sized to ``n_models``. Missing/blank entries are treated as
    ``auto`` so preprocessing can infer vector frame from metadata.
    """
    model_direction = module_vars.get('model_direction')
    if model_direction is None:
        model_direction = []
    if not isinstance(model_direction, list):
        raise ValueError("SIdrift 'model_direction' must be a list.")
    if len(model_direction) > n_models:
        raise ValueError(
            f"SIdrift 'model_direction' length ({len(model_direction)}) exceeds number of models ({n_models})."
        )

    allowed = {'lonlat', 'xy', 'other', 'auto'}
    normalized: List[str] = []
    for idx, value in enumerate(model_direction):
        raw_value = '' if value is None else str(value).strip().lower()
        if raw_value in {'', 'none', 'null'}:
            raw_value = 'auto'
        if raw_value not in allowed:
            raise ValueError(
                f"Invalid SIdrift model_direction[{idx}]='{value}'. "
                "Allowed values: lonlat, xy, other, auto (or blank for auto)."
            )
        normalized.append(raw_value)

    if len(normalized) < n_models:
        logger.warning(
            "SIdrift model_direction provides %d entries for %d models; "
            "defaulting missing entries to 'auto'.",
            len(normalized), n_models,
        )
        normalized.extend(['auto'] * (n_models - len(normalized)))

    return normalized


def _is_group_mean_label(label: Any) -> bool:
    """Return whether one dataset label is a synthetic group-mean label."""
    text = str(label if label is not None else '').strip()
    lower = text.lower()
    return (
        lower.startswith('groupmean[')
        or lower.startswith('groupmean(')
        or lower.startswith('group mean[')
        or lower.startswith('group mean(')
        or lower.startswith('gm[')
    )


def _format_group_mean_label(group_name: Any) -> str:
    """Build one stable group-mean display label."""
    raw = str(group_name if group_name is not None else '').strip()
    token = raw if raw else 'group'
    return f'{_GROUP_MEAN_LABEL_PREFIX}{token}{_GROUP_MEAN_LABEL_SUFFIX}'


def _normalize_group_token(value: Any) -> str:
    """Normalize one group member token for robust matching."""
    txt = str(value if value is not None else '').strip().lower()
    return re.sub(r'[^a-z0-9]+', '', txt)


def _resolve_group_mean_specs(
    module: str,
    module_vars: Dict[str, Any],
    common_config: Optional[Dict[str, Any]],
    model_labels: List[str],
) -> List[Dict[str, Any]]:
    """Resolve metric-level group-mean definitions for one module.

    Config keys (module-level has higher priority):
      - ``variables.<Module>.model_mean_groups``
      - ``variables.common.model_mean_groups``

    Format:
      ``{group_name: [member_label_or_index, ...]}``
    """
    cfg = None
    if isinstance(module_vars, dict):
        cfg = module_vars.get('model_mean_groups')
    if cfg is None and isinstance(common_config, dict):
        cfg = common_config.get('model_mean_groups')
    if cfg is None:
        return []
    if not isinstance(cfg, dict):
        logger.warning(
            "Ignoring invalid model_mean_groups for %s: expected mapping, got %s.",
            module, type(cfg).__name__,
        )
        return []

    if not model_labels:
        return []

    norm_to_indices: Dict[str, List[int]] = {}
    for idx, name in enumerate(model_labels):
        key = _normalize_group_token(name)
        norm_to_indices.setdefault(key, []).append(idx)

    used_group_labels: set = set()
    out: List[Dict[str, Any]] = []
    for raw_group_name, raw_members in cfg.items():
        group_name = str(raw_group_name if raw_group_name is not None else '').strip()
        if not group_name:
            continue
        if isinstance(raw_members, (list, tuple, set)):
            members = list(raw_members)
        else:
            members = [raw_members]

        member_indices: List[int] = []
        for item in members:
            idx_match: Optional[int] = None
            if isinstance(item, (int, np.integer)):
                idx_try = int(item) - 1
                if 0 <= idx_try < len(model_labels):
                    idx_match = idx_try
            else:
                token = str(item if item is not None else '').strip()
                if token.isdigit():
                    idx_try = int(token) - 1
                    if 0 <= idx_try < len(model_labels):
                        idx_match = idx_try
                if idx_match is None:
                    key = _normalize_group_token(token)
                    hit = norm_to_indices.get(key, [])
                    if hit:
                        idx_match = int(hit[0])

            if idx_match is None:
                logger.warning(
                    "Group '%s' in %s references unknown member '%s'; skipping this member.",
                    group_name, module, item,
                )
                continue
            if idx_match not in member_indices:
                member_indices.append(idx_match)

        if not member_indices:
            logger.warning(
                "Skipping empty group '%s' in %s (no valid members).",
                group_name, module,
            )
            continue

        group_label = _format_group_mean_label(group_name)
        if group_label in used_group_labels:
            logger.warning(
                "Duplicate group label '%s' in %s; keeping first definition only.",
                group_label, module,
            )
            continue
        used_group_labels.add(group_label)
        out.append({
            'name': group_name,
            'label': group_label,
            'member_indices': member_indices,
        })
    return out


def _nanmean_payload(values: List[Any]) -> Any:
    """Compute NaN-safe mean for nested metric payloads (numeric fields only)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None

    if all(isinstance(v, dict) for v in vals):
        common_keys = set(vals[0].keys())
        for node in vals[1:]:
            common_keys &= set(node.keys())
        out: Dict[str, Any] = {}
        for key in sorted(common_keys):
            child = _nanmean_payload([node.get(key) for node in vals])
            if child is not None:
                out[key] = child
        return out or None

    if all(isinstance(v, (list, tuple)) for v in vals):
        try:
            arr = np.asarray(vals, dtype=float)
        except Exception:
            return None
        if arr.ndim < 2:
            return None
        mean = np.nanmean(arr, axis=0)
        if all(isinstance(v, tuple) for v in vals):
            return tuple(np.asarray(mean).tolist())
        return list(np.asarray(mean).tolist())

    try:
        arr = np.asarray(vals, dtype=float)
    except Exception:
        return None
    if arr.size <= 0:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    # Shape mismatch often produces object arrays; skip those safely.
    if arr.dtype == np.dtype('O'):
        return None
    mean = np.nanmean(arr, axis=0)
    if np.isscalar(mean) or getattr(mean, 'ndim', 0) == 0:
        out_scalar = float(mean)
        return out_scalar if np.isfinite(out_scalar) else np.nan
    return np.asarray(mean, dtype=float)


def _nanstd_payload(values: List[Any]) -> Any:
    """Compute NaN-safe std for nested metric payloads (numeric fields only)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None

    if all(isinstance(v, dict) for v in vals):
        common_keys = set(vals[0].keys())
        for node in vals[1:]:
            common_keys &= set(node.keys())
        out: Dict[str, Any] = {}
        for key in sorted(common_keys):
            child = _nanstd_payload([node.get(key) for node in vals])
            if child is not None:
                out[key] = child
        return out or None

    if all(isinstance(v, (list, tuple)) for v in vals):
        try:
            arr = np.asarray(vals, dtype=float)
        except Exception:
            return None
        if arr.ndim < 2:
            return None
        std = np.nanstd(arr, axis=0)
        if all(isinstance(v, tuple) for v in vals):
            return tuple(np.asarray(std).tolist())
        return list(np.asarray(std).tolist())

    try:
        arr = np.asarray(vals, dtype=float)
    except Exception:
        return None
    if arr.size <= 0:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    if arr.dtype == np.dtype('O'):
        return None
    std = np.nanstd(arr, axis=0)
    if np.isscalar(std) or getattr(std, 'ndim', 0) == 0:
        out_scalar = float(std)
        return out_scalar if np.isfinite(out_scalar) else np.nan
    return np.asarray(std, dtype=float)


def _recompute_ano_series_stats(payload: Any) -> Any:
    """Recompute 1-D anomaly-derived std/trend fields from ``*_ano`` series.

    Rationale
    ---------
    Group-mean payloads are first built via numeric NaN-mean across model payloads.
    Directly averaging precomputed ``*_ano_std``/``*_ano_tr`` fields is not ideal:
    these diagnostics should instead be derived from the grouped ``*_ano`` series.

    This helper walks nested dictionaries and, for each 1-D ``*_ano`` key, writes
    anomaly-derived diagnostics:
      - ``<base>_std``
      - ``<base>_tr``
      - ``<base>_tr_p``

    Notes
    -----
    - Only 1-D anomaly series are handled here to avoid unit/scale ambiguities for
      map-style trends.
    - ``<base>_tr`` is written as a scipy linregress-like object (same style as
      native metric payloads for 1-D anomaly trends).
    """
    if not isinstance(payload, dict):
        return payload

    # Recurse first so nested nodes are normalized as well.
    for child_key, child_val in list(payload.items()):
        if isinstance(child_val, dict):
            payload[child_key] = _recompute_ano_series_stats(child_val)

    for key in list(payload.keys()):
        if not (isinstance(key, str) and key.endswith('_ano')):
            continue

        std_key = f'{key}_std'
        tr_key = f'{key}_tr'
        tr_p_key = f'{key}_tr_p'

        try:
            arr = np.asarray(payload.get(key), dtype=float)
        except Exception:
            continue
        if arr.ndim != 1 or arr.size <= 0:
            continue

        valid = np.isfinite(arr)
        std_val = float(np.nanstd(arr)) if np.any(valid) else np.nan
        payload[std_key] = std_val if np.isfinite(std_val) else np.nan

        reg = None
        if int(np.sum(valid)) >= 2:
            x = np.arange(arr.size, dtype=float)
            try:
                reg = stats.linregress(x[valid], arr[valid])
            except Exception:
                reg = None

        payload[tr_key] = reg if reg is not None else np.nan
        if reg is not None and np.isfinite(getattr(reg, 'pvalue', np.nan)):
            payload[tr_p_key] = float(reg.pvalue)
        else:
            payload[tr_p_key] = np.nan

    return payload


def _build_group_mean_std_payloads(
    model_payloads: List[Any],
    model_labels: List[str],
    group_specs: List[Dict[str, Any]],
) -> Tuple[List[Any], List[Any], List[str]]:
    """Build group-only mean/std payload triplets from one model payload list."""
    base_models = list(model_payloads or [])
    if not base_models or not group_specs:
        return [], [], []

    group_mean_payloads: List[Any] = []
    group_std_payloads: List[Any] = []
    group_labels: List[str] = []

    for spec in group_specs:
        idxs = [
            int(idx) for idx in (spec.get('member_indices') or [])
            if 0 <= int(idx) < len(base_models)
        ]
        if not idxs:
            continue
        members = [base_models[idx] for idx in idxs]
        mean_payload = _nanmean_payload(members)
        if mean_payload is None:
            logger.warning(
                "Failed to build group mean payload '%s' (incompatible members).",
                spec.get('label'),
            )
            continue
        mean_payload = _recompute_ano_series_stats(mean_payload)
        std_payload = _nanstd_payload(members)
        group_mean_payloads.append(mean_payload)
        group_std_payloads.append(std_payload)
        group_labels.append(str(spec.get('label') or _format_group_mean_label(spec.get('name'))))

    return group_mean_payloads, group_std_payloads, group_labels


def _build_group_member_file_map(
    file_paths: List[str],
    model_labels: List[str],
    group_specs: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Return group-label -> existing member-file list mapping."""
    base_files = [str(p) for p in (file_paths or [])]
    out: Dict[str, List[str]] = {}
    if (not base_files) or (not group_specs):
        return out
    for spec in group_specs:
        idxs = [
            int(idx) for idx in (spec.get('member_indices') or [])
            if 0 <= int(idx) < len(base_files)
        ]
        if not idxs:
            continue
        group_label = str(spec.get('label') or _format_group_mean_label(spec.get('name')))
        members = []
        for idx in idxs:
            fpath = base_files[idx]
            if fpath and Path(fpath).exists():
                members.append(str(Path(fpath)))
        if members:
            out[group_label] = members
    return out


def _build_group_mean_payloads(
    model_payloads: List[Any],
    diff_payloads: Optional[List[Any]],
    model_labels: List[str],
    group_specs: List[Dict[str, Any]],
) -> Tuple[List[Any], Optional[List[Any]], List[str], List[str]]:
    """Prepend group-mean payloads to model/diff payload lists."""
    base_models = list(model_payloads or [])
    base_diffs = list(diff_payloads) if isinstance(diff_payloads, list) else None
    base_labels = list(model_labels or [])
    if not base_models or not group_specs:
        return base_models, base_diffs, base_labels, []

    group_model_payloads: List[Any] = []
    group_diff_payloads: List[Any] = []
    group_labels: List[str] = []

    for spec in group_specs:
        idxs = [
            int(idx) for idx in (spec.get('member_indices') or [])
            if 0 <= int(idx) < len(base_models)
        ]
        if not idxs:
            continue
        metric_payload = _nanmean_payload([base_models[idx] for idx in idxs])
        if metric_payload is None:
            logger.warning(
                "Failed to build metric-level group mean '%s' (module payload incompatible).",
                spec.get('label'),
            )
            continue
        metric_payload = _recompute_ano_series_stats(metric_payload)
        group_model_payloads.append(metric_payload)
        group_labels.append(str(spec.get('label') or _format_group_mean_label(spec.get('name'))))

        if base_diffs is not None:
            valid_diff_idxs = [idx for idx in idxs if idx < len(base_diffs)]
            if valid_diff_idxs:
                group_diff_payloads.append(_nanmean_payload([base_diffs[idx] for idx in valid_diff_idxs]))
            else:
                group_diff_payloads.append(None)

    if not group_model_payloads:
        return base_models, base_diffs, base_labels, []

    merged_models = group_model_payloads + base_models
    merged_labels = group_labels + base_labels
    if base_diffs is None:
        return merged_models, None, merged_labels, group_labels
    merged_diffs = group_diff_payloads + base_diffs
    return merged_models, merged_diffs, merged_labels, group_labels


def _compute_group_mean_netcdf(
    member_files: List[str],
    output_file: Union[str, Path],
) -> Optional[str]:
    """Build one NaN-safe group-mean NetCDF from member files.

    The function averages **numeric** data variables with identical dims/shapes.
    Non-numeric or incompatible variables are skipped safely.
    """
    valid_members = [str(Path(p)) for p in (member_files or []) if p and Path(p).exists()]
    if not valid_members:
        return None
    if len(valid_members) == 1:
        return valid_members[0]

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        out_mtime = out_path.stat().st_mtime
        src_mtime = max(Path(p).stat().st_mtime for p in valid_members)
        if out_mtime >= src_mtime:
            return str(out_path)
    except Exception:
        pass

    with xr.open_dataset(valid_members[0]) as ds_ref:
        ref_attrs = dict(ds_ref.attrs)
        var_meta: Dict[str, Dict[str, Any]] = {}
        ref_coords: Dict[str, xr.DataArray] = {}
        ref_static_vars: Dict[str, xr.DataArray] = {}
        for var_name, da in ds_ref.data_vars.items():
            try:
                if not np.issubdtype(np.asarray(da).dtype, np.number):
                    continue
            except Exception:
                continue
            dim_coords: Dict[str, np.ndarray] = {}
            for dim_name in da.dims:
                if dim_name in ds_ref.coords:
                    dim_coords[dim_name] = np.asarray(ds_ref.coords[dim_name])
            var_meta[str(var_name)] = {
                'dims': tuple(str(d) for d in da.dims),
                'shape': tuple(int(s) for s in da.shape),
                'attrs': dict(da.attrs),
                'dim_coords': dim_coords,
            }

        for coord_name, coord_da in ds_ref.coords.items():
            dims = tuple(str(d) for d in coord_da.dims)
            if not all(dim in ds_ref.sizes for dim in dims):
                continue
            ref_coords[str(coord_name)] = xr.DataArray(
                np.asarray(coord_da),
                dims=dims,
                attrs=dict(coord_da.attrs),
            )

        static_var_candidates = (
            'lon', 'lat', 'longitude', 'latitude',
            'nav_lon', 'nav_lat', 'x', 'y',
        )
        for var_name in static_var_candidates:
            if var_name in var_meta:
                continue
            if var_name not in ds_ref.variables:
                continue
            da = ds_ref[var_name]
            dims = tuple(str(d) for d in da.dims)
            if not dims:
                continue
            if not all(dim in ds_ref.sizes for dim in dims):
                continue
            ref_static_vars[var_name] = xr.DataArray(
                np.asarray(da),
                dims=dims,
                attrs=dict(da.attrs),
            )

    if not var_meta:
        logger.warning(
            "Skip group-mean NetCDF build (no numeric variables): %s",
            valid_members[0],
        )
        return None

    out_vars: Dict[str, xr.DataArray] = {}
    for var_name, meta in var_meta.items():
        dims_ref = tuple(meta.get('dims') or ())
        shape_ref = tuple(meta.get('shape') or ())
        if not dims_ref or not shape_ref:
            continue

        sum_arr: Optional[np.ndarray] = None
        count_arr: Optional[np.ndarray] = None
        valid_var = True

        for member_path in valid_members:
            try:
                with xr.open_dataset(member_path) as ds:
                    if var_name not in ds.data_vars:
                        valid_var = False
                        break
                    da = ds[var_name]
                    if tuple(str(d) for d in da.dims) != dims_ref:
                        valid_var = False
                        break
                    if tuple(int(s) for s in da.shape) != shape_ref:
                        valid_var = False
                        break
                    arr = np.asarray(da, dtype=float)
            except Exception:
                valid_var = False
                break

            if sum_arr is None or count_arr is None:
                sum_arr = np.zeros(shape_ref, dtype=float)
                count_arr = np.zeros(shape_ref, dtype=np.int32)

            finite = np.isfinite(arr)
            if np.any(finite):
                sum_arr[finite] += arr[finite]
                count_arr[finite] += 1

        if (not valid_var) or (sum_arr is None) or (count_arr is None):
            continue

        mean_arr = np.full(shape_ref, np.nan, dtype=float)
        np.divide(sum_arr, count_arr, out=mean_arr, where=(count_arr > 0))
        out_vars[var_name] = xr.DataArray(
            mean_arr,
            dims=dims_ref,
            coords=dict(meta.get('dim_coords') or {}),
            attrs=dict(meta.get('attrs') or {}),
        )

    if not out_vars:
        logger.warning(
            "Skip group-mean NetCDF build (no compatible variables): members=%s",
            ', '.join(valid_members),
        )
        return None

    out_ds = xr.Dataset(out_vars, attrs=ref_attrs)

    compatible_coords: Dict[str, xr.DataArray] = {}
    for coord_name, coord_da in ref_coords.items():
        dims = tuple(str(d) for d in coord_da.dims)
        if all(
            (dim in out_ds.sizes) and
            (int(coord_da.sizes.get(dim, -1)) == int(out_ds.sizes[dim]))
            for dim in dims
        ):
            compatible_coords[coord_name] = coord_da
    if compatible_coords:
        out_ds = out_ds.assign_coords(compatible_coords)

    for var_name, var_da in ref_static_vars.items():
        if var_name in out_ds.variables:
            continue
        dims = tuple(str(d) for d in var_da.dims)
        if all(
            (dim in out_ds.sizes) and
            (int(var_da.sizes.get(dim, -1)) == int(out_ds.sizes[dim]))
            for dim in dims
        ):
            out_ds[var_name] = var_da
    tmp_path = out_path.with_suffix(f'{out_path.suffix}.tmp_{uuid.uuid4().hex[:8]}')
    try:
        encoding = {name: {'zlib': True, 'complevel': 1} for name in out_ds.data_vars}
        out_ds.to_netcdf(str(tmp_path), format='NETCDF4', encoding=encoding)
        os.replace(str(tmp_path), str(out_path))
    finally:
        try:
            out_ds.close()
        except Exception:
            pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return str(out_path)


def _build_group_mean_file_payloads(
    file_paths: List[str],
    model_labels: List[str],
    group_specs: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    file_tag: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Prepend group-mean files to one model-file list."""
    base_files = [str(p) for p in (file_paths or [])]
    base_labels = [str(lb) for lb in (model_labels or [])]
    if not base_files or not base_labels or not group_specs:
        return base_files, base_labels, []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_norm_labels = {_normalize_group_token(lb) for lb in base_labels}

    group_files: List[str] = []
    group_labels: List[str] = []
    for spec in group_specs:
        group_label = str(spec.get('label') or '')
        if not group_label:
            continue
        if _normalize_group_token(group_label) in existing_norm_labels:
            continue

        member_files: List[str] = []
        for member_idx in (spec.get('member_indices') or []):
            try:
                idx = int(member_idx)
            except Exception:
                continue
            if 0 <= idx < len(base_files):
                fp = base_files[idx]
                if fp and Path(fp).exists():
                    member_files.append(str(fp))
        if not member_files:
            continue

        group_name = str(spec.get('name') or group_label)
        safe_group = _sanitize_group_name(group_name) or f'group_{len(group_files) + 1}'
        out_file = out_dir / f'{file_tag}_{safe_group}.nc'
        mean_file = _compute_group_mean_netcdf(member_files, out_file)
        if not mean_file:
            continue

        group_files.append(str(mean_file))
        group_labels.append(group_label)

    if not group_files:
        return base_files, base_labels, []

    return group_files + base_files, group_labels + base_labels, group_labels


def _parse_table_numeric(value: Any) -> float:
    """Parse one table cell into float when possible."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        return val if np.isfinite(val) else np.nan
    txt = str(value).strip().lower()
    if not txt:
        return np.nan
    txt = txt.replace('%', '').replace('*', '').replace(',', '')
    try:
        out = float(txt)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _inject_group_rows_in_rows(
    rows: List[List[Any]],
    group_specs: List[Dict[str, Any]],
    model_labels: List[str],
) -> List[List[Any]]:
    """Insert group-mean rows into one tabular row block."""
    if not rows or not group_specs or not model_labels:
        return rows

    label_to_row_idx: Dict[str, int] = {}
    for ridx, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or not row:
            continue
        label_to_row_idx[_normalize_group_token(row[0])] = ridx

    model_row_indices: List[int] = []
    for label in model_labels:
        idx = label_to_row_idx.get(_normalize_group_token(label))
        if idx is not None:
            model_row_indices.append(int(idx))
    if not model_row_indices:
        return rows

    insert_at = min(model_row_indices)
    existing_group_labels = {
        _normalize_group_token(row[0])
        for row in rows
        if isinstance(row, (list, tuple)) and row and _is_group_mean_label(row[0])
    }

    new_group_rows: List[List[Any]] = []
    for spec in group_specs:
        label = str(spec.get('label') or '')
        if not label:
            continue
        if _normalize_group_token(label) in existing_group_labels:
            continue
        member_rows: List[List[Any]] = []
        for member_idx in (spec.get('member_indices') or []):
            if 0 <= int(member_idx) < len(model_labels):
                row_idx = label_to_row_idx.get(_normalize_group_token(model_labels[int(member_idx)]))
                if row_idx is not None and 0 <= row_idx < len(rows):
                    member_rows.append(list(rows[row_idx]))
        if not member_rows:
            continue

        n_cols = max(len(r) for r in member_rows)
        out_row: List[Any] = [label]
        for col in range(1, n_cols):
            col_vals = [_parse_table_numeric(r[col]) for r in member_rows if col < len(r)]
            valid = [v for v in col_vals if np.isfinite(v)]
            if not valid:
                out_row.append('nan')
                continue
            out_row.append(_format_min_sig(float(np.nanmean(valid)), min_sig=3))
        new_group_rows.append(out_row)

    if not new_group_rows:
        return rows
    return rows[:insert_at] + new_group_rows + rows[insert_at:]


def _inject_group_rows_into_table_payload(
    payload: Any,
    group_specs: List[Dict[str, Any]],
    model_labels: List[str],
) -> Any:
    """Recursively inject group-mean rows into one metric-table payload."""
    if not isinstance(payload, dict) or not group_specs or not model_labels:
        return payload

    ptype = str(payload.get('type', '')).strip().lower()

    if ptype == 'dual_table':
        sections = payload.get('sections', [])
        if isinstance(sections, list):
            payload['sections'] = [
                _inject_group_rows_into_table_payload(sec, group_specs, model_labels)
                if isinstance(sec, dict) else sec
                for sec in sections
            ]
        return payload

    if ptype.startswith('region_'):
        regions = payload.get('regions', {})
        if isinstance(regions, dict):
            payload['regions'] = {
                key: _inject_group_rows_into_table_payload(val, group_specs, model_labels)
                if isinstance(val, dict) else val
                for key, val in regions.items()
            }
        return payload

    if isinstance(payload.get('rows'), list):
        payload['rows'] = _inject_group_rows_in_rows(payload['rows'], group_specs, model_labels)

    for container_key in ('seasons', 'periods', 'phases'):
        container = payload.get(container_key)
        if isinstance(container, dict):
            for key, rows in list(container.items()):
                if isinstance(rows, list):
                    container[key] = _inject_group_rows_in_rows(rows, group_specs, model_labels)

    extra_tables = payload.get('extra_tables')
    if isinstance(extra_tables, list):
        payload['extra_tables'] = [
            _inject_group_rows_into_table_payload(tbl, group_specs, model_labels)
            if isinstance(tbl, dict) else tbl
            for tbl in extra_tables
        ]

    return payload


def _format_min_sig(value: Any, min_sig: int = 3, max_decimals: int = 10) -> str:
    """Format a scalar with at least ``min_sig`` significant digits."""
    if value is None:
        return 'NaN'
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 'NaN'
    if not np.isfinite(v):
        return 'NaN'
    if v == 0.0:
        return '0.000'

    abs_v = abs(v)
    sig = max(int(min_sig), 1)
    # Use scientific notation for very small/large values to keep tables readable.
    if abs_v < 1e-3 or abs_v >= 1e4:
        return f'{v:.{sig - 1}e}'

    exp10 = int(np.floor(np.log10(abs_v)))
    decimals = max(sig - exp10 - 1, 0)

    if decimals > int(max_decimals):
        return f'{v:.{sig - 1}e}'

    txt = f'{v:.{decimals}f}'
    sig = len(txt.lstrip('+-').replace('.', '').lstrip('0'))
    if sig < int(min_sig):
        if '.' not in txt:
            txt += '.'
        txt += '0' * (int(min_sig) - sig)
    return txt


def _calc_obs2_ratio(value: Any, obs2_value: Any, eps: float = 1e-9) -> float:
    """Backward-compatible helper: keep physical-difference value unchanged.

    NOTE:
    The public module helpers still call this function in several places where
    values used to be converted to ``|E_model| / |E_obs2|``. HTML scalar tables
    now keep physical units, so this function intentionally returns the input
    value (with finite guards) instead of ratio-transforming it.
    """
    _ = obs2_value
    _ = eps
    try:
        vv = float(value)
    except (TypeError, ValueError):
        return np.nan
    return vv if np.isfinite(vv) else np.nan


def _obs2_identity_ratio(obs2_value: Any, eps: float = 1e-9) -> float:
    """Backward-compatible helper: keep obs2 physical-difference value."""
    _ = eps
    try:
        bb = float(obs2_value)
    except (TypeError, ValueError):
        return np.nan
    return bb if np.isfinite(bb) else np.nan


def _build_metric_table(headers: list, obs2_diff, diff_dicts: list,
                        model_labels: list, units: list = None,
                        obs1_label: str = 'obs1 (baseline)',
                        obs2_label: str = 'obs2') -> dict:
    """Build a scalar metric summary table dict for the HTML report.

    Difference rows keep physical units when obs2 is available:
      - obs1 row: 0 (baseline marker)
      - obs2 row: obs1-vs-obs2 physical difference
      - model rows: obs1-vs-model physical difference
    """
    rows = [[str(obs1_label)] + [_format_min_sig(0.0, min_sig=3) for _ in headers]]
    obs2_vals = None
    if obs2_diff is not None:
        obs2_vals = [obs2_diff.get(k, float('nan')) for k in headers]
        rows.append(
            [str(obs2_label)] +
            [_format_min_sig(_obs2_identity_ratio(v), min_sig=3) for v in obs2_vals]
        )
    for label, d in zip(model_labels, diff_dicts):
        vals = [d.get(k, float('nan')) for k in headers]
        if obs2_vals is not None:
            vals = [_calc_obs2_ratio(v, b) for v, b in zip(vals, obs2_vals)]
        rows.append([label] + [_format_min_sig(v, min_sig=3) for v in vals])
    out_units = [''] + (units or [''] * len(headers))
    return {
        'headers': ['Dataset'] + headers,
        'rows': rows,
        'units': out_units,
    }


def _load_staged_payloads(stage_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load staged worker payloads in their original order."""
    payloads: List[Dict[str, Any]] = []
    for item in stage_refs:
        payload_file = Path(str(item['payload_file']))
        payloads.append(_load_pickle(payload_file))
    return payloads


def _format_siconc_value(value: float, digits: int = 3) -> str:
    """Format SIconc scalar values for report display."""
    if value is None:
        return 'nan'
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 'nan'
    return 'nan' if np.isnan(val) else f'{val:.{digits}f}'


def _format_siconc_trend(trend: float, pvalue: float, digits: int = 3) -> str:
    """Format trend with optional significance marker."""
    txt = _format_siconc_value(trend, digits=digits)
    if txt == 'nan':
        return txt
    try:
        pv = float(pvalue)
    except (TypeError, ValueError):
        pv = np.nan
    return f'{txt}*' if np.isfinite(pv) and pv < 0.05 else txt


def _siconc_season_months(hemisphere: str) -> Dict[str, Tuple[int, int, int]]:
    """Return canonical season -> month mapping for one hemisphere."""
    h = str(hemisphere or '').lower()
    if h == 'sh':
        return {
            'spring': (9, 10, 11),   # SON
            'summer': (12, 1, 2),    # DJF
            'autumn': (3, 4, 5),     # MAM
            'winter': (6, 7, 8),     # JJA
        }
    return {
        'spring': (3, 4, 5),         # MAM
        'summer': (6, 7, 8),         # JJA
        'autumn': (9, 10, 11),       # SON
        'winter': (12, 1, 2),        # DJF
    }


def _siconc_weighted_clim_value(monthly_vals: np.ndarray, months: Tuple[int, ...]) -> float:
    """Weighted mean over selected calendar months from 12-month climatology."""
    if monthly_vals.size < 12:
        return np.nan
    idx = [int(mm) - 1 for mm in months if 1 <= int(mm) <= 12]
    if not idx:
        return np.nan
    vals = np.asarray([monthly_vals[ii] for ii in idx], dtype=float)
    wts = np.asarray([float(DAYS_PER_MONTH[ii]) for ii in idx], dtype=float)
    valid = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if int(np.sum(valid)) <= 0:
        return np.nan
    vals = vals[valid]
    wts = wts[valid]
    if np.sum(wts) <= 0:
        return np.nan
    return float(np.nansum(vals * wts) / np.sum(wts))


def _build_siconc_iiee_period_stats(diff_stats: Dict[str, Any], hemisphere: str) -> Dict[str, float]:
    """Convert SIconc pairwise diff payload to period-wise IIEE means."""
    out = {k: np.nan for k in ('annual', 'spring', 'summer', 'autumn', 'winter', 'march', 'september')}
    if not isinstance(diff_stats, dict) or not diff_stats:
        return out

    clim = np.asarray(diff_stats.get('IIEE_clim_diff', np.array([])), dtype=float).reshape(-1)
    if clim.size >= 12:
        out['annual'] = _siconc_weighted_clim_value(clim, tuple(range(1, 13)))
        season_months = _siconc_season_months(hemisphere)
        for season in ('spring', 'summer', 'autumn', 'winter'):
            out[season] = _siconc_weighted_clim_value(clim, season_months[season])
        out['march'] = float(clim[2]) if np.isfinite(clim[2]) else np.nan
        out['september'] = float(clim[8]) if np.isfinite(clim[8]) else np.nan
    else:
        try:
            out['annual'] = float(diff_stats.get('IIEE_mean_diff', np.nan))
        except (TypeError, ValueError):
            out['annual'] = np.nan
    return out


def _build_siconc_period_table(obs1_stats: dict, obs2_stats: dict,
                               model_stats_list: list, model_labels: list,
                               obs1_label: str = 'obs1 (baseline)',
                               obs2_label: str = 'obs2') -> dict:
    """Build Shu-style period-tab table payload for SIconc."""
    headers = [
        'Dataset',
        'SIE_mean', 'SIE_trend', 'SIE_detrended_std',
        'SIA_mean', 'SIA_trend', 'SIA_detrended_std',
        'MIZ_mean', 'MIZ_trend', 'MIZ_detrended_std',
        'PIA_mean', 'PIA_trend', 'PIA_detrended_std',
    ]
    units = [
        '',
        '10⁶ km²', '10⁶ km²/decade', '10⁶ km²',
        '10⁶ km²', '10⁶ km²/decade', '10⁶ km²',
        '10⁶ km²', '10⁶ km²/decade', '10⁶ km²',
        '10⁶ km²', '10⁶ km²/decade', '10⁶ km²',
    ]

    period_map = [
        ('Annual', 'annual'),
        ('Spring', 'spring'),
        ('Summer', 'summer'),
        ('Autumn', 'autumn'),
        ('Winter', 'winter'),
        ('March', 'march'),
        ('September', 'september'),
    ]

    datasets = [(str(obs1_label), obs1_stats), (str(obs2_label), obs2_stats)] + list(zip(model_labels, model_stats_list))

    periods = {}
    metric_names = ('SIE', 'SIA', 'MIZ', 'PIA')
    for period_label, period_key in period_map:
        rows = []
        for label, s in datasets:
            p = s.get(period_key, {}) if isinstance(s, dict) else {}
            row = [label]
            for metric_name in metric_names:
                node = p.get(metric_name, {}) if isinstance(p, dict) else {}
                row.extend([
                    _format_siconc_value(node.get('mean')),
                    _format_siconc_trend(node.get('trend'), node.get('pvalue')),
                    _format_siconc_value(node.get('detrended_std')),
                ])
            rows.append(row)
        periods[period_label] = rows

    return {
        'type': 'period_table',
        'headers': headers,
        'units': units,
        'periods': periods,
    }


def _build_siconc_period_diff_table(obs1_stats: dict, obs2_stats: dict,
                                    model_stats_list: list, model_labels: list,
                                    iiee_period_obs2: Optional[Dict[str, float]] = None,
                                    iiee_period_model_list: Optional[List[Dict[str, float]]] = None,
                                    obs1_label: str = 'obs1 (baseline)',
                                    obs2_label: str = 'obs2') -> dict:
    """Build period-tab SIconc difference table relative to obs1."""
    headers = [
        'Dataset',
        'SIE_mean_diff', 'SIE_trend_diff', 'SIE_detrended_std_diff',
        'SIA_mean_diff', 'SIA_trend_diff', 'SIA_detrended_std_diff',
        'MIZ_mean_diff', 'MIZ_trend_diff', 'MIZ_detrended_std_diff',
        'PIA_mean_diff', 'PIA_trend_diff', 'PIA_detrended_std_diff',
        'IIEE_mean_diff',
    ]
    base_units = [
        '',
        '10⁶ km²', '10⁶ km²/decade', '10⁶ km²',
        '10⁶ km²', '10⁶ km²/decade', '10⁶ km²',
        '10⁶ km²', '10⁶ km²/decade', '10⁶ km²',
        '10⁶ km²', '10⁶ km²/decade', '10⁶ km²',
        '10⁶ km²',
    ]
    has_obs2 = isinstance(obs2_stats, dict) and bool(obs2_stats)
    units = base_units
    period_map = [
        ('Annual', 'annual'),
        ('Spring', 'spring'),
        ('Summer', 'summer'),
        ('Autumn', 'autumn'),
        ('Winter', 'winter'),
        ('March', 'march'),
        ('September', 'september'),
    ]

    model_datasets: List[Tuple[str, dict]] = [
        (model_labels[idx] if idx < len(model_labels) else f'model{idx + 1}', stats)
        for idx, stats in enumerate(model_stats_list or [])
    ]
    model_iiee_periods: List[Dict[str, float]] = list(iiee_period_model_list or [])
    obs2_iiee_period = iiee_period_obs2 if isinstance(iiee_period_obs2, dict) else {}

    def _to_float(value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return np.nan
        return out if np.isfinite(out) else np.nan

    def _diff_value(lhs: dict, rhs: dict, key: str) -> float:
        lv = _to_float(lhs.get(key))
        rv = _to_float(rhs.get(key))
        if not (np.isfinite(lv) and np.isfinite(rv)):
            return np.nan
        return abs(lv - rv)

    periods: Dict[str, List[List[str]]] = {}
    metric_names = ('SIE', 'SIA', 'MIZ', 'PIA')
    for period_label, period_key in period_map:
        ref_period = obs1_stats.get(period_key, {}) if isinstance(obs1_stats, dict) else {}
        ref_nodes = {
            metric_name: (ref_period.get(metric_name, {}) if isinstance(ref_period, dict) else {})
            for metric_name in metric_names
        }
        rows: List[List[str]] = []

        rows.append(
            [str(obs1_label)] + [_format_siconc_value(0.0) for _ in headers[1:]]
        )

        if has_obs2:
            period_stats = obs2_stats.get(period_key, {}) if isinstance(obs2_stats, dict) else {}
            row_vals: List[float] = []
            for metric_name in metric_names:
                node = period_stats.get(metric_name, {}) if isinstance(period_stats, dict) else {}
                ref_node = ref_nodes.get(metric_name, {})
                row_vals.extend([
                    _diff_value(node, ref_node, 'mean'),
                    _diff_value(node, ref_node, 'trend'),
                    _diff_value(node, ref_node, 'detrended_std'),
                ])
            row_vals.append(_to_float(obs2_iiee_period.get(period_key)))
            rows.append(
                [str(obs2_label)] +
                [_format_siconc_value(_obs2_identity_ratio(vv)) for vv in row_vals]
            )

        for idx, (label, stats) in enumerate(model_datasets):
            period_stats = stats.get(period_key, {}) if isinstance(stats, dict) else {}
            row_vals: List[float] = []
            for metric_name in metric_names:
                node = period_stats.get(metric_name, {}) if isinstance(period_stats, dict) else {}
                ref_node = ref_nodes.get(metric_name, {})
                row_vals.extend([
                    _diff_value(node, ref_node, 'mean'),
                    _diff_value(node, ref_node, 'trend'),
                    _diff_value(node, ref_node, 'detrended_std'),
                ])
            model_iiee = model_iiee_periods[idx] if idx < len(model_iiee_periods) and isinstance(model_iiee_periods[idx], dict) else {}
            row_vals.append(_to_float(model_iiee.get(period_key)))
            rows.append([label] + [_format_siconc_value(vv) for vv in row_vals])

        periods[period_label] = rows

    return {
        'type': 'period_table',
        'headers': headers,
        'units': units,
        'periods': periods,
    }


def _siconc_period_to_seasonal_table(period_table: dict) -> dict:
    """Convert SIconc period table payload to seasonal_table schema."""
    periods = period_table.get('periods', {}) if isinstance(period_table, dict) else {}
    season_order = [
        p for p in ('Annual', 'Spring', 'Summer', 'Autumn', 'Winter', 'March', 'September')
        if p in periods
    ]
    if not season_order:
        season_order = list(periods.keys())
    return {
        'type': 'seasonal_table',
        'season_order': season_order,
        'headers': list(period_table.get('headers', [])),
        'rows': [],
        'units': list(period_table.get('units', [])),
        'seasons': {k: list(periods.get(k, [])) for k in season_order},
    }


def _build_siconc_region_period_table(hemisphere: str,
                                      regional_stats: Dict[str, Dict[str, Any]],
                                      model_labels: list,
                                      obs_labels: Optional[List[str]] = None) -> dict:
    """Build region-aware SIconc table payload with Raw/Differences tabs."""
    region_order = utils.get_hemisphere_sectors(hemisphere, include_all=True)
    region_labels = {
        key: utils.get_sector_label(hemisphere, key)
        for key in region_order
    }

    obs1_label = (
        f'{obs_labels[0]} (baseline)'
        if isinstance(obs_labels, list) and len(obs_labels) >= 1 and str(obs_labels[0]).strip()
        else 'obs1 (baseline)'
    )
    obs2_label = (
        str(obs_labels[1]).strip()
        if isinstance(obs_labels, list) and len(obs_labels) >= 2 and str(obs_labels[1]).strip()
        else 'obs2'
    )

    regions_payload: Dict[str, Dict[str, Any]] = {}
    for sector in region_order:
        stats_pack = regional_stats.get(sector, {})
        raw_table = _build_siconc_period_table(
            obs1_stats=stats_pack.get('obs1_stats', {}),
            obs2_stats=stats_pack.get('obs2_stats', {}),
            model_stats_list=stats_pack.get('model_stats_list', []),
            model_labels=model_labels,
            obs1_label=obs1_label,
            obs2_label=obs2_label,
        )
        diff_table = _build_siconc_period_diff_table(
            obs1_stats=stats_pack.get('obs1_stats', {}),
            obs2_stats=stats_pack.get('obs2_stats', {}),
            model_stats_list=stats_pack.get('model_stats_list', []),
            model_labels=model_labels,
            iiee_period_obs2=stats_pack.get('obs2_iiee_period', {}),
            iiee_period_model_list=stats_pack.get('model_iiee_period_list', []),
            obs1_label=obs1_label,
            obs2_label=obs2_label,
        )
        regions_payload[sector] = {
            'type': 'dual_table',
            'sections': [
                {'id': 'raw', 'title': 'Raw Values', **_siconc_period_to_seasonal_table(raw_table)},
                {'id': 'diff', 'title': 'Differences', **_siconc_period_to_seasonal_table(diff_table)},
            ],
        }

    return {
        'type': 'region_dual_table',
        'region_order': region_order,
        'region_labels': region_labels,
        'regions': regions_payload,
    }


def _build_region_table_payload(hemisphere: str,
                                regional_tables: Dict[str, Any],
                                payload_type: str = 'region_table') -> dict:
    """Wrap per-sector table payloads for HTML region-tab rendering."""
    ordered = []
    try:
        for sector in utils.get_hemisphere_sectors(hemisphere, include_all=True):
            if sector in regional_tables:
                ordered.append(sector)
    except Exception:
        pass
    # Preserve any extra/custom sector keys not in the canonical order.
    for sector in regional_tables.keys():
        if sector not in ordered:
            ordered.append(sector)

    labels = {
        sector: utils.get_sector_label(hemisphere, sector)
        for sector in ordered
    }
    return {
        'type': payload_type,
        'region_order': ordered,
        'region_labels': labels,
        'regions': {sector: regional_tables[sector] for sector in ordered},
    }


def _run_preprocessing(case_name: str, module: str, recipe: RR.RecipeReader,
                       data_dir: Path, frequency: str = 'monthly',
                       obs_flag: Optional[str] = None,
                       overwrite_models: bool = False,
                       jobs: int = 1) -> Tuple[str, List[str], List[str]]:
    """Common preprocessing pipeline for all modules.

    Args:
        case_name: Evaluation experiment name.
        module: Module name (e.g., 'SIconc', 'SIdrift').
        recipe: Validated RecipeReader instance.
        data_dir: Directory for processed data files.
        frequency: Data frequency ('monthly' or 'daily').
        obs_flag: Optional flag name for observation files (e.g., '_sic', '_sidrift').
        overwrite_models: Whether to rebuild processed model files even if they exist.
        jobs: Maximum number of worker threads for model preprocessing.

    Returns:
        Tuple of (grid_file, obs_files, model_files).
    """
    preprocessor = PP.DataPreprocessor(case_name, module, hemisphere=recipe.hemisphere)
    grid_file = preprocessor.gen_eval_grid()
    file_groups = recipe.validate_module(module)

    # Ensure flag_name is a string (empty string if None)
    flag_name = obs_flag if obs_flag is not None else ''
    obs_files = preprocessor.prep_obs(
        frequency=frequency,
        output_dir=data_dir,
        flag_name=flag_name,
        jobs=jobs,
    )

    model_files = preprocessor.prep_models(
        file_groups, frequency=frequency, output_dir=data_dir, overwrite=overwrite_models, jobs=jobs
    ) if file_groups else []

    return grid_file, obs_files, model_files


def _get_eval_grid(case_name: str, module: str, hemisphere: str) -> str:
    """Generate or reuse the evaluation grid without running full preprocessing."""
    hms = str(hemisphere or '').lower()
    env_key = f'SITOOL_EVAL_GRID_{hms.upper()}'
    env_grid = str(os.environ.get(env_key, '')).strip()
    if env_grid and Path(env_grid).exists():
        return env_grid

    preprocessor = PP.DataPreprocessor(case_name, module, hemisphere=hemisphere)
    return preprocessor.gen_eval_grid()


def _is_grid_file_healthy(grid_file: Path) -> Tuple[bool, Optional[str]]:
    """Return True when one prebuilt evaluation grid can be opened safely."""
    nc_path = Path(grid_file)
    txt_path = nc_path.with_suffix('.txt')
    if not nc_path.exists():
        return False, f'missing NetCDF grid file: {nc_path}'
    if not txt_path.exists():
        return False, f'missing CDO grid descriptor: {txt_path}'

    try:
        with xr.open_dataset(nc_path) as ds:
            required_vars = ('lon', 'lat', 'cell_area', 'land_sea_mask', 'sea_ice_region')
            for name in required_vars:
                if name not in ds.variables:
                    return False, f"grid variable '{name}' is missing in {nc_path}"
    except Exception as exc:
        return False, str(exc)
    return True, None


def _prepare_eval_grids_serial(
    case_name: str,
    bootstrap_module: str,
    eval_hms: List[str],
) -> Dict[str, Path]:
    """Build (or repair) hemisphere grids serially before parallel module tasks."""
    prepared: Dict[str, Path] = {}
    for hms in eval_hms:
        env_key = f'SITOOL_EVAL_GRID_{str(hms).upper()}'
        os.environ.pop(env_key, None)

        grid_path = Path(_get_eval_grid(case_name, bootstrap_module, hms)).resolve()
        healthy, reason = _is_grid_file_healthy(grid_path)
        if not healthy:
            logger.warning(
                "[%s] Detected invalid evaluation grid (%s). Rebuilding serially.",
                str(hms).upper(), reason,
            )
            try:
                grid_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                grid_path.with_suffix('.txt').unlink(missing_ok=True)
            except Exception:
                pass
            grid_path = Path(_get_eval_grid(case_name, bootstrap_module, hms)).resolve()
            healthy, reason = _is_grid_file_healthy(grid_path)
            if not healthy:
                raise RuntimeError(
                    f"Failed to prepare healthy evaluation grid for hemisphere '{hms}': {reason}"
                )

        prepared[hms] = grid_path
        os.environ[env_key] = str(grid_path)
        logger.info("[%s] Prepared evaluation grid serially: %s", str(hms).upper(), grid_path)
    return prepared



def _init_case_dirs(case_dir: Path, eval_hms: List[str],
                    modules_to_run: List[str]) -> Tuple[Path, Path]:
    """Create and return (data_dir, output_dir) with all sub-directories pre-created.

    Creates:
      cases/<case>/Processed/<hms>/
      cases/<case>/Output/<hms>/<module>/   for every (hms, module) combination
      cases/<case>/metrics/

    Returns:
        (data_dir, output_dir) — base Path objects.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    data_dir = case_dir / 'Processed'
    output_dir = case_dir / 'Output'
    metrics_dir = case_dir / 'metrics'
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for hms in eval_hms:
        (data_dir / hms).mkdir(exist_ok=True)
        (output_dir / hms).mkdir(exist_ok=True)
        for module in modules_to_run:
            (output_dir / hms / module).mkdir(exist_ok=True)
    return data_dir, output_dir


# ---------------------------------------------------------------------------
# Module evaluation functions
# ---------------------------------------------------------------------------






def _run_one_module_task(
    case_name: str,
    hemisphere: str,
    module: str,
    data_dir: Path,
    output_dir: Path,
    recalculate: bool,
    jobs_for_module: int,
    isolated_worker_logging: bool = False,
) -> Dict[str, Any]:
    """Run one (hemisphere, module) task with an isolated recipe instance.

    Each task creates its own RecipeReader object to avoid race conditions when
    module tasks run concurrently and update `recipe.hemisphere`.
    """
    worker_log: Optional[Path] = None
    if isolated_worker_logging:
        worker_log = _configure_worker_logging(case_name, hemisphere, module)
        logger.info(
            "Worker initialized for [%s/%s] (pid=%d, log=%s)",
            hemisphere.upper(), module, os.getpid(), worker_log,
        )
    _install_thread_safe_plot_wrappers()
    task_recipe = RR.RecipeReader(case_name)
    task_recipe.hemisphere = hemisphere
    from scripts.pipeline import modules as eval_mods
    eval_funcs = {
        'SIconc': eval_mods.eval_sic,
        'SIthick': eval_mods.eval_sithick,
        'SNdepth': eval_mods.eval_sndepth,
        'SIdrift': eval_mods.eval_sidrift,
        'SICB': eval_mods.eval_sicb,
        'SItrans': eval_mods.eval_sitrans,
        'SIMbudget': eval_mods.eval_simbudget,
        'SNMbudget': eval_mods.eval_snmbudget,
    }
    requested_jobs = max(1, int(jobs_for_module))
    case_name_norm = str(case_name or '').strip().lower()
    aggressive_parallel = (
        case_name_norm.startswith('highres')
        or str(os.environ.get('SITOOL_AGGRESSIVE_PARALLEL', '')).strip().lower() in {
            '1', 'true', 'yes', 'on',
        }
    )
    prev_cdo_threads = os.environ.get('SITOOL_CDO_THREADS')
    prev_prep_parallel = os.environ.get('SITOOL_PREP_MAX_PARALLEL')
    prev_prep_force_serial = os.environ.get('SITOOL_PREP_FORCE_SERIAL')
    if not str(prev_cdo_threads or '').strip():
        cdo_threads = requested_jobs
        if aggressive_parallel:
            if module in {'SICB', 'SIdrift', 'SItrans'}:
                cdo_threads = min(requested_jobs, 4)
            elif module in {'SIconc', 'SIthick', 'SNdepth'}:
                cdo_threads = min(requested_jobs, 3)
        os.environ['SITOOL_CDO_THREADS'] = str(max(1, int(cdo_threads)))
    if not str(prev_prep_parallel or '').strip():
        prep_cap = _resolve_default_prep_parallel_cap(requested_jobs, module, case_name=case_name)
        # Keep Python-side preprocessing fan-out conservative by default to
        # reduce NetCDF/HDF5 lock contention risk while still enabling safe
        # low-degree parallel speedups in preprocess-heavy highres runs.
        os.environ['SITOOL_PREP_MAX_PARALLEL'] = str(prep_cap)
    if not str(prev_prep_force_serial or '').strip():
        os.environ['SITOOL_PREP_FORCE_SERIAL'] = '0'
    logger.info(
        "Worker parallel policy [%s/%s]: module_jobs=%d, CDO threads=%s, prep_python_workers_cap=%s",
        hemisphere.upper(),
        module,
        requested_jobs,
        os.environ.get('SITOOL_CDO_THREADS', '1'),
        os.environ.get('SITOOL_PREP_MAX_PARALLEL', '1'),
    )
    try:
        with rte.process_busy_scope(
            module=module,
            hemisphere=hemisphere,
            expected_threads=requested_jobs,
        ):
            result = eval_funcs[module](
                case_name, task_recipe, data_dir, output_dir,
                recalculate=recalculate, jobs=requested_jobs,
            )
        if isinstance(result, dict):
            try:
                module_vars = task_recipe.variables.get(module, {}) if isinstance(task_recipe.variables, dict) else {}
                n_models_declared = _model_count_for_module(module, module_vars)
                base_model_labels = _get_recipe_model_labels(module, module_vars, n_models_declared)
                group_specs = _resolve_group_mean_specs(
                    module=module,
                    module_vars=module_vars if isinstance(module_vars, dict) else {},
                    common_config=getattr(task_recipe, 'common_config', {}),
                    model_labels=base_model_labels,
                )
                if group_specs:
                    result = _inject_group_rows_into_table_payload(
                        payload=result,
                        group_specs=group_specs,
                        model_labels=base_model_labels,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to inject group-mean rows for [%s/%s] payload (%s).",
                    hemisphere.upper(), module, exc,
                )
    except Exception as exc:
        # Some third-party exceptions (for example python-cdo CDOException) are
        # not reliably pickleable across process boundaries. Re-raise as a plain
        # RuntimeError so the parent process can always receive the failure.
        logger.exception(
            "Worker task failed for [%s/%s]: %s",
            hemisphere.upper(), module, exc,
        )
        raise RuntimeError(
            f"Worker task failed for [{hemisphere.upper()}/{module}]: {exc}"
        ) from None
    finally:
        if prev_cdo_threads is None:
            os.environ.pop('SITOOL_CDO_THREADS', None)
        else:
            os.environ['SITOOL_CDO_THREADS'] = prev_cdo_threads
        if prev_prep_parallel is None:
            os.environ.pop('SITOOL_PREP_MAX_PARALLEL', None)
        else:
            os.environ['SITOOL_PREP_MAX_PARALLEL'] = prev_prep_parallel
        if prev_prep_force_serial is None:
            os.environ.pop('SITOOL_PREP_FORCE_SERIAL', None)
        else:
            os.environ['SITOOL_PREP_FORCE_SERIAL'] = prev_prep_force_serial
    return {
        'hemisphere': hemisphere,
        'module': module,
        'result': result,
        'worker_log': str(worker_log) if worker_log is not None else None,
    }


def _refresh_html_report(
    case_name: str,
    output_dir: Path,
    modules_to_run: List[str],
    metric_tables: Dict[str, Dict[str, Any]],
    context: str,
) -> None:
    """Regenerate summary_report.html after one module task completes."""
    report_file = Path(output_dir) / 'summary_report.html'

    def _metric_table_store_file() -> Path:
        return Path(output_dir) / '.metric_tables.pkl'

    def _save_metric_tables_snapshot() -> None:
        store_file = _metric_table_store_file()
        store_file.parent.mkdir(parents=True, exist_ok=True)
        with open(store_file, 'wb') as f:
            pickle.dump(metric_tables, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _resolve_report_modules() -> List[str]:
        module_set = set(modules_to_run or [])
        for hms_tables in (metric_tables or {}).values():
            if isinstance(hms_tables, dict):
                module_set.update(str(k) for k in hms_tables.keys())

        for hms in ('nh', 'sh'):
            hms_dir = Path(output_dir) / hms
            if not hms_dir.is_dir():
                continue
            for child in hms_dir.iterdir():
                if child.is_dir():
                    module_set.add(child.name)

        ordered = [m for m in ALL_MODULES if m in module_set]
        extras = sorted(m for m in module_set if m not in ALL_MODULES)
        return ordered + extras

    def _extract_panel_block(html_text: str, panel_id: str) -> Optional[str]:
        marker = f'<div class="panel" id="{panel_id}">'
        start = html_text.find(marker)
        if start < 0:
            return None
        tag_re = re.compile(r'<div\b|</div>')
        depth = 0
        for match in tag_re.finditer(html_text, start):
            token = match.group(0)
            if token.startswith('<div'):
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    return html_text[start:match.end()]
        return None

    try:
        _save_metric_tables_snapshot()
        report_modules = _resolve_report_modules()
        has_legacy_report = report_file.is_file()
        legacy_html = ''
        if has_legacy_report:
            try:
                legacy_html = report_file.read_text(encoding='utf-8')
            except Exception:
                legacy_html = ''
        partial_module_update = bool(
            has_legacy_report
            and modules_to_run
            and set(modules_to_run).issubset(set(report_modules))
            and set(modules_to_run) != set(report_modules)
        )

        if partial_module_update:
            old_html = legacy_html
            report.generate_html_report(
                case_name=case_name,
                output_dir=str(output_dir),
                modules_run=modules_to_run,
                metric_tables=metric_tables,
            )
            new_html = report_file.read_text(encoding='utf-8')

            merged_html = old_html
            merged_count = 0
            missing_panel_detected = False
            for hms in ('nh', 'sh'):
                for module_name in modules_to_run:
                    panel_id = f'tab-{hms}-{module_name}'
                    old_block = _extract_panel_block(merged_html, panel_id)
                    new_block = _extract_panel_block(new_html, panel_id)
                    # Any missing target panel implies the partial regenerated
                    # report and legacy report are not structurally aligned.
                    # Fall back to full report regeneration to avoid mixing
                    # stale and fresh module panels.
                    if old_block is None or new_block is None:
                        missing_panel_detected = True
                        continue
                    merged_html = merged_html.replace(old_block, new_block, 1)
                    merged_count += 1

            if missing_panel_detected:
                logger.warning(
                    "HTML merge after %s detected missing panel(s) in legacy report; "
                    "regenerating full report.",
                    context,
                )
                report.generate_html_report(
                    case_name=case_name,
                    output_dir=str(output_dir),
                    modules_run=report_modules,
                    metric_tables=metric_tables,
                )
                logger.info("Updated HTML report after: %s (full regeneration fallback)", context)
            elif merged_count > 0:
                report_file.write_text(merged_html, encoding='utf-8')
                logger.info(
                    "Updated HTML report after: %s (merged %d module panel(s), preserved untouched modules)",
                    context, merged_count
                )
            else:
                logger.warning(
                    "HTML panel merge after %s found no target panel; using regenerated report as fallback.",
                    context,
                )
        else:
            report.generate_html_report(
                case_name=case_name,
                output_dir=str(output_dir),
                modules_run=report_modules,
                metric_tables=metric_tables,
            )
            logger.info("Updated HTML report after: %s", context)
    except Exception as exc:
        logger.warning("Failed to refresh HTML report after %s (%s).", context, exc)


def _load_metric_table_store(
    output_dir: Path,
    hemispheres: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load persisted HTML metric table payloads from the previous run, if any."""
    base: Dict[str, Dict[str, Any]] = {}
    for hms in hemispheres or []:
        base[hms] = {}

    store_file = Path(output_dir) / '.metric_tables.pkl'
    if not store_file.is_file():
        return base

    try:
        with open(store_file, 'rb') as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return base
        for hms, tables in payload.items():
            if not isinstance(hms, str) or not isinstance(tables, dict):
                continue
            if hms not in base:
                base[hms] = {}
            for module, table_payload in tables.items():
                if isinstance(module, str):
                    base[hms][module] = table_payload
    except Exception as exc:
        logger.warning("Failed to load metric table snapshot (%s). Start with empty tables.", exc)
    return base


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Compatibility wrapper that delegates to scripts.pipeline.runner.main."""
    from scripts.pipeline.runner import main as _runner_main
    _runner_main()
