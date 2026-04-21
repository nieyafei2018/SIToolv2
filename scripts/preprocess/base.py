# -*- coding: utf-8 -*-
"""
Observation and model data preprocessing for SIToolv2 evaluation system.

Handles time-resolution matching, period selection, and spatial interpolation
via CDO for both observation and model datasets.
"""

import datetime
import json
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from cdo import Cdo

from scripts import SITOOL_CDO_COMPRESSION
from scripts import utils
from scripts.preprocess import vector_rotation as vr
from scripts.utils import runtime_efficiency as rte
from scripts.pipeline.recipe_reader import RecipeReader

logger = logging.getLogger(__name__)

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None

# Process-local file locks used to prevent concurrent writes to the same output
# path when multiple modules run in parallel threads.
_PATH_LOCKS: Dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _plan_nested_workers(total_jobs: int, n_groups: int) -> Tuple[int, int]:
    """Plan a bounded two-level worker layout.

    Returns:
        (outer_workers, inner_workers)

    Guarantees:
        outer_workers >= 1
        inner_workers >= 1
        outer_workers * inner_workers <= total_jobs
    """
    safe_jobs = max(1, int(total_jobs))
    groups = max(1, int(n_groups))

    if safe_jobs == 1:
        return 1, 1
    if groups == 1:
        return 1, safe_jobs

    outer = min(groups, safe_jobs)
    inner = max(1, safe_jobs // outer)

    # If every group is already active but no inner fan-out is possible,
    # rebalance to expose some segment-level concurrency.
    if outer == groups and inner == 1 and safe_jobs > groups and groups > 2:
        outer = max(2, groups - 1)
        inner = max(1, safe_jobs // outer)

    while outer * inner > safe_jobs and inner > 1:
        inner -= 1

    return max(1, outer), max(1, inner)


def _get_path_lock(path: str) -> threading.Lock:
    """Return a stable process-local lock object for one filesystem path."""
    norm_path = os.path.abspath(str(path))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(norm_path)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[norm_path] = lock
        return lock


@contextmanager
def _acquire_path_locks(paths: List[str]):
    """Acquire deterministic in-process and cross-process path locks."""
    normalized = sorted({os.path.abspath(str(p)) for p in paths})
    locks = [_get_path_lock(p) for p in normalized]
    fd_locks: List[int] = []

    for lock in locks:
        lock.acquire()

    try:
        if fcntl is not None:
            for path in normalized:
                lock_path = f"{path}.sitool.lock"
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except Exception:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                    raise
                fd_locks.append(fd)
        yield
    finally:
        for fd in reversed(fd_locks):
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(fd)
            except Exception:
                pass
        for lock in reversed(locks):
            lock.release()

class DataPreprocessorBase:
    """Base preprocessor with shared IO/CDO setup helpers."""

    def __init__(self, eval_ex: str, module_name: str, hemisphere: str = None):
        """Initialize the data preprocessor.

        Args:
            eval_ex: Evaluation experiment name.
            module_name: Module name (e.g., 'SIconc', 'SIdrift').
            hemisphere: Active hemisphere override ('nh' or 'sh').
                        If None, falls back to the first entry in the recipe's eval_hms list.
        """
        self.eval_ex = eval_ex
        self.module_name = module_name

        self.recipe_reader = RecipeReader(eval_ex)
        self.recipe_vars = self.recipe_reader.variables
        self.common_vars = self.recipe_vars['common']

        self.hemisphere = hemisphere if hemisphere is not None else self.recipe_reader.hemisphere
        self.grid_info = self.common_vars['grid_info']
        self.ref_data_path = self.common_vars['SIToolv2_RefData_path']
        self.model_data_path = self.common_vars['model_data_path']

        # Resolved output directory for this experiment.
        # Main can override this to a fallback run directory when
        # cases/<eval_ex> is not usable on the host filesystem.
        self.eval_dir = os.environ.get('SITOOL_CASE_RUN_DIR', os.path.join('cases', eval_ex))
        os.makedirs(self.eval_dir, exist_ok=True)
        # Use a case-local temporary root to avoid exhausting /tmp on long runs.
        self.temp_root = os.path.join(self.eval_dir, '_tmp')
        os.makedirs(self.temp_root, exist_ok=True)

    def _tempdir(self, prefix: str):
        """Create a case-local temporary directory context."""
        return tempfile.TemporaryDirectory(prefix=prefix, dir=self.temp_root)

    @staticmethod
    def _new_cdo_instance() -> Cdo:
        """Create a CDO instance for preprocessing.

        By default preprocessing writes compressed NetCDF4 with a lightweight
        deflate level to balance disk usage and parallel throughput. Thread
        count can be controlled by ``SITOOL_CDO_THREADS`` (mapped to ``cdo -P``).
        Set environment variable SITOOL_PREP_COMPRESS=0 to disable compression.
        """
        compress = str(os.environ.get('SITOOL_PREP_COMPRESS', '1')).strip().lower() in {
            '1', 'true', 'yes', 'on',
        }
        try:
            cdo_threads = max(1, int(os.environ.get('SITOOL_CDO_THREADS', '1')))
        except Exception:
            cdo_threads = 1

        opts: List[str] = []
        if cdo_threads > 1:
            opts.append(f'-P {cdo_threads}')
        if compress:
            opts.append(f'-f nc4 -z {SITOOL_CDO_COMPRESSION}')
        else:
            opts.append('-f nc4')
        return Cdo(options=opts)

    @staticmethod
    def _xarray_monmean(input_file: str, output_file: str) -> None:
        """Fallback monthly mean computation with xarray."""
        with xr.open_dataset(input_file) as ds_in:
            ds_mon = ds_in.resample(time='MS').mean(keep_attrs=True)
            utils.write_netcdf_compressed(ds_mon, output_file)

    @staticmethod
    def _xarray_seldate(input_file: str, output_file: str,
                        start_date: str, end_date: str) -> None:
        """Fallback time slicing with xarray."""
        with xr.open_dataset(input_file) as ds_in:
            ds_sel = ds_in.sel(time=slice(start_date, end_date))
            utils.write_netcdf_compressed(ds_sel, output_file)

    @staticmethod
    def _dataset_time_size(path: str) -> Optional[int]:
        """Return time-axis length for one dataset; None if unavailable/unreadable."""
        try:
            with xr.open_dataset(path, decode_times=False) as ds:
                if 'time' in ds.sizes:
                    return int(ds.sizes['time'])
                if 'time' in ds.dims:
                    return int(ds.dims['time'])
                return None
        except Exception:
            return None

    @staticmethod
    def _is_nonempty_time_file(path: str) -> bool:
        """Return True when dataset exists and has non-empty time axis (or no time axis)."""
        if not os.path.exists(path):
            return False
        tsize = DataPreprocessorBase._dataset_time_size(path)
        return (tsize is None) or (tsize > 0)

    def _safe_monmean(self, cdo_obj: Cdo, input_file: str, output_file: str) -> None:
        """Run CDO monmean with xarray fallback on transient backend failures."""
        try:
            cdo_obj.monmean(input=os.path.abspath(input_file), output=output_file)
        except Exception as cdo_err:
            logger.warning("CDO monmean failed (%s) — falling back to xarray.", cdo_err)
            self._xarray_monmean(os.path.abspath(input_file), output_file)

    def _safe_seldate(self, cdo_obj: Cdo, start_date: str, end_date: str,
                      input_file: str, output_file: str) -> None:
        """Run CDO seldate with xarray fallback on transient backend failures."""
        try:
            cdo_obj.seldate(start_date, end_date, input=input_file, output=output_file)
            out_t = self._dataset_time_size(output_file)
            in_t = self._dataset_time_size(input_file)
            if out_t == 0 and (in_t is None or in_t > 0):
                logger.warning(
                    "CDO seldate produced empty output for %s (%s to %s) — falling back to xarray.",
                    os.path.basename(str(input_file)), start_date, end_date,
                )
                self._xarray_seldate(input_file, output_file, start_date, end_date)
        except Exception as cdo_err:
            logger.warning("CDO seldate failed (%s) — falling back to xarray.", cdo_err)
            self._xarray_seldate(input_file, output_file, start_date, end_date)

__all__ = [
    "DataPreprocessorBase",
    "_plan_nested_workers",
    "_get_path_lock",
    "_acquire_path_locks",
]
