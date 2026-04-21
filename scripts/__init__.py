"""
scripts package initializer.

Centralizes imports of scientific computing libraries shared across all
SIToolv2 sub-modules and exports runtime-wide constants/env defaults.
"""

import os
import sys
import math
import time                # used for timing in SeaIceMetrics
import datetime
import tempfile
import pyproj              # map projections and coordinate transforms
import logging
import warnings
from tqdm import tqdm      # progress bars for long-running computations
warnings.filterwarnings('ignore', message='pkg_resources is deprecated', category=UserWarning)

# Some deployments run on mounted filesystems where HDF5 file-lock semantics
# can cause transient NetCDF write failures under multiprocessing workloads.
# Disable HDF5 file locking by default unless the user explicitly overrides it.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')


def _as_bool(value: str, default: bool = False) -> bool:
    """Parse one environment-like boolean string."""
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _as_bounded_int(value: str, default: int, lower: int, upper: int) -> int:
    """Parse one bounded integer with fallback."""
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = int(default)
    return max(int(lower), min(int(upper), parsed))


def _resolve_runtime_tmpdir() -> str:
    """Resolve runtime temp directory and avoid falling back to /tmp by default."""
    explicit = str(os.environ.get('SITOOL_TMPDIR', '')).strip()
    if explicit:
        target = explicit
    else:
        env_tmp = str(os.environ.get('TMPDIR', '')).strip()
        if (not env_tmp) or (os.path.abspath(env_tmp) == '/tmp'):
            target = os.getcwd()
        else:
            target = env_tmp

    target = os.path.abspath(target)
    os.makedirs(target, exist_ok=True)
    return target


SITOOL_TMPDIR = _resolve_runtime_tmpdir()
for _tmp_key in ('TMPDIR', 'TEMP', 'TMP'):
    os.environ[_tmp_key] = SITOOL_TMPDIR
tempfile.tempdir = SITOOL_TMPDIR
os.environ.setdefault('MPLCONFIGDIR', os.path.join(SITOOL_TMPDIR, '.matplotlib'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

SITOOL_NC_COMPRESS_LEVEL = _as_bounded_int(
    os.environ.get('SITOOL_NC_COMPRESS_LEVEL', '1'),
    default=1,
    lower=1,
    upper=9,
)
SITOOL_NC_SHUFFLE = _as_bool(
    os.environ.get('SITOOL_NC_SHUFFLE', '0'),
    default=False,
)
SITOOL_CDO_COMPRESSION = f'zip_{SITOOL_NC_COMPRESS_LEVEL}'

import numpy as np
import pandas as pd
import xarray as xr        # NetCDF / labelled multi-dimensional array I/O
from cdo import Cdo        # CDO (Climate Data Operators) Python bindings
from scipy import stats    # statistical tests and linear regression

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import List, Optional, Tuple, Dict, Union, Any

# Re-export project-level constants so callers only need one import
from scripts.config import (
    DAYS_PER_MONTH, DAYS_PER_SEASON,
)

# Fix random seed to ensure reproducibility for any stochastic operations
np.random.seed(0)
# Suppress noisy third-party deprecation warnings (e.g. xarray/CDO)
warnings.filterwarnings('ignore')
