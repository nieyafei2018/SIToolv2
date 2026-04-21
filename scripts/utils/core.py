# -*- coding: utf-8 -*-
"""
Basic utility functions for SIToolv2 evaluation system.

This module provides common utility functions including:
- Grid generation and manipulation
- Spatial interpolation
- Statistical calculations
- Data processing utilities

Created on 2023/11/4 15:31
"""

import re
import glob
import logging
import tempfile
from math import sin, asin, cos, radians, fabs, sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import numpy as np
import pyproj
import xarray as xr
from cdo import Cdo
from scipy.stats import pearsonr

from scripts import (
    SITOOL_TMPDIR,
    SITOOL_CDO_COMPRESSION,
    SITOOL_NC_COMPRESS_LEVEL,
    SITOOL_NC_SHUFFLE,
)

logger = logging.getLogger(__name__)

cdo = Cdo(options=[f'-f nc4 -z {SITOOL_CDO_COMPRESSION}'])

def _compressed_encoding(ds: xr.Dataset, base: Dict[str, Dict[str, object]] = None) -> Dict[str, Dict[str, object]]:
    """Build one NetCDF encoding map with lightweight compression for numeric data vars."""
    encoding: Dict[str, Dict[str, object]] = {}
    if base:
        encoding.update({k: dict(v) for k, v in base.items()})

    for var_name in ds.data_vars:
        da = ds[var_name]
        if da.dtype.kind in ('U', 'S', 'O'):
            continue
        var_enc = dict(encoding.get(var_name, {}))
        var_enc.setdefault('zlib', True)
        var_enc.setdefault('complevel', int(SITOOL_NC_COMPRESS_LEVEL))
        var_enc.setdefault('shuffle', bool(SITOOL_NC_SHUFFLE))
        encoding[var_name] = var_enc
    return encoding

def write_netcdf_compressed(ds: xr.Dataset, output_file: str, **kwargs) -> None:
    """Write one NetCDF file with default compression settings for numeric data vars."""
    params = dict(kwargs)
    base_encoding = params.pop('encoding', None)
    merged_encoding = _compressed_encoding(ds, base=base_encoding)
    if merged_encoding:
        params['encoding'] = merged_encoding
    ds.to_netcdf(output_file, **params)

__all__ = [
    "_compressed_encoding",
    "write_netcdf_compressed",
]
