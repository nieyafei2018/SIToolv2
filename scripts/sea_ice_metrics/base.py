# -*- coding: utf-8 -*-
"""
Sea ice metrics calculation for SIToolv2 evaluation system.

This module provides classes and methods to compute sea ice-related metrics,
including sea ice concentration, thickness, drift, and budget metrics.
It supports calculations for single datasets and comparisons between two datasets.

Classes:
    SIconcMetrics: Metrics for sea ice concentration (SIA, SIE, MIZ, PIA).
    Thickness_metrics: Metrics for sea ice thickness and volume.
    SID_metrics: Metrics for sea ice drift (velocity and vector correlation).
    SICB_metrics: Metrics for sea ice concentration budget components.
    SItrans_metrics: Metrics for sea ice transition dates (advance/retreat).

Created on 2023/10/31 11:35
"""

import json
import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scripts.utils.netcdf_io import open_netcdf_dataset

from scripts import SITOOL_NC_COMPRESS_LEVEL, SITOOL_NC_SHUFFLE

logger = logging.getLogger(__name__)

# ------------------------------------------------------


"""

For *_2M_metrics() function, the var1 is recommended to be the observations,
    the var2 is recommended to be the model results.

"""


class SeaIceMetricsBase:
    """Base class for sea ice metrics calculations."""

    # -----------------------------------------------------------------------------
    # Function: `__init__`.
    # Purpose: Handle init workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def __init__(self, grid_file: str, hemisphere: str, metric: str = 'MAE'):
        """Initialize the base metrics calculator.

        Args:
            grid_file: Path to grid file containing cell area and coordinates
            hemisphere: Hemisphere ('nh' for Northern or 'sh' for Southern)
            metric: Statistical metric for comparisons ('MAE', 'RMSE', etc.)
        """
        self.grid_file = grid_file
        self.hemisphere = hemisphere
        self.metric = metric

        if not os.path.exists(grid_file):
            raise FileNotFoundError(f"Grid file not found: {grid_file}")

        with open_netcdf_dataset(grid_file) as ds:
            self.lon = np.array(ds['lon'])
            self.lat = np.array(ds['lat'])
            self.cell_area = np.array(ds['cell_area'])

    # -----------------------------------------------------------------------------
    # Function: `resolve_key_months`.
    # Purpose: Handle resolve key months workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def resolve_key_months(hemisphere: str) -> List[int]:
        """Return hemisphere-aware key months for sea-ice extrema diagnostics."""
        h = (hemisphere or '').lower()
        return [3, 9] if h == 'nh' else [2, 9]

    # -----------------------------------------------------------------------------
    # Function: `month_tag`.
    # Purpose: Handle month tag workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def month_tag(month: int) -> str:
        """Return lowercase month tag used in file names and dictionary keys."""
        mapping = {
            1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
            7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
        }
        return mapping.get(int(month), f'm{int(month):02d}')

    # -----------------------------------------------------------------------------
    # Function: `month_label`.
    # Purpose: Handle month label workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def month_label(month: int) -> str:
        """Return title-case month label used in scalar table headers."""
        return SeaIceMetricsBase.month_tag(month).capitalize()

    # -----------------------------------------------------------------------------
    # Function: `_normalize_scalar`.
    # Purpose: Handle normalize scalar workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _normalize_scalar(value):
        """Convert NumPy scalar/bytes objects to native Python scalars."""
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode('utf-8')
        return value

    # -----------------------------------------------------------------------------
    # Function: `_to_jsonable`.
    # Purpose: Handle to jsonable workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _to_jsonable(value):
        """Recursively convert metric values to JSON-serializable objects."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): SeaIceMetricsBase._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [SeaIceMetricsBase._to_jsonable(v) for v in value]
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode('utf-8')
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    # -----------------------------------------------------------------------------
    # Function: `_is_linregress_like`.
    # Purpose: Handle is linregress like workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _is_linregress_like(value) -> bool:
        """Return True when *value* behaves like scipy.stats.linregress output."""
        required = ('slope', 'intercept', 'pvalue')
        return all(hasattr(value, k) for k in required)

    # -----------------------------------------------------------------------------
    # Function: `_coerce_numeric_array`.
    # Purpose: Handle coerce numeric array workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _coerce_numeric_array(value) -> Optional[np.ndarray]:
        """Return a numeric ndarray when conversion is safe, else None."""
        # scipy.stats.linregress returns a tuple-like object, but it should be
        # serialized as a structured trend payload (slope/pvalue/etc.), not as a
        # plain numeric vector.
        if SeaIceMetricsBase._is_linregress_like(value):
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
        else:
            return None

        if arr.dtype.kind in ('i', 'u', 'f', 'b'):
            return arr
        return None

    # -----------------------------------------------------------------------------
    # Function: `_sanitize_name_token`.
    # Purpose: Handle sanitize name token workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _sanitize_name_token(token: str) -> str:
        """Convert arbitrary key tokens to NetCDF-safe name fragments."""
        out = re.sub(r'[^0-9A-Za-z_]+', '_', str(token)).strip('_')
        if not out:
            out = 'item'
        if out[0].isdigit():
            out = f'k_{out}'
        return out

    # -----------------------------------------------------------------------------
    # Function: `_safe_dim_name`.
    # Purpose: Handle safe dim name workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _safe_dim_name(var_name: str, suffix: str) -> str:
        """Build a dimension name from one variable name with a fixed suffix."""
        return f"{SeaIceMetricsBase._sanitize_name_token(var_name)}_{SeaIceMetricsBase._sanitize_name_token(suffix)}"

    # -----------------------------------------------------------------------------
    # Function: `_linregress_fields`.
    # Purpose: Handle linregress fields workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _linregress_fields() -> Tuple[str, ...]:
        """Canonical field order for linregress-like payload vectors."""
        return (
            'slope',
            'intercept',
            'rvalue',
            'pvalue',
            'stderr',
            'intercept_stderr',
        )

    # -----------------------------------------------------------------------------
    # Function: `_linregress_to_vector`.
    # Purpose: Handle linregress to vector workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _linregress_to_vector(value) -> np.ndarray:
        """Convert one linregress-like object to a fixed-order numeric vector."""
        fields = SeaIceMetricsBase._linregress_fields()
        return np.asarray(
            [float(getattr(value, fld, np.nan)) for fld in fields],
            dtype=np.float64,
        )

    # -----------------------------------------------------------------------------
    # Function: `_vector_to_linregress`.
    # Purpose: Handle vector to linregress workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _vector_to_linregress(raw: object, fields: Tuple[str, ...]) -> SimpleNamespace:
        """Rebuild a linregress-like namespace from one cached numeric vector."""
        vec = np.asarray(raw, dtype=float).reshape(-1)
        payload: Dict[str, float] = {}
        for idx, fld in enumerate(fields):
            payload[fld] = float(vec[idx]) if idx < vec.size else np.nan
        return SimpleNamespace(**payload)

    # -----------------------------------------------------------------------------
    # Function: `_decode_object_path`.
    # Purpose: Handle decode object path workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _decode_object_path(attr_value: object) -> Optional[List[str]]:
        """Decode one object-path attribute into a list of dictionary keys."""
        if attr_value is None:
            return None
        text = str(attr_value).strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            if isinstance(payload, list) and payload:
                return [str(v) for v in payload]
        except Exception:
            return None
        return None

    # -----------------------------------------------------------------------------
    # Function: `_assign_path_value`.
    # Purpose: Handle assign path value workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _assign_path_value(root: Dict[str, object], path: List[str], value: object) -> None:
        """Assign one value into a nested dictionary path."""
        if not path:
            return
        cur: Dict[str, object] = root
        for seg in path[:-1]:
            child = cur.get(seg)
            if not isinstance(child, dict):
                child = {}
                cur[seg] = child
            cur = child
        cur[path[-1]] = value

    # -----------------------------------------------------------------------------
    # Function: `_numeric_to_dataarray`.
    # Purpose: Handle numeric to dataarray workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _numeric_to_dataarray(var_name: str,
                              logical_key: str,
                              arr: np.ndarray,
                              units: str,
                              long_name: str,
                              coords: Dict[str, xr.DataArray],
                              metric_context: Dict[str, object],
                              start_year: Optional[int],
                              end_year: Optional[int],
                              lon2d: Optional[np.ndarray] = None,
                              lat2d: Optional[np.ndarray] = None) -> xr.DataArray:
        """Serialize one numeric array/scalar into a CF-oriented DataArray."""
        k_lower = str(logical_key).lower()
        is_siconc_fraction_like = (
            arr.dtype.kind in ('i', 'u', 'f')
            and units == '%'
            and ('siconc_clim' in k_lower or k_lower == 'siconc')
            and all(tok not in k_lower for tok in ('ano', 'tr', 'std', 'diff'))
        )
        if is_siconc_fraction_like:
            finite = np.asarray(arr, dtype=float)
            if np.isfinite(finite).any() and np.nanmax(np.abs(finite)) <= 1.5:
                arr = finite * 100.0

        is_clim = SeaIceMetricsBase._is_climatology_key(logical_key)
        if arr.ndim >= 1 and is_clim:
            arr = SeaIceMetricsBase._normalize_climatology_array(
                logical_key, arr, metric_context, start_year=start_year, end_year=end_year
            )

        if arr.ndim == 0:
            da = xr.DataArray(SeaIceMetricsBase._normalize_scalar(arr))
            attrs = {
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'native',
            }
            fv = SeaIceMetricsBase._fill_value_for_array(np.asarray(arr))
            if fv is not None:
                attrs['_FillValue'] = fv
            da.attrs.update(attrs)
            return da

        if arr.ndim == 1 and k_lower in {'uni_mon', 'key_months'}:
            month_vals = SeaIceMetricsBase._sanitize_month_index_vector(
                arr, unique=(k_lower == 'uni_mon')
            )
            dim_name = SeaIceMetricsBase._safe_dim_name(var_name, 'index')
            da = xr.DataArray(month_vals.astype(np.int32, copy=False), dims=(dim_name,))
            da.attrs.update({
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'native',
            })
            return da

        if arr.ndim == 1 and is_clim and int(arr.shape[0]) == 12:
            if 'month' not in coords:
                coords['month'] = SeaIceMetricsBase._month_coord()
            da = xr.DataArray(arr, dims=('month',), coords={'month': coords['month']})
            da.attrs.update({
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'native',
            })
            fv = SeaIceMetricsBase._fill_value_for_array(arr)
            if fv is not None:
                da.attrs['_FillValue'] = fv
            return da

        if arr.ndim == 1:
            ntime = int(arr.shape[0])
            if 'time' not in coords or int(coords['time'].shape[0]) != ntime:
                coords['time'] = SeaIceMetricsBase._time_coord(
                    ntime, start_year=start_year, end_year=end_year, key=logical_key
                )
            da = xr.DataArray(arr, dims=('time',), coords={'time': coords['time']})
            da.attrs.update({
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'native',
            })
            fv = SeaIceMetricsBase._fill_value_for_array(arr)
            if fv is not None:
                da.attrs['_FillValue'] = fv
            return da

        if arr.ndim == 2 and k_lower in {'time_keys', 'yearmon_list'}:
            ntime, ncomp = int(arr.shape[0]), int(arr.shape[1])
            if 'time' not in coords or int(coords['time'].shape[0]) != ntime:
                coords['time'] = SeaIceMetricsBase._time_coord(
                    ntime, start_year=start_year, end_year=end_year, key=logical_key
                )
            comp_dim = SeaIceMetricsBase._safe_dim_name(var_name, 'component')
            comp_coord = xr.DataArray(np.arange(ncomp, dtype=np.int16), dims=(comp_dim,))
            comp_coord.attrs.update({
                'long_name': f'{k_lower}_component_index',
                'units': '1',
            })
            da = xr.DataArray(
                arr,
                dims=('time', comp_dim),
                coords={'time': coords['time'], comp_dim: comp_coord},
            )
            da.attrs.update({
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'native',
            })
            fv = SeaIceMetricsBase._fill_value_for_array(arr)
            if fv is not None:
                da.attrs['_FillValue'] = fv
            return da

        if arr.ndim == 2:
            ny, nx = int(arr.shape[0]), int(arr.shape[1])
            if 'y' not in coords or int(coords['y'].shape[0]) != ny or int(coords['x'].shape[0]) != nx:
                coords.update(SeaIceMetricsBase._xy_coords(ny, nx))
            da_coords: Dict[str, xr.DataArray] = {'y': coords['y'], 'x': coords['x']}
            if lon2d is not None and lat2d is not None and lon2d.shape == (ny, nx):
                da_coords['lon'] = xr.DataArray(lon2d, dims=('y', 'x'))
                da_coords['lat'] = xr.DataArray(lat2d, dims=('y', 'x'))
            da = xr.DataArray(arr, dims=('y', 'x'), coords=da_coords)
            da.attrs.update({
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'native',
            })
            fv = SeaIceMetricsBase._fill_value_for_array(arr)
            if fv is not None:
                da.attrs['_FillValue'] = fv
            return da

        if arr.ndim == 3 and is_clim and int(arr.shape[0]) == 12:
            nmon, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
            if 'month' not in coords or int(coords['month'].shape[0]) != nmon:
                coords['month'] = SeaIceMetricsBase._month_coord()
            if 'y' not in coords or int(coords['y'].shape[0]) != ny or int(coords['x'].shape[0]) != nx:
                coords.update(SeaIceMetricsBase._xy_coords(ny, nx))
            da_coords = {'month': coords['month'], 'y': coords['y'], 'x': coords['x']}
            if lon2d is not None and lat2d is not None and lon2d.shape == (ny, nx):
                da_coords['lon'] = xr.DataArray(lon2d, dims=('y', 'x'))
                da_coords['lat'] = xr.DataArray(lat2d, dims=('y', 'x'))
            da = xr.DataArray(arr, dims=('month', 'y', 'x'), coords=da_coords)
            da.attrs.update({
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'native',
            })
            fv = SeaIceMetricsBase._fill_value_for_array(arr)
            if fv is not None:
                da.attrs['_FillValue'] = fv
            return da

        if arr.ndim == 3:
            ntime, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
            if 'time' not in coords or int(coords['time'].shape[0]) != ntime:
                coords['time'] = SeaIceMetricsBase._time_coord(
                    ntime, start_year=start_year, end_year=end_year, key=logical_key
                )
            if 'y' not in coords or int(coords['y'].shape[0]) != ny or int(coords['x'].shape[0]) != nx:
                coords.update(SeaIceMetricsBase._xy_coords(ny, nx))
            da_coords = {'time': coords['time'], 'y': coords['y'], 'x': coords['x']}
            if lon2d is not None and lat2d is not None and lon2d.shape == (ny, nx):
                da_coords['lon'] = xr.DataArray(lon2d, dims=('y', 'x'))
                da_coords['lat'] = xr.DataArray(lat2d, dims=('y', 'x'))
            da = xr.DataArray(arr, dims=('time', 'y', 'x'), coords=da_coords)
            da.attrs.update({
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'native',
            })
            fv = SeaIceMetricsBase._fill_value_for_array(arr)
            if fv is not None:
                da.attrs['_FillValue'] = fv
            return da

        dims = tuple(SeaIceMetricsBase._safe_dim_name(var_name, f'dim{i}') for i in range(arr.ndim))
        da = xr.DataArray(arr, dims=dims)
        da.attrs.update({
            'units': units,
            'long_name': long_name,
            'sitool_encoding': 'native',
        })
        fv = SeaIceMetricsBase._fill_value_for_array(arr)
        if fv is not None:
            da.attrs['_FillValue'] = fv
        return da

    # -----------------------------------------------------------------------------
    # Function: `_fill_value_for_array`.
    # Purpose: Handle fill value for array workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _fill_value_for_array(arr: np.ndarray):
        """Return a CF-style fill value matching one array dtype."""
        if arr.dtype.kind in ('f', 'c'):
            return np.nan
        if arr.dtype.kind in ('i', 'u'):
            return -9999
        if arr.dtype.kind == 'b':
            return 255
        return None

    # -----------------------------------------------------------------------------
    # Function: `_month_coord`.
    # Purpose: Handle month coord workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _month_coord() -> xr.DataArray:
        """Create month coordinate [1..12] for climatology variables."""
        da = xr.DataArray(np.arange(1, 13, dtype=np.int16), dims=('month',))
        da.attrs.update({'long_name': 'calendar_month', 'units': '1'})
        return da

    # -----------------------------------------------------------------------------
    # Function: `_infer_key_month`.
    # Purpose: Handle infer key month workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _infer_key_month(key: str) -> Optional[int]:
        """Infer a fixed month from key suffix like *_mar, *_sep."""
        k = str(key or '').lower()
        tags = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        }
        for tag, month in tags.items():
            if k.endswith(f'_{tag}'):
                return month
        return None

    # -----------------------------------------------------------------------------
    # Function: `_time_coord`.
    # Purpose: Handle time coord workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _time_coord(length: int, start_year: Optional[int] = None,
                    end_year: Optional[int] = None,
                    key: str = '') -> xr.DataArray:
        """Create monthly CF-compliant time coordinate."""
        n = int(length)
        if n <= 0:
            vals = np.array([], dtype='datetime64[ns]')
        else:
            start_y = int(start_year) if start_year is not None else 2000
            key_month = SeaIceMetricsBase._infer_key_month(key)
            if key_month is not None:
                start = f'{start_y:04d}-{key_month:02d}-01'
                idx = pd.date_range(start=start, periods=n, freq='12MS')
            elif start_year is not None and end_year is not None:
                n_expected = (int(end_year) - int(start_year) + 1) * 12
                if n == n_expected:
                    start = f'{int(start_year):04d}-01-01'
                    idx = pd.date_range(start=start, periods=n, freq='MS')
                else:
                    start = f'{start_y:04d}-01-01'
                    idx = pd.date_range(start=start, periods=n, freq='MS')
            else:
                start = f'{start_y:04d}-01-01'
                idx = pd.date_range(start=start, periods=n, freq='MS')
            vals = idx.to_numpy(dtype='datetime64[ns]')

        da = xr.DataArray(vals, dims=('time',))
        da.attrs.update({
            'standard_name': 'time',
            'long_name': 'time',
        })
        return da

    # -----------------------------------------------------------------------------
    # Function: `_xy_coords`.
    # Purpose: Handle xy coords workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _xy_coords(ny: int, nx: int) -> Dict[str, xr.DataArray]:
        """Create index-based x/y coordinates."""
        y = xr.DataArray(np.arange(ny, dtype=np.int32), dims=('y',))
        x = xr.DataArray(np.arange(nx, dtype=np.int32), dims=('x',))
        y.attrs.update({'long_name': 'y_index', 'units': '1'})
        x.attrs.update({'long_name': 'x_index', 'units': '1'})
        return {'y': y, 'x': x}

    # -----------------------------------------------------------------------------
    # Function: `_is_climatology_key`.
    # Purpose: Handle is climatology key workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _is_climatology_key(key: str) -> bool:
        """Return True for climatology-style metric keys."""
        k = str(key or '').lower()
        return 'clim' in k

    # -----------------------------------------------------------------------------
    # Function: `_sanitize_month_index_vector`.
    # Purpose: Handle sanitize month index vector workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _sanitize_month_index_vector(values: object, *, unique: bool = False) -> np.ndarray:
        """Normalize month-index vectors to finite integers in [1, 12]."""
        try:
            arr = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return np.array([], dtype=np.int32)
        if arr.size <= 0:
            return np.array([], dtype=np.int32)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 0:
            return np.array([], dtype=np.int32)
        months = np.rint(arr).astype(np.int32, copy=False)
        months = months[(months >= 1) & (months <= 12)]
        if months.size <= 0:
            return np.array([], dtype=np.int32)
        if unique:
            out = []
            seen = set()
            for mm in months.tolist():
                key = int(mm)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
            return np.asarray(out, dtype=np.int32)
        return months.astype(np.int32, copy=False)

    # -----------------------------------------------------------------------------
    # Function: `_normalize_climatology_array`.
    # Purpose: Handle normalize climatology array workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _normalize_climatology_array(key: str, arr: np.ndarray,
                                     metric_dict: Dict[str, object],
                                     start_year: Optional[int] = None,
                                     end_year: Optional[int] = None) -> np.ndarray:
        """Normalize climatology arrays to 12 calendar months when possible.

        Priority:
          1. Use explicit ``uni_mon`` month indices when provided.
          2. Use CF-style monthly dates and ``groupby('time.month')`` averaging.
          3. Fall back to reshape-by-12 averaging when the leading axis is divisible by 12.
        """
        if not SeaIceMetricsBase._is_climatology_key(key) or arr.ndim == 0:
            return arr
        # Apply only to arrays with a leading temporal axis (1-D cycles or 3-D time-lat-lon fields).
        if arr.ndim not in (1, 3):
            return arr
        n0 = int(arr.shape[0])
        if n0 == 12:
            return arr

        uni_mon = metric_dict.get('uni_mon')
        if uni_mon is not None:
            months = np.asarray(uni_mon).astype(int).ravel()
            if months.size == n0:
                out_shape = (12,) + arr.shape[1:]
                out = np.full(out_shape, np.nan, dtype=float)
                arrf = np.asarray(arr, dtype=float)
                for ii, mm in enumerate(months):
                    if 1 <= int(mm) <= 12:
                        out[int(mm) - 1] = arrf[ii]
                return out

        if n0 >= 12:
            try:
                base_year = int(start_year) if start_year is not None else 2000
                idx = pd.date_range(start=f'{base_year:04d}-01-01', periods=n0, freq='MS')
                if arr.ndim == 1:
                    da = xr.DataArray(np.asarray(arr, dtype=float), dims=('time',), coords={'time': idx})
                    grouped = da.groupby('time.month').mean('time', skipna=True)
                    grouped = grouped.reindex(month=np.arange(1, 13, dtype=int))
                    return np.asarray(grouped.values, dtype=float)
                if arr.ndim == 3:
                    da = xr.DataArray(
                        np.asarray(arr, dtype=float),
                        dims=('time', 'y', 'x'),
                        coords={'time': idx},
                    )
                    grouped = da.groupby('time.month').mean('time', skipna=True)
                    grouped = grouped.reindex(month=np.arange(1, 13, dtype=int))
                    return np.asarray(grouped.values, dtype=float)
            except Exception:
                pass

        if n0 % 12 == 0:
            try:
                reshaped = np.asarray(arr, dtype=float).reshape((-1, 12) + arr.shape[1:])
                return np.nanmean(reshaped, axis=0)
            except Exception:
                return arr

        return arr

    # -----------------------------------------------------------------------------
    # Function: `to_xarray`.
    # Purpose: Handle to xarray workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def to_xarray(metric_dict: Dict[str, object],
                  units_map: Optional[Dict[str, str]] = None,
                  long_name_map: Optional[Dict[str, str]] = None,
                  grid_coords: Optional[Dict[str, np.ndarray]] = None,
                  start_year: Optional[int] = None,
                  end_year: Optional[int] = None) -> xr.Dataset:
        """Serialize one metric dictionary into one CF-oriented xarray Dataset.

        The serializer keeps the original metric keys as variable names, but
        normalizes dimensions for plotting-friendly NetCDF output:
          - 1-D time-series arrays use ``time`` (monthly dates).
          - 1-D climatologies use ``month`` (1..12).
          - 2-D arrays use ``y``/``x`` (+ optional 2-D lat/lon coords).
          - 3-D climatologies use ``month``/``y``/``x``.
          - 3-D time-series arrays use ``time``/``y``/``x``.
        """
        units_map = units_map or {}
        long_name_map = long_name_map or {}
        ds_vars: Dict[str, xr.DataArray] = {}
        coords: Dict[str, xr.DataArray] = {}
        used_var_names: set = set()
        lon2d = None
        lat2d = None
        if isinstance(grid_coords, dict):
            lon2d = grid_coords.get('lon')
            lat2d = grid_coords.get('lat')
            if not (isinstance(lon2d, np.ndarray) and isinstance(lat2d, np.ndarray)):
                lon2d = lat2d = None
            elif lon2d.shape != lat2d.shape or lon2d.ndim != 2:
                lon2d = lat2d = None

        entries: List[Dict[str, object]] = []

        # -----------------------------------------------------------------------------
        # Function: `_append_entry`.
        # Purpose: Handle append entry workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _append_entry(path: List[str],
                          kind: str,
                          value_obj: object,
                          units_obj: str,
                          long_name_obj: str,
                          context_obj: Dict[str, object]) -> None:
            entries.append({
                'path': list(path),
                'logical_key': str(path[-1]) if path else '',
                'kind': kind,
                'value': value_obj,
                'units': units_obj,
                'long_name': long_name_obj,
                'context': context_obj,
            })

        # -----------------------------------------------------------------------------
        # Function: `_walk_value`.
        # Purpose: Handle walk value workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _walk_value(top_key: str,
                        path: List[str],
                        value_obj: object,
                        units_obj: str,
                        long_name_obj: str,
                        context_obj: Dict[str, object]) -> None:
            if value_obj is None:
                return

            if isinstance(value_obj, dict):
                if not value_obj:
                    _append_entry(path, 'empty_dict', {}, units_obj, long_name_obj, context_obj)
                    return
                for sub_key, sub_val in value_obj.items():
                    sub_name = str(sub_key)
                    _walk_value(
                        top_key=top_key,
                        path=path + [sub_name],
                        value_obj=sub_val,
                        units_obj=units_map.get(sub_name, units_obj),
                        long_name_obj=f'{long_name_obj}.{sub_name}',
                        context_obj=value_obj,
                    )
                return

            if SeaIceMetricsBase._is_linregress_like(value_obj):
                _append_entry(path, 'linregress', value_obj, units_obj, long_name_obj, context_obj)
                return

            arr_obj = SeaIceMetricsBase._coerce_numeric_array(value_obj)
            if arr_obj is not None:
                _append_entry(path, 'numeric', arr_obj, units_obj, long_name_obj, context_obj)
                return

            if np.isscalar(value_obj):
                scalar_obj = SeaIceMetricsBase._normalize_scalar(value_obj)
                if isinstance(scalar_obj, (str, bytes, np.bytes_)):
                    _append_entry(path, 'string_code', str(scalar_obj), units_obj, long_name_obj, context_obj)
                else:
                    _append_entry(path, 'numeric', np.asarray(scalar_obj), units_obj, long_name_obj, context_obj)
                return

            if isinstance(value_obj, (list, tuple)):
                arr_obj = SeaIceMetricsBase._coerce_numeric_array(value_obj)
                if arr_obj is not None:
                    _append_entry(path, 'numeric', arr_obj, units_obj, long_name_obj, context_obj)
                    return
                payload = json.dumps(SeaIceMetricsBase._to_jsonable(value_obj), ensure_ascii=False)
                _append_entry(path, 'string_json', payload, units_obj, long_name_obj, context_obj)
                return

            _append_entry(path, 'string_code', str(value_obj), units_obj, long_name_obj, context_obj)

        for key, value in metric_dict.items():
            if value is None:
                continue
            _walk_value(
                top_key=str(key),
                path=[str(key)],
                value_obj=value,
                units_obj=units_map.get(str(key), 'unknown'),
                long_name_obj=long_name_map.get(str(key), str(key)),
                context_obj=metric_dict,
            )

        # -----------------------------------------------------------------------------
        # Function: `_make_var_name`.
        # Purpose: Handle make var name workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _make_var_name(path: List[str]) -> str:
            if len(path) == 1:
                base = str(path[0])
            else:
                toks = [SeaIceMetricsBase._sanitize_name_token(v) for v in path]
                base = f"obj__{'__'.join(toks)}"
            name = base
            idx = 2
            while name in used_var_names:
                name = f'{base}_{idx}'
                idx += 1
            used_var_names.add(name)
            return name

        for ent in entries:
            path = ent['path']
            if not isinstance(path, list) or not path:
                continue
            var_name = _make_var_name(path)
            logical_key = str(ent.get('logical_key', path[-1]))
            units = str(ent.get('units', 'unknown'))
            long_name = str(ent.get('long_name', logical_key))
            kind = str(ent.get('kind', 'numeric'))
            context = ent.get('context')
            if not isinstance(context, dict):
                context = metric_dict
            object_path_attr = json.dumps(path, ensure_ascii=False) if len(path) > 1 else None

            if kind == 'numeric':
                arr = np.asarray(ent.get('value'))
                da = SeaIceMetricsBase._numeric_to_dataarray(
                    var_name=var_name,
                    logical_key=logical_key,
                    arr=arr,
                    units=units,
                    long_name=long_name,
                    coords=coords,
                    metric_context=context,
                    start_year=start_year,
                    end_year=end_year,
                    lon2d=lon2d,
                    lat2d=lat2d,
                )
            elif kind == 'linregress':
                fields = SeaIceMetricsBase._linregress_fields()
                vec = SeaIceMetricsBase._linregress_to_vector(ent.get('value'))
                dim_name = SeaIceMetricsBase._safe_dim_name(var_name, 'linregress_component')
                dim_coord = xr.DataArray(np.arange(vec.size, dtype=np.int16), dims=(dim_name,))
                dim_coord.attrs.update({'long_name': 'linregress_component_index', 'units': '1'})
                da = xr.DataArray(vec, dims=(dim_name,), coords={dim_name: dim_coord})
                da.attrs.update({
                    'units': units,
                    'long_name': long_name,
                    'sitool_encoding': 'linregress_vector',
                    'sitool_linregress_fields': json.dumps(list(fields)),
                })
            elif kind == 'empty_dict':
                da = xr.DataArray(np.asarray(1, dtype=np.int8))
                da.attrs.update({
                    'units': units,
                    'long_name': long_name,
                    'sitool_encoding': 'object_empty_dict',
                })
            else:
                payload_text = str(ent.get('value', ''))
                da = xr.DataArray(np.asarray(0, dtype=np.int32))
                da.attrs.update({
                    'units': units,
                    'long_name': long_name,
                    'sitool_encoding': 'string_code',
                    'sitool_string_values': json.dumps([payload_text], ensure_ascii=False),
                })
                if kind == 'string_json':
                    da.attrs['sitool_string_json'] = '1'

            if object_path_attr is not None:
                da.attrs['sitool_object_path'] = object_path_attr
            ds_vars[var_name] = da

        ds = xr.Dataset(ds_vars)
        if 'time' in ds.coords:
            ds['time'].attrs.update({
                'standard_name': 'time',
                'long_name': 'time',
            })
        if 'month' in ds.coords:
            ds['month'].attrs.update({'long_name': 'calendar_month', 'units': '1'})
        if 'lat' in ds.coords:
            ds['lat'].attrs.update({
                'standard_name': 'latitude',
                'long_name': 'latitude',
                'units': 'degrees_north',
            })
        if 'lon' in ds.coords:
            ds['lon'].attrs.update({
                'standard_name': 'longitude',
                'long_name': 'longitude',
                'units': 'degrees_east',
            })
        return ds

    # -----------------------------------------------------------------------------
    # Function: `save_to_nc`.
    # Purpose: Handle save to nc workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def save_to_nc(output_file: str,
                   metric_dict: Dict[str, object],
                   group: Optional[str] = None,
                   mode: str = 'a',
                   engines: Tuple[Optional[str], ...] = ('netcdf4', 'h5netcdf', None),
                   units_map: Optional[Dict[str, str]] = None,
                   long_name_map: Optional[Dict[str, str]] = None,
                   grid_coords: Optional[Dict[str, np.ndarray]] = None,
                   start_year: Optional[int] = None,
                   end_year: Optional[int] = None) -> None:
        """Serialize and save one metric dictionary into a NetCDF file/group.

        String/object variables are always written with ``_FillValue=None`` to
        avoid netCDF4 variable-length string encoding errors.
        """
        ds = SeaIceMetricsBase.to_xarray(
            metric_dict=metric_dict,
            units_map=units_map,
            long_name_map=long_name_map,
            grid_coords=grid_coords,
            start_year=start_year,
            end_year=end_year,
        )
        ds_out = ds.copy(deep=False)

        encoding: Dict[str, Dict[str, object]] = {}
        for var_name, da in ds_out.data_vars.items():
            kind = da.dtype.kind
            fv = da.attrs.pop('_FillValue', None)
            if kind in ('U', 'S', 'O'):
                encoding[var_name] = {'_FillValue': None}
            else:
                var_enc: Dict[str, object] = {
                    'zlib': True,
                    'complevel': int(SITOOL_NC_COMPRESS_LEVEL),
                    'shuffle': bool(SITOOL_NC_SHUFFLE),
                }
                if fv is not None and not isinstance(fv, (str, bytes)):
                    var_enc['_FillValue'] = fv
                encoding[var_name] = var_enc

        out_dir = os.path.dirname(os.path.abspath(str(output_file)))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # -----------------------------------------------------------------------------
        # Function: `_retryable`.
        # Purpose: Handle retryable workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _retryable(exc: Exception) -> bool:
            text = str(exc).lower()
            if isinstance(exc, (PermissionError, BlockingIOError)):
                return True
            return any(
                tok in text for tok in (
                    'permission denied',
                    'access is denied',
                    'resource temporarily unavailable',
                    'file is already open',
                    'unable to open file',
                    'hdf error',
                    'locking',
                )
            )

        modes = [mode]
        if mode == 'a' and not os.path.exists(str(output_file)):
            modes = ['w', 'a']

        errors: List[str] = []
        for write_mode in modes:
            for attempt in range(4):
                attempt_errors: List[str] = []
                retryable_hit = False
                for engine in engines:
                    try:
                        kwargs: Dict[str, object] = {'mode': write_mode}
                        if group is not None:
                            kwargs['group'] = group
                        if engine is not None:
                            kwargs['engine'] = engine
                        if encoding:
                            kwargs['encoding'] = encoding
                        ds_out.to_netcdf(output_file, **kwargs)
                        return
                    except Exception as exc:
                        attempt_errors.append(f'mode={write_mode}, {engine or "default"}: {exc}')
                        retryable_hit = retryable_hit or _retryable(exc)
                errors.extend(attempt_errors)
                if (not retryable_hit) or attempt >= 3:
                    break
                time.sleep(0.25 * (attempt + 1))

        raise OSError(
            f'Unable to write NetCDF {output_file}::{group}. '
            f'Tried: {" | ".join(errors)}'
        )

    # -----------------------------------------------------------------------------
    # Function: `from_xarray`.
    # Purpose: Handle from xarray workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def from_xarray(ds: xr.Dataset) -> Dict[str, object]:
        """Rebuild a metric dictionary from cached xarray datasets."""
        metric: Dict[str, object] = {}
        if 'time' in ds.coords:
            time_coord = None
            try:
                # Cache files are often opened with decode_times=False in main.py.
                # Decode here so plotting can recover real datetime axes.
                if np.issubdtype(ds['time'].dtype, np.datetime64):
                    time_coord = np.asarray(ds['time'].values)
                else:
                    tds = ds['time'].to_dataset(name='time_coord')
                    tds = xr.decode_cf(tds)
                    time_coord = np.asarray(tds['time_coord'].values)
            except Exception:
                try:
                    time_coord = np.asarray(ds['time'].values)
                except Exception:
                    time_coord = None
            if time_coord is not None:
                metric['time_coord'] = time_coord

        for key, da in ds.data_vars.items():
            encoding = da.attrs.get('sitool_encoding', da.attrs.get('encoding', 'native'))
            values = da.values
            if isinstance(values, np.ndarray) and values.ndim == 0:
                raw = values.item()
            else:
                raw = values

            decoded_value: object
            if encoding == 'json':
                text = SeaIceMetricsBase._normalize_scalar(raw)
                try:
                    payload = json.loads(text)
                except Exception:
                    decoded_value = text
                else:
                    p_type = payload.get('type')
                    p_value = payload.get('value')
                    if p_type == 'linregress' and isinstance(p_value, dict):
                        decoded_value = SimpleNamespace(**p_value)
                    else:
                        decoded_value = p_value
            elif encoding == 'linregress_vector':
                fields_attr = da.attrs.get('sitool_linregress_fields')
                fields = SeaIceMetricsBase._linregress_fields()
                if fields_attr is not None:
                    try:
                        parsed = json.loads(str(fields_attr))
                        if isinstance(parsed, list) and parsed:
                            fields = tuple(str(v) for v in parsed)
                    except Exception:
                        pass
                decoded_value = SeaIceMetricsBase._vector_to_linregress(raw, fields)
            elif encoding == 'string_code':
                labels_attr = da.attrs.get('sitool_string_values', '[]')
                labels: List[str] = []
                try:
                    parsed = json.loads(str(labels_attr))
                    if isinstance(parsed, list):
                        labels = [str(v) for v in parsed]
                except Exception:
                    labels = []
                idx = 0
                try:
                    idx = int(np.asarray(raw).reshape(()))
                except Exception:
                    idx = 0
                if 0 <= idx < len(labels):
                    decoded_value = labels[idx]
                elif labels:
                    decoded_value = labels[0]
                else:
                    decoded_value = ''
                if str(da.attrs.get('sitool_string_json', '')).strip() == '1':
                    try:
                        decoded_value = json.loads(str(decoded_value))
                    except Exception:
                        pass
            elif encoding == 'object_empty_dict':
                decoded_value = {}
            elif encoding == 'object_empty_list':
                decoded_value = []
            else:
                semantic_key = str(key)
                path = SeaIceMetricsBase._decode_object_path(da.attrs.get('sitool_object_path'))
                if path:
                    semantic_key = str(path[-1])
                if isinstance(raw, np.ndarray) and raw.ndim > 0:
                    k_lower = semantic_key.lower()
                    if k_lower in {'uni_mon', 'key_months'}:
                        decoded_value = SeaIceMetricsBase._sanitize_month_index_vector(
                            raw, unique=(k_lower == 'uni_mon')
                        )
                    elif k_lower in {'time_keys', 'yearmon_list'} and raw.ndim == 2:
                        out_rows: List[Tuple[int, ...]] = []
                        arr_rows = np.asarray(raw)
                        for row in arr_rows:
                            try:
                                vals = tuple(int(v) for v in np.asarray(row).reshape(-1).tolist())
                            except Exception:
                                continue
                            out_rows.append(vals)
                        decoded_value = out_rows
                    else:
                        decoded_value = np.asarray(raw)
                else:
                    decoded_value = SeaIceMetricsBase._normalize_scalar(raw)

            object_path = SeaIceMetricsBase._decode_object_path(da.attrs.get('sitool_object_path'))
            if object_path:
                SeaIceMetricsBase._assign_path_value(metric, object_path, decoded_value)
            else:
                metric[key] = decoded_value

        return metric

    # -----------------------------------------------------------------------------
    # Function: `valid_data_mask`.
    # Purpose: Handle valid data mask workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def valid_data_mask(data: np.ndarray, positive_only: bool = False) -> np.ndarray:
        """Return a boolean valid-data mask for scalar fields."""
        mask = np.isfinite(data)
        if positive_only:
            mask &= (data > 0)
        return mask

    # -----------------------------------------------------------------------------
    # Function: `vector_valid_data_mask`.
    # Purpose: Handle vector valid data mask workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def vector_valid_data_mask(u: np.ndarray, v: np.ndarray, positive_speed: bool = False) -> np.ndarray:
        """Return a boolean valid-data mask for vector fields."""
        mask = np.isfinite(u) & np.isfinite(v)
        if positive_speed:
            mask &= ((u != 0) | (v != 0))
        return mask
