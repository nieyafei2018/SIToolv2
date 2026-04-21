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
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from scripts import SITOOL_NC_COMPRESS_LEVEL, SITOOL_NC_SHUFFLE

logger = logging.getLogger(__name__)

# ------------------------------------------------------


"""

For *_2M_metrics() function, the var1 is recommended to be the observations,
    the var2 is recommended to be the model results.

"""


class SeaIceMetricsBase:
    """Base class for sea ice metrics calculations."""

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

        with xr.open_dataset(grid_file) as ds:
            self.lon = np.array(ds['lon'])
            self.lat = np.array(ds['lat'])
            self.cell_area = np.array(ds['cell_area'])

    @staticmethod
    def resolve_key_months(hemisphere: str) -> List[int]:
        """Return hemisphere-aware key months for sea-ice extrema diagnostics."""
        h = (hemisphere or '').lower()
        return [3, 9] if h == 'nh' else [2, 9]

    @staticmethod
    def month_tag(month: int) -> str:
        """Return lowercase month tag used in file names and dictionary keys."""
        mapping = {
            1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
            7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
        }
        return mapping.get(int(month), f'm{int(month):02d}')

    @staticmethod
    def month_label(month: int) -> str:
        """Return title-case month label used in scalar table headers."""
        return SeaIceMetricsBase.month_tag(month).capitalize()

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

    @staticmethod
    def _is_linregress_like(value) -> bool:
        """Return True when *value* behaves like scipy.stats.linregress output."""
        required = ('slope', 'intercept', 'pvalue')
        return all(hasattr(value, k) for k in required)

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

    @staticmethod
    def _month_coord() -> xr.DataArray:
        """Create month coordinate [1..12] for climatology variables."""
        da = xr.DataArray(np.arange(1, 13, dtype=np.int16), dims=('month',))
        da.attrs.update({'long_name': 'calendar_month', 'units': '1'})
        return da

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

    @staticmethod
    def _xy_coords(ny: int, nx: int) -> Dict[str, xr.DataArray]:
        """Create index-based x/y coordinates."""
        y = xr.DataArray(np.arange(ny, dtype=np.int32), dims=('y',))
        x = xr.DataArray(np.arange(nx, dtype=np.int32), dims=('x',))
        y.attrs.update({'long_name': 'y_index', 'units': '1'})
        x.attrs.update({'long_name': 'x_index', 'units': '1'})
        return {'y': y, 'x': x}

    @staticmethod
    def _is_climatology_key(key: str) -> bool:
        """Return True for climatology-style metric keys."""
        k = str(key or '').lower()
        return 'clim' in k

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
        lon2d = None
        lat2d = None
        if isinstance(grid_coords, dict):
            lon2d = grid_coords.get('lon')
            lat2d = grid_coords.get('lat')
            if not (isinstance(lon2d, np.ndarray) and isinstance(lat2d, np.ndarray)):
                lon2d = lat2d = None
            elif lon2d.shape != lat2d.shape or lon2d.ndim != 2:
                lon2d = lat2d = None

        for key, value in metric_dict.items():
            if value is None:
                continue

            units = units_map.get(key, 'unknown')
            long_name = long_name_map.get(key, key)

            arr = SeaIceMetricsBase._coerce_numeric_array(value)
            if arr is not None:
                k_lower = key.lower()
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

                is_clim = SeaIceMetricsBase._is_climatology_key(key)
                if arr.ndim >= 1 and is_clim:
                    arr = SeaIceMetricsBase._normalize_climatology_array(
                        key, arr, metric_dict, start_year=start_year, end_year=end_year
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
                    ds_vars[key] = da
                elif arr.ndim == 1 and k_lower in {'uni_mon', 'key_months'}:
                    month_vals = SeaIceMetricsBase._sanitize_month_index_vector(
                        arr, unique=(k_lower == 'uni_mon')
                    )
                    dim_name = f'{k_lower}_index'
                    da = xr.DataArray(month_vals.astype(np.int32, copy=False), dims=(dim_name,))
                    da.attrs.update({
                        'units': units,
                        'long_name': long_name,
                        'sitool_encoding': 'native',
                    })
                    ds_vars[key] = da
                elif arr.ndim == 1 and is_clim and int(arr.shape[0]) == 12:
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
                    ds_vars[key] = da
                elif arr.ndim == 1:
                    ntime = int(arr.shape[0])
                    if 'time' not in coords or int(coords['time'].shape[0]) != ntime:
                        coords['time'] = SeaIceMetricsBase._time_coord(
                            ntime, start_year=start_year, end_year=end_year, key=key
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
                    ds_vars[key] = da
                elif arr.ndim == 2:
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
                    ds_vars[key] = da
                elif arr.ndim == 3 and is_clim and int(arr.shape[0]) == 12:
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
                    ds_vars[key] = da
                elif arr.ndim == 3:
                    ntime, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
                    if 'time' not in coords or int(coords['time'].shape[0]) != ntime:
                        coords['time'] = SeaIceMetricsBase._time_coord(
                            ntime, start_year=start_year, end_year=end_year, key=key
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
                    ds_vars[key] = da
                else:
                    dims = tuple(f'{key}_dim{i}' for i in range(arr.ndim))
                    da = xr.DataArray(arr, dims=dims)
                    da.attrs.update({
                        'units': units,
                        'long_name': long_name,
                        'sitool_encoding': 'native',
                    })
                    fv = SeaIceMetricsBase._fill_value_for_array(arr)
                    if fv is not None:
                        da.attrs['_FillValue'] = fv
                    ds_vars[key] = da
                continue

            if np.isscalar(value):
                scalar_value = SeaIceMetricsBase._normalize_scalar(value)
                da = xr.DataArray(scalar_value)
                attrs = {
                    'units': units,
                    'long_name': long_name,
                    'sitool_encoding': 'native',
                }
                fv = SeaIceMetricsBase._fill_value_for_array(np.asarray(scalar_value))
                if fv is not None:
                    attrs['_FillValue'] = fv
                da.attrs.update(attrs)
                ds_vars[key] = da
                continue

            if SeaIceMetricsBase._is_linregress_like(value):
                payload = {
                    'type': 'linregress',
                    'value': {
                        'slope': float(getattr(value, 'slope', np.nan)),
                        'intercept': float(getattr(value, 'intercept', np.nan)),
                        'rvalue': float(getattr(value, 'rvalue', np.nan)),
                        'pvalue': float(getattr(value, 'pvalue', np.nan)),
                        'stderr': float(getattr(value, 'stderr', np.nan)),
                        'intercept_stderr': float(getattr(value, 'intercept_stderr', np.nan)),
                    },
                }
            elif isinstance(value, dict):
                payload = {'type': 'dict', 'value': SeaIceMetricsBase._to_jsonable(value)}
            elif isinstance(value, (list, tuple)):
                payload = {'type': 'sequence', 'value': SeaIceMetricsBase._to_jsonable(value)}
            else:
                payload = {'type': 'fallback', 'value': SeaIceMetricsBase._to_jsonable(value)}

            da = xr.DataArray(np.asarray(json.dumps(payload), dtype=str))
            da.attrs.update({
                'units': units,
                'long_name': long_name,
                'sitool_encoding': 'json',
            })
            ds_vars[key] = da

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

            if encoding == 'json':
                text = SeaIceMetricsBase._normalize_scalar(raw)
                try:
                    payload = json.loads(text)
                except Exception:
                    metric[key] = text
                    continue

                p_type = payload.get('type')
                p_value = payload.get('value')
                if p_type == 'linregress' and isinstance(p_value, dict):
                    metric[key] = SimpleNamespace(**p_value)
                else:
                    metric[key] = p_value
            else:
                if isinstance(raw, np.ndarray) and raw.ndim > 0:
                    k_lower = str(key).lower()
                    if k_lower in {'uni_mon', 'key_months'}:
                        metric[key] = SeaIceMetricsBase._sanitize_month_index_vector(
                            raw, unique=(k_lower == 'uni_mon')
                        )
                    else:
                        metric[key] = np.asarray(raw)
                else:
                    metric[key] = SeaIceMetricsBase._normalize_scalar(raw)

        return metric

    @staticmethod
    def valid_data_mask(data: np.ndarray, positive_only: bool = False) -> np.ndarray:
        """Return a boolean valid-data mask for scalar fields."""
        mask = np.isfinite(data)
        if positive_only:
            mask &= (data > 0)
        return mask

    @staticmethod
    def vector_valid_data_mask(u: np.ndarray, v: np.ndarray, positive_speed: bool = False) -> np.ndarray:
        """Return a boolean valid-data mask for vector fields."""
        mask = np.isfinite(u) & np.isfinite(v)
        if positive_speed:
            mask &= ((u != 0) | (v != 0))
        return mask
