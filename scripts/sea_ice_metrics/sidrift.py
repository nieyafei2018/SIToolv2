# -*- coding: utf-8 -*-
"""Sea-ice metric implementations by diagnostic family."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from scripts import utils
from scripts.config import DAYS_PER_MONTH
from scripts.sea_ice_metrics.base import SeaIceMetricsBase

logger = logging.getLogger(__name__)


class SIDMetrics(SeaIceMetricsBase):
    """
    Calculate sea ice drift metrics including velocity and vector correlation.

    All calculations are performed in the x-y direction.
    """

    def __init__(self, grid_file: str, hemisphere: str, time_sta: int, time_end: int,
                 metric: str = 'MAE', projection: str = 'stere'):
        """
        Initialize the sea ice drift metrics calculator with Cartopy.

        Args:
            grid_file: Path to grid file containing cell area and coordinates.
            hemisphere: Hemisphere ('nh' for Northern or 'sh' for Southern).
            time_sta: Start year for analysis.
            time_end: End year for analysis.
            metric: Statistical metric for comparisons.
            projection: Map projection ('stere' for stereographic, 'laea' for equal area).
        """
        super().__init__(grid_file, hemisphere, metric)
        self.time_sta = time_sta
        self.time_end = time_end
        self.projection = projection

        # Initialize Cartopy projection
        self._init_cartopy_projection()

    def _init_cartopy_projection(self):
        """
        Initialize Cartopy projection based on hemisphere and projection type.
        """
        cl, tsl = (-90, -70) if self.hemisphere == 'sh' else (90, 70)

        if self.projection == 'stere':
            self.proj = ccrs.Stereographic(central_latitude=cl, central_longitude=0,
                                           true_scale_latitude=tsl)
        elif self.projection == 'laea':
            self.proj = ccrs.LambertAzimuthalEqualArea(central_latitude=cl,
                                                       central_longitude=0)
        else:
            raise ValueError(f'\tUnsupport projection: {self.projection}! Should be one of "stere" or "laea"')

        # Source projection (geographic coordinates)
        self.src_proj = ccrs.PlateCarree()

    @staticmethod
    def _infer_direction_from_attrs(u_attrs: Dict[str, Any],
                                    v_attrs: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Infer vector frame from variable metadata.

        Returns:
            (direction, reason)
            direction is one of {'xy', 'lonlat', None}.
        """
        def _norm_text(attrs: Dict[str, Any]) -> str:
            parts = []
            for key in ('standard_name', 'long_name', 'comment', 'description'):
                val = attrs.get(key)
                if val is not None:
                    parts.append(str(val).lower())
            return ' | '.join(parts)

        u_txt = _norm_text(u_attrs or {})
        v_txt = _norm_text(v_attrs or {})
        merged = f"{u_txt} || {v_txt}"

        # Typical CF-style geographic components.
        if (
            ('eastward' in u_txt or 'zonal' in u_txt)
            and ('northward' in v_txt or 'meridional' in v_txt)
        ):
            return 'lonlat', 'eastward/northward metadata'

        # Typical model-grid components (e.g., CMIP siu/siv).
        if (
            'sea_ice_x_velocity' in u_txt
            and 'sea_ice_y_velocity' in v_txt
        ):
            return 'xy', 'sea_ice_x/y_velocity metadata'
        if (
            ('x-component' in u_txt and 'y-component' in v_txt)
            or ('native model grid' in merged)
            or ('native model' in merged and 'x-velocity' in merged)
        ):
            return 'xy', 'x/y native-grid metadata'

        return None, 'no decisive metadata'

    @staticmethod
    def _get_time_dimension(dataset: xr.Dataset, var_key: Optional[str] = None) -> str:
        """Identify time dimension name in dataset."""
        if var_key and var_key in dataset.variables:
            for dim in dataset[var_key].dims:
                dlow = dim.lower()
                if 'time' in dlow and 'bnds' not in dlow and 'bounds' not in dlow:
                    return dim

        time_candidates = [
            dim for dim in dataset.dims
            if ('time' in dim.lower() and 'bnds' not in dim.lower() and 'bounds' not in dim.lower())
        ]
        if not time_candidates:
            raise ValueError("No time dimension found in dataset.")
        return time_candidates[0]

    @staticmethod
    def _time_values_to_keys(time_values: Any) -> List[Tuple[int, int, int]]:
        """Convert decoded time values into `(year, month, day)` tuples.

        Supports both NumPy/pandas datetime-like values and CFTime calendars
        (for example `cftime.DatetimeNoLeap`).
        """
        vals = np.asarray(time_values).ravel()
        if vals.size == 0:
            return []

        first = vals[0]
        if hasattr(first, 'year') and hasattr(first, 'month'):
            keys: List[Tuple[int, int, int]] = []
            for item in vals:
                yy = int(getattr(item, 'year'))
                mm = int(getattr(item, 'month'))
                dd = int(getattr(item, 'day', 1))
                keys.append((yy, mm, dd))
            return keys

        time_vals = pd.to_datetime(vals)
        return [(int(t.year), int(t.month), int(t.day)) for t in time_vals]

    @staticmethod
    def _normalize_time_keys(time_keys: Optional[list], nt: int) -> List[Tuple[int, ...]]:
        """Normalize arbitrary time key containers into tuples for set operations."""
        if isinstance(time_keys, (list, tuple)) and len(time_keys) == nt:
            norm: List[Tuple[int, ...]] = []
            for item in time_keys:
                if isinstance(item, (tuple, list, np.ndarray)):
                    vals = tuple(int(v) for v in np.asarray(item).astype(int).ravel().tolist())
                else:
                    vals = (int(item),)
                norm.append(vals)
            return norm
        # Fallback to index-based alignment when explicit time keys are absent.
        return [(idx,) for idx in range(nt)]

    @staticmethod
    def _time_keys_to_yearmon(time_keys: List[Tuple[int, ...]]) -> List[Tuple[int, int]]:
        """Convert normalized time keys into (year, month) tuples when available."""
        out: List[Tuple[int, int]] = []
        for key in time_keys:
            if len(key) >= 2:
                out.append((int(key[0]), int(key[1])))
        return out

    @classmethod
    def _align_to_common_times(cls,
                               u1: np.ndarray, v1: np.ndarray, time1: Optional[list],
                               u2: np.ndarray, v2: np.ndarray, time2: Optional[list],
                               obs_u: Optional[np.ndarray] = None,
                               obs_v: Optional[np.ndarray] = None,
                               obs_time: Optional[list] = None,
                               key_mode: str = 'exact') -> Tuple[
                                   np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                   Optional[np.ndarray], Optional[np.ndarray], List[Tuple[int, ...]]
                               ]:
        """Align two (or three) vector datasets onto their common time keys.

        Args:
            key_mode:
                - 'exact': use full normalized keys as-is.
                - 'yearmon': compare only (year, month), ignoring day-of-month.
                  This is recommended for monthly-mean products whose timestamps
                  may differ by representative day (for example 15th vs 16th).
        """
        nt1, nt2 = int(u1.shape[0]), int(u2.shape[0])
        keys1_raw = cls._normalize_time_keys(time1, nt1)
        keys2_raw = cls._normalize_time_keys(time2, nt2)

        def _canon_key(key: Tuple[int, ...]) -> Tuple[int, ...]:
            if key_mode == 'yearmon' and len(key) >= 2:
                return (int(key[0]), int(key[1]))
            return key

        keys1 = [_canon_key(key) for key in keys1_raw]
        keys2 = [_canon_key(key) for key in keys2_raw]
        map2 = {k: idx for idx, k in enumerate(keys2)}

        use_obs = obs_u is not None and obs_v is not None
        map_obs = {}
        if use_obs:
            nt_obs = int(obs_u.shape[0])
            obs_keys_raw = cls._normalize_time_keys(obs_time, nt_obs)
            obs_keys = [_canon_key(key) for key in obs_keys_raw]
            map_obs = {k: idx for idx, k in enumerate(obs_keys)}
        else:
            obs_keys = []

        idx1: List[int] = []
        idx2: List[int] = []
        idx_obs: List[int] = []
        common_keys: List[Tuple[int, ...]] = []
        seen: set = set()
        for i1, key in enumerate(keys1):
            if key in seen:
                continue
            i2 = map2.get(key)
            if i2 is None:
                continue
            if use_obs:
                io = map_obs.get(key)
                if io is None:
                    continue
                idx_obs.append(io)
            idx1.append(i1)
            idx2.append(i2)
            common_keys.append(key)
            seen.add(key)

        if not idx1:
            empty_shape = (0,) + tuple(u1.shape[1:])
            empty = np.empty(empty_shape, dtype=float)
            empty_obs = np.empty(empty_shape, dtype=float) if use_obs else None
            return empty, empty.copy(), empty.copy(), empty.copy(), empty_obs, empty_obs.copy() if use_obs else None, []

        u1a = u1[idx1, :, :]
        v1a = v1[idx1, :, :]
        u2a = u2[idx2, :, :]
        v2a = v2[idx2, :, :]
        if use_obs:
            obs_ua = obs_u[idx_obs, :, :]
            obs_va = obs_v[idx_obs, :, :]
        else:
            obs_ua = None
            obs_va = None
        return u1a, v1a, u2a, v2a, obs_ua, obs_va, common_keys

    def SID_1M_metrics(self, u_file: str, u_key: str, v_file: str, v_key: str,
                       rotate: Optional[bool] = None,
                       model_direction: str = 'xy',
                       sector: str = 'All') -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate sea ice drift metrics for a single dataset.

        Args:
            u_file: File containing u-component of velocity.
            u_key: Variable name for u-component.
            v_file: File containing v-component of velocity (if None, search in u_file).
            v_key: Variable name for v-component.
            rotate: Backward-compatible switch. True->lonlat, False->xy.
            model_direction: One of {'xy', 'lonlat', 'other'}.
                - xy: vectors are already in evaluation-grid x/y direction.
                - lonlat: vectors are geographic east/north and need cartopy rotation.
                - other: placeholder path for custom-angle rotation (TODO).
            sector: Geographic sector for regional analysis.

        Returns:
            Dictionary containing drift metrics including:
            - Monthly climatologies: u_clim, v_clim, MKE_clim, speed_clim
            - Time series climatologies: MKE_ts_clim, speed_ts_clim
            - Anomalies: MKE_ano, speed_ano, MKE_ts_ano, speed_ts_ano
            - Trends: MKE_ano_tr, speed_ano_tr, MKE_ts_ano_tr, speed_ts_ano_tr
            - Variabilities: MKE_ts_ano_std, speed_ts_ano_std
        """
        logger.info("Calculating SID metrics for %s ...", u_file)

        # Read data
        with xr.open_dataset(u_file) as ds:
            u = np.array(ds[u_key])
            lon = np.array(ds['lon'])
            lat = np.array(ds['lat'])
            u_attrs = dict(ds[u_key].attrs)
            time_name = self._get_time_dimension(ds, u_key)
            time_keys = self._time_values_to_keys(ds[time_name].values)
        with xr.open_dataset(v_file) as ds:
            v = np.array(ds[v_key])
            v_attrs = dict(ds[v_key].attrs)

        nt, nx, ny = u.shape
        logger.info("\tProcessing %d time steps ...", nt)

        # --- Vector rotation handling ---
        direction = str(model_direction or 'xy').strip().lower()
        if rotate is not None:
            # Backward compatibility with older rotate=True/False call sites.
            direction = 'lonlat' if bool(rotate) else 'xy'

        inferred_direction, infer_reason = self._infer_direction_from_attrs(u_attrs, v_attrs)
        if inferred_direction is not None and direction != inferred_direction:
            logger.warning(
                "Requested model_direction='%s' but %s in %s/%s suggests '%s'. "
                "Keeping requested direction from recipe.",
                direction, infer_reason, os.path.basename(u_file), os.path.basename(v_file),
                inferred_direction,
            )

        if direction not in ('xy', 'lonlat', 'other'):
            raise ValueError(
                f"Unsupported model_direction='{model_direction}'. "
                "Allowed values: xy, lonlat, other."
            )

        # Case 1: already on target x/y orientation.
        if direction == 'xy':
            logger.info("\tUsing vectors as-is (model_direction='xy').")
        # Case 2: geographic east/north -> projection x/y.
        elif direction == 'lonlat':
            logger.info("\tRotating vectors from lon/lat to x/y using Cartopy.")
            u_rot = np.full_like(u, np.nan)
            v_rot = np.full_like(v, np.nan)
            for ii in range(nt):
                u_rot[ii], v_rot[ii] = utils.rotate_vector_formula(
                    lons=lon,
                    hemisphere=self.hemisphere,
                    u=u[ii],
                    v=v[ii],
                )
            u, v = u_rot, v_rot
            logger.info("Vectors rotated to xy direction. Source: %s", u_file)
        # Case 3: placeholder for custom-grid vectors.
        else:
            logger.warning(
                "\tmodel_direction='other': applying placeholder angle rotation (0 rad). "
                "Replace TODO with real angle field."
            )
            # TODO: Load/provide model-specific rotation angle field (theta) and
            # rotate to evaluation-grid x/y using:
            #   u_rot[ii], v_rot[ii] = utils.rotate_vector_by_angle(
            #       u[ii], v[ii], theta, theta_unit='radian'
            #   )
            theta = np.zeros_like(lon, dtype=float)
            u_rot = np.full_like(u, np.nan)
            v_rot = np.full_like(v, np.nan)
            for ii in range(nt):
                u_rot[ii], v_rot[ii] = utils.rotate_vector_by_angle(
                    u[ii], v[ii], theta, theta_unit='radian'
                )
            u, v = u_rot, v_rot
            logger.info("Applied placeholder custom-angle rotation for: %s", u_file)

        # === Check inputs and quality control ===
        if np.nanmax(u) > 5:  # likely cm/s
            logger.warning("max|u|=%.2f — units may be cm/s; converting to m/s.", np.nanmax(u))
            u, v = u / 100, v / 100
        u[u == 0], v[v == 0] = np.nan, np.nan

        # Apply regional mask
        sec_index = utils.region_index(self.grid_file, self.hemisphere, sector)
        u[:, ~sec_index], v[:, ~sec_index] = np.nan, np.nan

        # Calculate drift metrics
        out = self._calculate_drift_metrics(u, v)
        out['source_direction'] = direction
        out['vector_frame'] = 'xy'
        out['time_keys'] = time_keys
        out['yearmon_list'] = [(int(y), int(m)) for y, m, *_ in time_keys]
        return out

    def _calculate_drift_metrics(self, u: np.ndarray, v: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate comprehensive drift metrics from u and v components.
        """

        nt, nx, ny = u.shape

        # Calculate mean kinetic energy and speed
        # TODO: use MKE or speed as the primary metric?
        MKE = (u**2 + v**2) / 2
        speed = np.sqrt(u**2 + v**2)

        # Time series of domain averages
        MKE_ts = np.array([np.nanmean(MKE[m, :, :]) for m in range(nt)])
        speed_ts = np.array([np.nanmean(speed[m, :, :]) for m in range(nt)])

        # 1. Monthly climatologies
        u_clim = np.array([np.nanmean(u[m::12, :, :], axis=0) for m in range(12)])
        v_clim = np.array([np.nanmean(v[m::12, :, :], axis=0) for m in range(12)])
        MKE_clim = np.array([np.nanmean(MKE[m::12, :, :], axis=0) for m in range(12)])
        speed_clim = np.array([np.nanmean(speed[m::12, :, :], axis=0) for m in range(12)])

        MKE_ts_clim = np.array([np.nanmean(MKE_ts[m::12]) for m in range(12)])
        speed_ts_clim = np.array([np.nanmean(speed_ts[m::12]) for m in range(12)])

        # 2. Interannual variabilities
        # The anomalies are defined as the signal minus the mean seasonal cycle
        MKE_ano = np.array([MKE[j, :, :] - MKE_clim[j % 12, :, :] for j in range(nt)])
        speed_ano = np.array([speed[j, :, :] - speed_clim[j % 12, :, :] for j in range(nt)])

        MKE_ts_ano = np.array([MKE_ts[j] - MKE_ts_clim[j % 12] for j in range(nt)])
        speed_ts_ano = np.array([speed_ts[j] - speed_ts_clim[j % 12] for j in range(nt)])

        # 3. Trend
        months = np.arange(1, nt + 1)
        if nt / 12 < 15:
            logger.warning("Time series shorter than 15 years — trends may be uncertain.")

        # Spatial trends
        MKE_ano_tr = np.full((nx, ny), np.nan)
        MKE_ano_tr_p = np.full((nx, ny), np.nan)
        speed_ano_tr = np.full((nx, ny), np.nan)
        speed_ano_tr_p = np.full((nx, ny), np.nan)
        for jx in np.arange(nx):
            for jy in np.arange(ny):
                # MKE trends
                ano_ts = MKE_ano[:, jx, jy]
                valid_idx = ~np.isnan(ano_ts)
                if np.sum(valid_idx) > nt / 2:
                    slope, _, _, p_value, _ = stats.linregress(months[valid_idx], ano_ts[valid_idx])
                    MKE_ano_tr[jx, jy] = slope * 12 * 10  # Convert to per decade
                    MKE_ano_tr_p[jx, jy] = p_value

                # Speed trends
                ano_ts_speed = speed_ano[:, jx, jy]
                valid_idx_speed = ~np.isnan(ano_ts_speed)
                if np.sum(valid_idx_speed) > nt / 2:
                    slope, _, _, p_value, _ = stats.linregress(months[valid_idx_speed], ano_ts_speed[valid_idx_speed])
                    speed_ano_tr[jx, jy] = slope * 12 * 10
                    speed_ano_tr_p[jx, jy] = p_value

        # Time series trends
        MKE_ts_ano_tr = stats.linregress(months, MKE_ts_ano)
        speed_ts_ano_tr = stats.linregress(months, speed_ts_ano)

        # Scalar differences derived from the current evaluation footprint
        MKE_std_diff = float(np.nanmean(np.nanstd(MKE_ano, axis=0)))
        MKE_trend_diff = float(np.nanmean(MKE_ano_tr))
        MKE_ts_trend_diff = float(MKE_ts_ano_tr.slope * 12 * 10)

        return {'u': u, 'v': v,
                'u_clim': u_clim, 'v_clim': v_clim,
                'MKE_clim': MKE_clim, 'speed_clim': speed_clim,
                'MKE_ts_clim': MKE_ts_clim, 'speed_ts_clim': speed_ts_clim,
                'MKE_ano': MKE_ano, 'speed_ano': speed_ano,
                'MKE_ts_ano': MKE_ts_ano, 'speed_ts_ano': speed_ts_ano,
                'MKE_ano_tr': MKE_ano_tr, 'MKE_ano_tr_p': MKE_ano_tr_p, 'speed_ano_tr': speed_ano_tr,
                'speed_ano_tr_p': speed_ano_tr_p,
                'MKE_ts_ano_tr': MKE_ts_ano_tr, 'speed_ts_ano_tr': speed_ts_ano_tr,
                'MKE_ts_ano_std': np.nanstd(MKE_ts_ano), 'speed_ts_ano_std': np.nanstd(speed_ts_ano),
                'MKE_std_diff': MKE_std_diff, 'MKE_trend_diff': MKE_trend_diff,
                'MKE_ts_trend_diff': MKE_ts_trend_diff}

    def SID_2M_metrics(self, u_file1: str, u_key1: str, v_file1: str, v_key1: str,
                       u_file2: str, u_key2: str, v_file2: str, v_key2: str,
                       rotate1: Optional[bool] = None, rotate2: Optional[bool] = None,
                       model_direction1: str = 'xy', model_direction2: str = 'xy',
                       strict_obs_match: bool = False,
                       sector: str = 'All',
                       obs_match_u_file: Optional[str] = None,
                       obs_match_u_key: Optional[str] = None,
                       obs_match_v_file: Optional[str] = None,
                       obs_match_v_key: Optional[str] = None,
                       obs_match_direction: str = 'xy') -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate the mean Mean Kinetic Energy (MKE) difference between two spatio-temporal velocity fields,
            the vector correlation coefficient in each grid cell,
            and the mean vector correlation coefficient

        Args:
            u_file_1: Observation dataset u-component file.
            u_key_1: Variable name for u-component in observation file.
            u_file_2: Model dataset u-component file.
            u_key_2: Variable name for u-component in model file.
            v_file_1: Observation dataset v-component file (default: same as u_file_1).
            v_key_1: Variable name for v-component in observation file
                    (default: u_key_1 with 'u' replaced by 'v').
            v_file_2: Model dataset v-component file (default: same as u_file_2).
            v_key_2: Variable name for v-component in model file
                    (default: u_key_2 with 'u' replaced by 'v').
            rotate1: Backward-compatible switch for dataset-1 (True->lonlat, False->xy).
            rotate2: Backward-compatible switch for dataset-2 (True->lonlat, False->xy).
            model_direction1: Direction tag for dataset-1.
            model_direction2: Direction tag for dataset-2.
            strict_obs_match:
                When True, apply obs1-valid footprint matching before metric
                diagnostics, i.e., keep only cells/time where obs1 and model
                are all finite in u/v components.
            sector: Geographic sector for regional analysis.
            obs_match_u_file / obs_match_v_file:
                Optional second-observation files used to enforce common
                obs1+obs2+model temporal/spatial coverage in strict mode.
            obs_match_u_key / obs_match_v_key:
                Variable names for the optional second-observation files.
            obs_match_direction:
                Direction tag for optional second-observation vectors.

        Returns
        -------
        A dict variable contains the following metrics,
            · Mean state difference (1, ):                       MKE_mean_diff, X_mean_diff, IIEE_mean_diff
            · Mean anomaly varaiance difference (1, ):           siconc_std_diff, X_std_diff
            · Mean trend differnece (1, ):                       siconc_trend_diff, X_trend_diff

        Notes
        -----
        The projection of both datasets should be the same (e.g., both in polar stereographic).
            Typical setup: observations in 'xy' and models in 'xy' or 'lonlat',
            controlled by ``model_direction1`` and ``model_direction2``.

        """
        logger.info("Comparing SID datasets: %s vs %s", u_file1, u_file2)

        # === Calculate sea ice drift metrics ===
        sid1 = self.SID_1M_metrics(
            u_file1, u_key1, v_file1, v_key1,
            rotate=rotate1, model_direction=model_direction1, sector=sector
        )
        sid2 = self.SID_1M_metrics(
            u_file2, u_key2, v_file2, v_key2,
            rotate=rotate2, model_direction=model_direction2, sector=sector
        )

        u1, v1 = sid1['u'], sid1['v']
        u2, v2 = sid2['u'], sid2['v']
        u1, v1, u2, v2, _, _, common_time_keys = self._align_to_common_times(
            u1, v1, sid1.get('time_keys'),
            u2, v2, sid2.get('time_keys'),
            key_mode='yearmon',
        )
        if not common_time_keys:
            logger.warning(
                "No common time coverage for SID comparison: %s vs %s",
                u_file1, u_file2,
            )
            nx, ny = self.lon.shape
            nan2d = np.full((nx, ny), np.nan)
            return {
                'MKE_mean_diff': np.nan,
                'vectcorr': nan2d,
                'vectcorr_mean': np.nan,
                'match_mode': 'pairwise',
                'sid1_metric': sid1,
                'sid2_metric': sid2,
                'MKE_std_diff': np.nan,
                'MKE_ts_ano_std': np.nan,
                'MKE_trend_diff': np.nan,
                'MKE_ts_trend_diff': np.nan,
            }

        sid1['time_keys'] = common_time_keys
        sid2['time_keys'] = common_time_keys
        sid1['yearmon_list'] = self._time_keys_to_yearmon(common_time_keys)
        sid2['yearmon_list'] = self._time_keys_to_yearmon(common_time_keys)

        match_mode = 'obs1_strict' if strict_obs_match else 'pairwise'
        if strict_obs_match:
            obs_match_u = None
            obs_match_v = None
            if obs_match_u_file and obs_match_v_file:
                match_u_key = obs_match_u_key or u_key1
                match_v_key = obs_match_v_key or v_key1
                sid_match = self.SID_1M_metrics(
                    obs_match_u_file, match_u_key, obs_match_v_file, match_v_key,
                    model_direction=obs_match_direction, sector=sector
                )
                obs_match_u = sid_match.get('u')
                obs_match_v = sid_match.get('v')
                u1, v1, u2, v2, obs_match_u, obs_match_v, common_time_keys = self._align_to_common_times(
                    u1, v1, sid1.get('time_keys'),
                    u2, v2, sid2.get('time_keys'),
                    obs_match_u, obs_match_v, sid_match.get('time_keys'),
                    key_mode='yearmon',
                )
                if not common_time_keys:
                    logger.warning(
                        "No common obs1/obs2/model time coverage in strict SID match: %s | %s | %s",
                        u_file1, obs_match_u_file, u_file2,
                    )
                    nx, ny = self.lon.shape
                    nan2d = np.full((nx, ny), np.nan)
                    return {
                        'MKE_mean_diff': np.nan,
                        'vectcorr': nan2d,
                        'vectcorr_mean': np.nan,
                        'match_mode': 'obs_dual_strict',
                        'sid1_metric': sid1,
                        'sid2_metric': sid2,
                        'MKE_std_diff': np.nan,
                        'MKE_ts_ano_std': np.nan,
                        'MKE_trend_diff': np.nan,
                        'MKE_ts_trend_diff': np.nan,
                    }
                sid1['time_keys'] = common_time_keys
                sid2['time_keys'] = common_time_keys
                sid1['yearmon_list'] = self._time_keys_to_yearmon(common_time_keys)
                sid2['yearmon_list'] = self._time_keys_to_yearmon(common_time_keys)
                match_mode = 'obs_dual_strict'

            u1, v1, u2, v2 = self._validate_uv_match(
                u1, v1, u2, v2, strict_obs_match=True,
                obs_match_u=obs_match_u, obs_match_v=obs_match_v,
            )
            sid1 = self._calculate_drift_metrics(u1, v1)
            sid2 = self._calculate_drift_metrics(u2, v2)
            sid1['u'], sid1['v'] = u1, v1
            sid2['u'], sid2['v'] = u2, v2
            sid1['time_keys'] = common_time_keys
            sid2['time_keys'] = common_time_keys
            sid1['yearmon_list'] = self._time_keys_to_yearmon(common_time_keys)
            sid2['yearmon_list'] = self._time_keys_to_yearmon(common_time_keys)

        # === 1. MKE diff ===
        error_mean_monthly = np.full((12,), np.nan)
        for m in range(12):
            error_mean_monthly[m] = utils.MatrixDiff(sid1['MKE_clim'][m, :, :], sid2['MKE_clim'][m, :, :],
                                                     metric=self.metric, mask=True)
        MKE_mean_diff = np.sum(error_mean_monthly * DAYS_PER_MONTH) / np.sum(DAYS_PER_MONTH)

        vectcorr = self._calculate_vector_correlation(u1, v1, u2, v2)
        sid1.setdefault('MKE_std_diff', sid1.get('MKE_std_diff', np.nan))
        sid1.setdefault('MKE_trend_diff', sid1.get('MKE_trend_diff', np.nan))
        sid1.setdefault('MKE_ts_trend_diff', sid1.get('MKE_ts_trend_diff', np.nan))

        return {'MKE_mean_diff': MKE_mean_diff,
                'vectcorr': vectcorr,
                'vectcorr_mean': np.nanmean(vectcorr),
                'match_mode': match_mode,
                'sid1_metric': sid1,
                'sid2_metric': sid2,
                'MKE_std_diff': utils.MatrixDiff(sid1['MKE_ano'], sid2['MKE_ano'], metric=self.metric, mask=True),
                'MKE_ts_ano_std': utils.MatrixDiff(sid1['MKE_ts_ano_std'], sid2['MKE_ts_ano_std'], metric=self.metric, mask=True),
                'MKE_trend_diff': utils.MatrixDiff(sid1['MKE_ano_tr'], sid2['MKE_ano_tr'], metric='MAE', mask=True),
                'MKE_ts_trend_diff': sid1['MKE_ts_trend_diff'] - sid2['MKE_ts_trend_diff']}

    @staticmethod
    def _validate_uv_match(u1: np.ndarray, v1: np.ndarray,
                           u2: np.ndarray, v2: np.ndarray,
                           strict_obs_match: bool = False,
                           obs_match_u: Optional[np.ndarray] = None,
                           obs_match_v: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply co-located/matched valid mask to u/v arrays.

        Default mode (strict_obs_match=False):
            Keep only pairwise finite samples for u/v between two datasets.

        strict_obs_match=True:
            Enforce obs1-valid footprint matching:
              - observation (u1/v1) must be finite
              - model (u2/v2) must be finite
              - if provided, obs2 (obs_match_u/obs_match_v) must be finite
            The resulting mask is applied to all four arrays.
        """
        u1o = u1.copy()
        v1o = v1.copy()
        u2o = u2.copy()
        v2o = v2.copy()
        if strict_obs_match:
            obs_invalid = (~np.isfinite(u1)) | (~np.isfinite(v1))
            model_invalid = (~np.isfinite(u2)) | (~np.isfinite(v2))
            invalid = obs_invalid | model_invalid
            if obs_match_u is not None and obs_match_v is not None:
                if np.shape(obs_match_u) != np.shape(u1) or np.shape(obs_match_v) != np.shape(v1):
                    raise ValueError(
                        "obs_match_u/v shape mismatch in strict SID matching: "
                        f"{np.shape(obs_match_u)}, {np.shape(obs_match_v)} vs {np.shape(u1)}, {np.shape(v1)}"
                    )
                obs2_invalid = (~np.isfinite(obs_match_u)) | (~np.isfinite(obs_match_v))
                invalid = invalid | obs2_invalid
        else:
            invalid = (
                (~np.isfinite(u1)) | (~np.isfinite(v1))
                | (~np.isfinite(u2)) | (~np.isfinite(v2))
            )
        u1o[invalid] = np.nan
        v1o[invalid] = np.nan
        u2o[invalid] = np.nan
        v2o[invalid] = np.nan
        return u1o, v1o, u2o, v2o

    def _calculate_vector_correlation(self, u1: np.ndarray, v1: np.ndarray,
                                      u2: np.ndarray, v2: np.ndarray):
        """
        Compute the Crosby et al. (1993) vector correlation coefficient at each grid cell.

        The coefficient ρ² ∈ [0, 2] measures how well two 2-D vector fields co-vary:
            ρ² = 0  → no correlation
            ρ² = 1  → same as two independent scalar correlations of 1
            ρ² = 2  → perfect vector correlation (both magnitude and direction)

        Reference: Crosby, D. S., Breaker, L. C., & Gemmill, W. H. (1993).
            A proposed definition for vector correlation in geophysics: Theory and application.
            Journal of Atmospheric and Oceanic Technology, 10(3), 355–367.
        """
        _, nx, ny = u1.shape

        logger.info("\n\t --- Computing vector correlation (vectorized), this may take a few minutes ---\n")

        # Propagate one common valid-time mask across all components so each
        # covariance term uses exactly the same samples.
        valid = np.isfinite(u1) & np.isfinite(v1) & np.isfinite(u2) & np.isfinite(v2)
        u1m = np.where(valid, u1, np.nan)
        v1m = np.where(valid, v1, np.nan)
        u2m = np.where(valid, u2, np.nan)
        v2m = np.where(valid, v2, np.nan)

        n_valid = np.sum(valid, axis=0).astype(float)
        denom = n_valid - 1.0

        def _covariance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Sample covariance with NaN-safe common mask and ddof=1."""
            mean_a = np.divide(
                np.nansum(a, axis=0, dtype=float),
                n_valid,
                out=np.full((nx, ny), np.nan, dtype=float),
                where=n_valid > 0,
            )
            mean_b = np.divide(
                np.nansum(b, axis=0, dtype=float),
                n_valid,
                out=np.full((nx, ny), np.nan, dtype=float),
                where=n_valid > 0,
            )
            numerator = np.nansum((a - mean_a) * (b - mean_b), axis=0, dtype=float)
            return np.divide(
                numerator,
                denom,
                out=np.full((nx, ny), np.nan, dtype=float),
                where=denom > 0,
            )

        # Covariance terms required by Crosby (1993)
        u1u2 = _covariance(u1m, u2m)
        u1u1 = _covariance(u1m, u1m)
        u2u2 = _covariance(u2m, u2m)
        u1v1 = _covariance(u1m, v1m)
        v1v1 = _covariance(v1m, v1m)
        u1v2 = _covariance(u1m, v2m)
        v2v2 = _covariance(v2m, v2m)
        u2v2 = _covariance(u2m, v2m)
        v1v2 = _covariance(v1m, v2m)
        v1u2 = _covariance(v1m, u2m)

        # --- Crosby (1993) vector correlation formula ---
        # Numerator f = f0 + f1 encodes cross-covariance between the two vector fields
        f0 = u1u1 * (u2u2 * v1v2 ** 2 + v2v2 * v1u2 ** 2) + v1v1 * (u2u2 * u1v2 ** 2 + v2v2 * u1u2 ** 2) + 2 * (
                u1v1 * u1v2 * v1u2 * u2v2) + 2 * (u1v1 * u1u2 * v1v2 * u2v2)
        f1 = -2 * (u1u1 * v1u2 * v1v2 * u2v2) - 2 * (v1v1 * u1u2 * u1v2 * u2v2) - 2 * (
                u2u2 * u1v1 * u1v2 * v1v2) - 2 * (
                     v2v2 * u1v1 * u1u2 * v1u2)

        f = f0 + f1
        # Denominator g = product of the two vector field variances (det of each covariance matrix)
        g = (u1u1 * v1v1 - u1v1 ** 2) * (u2u2 * v2v2 - u2v2 ** 2)
        idx = np.where(g == 0)
        g[idx] = np.nan  # avoid division by zero at ice-free cells
        vectcorr = f / g / 2

        # Mask cells with fewer than 11 valid time steps — correlation is unreliable
        # (keeps only coefficients significant at the 99% level)
        vectcorr[n_valid < 11] = np.nan

        return vectcorr
