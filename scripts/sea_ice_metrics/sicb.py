# -*- coding: utf-8 -*-
"""Sea-ice metric implementations by diagnostic family."""

import datetime as _dt
import logging
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from scripts import SITOOL_NC_COMPRESS_LEVEL, SITOOL_NC_SHUFFLE
from scripts import utils
from scripts.utils import runtime_efficiency as rte
from scripts.config import DAYS_PER_MONTH, DAYS_PER_SEASON
from scripts.sea_ice_metrics.base import SeaIceMetricsBase

logger = logging.getLogger(__name__)

# Aggregation metadata version:
# 1 => seasonal/monthly budget climatologies use a common-valid mask across
#      dadt/adv/div/res before temporal aggregation.
_SICB_COMMON_VALID_AGGREGATION = 1


class SICBMetrics(SeaIceMetricsBase):
    """
    Calculate sea ice concentration budget (SICB) metrics.

    This class provides methods to compute sea ice concentration budget components
    including advection, divergence, and residual terms. It supports both single
    dataset analysis and comparisons between two datasets.

    Attributes:
        grid_file (str): Path to grid file containing cell area and coordinates.
        time_sta (str): Start date for analysis (format: 'YYYY-MM-DD').
        time_end (str): End date for analysis (format: 'YYYY-MM-DD').
        hemisphere (str): Hemisphere ('nh' for Northern or 'sh' for Southern).
        metric (str): Statistical metric for comparisons ('MAE', 'RMSE', etc.).
    """

    def __init__(self, grid_file: str, date_start: str, date_end: str,
                 hemisphere: str, metric: str = 'MAE'):
        """
        Initialize the SICB metrics calculator.

        Args:
            grid_file: NetCDF file with 'cell_area' variable (units: m²).
            date_start: Start date for analysis period (format: 'YYYY-MM-DD').
            date_end: End date for analysis period (format: 'YYYY-MM-DD').
            hemisphere: Target hemisphere ('nh' for Northern or 'sh' for Southern).
            metric: Statistical metric for comparisons.
        """
        super().__init__(grid_file, hemisphere, metric)
        self.date_sta = date_start
        self.date_end = date_end

        # Convert cell area to km² for budget calculations
        self.cell_area_km2 = self.cell_area / 1e6  # Convert m² to km²

        # Parse date range
        date_format = "%Y-%m-%d"
        try:
            date_start = _dt.datetime.strptime(date_start, date_format)
            date_end = _dt.datetime.strptime(date_end, date_format)
            self.date_range = np.arange(date_start, date_end, _dt.timedelta(days=1))
            self.year_sta, self.year_end = date_start.year, date_end.year
            self.years = np.arange(self.year_sta, self.year_end + 1)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD'. Error: {e}")

    def Cal_SIC_budget(self, sic_file: str, sic_key: str, u_file: str, u_key: str, v_file: str, v_key: str,
                       hemisphere: str,
                       rotate: bool = False,
                       ngrid_filter: int = 3, sic_threshold: int = 15,
                       time_frequency: str = 'daily',
                       jobs: int = 1,
                       output_folder: Optional[str] = './',
                       output_label: Optional[str] = '',
                       valid_mask: Optional[np.ndarray] = None,
                       normalize_by_valid_days: bool = False,
                       reuse_existing: bool = True) -> Tuple[str, str]:
        """
        Calculate seasonal climatologies for sea ice concentration budget components.

        Args:
            sic_file: NetCDF file containing daily sea ice concentration data.
            siuv_file: NetCDF file containing daily sea ice drift data.
            hemisphere: Hemisphere for analysis.
            sic_key: Variable name for sea ice concentration.
            u_key: Variable name for u-component of velocity.
            v_key: Variable name for v-component of velocity.
            rotate_func: Optional function to rotate velocity vectors.
            ngrid_filter: Size of ice velocity filter (default: 3x3).
            sic_threshold: SIC threshold for valid ice motion ('15%').
            time_frequency: Temporal frequency ('daily' or 'monthly').
            jobs: Number of timestep worker threads for SICB inner splitting.
            output_folder: Directory for output files.
            output_label: Label for output file naming.
            valid_mask: Optional external common-valid mask with shape
                ``(time, y, x)``. Cells/times outside the mask are excluded
                from SIC/U/V before daily budget calculations.
            normalize_by_valid_days: Whether to normalize seasonal/monthly
                climatologies by valid-day count before converting to canonical
                season/month totals. This avoids artificial amplitude shrinkage
                when matched mode has fewer valid samples.
            reuse_existing: Reuse existing daily/seasonal/monthly budget files
                when metadata is compatible. Set False to force recomputation.

        Returns:
            Tuple containing paths to seasonal and monthly climatology files.

        Methods
        -------
            Step 1. Calculate daily adv and div using daily sic and filtered sid
            Step 2. Calculate seasonal climatologies from daily fields

        References
        ----------
        Holland, P. R., & Kimura, N. (2016). Observed Concentration Budgets of Arctic and Antarctic Sea Ice.
            Journal of Climate, 29(14), 5241–5249. https://doi.org/10.1175/JCLI-D-16-0121.1
        Holmes, C. R., Holland, P. R., & Bracegirdle, T. J. (2019). Compensating Biases and a Noteworthy
            Success in the CMIP5 Representation of Antarctic Sea Ice Processes. Geophysical Research Letters, 46(8),
            4299–4307. https://doi.org/10.1029/2018GL081796
        Nie, Y., Uotila, P., Cheng, B., Massonnet, F., Kimura, N., Cipollone, A., & Lv, X. (2022). Southern Ocean
            sea ice concentration budgets of five ocean-sea ice reanalyses. Climate Dynamics, 59(11–12), 3265–3285.
            https://doi.org/10.1007/s00382-022-06260-x
        """
        logger.info("Calculating sea ice concentration budget ...")

        # Generate output file names
        save_prefix = f'{output_folder}SICB_{hemisphere}_{output_label}_{self.year_sta}-{self.year_end}'
        daily_budget_file = f'{save_prefix}_daily.nc'
        seas_clim_file = f'{save_prefix}_SeasClim.nc'
        mon_clim_file = f'{save_prefix}_MonClim.nc'

        if (
            bool(reuse_existing)
            and
            os.path.exists(daily_budget_file)
            and os.path.exists(seas_clim_file)
            and os.path.exists(mon_clim_file)
        ):
            expected_norm = int(bool(normalize_by_valid_days))
            expected_ridging = 1
            expected_common_valid = int(_SICB_COMMON_VALID_AGGREGATION)

            def _read_flag(path: str, key: str, default: int = 0) -> int:
                try:
                    with xr.open_dataset(path) as ds_tmp:
                        raw = ds_tmp.attrs.get(key, default)
                        return int(raw)
                except Exception:
                    return int(default)

            seas_norm = _read_flag(seas_clim_file, 'sicb_normalized_by_valid_days', 0)
            mon_norm = _read_flag(mon_clim_file, 'sicb_normalized_by_valid_days', 0)
            daily_ridging = _read_flag(daily_budget_file, 'sicb_ridging_enabled', 0)
            seas_ridging = _read_flag(seas_clim_file, 'sicb_ridging_enabled', 0)
            mon_ridging = _read_flag(mon_clim_file, 'sicb_ridging_enabled', 0)
            seas_common_valid = _read_flag(seas_clim_file, 'sicb_common_valid_aggregation', 0)
            mon_common_valid = _read_flag(mon_clim_file, 'sicb_common_valid_aggregation', 0)
            if (
                seas_norm == expected_norm
                and mon_norm == expected_norm
                and daily_ridging == expected_ridging
                and seas_ridging == expected_ridging
                and mon_ridging == expected_ridging
                and seas_common_valid == expected_common_valid
                and mon_common_valid == expected_common_valid
            ):
                logger.info(
                    "Using existing budget files: %s, %s",
                    seas_clim_file, mon_clim_file,
                )
                return seas_clim_file, mon_clim_file
            logger.info(
                "Recomputing SICB outputs due to metadata mismatch: "
                "norm expected=%d(seas=%d, mon=%d), "
                "ridging expected=%d(daily=%d, seas=%d, mon=%d), "
                "common-valid expected=%d(seas=%d, mon=%d)",
                expected_norm, seas_norm, mon_norm,
                expected_ridging, daily_ridging, seas_ridging, mon_ridging,
                expected_common_valid, seas_common_valid, mon_common_valid,
            )

        logger.info("Calculating daily advection and divergence ...")
        start_time = time.time()

        # Main calculation logic integrated here
        daily_budget_file = self._calculate_daily_budget_components(
            sic_file, sic_key, u_file, u_key, v_file, v_key, rotate,
            ngrid_filter, sic_threshold, time_frequency, jobs, output_folder, output_label,
            valid_mask=valid_mask,
        )

        # Calculate seasonal and monthly climatologies
        seas_clim_file, mon_clim_file = self._calculate_climatologies(
            daily_budget_file, save_prefix, normalize_by_valid_days=normalize_by_valid_days
        )

        elapsed_time = time.time() - start_time
        logger.info("Sea ice budget calculations completed! Time: %.1fs", elapsed_time)

        return seas_clim_file, mon_clim_file

    def _calculate_daily_budget_components(self, sic_file: str, sic_key: str, u_file: str, u_key: str, v_file: str, v_key: str,
                                           rotate: bool,
                                           ngrid_filter: int, sic_threshold: int,
                                           time_frequency: str, jobs: int, output_folder: str,
                                           output_label: str,
                                           valid_mask: Optional[np.ndarray] = None) -> str:
        """
        Main method for calculating daily budget components with integrated logic.
        sic_threshold: values between 0 and 100
        """
        # Keep all input datasets open for the entire computation scope.
        # This avoids using a closed xarray Dataset object outside its context,
        # which can trigger unstable backend re-open behavior under parallel runs.
        with xr.open_dataset(sic_file) as ds_sic, xr.open_dataset(u_file) as ds_u, xr.open_dataset(v_file) as ds_v:
            # Load all required fields to NumPy arrays once so timestep workers
            # never touch NetCDF backends in parallel.
            lon, lat = np.array(ds_sic['lon']), np.array(ds_sic['lat'])
            sic = np.array(ds_sic[sic_key], dtype=np.float32)
            u = np.array(ds_u[u_key], dtype=np.float32)
            v = np.array(ds_v[v_key], dtype=np.float32)

            nt = int(sic.shape[0])
            nx = int(sic.shape[1])
            ny = int(sic.shape[2])

            if nt <= 0:
                raise ValueError(
                    f"No timesteps available after preprocessing for SICB input: {sic_file}"
                )

            # Calculate distance matrices.
            hx, hy = utils.ll_dist_matrix(lon, lat)

            # Unit conversion and quality control for SIC once per file.
            if float(np.nanmax(sic)) < 10.0:
                sic = sic * 100.0
            sic[(sic > 100.0) | (sic < 0.0)] = np.nan

            if not rotate:
                logger.info("\tUsing vectors as-is (assumed to be in xy direction).")
            else:
                logger.info("\tRotating vectors from lon/lat to x/y using Cartopy.")
                u_rot = np.full_like(u, np.nan, dtype=np.float32)
                v_rot = np.full_like(v, np.nan, dtype=np.float32)
                for ii in range(nt):
                    u_rot[ii], v_rot[ii] = utils.rotate_vector_formula(
                        lons=lon,
                        hemisphere=self.hemisphere,
                        u=u[ii],
                        v=v[ii],
                    )
                u, v = u_rot, v_rot
                logger.info("u v have been rotated to xy direction.")

            if valid_mask is not None:
                valid = np.asarray(valid_mask)
                if valid.shape != sic.shape:
                    raise ValueError(
                        "valid_mask shape mismatch for SICB matching: "
                        f"{valid.shape} vs {sic.shape}"
                    )
                valid = np.isfinite(valid) & (valid > 0)
                keep_ratio = float(np.nanmean(valid.astype(np.float32)))
                logger.info(
                    "Applying external SICB common-valid mask (keep %.1f%% of spatiotemporal samples).",
                    keep_ratio * 100.0,
                )
                invalid = ~valid
                sic[invalid] = np.nan
                u[invalid] = np.nan
                v[invalid] = np.nan

            # Initialize arrays for budget components.
            adv_daily = np.full((nt, nx, ny), np.nan, dtype=np.float32)
            div_daily = np.full((nt, nx, ny), np.nan, dtype=np.float32)
            adv_temp = np.full((nt, nx, ny), np.nan, dtype=np.float32)
            div_temp = np.full((nt, nx, ny), np.nan, dtype=np.float32)
            dadt_daily = np.full((nt, nx, ny), np.nan, dtype=np.float32)

            # Calculate day weights.
            if time_frequency == 'daily':
                day_weights = np.ones(nt, dtype=np.float32)
            else:
                repeats = max(1, int(math.ceil(float(nt) / 12.0)))
                day_weights = np.tile(np.asarray(DAYS_PER_MONTH, dtype=np.float32), repeats)[:nt]

            time_values = np.array(ds_sic['time'].values)

            def _label_time(tt: int) -> str:
                if 0 <= tt < len(time_values):
                    return str(time_values[tt])[:10]
                return str(tt)

            def _process_range(tt_start: int, tt_end: int) -> Tuple[int, int]:
                """Process one contiguous timestep block [tt_start, tt_end)."""
                processed = 0
                failed = 0
                for tt in range(tt_start, tt_end):
                    try:
                        sic_current = sic[tt]
                        sic_prev = sic[tt - 2]

                        # Apply median filter to velocity to reduce noise
                        # [Following Holland and Kimura (2016) and Nie et al. (2022)]
                        u_filtered = utils.median_filter(u[tt], ngrid_filter)
                        v_filtered = utils.median_filter(v[tt], ngrid_filter)

                        # Mask velocity where SIC is below threshold (no meaningful ice motion)
                        invalid_mask = sic_current < sic_threshold
                        u_filtered[invalid_mask] = np.nan
                        v_filtered[invalid_mask] = np.nan

                        # dA/dt via central differencing (units: % per day)
                        dadt = (sic_current - sic_prev) / 2.0

                        # Spatial gradients of SIC and velocity divergence
                        Ax = utils.xy_gradient(sic_current, hx, 'x')   # ∂A/∂x
                        Ay = utils.xy_gradient(sic_current, hy, 'y')   # ∂A/∂y
                        ux = utils.xy_gradient(u_filtered, hx, 'x')    # ∂u/∂x
                        vy = utils.xy_gradient(v_filtered, hy, 'y')    # ∂v/∂y

                        # Budget components (% per day):
                        #   advection = -(u·∂A/∂x + v·∂A/∂y)   [ice transported into cell]
                        #   divergence = -A·(∂u/∂x + ∂v/∂y)    [ice area change due to spreading]
                        adv = -(u_filtered * Ax + v_filtered * Ay) * 86400.0 * day_weights[tt]
                        div = -sic_current * (ux + vy) * 86400.0 * day_weights[tt]

                        idx = tt - 1
                        dadt_daily[idx] = np.asarray(dadt, dtype=np.float32)
                        adv_temp[idx] = np.asarray(adv, dtype=np.float32)
                        div_temp[idx] = np.asarray(div, dtype=np.float32)
                        processed += 1
                    except Exception as exc:
                        failed += 1
                        logger.warning("Error processing time step %s: %s", _label_time(tt), exc)
                return processed, failed

            # Main processing loop (time-split execution for SICB long-tail acceleration).
            safe_jobs = max(1, int(jobs))
            n_steps = max(0, nt - 2)  # need at least 3 time steps for central differencing
            if safe_jobs > 1 and n_steps > 0:
                n_workers = min(safe_jobs, n_steps)
                chunk = max(1, int(math.ceil(float(n_steps) / float(n_workers))))
                ranges: List[Tuple[int, int]] = []
                for ii in range(n_workers):
                    st = 2 + ii * chunk
                    en = min(nt, st + chunk)
                    if st < en:
                        ranges.append((st, en))

                logger.info(
                    "SICB daily budget split: workers=%d, chunks=%d, time_steps=%d",
                    n_workers, len(ranges), n_steps,
                )
                failed_total = 0
                with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix='sicb-daily') as pool:
                    futs = [
                        pool.submit(rte.run_tracked_task, _process_range, st, en)
                        for (st, en) in ranges
                    ]
                    for fut in as_completed(futs):
                        _processed, _failed = fut.result()
                        failed_total += int(_failed)
                if failed_total > 0:
                    logger.warning("SICB daily budget completed with %d failed timesteps.", failed_total)
            else:
                disable_progress = str(os.environ.get('SITOOL_DISABLE_TQDM', '0')).strip().lower() in (
                    '1', 'true', 'yes', 'on'
                )
                step_iter = range(2, nt)
                if not disable_progress:
                    step_iter = tqdm(step_iter, desc="Processing daily budget", disable=False)
                for tt in step_iter:
                    _process_range(tt, tt + 1)

            # Apply 3-day running mean to smooth daily budget components
            for tt in range(nt):
                adv_daily[tt - 1] = np.nanmean(adv_temp[tt - 2:tt + 1], axis=0)
                div_daily[tt - 1] = np.nanmean(div_temp[tt - 2:tt + 1], axis=0)

            # Residual = total tendency − advection − divergence
            # Captures thermodynamic processes (freezing/melting) and mechanical redistribution
            res_daily = dadt_daily - adv_daily - div_daily

            # Daily ridging-grid ratio over ice-covered grids.
            # Ridging grid definition:
            #   (1) SIC > 90%
            #   (2) residual < 0
            #   (3) divergence > 0
            # Ice-covered denominator follows SIC threshold used by SICB motion filtering.
            ice_mask = np.isfinite(sic) & (sic >= float(sic_threshold))
            ridging_mask = (
                ice_mask
                & np.isfinite(res_daily)
                & np.isfinite(div_daily)
                & (sic > 90.0)
                & (res_daily < 0.0)
                & (div_daily > 0.0)
            )
            ice_count = np.sum(ice_mask, axis=(1, 2), dtype=np.float64)
            ridging_count = np.sum(ridging_mask, axis=(1, 2), dtype=np.float64)
            ridging_ratio_daily = np.where(ice_count > 0.0, ridging_count / ice_count, np.nan).astype(np.float32)

            # Create output dataset
            output_file = self._create_budget_dataset(
                ds_sic=ds_sic,
                sic=sic,
                dadt=dadt_daily,
                adv=adv_daily,
                div=div_daily,
                res=res_daily,
                ridging_ratio=ridging_ratio_daily,
                sic_threshold=sic_threshold,
                output_folder=output_folder,
                output_label=output_label,
            )

        return output_file

    @staticmethod
    def _write_dataset_atomic(ds_out: xr.Dataset, output_file: str, retries: int = 3, backoff_s: float = 1.0) -> None:
        """Write one NetCDF file atomically with retry on transient permission errors.

        The method writes to a temporary path in the same directory and then
        replaces the final path with ``os.replace``. This prevents partial files
        from becoming visible and avoids append/open races on an existing target.
        """
        out_dir = os.path.dirname(output_file) or '.'
        os.makedirs(out_dir, exist_ok=True)
        out_base = os.path.basename(output_file)

        for attempt in range(1, max(1, int(retries)) + 1):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f'.{out_base}.tmp.',
                    suffix='.nc',
                    dir=out_dir,
                    delete=False,
                ) as tf:
                    tmp_path = tf.name

                # Write a brand-new file first, then atomically swap it in place.
                encoding: Dict[str, Dict[str, object]] = {}
                for var_name, da in ds_out.data_vars.items():
                    if da.dtype.kind in ('U', 'S', 'O'):
                        continue
                    encoding[var_name] = {
                        'zlib': True,
                        'complevel': int(SITOOL_NC_COMPRESS_LEVEL),
                        'shuffle': bool(SITOOL_NC_SHUFFLE),
                    }
                kwargs: Dict[str, object] = {'mode': 'w'}
                if encoding:
                    kwargs['encoding'] = encoding
                ds_out.to_netcdf(tmp_path, **kwargs)
                os.replace(tmp_path, output_file)
                return
            except PermissionError:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if attempt >= retries:
                    raise
                wait_seconds = float(backoff_s) * float(attempt)
                logger.warning(
                    "Permission denied while writing '%s' (attempt %d/%d). Retrying in %.1fs.",
                    output_file, attempt, retries, wait_seconds,
                )
                time.sleep(wait_seconds)
            except Exception:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

    def _create_budget_dataset(self, ds_sic: xr.Dataset, sic: np.ndarray,
                               dadt: np.ndarray, adv: np.ndarray,
                               div: np.ndarray, res: np.ndarray,
                               ridging_ratio: np.ndarray,
                               sic_threshold: int,
                               output_folder: str,
                               output_label: str) -> str:
        """Create NetCDF dataset with budget components."""
        output_file = f'{output_folder}SICB_{self.hemisphere}_{output_label}_{self.year_sta}-{self.year_end}_daily.nc'

        # Create data arrays
        sic_da = xr.DataArray(np.asarray(sic, dtype=np.float32), dims=('time', 'y', 'x'),
                              coords={'time': ds_sic.time, 'lat': ds_sic.lat, 'lon': ds_sic.lon})
        dadt_da = xr.DataArray(np.asarray(dadt, dtype=np.float32), dims=('time', 'y', 'x'),
                               coords={'time': ds_sic.time, 'lat': ds_sic.lat, 'lon': ds_sic.lon})
        adv_da = xr.DataArray(np.asarray(adv, dtype=np.float32), dims=('time', 'y', 'x'),
                              coords={'time': ds_sic.time, 'lat': ds_sic.lat, 'lon': ds_sic.lon})
        div_da = xr.DataArray(np.asarray(div, dtype=np.float32), dims=('time', 'y', 'x'),
                              coords={'time': ds_sic.time, 'lat': ds_sic.lat, 'lon': ds_sic.lon})
        res_da = xr.DataArray(np.asarray(res, dtype=np.float32), dims=('time', 'y', 'x'),
                              coords={'time': ds_sic.time, 'lat': ds_sic.lat, 'lon': ds_sic.lon})
        ridging_ratio_da = xr.DataArray(
            np.asarray(ridging_ratio, dtype=np.float32), dims=('time',),
            coords={'time': ds_sic.time},
        )

        # Create dataset
        ds_out = xr.Dataset({
            'sic': sic_da,
            'dadt': dadt_da,
            'adv': adv_da,
            'div': div_da,
            'res': res_da,
            'ridging_ratio': ridging_ratio_da,
        })

        # Add attributes
        ds_out.sic.attrs = {'long_name': 'Sea ice concentration', 'units': '%'}
        ds_out.dadt.attrs = {'long_name': 'Rate of change of sea ice concentration', 'units': '%/day'}
        ds_out.adv.attrs = {'long_name': 'Advection term', 'units': '%/day'}
        ds_out.div.attrs = {'long_name': 'Divergence term', 'units': '%/day'}
        ds_out.res.attrs = {'long_name': 'Residual term', 'units': '%/day'}
        ds_out.ridging_ratio.attrs = {
            'long_name': 'Daily ridging-grid ratio in ice-covered region',
            'units': '1',
            'description': (
                'Ridging grid is defined by sic>90%, res<0 and div>0. '
                'Ratio = count(ridging grids) / count(ice grids)'
            ),
        }
        ds_out.attrs.update({
            'sicb_ridging_enabled': 1,
            'sicb_ridging_ice_threshold': float(sic_threshold),
            'sicb_ridging_definition': 'sic>90%,res<0,div>0',
        })

        # Use atomic replace so readers never observe a partially written file.
        self._write_dataset_atomic(ds_out, output_file)
        return output_file

    def _calculate_climatologies(
        self,
        daily_file: str,
        save_prefix: str,
        normalize_by_valid_days: bool = False,
    ) -> Tuple[str, str]:
        """Compute seasonal and monthly climatologies from one daily SICB file.

        Seasonal definition follows CDO-style order:
          DJF, MAM, JJA, SON
        Monthly output is ordered:
          Jan ... Dec
        """
        seas_clim_file = f'{save_prefix}_SeasClim.nc'
        mon_clim_file = f'{save_prefix}_MonClim.nc'

        season_order = ['DJF', 'MAM', 'JJA', 'SON']
        month_order = np.arange(1, 13, dtype=np.int16)

        def _scale_sum_by_valid_count(
            ds_sum: xr.Dataset,
            ds_count: xr.Dataset,
            scale_days: xr.DataArray,
        ) -> xr.Dataset:
            """Scale per-period sums by valid-day ratio to canonical day counts."""
            out_vars: Dict[str, xr.DataArray] = {}
            for vname in ds_sum.data_vars:
                sum_da = ds_sum[vname]
                cnt_da = ds_count[vname]
                scaled = xr.where(
                    cnt_da > 0,
                    (sum_da / cnt_da) * scale_days,
                    np.nan,
                )
                out_vars[vname] = scaled
            ds_out = xr.Dataset(out_vars, coords=ds_sum.coords, attrs=ds_sum.attrs)
            return ds_out

        def _build_and_write(ds_daily: xr.Dataset) -> None:
            """Build seasonal/monthly climatologies from an opened dataset."""
            budget_vars = [v for v in ('dadt', 'adv', 'div', 'res') if v in ds_daily.data_vars]
            if not budget_vars:
                raise ValueError("Daily SICB file does not contain required budget variables.")
            ds_budget_daily = ds_daily[budget_vars]
            common_terms = ('dadt', 'adv', 'div', 'res')
            if all(v in ds_budget_daily.data_vars for v in common_terms):
                common_valid = (
                    np.isfinite(ds_budget_daily['dadt'])
                    & np.isfinite(ds_budget_daily['adv'])
                    & np.isfinite(ds_budget_daily['div'])
                    & np.isfinite(ds_budget_daily['res'])
                )
                keep_ratio = float(common_valid.astype(np.float32).mean().values)
                logger.info(
                    "Applying SICB common-valid aggregation mask (keep %.1f%% samples).",
                    keep_ratio * 100.0,
                )
                for vname in common_terms:
                    ds_budget_daily[vname] = ds_budget_daily[vname].where(common_valid)

            def _seasonal_mean_climatology(da: xr.DataArray) -> xr.DataArray:
                out = da.resample(time='QS-DEC').mean(skipna=True, keep_attrs=True)
                out = out.groupby('time.season').mean(dim='time', skipna=True, keep_attrs=True)
                out = out.reindex(season=season_order)
                out = out.rename({'season': 'time'})
                out = out.assign_coords(time=np.arange(1, 5, dtype=np.int16))
                return out

            def _monthly_mean_climatology(da: xr.DataArray) -> xr.DataArray:
                out = da.resample(time='MS').mean(skipna=True, keep_attrs=True)
                out = out.groupby('time.month').mean(dim='time', skipna=True, keep_attrs=True)
                out = out.reindex(month=month_order)
                out = out.rename({'month': 'time'})
                out = out.assign_coords(time=month_order)
                return out

            # Seasonal climatology:
            #   daily -> per-year seasonal sum -> multi-year mean.
            ds_seasonal = ds_budget_daily.resample(time='QS-DEC').sum(skipna=True, keep_attrs=True)
            if normalize_by_valid_days:
                ds_season_count = ds_budget_daily.notnull().resample(time='QS-DEC').sum()
                season_time = pd.to_datetime(ds_seasonal['time'].values)
                season_days = np.zeros((len(season_time),), dtype=np.float32)
                for ii, ts in enumerate(season_time):
                    # QS-DEC timestamps are anchored at Dec/Mar/Jun/Sep.
                    month = int(pd.Timestamp(ts).month)
                    if month == 12:
                        season_days[ii] = float(DAYS_PER_SEASON[0])  # DJF
                    elif month == 3:
                        season_days[ii] = float(DAYS_PER_SEASON[1])  # MAM
                    elif month == 6:
                        season_days[ii] = float(DAYS_PER_SEASON[2])  # JJA
                    elif month == 9:
                        season_days[ii] = float(DAYS_PER_SEASON[3])  # SON
                    else:
                        season_days[ii] = np.nan
                season_scale = xr.DataArray(
                    season_days, dims=('time',), coords={'time': ds_seasonal['time']}
                )
                ds_seasonal = _scale_sum_by_valid_count(ds_seasonal, ds_season_count, season_scale)
            ds_seasonal = ds_seasonal.groupby('time.season').mean(dim='time', skipna=True, keep_attrs=True)
            ds_seasonal = ds_seasonal.reindex(season=season_order)
            ds_seasonal = ds_seasonal.rename({'season': 'time'})
            ds_seasonal = ds_seasonal.assign_coords(time=np.arange(1, 5, dtype=np.int16))
            ds_seasonal['time'].attrs.update({
                'long_name': 'season_index',
                'description': '1=DJF,2=MAM,3=JJA,4=SON',
            })

            sic_mean = None
            div_mean = None
            res_mean = None
            if 'sic' in ds_daily:
                sic_mean = _seasonal_mean_climatology(ds_daily['sic']).astype(np.float32)
                sic_mean.attrs.update({
                    'long_name': 'Seasonal mean sea ice concentration',
                    'units': '%',
                })
                ds_seasonal['sic_mean'] = sic_mean
            if 'div' in ds_daily:
                div_mean = _seasonal_mean_climatology(ds_daily['div']).astype(np.float32)
                div_mean.attrs.update({
                    'long_name': 'Seasonal mean divergence term',
                    'units': '%/day',
                })
                ds_seasonal['div_mean'] = div_mean
            if 'res' in ds_daily:
                res_mean = _seasonal_mean_climatology(ds_daily['res']).astype(np.float32)
                res_mean.attrs.update({
                    'long_name': 'Seasonal mean residual term',
                    'units': '%/day',
                })
                ds_seasonal['res_mean'] = res_mean
            if (sic_mean is not None) and (div_mean is not None) and (res_mean is not None):
                ridging_raw = (
                    (sic_mean > 90.0)
                    & (res_mean < 0.0)
                    & (div_mean > 0.0)
                )
                ridging_valid = np.isfinite(sic_mean) & np.isfinite(div_mean) & np.isfinite(res_mean)
                ridging_mask = xr.where(ridging_valid, ridging_raw.astype(np.float32), np.nan).astype(np.float32)
                ridging_mask.attrs.update({
                    'long_name': 'Seasonal ridging mask',
                    'units': '1',
                    'description': '1=ridging region by seasonal mean criteria; NaN=invalid',
                    'criteria': 'sic_mean>90%,res_mean<0,div_mean>0',
                })
                ds_seasonal['ridging_mask'] = ridging_mask

            ds_seasonal.attrs.update({
                'sicb_normalized_by_valid_days': int(bool(normalize_by_valid_days)),
                'sicb_ridging_enabled': 1,
                'sicb_common_valid_aggregation': int(_SICB_COMMON_VALID_AGGREGATION),
            })

            for name in list(ds_seasonal.data_vars):
                if ds_seasonal[name].dtype.kind == 'f' and ds_seasonal[name].dtype.itemsize > 4:
                    ds_seasonal[name] = ds_seasonal[name].astype(np.float32)
            self._write_dataset_atomic(ds_seasonal, seas_clim_file)

            # Monthly climatology:
            #   daily -> per-month sum -> multi-year monthly mean.
            logger.info("Computing monthly climatology (xarray) ...")
            ds_monthly = ds_budget_daily.resample(time='MS').sum(skipna=True, keep_attrs=True)
            if normalize_by_valid_days:
                ds_month_count = ds_budget_daily.notnull().resample(time='MS').sum()
                month_time = pd.to_datetime(ds_monthly['time'].values)
                month_days = np.array(
                    [float(DAYS_PER_MONTH[int(pd.Timestamp(ts).month) - 1]) for ts in month_time],
                    dtype=np.float32,
                )
                month_scale = xr.DataArray(
                    month_days, dims=('time',), coords={'time': ds_monthly['time']}
                )
                ds_monthly = _scale_sum_by_valid_count(ds_monthly, ds_month_count, month_scale)
            ds_monthly = ds_monthly.groupby('time.month').mean(dim='time', skipna=True, keep_attrs=True)
            ds_monthly = ds_monthly.reindex(month=month_order)
            ds_monthly = ds_monthly.rename({'month': 'time'})
            ds_monthly = ds_monthly.assign_coords(time=month_order)
            ds_monthly['time'].attrs.update({
                'long_name': 'month',
                'units': '1-12',
            })

            if 'sic' in ds_daily:
                sic_monthly_mean = _monthly_mean_climatology(ds_daily['sic']).astype(np.float32)
                sic_monthly_mean.attrs.update({
                    'long_name': 'Monthly mean sea ice concentration climatology',
                    'units': '%',
                })
                ds_monthly['sic_mean'] = sic_monthly_mean
            if 'div' in ds_daily:
                div_monthly_mean = _monthly_mean_climatology(ds_daily['div']).astype(np.float32)
                div_monthly_mean.attrs.update({
                    'long_name': 'Monthly mean divergence climatology',
                    'units': '%/day',
                })
                ds_monthly['div_mean'] = div_monthly_mean
            if 'res' in ds_daily:
                res_monthly_mean = _monthly_mean_climatology(ds_daily['res']).astype(np.float32)
                res_monthly_mean.attrs.update({
                    'long_name': 'Monthly mean residual climatology',
                    'units': '%/day',
                })
                ds_monthly['res_mean'] = res_monthly_mean
            if 'ridging_ratio' in ds_daily:
                ridging_monthly_mean = _monthly_mean_climatology(ds_daily['ridging_ratio']).astype(np.float32)
                ridging_monthly_mean.attrs.update({
                    'long_name': 'Monthly mean ridging-grid ratio climatology',
                    'units': '1',
                })
                ds_monthly['ridging_ratio_mean'] = ridging_monthly_mean

            ds_monthly.attrs.update({
                'sicb_normalized_by_valid_days': int(bool(normalize_by_valid_days)),
                'sicb_ridging_enabled': 1,
                'sicb_common_valid_aggregation': int(_SICB_COMMON_VALID_AGGREGATION),
            })

            for name in list(ds_monthly.data_vars):
                if ds_monthly[name].dtype.kind == 'f' and ds_monthly[name].dtype.itemsize > 4:
                    ds_monthly[name] = ds_monthly[name].astype(np.float32)
            self._write_dataset_atomic(ds_monthly, mon_clim_file)

        logger.info("Computing seasonal climatology (xarray) ...")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt == 1:
                    with xr.open_dataset(daily_file) as ds_daily:
                        _build_and_write(ds_daily)
                else:
                    # Fallback path for rare transient NetCDF backend handle issues:
                    # eagerly load once, then compute climatologies from in-memory arrays.
                    with xr.open_dataset(daily_file) as ds_daily:
                        ds_mem = ds_daily.load()
                    _build_and_write(ds_mem)
                break
            except RuntimeError as exc:
                transient_nc = 'NetCDF: Not a valid ID' in str(exc)
                if not transient_nc or attempt >= max_attempts:
                    raise
                wait_seconds = float(attempt) * 0.5
                logger.warning(
                    "Transient NetCDF handle error while computing SICB climatology "
                    "(attempt %d/%d): %s. Retrying in %.1fs ...",
                    attempt, max_attempts, exc, wait_seconds,
                )
                time.sleep(wait_seconds)

        return seas_clim_file, mon_clim_file

    def SICB_1M_metrics(self, sicb_file):
        """
        Purpose
        -------
        Calculate the area integral of each component to elucidate the contribution of dynamics
            (divergence and advection) and thermodynamics (the residual) to the total ice budget

        Parameters
        ----------
        sicb_file: <string> .nc file's name
            including 4 <3-d array> in shape of (nt, nx, ny), named dadt, adv, div, res,
            which are calculated by ** SITool_sicb **,
            could be seasonal climatological, monthly mean, and monthly climatological, etc.

        Returns
        -------
        A dict variable contains the following metrics, X represent adv/div/res
            · Area integral of positive values (nt, ):                                         dadt_psum, X_psum, dyn_psum
            · Area integral of negative values (nt, ):                                         dadt_nsum, X_nsum, dyn_nsum
            · Area integral of all values (nt, ):                                              dadt_sum, X_sum
            · Area integral of X component as proportions of the total change in SIC (nt, ):   X_p

        Notes
        -----
        The res term contains not only thermodynamic processes (freezing and melting) but also sea ice mechanical
             redistribution (ridging and rafting). The mechanical redistribution is small in observations,
             but large in many model results. So be careful when interpreting the area integral results.

        Reference
        ---------
        Holmes, C. R., Holland, P. R., & Bracegirdle, T. J. (2019). Compensating Biases and a Noteworthy Success in
            the CMIP5 Representation of Antarctic Sea Ice Processes. Geophysical Research Letters, 46(8), 4299–4307.
            https://doi.org/10.1029/2018GL081796
        Uotila, P., Holland, P. R., Vihma, T., Marsland, S. J., & Kimura, N. (2014). Is realistic Antarctic sea-ice
            extent in climate models the result of excessive ice drift? Ocean Modelling, 79, 33–42.
            https://doi.org/10.1016/j.ocemod.2014.04.004
        """
        logger.info("Calculating SICB metrics for %s ...", sicb_file)

        with xr.open_dataset(sicb_file) as ds:
            dadt = np.array(ds['dadt'])
            adv = np.array(ds['adv'])
            div = np.array(ds['div'])
            res = np.array(ds['res'])

        # Validate input dimensions
        if dadt.ndim != 3:
            raise ValueError("Budget components must be 3D arrays [time, x, y].")
        if self.cell_area.shape != dadt[0, :, :].shape:
            raise ValueError(f"Grid cell area and budget data must have same spatial dimensions.")

        nt, nx, ny = dadt.shape

        # Main calculation logic integrated here
        dadt_sum = np.full(nt, np.nan)
        adv_psum, adv_nsum = np.full(nt, np.nan), np.full(nt, np.nan)
        div_psum, div_nsum = np.full(nt, np.nan), np.full(nt, np.nan)
        res_psum, res_nsum = np.full(nt, np.nan), np.full(nt, np.nan)
        for jt in range(nt):
            # Calculate area integrals
            dadt_sum[jt] = np.nansum(dadt[jt] * self.cell_area_km2)

            # Positive and negative parts
            adv_psum[jt] = self._integrate_positive_negative(adv[jt], '+')
            adv_nsum[jt] = self._integrate_positive_negative(adv[jt], '-')
            div_psum[jt] = self._integrate_positive_negative(div[jt], '+')
            div_nsum[jt] = self._integrate_positive_negative(div[jt], '-')
            res_psum[jt] = self._integrate_positive_negative(res[jt], '+')
            res_nsum[jt] = self._integrate_positive_negative(res[jt], '-')

        # Calculate derived metrics
        adv_sum = adv_psum + adv_nsum
        div_sum = div_psum + div_nsum
        res_sum = res_psum + res_nsum

        # Proportional contributions
        dadt_abs = np.where(dadt_sum != 0, np.abs(dadt_sum), np.nan)
        adv_p = np.where(dadt_abs != 0, adv_sum / dadt_abs * 100, np.nan)
        div_p = np.where(dadt_abs != 0, div_sum / dadt_abs * 100, np.nan)
        res_p = np.where(dadt_sum != 0, res_sum / dadt_sum * 100, np.nan)

        return {
            'dadt_sum': dadt_sum,
            'adv_psum': adv_psum, 'adv_nsum': adv_nsum, 'adv_sum': adv_sum, 'adv_p': adv_p,
            'div_psum': div_psum, 'div_nsum': div_nsum, 'div_sum': div_sum, 'div_p': div_p,
            'res_psum': res_psum, 'res_nsum': res_nsum, 'res_sum': res_sum, 'res_p': res_p
        }

    def _integrate_positive_negative(self, data: np.ndarray, sign: str) -> float:
        """Calculate area integral of positive or negative values."""
        if sign == '+':
            mask = data > 0
        elif sign == '-':
            mask = data < 0
        else:
            raise ValueError("Sign must be '+' or '-'")

        if np.any(mask):
            return np.nansum(data[mask] * self.cell_area_km2[mask])
        else:
            return 0.0

    def summarize_budget_closure(self, mon_clim_file: str,
                                dataset_name: Optional[str] = None) -> Dict[str, float]:
        """Summarize SICB annual budget closure terms for one monthly climatology file.

        Terms (paper-style):
            total = dadt
            dyn = adv + div
            therm = res
            residual_res = total - (dyn + therm)

        Returns a single-row dictionary for HTML scalar tables with units
        10^6 km^2/month.
        """

        def _to_3d(field: xr.DataArray) -> np.ndarray:
            arr = np.array(field, dtype=float)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            return arr

        def _annual_integrated_million_km2(term_3d: np.ndarray) -> float:
            ntime = term_3d.shape[0]
            monthly_vals = np.full((ntime,), np.nan)

            for jt in range(ntime):
                term_map = term_3d[jt, :, :]
                valid = np.isfinite(term_map) & np.isfinite(self.cell_area_km2)
                if not np.any(valid):
                    continue

                area_valid = self.cell_area_km2[valid]
                term_valid = term_map[valid]
                area_sum = np.nansum(area_valid)
                if area_sum <= 0:
                    continue

                # Area-weighted monthly mean (%/month), then convert to integrated
                # 10^6 km^2/month using the same percent×area basis as existing code.
                weighted_mean = np.nansum(term_valid * area_valid) / area_sum
                monthly_vals[jt] = weighted_mean * (area_sum / 100.0) / 1e6

            return float(np.nanmean(monthly_vals))

        with xr.open_dataset(mon_clim_file) as ds:
            total_3d = _to_3d(ds['dadt'])
            dyn_3d = _to_3d(ds['adv'] + ds['div'])
            therm_3d = _to_3d(ds['res'])
            residual_3d = total_3d - (dyn_3d + therm_3d)

        total_change = _annual_integrated_million_km2(total_3d)
        dynamic_term = _annual_integrated_million_km2(dyn_3d)
        thermodynamic_term = _annual_integrated_million_km2(therm_3d)
        residual_res = _annual_integrated_million_km2(residual_3d)

        row = {
            'total_change': total_change,
            'dynamic_term': dynamic_term,
            'thermodynamic_term': thermodynamic_term,
            'residual_res': residual_res,
            'sum_check': abs(total_change - (dynamic_term + thermodynamic_term + residual_res)),
        }
        if dataset_name is not None:
            row['dataset_name'] = dataset_name

        return row

    def summarize_seasonal_budget_table(self, seas_clim_file: str,
                                       dataset_name: Optional[str] = None,
                                       sector: str = 'All') -> Dict[str, Dict[str, float]]:
        """Summarize SICB seasonal integrated terms from one *_SeasClim.nc file.

        Returns a dict keyed by canonical seasons (Spring, Summer, Autumn, Winter),
        where each season contains integrated totals of dadt/adv/div/res and
        percentage contributions relative to dadt.
        """
        sec_index = utils.region_index(
            grid_file=self.grid_file,
            hms=self.hemisphere,
            sector=sector,
        )

        def _integrated_total(term_map: np.ndarray) -> float:
            valid = np.isfinite(term_map) & np.isfinite(self.cell_area_km2) & sec_index
            if not np.any(valid):
                return np.nan
            # percent × area integration to 10^6 km^2/month
            return float(np.nansum(term_map[valid] * self.cell_area_km2[valid]) / 100.0 / 1e6)

        with xr.open_dataset(seas_clim_file) as ds:
            dadt = np.array(ds['dadt'], dtype=float)
            adv = np.array(ds['adv'], dtype=float)
            div = np.array(ds['div'], dtype=float)
            res = np.array(ds['res'], dtype=float)

        if dadt.ndim != 3 or adv.ndim != 3 or div.ndim != 3 or res.ndim != 3:
            raise ValueError("Seasonal budget components must be 3D arrays [season, x, y].")
        if dadt.shape[0] < 4:
            raise ValueError(f"Expected at least 4 seasonal slices, got {dadt.shape[0]}.")
        if dadt.shape[1:] != self.cell_area_km2.shape:
            raise ValueError(
                "Grid cell area and budget data must have same spatial dimensions."
                f"\n\tshape of dadt spatial dims: {dadt.shape[1:]}"
                f"\n\tshape of cell_area: {self.cell_area_km2.shape}"
            )

        # Input file seasonal order from CDO yseassum: DJF, MAM, JJA, SON
        if self.hemisphere == 'nh':
            season_to_idx = {
                'Spring': 1,  # MAM
                'Summer': 2,  # JJA
                'Autumn': 3,  # SON
                'Winter': 0,  # DJF
            }
        else:
            season_to_idx = {
                'Spring': 3,  # SON
                'Summer': 0,  # DJF
                'Autumn': 1,  # MAM
                'Winter': 2,  # JJA
            }

        out = {}
        for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
            idx = season_to_idx[season]
            dadt_total = _integrated_total(dadt[idx])
            adv_total = _integrated_total(adv[idx])
            div_total = _integrated_total(div[idx])
            res_total = _integrated_total(res[idx])

            if np.isfinite(dadt_total) and dadt_total != 0:
                adv_pct = adv_total / dadt_total * 100.0
                div_pct = div_total / dadt_total * 100.0
                res_pct = res_total / dadt_total * 100.0
            else:
                adv_pct = np.nan
                div_pct = np.nan
                res_pct = np.nan

            out[season] = {
                'dadt': dadt_total,
                'adv': adv_total,
                'adv_dadt_pct': adv_pct,
                'div': div_total,
                'div_dadt_pct': div_pct,
                'res': res_total,
                'res_dadt_pct': res_pct,
            }

        if dataset_name is not None:
            out['dataset_name'] = dataset_name

        return out

    def summarize_budget_components(self, obs_mon_clim_file: str, target_mon_clim_file: str,
                                    aggregate: str = 'annual') -> List[Dict[str, float]]:
        """Summarize SICB component statistics for scalar table output.

        Provenance note:
        - This method follows SIC budget equations already documented in this codebase
          (Holland & Kimura, 2016; Holmes et al., 2019; Nie et al., 2022).
        - PDF page-level extraction is unavailable in this environment, so references are
          taken from in-code docstrings.

        Equations used (monthly climatology fields):
        - $\\Delta SIC = Dyn + Therm + Res$
        - $Dyn = Adv + Div$
        - $Bias = \\overline{M - O}$

        Notes:
        - Therm is represented by the residual proxy ("res") that contains
          thermodynamic + unresolved mechanical contributions.
        - Res is defined as closure residual: dadt - dyn - therm.
        """

        def _collapse_to_map(field: xr.DataArray) -> np.ndarray:
            if 'time' not in field.dims:
                return np.array(field, dtype=float)
            if aggregate == 'annual':
                return np.array(field.mean(dim='time', skipna=True), dtype=float)
            if aggregate == 'seasonal':
                by_season = field.groupby('time.season').mean(dim='time', skipna=True)
                return np.array(by_season.mean(dim='season', skipna=True), dtype=float)
            raise ValueError("aggregate must be 'annual' or 'seasonal'.")

        def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
            valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
            if np.sum(valid) < 2:
                return np.nan
            x = a[valid].reshape(-1)
            y = b[valid].reshape(-1)
            ww = w[valid].reshape(-1)
            wsum = np.sum(ww)
            if wsum <= 0:
                return np.nan
            ww = ww / wsum

            mx = np.sum(ww * x)
            my = np.sum(ww * y)
            cov = np.sum(ww * (x - mx) * (y - my))
            vx = np.sum(ww * (x - mx) ** 2)
            vy = np.sum(ww * (y - my) ** 2)
            if vx <= 0 or vy <= 0:
                return np.nan
            return float(cov / np.sqrt(vx * vy))

        def _integrated_series_million_km2(field: xr.DataArray) -> np.ndarray:
            arr = np.array(field, dtype=float)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            # Convert percent to fraction and integrate area.
            return np.nansum(arr * (self.cell_area_km2 / 100.0)[np.newaxis, :, :], axis=(1, 2)) / 1e6

        with xr.open_dataset(obs_mon_clim_file) as obs_ds, xr.open_dataset(target_mon_clim_file) as tgt_ds:
            obs_dyn = obs_ds['adv'] + obs_ds['div']
            tgt_dyn = tgt_ds['adv'] + tgt_ds['div']

            # Residual proxy for thermodynamic + unresolved mechanical contribution.
            obs_therm = obs_ds['res']
            tgt_therm = tgt_ds['res']

            # Closure residual (expected to be near zero): Res = dadt - Dyn - Therm.
            obs_res = obs_ds['dadt'] - obs_dyn - obs_therm
            tgt_res = tgt_ds['dadt'] - tgt_dyn - tgt_therm

            components = [
                ('Dyn', obs_dyn, tgt_dyn),
                ('Therm', obs_therm, tgt_therm),
                ('Res', obs_res, tgt_res),
            ]

            rows: List[Dict[str, float]] = []
            for comp_name, obs_comp, tgt_comp in components:
                obs_map = _collapse_to_map(obs_comp)
                tgt_map = _collapse_to_map(tgt_comp)

                corr = _weighted_corr(obs_map, tgt_map, self.cell_area)
                mean_value = float(np.nanmean(_integrated_series_million_km2(tgt_comp)))
                bias = float(np.nanmean(_integrated_series_million_km2(tgt_comp - obs_comp)))

                rows.append({
                    'component': comp_name,
                    'mean_value': mean_value,
                    'correlation_with_obs': corr,
                    'bias_vs_obs': bias,
                })

        return rows

    def SICB_2M_metrics(self, sicb1_file: str, sicb2_file: str) -> Dict[str, Union[float, np.ndarray]]:
        """
        Purpose
        -------
        Calculate the difference between two SIC budget datasets, the metrics include RMSE, MAE, Corr, and AIPE
        and time series' difference

        Args:
            sicb1_file: Reference dataset file (typically observations).
            sicb2_file: Model or comparison dataset file.

        Returns
        -------
        A dict variable contains the following metrics, X represent dadt/adv/div/res
            ·  Mean difference in each time slice (nt, ):                     X_diff
            ·  Mean state difference <float value>:                           X_mean_diff
            ·  Error of the area integral of SIC budget component
               as a proportion of the total SIC change (AIPE) (nt, ):         AIPE_adv, AIPE_div

        Notes
        -----
        We recommand to do the diagnosis on seasonal scale, as there are high uncertainties on ice drift observations
            and thus SIC budget observations. Diagnostic on seasonal scale would reduce the impact of uncertainty.

        Reference
        ---------


        """
        # === Read data ===
        with xr.open_dataset(sicb1_file) as ds:
            dadt1, adv1, div1, res1 = np.array(ds['dadt']), np.array(ds['adv']), np.array(ds['div']), np.array(
                ds['res'])
        with xr.open_dataset(sicb2_file) as ds:
            dadt2, adv2, div2, res2 = np.array(ds['dadt']), np.array(ds['adv']), np.array(ds['div']), np.array(
                ds['res'])

        # === Check inputs ===
        if dadt2.shape != dadt1.shape:
            raise ValueError(f'dadt1 and dadt2 must have the same shape!')
        if self.cell_area.shape != dadt1[0, :, :].shape:
            raise ValueError(f'dadt1 and cell_area must have the same spatial shape!'
                             f'\n\tshape of dadt1: {dadt1.shape}'
                             f'\n\tshape of cell_area: {self.cell_area.shape}')

        nt, nx, ny = dadt1.shape

        # === Calculate SIC budget metrics ===
        dadt_diff = np.full((nt,), np.nan)
        adv_diff = np.full((nt,), np.nan)
        div_diff = np.full((nt,), np.nan)
        res_diff = np.full((nt,), np.nan)
        for m in range(nt):
            dadt_diff[m] = utils.MatrixDiff(dadt1[m, :, :], dadt2[m, :, :], self.cell_area, self.metric, mask=True)
            adv_diff[m] = utils.MatrixDiff(adv1[m, :, :], adv2[m, :, :], self.cell_area, self.metric, mask=True)
            div_diff[m] = utils.MatrixDiff(div1[m, :, :], div2[m, :, :], self.cell_area, self.metric, mask=True)
            res_diff[m] = utils.MatrixDiff(res1[m, :, :], res2[m, :, :], self.cell_area, self.metric, mask=True)

        # --- mean diff ---
        dadt_mean_diff = np.sum(dadt_diff * DAYS_PER_SEASON) / np.sum(DAYS_PER_SEASON)
        adv_mean_diff = np.sum(adv_diff * DAYS_PER_SEASON) / np.sum(DAYS_PER_SEASON)
        div_mean_diff = np.sum(div_diff * DAYS_PER_SEASON) / np.sum(DAYS_PER_SEASON)
        res_mean_diff = np.sum(res_diff * DAYS_PER_SEASON) / np.sum(DAYS_PER_SEASON)

        sicb1 = self.SICB_1M_metrics(sicb_file=sicb1_file)
        sicb2 = self.SICB_1M_metrics(sicb_file=sicb2_file)

        AIPE_adv = sicb2['adv_p'] - sicb1['adv_p']  # positive: model overestimates adv-driven ice increase; negative: underestimates
        AIPE_div = sicb2['div_p'] - sicb1['div_p']  # positive: overestimates convergence contribution; negative: overestimates divergence
        AIPE_res_p = (sicb2['res_psum'] / abs(sicb2['dadt_sum']) - sicb1['res_psum'] / abs(
            sicb1['dadt_sum'])) * 100  # positive: overestimates freezing; negative: underestimates freezing
        AIPE_res_n = (sicb2['res_nsum'] / abs(sicb2['dadt_sum']) - sicb1['res_nsum'] / abs(
            sicb1['dadt_sum'])) * 100  # negative: overestimates melting; positive: underestimates melting

        return {'dadt_diff': dadt_diff, 'dadt_mean_diff': dadt_mean_diff,
                'adv_diff': adv_diff, 'adv_mean_diff': adv_mean_diff,
                'div_diff': div_diff, 'div_mean_diff': div_mean_diff,
                'res_diff': res_diff, 'res_mean_diff': res_mean_diff,
                'AIPE_adv': np.nanmean(AIPE_adv), 'AIPE_div': np.nanmean(AIPE_div),
                'AIPE_res_p': np.nanmean(AIPE_res_p), 'AIPE_res_n': np.nanmean(AIPE_res_n)}
