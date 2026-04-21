# -*- coding: utf-8 -*-
"""Sea-ice metric implementations by diagnostic family."""

import datetime as _dt
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.ndimage import convolve1d

from scripts import utils
from scripts.sea_ice_metrics.base import SeaIceMetricsBase

logger = logging.getLogger(__name__)


class SItransMetrics(SeaIceMetricsBase):
    """Calculate sea ice transition (advance and retreat) dates."""

    def __init__(self, grid_file: str, date_sta: str, date_end: str,
                 hemisphere: str = None, metric: str = 'MAE',
                 threshold: float = 15.0, smooth_window_days: int = 15,
                 persistence_days: int = 7, trend_sig_p: float = 0.05):
        """Initialize the sea ice transition metrics calculator.

        Args:
            grid_file: NetCDF file with 'cell_area' variable (units: m²).
            date_sta: Start date string ('YYYY-MM-DD').
            date_end: End date string ('YYYY-MM-DD').
            hemisphere: Target hemisphere ('nh' or 'sh').
            metric: Statistical metric for comparisons.
            threshold: SIC threshold (%) used for transition-date detection.
            smooth_window_days: Running-mean window length (days).
            persistence_days: Consecutive-day persistence requirement.
            trend_sig_p: P-value threshold for trend significance masks.
        """
        super().__init__(grid_file, hemisphere, metric)

        self.date_sta = date_sta
        self.date_end = date_end

        try:
            self.default_threshold = float(threshold)
        except Exception:
            self.default_threshold = 15.0

        try:
            self.sequence_length = max(2, int(round(float(persistence_days))))
        except Exception:
            self.sequence_length = 7

        # Smoothing window: 15-day running mean to remove short-term dynamical events.
        # (Lebrun et al., 2019, TC; Himmich et al., 2024, NC)
        try:
            window_size = max(1, int(round(float(smooth_window_days))))
        except Exception:
            window_size = 15
        self.smooth_window_days = int(window_size)
        self.window = np.ones(window_size) / window_size

        try:
            _p = float(trend_sig_p)
        except Exception:
            _p = 0.05
        self.trend_sig_p = _p if 0.0 < _p < 1.0 else 0.05

        date_format = "%Y-%m-%d"
        date_start_dt = _dt.datetime.strptime(date_sta, date_format)
        date_end_dt = _dt.datetime.strptime(date_end, date_format)
        self.date_range = np.arange(date_start_dt, date_end_dt, _dt.timedelta(days=1))

        self.year_sta, self.year_end = date_start_dt.year, date_end_dt.year
        self.year = np.arange(self.year_sta, self.year_end + 1)

    def _transition_cycle_start(self) -> Tuple[int, int]:
        """Return (month, day) for the hemisphere-specific transition-year start.

        Physical rationale:
        - NH starts on 1 Sep because Arctic minimum sea-ice extent occurs in
          late boreal summer; this places autumn advance and following spring
          retreat in one contiguous transition year.
        - SH starts on 1 Mar because Antarctic minimum occurs in late austral
          summer; this similarly keeps advance and retreat within one cycle.
        """
        if self.hemisphere == 'nh':
            return 9, 1
        return 3, 1

    def _get_transition_window(self, start_year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Build hemisphere-aware [start, end] timestamps for one transition year."""
        start_month, start_day = self._transition_cycle_start()
        start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
        # One full transition year minus one day (365/366-day safe).
        end_date = start_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        return start_date, end_date

    @staticmethod
    def _time_values_to_keys(time_values) -> list[Tuple[int, int, int]]:
        """Convert decoded time coordinate values into `(year, month, day)` keys."""
        vals = np.asarray(time_values).ravel()
        if vals.size == 0:
            return []

        first = vals[0]
        if hasattr(first, 'year') and hasattr(first, 'month'):
            keys = []
            for item in vals:
                yy = int(getattr(item, 'year'))
                mm = int(getattr(item, 'month'))
                dd = int(getattr(item, 'day', 1))
                keys.append((yy, mm, dd))
            return keys

        time_vals = pd.to_datetime(vals)
        return [(int(t.year), int(t.month), int(t.day)) for t in time_vals]

    def _transition_mask(self, time_keys: list[Tuple[int, int, int]], start_year: int) -> np.ndarray:
        """Return boolean mask for one hemisphere-aware transition-year window."""
        start_month, start_day = self._transition_cycle_start()
        end_year = int(start_year) + 1
        mask = np.zeros(len(time_keys), dtype=bool)

        for idx, (yy, mm, dd) in enumerate(time_keys):
            in_first_part = (
                int(yy) == int(start_year)
                and (int(mm) > start_month or (int(mm) == start_month and int(dd) >= start_day))
            )
            in_second_part = (
                int(yy) == end_year
                and (int(mm) < start_month or (int(mm) == start_month and int(dd) < start_day))
            )
            mask[idx] = in_first_part or in_second_part
        return mask

    def cal_sitrans(self, sic_file, threshold=None, sic_name='siconc',
                    advance_ctrl=True, retreat_ctrl=True):
        """Calculate sea ice advance and retreat dates for each year.

        Uses a 15-day running mean to remove short-term dynamical events.
        Advance date: first day SIC rises above threshold and stays above for
            at least sequence_length days.
        Retreat date: first day SIC falls below threshold and stays below for
            at least sequence_length days.

        Args:
            sic_file: NetCDF file with daily sea ice concentration.
            threshold: SIC threshold (%) for advance/retreat detection.
            sic_name: Variable name for SIC in the file.
            advance_ctrl: Remove advance dates that exceed retreat dates.
            retreat_ctrl: Remove retreat dates that precede advance dates.

        Returns:
            Dict with 'advance_day' and 'retreat_day' lists (one array per year).

        Notes:
            The SIC threshold choice has no significant impact on results.
            (Lebrun et al., 2019, TC)
        """
        if threshold is None:
            threshold = self.default_threshold

        try:
            threshold = float(threshold)
        except Exception:
            threshold = self.default_threshold

        i_retreat_results, i_advance_results = [], []
        annual_data_results, annual_data_results_ma = [], []
        cycle_lengths: list = []
        start_years: list = []

        with xr.open_dataset(sic_file) as ds:
            sic_da = ds[sic_name].astype('float32')
            lon = ds['lon']

            time_candidates = [
                dim for dim in sic_da.dims
                if ('time' in dim.lower() and 'bnds' not in dim.lower() and 'bounds' not in dim.lower())
            ]
            if not time_candidates:
                raise ValueError(f"No time-like dimension found in {sic_file} for variable {sic_name}.")
            time_name = time_candidates[0]

            # Harmonize SIC units across products: many datasets provide 0–100 (%),
            # while some CMIP products provide 0–1 (fraction). SItrans threshold is
            # defined in percent space (default 15), so convert fractional inputs.
            try:
                n_sample = int(min(366, int(sic_da.sizes.get(time_name, 0))))
                sample = sic_da.isel({time_name: slice(0, n_sample)}).values if n_sample > 0 else sic_da.values
                sample = np.asarray(sample, dtype=float)
                sample_finite = sample[np.isfinite(sample)]
                sample_hi = float(np.nanpercentile(sample_finite, 99)) if sample_finite.size else np.nan
            except Exception:
                sample_hi = np.nan
            if np.isfinite(sample_hi) and sample_hi <= 1.5 and float(threshold) > 1.0:
                sic_da = sic_da * 100.0
                logger.info(
                    "SItrans auto-unit conversion applied (%s): detected fractional SIC (p99=%.3f), scaled to percent.",
                    str(sic_file),
                    sample_hi,
                )

            time_values = sic_da[time_name].values if time_name in sic_da.coords else ds[time_name].values
            time_keys = self._time_values_to_keys(time_values)

            for start_year in np.arange(self.year_sta, self.year_end):
                # Hemisphere-aware transition-year window:
                # - NH: Sep 1 -> Aug 31 (captures autumn advance then spring retreat)
                # - SH: Mar 1 -> Feb 28/29 (captures austral advance then retreat)
                mask = self._transition_mask(time_keys, int(start_year))
                if np.any(mask):
                    annual_data = sic_da.isel({time_name: np.flatnonzero(mask)})
                else:
                    annual_data = sic_da.isel({time_name: slice(0, 0)})

                time_size = annual_data.shape[0]
                cycle_lengths.append(int(time_size))
                start_years.append(int(start_year))

                annual_values = annual_data.values
                if time_size <= self.sequence_length + 1:
                    i_retreat_results.append(np.full(lon.shape, np.nan))
                    i_advance_results.append(np.full(lon.shape, np.nan))
                    annual_data_results.append(annual_values)
                    annual_data_results_ma.append(annual_values)
                    continue

                smoothed_values = convolve1d(annual_values, self.window, axis=0, mode='nearest')
                i_retreat_result = np.full(lon.shape, np.nan)
                i_advance_result = np.full(lon.shape, np.nan)

                # Detect retreat: SIC drops below threshold and stays below
                for t in range(time_size - self.sequence_length, 0, -1):
                    cond1 = (smoothed_values[t, :, :] <= threshold) & (smoothed_values[t - 1, :, :] > threshold)
                    cond2 = np.all(smoothed_values[t + 1:t + self.sequence_length, :, :] < threshold, axis=0)
                    valid = cond1 & cond2
                    i_retreat_result[valid & np.isnan(i_retreat_result)] = t

                # Detect advance: SIC rises above threshold and stays above
                for t in range(1, time_size - self.sequence_length):
                    cond4 = (smoothed_values[t, :, :] >= threshold) & (smoothed_values[t - 1, :, :] < threshold)
                    cond5 = np.all(smoothed_values[t + 1:t + self.sequence_length, :, :] > threshold, axis=0)
                    valid = cond4 & cond5
                    i_advance_result[valid & np.isnan(i_advance_result)] = t

                # Keep original QC intent but derive limits from cycle length so
                # behavior remains consistent across hemispheres and leap years.
                early_cutoff = int(round(0.28 * time_size))
                late_cutoff = int(round(0.66 * time_size))

                if advance_ctrl:
                    idx = (i_advance_result > i_retreat_result) | (i_advance_result > late_cutoff)
                    i_advance_result[idx] = np.nan
                if retreat_ctrl:
                    idx = (i_retreat_result < i_advance_result) | (i_retreat_result < early_cutoff)
                    i_retreat_result[idx] = np.nan

                annual_data_results.append(annual_values)
                annual_data_results_ma.append(smoothed_values)
                i_retreat_results.append(i_retreat_result)
                i_advance_results.append(i_advance_result)

        return {
            'advance_day': i_advance_results,
            'retreat_day': i_retreat_results,
            'cycle_length': np.asarray(cycle_lengths, dtype=float),
            'start_year': np.asarray(start_years, dtype=int),
        }

    def SItrans_2M_metrics(self, sic_file1, sic_file2,
                           sic_name1='siconc', sic_name2='siconc',
                           sector: str = 'All'):
        """Calculate SItrans diagnostics for observation-vs-model (or obs1-vs-obs2).

        Returns a structured dictionary containing climatology, bias/variability/
        trend maps, and scalar summary statistics for advance/retreat dates.
        """

        def _calc_trend_map(data_3d: np.ndarray, min_valid_ratio: float = 0.6):
            """Per-grid trend, p-value, and valid-sample count."""
            n_years, nx, ny = data_3d.shape
            years = np.arange(n_years, dtype=float)
            trend = np.full((nx, ny), np.nan, dtype=float)
            pvalue = np.full((nx, ny), np.nan, dtype=float)
            valid_count = np.full((nx, ny), np.nan, dtype=float)
            min_valid = max(3, int(np.ceil(min_valid_ratio * n_years)))

            for ii in range(nx):
                for jj in range(ny):
                    ts = data_3d[:, ii, jj]
                    valid = np.isfinite(ts)
                    n_valid = int(np.sum(valid))
                    if n_valid < min_valid:
                        continue
                    reg = stats.linregress(years[valid], ts[valid])
                    trend[ii, jj] = float(reg.slope)
                    pvalue[ii, jj] = float(reg.pvalue)
                    valid_count[ii, jj] = float(n_valid)

            return trend, pvalue, valid_count

        def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
            """Area-weighted Pearson correlation for finite paired grid cells."""
            valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
            if np.sum(valid) < 3:
                return np.nan
            x = a[valid].reshape(-1).astype(float)
            y = b[valid].reshape(-1).astype(float)
            ww = w[valid].reshape(-1).astype(float)
            wsum = np.sum(ww)
            if not np.isfinite(wsum) or wsum <= 0:
                return np.nan
            ww = ww / wsum
            xbar = np.sum(ww * x)
            ybar = np.sum(ww * y)
            cov = np.sum(ww * (x - xbar) * (y - ybar))
            varx = np.sum(ww * (x - xbar) ** 2)
            vary = np.sum(ww * (y - ybar) ** 2)
            if varx <= 0 or vary <= 0:
                return np.nan
            return float(cov / np.sqrt(varx * vary))

        def _weighted_mean_field(arr: np.ndarray, w: np.ndarray) -> float:
            valid = np.isfinite(arr) & np.isfinite(w) & (w > 0)
            if np.sum(valid) < 1:
                return np.nan
            ww = w[valid].astype(float)
            vv = arr[valid].astype(float)
            wsum = np.sum(ww)
            if wsum <= 0:
                return np.nan
            return float(np.sum(vv * ww) / wsum)

        def _area_weighted_mean_series(data_3d: np.ndarray, w: np.ndarray) -> np.ndarray:
            n_years = data_3d.shape[0]
            series = np.full((n_years,), np.nan, dtype=float)
            for yy in range(n_years):
                series[yy] = _weighted_mean_field(data_3d[yy, :, :], w)
            return series

        def _safe_matrix_diff(obs_map: np.ndarray, mod_map: np.ndarray, metric_name: str) -> float:
            try:
                val = utils.MatrixDiff(
                    obs_map, mod_map,
                    weights=self.cell_area,
                    metric=metric_name,
                    mask=True,
                )
                valf = float(val)
                return valf if np.isfinite(valf) else np.nan
            except Exception:
                return np.nan

        def _weighted_sig_fraction(sigmask: np.ndarray, w: np.ndarray) -> float:
            valid = np.isfinite(sigmask) & np.isfinite(w) & (w > 0)
            if np.sum(valid) < 1:
                return np.nan
            ww = w[valid].astype(float)
            vv = sigmask[valid].astype(float)
            denom = np.sum(ww)
            if denom <= 0:
                return np.nan
            return float(100.0 * np.sum(ww * vv) / denom)

        def _calc_relationship(ret_series: np.ndarray, adv_series: np.ndarray) -> Dict[str, float]:
            def _fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
                valid = np.isfinite(x) & np.isfinite(y)
                if np.sum(valid) < 3:
                    return {
                        'corr': np.nan,
                        'slope': np.nan,
                        'pvalue': np.nan,
                        'r2': np.nan,
                    }
                xv = np.asarray(x[valid], dtype=float)
                yv = np.asarray(y[valid], dtype=float)
                reg = stats.linregress(xv, yv)
                corr = stats.pearsonr(xv, yv)[0] if xv.size >= 3 else np.nan
                return {
                    'corr': float(corr) if np.isfinite(corr) else np.nan,
                    'slope': float(reg.slope) if np.isfinite(reg.slope) else np.nan,
                    'pvalue': float(reg.pvalue) if np.isfinite(reg.pvalue) else np.nan,
                    'r2': float(reg.rvalue ** 2) if np.isfinite(reg.rvalue) else np.nan,
                }

            lag0 = _fit(ret_series, adv_series)
            if ret_series.size >= 2 and adv_series.size >= 2:
                lag1 = _fit(ret_series[:-1], adv_series[1:])
            else:
                lag1 = {'corr': np.nan, 'slope': np.nan, 'pvalue': np.nan, 'r2': np.nan}
            return {'lag0': lag0, 'lag1': lag1}

        sitr1 = self.cal_sitrans(sic_file=sic_file1, threshold=self.default_threshold, sic_name=sic_name1)
        sitr2 = self.cal_sitrans(sic_file=sic_file2, threshold=self.default_threshold, sic_name=sic_name2)

        advance_obs = np.array(sitr1['advance_day'], dtype=float)
        advance_mod = np.array(sitr2['advance_day'], dtype=float)
        retreat_obs = np.array(sitr1['retreat_day'], dtype=float)
        retreat_mod = np.array(sitr2['retreat_day'], dtype=float)
        cycle_length_obs = np.array(sitr1.get('cycle_length', []), dtype=float)
        cycle_length_mod = np.array(sitr2.get('cycle_length', []), dtype=float)
        year_axis = np.array(sitr1.get('start_year', []), dtype=float)
        if year_axis.size != advance_obs.shape[0]:
            year_axis = np.arange(advance_obs.shape[0], dtype=float) + float(self.year_sta)

        sec_index = utils.region_index(
            grid_file=self.grid_file,
            hms=self.hemisphere,
            sector=sector,
        )
        advance_obs[:, ~sec_index] = np.nan
        advance_mod[:, ~sec_index] = np.nan
        retreat_obs[:, ~sec_index] = np.nan
        retreat_mod[:, ~sec_index] = np.nan

        # Climatology fields
        advance_clim_obs = np.nanmean(advance_obs, axis=0)
        advance_clim_mod = np.nanmean(advance_mod, axis=0)
        retreat_clim_obs = np.nanmean(retreat_obs, axis=0)
        retreat_clim_mod = np.nanmean(retreat_mod, axis=0)

        # Mean-bias maps (model - obs)
        advance_bias_map = advance_clim_mod - advance_clim_obs
        retreat_bias_map = retreat_clim_mod - retreat_clim_obs

        # Interannual variability maps
        advance_std_obs = np.nanstd(advance_obs, axis=0)
        advance_std_mod = np.nanstd(advance_mod, axis=0)
        retreat_std_obs = np.nanstd(retreat_obs, axis=0)
        retreat_std_mod = np.nanstd(retreat_mod, axis=0)

        advance_std_diff_map = advance_std_mod - advance_std_obs
        retreat_std_diff_map = retreat_std_mod - retreat_std_obs

        # Trend maps (days/year)
        advance_trend_obs, advance_trend_pvalue_obs, advance_valid_year_count_obs = _calc_trend_map(advance_obs)
        advance_trend_mod, advance_trend_pvalue_mod, advance_valid_year_count_mod = _calc_trend_map(advance_mod)
        retreat_trend_obs, retreat_trend_pvalue_obs, retreat_valid_year_count_obs = _calc_trend_map(retreat_obs)
        retreat_trend_mod, retreat_trend_pvalue_mod, retreat_valid_year_count_mod = _calc_trend_map(retreat_mod)

        advance_trend_sigmask_obs = np.where(
            np.isfinite(advance_trend_pvalue_obs),
            (advance_trend_pvalue_obs < self.trend_sig_p).astype(float),
            np.nan,
        )
        advance_trend_sigmask_mod = np.where(
            np.isfinite(advance_trend_pvalue_mod),
            (advance_trend_pvalue_mod < self.trend_sig_p).astype(float),
            np.nan,
        )
        retreat_trend_sigmask_obs = np.where(
            np.isfinite(retreat_trend_pvalue_obs),
            (retreat_trend_pvalue_obs < self.trend_sig_p).astype(float),
            np.nan,
        )
        retreat_trend_sigmask_mod = np.where(
            np.isfinite(retreat_trend_pvalue_mod),
            (retreat_trend_pvalue_mod < self.trend_sig_p).astype(float),
            np.nan,
        )

        advance_trend_diff_map = advance_trend_mod - advance_trend_obs
        retreat_trend_diff_map = retreat_trend_mod - retreat_trend_obs

        # Derived duration diagnostics
        ice_season_obs = retreat_obs - advance_obs
        ice_season_mod = retreat_mod - advance_mod
        ice_season_obs = np.where(ice_season_obs > 0, ice_season_obs, np.nan)
        ice_season_mod = np.where(ice_season_mod > 0, ice_season_mod, np.nan)

        n_years = advance_obs.shape[0]
        if cycle_length_obs.size != n_years:
            cycle_length_obs = np.full((n_years,), np.nan, dtype=float)
        if cycle_length_mod.size != n_years:
            cycle_length_mod = np.full((n_years,), np.nan, dtype=float)

        open_water_obs = cycle_length_obs[:, None, None] - ice_season_obs
        open_water_mod = cycle_length_mod[:, None, None] - ice_season_mod
        open_water_obs = np.where(open_water_obs > 0, open_water_obs, np.nan)
        open_water_mod = np.where(open_water_mod > 0, open_water_mod, np.nan)

        ice_season_clim_obs = np.nanmean(ice_season_obs, axis=0)
        ice_season_clim_mod = np.nanmean(ice_season_mod, axis=0)
        open_water_clim_obs = np.nanmean(open_water_obs, axis=0)
        open_water_clim_mod = np.nanmean(open_water_mod, axis=0)

        ice_season_mean_bias = _safe_matrix_diff(ice_season_clim_obs, ice_season_clim_mod, 'Bias')
        ice_season_rmse = _safe_matrix_diff(ice_season_clim_obs, ice_season_clim_mod, 'RMSE')
        open_water_mean_bias = _safe_matrix_diff(open_water_clim_obs, open_water_clim_mod, 'Bias')
        open_water_rmse = _safe_matrix_diff(open_water_clim_obs, open_water_clim_mod, 'RMSE')

        advance_series_obs = _area_weighted_mean_series(advance_obs, self.cell_area)
        advance_series_mod = _area_weighted_mean_series(advance_mod, self.cell_area)
        retreat_series_obs = _area_weighted_mean_series(retreat_obs, self.cell_area)
        retreat_series_mod = _area_weighted_mean_series(retreat_mod, self.cell_area)
        ice_season_series_obs = _area_weighted_mean_series(ice_season_obs, self.cell_area)
        ice_season_series_mod = _area_weighted_mean_series(ice_season_mod, self.cell_area)
        open_water_series_obs = _area_weighted_mean_series(open_water_obs, self.cell_area)
        open_water_series_mod = _area_weighted_mean_series(open_water_mod, self.cell_area)

        rel_obs = _calc_relationship(retreat_series_obs, advance_series_obs)
        rel_mod = _calc_relationship(retreat_series_mod, advance_series_mod)

        # Scalar summaries
        advance_corr = _weighted_corr(advance_clim_obs, advance_clim_mod, self.cell_area)
        retreat_corr = _weighted_corr(retreat_clim_obs, retreat_clim_mod, self.cell_area)

        advance_rmse = _safe_matrix_diff(advance_clim_obs, advance_clim_mod, 'RMSE')
        retreat_rmse = _safe_matrix_diff(retreat_clim_obs, retreat_clim_mod, 'RMSE')
        advance_mean_bias = _safe_matrix_diff(advance_clim_obs, advance_clim_mod, 'Bias')
        retreat_mean_bias = _safe_matrix_diff(retreat_clim_obs, retreat_clim_mod, 'Bias')
        advance_trend_bias = _safe_matrix_diff(advance_trend_obs, advance_trend_mod, 'Bias')
        retreat_trend_bias = _safe_matrix_diff(retreat_trend_obs, retreat_trend_mod, 'Bias')

        advance_trend_sig_fraction_obs = _weighted_sig_fraction(advance_trend_sigmask_obs, self.cell_area)
        advance_trend_sig_fraction_mod = _weighted_sig_fraction(advance_trend_sigmask_mod, self.cell_area)
        retreat_trend_sig_fraction_obs = _weighted_sig_fraction(retreat_trend_sigmask_obs, self.cell_area)
        retreat_trend_sig_fraction_mod = _weighted_sig_fraction(retreat_trend_sigmask_mod, self.cell_area)

        advance_valid_year_mean_obs = _weighted_mean_field(advance_valid_year_count_obs, self.cell_area)
        advance_valid_year_mean_mod = _weighted_mean_field(advance_valid_year_count_mod, self.cell_area)
        retreat_valid_year_mean_obs = _weighted_mean_field(retreat_valid_year_count_obs, self.cell_area)
        retreat_valid_year_mean_mod = _weighted_mean_field(retreat_valid_year_count_mod, self.cell_area)

        return {
            'advance_clim_obs': advance_clim_obs,
            'advance_clim_mod': advance_clim_mod,
            'retreat_clim_obs': retreat_clim_obs,
            'retreat_clim_mod': retreat_clim_mod,

            'advance_bias_map': advance_bias_map,
            'retreat_bias_map': retreat_bias_map,

            'advance_std_obs': advance_std_obs,
            'advance_std_mod': advance_std_mod,
            'retreat_std_obs': retreat_std_obs,
            'retreat_std_mod': retreat_std_mod,
            'advance_std_diff_map': advance_std_diff_map,
            'retreat_std_diff_map': retreat_std_diff_map,

            'advance_trend_obs': advance_trend_obs,
            'advance_trend_mod': advance_trend_mod,
            'retreat_trend_obs': retreat_trend_obs,
            'retreat_trend_mod': retreat_trend_mod,
            'advance_trend_diff_map': advance_trend_diff_map,
            'retreat_trend_diff_map': retreat_trend_diff_map,
            'advance_trend_pvalue_obs': advance_trend_pvalue_obs,
            'advance_trend_pvalue_mod': advance_trend_pvalue_mod,
            'retreat_trend_pvalue_obs': retreat_trend_pvalue_obs,
            'retreat_trend_pvalue_mod': retreat_trend_pvalue_mod,
            'advance_trend_sigmask_obs': advance_trend_sigmask_obs,
            'advance_trend_sigmask_mod': advance_trend_sigmask_mod,
            'retreat_trend_sigmask_obs': retreat_trend_sigmask_obs,
            'retreat_trend_sigmask_mod': retreat_trend_sigmask_mod,
            'advance_valid_year_count_obs': advance_valid_year_count_obs,
            'advance_valid_year_count_mod': advance_valid_year_count_mod,
            'retreat_valid_year_count_obs': retreat_valid_year_count_obs,
            'retreat_valid_year_count_mod': retreat_valid_year_count_mod,

            'ice_season_clim_obs': ice_season_clim_obs,
            'ice_season_clim_mod': ice_season_clim_mod,
            'open_water_clim_obs': open_water_clim_obs,
            'open_water_clim_mod': open_water_clim_mod,

            'year': year_axis,
            'advance_series_obs': advance_series_obs,
            'advance_series_mod': advance_series_mod,
            'retreat_series_obs': retreat_series_obs,
            'retreat_series_mod': retreat_series_mod,
            'ice_season_series_obs': ice_season_series_obs,
            'ice_season_series_mod': ice_season_series_mod,
            'open_water_series_obs': open_water_series_obs,
            'open_water_series_mod': open_water_series_mod,

            'advance_corr': advance_corr,
            'advance_rmse': advance_rmse,
            'advance_mean_bias': advance_mean_bias,
            'advance_trend_bias': advance_trend_bias,
            'retreat_corr': retreat_corr,
            'retreat_rmse': retreat_rmse,
            'retreat_mean_bias': retreat_mean_bias,
            'retreat_trend_bias': retreat_trend_bias,

            'advance_trend_sig_fraction_obs': advance_trend_sig_fraction_obs,
            'advance_trend_sig_fraction_mod': advance_trend_sig_fraction_mod,
            'retreat_trend_sig_fraction_obs': retreat_trend_sig_fraction_obs,
            'retreat_trend_sig_fraction_mod': retreat_trend_sig_fraction_mod,
            'advance_valid_year_mean_obs': advance_valid_year_mean_obs,
            'advance_valid_year_mean_mod': advance_valid_year_mean_mod,
            'retreat_valid_year_mean_obs': retreat_valid_year_mean_obs,
            'retreat_valid_year_mean_mod': retreat_valid_year_mean_mod,

            'ice_season_rmse': ice_season_rmse,
            'ice_season_mean_bias': ice_season_mean_bias,
            'open_water_rmse': open_water_rmse,
            'open_water_mean_bias': open_water_mean_bias,

            'retreat_advance_corr_obs': rel_obs['lag0']['corr'],
            'retreat_advance_corr_mod': rel_mod['lag0']['corr'],
            'retreat_advance_corr_bias': rel_mod['lag0']['corr'] - rel_obs['lag0']['corr'],
            'retreat_advance_slope_obs': rel_obs['lag0']['slope'],
            'retreat_advance_slope_mod': rel_mod['lag0']['slope'],
            'retreat_advance_slope_bias': rel_mod['lag0']['slope'] - rel_obs['lag0']['slope'],
            'retreat_advance_r2_obs': rel_obs['lag0']['r2'],
            'retreat_advance_r2_mod': rel_mod['lag0']['r2'],
            'retreat_advance_r2_bias': rel_mod['lag0']['r2'] - rel_obs['lag0']['r2'],

            'retreat_to_next_advance_corr_obs': rel_obs['lag1']['corr'],
            'retreat_to_next_advance_corr_mod': rel_mod['lag1']['corr'],
            'retreat_to_next_advance_corr_bias': rel_mod['lag1']['corr'] - rel_obs['lag1']['corr'],
            'retreat_to_next_advance_slope_obs': rel_obs['lag1']['slope'],
            'retreat_to_next_advance_slope_mod': rel_mod['lag1']['slope'],
            'retreat_to_next_advance_slope_bias': rel_mod['lag1']['slope'] - rel_obs['lag1']['slope'],
            'retreat_to_next_advance_r2_obs': rel_obs['lag1']['r2'],
            'retreat_to_next_advance_r2_mod': rel_mod['lag1']['r2'],
            'retreat_to_next_advance_r2_bias': rel_mod['lag1']['r2'] - rel_obs['lag1']['r2'],

            'trend_sig_p': float(self.trend_sig_p),
        }

    def calculate_edge_area_difference(self, obs_array, mod_array,
                                       min_day=1, max_day=365, step=5, is_advance=True):
        """
        Compute the IIEE between simulated and observed sea-ice edge areas.

        Parameters
        ----------
        obs_array : ndarray (nx, ny)
            Observed advance/retreat day array.
        mod_array : ndarray (nx, ny)
            Simulated advance/retreat day array.
        min_day, max_day : int
            Day-of-year range for the threshold sweep.
        step : int
            Step size for the threshold sweep.
        is_advance : bool
            True for advance day (threshold: days <= threshold);
            False for retreat day (threshold: days >= threshold).

        Returns
        -------
        IIEE_ts_O : ndarray
            Over-estimation component (10^12 m^2).
        IIEE_ts_U : ndarray
            Under-estimation component (10^12 m^2).
        """
        days_range = np.arange(min_day, max_day + 1, step)
        nt = len(days_range)

        nx, ny = obs_array.shape

        # IIEE
        mask1 = np.zeros((nt, nx, ny))
        mask2 = np.zeros((nt, nx, ny))

        # Build binary masks for each threshold day
        for i, day in enumerate(days_range):
            if is_advance:
                # advance: ice present on days <= threshold
                mask1[i, :, :] = obs_array <= day
                mask2[i, :, :] = mod_array <= day
            else:
                # retreat: ice present on days >= threshold
                mask1[i, :, :] = obs_array >= day
                mask2[i, :, :] = mod_array >= day

        # --- Formula: IIEE = O + U, O = integer_A( max(c_f - c_t, 0) dA ), U = integer_A( max(c_t - c_f, 0) dA )
        diff_temp = mask2 - mask1

        diff_nb = np.where(diff_temp > 0, np.nan, abs(diff_temp))  # negative bias (but still positive values)
        diff_pb = np.where(diff_temp < 0, np.nan, diff_temp)  # positive bias

        IIEE_ts_O = np.array([np.nansum(diff_pb[jt, :, :] * self.cell_area) for jt in range(nt)]) / 1e12
        IIEE_ts_U = np.array([np.nansum(diff_nb[jt, :, :] * self.cell_area) for jt in range(nt)]) / 1e12

        return IIEE_ts_O, IIEE_ts_U
