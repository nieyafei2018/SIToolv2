# -*- coding: utf-8 -*-
"""Sea-ice metric implementations by diagnostic family."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from scripts import utils
from scripts.config import DAYS_PER_MONTH
from scripts.sea_ice_metrics.base import SeaIceMetricsBase

logger = logging.getLogger(__name__)


class ThicknessMetrics(SeaIceMetricsBase):
    """
    Purpose
    -------
    Evaluate sea ice thickness or snow layer thickness (i.e., snow depth).

    For CMIP6 models,
        sisnconc: surface snow area fraction [Snow Area Percentage] (%)
        sisnthick: actual thickness of snow (snow volume divided by snow-covered area)
    """

    def __init__(self, grid_file: str, hemisphere: str, year_sta: int, year_end: int,
                 metric: str = 'MAE'):
        """Initialize the thickness metrics calculator.

        Args:
            grid_file: NetCDF file with 'cell_area' variable (units: m²).
            hemisphere: Target hemisphere ('nh' or 'sh').
            year_sta: First year of the evaluation period (inclusive).
            year_end: Last year of the evaluation period (inclusive).
            metric: Statistical metric for comparisons ('MAE', 'RMSE', etc.).
        """
        super().__init__(grid_file, hemisphere, metric)
        self.year_sta = year_sta
        self.year_end = year_end
        self.years = np.arange(year_sta, year_end + 1)

    @staticmethod
    def _detrended_std_and_trend_map(field_ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate detrended STD and trend (per decade) for a yearly map series.

        Note: This function is designed for yearly-sampled data (e.g., one value per year
        for a specific month like March). The time axis x = [0, 1, 2, ..., n_years-1]
        represents years, so the slope is in units/year. Multiplying by 10 converts to
        units/decade.
        """
        arr = np.asarray(field_ts, dtype=float)
        nt, nx, ny = arr.shape
        std_map = np.full((nx, ny), np.nan, dtype=float)
        trend_map = np.full((nx, ny), np.nan, dtype=float)
        p_map = np.full((nx, ny), np.nan, dtype=float)
        if nt < 2:
            return std_map, trend_map, p_map

        x = np.arange(nt, dtype=float).reshape(nt, 1, 1)
        valid = np.isfinite(arr)
        n = np.sum(valid, axis=0).astype(float)
        good_n = n >= 2
        if not np.any(good_n):
            return std_map, trend_map, p_map

        y0 = np.where(valid, arr, 0.0)
        x0 = np.where(valid, x, 0.0)
        sum_x = np.sum(x0, axis=0)
        sum_y = np.sum(y0, axis=0)
        sum_xx = np.sum(x0 * x0, axis=0)
        sum_xy = np.sum(x0 * y0, axis=0)

        den = n * sum_xx - sum_x * sum_x
        good = good_n & np.isfinite(den) & (np.abs(den) > 0.0)
        if not np.any(good):
            return std_map, trend_map, p_map

        slope = np.full((nx, ny), np.nan, dtype=float)
        intercept = np.full((nx, ny), np.nan, dtype=float)
        slope[good] = (n[good] * sum_xy[good] - sum_x[good] * sum_y[good]) / den[good]
        intercept[good] = (sum_y[good] - slope[good] * sum_x[good]) / n[good]

        fit = slope[None, :, :] * x + intercept[None, :, :]
        residual = np.where(valid, arr - fit, np.nan)
        with np.errstate(invalid='ignore'):
            std_map = np.nanstd(residual, axis=0)

        trend_map[good] = slope[good] * 10.0

        mean_x = np.full((nx, ny), np.nan, dtype=float)
        mean_y = np.full((nx, ny), np.nan, dtype=float)
        mean_x[good] = sum_x[good] / n[good]
        mean_y[good] = sum_y[good] / n[good]
        xc = np.where(valid, x - mean_x[None, :, :], 0.0)
        yc = np.where(valid, arr - mean_y[None, :, :], 0.0)
        ss_xx = np.sum(xc * xc, axis=0)
        ss_yy = np.sum(yc * yc, axis=0)
        ss_xy = np.sum(xc * yc, axis=0)

        good_r = good & (ss_xx > 0.0) & (ss_yy > 0.0)
        if np.any(good_r):
            r_map = np.full((nx, ny), np.nan, dtype=float)
            r_map[good_r] = ss_xy[good_r] / np.sqrt(ss_xx[good_r] * ss_yy[good_r])
            r_map = np.where(np.isfinite(r_map), np.clip(r_map, -1.0, 1.0), np.nan)
            df = n - 2.0
            good_p = good_r & (df > 0.0)
            if np.any(good_p):
                ridx = np.where(good_p)
                abs_r = np.abs(r_map[ridx])
                df_vals = df[ridx]
                t_stat = np.full(abs_r.shape, np.inf, dtype=float)
                not_one = abs_r < (1.0 - 1e-12)
                denom_t = np.maximum(1e-12, 1.0 - abs_r[not_one] ** 2)
                t_stat[not_one] = abs_r[not_one] * np.sqrt(df_vals[not_one] / denom_t)
                p_map[ridx] = 2.0 * stats.t.sf(t_stat, df_vals)

        return std_map, trend_map, p_map

    @staticmethod
    def _calc_month_series_skill(series_ref: np.ndarray, series_cmp: np.ndarray) -> Dict[str, float]:
        """Return Corr/RMSE between two month-filtered 1-D series."""
        y1 = np.asarray(series_ref, dtype=float)
        y2 = np.asarray(series_cmp, dtype=float)
        valid = np.isfinite(y1) & np.isfinite(y2)
        if np.sum(valid) < 2:
            return {'corr': np.nan, 'rmse': np.nan}
        corr = float(np.corrcoef(y1[valid], y2[valid])[0, 1])
        rmse = float(np.sqrt(np.nanmean((y2[valid] - y1[valid]) ** 2)))
        return {'corr': corr, 'rmse': rmse}

    def Thickness_1M_metrics(self, thickness, month_list: np.array) -> Dict[str, Union[np.ndarray, object]]:
        """
        Calculate comprehensive metrics for a single thickness dataset.

        Metrics include climatologies, anomalies, variability, and trends for both
        spatial thickness fields and integrated volume time series.

        Args:
            thickness: Thickness data array (shape: [time, x, y]).
            month_list: List of months corresponding to each time step.

        Returns
        -------
        A dict variable contains the following metrics, nm represents how many unique months
            · Monthly climatology (nm, nx, ny):                   thick_clim
            · Monthly climatology (nm, ):                         Vol_clim
            · Monthly anomaly (nt, nx, ny):                       thick_ano
            · Monthly anomaly (nt, ):                             Vol_ano
            · Trends (nx, ny) and p-value (nx, ny):               thick_ano_tr, thick_ano_tr_p
            · Linear regression results (object)
              including attributes: slope, intercept, pvalue... : Vol_ano_tr
        """
        # Validate input data
        thickness = np.array(thickness, copy=True)
        if thickness.ndim != 3:
            raise ValueError("Thickness data must be a 3D array [time, x, y].")
        if self.cell_area.shape != thickness[0, :, :].shape:
            raise ValueError(f"Grid cell area and thickness data must have same spatial dimensions."
                             f"\n\tshape of thickness: {thickness.shape}"
                             f"\n\tshape of cell_area: {self.cell_area.shape}")

        # Quality control
        thickness[thickness > 200] = np.nan
        thickness[thickness < 0] = np.nan


        nt, nx, ny = thickness.shape
        logger.info("Processing %d time steps over %d years ...", nt, len(self.years))

        # Calculate volume time series with validation for zero valid pixels
        Vol_ts = np.full(nt, np.nan)
        for m in range(nt):
            valid_mask = np.isfinite(thickness[m, :, :])
            valid_count = np.sum(valid_mask)
            if valid_count == 0:
                logger.debug("Timestep %d has zero valid pixels, setting volume to NaN.", m)
                Vol_ts[m] = np.nan
            else:
                Vol_ts[m] = np.nansum(thickness[m, :, :] / 1e3 * self.cell_area / 1e6) / 1e3
        # unit: 10^3 km^3

        # 1. Calculate monthly climatologies
        unique_months = np.unique(month_list)
        positions = {month: [idx for idx, m in enumerate(month_list) if m == month] for month in unique_months}

        thick_clim = np.array([np.nanmean(thickness[positions[month], :, :], axis=0) for month in unique_months])
        Vol_clim = np.array([np.nanmean(Vol_ts[positions[month]]) for month in unique_months])

        # 2. Calculate anomalies
        thick_ano = np.full((nt, nx, ny), np.nan)
        Vol_ano = np.full(nt, np.nan)

        for i, month in enumerate(unique_months):
            thick_ano[positions[month], :, :] = (thickness[positions[month], :, :] - thick_clim[i, :, :])
            Vol_ano[positions[month]] = Vol_ts[positions[month]] - Vol_clim[i]

        # 3. Calculate trends
        months = np.arange(1, nt + 1)
        if nt / 12 < 15:
            logger.warning("Time series shorter than 15 years — trends may be uncertain.")

        # Spatial trends with detrended STD
        thick_ano_tr = np.full((nx, ny), np.nan)
        thick_ano_tr_p = np.full((nx, ny), np.nan)
        thick_ano_std_detrended = np.full((nx, ny), np.nan)  # Detrended STD
        for jx in np.arange(nx):
            for jy in np.arange(ny):
                ano_ts = thick_ano[:, jx, jy]
                valid_idx = ~np.isnan(ano_ts)
                if np.sum(valid_idx) > nt / 2:  # Require at least 50% valid data
                    slope, intercept, _, p_value, _ = stats.linregress(months[valid_idx], ano_ts[valid_idx])
                    thick_ano_tr[jx, jy] = slope * 12 * 10  # Convert to per decade
                    thick_ano_tr_p[jx, jy] = p_value
                    # Calculate detrended STD
                    fit = slope * months[valid_idx] + intercept
                    detrended = ano_ts[valid_idx] - fit
                    thick_ano_std_detrended[jx, jy] = np.nanstd(detrended)

        Vol_ano_tr = stats.linregress(months, Vol_ano)

        key_month_products: Dict[str, Union[np.ndarray, object]] = {}
        key_months = self.resolve_key_months(self.hemisphere)
        for month in key_months:
            mtag = self.month_tag(month)
            month_mask = (np.asarray(month_list, dtype=int) == month)
            if np.any(month_mask):
                month_ts = thickness[month_mask, :, :]
                key_month_products[f'thick_clim_{mtag}'] = np.nanmean(month_ts, axis=0)
                std_map, tr_map, tr_p = self._detrended_std_and_trend_map(month_ts)
                key_month_products[f'thick_ano_std_{mtag}'] = std_map
                key_month_products[f'thick_ano_tr_{mtag}'] = tr_map
                key_month_products[f'thick_ano_tr_p_{mtag}'] = tr_p
                key_month_products[f'Vol_ano_{mtag}'] = np.asarray(Vol_ano[month_mask], dtype=float)
            else:
                key_month_products[f'thick_clim_{mtag}'] = np.full((nx, ny), np.nan)
                key_month_products[f'thick_ano_std_{mtag}'] = np.full((nx, ny), np.nan)
                key_month_products[f'thick_ano_tr_{mtag}'] = np.full((nx, ny), np.nan)
                key_month_products[f'thick_ano_tr_p_{mtag}'] = np.full((nx, ny), np.nan)
                key_month_products[f'Vol_ano_{mtag}'] = np.array([], dtype=float)
        key_month_products['key_months'] = np.array(key_months, dtype=int)

        return {'uni_mon': unique_months,
                'thick_clim': thick_clim, 'Vol_clim': Vol_clim,
                'thick_ano': thick_ano, 'Vol_ano': Vol_ano,
                'thick_ano_std': np.nanstd(thick_ano, axis=0), 'Vol_ano_std': np.nanstd(Vol_ano),
                'thick_ano_tr': thick_ano_tr, 'thick_ano_tr_p': thick_ano_tr_p,
                'Vol_ano_tr': Vol_ano_tr,
                **key_month_products}

    def Thickness_2M_metrics(self, thick1_file: str, thick1_key: str,
                             thick2_file: str, thick2_key: str,
                             strict_obs_match: bool = False,
                             sector: str = 'All',
                             obs_match_file: Optional[str] = None,
                             obs_match_key: Optional[str] = None) -> Optional[Dict[str, Union[float, np.ndarray]]]:
        """
        Calculate difference metrics between two thickness datasets.

        Typically used for model-observation comparisons of sea ice thickness
        or snow depth data.

        Args:
            thick1_file: Reference dataset file (typically observations).
            thick1_key: Variable name in observation file.
            thick2_file: Model or comparison dataset file.
            thick2_key: Variable name in model file.
            strict_obs_match: Whether to enforce obs1-valid-cell matching.
            sector: Geographic sector for regional analysis.
            obs_match_file: Optional second-observation file used to enforce
                shared observation coverage in strict mode.
            obs_match_key: Variable key for *obs_match_file* (defaults to
                ``thick1_key`` when omitted).

        Returns
        -------
        A dict variable contains the following metrics,
            · Mean state difference <float value>:               thick_mean_diff   unit: [m]
                                                                 Vol_mean_diff     unit: [10^3 km^3]
            · Mean anomaly varaiance difference <float value>:   thick_std_diff    unit: [m]
                                                                 Vol_std_diff      unit: [10^3 km^3]
            · Mean trend differnece <float value>:               thick_trend_diff  unit: [m/decade]
                                                                 Vol_trend_diff    unit: [10^3 km^3/decade]

        Notes
        -----
        · Should keep in mind that the sea ice thickness observations in both hemisphere have considerable missing values.
                      For instance, there is a circle of missing measurements near the North Pole in the period
                      200206-201010 for Envisat-based observations. So be careful when calculating the total
                      sea ice volume.
        · All reference .nc files have 'longitude', 'latitude', 'time'/'Time', and 'heff'/'sit'/'SNdepth' variables
        """
        logger.info("Comparing thickness datasets: %s vs %s", thick1_file, thick2_file)

        # Read and preprocess data on each pair's own temporal overlap.
        # In strict mode, optionally constrain to months shared by obs1/obs2/model.
        allowed_yearmon = None
        use_obs_common_match = bool(strict_obs_match and obs_match_file)
        if use_obs_common_match:
            match_key = obs_match_key or thick1_key
            allowed_yearmon = (
                self._month_coverage(thick1_file, thick1_key)
                & self._month_coverage(thick2_file, thick2_key)
                & self._month_coverage(obs_match_file, match_key)
            )
            if not allowed_yearmon:
                logger.warning(
                    "No common monthly coverage across obs1/obs2/model for strict matching: %s, %s, %s",
                    thick1_file, obs_match_file, thick2_file,
                )
                return None

        thick1, thick2, yearmon_list = self._read_thickness_data(
            thick1_file, thick1_key, thick2_file, thick2_key,
            allowed_yearmon=allowed_yearmon,
        )

        if thick1 is None or thick2 is None or not yearmon_list:
            logger.warning(
                "No overlapping months found for %s vs %s (skip pairwise metrics).",
                thick1_file, thick2_file,
            )
            return None

        obs_match = None
        if use_obs_common_match:
            match_key = obs_match_key or thick1_key
            _, obs_match_raw, obs_match_yearmon = self._read_thickness_data(
                thick1_file, thick1_key, obs_match_file, match_key,
                allowed_yearmon=set(yearmon_list),
            )
            if obs_match_raw is None or not obs_match_yearmon:
                logger.warning(
                    "Strict obs-common matching skipped (%s vs %s has no shared months).",
                    thick1_file, obs_match_file,
                )
                return None

            obs_match_map = {
                ym: arr for ym, arr in zip(obs_match_yearmon, np.asarray(obs_match_raw))
            }
            keep_idx = []
            aligned_match = []
            for idx, ym in enumerate(yearmon_list):
                match_arr = obs_match_map.get(ym)
                if match_arr is None:
                    continue
                keep_idx.append(idx)
                aligned_match.append(match_arr)

            if not keep_idx:
                logger.warning(
                    "No common time samples remain after obs-common alignment: %s, %s, %s",
                    thick1_file, obs_match_file, thick2_file,
                )
                return None

            thick1 = thick1[keep_idx, :, :]
            thick2 = thick2[keep_idx, :, :]
            yearmon_list = [yearmon_list[idx] for idx in keep_idx]
            obs_match = np.asarray(aligned_match)

        # Apply sector footprint before pairwise validation.
        sec_index = utils.region_index(
            grid_file=self.grid_file,
            hms=self.hemisphere,
            sector=sector,
        )
        thick1[:, ~sec_index] = np.nan
        thick2[:, ~sec_index] = np.nan
        if obs_match is not None:
            obs_match[:, ~sec_index] = np.nan

        # Apply spatial matching:
        #   - default mode: pairwise co-location (legacy behavior)
        #   - strict_obs_match=True: retain only cells where obs1/model are valid
        #     and, when provided, obs2 is also valid.
        thick01, thick02 = self._validate_ts_match(
            thick1, thick2, strict_obs_match=strict_obs_match, obs_match=obs_match
        )

        month_list = [month for year, month in yearmon_list]

        # Pairwise spatial diagnostics use co-located arrays on the shared overlap.
        # For integrated volume:
        #   - strict_obs_match=False: preserve each dataset's own valid cells (legacy)
        #   - strict_obs_match=True: use the same obs1-valid footprint for both series
        sit1 = self.Thickness_1M_metrics(thick01, month_list)
        sit2 = self.Thickness_1M_metrics(thick02, month_list)
        if not strict_obs_match:
            sit1_full = self.Thickness_1M_metrics(thick1.copy(), month_list)
            sit2_full = self.Thickness_1M_metrics(thick2.copy(), month_list)
            for key in ('Vol_clim', 'Vol_ano', 'Vol_ano_std', 'Vol_ano_tr'):
                sit1[key] = sit1_full[key]
                sit2[key] = sit2_full[key]

        uni_mon = sit1['uni_mon']
        nm = len(uni_mon)

        # 1. Mean cycle
        temp = np.full((nm,), np.nan)
        for m in range(nm):
            temp[m] = utils.MatrixDiff(sit1['thick_clim'][m, :, :], sit2['thick_clim'][m, :, :], self.cell_area,
                                       self.metric, mask=True)

        DAYS_PER_MONTH_t = np.array([DAYS_PER_MONTH[i - 1] for i in uni_mon])

        thick_mean_diff = np.sum(temp * DAYS_PER_MONTH_t) / np.sum(DAYS_PER_MONTH_t)
        Vol_mean_diff = utils.MatrixDiff(sit1['Vol_clim'], sit2['Vol_clim'], DAYS_PER_MONTH_t, self.metric)

        # 2. Anomaly varaiance
        thick_std_diff = utils.MatrixDiff(sit1['thick_ano_std'], sit2['thick_ano_std'], self.cell_area, self.metric,
                                          mask=True)
        Vol_std_diff = utils.MatrixDiff(sit1['Vol_ano_std'], sit2['Vol_ano_std'], metric=self.metric)

        # 3. Trend
        nt, nx, ny = thick01.shape
        if nt / 12 < 15:
            logger.warning("Years analyzed < 15 — trends have considerable uncertainty.")
        thick_trend_diff = utils.MatrixDiff(sit1['thick_ano_tr'],
                                            sit2['thick_ano_tr'],
                                            self.cell_area, 'MAE', mask=True)

        if sit1['Vol_ano_tr'].pvalue > 0.05:
            logger.warning("Vol trend of thick1 is not significant.")
        Vol_trend_diff = 12 * 10 * (sit1['Vol_ano_tr'].slope - sit2['Vol_ano_tr'].slope)  # 10^3 km3/decade

        key_month_diff: Dict[str, float] = {}
        for month in self.resolve_key_months(self.hemisphere):
            mtag = self.month_tag(month)
            mlabel = self.month_label(month)
            key_month_diff[f'thick_mean_diff_{mtag}'] = utils.MatrixDiff(
                sit1[f'thick_clim_{mtag}'], sit2[f'thick_clim_{mtag}'],
                self.cell_area, self.metric, mask=True
            )
            key_month_diff[f'thick_std_diff_{mtag}'] = utils.MatrixDiff(
                sit1[f'thick_ano_std_{mtag}'], sit2[f'thick_ano_std_{mtag}'],
                self.cell_area, self.metric, mask=True
            )
            key_month_diff[f'thick_trend_diff_{mtag}'] = utils.MatrixDiff(
                sit1[f'thick_ano_tr_{mtag}'], sit2[f'thick_ano_tr_{mtag}'],
                self.cell_area, 'MAE', mask=True
            )
            _skill = self._calc_month_series_skill(sit1[f'Vol_ano_{mtag}'], sit2[f'Vol_ano_{mtag}'])
            key_month_diff[f'{mlabel}_Corr'] = _skill['corr']
            key_month_diff[f'{mlabel}_RMSE'] = _skill['rmse']

        return {'thick1_metric': sit1, 'thick2_metric': sit2, 'yearmon_list': yearmon_list,
                'match_mode': ('obs_dual_strict' if use_obs_common_match else 'obs1_strict') if strict_obs_match else 'pairwise',
                'thick_mean_diff': thick_mean_diff,
                'Vol_mean_diff': Vol_mean_diff,
                'thick_std_diff': thick_std_diff,
                'Vol_std_diff': Vol_std_diff,
                'thick_trend_diff': thick_trend_diff,
                'Vol_trend_diff': Vol_trend_diff,
                **key_month_diff}

    def _month_coverage(self, file_path: str, var_key: str) -> set:
        """Return the set of (year, month) tuples present in *file_path* within [year_sta, year_end].

        A month is considered present if the file contains at least one timestep
        in that calendar month within the analysis period.

        Args:
            file_path: Path to a preprocessed NetCDF file.
            var_key:   Variable name (used only to confirm the file is readable).

        Returns:
            Set of (year, month) tuples.
        """
        covered: set = set()
        try:
            with xr.open_dataset(file_path) as ds:
                time_name = self._get_time_dimension(ds)
                for year, month in self._time_values_to_yearmon(ds[time_name].values):
                    covered.add((year, month))
        except Exception as exc:
            logger.warning("_month_coverage: could not read %s (%s)", file_path, exc)
        return covered

    def _time_values_to_yearmon(self, time_values: np.ndarray) -> List[Tuple[int, int]]:
        """Convert heterogeneous time arrays to ``[(year, month), ...]``.

        Supports NumPy/Pandas datetimes and CFTime objects (e.g.
        ``cftime.Datetime360Day``) by preferring native ``year``/``month``
        attributes and falling back to ``pd.Timestamp`` conversion.
        """
        out: List[Tuple[int, int]] = []
        values = np.asarray(time_values).reshape(-1)
        for raw in values:
            year = None
            month = None
            if hasattr(raw, 'year') and hasattr(raw, 'month'):
                try:
                    year = int(raw.year)
                    month = int(raw.month)
                except Exception:
                    year = None
                    month = None
            if year is None or month is None:
                try:
                    ts = pd.Timestamp(raw)
                    year = int(ts.year)
                    month = int(ts.month)
                except Exception:
                    continue
            if 1 <= month <= 12 and self.year_sta <= year <= self.year_end:
                out.append((year, month))
        return out

    def _validate_ts_match(self, thick1: np.ndarray, thick2: np.ndarray,
                           strict_obs_match: bool = False,
                           obs_match: Optional[np.ndarray] = None):
        """Apply a co-located valid-data mask to both thickness arrays.

        Default (legacy) mode:
            Cells are masked where both arrays are zero (open ocean) or either
            array is NaN.

        strict_obs_match=True mode:
            Cells are retained only when observation (thick1) is finite and > 0.
            Model NaNs are also excluded. This is used for strict obs-footprint
            matching in integrated-volume diagnostics. When *obs_match* is
            provided, those cells must also be finite and > 0.

        Args:
            thick1: Observation thickness array (nt, nx, ny).
            thick2: Model thickness array (nt, nx, ny).
            strict_obs_match: Whether to enforce obs1-valid-cell matching.
            obs_match: Optional auxiliary observation array for strict common
                obs-coverage matching.

        Returns:
            Tuple (thick01, thick02) with invalid cells set to NaN in both arrays.
        """
        thick01 = thick1.copy()
        thick02 = thick2.copy()
        if strict_obs_match:
            obs_invalid = (~np.isfinite(thick1)) | (thick1 <= 0)
            model_invalid = ~np.isfinite(thick2)
            invalid = obs_invalid | model_invalid
            if obs_match is not None:
                if np.shape(obs_match) != np.shape(thick1):
                    raise ValueError(
                        "obs_match shape mismatch in strict thickness matching: "
                        f"{np.shape(obs_match)} vs {np.shape(thick1)}"
                    )
                obs2_invalid = (~np.isfinite(obs_match)) | (obs_match <= 0)
                invalid = invalid | obs2_invalid
        else:
            # A cell is invalid if both values are zero (open ocean) or either is NaN
            invalid = ((thick1 == 0) & (thick2 == 0)) | np.isnan(thick1) | np.isnan(thick2)
        thick01[invalid] = np.nan
        thick02[invalid] = np.nan
        return thick01, thick02

    def _read_thickness_data(self, thick1_file: str, thick1_key: str,
                             thick2_file: str, thick2_key: str,
                             allowed_yearmon: Optional[set] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Tuple[int, int]]]:
        """
        Read and temporally match thickness data from two datasets.

        The matching is performed at (year, month) resolution over the analysis
        period [year_sta, year_end]. Only months available in *both* datasets are
        retained (strict temporal overlap).

        Returns:
            Tuple containing:
            - thick1: Data from first dataset (nt, nx, ny), or None if no overlap
            - thick2: Data from second dataset (nt, nx, ny), or None if no overlap
            - yearmon_list: List of (year, month) tuples for each time step
        """
        model_data: List[np.ndarray] = []
        obs_data: List[np.ndarray] = []
        yearmon_list: List[Tuple[int, int]] = []

        with xr.open_dataset(f'{thick1_file}') as ds_o:
            with xr.open_dataset(f'{thick2_file}') as ds_m:
                # Identify time dimension names
                time_name_o = self._get_time_dimension(ds_o)
                time_name_m = self._get_time_dimension(ds_m)

                otime_ym = set(self._time_values_to_yearmon(ds_o[time_name_o].values))
                mtime_ym = set(self._time_values_to_yearmon(ds_m[time_name_m].values))

                # Temporal matching
                for year in range(self.year_sta, self.year_end + 1):
                    for month in range(1, 13):
                        if allowed_yearmon is not None and (year, month) not in allowed_yearmon:
                            continue
                        if (year, month) not in otime_ym or (year, month) not in mtime_ym:
                            continue

                        # Select all samples in this (year, month)
                        time_sel_o = (ds_o[f'{time_name_o}.year'] == year) & (ds_o[f'{time_name_o}.month'] == month)
                        time_sel_o_dict = {time_name_o: time_sel_o[time_name_o][time_sel_o == True]}
                        o_da = ds_o[thick1_key].sel(time_sel_o_dict)

                        time_sel_m = (ds_m[f'{time_name_m}.year'] == year) & (ds_m[f'{time_name_m}.month'] == month)
                        time_sel_m_dict = {time_name_m: time_sel_m[time_name_m][time_sel_m == True]}
                        m_da = ds_m[thick2_key].sel(time_sel_m_dict)

                        # Reduce to a single 2-D field per month
                        if time_name_o in o_da.dims:
                            o_da = o_da.mean(dim=time_name_o, skipna=True)
                        if time_name_m in m_da.dims:
                            m_da = m_da.mean(dim=time_name_m, skipna=True)

                        o_data = np.array(o_da)
                        m_data = np.array(m_da)

                        # Skip unexpected shapes (must be 2-D spatial fields)
                        if o_data.ndim != 2 or m_data.ndim != 2:
                            logger.warning(
                                "Skipping %04d-%02d due to non-2D monthly fields: obs ndim=%s, model ndim=%s",
                                year, month, o_data.ndim, m_data.ndim,
                            )
                            continue

                        obs_data.append(o_data)
                        model_data.append(m_data)
                        yearmon_list.append((year, month))

        if not yearmon_list:
            return None, None, []

        thick1 = np.array(obs_data)
        thick2 = np.array(model_data)

        # Quality control
        thick1[(thick1 < 0) | (thick1 > 200)] = np.nan
        thick2[(thick2 < 0) | (thick2 > 200)] = np.nan

        return thick1, thick2, yearmon_list

    def _get_time_dimension(self, dataset: xr.Dataset) -> str:
        """Identify time dimension name in dataset."""
        time_candidates = [dim for dim in dataset.dims
                           if ("time" in dim.lower() and
                               "bnds" not in dim.lower() and
                               "bounds" not in dim.lower())]

        if not time_candidates:
            raise ValueError("No time dimension found in dataset.")

        return time_candidates[0]

    def _has_data_for_period(self, otime: pd.DatetimeIndex, mtime: pd.DatetimeIndex,
                             year: int, month: int) -> bool:
        """Check if both datasets have data for the specified period."""
        if month == 12:
            next_year, next_month = year + 1, 1
        else:
            next_year, next_month = year, month + 1

        start_date = pd.Timestamp(year, month, 1)
        end_date = pd.Timestamp(next_year, next_month, 1)  # The first day of next month

        o_has_data = any((otime >= start_date) & (otime < end_date))
        m_has_data = any((mtime >= start_date) & (mtime < end_date))
        return o_has_data and m_has_data
