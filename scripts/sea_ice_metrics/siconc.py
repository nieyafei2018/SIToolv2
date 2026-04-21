# -*- coding: utf-8 -*-
"""Sea-ice metric implementations by diagnostic family."""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr
from scipy import stats

from scripts import utils
from scripts.config import DAYS_PER_MONTH
from scripts.sea_ice_metrics.base import SeaIceMetricsBase

logger = logging.getLogger(__name__)


class SIconcMetrics(SeaIceMetricsBase):
    """
    Calculate sea ice concentration metrics including SIA, SIE, MIZ, and PIA.

    Attributes:
        grid_file (str): Path to grid file containing cell area and coordinates.
        hemisphere (str): Hemisphere, 'nh' for Northern or 'sh' for Southern.
        metric (str): Statistical metric for comparisons ('MAE', 'RMSE', etc.).
    """
    def __init__(self, grid_file: str, hemisphere: str = None, metric: str = 'MAE') -> None:
        """Initialize the SIconcMetrics calculator.

        Args:
            grid_file: NetCDF file with 'cell_area' variable (units: m²).
            hemisphere: Target hemisphere ('nh' or 'sh').
            metric: Metric for error calculation ('MAE', 'RMSE', etc.).
        """
        super().__init__(grid_file, hemisphere, metric)

    @staticmethod
    def _season_months(hemisphere: str) -> Dict[str, Tuple[int, int, int]]:
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

    @staticmethod
    def _seasonal_mean_series(monthly_series: np.ndarray, months: Tuple[int, ...]) -> np.ndarray:
        """Return year-wise seasonal means (weighted by month length)."""
        arr = np.asarray(monthly_series, dtype=float).reshape(-1)
        if arr.size <= 0:
            return np.array([], dtype=float)

        month_tuple = tuple(int(m) for m in months)
        cross_year = (12 in month_tuple and 1 in month_tuple)
        grouped: Dict[int, List[Tuple[float, float]]] = {}

        for ii, val in enumerate(arr):
            mm = int(ii % 12) + 1
            if mm not in month_tuple or not np.isfinite(val):
                continue
            yy = int(ii // 12)
            sy = yy + 1 if (cross_year and mm == 12) else yy
            grouped.setdefault(sy, []).append((float(val), float(DAYS_PER_MONTH[mm - 1])))

        out: List[float] = []
        for sy in sorted(grouped.keys()):
            vals = grouped[sy]
            if len(vals) < 2:
                continue
            vv = np.asarray([v for v, _ in vals], dtype=float)
            ww = np.asarray([w for _, w in vals], dtype=float)
            valid = np.isfinite(vv) & np.isfinite(ww) & (ww > 0)
            if int(np.sum(valid)) <= 0:
                continue
            vv = vv[valid]
            ww = ww[valid]
            if np.sum(ww) <= 0:
                continue
            out.append(float(np.nansum(vv * ww) / np.sum(ww)))

        return np.asarray(out, dtype=float)

    def _subset_period_series(self, monthly_series: np.ndarray, period: str) -> np.ndarray:
        """Return a 1-D series for annual/season/key-month diagnostics."""
        period_key = str(period or '').strip().lower()
        if period_key == 'march':
            return np.asarray(monthly_series[2::12], dtype=float)
        if period_key == 'september':
            return np.asarray(monthly_series[8::12], dtype=float)
        if period_key == 'annual':
            n_years = len(monthly_series) // 12
            if n_years == 0:
                return np.array([], dtype=float)
            trimmed = np.asarray(monthly_series[:n_years * 12], dtype=float)
            return np.nanmean(trimmed.reshape(n_years, 12), axis=1)

        season_map = self._season_months(self.hemisphere)
        if period_key in season_map:
            return self._seasonal_mean_series(np.asarray(monthly_series, dtype=float), season_map[period_key])
        raise ValueError(f"Unsupported period: {period}")

    @staticmethod
    def _extract_month_series(monthly_series: np.ndarray, month: int) -> np.ndarray:
        """Return month-filtered 1-D series (month is 1-12)."""
        return np.asarray(monthly_series[int(month) - 1::12], dtype=float)

    @staticmethod
    def _month_clim_from_series(field_ts: np.ndarray, month: int) -> np.ndarray:
        """Return climatological map for a selected calendar month."""
        return np.nanmean(field_ts[int(month) - 1::12, :, :], axis=0)

    @staticmethod
    def _detrended_std_and_trend_map(field_ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate detrended STD and trend (per decade) for a yearly map series.

        Note: This function is designed for yearly-sampled data (e.g., one value per year
        for a specific month like March). The time axis x = [0, 1, 2, ..., n_years-1]
        represents years, so the slope is in units/year. Multiplying by 10 converts to
        units/decade.
        """
        nt, nx, ny = field_ts.shape
        x = np.arange(nt, dtype=float)  # Year indices: 0, 1, 2, ..., n_years-1
        std_map = np.full((nx, ny), np.nan)
        trend_map = np.full((nx, ny), np.nan)
        p_map = np.full((nx, ny), np.nan)

        for jx in range(nx):
            for jy in range(ny):
                y = field_ts[:, jx, jy]
                valid = np.isfinite(y)
                if np.sum(valid) < 2:
                    continue
                reg = stats.linregress(x[valid], y[valid])
                fit = reg.slope * x[valid] + reg.intercept
                std_map[jx, jy] = np.nanstd(y[valid] - fit)
                # FIX: slope is in units/year (not units/month), so multiply by 10 only
                trend_map[jx, jy] = reg.slope * 10
                p_map[jx, jy] = reg.pvalue
        return std_map, trend_map, p_map

    @staticmethod
    def _calc_month_series_skill(series_ref: np.ndarray, series_cmp: np.ndarray) -> Dict[str, float]:
        """Return Corr/RMSE between two month-filtered 1-D anomaly series."""
        y1 = np.asarray(series_ref, dtype=float)
        y2 = np.asarray(series_cmp, dtype=float)
        valid = np.isfinite(y1) & np.isfinite(y2)
        if np.sum(valid) < 2:
            return {'corr': np.nan, 'rmse': np.nan}
        corr = float(np.corrcoef(y1[valid], y2[valid])[0, 1])
        rmse = float(np.sqrt(np.nanmean((y2[valid] - y1[valid]) ** 2)))
        return {'corr': corr, 'rmse': rmse}

    def _build_key_month_products(self, siconc: np.ndarray,
                                  SIA_ano: np.ndarray,
                                  SIE_ano: np.ndarray,
                                  PIA_ano: np.ndarray,
                                  MIZ_ano: np.ndarray) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Build hemisphere-aware key-month products for SIC diagnostics."""
        out: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}
        key_months = self.resolve_key_months(self.hemisphere)

        for month in key_months:
            mtag = self.month_tag(month)
            # Extract time series for this specific month
            month_ts = siconc[month - 1::12, :, :]
            # Calculate climatology for this month
            out[f'siconc_clim_{mtag}'] = self._month_clim_from_series(siconc, month)

            # Calculate anomalies by removing climatology from each year's value
            month_clim = out[f'siconc_clim_{mtag}']
            month_ano = month_ts - month_clim[np.newaxis, :, :]  # Broadcast climatology to all years

            # Calculate detrended STD and trend on ANOMALIES (not raw data)
            std_map, tr_map, tr_p = self._detrended_std_and_trend_map(month_ano)
            out[f'siconc_ano_std_{mtag}'] = std_map
            out[f'siconc_ano_tr_{mtag}'] = tr_map
            out[f'siconc_ano_tr_p_{mtag}'] = tr_p

            out[f'SIA_ano_{mtag}'] = self._extract_month_series(SIA_ano, month)
            out[f'SIE_ano_{mtag}'] = self._extract_month_series(SIE_ano, month)
            out[f'PIA_ano_{mtag}'] = self._extract_month_series(PIA_ano, month)
            out[f'MIZ_ano_{mtag}'] = self._extract_month_series(MIZ_ano, month)

        out['key_months'] = np.array(key_months, dtype=int)
        return out

    def _build_key_month_diff(self, sic1: Dict[str, np.ndarray],
                              sic2: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Build month-specific spatial/scalar differences and anomaly-series skill."""
        out: Dict[str, float] = {}
        for month in self.resolve_key_months(self.hemisphere):
            mtag = self.month_tag(month)
            mlabel = self.month_label(month)

            out[f'siconc_mean_diff_{mtag}'] = utils.MatrixDiff(
                sic1[f'siconc_clim_{mtag}'], sic2[f'siconc_clim_{mtag}'],
                self.cell_area, self.metric, mask=True
            )
            out[f'siconc_std_diff_{mtag}'] = utils.MatrixDiff(
                sic1[f'siconc_ano_std_{mtag}'], sic2[f'siconc_ano_std_{mtag}'],
                self.cell_area, self.metric, mask=True
            )
            out[f'siconc_trend_diff_{mtag}'] = utils.MatrixDiff(
                sic1[f'siconc_ano_tr_{mtag}'], sic2[f'siconc_ano_tr_{mtag}'],
                self.cell_area, 'MAE', mask=True
            )

            skill_sie = self._calc_month_series_skill(
                sic1[f'SIE_ano_{mtag}'], sic2[f'SIE_ano_{mtag}']
            )
            skill_sia = self._calc_month_series_skill(
                sic1[f'SIA_ano_{mtag}'], sic2[f'SIA_ano_{mtag}']
            )
            out[f'{mlabel}_Corr'] = skill_sie['corr']
            out[f'{mlabel}_RMSE'] = skill_sie['rmse']
            out[f'{mlabel}_SIA_Corr'] = skill_sia['corr']
            out[f'{mlabel}_SIA_RMSE'] = skill_sia['rmse']

        return out

    @staticmethod
    def _calc_series_stats(series: np.ndarray, trend_factor: float) -> Dict[str, float]:
        """Compute mean, linear trend, detrended STD, and significance diagnostics."""
        y = np.asarray(series, dtype=float)
        mean = float(np.nanmean(y)) if y.size else np.nan

        valid = np.isfinite(y)
        if np.sum(valid) < 2:
            return {
                'mean': mean,
                'trend': np.nan,
                'detrended_std': np.nan,
                'pvalue': np.nan,
                'significant': False,
            }

        x = np.arange(y.size, dtype=float)
        reg = stats.linregress(x[valid], y[valid])
        fit = reg.slope * x[valid] + reg.intercept
        residuals = y[valid] - fit

        return {
            'mean': mean,
            'trend': float(reg.slope * trend_factor),
            'detrended_std': float(np.nanstd(residuals)),
            'pvalue': float(reg.pvalue),
            'significant': bool(reg.pvalue < 0.05),
        }

    def _build_period_stats(
            self,
            SIA_ts: np.ndarray,
            SIE_ts: np.ndarray,
            MIZ_ts: np.ndarray = None,
            PIA_ts: np.ndarray = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Build Shu-style scalar statistics across requested periods."""
        period_cfg = [
            ('annual', 10.0),
            ('spring', 10.0),
            ('summer', 10.0),
            ('autumn', 10.0),
            ('winter', 10.0),
            ('march', 12.0 * 10.0),
            ('september', 12.0 * 10.0),
        ]
        metric_series: Dict[str, np.ndarray] = {
            'SIE': np.asarray(SIE_ts, dtype=float),
            'SIA': np.asarray(SIA_ts, dtype=float),
        }
        if MIZ_ts is not None:
            metric_series['MIZ'] = np.asarray(MIZ_ts, dtype=float)
        if PIA_ts is not None:
            metric_series['PIA'] = np.asarray(PIA_ts, dtype=float)

        stats_dict: Dict[str, Dict[str, Dict[str, float]]] = {}
        for period, trend_factor in period_cfg:
            period_metrics: Dict[str, Dict[str, float]] = {}
            for metric_name, series in metric_series.items():
                period_series = self._subset_period_series(series, period)
                period_metrics[metric_name] = self._calc_series_stats(period_series, trend_factor=trend_factor)
            stats_dict[period] = period_metrics
        return stats_dict

    def SIC_period_stats(self, sic_file: str, sic_key: str, sector: str = 'All') -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute lightweight period scalar stats for one sector.

        This helper intentionally skips spatial anomaly/trend products and only
        derives area-integrated time series required by the HTML scalar summary table.
        """
        with xr.open_dataset(sic_file) as ds:
            siconc = np.array(ds[sic_key])

        if siconc.ndim != 3:
            raise ValueError("Sea ice concentration must be a 3D array [time, x, y].")
        if self.cell_area.shape != siconc[0, :, :].shape:
            raise ValueError(
                'Grid cell area and SIC data must have the same spatial dimensions.'
                f'\n\tshape of siconc: {siconc.shape}'
                f'\n\tshape of cell_area: {self.cell_area.shape}'
            )

        # Convert fractions to percent when required and apply valid bounds.
        if np.nanmax(siconc) < 10:
            siconc = siconc * 100
        siconc[(siconc > 100) | (siconc < 0)] = np.nan

        sec_index = utils.region_index(
            grid_file=self.grid_file,
            hms=self.hemisphere,
            sector=sector,
        )
        siconc[:, ~sec_index] = np.nan

        nt = siconc.shape[0]
        SIA_ts = np.array([
            np.nansum((siconc[jt, :, :] / 100 * self.cell_area) * (siconc[jt, :, :] > 15))
            for jt in range(nt)
        ]) / 1e12
        SIE_ts = np.array([
            np.nansum((siconc[jt, :, :] > 15) * self.cell_area)
            for jt in range(nt)
        ]) / 1e12
        MIZ_ts = np.array([
            np.nansum(((siconc[jt, :, :] > 15) & (siconc[jt, :, :] < 90)) * self.cell_area)
            for jt in range(nt)
        ]) / 1e12
        PIA_ts = np.array([
            np.nansum((siconc[jt, :, :] >= 90) * self.cell_area)
            for jt in range(nt)
        ]) / 1e12
        return self._build_period_stats(SIA_ts=SIA_ts, SIE_ts=SIE_ts, MIZ_ts=MIZ_ts, PIA_ts=PIA_ts)

    def SIC_1M_metrics(self, sic_file: str, sic_key: str, sector: str = 'All') -> Dict[str, np.ndarray]:
        """
        Calculate some key metrics for a single sea ice concentration dataset, including
            SIA: the integral of grid cells areas multiplied by the SIC in each grid cell,
            SIE: the integral of grid cells areas where the SIC is larger than 15%,
            MIZ: area covered with SIC between 15% and 90%,
            PIA: area covered with SIC higher than 90%,

        Args:
            sic_file: NetCDF file with sea ice concentration data (shape: [time, x, y]).
            sic_key: Variable name for sea ice concentration in sic_file.
            sector: Geographic sector for regional analysis (e.g., 'Weddell' for Antarctic).

        Returns
        -------
        A dict variable contains the following metrics, where X stands for SIA/SIE/MIZ/PIA
            · Time series (nt, ):                                X_ts
            · Monthly climatologies (12, nx, ny):                siconc_clim
            · Monthly climatologies (12, ):                      X_clim
            · Monthly anomalies (nt, nx, ny):                    siconc_ano
            · Monthly anomalies (nt, ):                          X_ano
            · Interannual variabilities (nx, ny):                siconc_ano_std
            · Interannual variabilities (1, ):                   X_ano_std
            · Trends (nx, ny) and p-value (nx, ny):              siconc_ano_tr, siconc_ano_tr_p
            · Linear regression results (object)
              including attributes: slope, intercept, pvalue... : X_ano_tr
            · Linear regression results for March and September : X_ano_tr_Mar, X_ano_tr_Sep

        Notes
        -----
        Only support year-round data records!

        """
        logger.info("Calculating SIC metrics for %s ...", sic_file)

        with xr.open_dataset(sic_file) as ds:
            siconc = np.array(ds[sic_key])

        # Validate input dimensions
        if siconc.ndim != 3:
            raise ValueError("Sea ice concentration must be a 3D array [time, x, y].")
        if self.cell_area.shape != siconc[0, :, :].shape:
            raise ValueError(f'Grid cell area and SIC data must have the same spatial dimensions.'
                             f'\n\tshape of siconc: {siconc.shape}'
                             f'\n\tshape of cell_area: {self.cell_area.shape}')

        # Convert units to percentage if needed
        if np.nanmax(siconc) < 10:
            siconc = siconc * 100
        siconc[(siconc > 100) | (siconc < 0)] = np.nan


        # Apply regional mask if sector is specified
        sec_index = utils.region_index(grid_file=self.grid_file, hms=self.hemisphere, sector=sector)
        siconc[:, ~sec_index] = np.nan

        nt, nx, ny = siconc.shape
        logger.info("Processing %d time steps ...", nt)

        # --- Compute area-integrated metrics at each time step (units: 10^6 km^2) ---
        # SIA: sea ice area    = Σ(SIC/100 × cell_area), cells with SIC > 15% only
        # SIE: sea ice extent  = Σ(cell_area),           cells with SIC > 15% only
        # MIZ: marginal ice zone area = Σ(cell_area),    15% < SIC < 90%
        # PIA: pack ice area   = Σ(cell_area),           SIC ≥ 90%
        SIA_ts = np.array([np.nansum((siconc[jt, :, :] / 100 * self.cell_area) * (siconc[jt, :, :] > 15)) for jt in
                           range(nt)]) / 1e12
        SIE_ts = np.array([np.nansum((siconc[jt, :, :] > 15) * self.cell_area) for jt in range(nt)]) / 1e12
        MIZ_ts = np.array([np.nansum(((siconc[jt, :, :] > 15) & (siconc[jt, :, :] < 90)) * self.cell_area)
                           for jt in range(nt)]) / 1e12
        PIA_ts = np.array([np.nansum((siconc[jt, :, :] >= 90) * self.cell_area) for jt in range(nt)]) / 1e12

        # --- Monthly climatology: multi-year mean for each calendar month (0::12, 1::12, ...) ---
        siconc_clim = np.array([np.nanmean(siconc[m::12, :, :], axis=0) for m in range(12)])
        SIA_clim = np.array([np.nanmean(SIA_ts[m::12]) for m in range(12)])
        SIE_clim = np.array([np.nanmean(SIE_ts[m::12]) for m in range(12)])
        MIZ_clim = np.array([np.nanmean(MIZ_ts[m::12]) for m in range(12)])
        PIA_clim = np.array([np.nanmean(PIA_ts[m::12]) for m in range(12)])

        # --- Monthly anomalies = raw value − corresponding monthly climatology ---
        siconc_ano = np.array([siconc[j, :, :] - siconc_clim[j % 12, :, :] for j in range(nt)])
        SIA_ano = np.array([SIA_ts[j] - SIA_clim[j % 12] for j in range(nt)])
        SIE_ano = np.array([SIE_ts[j] - SIE_clim[j % 12] for j in range(nt)])
        MIZ_ano = np.array([MIZ_ts[j] - MIZ_clim[j % 12] for j in range(nt)])
        PIA_ano = np.array([PIA_ts[j] - PIA_clim[j % 12] for j in range(nt)])

        # --- Linear trend analysis ---
        months = np.arange(1, nt + 1, 1)
        years = np.arange(1, nt / 12 + 1, 1)
        if nt / 12 < 15:
            logger.warning("Time series shorter than 15 years — trends may be uncertain.")

        # Per-grid-point spatial trend: linear regression on the anomaly time series.
        # slope × 12 × 10 converts monthly slope to "per decade" units.
        # Only computed when valid data exceed half the total time steps (avoids spurious trends from sparse data).
        siconc_ano_tr = np.full((nx, ny), np.nan)
        siconc_ano_tr_p = np.full((nx, ny), np.nan)
        siconc_ano_std_detrended = np.full((nx, ny), np.nan)  # Detrended STD
        for jx in range(nx):
            for jy in range(ny):
                ano_ts = siconc_ano[:, jx, jy]
                valid_idx = ~np.isnan(ano_ts)
                if np.sum(valid_idx) > nt / 2:
                    slope, intercept, _, p_value, _ = stats.linregress(months[valid_idx], ano_ts[valid_idx])
                    siconc_ano_tr[jx, jy] = slope * 12 * 10
                    siconc_ano_tr_p[jx, jy] = p_value
                    # Calculate detrended STD: remove linear trend from anomalies
                    fit = slope * months[valid_idx] + intercept
                    detrended = ano_ts[valid_idx] - fit
                    siconc_ano_std_detrended[jx, jy] = np.nanstd(detrended)

        # Trend of area-integrated quantities (returns a linregress object with slope, pvalue, etc.)
        SIA_ano_tr = stats.linregress(months, SIA_ano)
        SIE_ano_tr = stats.linregress(months, SIE_ano)
        MIZ_ano_tr = stats.linregress(months, MIZ_ano)
        PIA_ano_tr = stats.linregress(months, PIA_ano)

        siconc_stats = self._build_period_stats(SIA_ts=SIA_ts, SIE_ts=SIE_ts, MIZ_ts=MIZ_ts, PIA_ts=PIA_ts)
        key_month_products = self._build_key_month_products(
            siconc=siconc,
            SIA_ano=SIA_ano,
            SIE_ano=SIE_ano,
            PIA_ano=PIA_ano,
            MIZ_ano=MIZ_ano,
        )

        return {'SIA_ts': SIA_ts, 'SIE_ts': SIE_ts, 'MIZ_ts': MIZ_ts, 'PIA_ts': PIA_ts,
                'siconc_clim': siconc_clim,
                'SIA_clim': SIA_clim, 'SIE_clim': SIE_clim, 'MIZ_clim': MIZ_clim, 'PIA_clim': PIA_clim,
                'siconc_ano': siconc_ano, 'siconc_ano_std': siconc_ano_std_detrended,
                'SIA_ano': SIA_ano, 'SIA_ano_std': np.nanstd(SIA_ano),
                'SIE_ano': SIE_ano, 'SIE_ano_std': np.nanstd(SIE_ano),
                'MIZ_ano': MIZ_ano, 'MIZ_ano_std': np.nanstd(MIZ_ano),
                'PIA_ano': PIA_ano, 'PIA_ano_std': np.nanstd(PIA_ano),
                'siconc_ano_tr': siconc_ano_tr, 'siconc_ano_tr_p': siconc_ano_tr_p,
                'SIA_ano_tr': SIA_ano_tr,
                'SIE_ano_tr': SIE_ano_tr,
                'MIZ_ano_tr': MIZ_ano_tr,
                'PIA_ano_tr': PIA_ano_tr,
                'siconc_stats': siconc_stats,
                **key_month_products}

    def SIC_2M_metrics(self, sic1_file: str, sic1_key: str, sic2_file: str,
                       sic2_key: str, sector: str = 'All') -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate difference metrics between two sea ice concentration datasets,
        typically used for model-observation comparisons.

        Args:
            sic1_file: Reference dataset (typically observations).
            sic1_key: Variable name in sic1_file.
            sic2_file: Model or comparison dataset.
            sic2_key: Variable name in sic2_file.
            sector: Geographic sector for regional analysis.

        Returns:
        --------
        Dictionary containing the following metrics, where X stands for SIA/SIE/MIZ/PIA
            · Mean state difference (1, ):                       siconc_mean_diff, X_mean_diff, IIEE_mean_diff
            · Mean anomaly varaiance difference (1, ):           siconc_std_diff, X_std_diff
            · Mean trend differnece (1, ):                       siconc_trend_diff, X_trend_diff
            · Monthly climatologies (12, ):                      IIEE_clim_diff

        Notes
        -----
        Only support year-round model data evaluation, and on month-scale!

        """
        logger.info("Comparing SIC datasets: %s vs %s", sic1_file, sic2_file)

        # Read data
        with xr.open_dataset(sic1_file) as ds:
            siconc1 = np.array(ds[sic1_key])
        with xr.open_dataset(sic2_file) as ds:
            siconc2 = np.array(ds[sic2_key])

        if np.nanmax(siconc1) < 10: siconc1 = siconc1 * 100  # unit: %
        siconc1[(siconc1 > 100) | (siconc1 < 0)] = np.nan
        if np.nanmax(siconc2) < 10: siconc2 = siconc2 * 100  # unit: %
        siconc2[(siconc2 > 100) | (siconc2 < 0)] = np.nan

        nt, nx, ny = siconc1.shape

        # Apply regional mask
        sec_index = utils.region_index(grid_file=self.grid_file, hms=self.hemisphere, sector=sector)
        siconc1[:, ~sec_index], siconc2[:, ~sec_index] = np.nan, np.nan

        # Compute single-dataset metrics for each of the two datasets independently
        sic1 = self.SIC_1M_metrics(sic1_file, sic1_key, sector=sector)
        sic2 = self.SIC_1M_metrics(sic2_file, sic2_key, sector=sector)

        # --- 1. Mean-state difference ---
        # Compute spatial error metrics month by month, then weight-average to annual using DAYS_PER_MONTH
        temp = np.full((12,), np.nan)
        for m in range(12):
            temp[m] = utils.MatrixDiff(
                sic1['siconc_clim'][m, :, :], sic2['siconc_clim'][m, :, :],
                metric=self.metric, mask=True
            )

        # DAYS_PER_MONTH-weighted annual mean: Σ(monthly error × days per month) / Σ(days per month)
        siconc_mean_diff = np.sum(temp * DAYS_PER_MONTH) / np.sum(DAYS_PER_MONTH)
        SIA_mean_diff = utils.MatrixDiff(sic1['SIA_clim'], sic2['SIA_clim'], DAYS_PER_MONTH, self.metric)
        SIE_mean_diff = utils.MatrixDiff(sic1['SIE_clim'], sic2['SIE_clim'], DAYS_PER_MONTH, self.metric)
        MIZ_mean_diff = utils.MatrixDiff(sic1['MIZ_clim'], sic2['MIZ_clim'], DAYS_PER_MONTH, self.metric)
        PIA_mean_diff = utils.MatrixDiff(sic1['PIA_clim'], sic2['PIA_clim'], DAYS_PER_MONTH, self.metric)

        # 2. Anomaly variance differences
        siconc_std_diff = utils.MatrixDiff(
            sic1['siconc_ano_std'], sic2['siconc_ano_std'],
            self.cell_area, self.metric, mask=True
        )
        SIA_std_diff = utils.MatrixDiff(sic1['SIA_ano_std'], sic2['SIA_ano_std'], metric=self.metric)
        SIE_std_diff = utils.MatrixDiff(sic1['SIE_ano_std'], sic2['SIE_ano_std'], metric=self.metric)
        MIZ_std_diff = utils.MatrixDiff(sic1['MIZ_ano_std'], sic2['MIZ_ano_std'], metric=self.metric)
        PIA_std_diff = utils.MatrixDiff(sic1['PIA_ano_std'], sic2['PIA_ano_std'], metric=self.metric)

        # 3. Trend differences
        siconc_trend_diff = utils.MatrixDiff(
            sic1['siconc_ano_tr'],
            sic2['siconc_ano_tr'],
            self.cell_area, 'MAE', mask=True
        )
        if sic1['SIA_ano_tr'].pvalue > 0.05:
            logger.warning("SIA trend of sic1 is not significant.")
        SIA_trend_diff = 12 * 10 * abs(sic1['SIA_ano_tr'].slope - sic2['SIA_ano_tr'].slope)

        if sic1['SIE_ano_tr'].pvalue > 0.05:
            logger.warning("SIE trend of sic1 is not significant.")
        SIE_trend_diff = 12 * 10 * abs(sic1['SIE_ano_tr'].slope - sic2['SIE_ano_tr'].slope)

        if sic1['MIZ_ano_tr'].pvalue > 0.05:
            logger.warning("MIZ trend of sic1 is not significant.")
        MIZ_trend_diff = 12 * 10 * abs(sic1['MIZ_ano_tr'].slope - sic2['MIZ_ano_tr'].slope)

        if sic1['PIA_ano_tr'].pvalue > 0.05:
            logger.warning("PIA trend of sic1 is not significant.")
        PIA_trend_diff = 12 * 10 * abs(sic1['PIA_ano_tr'].slope - sic2['PIA_ano_tr'].slope)

        # === 4. IIEE ===
        conc1mask = np.zeros((nt, nx, ny))
        conc2mask = np.zeros((nt, nx, ny))
        conc1mask[np.where(siconc1 >= 15)] = 1
        conc2mask[np.where(siconc2 >= 15)] = 1

        # --- Formula: IIEE = O + U, O = integer_A( max(c_f - c_t, 0) dA ), U = integer_A( max(c_t - c_f, 0) dA )
        diff_temp = np.full(conc1mask.shape, np.nan)
        for jt in range(nt):
            diff_temp[jt, :, :] = conc2mask[jt, :, :] - conc1mask[jt, :, :]

        diff_nb = np.where(diff_temp > 0, np.nan, abs(diff_temp))  # negative bias (but still positive values)
        diff_pb = np.where(diff_temp < 0, np.nan, diff_temp)  # positive bias

        IIEE_ts_O = np.array([np.nansum(diff_pb[jt, :, :] * self.cell_area) for jt in range(nt)]) / 1e12
        IIEE_ts_U = np.array([np.nansum(diff_nb[jt, :, :] * self.cell_area) for jt in range(nt)]) / 1e12
        IIEE_ts_diff = IIEE_ts_O + IIEE_ts_U

        IIEE_clim_diff = np.array([np.nanmean(IIEE_ts_diff[m::12]) for m in range(12)])
        IIEE_mean_diff = np.sum(IIEE_clim_diff * DAYS_PER_MONTH) / np.sum(DAYS_PER_MONTH)

        # --- Formula: IIEE = AEE + ME, AEE = |O - U|, ME = 2·min(O, U)
        O_U_ts_diff = IIEE_ts_O - IIEE_ts_U  # if > 0, means overestimation dominate
        O_U_clim_diff = np.array([np.nanmean(O_U_ts_diff[m::12]) for m in range(12)])
        O_U_mean_diff = np.sum(O_U_clim_diff * DAYS_PER_MONTH) / np.sum(DAYS_PER_MONTH)

        # ME_ts_diff = IIEE_ts_diff - AEE_ts_diff  # misplacement error

        key_month_diff = self._build_key_month_diff(sic1=sic1, sic2=sic2)

        return {'siconc_mean_diff': siconc_mean_diff,
                'SIA_mean_diff': SIA_mean_diff,
                'SIE_mean_diff': SIE_mean_diff,
                'MIZ_mean_diff': MIZ_mean_diff,
                'PIA_mean_diff': PIA_mean_diff,
                'siconc_std_diff': siconc_std_diff,
                'SIA_std_diff': SIA_std_diff,
                'SIE_std_diff': SIE_std_diff,
                'MIZ_std_diff': MIZ_std_diff,
                'PIA_std_diff': PIA_std_diff,
                'siconc_trend_diff': siconc_trend_diff,
                'SIA_trend_diff': SIA_trend_diff,
                'SIE_trend_diff': SIE_trend_diff,
                'MIZ_trend_diff': MIZ_trend_diff,
                'PIA_trend_diff': PIA_trend_diff,
                'IIEE_clim_diff': IIEE_clim_diff,
                'IIEE_mean_diff': IIEE_mean_diff,
                'O_U_clim_diff': O_U_clim_diff,
                'O_U_mean_diff': O_U_mean_diff,
                **key_month_diff}
