# -*- coding: utf-8 -*-
"""Pipeline module evaluation helpers."""

from scripts.pipeline import app as _app

# Reuse runtime namespace (imports/constants/helpers) initialized in app.py.
globals().update({k: v for k, v in _app.__dict__.items() if k not in globals()})

from scripts.config import DAYS_PER_MONTH
from scripts.pipeline.modules.common import (
    _apply_plot_runtime_options,
    _check_outputs_exist,
    _load_plot_options,
    _plot_options_get_module,
)


def _sid_month_mask(time_keys: Any, size: int, month: int) -> np.ndarray:
    """Build month selector from cached (year, month[, day]) keys."""
    size = max(0, int(size))
    month = int(month)
    mask = np.zeros(size, dtype=bool)
    if isinstance(time_keys, (list, tuple)) and len(time_keys) >= size:
        for ii, key in enumerate(time_keys[:size]):
            mm = None
            if isinstance(key, (list, tuple, np.ndarray)) and len(key) >= 2:
                try:
                    mm = int(key[1])
                except Exception:
                    mm = None
            if mm == month:
                mask[ii] = True
    if not np.any(mask) and size > 0:
        # Fallback for synthetic monthly index series.
        idx = np.arange(size)
        mask = ((idx % 12) + 1) == month
    return mask


def _sid_season_months(hemisphere: str) -> Dict[str, Tuple[int, int, int]]:
    """Return canonical season -> month mapping for one hemisphere."""
    h = str(hemisphere or '').lower()
    if h == 'sh':
        return {
            'Spring': (9, 10, 11),   # SON
            'Summer': (12, 1, 2),    # DJF
            'Autumn': (3, 4, 5),     # MAM
            'Winter': (6, 7, 8),     # JJA
        }
    return {
        'Spring': (3, 4, 5),         # MAM
        'Summer': (6, 7, 8),         # JJA
        'Autumn': (9, 10, 11),       # SON
        'Winter': (12, 1, 2),        # DJF
    }


def _sid_domain_specs(hemisphere: str) -> List[Tuple[str, Tuple[str, Optional[Any]]]]:
    """Return ordered domain specs (annual + seasons + key months)."""
    specs: List[Tuple[str, Tuple[str, Optional[Any]]]] = [('Annual', ('annual', None))]
    for season_name in ('Spring', 'Summer', 'Autumn', 'Winter'):
        specs.append((season_name, ('season', season_name)))
    specs.extend([
        ('March', ('month', 3)),
        ('September', ('month', 9)),
    ])
    return specs


def _sid_yearmon_series(metric_dict: Dict[str, Any], n_time: int) -> List[Tuple[int, int]]:
    """Infer (year, month) per time step for one SID metric payload."""
    if n_time <= 0:
        return []
    ym = metric_dict.get('yearmon_list', [])
    if isinstance(ym, (list, tuple)) and len(ym) == n_time:
        out: List[Tuple[int, int]] = []
        for item in ym:
            try:
                if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                    out.append((int(item[0]), int(item[1])))
            except Exception:
                continue
        if len(out) == n_time:
            return out

    tkeys = metric_dict.get('time_keys', [])
    if isinstance(tkeys, (list, tuple)) and len(tkeys) == n_time:
        out = []
        for item in tkeys:
            try:
                if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                    out.append((int(item[0]), int(item[1])))
            except Exception:
                continue
        if len(out) == n_time:
            return out

    mons = np.tile(np.arange(1, 13, dtype=int), int(np.ceil(float(n_time) / 12.0)))[:n_time]
    years = np.arange(n_time, dtype=int) // 12
    return [(int(years[ii]), int(mons[ii])) for ii in range(n_time)]


def _sid_matrix_seasonal_mean(field_mon: np.ndarray, months: Tuple[int, ...]) -> np.ndarray:
    """Weighted seasonal mean map from 12-month climatology field."""
    arr = np.asarray(field_mon, dtype=float)
    if arr.ndim != 3 or arr.shape[0] < 12:
        return np.array([])
    maps = []
    weights = []
    for mm in months:
        idx = int(mm) - 1
        if idx < 0 or idx >= arr.shape[0]:
            continue
        maps.append(np.asarray(arr[idx], dtype=float))
        weights.append(float(DAYS_PER_MONTH[idx]))
    if not maps:
        return np.array([])
    maps_arr = np.asarray(maps, dtype=float)
    w = np.asarray(weights, dtype=float)
    if np.sum(w) <= 0:
        return np.array([])
    return np.nansum(maps_arr * w[:, None, None], axis=0) / np.sum(w)


def _sid_seasonal_diagnostics(
    metric_dict: Optional[dict],
    season: str,
    hemisphere: str,
) -> Dict[str, Any]:
    """Build seasonal diagnostics (clim/std/trend maps + anomaly series)."""
    out: Dict[str, Any] = {
        'season': str(season),
        'months': tuple(),
        'clim_map': np.array([]),
        'u_clim_map': np.array([]),
        'v_clim_map': np.array([]),
        'std_map': np.array([]),
        'trend_map': np.array([]),
        'trend_p_map': np.array([]),
        'ts_years': np.array([], dtype=int),
        'ts_ano': np.array([], dtype=float),
    }
    if not isinstance(metric_dict, dict):
        return out

    months = _sid_season_months(hemisphere).get(str(season), tuple())
    if not months:
        return out
    out['months'] = tuple(int(m) for m in months)

    mke_clim = np.asarray(metric_dict.get('MKE_clim', np.array([])), dtype=float)
    if mke_clim.ndim != 3 or mke_clim.shape[0] < 12:
        mke_clim = np.asarray(metric_dict.get('speed_clim', np.array([])), dtype=float)
    out['clim_map'] = _sid_matrix_seasonal_mean(mke_clim, out['months'])

    u_clim = np.asarray(metric_dict.get('u_clim', np.array([])), dtype=float)
    v_clim = np.asarray(metric_dict.get('v_clim', np.array([])), dtype=float)
    out['u_clim_map'] = _sid_matrix_seasonal_mean(u_clim, out['months'])
    out['v_clim_map'] = _sid_matrix_seasonal_mean(v_clim, out['months'])

    speed_ano = np.asarray(metric_dict.get('speed_ano', np.array([])), dtype=float)
    if speed_ano.ndim != 3:
        speed_ano = np.asarray(metric_dict.get('MKE_ano', np.array([])), dtype=float)
    speed_ts_ano = np.asarray(metric_dict.get('speed_ts_ano', np.array([])), dtype=float).squeeze()
    if speed_ts_ano.ndim != 1:
        speed_ts_ano = np.asarray(metric_dict.get('MKE_ts_ano', np.array([])), dtype=float).squeeze()

    if speed_ano.ndim != 3 or speed_ts_ano.ndim != 1:
        return out
    n_use = min(int(speed_ano.shape[0]), int(speed_ts_ano.size))
    if n_use <= 0:
        return out
    speed_ano = speed_ano[:n_use, :, :]
    speed_ts_ano = speed_ts_ano[:n_use]

    ym = _sid_yearmon_series(metric_dict, n_use)
    cross_year = (12 in out['months'] and 1 in out['months'])
    grouped: Dict[int, List[int]] = {}
    for ii, (yy, mm) in enumerate(ym):
        if int(mm) not in out['months']:
            continue
        sy = int(yy) + 1 if (cross_year and int(mm) == 12) else int(yy)
        grouped.setdefault(sy, []).append(ii)

    season_years = sorted(grouped.keys())
    if not season_years:
        return out

    map_list = []
    ts_list = []
    year_list = []
    for sy in season_years:
        idx = grouped.get(sy, [])
        if len(idx) < 2:
            continue
        map_list.append(np.nanmean(speed_ano[idx, :, :], axis=0))
        ts_list.append(float(np.nanmean(speed_ts_ano[idx])))
        year_list.append(int(sy))

    if not map_list:
        return out

    map_ts = np.asarray(map_list, dtype=float)
    out['ts_years'] = np.asarray(year_list, dtype=int)
    out['ts_ano'] = np.asarray(ts_list, dtype=float)
    if map_ts.ndim == 3 and map_ts.shape[0] >= 2:
        std_map, tr_map, tr_p = SIM.ThicknessMetrics._detrended_std_and_trend_map(map_ts)
        out['std_map'] = std_map
        out['trend_map'] = tr_map
        out['trend_p_map'] = tr_p

    return out


def _sid_matrix_corr(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Spatial correlation over finite overlap of two 2D arrays."""
    a = np.asarray(arr1, dtype=float)
    b = np.asarray(arr2, dtype=float)
    if a.shape != b.shape:
        return np.nan
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 2:
        return np.nan
    va = a[valid].ravel()
    vb = b[valid].ravel()
    sa = np.nanstd(va)
    sb = np.nanstd(vb)
    if not (np.isfinite(sa) and np.isfinite(sb)) or sa <= 0.0 or sb <= 0.0:
        return np.nan
    return float(np.corrcoef(va, vb)[0, 1])


def _sid_vector_cosine_mean(
    u1_map: np.ndarray,
    v1_map: np.ndarray,
    u2_map: np.ndarray,
    v2_map: np.ndarray,
) -> float:
    """Return mean cosine similarity between two vector maps."""
    u1 = np.asarray(u1_map, dtype=float)
    v1 = np.asarray(v1_map, dtype=float)
    u2 = np.asarray(u2_map, dtype=float)
    v2 = np.asarray(v2_map, dtype=float)
    if u1.ndim != 2 or v1.ndim != 2 or u2.ndim != 2 or v2.ndim != 2:
        return np.nan
    if u1.shape != v1.shape or u2.shape != v2.shape or u1.shape != u2.shape:
        return np.nan
    n1 = np.sqrt(u1 ** 2 + v1 ** 2)
    n2 = np.sqrt(u2 ** 2 + v2 ** 2)
    denom = n1 * n2
    valid = np.isfinite(u1) & np.isfinite(v1) & np.isfinite(u2) & np.isfinite(v2) & (denom > 0)
    if int(np.sum(valid)) < 2:
        return np.nan
    cosv = (u1 * u2 + v1 * v2) / denom
    vv = cosv[valid]
    if vv.size <= 0:
        return np.nan
    return float(np.nanmean(vv))


def _sid_extract_month_skill(diff_dict: Optional[dict], month: Optional[int]) -> Tuple[float, float]:
    """Return (Corr, RMSE) for one key month (or annual) from one SID 2M payload."""
    if not isinstance(diff_dict, dict):
        return np.nan, np.nan

    sid1 = diff_dict.get('sid1_metric') if isinstance(diff_dict.get('sid1_metric'), dict) else {}
    sid2 = diff_dict.get('sid2_metric') if isinstance(diff_dict.get('sid2_metric'), dict) else {}

    # Prefer speed-based monthly anomaly skill (more intuitive than MKE),
    # and fall back to legacy MKE-based series when needed.
    for metric_key in ('speed_ts_ano', 'MKE_ts_ano'):
        s1 = np.asarray(sid1.get(metric_key, np.array([])), dtype=float).squeeze()
        s2 = np.asarray(sid2.get(metric_key, np.array([])), dtype=float).squeeze()
        if s1.ndim == 1 and s2.ndim == 1 and s1.size > 0 and s2.size > 0:
            n_use = min(s1.size, s2.size)
            s1 = s1[:n_use]
            s2 = s2[:n_use]
            keys = sid1.get('yearmon_list') or sid2.get('yearmon_list') or sid1.get('time_keys') or sid2.get('time_keys')
            mon_mask = np.ones(n_use, dtype=bool) if month is None else _sid_month_mask(keys, n_use, int(month))
            valid = mon_mask & np.isfinite(s1) & np.isfinite(s2)
            if int(np.sum(valid)) >= 2:
                s1m = s1[valid]
                s2m = s2[valid]
                corr = np.nan
                if np.nanstd(s1m) > 0 and np.nanstd(s2m) > 0:
                    corr = float(np.corrcoef(s1m, s2m)[0, 1])
                rmse = float(np.sqrt(np.nanmean((s1m - s2m) ** 2)))
                return corr, rmse

    if month is None:
        corr = np.nan
        rmse = np.nan
        for clim_key in ('speed_clim', 'MKE_clim'):
            c1 = np.asarray(sid1.get(clim_key, np.array([])), dtype=float)
            c2 = np.asarray(sid2.get(clim_key, np.array([])), dtype=float)
            if c1.ndim == 3 and c2.ndim == 3 and c1.shape[0] > 0 and c2.shape[0] > 0:
                nmon = min(c1.shape[0], c2.shape[0], 12)
                if nmon <= 0:
                    continue
                corr_vals = []
                rmse_vals = []
                weights = np.asarray(DAYS_PER_MONTH[:nmon], dtype=float)
                for mm in range(nmon):
                    cc = _sid_matrix_corr(c1[mm], c2[mm])
                    rr = utils.MatrixDiff(c1[mm], c2[mm], metric='RMSE', mask=True)
                    corr_vals.append(cc if np.isfinite(cc) else np.nan)
                    rmse_vals.append(rr if np.isfinite(rr) else np.nan)
                corr_arr = np.asarray(corr_vals, dtype=float)
                rmse_arr = np.asarray(rmse_vals, dtype=float)
                wc = weights.copy()
                wr = weights.copy()
                wc[~np.isfinite(corr_arr)] = 0.0
                wr[~np.isfinite(rmse_arr)] = 0.0
                if np.sum(wc) > 0:
                    corr = float(np.nansum(corr_arr * wc) / np.sum(wc))
                if np.sum(wr) > 0:
                    rmse = float(np.nansum(rmse_arr * wr) / np.sum(wr))
                break
        return float(corr) if np.isfinite(corr) else np.nan, float(rmse) if np.isfinite(rmse) else np.nan

    mlabel = SIM.SeaIceMetricsBase.month_label(month)
    corr_key = f'{mlabel}_Corr'
    rmse_key = f'{mlabel}_RMSE'
    corr = diff_dict.get(corr_key, np.nan)
    rmse = diff_dict.get(rmse_key, np.nan)
    if np.isfinite(corr) and np.isfinite(rmse):
        return float(corr), float(rmse)

    m_idx = int(month) - 1
    corr = np.nan
    rmse = np.nan
    for clim_key in ('speed_clim', 'MKE_clim'):
        c1 = np.asarray(sid1.get(clim_key, np.array([])), dtype=float)
        c2 = np.asarray(sid2.get(clim_key, np.array([])), dtype=float)
        if c1.ndim == 3 and c2.ndim == 3 and c1.shape[0] > m_idx and c2.shape[0] > m_idx:
            corr = _sid_matrix_corr(c1[m_idx], c2[m_idx])
            rmse = utils.MatrixDiff(c1[m_idx], c2[m_idx], metric='RMSE', mask=True)
            break

    return float(corr) if np.isfinite(corr) else np.nan, float(rmse) if np.isfinite(rmse) else np.nan


def _sid_month_trend_map(field_ts: np.ndarray, samples_per_year: float = 1.0) -> np.ndarray:
    """Return per-grid linear trend (per decade) for a map time series."""
    arr = np.asarray(field_ts, dtype=float)
    if arr.ndim != 3 or arr.shape[0] < 2:
        shape = arr.shape[1:] if arr.ndim >= 2 else (0, 0)
        return np.full(shape, np.nan, dtype=float)

    nt = int(arr.shape[0])
    x = np.arange(nt, dtype=float)[:, None, None]
    valid = np.isfinite(arr)
    n_valid = np.sum(valid, axis=0).astype(float)

    mean_x = np.divide(
        np.sum(x * valid, axis=0, dtype=float),
        n_valid,
        out=np.full(n_valid.shape, np.nan, dtype=float),
        where=n_valid > 0,
    )
    mean_y = np.divide(
        np.nansum(arr, axis=0, dtype=float),
        n_valid,
        out=np.full(n_valid.shape, np.nan, dtype=float),
        where=n_valid > 0,
    )

    x_center = x - mean_x[None, :, :]
    y_center = arr - mean_y[None, :, :]

    cov_num = np.nansum(x_center * y_center, axis=0, dtype=float)
    var_num = np.nansum((x_center ** 2) * valid, axis=0, dtype=float)
    denom = n_valid - 1.0

    cov_xy = np.divide(
        cov_num,
        denom,
        out=np.full(n_valid.shape, np.nan, dtype=float),
        where=denom > 0,
    )
    var_x = np.divide(
        var_num,
        denom,
        out=np.full(n_valid.shape, np.nan, dtype=float),
        where=denom > 0,
    )

    slope_per_sample = np.divide(
        cov_xy,
        var_x,
        out=np.full(n_valid.shape, np.nan, dtype=float),
        where=var_x > 0,
    )
    slope_per_sample[n_valid < 2] = np.nan
    spp = float(samples_per_year) if np.isfinite(samples_per_year) and samples_per_year > 0 else 1.0
    return slope_per_sample * spp * 10.0


def _sid_extract_month_map_diffs(
    sid1: dict, sid2: dict, month: Optional[int]
) -> Tuple[float, float]:
    """Return (std_map_diff, trend_map_diff) for one key month or annual mode."""
    for metric_key in ('speed_ano', 'MKE_ano'):
        field_ano1 = np.asarray(sid1.get(metric_key, np.array([])), dtype=float)
        field_ano2 = np.asarray(sid2.get(metric_key, np.array([])), dtype=float)
        if field_ano1.ndim != 3 or field_ano2.ndim != 3:
            continue

        n_use = min(field_ano1.shape[0], field_ano2.shape[0])
        if n_use < 2:
            continue

        field_ano1 = field_ano1[:n_use, :, :]
        field_ano2 = field_ano2[:n_use, :, :]
        keys = sid1.get('yearmon_list') or sid2.get('yearmon_list') or sid1.get('time_keys') or sid2.get('time_keys')
        mon_mask = np.ones(n_use, dtype=bool) if month is None else _sid_month_mask(keys, n_use, int(month))
        if int(np.sum(mon_mask)) < 2:
            continue

        month_ano1 = field_ano1[mon_mask, :, :]
        month_ano2 = field_ano2[mon_mask, :, :]
        std1 = np.nanstd(month_ano1, axis=0)
        std2 = np.nanstd(month_ano2, axis=0)
        samples_per_year = 12.0 if month is None else 1.0
        tr1 = _sid_month_trend_map(month_ano1, samples_per_year=samples_per_year)
        tr2 = _sid_month_trend_map(month_ano2, samples_per_year=samples_per_year)

        std_map_diff = utils.MatrixDiff(std1, std2, metric='MAE', mask=True)
        trend_map_diff = utils.MatrixDiff(tr1, tr2, metric='MAE', mask=True)
        return (
            float(std_map_diff) if np.isfinite(std_map_diff) else np.nan,
            float(trend_map_diff) if np.isfinite(trend_map_diff) else np.nan,
        )

    return np.nan, np.nan


def _sid_extract_month_extended_stats(
    diff_dict: Optional[dict], month: Optional[int]
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return month-level SID diagnostics for one key month (or annual mode).

    Returns:
        (mean_diff, ano_std_diff, ano_trend_diff, std_map_diff, trend_map_diff,
         vectcorr_mean, corr, rmse)
    """
    corr, rmse = _sid_extract_month_skill(diff_dict, month)
    if not isinstance(diff_dict, dict):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, corr, rmse

    mean_diff = np.nan
    ano_std_diff = np.nan
    ano_trend_diff = np.nan
    std_map_diff = np.nan
    trend_map_diff = np.nan
    vectcorr_mean = diff_dict.get('vectcorr_mean', np.nan)

    sid1 = diff_dict.get('sid1_metric') if isinstance(diff_dict.get('sid1_metric'), dict) else {}
    sid2 = diff_dict.get('sid2_metric') if isinstance(diff_dict.get('sid2_metric'), dict) else {}

    if month is None:
        for clim_key in ('speed_clim', 'MKE_clim'):
            c1 = np.asarray(sid1.get(clim_key, np.array([])), dtype=float)
            c2 = np.asarray(sid2.get(clim_key, np.array([])), dtype=float)
            if c1.ndim == 3 and c2.ndim == 3 and c1.shape[0] > 0 and c2.shape[0] > 0:
                nmon = min(c1.shape[0], c2.shape[0], 12)
                if nmon <= 0:
                    continue
                mm_diff = np.asarray([
                    utils.MatrixDiff(c1[mm], c2[mm], metric='MAE', mask=True)
                    for mm in range(nmon)
                ], dtype=float)
                w = np.asarray(DAYS_PER_MONTH[:nmon], dtype=float)
                w[~np.isfinite(mm_diff)] = 0.0
                if np.sum(w) > 0:
                    mean_diff = float(np.nansum(mm_diff * w) / np.sum(w))
                break
    else:
        m_idx = int(month) - 1
        for clim_key in ('speed_clim', 'MKE_clim'):
            c1 = np.asarray(sid1.get(clim_key, np.array([])), dtype=float)
            c2 = np.asarray(sid2.get(clim_key, np.array([])), dtype=float)
            if c1.ndim == 3 and c2.ndim == 3 and c1.shape[0] > m_idx and c2.shape[0] > m_idx:
                mean_diff = utils.MatrixDiff(c1[m_idx], c2[m_idx], metric='MAE', mask=True)
                break

    smd, tmd = _sid_extract_month_map_diffs(sid1, sid2, month)
    if np.isfinite(smd):
        std_map_diff = smd
    if np.isfinite(tmd):
        trend_map_diff = tmd

    for ts_key in ('speed_ts_ano', 'MKE_ts_ano'):
        s1 = np.asarray(sid1.get(ts_key, np.array([])), dtype=float).squeeze()
        s2 = np.asarray(sid2.get(ts_key, np.array([])), dtype=float).squeeze()
        if s1.ndim == 1 and s2.ndim == 1 and s1.size > 0 and s2.size > 0:
            n_use = min(s1.size, s2.size)
            s1 = s1[:n_use]
            s2 = s2[:n_use]
            keys = sid1.get('yearmon_list') or sid2.get('yearmon_list') or sid1.get('time_keys') or sid2.get('time_keys')
            mon_mask = np.ones(n_use, dtype=bool) if month is None else _sid_month_mask(keys, n_use, int(month))
            valid = mon_mask & np.isfinite(s1) & np.isfinite(s2)
            if int(np.sum(valid)) >= 2:
                s1m = s1[valid]
                s2m = s2[valid]
                ano_std_diff = abs(float(np.nanstd(s1m)) - float(np.nanstd(s2m)))
                if s1m.size >= 2 and s2m.size >= 2:
                    x = np.arange(s1m.size, dtype=float)
                    try:
                        slope1 = float(np.polyfit(x, s1m, 1)[0])
                        slope2 = float(np.polyfit(x, s2m, 1)[0])
                        factor = 12.0 * 10.0 if month is None else 10.0
                        ano_trend_diff = abs((slope1 - slope2) * factor)
                    except Exception:
                        ano_trend_diff = np.nan
                break

    if not np.isfinite(ano_std_diff):
        ano_std_diff = diff_dict.get('speed_ts_ano_std', np.nan)
    if not np.isfinite(ano_std_diff):
        ano_std_diff = diff_dict.get('MKE_ts_ano_std', np.nan)

    if not np.isfinite(ano_trend_diff):
        tr1 = sid1.get('speed_ts_ano_tr')
        tr2 = sid2.get('speed_ts_ano_tr')
        if tr1 is not None and tr2 is not None and hasattr(tr1, 'slope') and hasattr(tr2, 'slope'):
            try:
                factor = 12.0 * 10.0 if month is None else 10.0
                ano_trend_diff = abs((float(tr1.slope) - float(tr2.slope)) * factor)
            except Exception:
                ano_trend_diff = np.nan
    if not np.isfinite(ano_trend_diff):
        ano_trend_diff = diff_dict.get('MKE_ts_trend_diff', np.nan)

    if not np.isfinite(std_map_diff):
        std_map_diff = diff_dict.get('MKE_std_diff', np.nan)
    if not np.isfinite(trend_map_diff):
        trend_map_diff = diff_dict.get('MKE_trend_diff', np.nan)

    return (
        float(mean_diff) if np.isfinite(mean_diff) else np.nan,
        float(ano_std_diff) if np.isfinite(ano_std_diff) else np.nan,
        float(ano_trend_diff) if np.isfinite(ano_trend_diff) else np.nan,
        float(std_map_diff) if np.isfinite(std_map_diff) else np.nan,
        float(trend_map_diff) if np.isfinite(trend_map_diff) else np.nan,
        float(vectcorr_mean) if np.isfinite(vectcorr_mean) else np.nan,
        float(corr) if np.isfinite(corr) else np.nan,
        float(rmse) if np.isfinite(rmse) else np.nan,
    )


def _fmt_sid_month_stat(value: float) -> str:
    return _format_min_sig(value, min_sig=3)


def _sid_extract_month_absolute_stats(
    metric_dict: Optional[dict], month: Optional[int]
) -> Tuple[float, float, float, float, float]:
    """Return absolute SID diagnostics for one key month (or annual mode).

    Returns:
        (mean_value, ano_ts_std, ano_ts_trend, std_map_mean, trend_map_mean_abs)
    """
    if not isinstance(metric_dict, dict):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mean_value = np.nan
    ano_ts_std = np.nan
    ano_ts_trend = np.nan
    std_map_mean = np.nan
    trend_map_mean = np.nan

    if month is None:
        for clim_key in ('speed_clim', 'MKE_clim'):
            clim = np.asarray(metric_dict.get(clim_key, np.array([])), dtype=float)
            if clim.ndim != 3 or clim.shape[0] <= 0:
                continue
            n_use = min(int(clim.shape[0]), 12)
            if n_use <= 0:
                continue
            month_means = np.array([np.nanmean(clim[ii]) for ii in range(n_use)], dtype=float)
            weights = np.asarray(DAYS_PER_MONTH[:n_use], dtype=float)
            valid = np.isfinite(month_means)
            if np.any(valid):
                w = weights[valid]
                if np.sum(w) > 0:
                    mean_value = float(np.nansum(month_means[valid] * w) / np.sum(w))
                    break
    else:
        m_idx = int(month) - 1
        for clim_key in ('speed_clim', 'MKE_clim'):
            clim = np.asarray(metric_dict.get(clim_key, np.array([])), dtype=float)
            if clim.ndim == 3 and clim.shape[0] > m_idx:
                vv = np.nanmean(clim[m_idx])
                if np.isfinite(vv):
                    mean_value = float(vv)
                    break

    keys = None
    for cand in (
        metric_dict.get('yearmon_list'),
        metric_dict.get('time_keys'),
        metric_dict.get('month_list'),
    ):
        if cand is None:
            continue
        if isinstance(cand, np.ndarray) and cand.size <= 0:
            continue
        if isinstance(cand, (list, tuple)) and len(cand) <= 0:
            continue
        keys = cand
        break
    for ts_key in ('speed_ts_ano', 'MKE_ts_ano'):
        ts = np.asarray(metric_dict.get(ts_key, np.array([])), dtype=float).squeeze()
        if ts.ndim != 1 or ts.size <= 0:
            continue
        n_use = int(ts.size)
        mask = np.ones(n_use, dtype=bool) if month is None else _sid_month_mask(keys, n_use, int(month))
        valid = mask & np.isfinite(ts)
        if int(np.sum(valid)) <= 0:
            continue
        vals = ts[valid]
        ano_ts_std = float(np.nanstd(vals))
        if vals.size >= 2:
            x = np.arange(vals.size, dtype=float)
            try:
                slope = float(np.polyfit(x, vals, 1)[0])
                factor = 12.0 * 10.0 if month is None else 10.0
                ano_ts_trend = slope * factor
            except Exception:
                ano_ts_trend = np.nan
        break

    if not np.isfinite(ano_ts_trend):
        for tr_key in ('speed_ts_ano_tr', 'MKE_ts_ano_tr'):
            tr = metric_dict.get(tr_key)
            if month is not None:
                continue
            if hasattr(tr, 'slope'):
                try:
                    ano_ts_trend = float(tr.slope) * 12.0 * 10.0
                    break
                except Exception:
                    ano_ts_trend = np.nan

    for ano_key in ('speed_ano', 'MKE_ano'):
        ano = np.asarray(metric_dict.get(ano_key, np.array([])), dtype=float)
        if ano.ndim != 3 or ano.shape[0] < 2:
            continue
        n_use = int(ano.shape[0])
        mask = np.ones(n_use, dtype=bool) if month is None else _sid_month_mask(keys, n_use, int(month))
        if int(np.sum(mask)) < 2:
            continue
        month_ano = ano[mask, :, :]
        std_map = np.nanstd(month_ano, axis=0)
        std_map_mean = float(np.nanmean(std_map)) if np.any(np.isfinite(std_map)) else np.nan
        tr_map = _sid_month_trend_map(
            month_ano,
            samples_per_year=(12.0 if month is None else 1.0),
        )
        trend_map_mean = (
            float(np.nanmean(np.abs(tr_map)))
            if np.any(np.isfinite(tr_map))
            else np.nan
        )
        break

    if not np.isfinite(std_map_mean):
        std_map = np.asarray(metric_dict.get('MKE_ano_std', np.array([])), dtype=float)
        if std_map.ndim == 2 and np.any(np.isfinite(std_map)):
            std_map_mean = float(np.nanmean(std_map))
    if not np.isfinite(trend_map_mean):
        tr_map = np.asarray(metric_dict.get('MKE_ano_tr', np.array([])), dtype=float)
        if tr_map.ndim == 2 and np.any(np.isfinite(tr_map)):
            trend_map_mean = float(np.nanmean(np.abs(tr_map)))

    return (
        float(mean_value) if np.isfinite(mean_value) else np.nan,
        float(ano_ts_std) if np.isfinite(ano_ts_std) else np.nan,
        float(ano_ts_trend) if np.isfinite(ano_ts_trend) else np.nan,
        float(std_map_mean) if np.isfinite(std_map_mean) else np.nan,
        float(trend_map_mean) if np.isfinite(trend_map_mean) else np.nan,
    )


def _sid_extract_domain_absolute_stats(
    metric_dict: Optional[dict],
    domain: Tuple[str, Optional[Any]],
    hemisphere: str,
) -> Tuple[float, float, float, float, float]:
    """Return absolute SID stats for one domain selector (annual/month/season)."""
    kind = str(domain[0]).lower() if domain else 'annual'
    val = domain[1] if domain else None
    if kind == 'annual':
        return _sid_extract_month_absolute_stats(metric_dict, None)
    if kind == 'month':
        try:
            month = int(val) if val is not None else None
        except Exception:
            month = None
        return _sid_extract_month_absolute_stats(metric_dict, month)
    if kind == 'season':
        season_name = str(val or '')
        sdiag = _sid_seasonal_diagnostics(metric_dict, season_name, hemisphere)
        clim_map = np.asarray(sdiag.get('clim_map', np.array([])), dtype=float)
        std_map = np.asarray(sdiag.get('std_map', np.array([])), dtype=float)
        tr_map = np.asarray(sdiag.get('trend_map', np.array([])), dtype=float)
        ts_ano = np.asarray(sdiag.get('ts_ano', np.array([])), dtype=float).reshape(-1)
        mean_value = float(np.nanmean(clim_map)) if clim_map.ndim == 2 and np.any(np.isfinite(clim_map)) else np.nan
        std_map_mean = float(np.nanmean(std_map)) if std_map.ndim == 2 and np.any(np.isfinite(std_map)) else np.nan
        trend_map_mean = float(np.nanmean(np.abs(tr_map))) if tr_map.ndim == 2 and np.any(np.isfinite(tr_map)) else np.nan
        ano_ts_std = float(np.nanstd(ts_ano)) if ts_ano.size > 0 and np.any(np.isfinite(ts_ano)) else np.nan
        ano_ts_trend = np.nan
        if ts_ano.size >= 2:
            valid = np.isfinite(ts_ano)
            if int(np.sum(valid)) >= 2:
                y = ts_ano[valid]
                xx = np.arange(y.size, dtype=float)
                try:
                    ano_ts_trend = float(np.polyfit(xx, y, 1)[0]) * 10.0
                except Exception:
                    ano_ts_trend = np.nan
        return (
            float(mean_value) if np.isfinite(mean_value) else np.nan,
            float(ano_ts_std) if np.isfinite(ano_ts_std) else np.nan,
            float(ano_ts_trend) if np.isfinite(ano_ts_trend) else np.nan,
            float(std_map_mean) if np.isfinite(std_map_mean) else np.nan,
            float(trend_map_mean) if np.isfinite(trend_map_mean) else np.nan,
        )
    return _sid_extract_month_absolute_stats(metric_dict, None)


def _sid_extract_domain_extended_stats(
    diff_dict: Optional[dict],
    domain: Tuple[str, Optional[Any]],
    hemisphere: str,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return SID diff stats for one domain selector (annual/month/season)."""
    kind = str(domain[0]).lower() if domain else 'annual'
    val = domain[1] if domain else None
    if kind == 'annual':
        return _sid_extract_month_extended_stats(diff_dict, None)
    if kind == 'month':
        try:
            month = int(val) if val is not None else None
        except Exception:
            month = None
        return _sid_extract_month_extended_stats(diff_dict, month)
    if kind == 'season':
        if not isinstance(diff_dict, dict):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        season_name = str(val or '')
        sid1 = diff_dict.get('sid1_metric') if isinstance(diff_dict.get('sid1_metric'), dict) else {}
        sid2 = diff_dict.get('sid2_metric') if isinstance(diff_dict.get('sid2_metric'), dict) else {}
        d1 = _sid_seasonal_diagnostics(sid1, season_name, hemisphere)
        d2 = _sid_seasonal_diagnostics(sid2, season_name, hemisphere)

        c1 = np.asarray(d1.get('clim_map', np.array([])), dtype=float)
        c2 = np.asarray(d2.get('clim_map', np.array([])), dtype=float)
        s1 = np.asarray(d1.get('std_map', np.array([])), dtype=float)
        s2 = np.asarray(d2.get('std_map', np.array([])), dtype=float)
        t1 = np.asarray(d1.get('trend_map', np.array([])), dtype=float)
        t2 = np.asarray(d2.get('trend_map', np.array([])), dtype=float)
        ts1 = np.asarray(d1.get('ts_ano', np.array([])), dtype=float).reshape(-1)
        ts2 = np.asarray(d2.get('ts_ano', np.array([])), dtype=float).reshape(-1)

        mean_diff = (
            utils.MatrixDiff(c1, c2, metric='MAE', mask=True)
            if c1.ndim == 2 and c2.ndim == 2 and c1.shape == c2.shape
            else np.nan
        )
        std_map_diff = (
            utils.MatrixDiff(s1, s2, metric='MAE', mask=True)
            if s1.ndim == 2 and s2.ndim == 2 and s1.shape == s2.shape
            else np.nan
        )
        trend_map_diff = (
            utils.MatrixDiff(t1, t2, metric='MAE', mask=True)
            if t1.ndim == 2 and t2.ndim == 2 and t1.shape == t2.shape
            else np.nan
        )
        vectcorr_mean = _sid_vector_cosine_mean(
            np.asarray(d1.get('u_clim_map', np.array([])), dtype=float),
            np.asarray(d1.get('v_clim_map', np.array([])), dtype=float),
            np.asarray(d2.get('u_clim_map', np.array([])), dtype=float),
            np.asarray(d2.get('v_clim_map', np.array([])), dtype=float),
        )
        if not np.isfinite(vectcorr_mean):
            vectcorr_mean = diff_dict.get('vectcorr_mean', np.nan)

        ano_std_diff = np.nan
        ano_trend_diff = np.nan
        corr = np.nan
        rmse = np.nan
        n_use = min(ts1.size, ts2.size)
        if n_use >= 2:
            y1 = ts1[:n_use]
            y2 = ts2[:n_use]
            valid = np.isfinite(y1) & np.isfinite(y2)
            if int(np.sum(valid)) >= 2:
                y1v = y1[valid]
                y2v = y2[valid]
                ano_std_diff = abs(float(np.nanstd(y1v)) - float(np.nanstd(y2v)))
                if np.nanstd(y1v) > 0 and np.nanstd(y2v) > 0:
                    corr = float(np.corrcoef(y1v, y2v)[0, 1])
                rmse = float(np.sqrt(np.nanmean((y1v - y2v) ** 2)))
                xx = np.arange(y1v.size, dtype=float)
                try:
                    slope1 = float(np.polyfit(xx, y1v, 1)[0])
                    slope2 = float(np.polyfit(xx, y2v, 1)[0])
                    ano_trend_diff = abs((slope1 - slope2) * 10.0)
                except Exception:
                    ano_trend_diff = np.nan

        return (
            float(mean_diff) if np.isfinite(mean_diff) else np.nan,
            float(ano_std_diff) if np.isfinite(ano_std_diff) else np.nan,
            float(ano_trend_diff) if np.isfinite(ano_trend_diff) else np.nan,
            float(std_map_diff) if np.isfinite(std_map_diff) else np.nan,
            float(trend_map_diff) if np.isfinite(trend_map_diff) else np.nan,
            float(vectcorr_mean) if np.isfinite(vectcorr_mean) else np.nan,
            float(corr) if np.isfinite(corr) else np.nan,
            float(rmse) if np.isfinite(rmse) else np.nan,
        )
    return _sid_extract_month_extended_stats(diff_dict, None)


def _build_sid_month_period_raw_table(
    obs1_metric: Optional[dict],
    obs2_metric: Optional[dict],
    model_metrics: List[Optional[dict]],
    model_labels: List[str],
    hemisphere: str = 'nh',
    obs1_label: str = 'obs1 (baseline)',
    obs2_label: str = 'obs2',
) -> Dict[str, Any]:
    """Build annual/season/month raw (absolute) table for SID."""
    headers = [
        'Model/Obs Name',
        'Speed Clim Mean',
        'Speed Ano TS Std',
        'Speed Ano TS Trend',
        'Speed Ano Std Map Mean',
        'Speed Ano Trend Map Mean',
    ]
    units = ['', 'm/s', 'm/s', 'm/s/decade', 'm/s', 'm/s/decade']
    specs = _sid_domain_specs(hemisphere)
    season_order = [name for name, _ in specs]
    seasons: Dict[str, List[List[str]]] = {}

    for season_name, domain in specs:
        rows: List[List[str]] = []
        if isinstance(obs1_metric, dict):
            m0, s0, t0, sm0, tm0 = _sid_extract_domain_absolute_stats(obs1_metric, domain, hemisphere)
            rows.append([
                str(obs1_label),
                _fmt_sid_month_stat(m0),
                _fmt_sid_month_stat(s0),
                _fmt_sid_month_stat(t0),
                _fmt_sid_month_stat(sm0),
                _fmt_sid_month_stat(tm0),
            ])
        if isinstance(obs2_metric, dict):
            m0, s0, t0, sm0, tm0 = _sid_extract_domain_absolute_stats(obs2_metric, domain, hemisphere)
            rows.append([
                str(obs2_label),
                _fmt_sid_month_stat(m0),
                _fmt_sid_month_stat(s0),
                _fmt_sid_month_stat(t0),
                _fmt_sid_month_stat(sm0),
                _fmt_sid_month_stat(tm0),
            ])
        for ii, metric_dict in enumerate(model_metrics):
            label = model_labels[ii] if ii < len(model_labels) else f'model{ii + 1}'
            m1, s1, t1, sm1, tm1 = _sid_extract_domain_absolute_stats(metric_dict, domain, hemisphere)
            rows.append([
                label,
                _fmt_sid_month_stat(m1),
                _fmt_sid_month_stat(s1),
                _fmt_sid_month_stat(t1),
                _fmt_sid_month_stat(sm1),
                _fmt_sid_month_stat(tm1),
            ])
        seasons[season_name] = rows

    return {
        'type': 'seasonal_table',
        'season_order': season_order,
        'headers': headers,
        'rows': [],
        'units': units,
        'seasons': seasons,
    }


def _build_sid_month_period_table(obs2_diff: Optional[dict],
                                  diff_dicts: List[dict],
                                  model_labels: List[str],
                                  hemisphere: str = 'nh',
                                  obs1_label: str = 'obs1 (baseline)',
                                  obs2_label: str = 'obs2') -> Dict[str, Any]:
    """Build annual/season/month period tabs for SID scalar table."""
    headers = [
        'Model/Obs Name',
        'Speed Clim MAE',
        'Speed Ano TS Std Diff',
        'Speed Ano TS Trend Diff',
        'Speed Ano Std Map Diff',
        'Speed Ano Trend Map Diff',
        'VectCorr Mean',
        'Ano TS Corr',
        'Ano TS RMSE',
    ]
    units = ['', 'm/s', 'm/s', 'm/s/decade', 'm/s', 'm/s/decade', '', '', 'm/s']
    specs = _sid_domain_specs(hemisphere)
    season_order = [name for name, _ in specs]
    seasons: Dict[str, List[List[str]]] = {}

    for season_name, domain in specs:
        rows: List[List[str]] = [[
            str(obs1_label),
            _fmt_sid_month_stat(0.0),
            _fmt_sid_month_stat(0.0),
            _fmt_sid_month_stat(0.0),
            _fmt_sid_month_stat(0.0),
            _fmt_sid_month_stat(0.0),
            _fmt_sid_month_stat(0.0),
            _fmt_sid_month_stat(0.0),
            _fmt_sid_month_stat(0.0),
        ]]
        obs2_vals: Optional[List[float]] = None
        if isinstance(obs2_diff, dict):
            md0, sd0, td0, sm0, tm0, vc0, c0, r0 = _sid_extract_domain_extended_stats(
                obs2_diff, domain, hemisphere
            )
            obs2_vals = [md0, sd0, td0, sm0, tm0, vc0, c0, r0]
            rows.append([
                str(obs2_label),
                _fmt_sid_month_stat(_obs2_identity_ratio(md0)),
                _fmt_sid_month_stat(_obs2_identity_ratio(sd0)),
                _fmt_sid_month_stat(_obs2_identity_ratio(td0)),
                _fmt_sid_month_stat(_obs2_identity_ratio(sm0)),
                _fmt_sid_month_stat(_obs2_identity_ratio(tm0)),
                _fmt_sid_month_stat(_obs2_identity_ratio(vc0)),
                _fmt_sid_month_stat(_obs2_identity_ratio(c0)),
                _fmt_sid_month_stat(_obs2_identity_ratio(r0)),
            ])

        for ii, dct in enumerate(diff_dicts):
            label = model_labels[ii] if ii < len(model_labels) else f'model{ii + 1}'
            md, sd, td, sm, tm, vc, cc, rr = _sid_extract_domain_extended_stats(
                dct, domain, hemisphere
            )
            vals = [md, sd, td, sm, tm, vc, cc, rr]
            if obs2_vals is not None:
                vals = [_calc_obs2_ratio(v, b) for v, b in zip(vals, obs2_vals)]
            rows.append([
                label,
                _fmt_sid_month_stat(vals[0]),
                _fmt_sid_month_stat(vals[1]),
                _fmt_sid_month_stat(vals[2]),
                _fmt_sid_month_stat(vals[3]),
                _fmt_sid_month_stat(vals[4]),
                _fmt_sid_month_stat(vals[5]),
                _fmt_sid_month_stat(vals[6]),
                _fmt_sid_month_stat(vals[7]),
            ])
        seasons[season_name] = rows

    return {
        'type': 'seasonal_table',
        'season_order': season_order,
        'headers': headers,
        'rows': [],
        'units': units,
        'seasons': seasons,
    }


def eval_sidrift(case_name: str, recipe: RR.RecipeReader,
                 data_dir: str, output_dir: str,
                 recalculate: bool = False,
                 jobs: int = 1) -> Optional[dict]:
    """Evaluate sea ice drift (SIdrift)."""
    module = 'SIdrift'
    _log_module_header(module)

    module_vars = recipe.variables[module]
    hemisphere = recipe.hemisphere
    year_sta, year_end = _get_year_range(module_vars)
    plot_opts = _load_plot_options(case_name)
    _apply_plot_runtime_options(plot_opts, module)
    line_colors = _plot_options_get_module(plot_opts, module, ['line', 'model_colors'], None)
    if not isinstance(line_colors, (list, tuple)) or len(line_colors) == 0:
        line_colors = None
    line_styles = _plot_options_get_module(plot_opts, module, ['line', 'model_linestyles'], None)
    if not isinstance(line_styles, (list, tuple)) or len(line_styles) == 0:
        line_styles = None

    mke_raw_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'mke_raw_cmap'], 'viridis')
    mke_diff_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'mke_diff_cmap'], 'RdBu_r')
    vectcorr_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'vectcorr_cmap'], 'viridis')
    std_raw_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'std_raw_cmap'], 'Purples')
    std_diff_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'std_diff_cmap'], 'RdBu_r')
    trend_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'trend_cmap'], 'RdBu_r')
    heatmap_cmap = _plot_options_get_module(plot_opts, module, ['heatmap', 'cmap'], 'RdBu_r')
    heat_ratio_vmin = float(_plot_options_get_module(plot_opts, module, ['heatmap', 'ratio_vmin'], 0.5))
    heat_ratio_vmax = float(_plot_options_get_module(plot_opts, module, ['heatmap', 'ratio_vmax'], 2.0))
    heat_main_height_scale = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'main_height_scale'], 0.50,
    ))
    heat_season_wspace = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_subplot_wspace'], 0.30,
    ))
    heat_season_hspace = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_subplot_hspace'], 0.18,
    ))
    heat_season_cbar_fraction = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_colorbar_fraction'], 0.018,
    ))
    heat_season_cbar_pad = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_colorbar_pad'], 0.055,
    ))
    heat_season_cbar_aspect = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_colorbar_aspect'], 40.0,
    ))
    heat_season_height_scale = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_height_scale'], 0.84,
    ))
    heat_season_right = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_right_margin'], 0.88,
    ))
    key_month_legend_gap = float(_plot_options_get_module(
        plot_opts, module, ['line', 'key_month_ano_legend_y_gap'], 0.060,
    ))

    vector_quiver_skip = int(_plot_options_get_module(plot_opts, module, ['vector', 'quiver_skip'], 3))
    vector_speed_vmin = float(_plot_options_get_module(plot_opts, module, ['vector', 'speed_vmin'], 0.0))
    vector_speed_vmax = float(_plot_options_get_module(plot_opts, module, ['vector', 'speed_vmax'], 0.3))
    vector_quiver_scale = float(_plot_options_get_module(plot_opts, module, ['vector', 'quiver_scale'], 1.0))
    vector_quiver_width = float(_plot_options_get_module(plot_opts, module, ['vector', 'quiver_width'], 0.004))
    vector_quiver_min_speed = float(_plot_options_get_module(plot_opts, module, ['vector', 'quiver_min_speed'], 0.0))
    n_models_declared = len(module_vars.get('model_file_u', []))
    model_direction = _validate_sidrift_direction_config(module_vars, n_models_declared)

    fig_dir = Path(output_dir) / module
    if _check_outputs_exist(module, fig_dir, hemisphere, recalculate=recalculate):
        logger.info(f"{module} evaluation skipped - all outputs exist.")
        return None

    cache_file = _get_metrics_cache_file(case_name, output_dir, hemisphere, module)
    grid_file = _get_eval_grid(case_name, module, hemisphere)

    model_labels = _get_recipe_model_labels(module, module_vars, len(module_vars.get('model_file_u') or []))
    obs_labels = _get_reference_labels(module_vars, hemisphere)
    obs1_plot_label = obs_labels[0] if len(obs_labels) >= 1 else 'obs1'
    obs2_plot_label = obs_labels[1] if len(obs_labels) >= 2 else 'obs2'
    obs_dict = None
    obs2_dict = None
    obs_diff_dict = None
    obs_diff_dict_matched = None
    model_dicts: List[dict] = []
    diff_dicts: List[dict] = []
    matched_diff_dicts: List[dict] = []
    matched_model_labels_used: List[str] = []
    model_frames: List[str] = []
    obs_files: List[str] = []
    obs_u_files: List[str] = []
    obs_v_files: List[str] = []
    model_u_files: List[str] = []
    model_v_files: List[str] = []
    u_key = module_vars.get('ref_var_u', 'u')
    v_key = module_vars.get('ref_var_v', 'v')
    model_u_key = module_vars.get('model_var_u', 'u')
    model_v_key = module_vars.get('model_var_v', 'v')
    cache_loaded = False

    if not recalculate:
        cached = _load_module_cache(cache_file, module, hemisphere)
        if cached is not None and cached.get('payload_kind') == module:
            try:
                records = cached.get('records', {})
                obs_dict = records.get('obs1_1m')
                obs2_dict = records.get('obs2_1m')
                obs_diff_dict = records.get('obs2_vs_obs1_2m')
                obs_diff_dict_matched = records.get('obs2_vs_obs1_matched_2m')
                model_records = cached.get('model_records', [])
                diff_records = cached.get('diff_records', [])
                matched_diff_records = cached.get('matched_diff_records', [])
                model_dicts = [records[r] for r in model_records if r in records]
                diff_dicts = [records[r] for r in diff_records if r in records]
                matched_diff_dicts = [records[r] for r in matched_diff_records if r in records]
                model_labels = cached.get('model_labels', model_labels)
                obs_u_files = list(cached.get('obs_u_files', obs_u_files) or [])
                obs_v_files = list(cached.get('obs_v_files', obs_v_files) or [])
                model_u_files = list(cached.get('model_u_files', model_u_files) or [])
                model_v_files = list(cached.get('model_v_files', model_v_files) or [])
                matched_model_labels_used = list(
                    cached.get('matched_model_labels_used', matched_model_labels_used)
                )
                if not matched_model_labels_used and matched_diff_dicts:
                    matched_model_labels_used = _get_recipe_model_labels(
                        module, module_vars, len(matched_diff_dicts)
                    )
                if not model_labels and model_dicts:
                    model_labels = _get_recipe_model_labels(module, module_vars, len(model_dicts))
                cache_loaded = obs_dict is not None and bool(model_dicts) and bool(diff_dicts) and bool(matched_diff_dicts)
                if cache_loaded:
                    logger.info("Loaded %s metrics from cache: %s", module, cache_file)
            except Exception as exc:
                logger.warning("Cache payload for %s is incomplete (%s). Recalculating.", module, exc)
                cache_loaded = False

    if not cache_loaded:
        preprocessor = PP.DataPreprocessor(case_name, module, hemisphere=hemisphere)
        grid_file = preprocessor.gen_eval_grid()
        file_groups = recipe.validate_module(module)
        obs_files = preprocessor.prep_obs(
            frequency='monthly',
            output_dir=data_dir,
            jobs=jobs,
        )
        obs_u_files = obs_v_files = obs_files

        if file_groups:
            if len(file_groups) % 2 != 0:
                raise ValueError(
                    f"SIdrift model file groups should be paired u/v, got {len(file_groups)} groups."
                )
            n_models = len(file_groups) // 2
            model_u_groups = file_groups[:n_models]
            model_v_groups = file_groups[n_models:]
        else:
            n_models = 0
            model_u_groups = []
            model_v_groups = []

        if len(model_direction) != n_models:
            raise ValueError(
                f"SIdrift model_direction count ({len(model_direction)}) must match "
                f"processed model count ({n_models})."
            )

        model_angle = module_vars.get('model_angle', [])
        model_u_files, model_v_files, _sidrift_prep_audit = preprocessor.prep_sidrift_models(
            u_file_groups=model_u_groups,
            v_file_groups=model_v_groups,
            model_direction=model_direction,
            model_angle=model_angle,
            frequency='monthly',
            output_dir=data_dir,
            overwrite=recalculate,
            prefer_metadata=True,
            jobs=jobs,
        )

        if len(model_u_files) != len(model_v_files):
            raise ValueError(
                f"SIdrift processed u/v count mismatch: {len(model_u_files)} vs {len(model_v_files)}."
            )
        if len(model_u_files) != n_models:
            if n_models > 0 and len(model_u_files) == 0:
                worker_log_dir = Path('cases') / case_name / 'logs' / 'workers'
                raise RuntimeError(
                    "SIdrift model preprocessing produced 0 outputs while recipe declares "
                    f"{n_models} model(s). Check worker logs under "
                    f"{worker_log_dir} for preprocessing errors."
                )
            logger.warning(
                "SIdrift processed model count (%d) differs from declared count (%d).",
                len(model_u_files), n_models,
            )
            n_models = len(model_u_files)

        metric = _get_metric(module_vars)
        model_labels = _get_recipe_model_labels(module, module_vars, n_models)
        sid_metrics = SIM.SIDMetrics(
            grid_file=grid_file,
            hemisphere=hemisphere,
            time_sta=year_sta,
            time_end=year_end,
            metric=metric,
            projection=module_vars.get('projection', 'stere'),
        )

        obs_direction = 'xy'

        obs_dict = None
        obs2_dict = None
        obs_diff_dict = None
        obs_diff_dict_matched = None
        obs_u = None
        obs_v = None
        obs2_u = None
        obs2_v = None
        if obs_u_files and obs_v_files:
            obs_u = os.path.join(data_dir, obs_u_files[0])
            obs_v = os.path.join(data_dir, obs_v_files[0])
            obs_dict = sid_metrics.SID_1M_metrics(
                obs_u, u_key, obs_v, v_key, model_direction=obs_direction
            )
            if len(obs_files) >= 2:
                obs2_u = os.path.join(data_dir, obs_u_files[1])
                obs2_v = os.path.join(data_dir, obs_v_files[1])
                obs2_dict = sid_metrics.SID_1M_metrics(
                    obs2_u, u_key, obs2_v, v_key, model_direction=obs_direction
                )
                obs_diff_dict = sid_metrics.SID_2M_metrics(
                    obs_u, u_key, obs_v, v_key,
                    obs2_u, u_key, obs2_v, v_key,
                    model_direction1=obs_direction, model_direction2=obs_direction,
                )
                obs_diff_dict_matched = sid_metrics.SID_2M_metrics(
                    obs_u, u_key, obs_v, v_key,
                    obs2_u, u_key, obs2_v, v_key,
                    model_direction1=obs_direction, model_direction2=obs_direction,
                    strict_obs_match=True,
                    obs_match_u_file=obs2_u,
                    obs_match_u_key=u_key,
                    obs_match_v_file=obs2_v,
                    obs_match_v_key=v_key,
                    obs_match_direction=obs_direction,
                )

        model_dicts = []
        diff_dicts = []
        matched_diff_dicts = []
        matched_model_labels_used = []
        model_frames: List[str] = []

        model_tasks = list(enumerate(zip(model_u_files, model_v_files)))
        if jobs > 1 and model_tasks:
            stage_dir = _get_stage_dir(case_name, output_dir, hemisphere, module) / 'model_metrics'
            stage_dir.mkdir(parents=True, exist_ok=True)

            def _worker(task: Tuple[int, Tuple[str, str]]) -> Dict[str, Any]:
                idx, (mf_u, mf_v) = task
                model_dir = 'xy'
                model_u_path = os.path.join(data_dir, mf_u)
                model_v_path = os.path.join(data_dir, mf_v)
                m_dict = sid_metrics.SID_1M_metrics(
                    model_u_path, model_u_key, model_v_path, model_v_key,
                    model_direction=model_dir,
                )
                model_source_dir = str(m_dict.get('source_direction', model_dir))
                model_frame = str(m_dict.get('vector_frame', 'xy'))
                d_dict = None
                d_matched_dict = None
                if obs_dict is not None:
                    d_dict = sid_metrics.SID_2M_metrics(
                        obs_u, u_key, obs_v, v_key,
                        model_u_path, model_u_key, model_v_path, model_v_key,
                        model_direction1=obs_direction, model_direction2=model_source_dir,
                    )
                    d_matched_dict = sid_metrics.SID_2M_metrics(
                        obs_u, u_key, obs_v, v_key,
                        model_u_path, model_u_key, model_v_path, model_v_key,
                        model_direction1=obs_direction, model_direction2=model_source_dir,
                        strict_obs_match=True,
                        obs_match_u_file=obs2_u,
                        obs_match_u_key=u_key,
                        obs_match_v_file=obs2_v,
                        obs_match_v_key=v_key,
                        obs_match_direction=obs_direction,
                    )
                payload_file = stage_dir / f'model_{idx:04d}.pkl'
                _save_pickle_atomic(payload_file, {
                    'model_index': idx,
                    'model_1m': m_dict,
                    'diff_2m': d_dict,
                    'diff_matched_2m': d_matched_dict,
                    'model_frame': model_frame,
                })
                return {'payload_file': str(payload_file)}

            stage_refs = _parallel_map_ordered(
                items=model_tasks,
                worker_fn=_worker,
                max_workers=jobs,
                task_label=f'{hemisphere}/{module}/model-metrics',
            )
            staged_payloads = _load_staged_payloads(stage_refs)
            for payload in staged_payloads:
                model_dicts.append(payload['model_1m'])
                model_frames.append(str(payload.get('model_frame', 'xy')))
                if payload.get('diff_2m') is not None:
                    diff_dicts.append(payload['diff_2m'])
                if payload.get('diff_matched_2m') is not None:
                    matched_diff_dicts.append(payload['diff_matched_2m'])
                    model_idx = int(payload.get('model_index', len(matched_model_labels_used)))
                    if model_idx < len(model_labels):
                        matched_model_labels_used.append(model_labels[model_idx])
                    else:
                        matched_model_labels_used.append(f'model{model_idx + 1}')
        else:
            for idx, (mf_u, mf_v) in model_tasks:
                # Model vectors are normalized to evaluation-grid x/y during preprocessing.
                model_dir = 'xy'
                model_u_path = os.path.join(data_dir, mf_u)
                model_v_path = os.path.join(data_dir, mf_v)
                m_dict = sid_metrics.SID_1M_metrics(
                    model_u_path, model_u_key, model_v_path, model_v_key,
                    model_direction=model_dir,
                )
                model_source_dir = str(m_dict.get('source_direction', model_dir))
                model_frame = str(m_dict.get('vector_frame', 'xy'))
                model_frames.append(model_frame)
                model_dicts.append(m_dict)
                if obs_dict is not None:
                    d_dict = sid_metrics.SID_2M_metrics(
                        obs_u, u_key, obs_v, v_key,
                        model_u_path, model_u_key, model_v_path, model_v_key,
                        model_direction1=obs_direction, model_direction2=model_source_dir,
                    )
                    diff_dicts.append(d_dict)
                    d_matched_dict = sid_metrics.SID_2M_metrics(
                        obs_u, u_key, obs_v, v_key,
                        model_u_path, model_u_key, model_v_path, model_v_key,
                        model_direction1=obs_direction, model_direction2=model_source_dir,
                        strict_obs_match=True,
                        obs_match_u_file=obs2_u,
                        obs_match_u_key=u_key,
                        obs_match_v_file=obs2_v,
                        obs_match_v_key=v_key,
                        obs_match_direction=obs_direction,
                    )
                    matched_diff_dicts.append(d_matched_dict)
                    matched_model_labels_used.append(model_labels[idx] if idx < len(model_labels) else f'model{idx + 1}')

        model_records = [f'model{i}_1m' for i in range(len(model_dicts))]
        diff_records = [f'model{i}_vs_obs1_2m' for i in range(len(diff_dicts))]
        matched_diff_records = [f'model{i}_vs_obs1_matched_2m' for i in range(len(matched_diff_dicts))]
        records = {}
        if obs_dict is not None:
            records['obs1_1m'] = obs_dict
        if obs2_dict is not None:
            records['obs2_1m'] = obs2_dict
        if obs_diff_dict is not None:
            records['obs2_vs_obs1_2m'] = obs_diff_dict
        if obs_diff_dict_matched is not None:
            records['obs2_vs_obs1_matched_2m'] = obs_diff_dict_matched
        records.update({name: d for name, d in zip(model_records, model_dicts)})
        records.update({name: d for name, d in zip(diff_records, diff_dicts)})
        records.update({name: d for name, d in zip(matched_diff_records, matched_diff_dicts)})

        used_entities: set = set()
        entity_groups: Dict[str, str] = {}

        obs1_entity = None
        if obs_dict is not None:
            obs1_entity = _unique_entity_name(
            preferred=obs_labels[0] if len(obs_labels) >= 1 else 'Reference_1',
            fallback='Reference_1',
                used=used_entities,
            )
            entity_groups['obs1_1m'] = obs1_entity

        if obs2_dict is not None:
            obs2_entity = _unique_entity_name(
                preferred=obs_labels[1] if len(obs_labels) >= 2 else 'Reference_2',
                fallback='Reference_2',
                used=used_entities,
            )
            entity_groups['obs2_1m'] = obs2_entity
            if obs_diff_dict is not None and obs1_entity is not None:
                entity_groups['obs2_vs_obs1_2m'] = _unique_entity_name(
                    preferred=f'{obs1_entity}_vs_{obs2_entity}',
                    fallback='Reference_1_vs_Reference_2',
                    used=used_entities,
                )

        for i, rec_name in enumerate(model_records):
            model_entity = _unique_entity_name(
                preferred=model_labels[i] if i < len(model_labels) else f'{module}_dataset_{i + 1}',
                fallback=f'{module}_dataset_{i + 1}',
                used=used_entities,
            )
            entity_groups[rec_name] = model_entity

        if obs1_entity is not None:
            for i, rec_name in enumerate(diff_records):
                model_entity = entity_groups.get(model_records[i], f'{module}_dataset_{i + 1}')
                entity_groups[rec_name] = _unique_entity_name(
                    preferred=f'{obs1_entity}_vs_{model_entity}',
                    fallback=f'Reference_1_vs_{module}_dataset_{i + 1}',
                    used=used_entities,
                )
            for i, rec_name in enumerate(matched_diff_records):
                model_name = (
                    matched_model_labels_used[i]
                    if i < len(matched_model_labels_used)
                    else f'{module}_dataset_{i + 1}'
                )
                entity_groups[rec_name] = _unique_entity_name(
                    preferred=f'{obs1_entity}_vs_{model_name}_matched',
                    fallback=f'Reference_1_vs_{module}_dataset_{i + 1}_matched',
                    used=used_entities,
                )

        try:
            _save_module_cache(
                cache_file=cache_file,
                case_name=case_name,
                module=module,
                hemisphere=hemisphere,
                start_year=year_sta,
                end_year=year_end,
                payload_meta={
                    'payload_kind': module,
                    'model_labels': model_labels,
                    'matched_model_labels_used': matched_model_labels_used,
                    'model_records': model_records,
                    'diff_records': diff_records,
                    'matched_diff_records': matched_diff_records,
                    'obs_u_files': list(obs_u_files),
                    'obs_v_files': list(obs_v_files),
                    'model_u_files': list(model_u_files),
                    'model_v_files': list(model_v_files),
                },
                records=records,
                entity_groups=entity_groups,
                grid_file=grid_file,
            )
            logger.info("Saved %s metrics to cache: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to save %s cache (%s).", module, exc)

    cached_for_plot = _load_module_cache(cache_file, module, hemisphere)
    if cached_for_plot is not None and cached_for_plot.get('payload_kind') == module:
        try:
            records = cached_for_plot.get('records', {})
            obs_dict_plot = records.get('obs1_1m')
            model_records_plot = cached_for_plot.get('model_records', [])
            diff_records_plot = cached_for_plot.get('diff_records', [])
            matched_diff_records_plot = cached_for_plot.get('matched_diff_records', [])
            model_dicts_plot = [records[r] for r in model_records_plot if r in records]
            matched_diff_dicts_plot = [records[r] for r in matched_diff_records_plot if r in records]
            if obs_dict_plot is not None and model_dicts_plot:
                obs_dict = obs_dict_plot
                obs2_dict = records.get('obs2_1m')
                obs_diff_dict = records.get('obs2_vs_obs1_2m')
                obs_diff_dict_matched = records.get('obs2_vs_obs1_matched_2m')
                model_dicts = model_dicts_plot
                model_frames = [str(d.get('vector_frame', 'xy')) for d in model_dicts]
                diff_dicts = [records[r] for r in diff_records_plot if r in records]
                matched_diff_dicts = matched_diff_dicts_plot
                model_labels = cached_for_plot.get('model_labels', model_labels) or model_labels
                obs_u_files = list(cached_for_plot.get('obs_u_files', obs_u_files) or [])
                obs_v_files = list(cached_for_plot.get('obs_v_files', obs_v_files) or [])
                model_u_files = list(cached_for_plot.get('model_u_files', model_u_files) or [])
                model_v_files = list(cached_for_plot.get('model_v_files', model_v_files) or [])
                matched_model_labels_used = list(
                    cached_for_plot.get('matched_model_labels_used', matched_model_labels_used)
                )
                if not matched_model_labels_used and matched_diff_dicts:
                    matched_model_labels_used = _get_recipe_model_labels(
                        module, module_vars, len(matched_diff_dicts)
                    )
                logger.info("Using cache-backed %s payload for plotting: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to reload %s cache for plotting (%s). Using in-memory payload.", module, exc)

    # Metric-level multi-model group means (computed after per-model metrics exist).
    base_model_dicts_for_ts = list(model_dicts)
    base_model_labels_for_group = list(model_labels)
    base_matched_diff_dicts_for_ts = list(matched_diff_dicts)
    base_matched_labels_for_group = list(matched_model_labels_used)
    group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=base_model_labels_for_group,
    )
    group_labels: List[str] = []
    group_model_means_for_ts: List[Any] = []
    group_model_stds_for_ts: List[Any] = []
    if group_specs and base_model_dicts_for_ts:
        group_model_means_for_ts, group_model_stds_for_ts, _g_labels = _build_group_mean_std_payloads(
            model_payloads=base_model_dicts_for_ts,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
        )
    if group_specs and model_dicts:
        model_dicts_grouped, diff_dicts_grouped, model_labels_grouped, group_labels = _build_group_mean_payloads(
            model_payloads=model_dicts,
            diff_payloads=diff_dicts,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
        )
        model_dicts = list(model_dicts_grouped)
        diff_dicts = list(diff_dicts_grouped or [])
        model_labels = list(model_labels_grouped)
        if group_labels:
            model_frames = (['xy'] * len(group_labels)) + list(model_frames)
            logger.info(
                "Enabled metric-level group means for %s [%s]: %s",
                module, hemisphere.upper(), ', '.join(group_labels),
            )

    matched_group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=list(base_matched_labels_for_group),
    )
    matched_group_labels: List[str] = []
    matched_group_model_means_for_ts: List[Any] = []
    matched_group_model_stds_for_ts: List[Any] = []
    if matched_group_specs and base_matched_diff_dicts_for_ts:
        base_matched_model_payloads = [
            d.get('sid2_metric') if isinstance(d, dict) else None
            for d in base_matched_diff_dicts_for_ts
        ]
        matched_group_model_means_for_ts, matched_group_model_stds_for_ts, _mg_labels = _build_group_mean_std_payloads(
            model_payloads=base_matched_model_payloads,
            model_labels=list(base_matched_labels_for_group),
            group_specs=matched_group_specs,
        )
    if matched_group_specs and matched_diff_dicts:
        matched_model_payloads = [
            d.get('sid2_metric') if isinstance(d, dict) else None
            for d in matched_diff_dicts
        ]
        _matched_grouped_models, matched_grouped_diffs, matched_grouped_labels, matched_group_labels = _build_group_mean_payloads(
            model_payloads=matched_model_payloads,
            diff_payloads=matched_diff_dicts,
            model_labels=list(matched_model_labels_used),
            group_specs=matched_group_specs,
        )
        matched_diff_dicts = list(matched_grouped_diffs or [])
        matched_model_labels_used = list(matched_grouped_labels)
        if matched_group_labels:
            logger.info(
                "Enabled metric-level matched group means for %s [%s]: %s",
                module, hemisphere.upper(), ', '.join(matched_group_labels),
            )

    if obs_dict is not None and model_dicts:
        logger.info('Generating figures ...')
        fig_dir = Path(output_dir) / module
        fig_dir.mkdir(parents=True, exist_ok=True)

        obs_list = [obs_dict] + ([obs2_dict] if obs2_dict is not None else [])
        all_labels = [obs1_plot_label] + ([obs2_plot_label] if obs2_dict is not None else []) + model_labels
        base_labels_for_ts = [obs1_plot_label] + ([obs2_plot_label] if obs2_dict is not None else []) + base_model_labels_for_group

        with xr.open_dataset(grid_file) as ds:
            _lon, _lat = np.array(ds['lon']), np.array(ds['lat'])

        pf.plot_SID_ts(
            obs_list, base_model_dicts_for_ts, base_labels_for_ts,
            fig_name=str(fig_dir / 'SID_ts.png'),
            line_style=line_styles, color=line_colors,
        )
        pf.plot_SID_ano(
            obs_list, base_model_dicts_for_ts, base_labels_for_ts, module_vars['year_range'],
            fig_name=str(fig_dir / 'SID_ano.png'), hms=hemisphere,
            line_style=line_styles, color=line_colors,
        )
        for month in (3, 9):
            mtag = SIM.SeaIceMetricsBase.month_tag(month)
            pf.plot_SID_key_month_ano(
                obs_list, base_model_dicts_for_ts, base_labels_for_ts, module_vars['year_range'],
                month=month, fig_name=str(fig_dir / f'sidrift_ano_timeseries_{mtag}.png'),
                hms=hemisphere,
                line_style=line_styles, color=line_colors,
                legend_y_gap=key_month_legend_gap,
            )
            if group_labels and group_model_means_for_ts:
                pf.plot_SID_key_month_ano(
                    obs_list,
                    group_model_means_for_ts,
                    [obs1_plot_label] + ([obs2_plot_label] if obs2_dict is not None else []) + group_labels,
                    module_vars['year_range'],
                    month=month,
                    fig_name=str(fig_dir / f'sidrift_ano_timeseries_{mtag}_groupmean.png'),
                    hms=hemisphere,
                    line_style=line_styles, color=line_colors,
                    legend_y_gap=key_month_legend_gap,
                    model_spread_payloads=group_model_stds_for_ts,
                )
        if group_labels and group_model_means_for_ts:
            group_labels_for_ts = [obs1_plot_label] + ([obs2_plot_label] if obs2_dict is not None else []) + group_labels
            pf.plot_SID_ts(
                obs_list, group_model_means_for_ts, group_labels_for_ts,
                fig_name=str(fig_dir / 'SID_ts_groupmean.png'),
                line_style=line_styles, color=line_colors,
                model_spread_payloads=group_model_stds_for_ts,
            )
            pf.plot_SID_ano(
                obs_list, group_model_means_for_ts, group_labels_for_ts, module_vars['year_range'],
                fig_name=str(fig_dir / 'SID_ano_groupmean.png'), hms=hemisphere,
                line_style=line_styles, color=line_colors,
                model_spread_payloads=group_model_stds_for_ts,
            )

        _all_dicts = obs_list + model_dicts
        _mke_data = np.array([np.nanmean(d['MKE_clim'], axis=0) for d in _all_dicts])
        pf.plot_MKE_map(_lon, _lat, _mke_data, all_labels, hemisphere,
                        plot_mode='raw', raw_cmap=mke_raw_cmap, diff_cmap=mke_diff_cmap,
                        fig_name=str(fig_dir / 'MKE_map_raw.png'))
        pf.plot_MKE_map(_lon, _lat, _mke_data, all_labels, hemisphere,
                        plot_mode='diff', raw_cmap=mke_raw_cmap, diff_cmap=mke_diff_cmap,
                        fig_name=str(fig_dir / 'MKE_map_diff.png'))

        if all(('MKE_ano' in d) for d in _all_dicts):
            _mke_std_maps = np.array([np.nanstd(d['MKE_ano'], axis=0) for d in _all_dicts])
            _std_max = max(0.001, round(float(np.nanpercentile(_mke_std_maps[0], 95)) * 1000) / 1000)
            pf.plot_SIC_map(grid_file, _mke_std_maps, all_labels, hemisphere,
                            sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                            sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit=r'$m^2/s^2$',
                            plot_mode='raw', fig_name=str(fig_dir / 'MKE_std_map_raw.png'))
            pf.plot_SIC_map(grid_file, _mke_std_maps, all_labels, hemisphere,
                            sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                            sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit=r'$m^2/s^2$',
                            plot_mode='diff', fig_name=str(fig_dir / 'MKE_std_map_diff.png'))

        if all(('MKE_ano_tr' in d) for d in _all_dicts):
            _mke_tr_maps = np.array([d['MKE_ano_tr'] for d in _all_dicts])
            _mke_tr_pval = np.array([d['MKE_ano_tr_p'] for d in _all_dicts])
            _tr_max = max(0.001, round(float(np.nanpercentile(np.abs(_mke_tr_maps[0]), 95)) * 1000) / 1000)
            pf.plot_trend_map(grid_file, _mke_tr_maps, _mke_tr_pval, all_labels, hemisphere,
                              trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit=r'$m^2/s^2$/decade',
                              plot_mode='raw', fig_name=str(fig_dir / 'MKE_trend_map_raw.png'))
            pf.plot_trend_map(grid_file, _mke_tr_maps, _mke_tr_pval, all_labels, hemisphere,
                              trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit=r'$m^2/s^2$/decade',
                              plot_mode='diff', fig_name=str(fig_dir / 'MKE_trend_map_diff.png'))

        _season_names = ['Spring', 'Summer', 'Autumn', 'Winter']

        def _stack_sid_season_maps(diag_list: List[Dict[str, Any]], key: str) -> Optional[np.ndarray]:
            arrs = [np.asarray(d.get(key, np.array([])), dtype=float) for d in diag_list]
            if not arrs or any(a.ndim != 2 for a in arrs):
                return None
            shp = arrs[0].shape
            if any(a.shape != shp for a in arrs):
                return None
            return np.asarray(arrs, dtype=float)

        for _season in _season_names:
            _diags = [_sid_seasonal_diagnostics(mdict, _season, hemisphere) for mdict in _all_dicts]

            _clim_stack = _stack_sid_season_maps(_diags, 'clim_map')
            if _clim_stack is not None:
                pf.plot_MKE_map(
                    _lon, _lat, _clim_stack, all_labels, hemisphere,
                    plot_mode='raw', raw_cmap=mke_raw_cmap, diff_cmap=mke_diff_cmap,
                    fig_name=str(fig_dir / f'SID_seasonal_clim_{_season}_raw.png'),
                )
                pf.plot_MKE_map(
                    _lon, _lat, _clim_stack, all_labels, hemisphere,
                    plot_mode='diff', raw_cmap=mke_raw_cmap, diff_cmap=mke_diff_cmap,
                    fig_name=str(fig_dir / f'SID_seasonal_clim_{_season}_diff.png'),
                )

            _std_stack = _stack_sid_season_maps(_diags, 'std_map')
            if _std_stack is not None:
                _std_max = max(0.001, round(float(np.nanpercentile(_std_stack[0], 95)) * 1000) / 1000)
                pf.plot_SIC_map(
                    grid_file, _std_stack, all_labels, hemisphere,
                    sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                    sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m/s',
                    plot_mode='raw', fig_name=str(fig_dir / f'SID_seasonal_std_{_season}_raw.png'),
                )
                pf.plot_SIC_map(
                    grid_file, _std_stack, all_labels, hemisphere,
                    sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                    sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m/s',
                    plot_mode='diff', fig_name=str(fig_dir / f'SID_seasonal_std_{_season}_diff.png'),
                )

            _tr_stack = _stack_sid_season_maps(_diags, 'trend_map')
            _trp_stack = _stack_sid_season_maps(_diags, 'trend_p_map')
            if _tr_stack is not None and _trp_stack is not None:
                _tr_max = max(0.001, round(float(np.nanpercentile(np.abs(_tr_stack[0]), 95)) * 1000) / 1000)
                pf.plot_trend_map(
                    grid_file, _tr_stack, _trp_stack, all_labels, hemisphere,
                    trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/s/decade',
                    plot_mode='raw', fig_name=str(fig_dir / f'SID_seasonal_trend_{_season}_raw.png'),
                )
                pf.plot_trend_map(
                    grid_file, _tr_stack, _trp_stack, all_labels, hemisphere,
                    trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/s/decade',
                    plot_mode='diff', fig_name=str(fig_dir / f'SID_seasonal_trend_{_season}_diff.png'),
                )

        _sid_keys = ['MKE_mean_diff', 'MKE_std_diff', 'MKE_ts_ano_std', 'MKE_trend_diff', 'MKE_ts_trend_diff']
        _sid_season_core = [
            'MeanDiff',
            'AnoStdDiff',
            'AnoTrendDiff',
            'StdMapDiff',
            'TrendMapDiff',
            'RMSE',
        ]
        _sid_season_keys: List[str] = []
        for _season in _season_names:
            _sid_season_keys.extend([
                f'{_season}_{_kk}' for _kk in _sid_season_core
            ])

        if diff_dicts:
            _vcorr_data: List[np.ndarray] = []
            _vcorr_labels: List[str] = []
            if isinstance(obs_diff_dict, dict) and isinstance(obs_diff_dict.get('vectcorr'), np.ndarray):
                _vcorr_data.append(np.asarray(obs_diff_dict.get('vectcorr'), dtype=float))
                _vcorr_labels.append(f'{obs2_plot_label} vs {obs1_plot_label}')
            _vcorr_data.extend(np.asarray(d['vectcorr'], dtype=float) for d in diff_dicts)
            _vcorr_labels.extend(model_labels[:len(diff_dicts)])
            if _vcorr_data:
                pf.plot_VectCorr_map(
                    _lon, _lat, np.asarray(_vcorr_data, dtype=float), _vcorr_labels, hemisphere,
                    data_cm=vectcorr_cmap, unit='Vector Correlation',
                    fig_name=str(fig_dir / 'VectCorr_map.png'),
                )

            _sid_heat = np.array([[d.get(k, float('nan')) for k in _sid_keys] for d in diff_dicts])
            with _PLOT_LOCK:
                _fig, _ax = plt.subplots(
                    figsize=(
                        max(8, len(_sid_keys) * 0.8),
                        max(3, 1 + len(diff_dicts)) * max(0.1, heat_main_height_scale),
                    )
                )
                if obs_diff_dict is not None:
                    _obs2_row = np.array([abs(obs_diff_dict.get(k, float('nan'))) for k in _sid_keys])
                    pf.plot_heat_map(
                        _sid_heat, model_labels, _sid_keys, ax=_ax,
                        cbarlabel='Ratio to obs uncertainty',
                        obs_row=_obs2_row, obs_row_label=obs2_plot_label,
                        ratio_vmin=heat_ratio_vmin, ratio_vmax=heat_ratio_vmax,
                        cmap=heatmap_cmap,
                    )
                else:
                    pf.plot_heat_map(_sid_heat, model_labels, _sid_keys, ax=_ax, cbarlabel=r'$m^2/s^2$', cmap=heatmap_cmap)
                _fig.tight_layout()
                pf._save_fig(str(fig_dir / 'heat_map.png'), close=False)
                plt.close(_fig)

            _sid_heat_season = []
            for _dct in diff_dicts:
                _row = []
                for _season in _season_names:
                    md, sd, td, sm, tm, vc, cc, rr = _sid_extract_domain_extended_stats(
                        _dct, ('season', _season), hemisphere
                    )
                    _row.extend([md, sd, td, sm, tm, rr])
                _sid_heat_season.append(_row)
            _sid_heat_season = np.asarray(_sid_heat_season, dtype=float)
            if _sid_heat_season.size > 0:
                with _PLOT_LOCK:
                    _n_metric = len(_sid_season_core)
                    _n_rows = max(1, 1 + len(diff_dicts))
                    _panel_w = max(8.0, _n_metric * 0.76)
                    _panel_h = max(3.2, _n_rows * 0.45 + 1.1)
                    _fig, _axes = plt.subplots(
                        2, 2,
                        figsize=(
                            _panel_w * 2.0 + 1.0,
                            (_panel_h * 2.0 + 0.4) * max(0.5, heat_season_height_scale),
                        ),
                        squeeze=False,
                    )
                    _axes_flat = _axes.ravel()
                    _used_axes = []
                    _first_im = None

                    if obs_diff_dict is not None:
                        _obs2_row = []
                        for _season in _season_names:
                            md, sd, td, sm, tm, vc, cc, rr = _sid_extract_domain_extended_stats(
                                obs_diff_dict, ('season', _season), hemisphere
                            )
                            _obs2_row.extend([md, sd, td, sm, tm, rr])
                        _obs2_row = np.asarray(_obs2_row, dtype=float)
                    else:
                        _obs2_row = None

                    for _ii, _season in enumerate(_season_names):
                        _ax = _axes_flat[_ii]
                        _j0 = _ii * _n_metric
                        _j1 = _j0 + _n_metric
                        _im = pf.plot_heat_map(
                            _sid_heat_season[:, _j0:_j1],
                            model_labels,
                            _sid_season_core,
                            ax=_ax,
                            cbarlabel='',
                            obs_row=(None if _obs2_row is None else _obs2_row[_j0:_j1]),
                            obs_row_label=obs2_plot_label,
                            ratio_vmin=heat_ratio_vmin,
                            ratio_vmax=heat_ratio_vmax,
                            cmap=heatmap_cmap,
                            add_colorbar=False,
                        )
                        _ax.set_title(str(_season), fontsize=17, pad=8)
                        _used_axes.append(_ax)
                        if _first_im is None:
                            _first_im = _im

                    for _jj in range(len(_season_names), len(_axes_flat)):
                        _fig.delaxes(_axes_flat[_jj])

                    if _first_im is not None and _used_axes:
                        _cb = _fig.colorbar(
                            _first_im,
                            ax=_used_axes,
                            orientation='vertical',
                            fraction=max(0.001, heat_season_cbar_fraction),
                            pad=max(0.0, heat_season_cbar_pad),
                            aspect=max(1.0, heat_season_cbar_aspect),
                            extend=pf._resolve_colorbar_extend(
                                _first_im,
                                data=np.abs(_sid_heat_season),
                                vmin=heat_ratio_vmin,
                                vmax=heat_ratio_vmax,
                                extend='auto',
                            ),
                        )
                        _cb.set_label('Ratio to obs uncertainty', fontsize=16)
                        _cb.ax.tick_params(labelsize=15)

                    _fig.subplots_adjust(
                        left=0.06,
                        right=min(0.98, max(0.70, heat_season_right)),
                        top=0.93,
                        bottom=0.10,
                        wspace=max(0.0, heat_season_wspace),
                        hspace=max(0.0, heat_season_hspace),
                    )
                    pf._save_fig(str(fig_dir / 'heat_map_seasonal.png'), close=False)
                    plt.close(_fig)

        _sid_map_data = np.array([[d['u_clim'], d['v_clim']] for d in _all_dicts])
        _kk = int(module_vars.get('quiver_skip', vector_quiver_skip))
        model_rotate_flags = [f == 'xy' for f in model_frames]
        sid_rotate_flags = [True] * len(obs_list) + model_rotate_flags
        if len(sid_rotate_flags) < len(_all_dicts):
            sid_rotate_flags.extend([True] * (len(_all_dicts) - len(sid_rotate_flags)))
        pf.plot_SID_map(
            _lon, _lat, _sid_map_data, all_labels, hemisphere, _kk,
            sid_range=[vector_speed_vmin, vector_speed_vmax], rotate_flags=sid_rotate_flags,
            quiver_scale=vector_quiver_scale,
            quiver_width=vector_quiver_width,
            quiver_min_speed=vector_quiver_min_speed,
            fig_name=str(fig_dir / 'SID_map.png')
        )

        if matched_diff_dicts:
            matched_diffs_for_plot: List[dict] = []
            matched_model_dicts: List[dict] = []
            matched_labels_for_plot: List[str] = []
            matched_obs_dict: Optional[dict] = None
            for idx, d in enumerate(base_matched_diff_dicts_for_ts):
                if not isinstance(d, dict):
                    continue
                sid1_metric = d.get('sid1_metric')
                sid2_metric = d.get('sid2_metric')
                if matched_obs_dict is None and isinstance(sid1_metric, dict):
                    matched_obs_dict = sid1_metric
                if not isinstance(sid2_metric, dict):
                    continue
                matched_diffs_for_plot.append(d)
                matched_model_dicts.append(sid2_metric)
                if idx < len(base_matched_labels_for_group):
                    matched_labels_for_plot.append(base_matched_labels_for_group[idx])
                else:
                    matched_labels_for_plot.append(f'model{idx + 1}')

            if matched_obs_dict is not None and matched_model_dicts:
                if obs_diff_dict_matched is not None:
                    matched_obs_list = [
                        obs_diff_dict_matched.get('sid1_metric'),
                        obs_diff_dict_matched.get('sid2_metric'),
                    ]
                    matched_obs_list = [item for item in matched_obs_list if isinstance(item, dict)]
                    matched_panel_labels = [obs1_plot_label, obs2_plot_label][:len(matched_obs_list)] + matched_labels_for_plot
                else:
                    matched_obs_list = [matched_obs_dict]
                    matched_panel_labels = [obs1_plot_label] + matched_labels_for_plot

                pf.plot_SID_ts(
                    matched_obs_list, matched_model_dicts, matched_panel_labels,
                    fig_name=str(fig_dir / 'SID_ts_matched.png'),
                    line_style=line_styles, color=line_colors,
                )
                pf.plot_SID_ano(
                    matched_obs_list, matched_model_dicts, matched_panel_labels,
                    module_vars['year_range'],
                    fig_name=str(fig_dir / 'SID_ano_matched.png'),
                    hms=hemisphere,
                    line_style=line_styles, color=line_colors,
                )
                for month in (3, 9):
                    mtag = SIM.SeaIceMetricsBase.month_tag(month)
                    pf.plot_SID_key_month_ano(
                        matched_obs_list, matched_model_dicts, matched_panel_labels,
                        module_vars['year_range'],
                        month=month, hms=hemisphere,
                        fig_name=str(fig_dir / f'sidrift_ano_timeseries_{mtag}_matched.png'),
                        line_style=line_styles, color=line_colors,
                        legend_y_gap=key_month_legend_gap,
                    )
                    if matched_group_labels and matched_group_model_means_for_ts:
                        pf.plot_SID_key_month_ano(
                            matched_obs_list, matched_group_model_means_for_ts,
                            [obs1_plot_label] + ([obs2_plot_label] if obs_diff_dict_matched is not None else []) + matched_group_labels,
                            module_vars['year_range'],
                            month=month, hms=hemisphere,
                            fig_name=str(fig_dir / f'sidrift_ano_timeseries_{mtag}_matched_groupmean.png'),
                            line_style=line_styles, color=line_colors,
                            legend_y_gap=key_month_legend_gap,
                            model_spread_payloads=matched_group_model_stds_for_ts,
                        )

                if matched_group_labels and matched_group_model_means_for_ts:
                    matched_group_panel_labels = [obs1_plot_label] + ([obs2_plot_label] if obs_diff_dict_matched is not None else []) + matched_group_labels
                    pf.plot_SID_ts(
                        matched_obs_list, matched_group_model_means_for_ts, matched_group_panel_labels,
                        fig_name=str(fig_dir / 'SID_ts_matched_groupmean.png'),
                        line_style=line_styles, color=line_colors,
                        model_spread_payloads=matched_group_model_stds_for_ts,
                    )
                    pf.plot_SID_ano(
                        matched_obs_list, matched_group_model_means_for_ts, matched_group_panel_labels,
                        module_vars['year_range'],
                        fig_name=str(fig_dir / 'SID_ano_matched_groupmean.png'),
                        hms=hemisphere,
                        line_style=line_styles, color=line_colors,
                        model_spread_payloads=matched_group_model_stds_for_ts,
                    )

                _all_dicts_matched = matched_obs_list + matched_model_dicts
                _mke_data_matched = np.array([np.nanmean(d['MKE_clim'], axis=0) for d in _all_dicts_matched])
                pf.plot_MKE_map(
                    _lon, _lat, _mke_data_matched, matched_panel_labels, hemisphere,
                    plot_mode='raw', raw_cmap=mke_raw_cmap, diff_cmap=mke_diff_cmap,
                    fig_name=str(fig_dir / 'MKE_map_matched_raw.png'),
                )
                pf.plot_MKE_map(
                    _lon, _lat, _mke_data_matched, matched_panel_labels, hemisphere,
                    plot_mode='diff', raw_cmap=mke_raw_cmap, diff_cmap=mke_diff_cmap,
                    fig_name=str(fig_dir / 'MKE_map_matched_diff.png'),
                )

                if all(('MKE_ano' in d) for d in _all_dicts_matched):
                    _mke_std_maps_matched = np.array([np.nanstd(d['MKE_ano'], axis=0) for d in _all_dicts_matched])
                    _std_max_matched = max(0.001, round(float(np.nanpercentile(_mke_std_maps_matched[0], 95)) * 1000) / 1000)
                    pf.plot_SIC_map(
                        grid_file, _mke_std_maps_matched, matched_panel_labels, hemisphere,
                        sic_range=[0, _std_max_matched], diff_range=[-_std_max_matched * 0.5, _std_max_matched * 0.5],
                        sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit=r'$m^2/s^2$',
                        plot_mode='raw', fig_name=str(fig_dir / 'MKE_std_map_matched_raw.png'),
                    )
                    pf.plot_SIC_map(
                        grid_file, _mke_std_maps_matched, matched_panel_labels, hemisphere,
                        sic_range=[0, _std_max_matched], diff_range=[-_std_max_matched * 0.5, _std_max_matched * 0.5],
                        sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit=r'$m^2/s^2$',
                        plot_mode='diff', fig_name=str(fig_dir / 'MKE_std_map_matched_diff.png'),
                    )

                if all(('MKE_ano_tr' in d) for d in _all_dicts_matched):
                    _mke_tr_maps_matched = np.array([d['MKE_ano_tr'] for d in _all_dicts_matched])
                    _mke_tr_pval_matched = np.array([d['MKE_ano_tr_p'] for d in _all_dicts_matched])
                    _tr_max_matched = max(0.001, round(float(np.nanpercentile(np.abs(_mke_tr_maps_matched[0]), 95)) * 1000) / 1000)
                    pf.plot_trend_map(
                        grid_file, _mke_tr_maps_matched, _mke_tr_pval_matched, matched_panel_labels, hemisphere,
                        trend_range=[-_tr_max_matched, _tr_max_matched], cmap=trend_cmap, unit=r'$m^2/s^2$/decade',
                        plot_mode='raw', fig_name=str(fig_dir / 'MKE_trend_map_matched_raw.png'),
                    )
                    pf.plot_trend_map(
                        grid_file, _mke_tr_maps_matched, _mke_tr_pval_matched, matched_panel_labels, hemisphere,
                        trend_range=[-_tr_max_matched, _tr_max_matched], cmap=trend_cmap, unit=r'$m^2/s^2$/decade',
                        plot_mode='diff', fig_name=str(fig_dir / 'MKE_trend_map_matched_diff.png'),
                    )

                for _season in _season_names:
                    _diags_matched = [_sid_seasonal_diagnostics(mdict, _season, hemisphere) for mdict in _all_dicts_matched]

                    _clim_stack_matched = _stack_sid_season_maps(_diags_matched, 'clim_map')
                    if _clim_stack_matched is not None:
                        pf.plot_MKE_map(
                            _lon, _lat, _clim_stack_matched, matched_panel_labels, hemisphere,
                            plot_mode='raw', raw_cmap=mke_raw_cmap, diff_cmap=mke_diff_cmap,
                            fig_name=str(fig_dir / f'SID_seasonal_clim_{_season}_matched_raw.png'),
                        )
                        pf.plot_MKE_map(
                            _lon, _lat, _clim_stack_matched, matched_panel_labels, hemisphere,
                            plot_mode='diff', raw_cmap=mke_raw_cmap, diff_cmap=mke_diff_cmap,
                            fig_name=str(fig_dir / f'SID_seasonal_clim_{_season}_matched_diff.png'),
                        )

                    _std_stack_matched = _stack_sid_season_maps(_diags_matched, 'std_map')
                    if _std_stack_matched is not None:
                        _std_max_matched = max(0.001, round(float(np.nanpercentile(_std_stack_matched[0], 95)) * 1000) / 1000)
                        pf.plot_SIC_map(
                            grid_file, _std_stack_matched, matched_panel_labels, hemisphere,
                            sic_range=[0, _std_max_matched], diff_range=[-_std_max_matched * 0.5, _std_max_matched * 0.5],
                            sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m/s',
                            plot_mode='raw', fig_name=str(fig_dir / f'SID_seasonal_std_{_season}_matched_raw.png'),
                        )
                        pf.plot_SIC_map(
                            grid_file, _std_stack_matched, matched_panel_labels, hemisphere,
                            sic_range=[0, _std_max_matched], diff_range=[-_std_max_matched * 0.5, _std_max_matched * 0.5],
                            sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m/s',
                            plot_mode='diff', fig_name=str(fig_dir / f'SID_seasonal_std_{_season}_matched_diff.png'),
                        )

                    _tr_stack_matched = _stack_sid_season_maps(_diags_matched, 'trend_map')
                    _trp_stack_matched = _stack_sid_season_maps(_diags_matched, 'trend_p_map')
                    if _tr_stack_matched is not None and _trp_stack_matched is not None:
                        _tr_max_matched = max(0.001, round(float(np.nanpercentile(np.abs(_tr_stack_matched[0]), 95)) * 1000) / 1000)
                        pf.plot_trend_map(
                            grid_file, _tr_stack_matched, _trp_stack_matched, matched_panel_labels, hemisphere,
                            trend_range=[-_tr_max_matched, _tr_max_matched], cmap=trend_cmap, unit='m/s/decade',
                            plot_mode='raw', fig_name=str(fig_dir / f'SID_seasonal_trend_{_season}_matched_raw.png'),
                        )
                        pf.plot_trend_map(
                            grid_file, _tr_stack_matched, _trp_stack_matched, matched_panel_labels, hemisphere,
                            trend_range=[-_tr_max_matched, _tr_max_matched], cmap=trend_cmap, unit='m/s/decade',
                            plot_mode='diff', fig_name=str(fig_dir / f'SID_seasonal_trend_{_season}_matched_diff.png'),
                        )

                _vcorr_data_matched: List[np.ndarray] = []
                _vcorr_labels_matched: List[str] = []
                if isinstance(obs_diff_dict_matched, dict) and isinstance(obs_diff_dict_matched.get('vectcorr'), np.ndarray):
                    _vcorr_data_matched.append(np.asarray(obs_diff_dict_matched.get('vectcorr'), dtype=float))
                    _vcorr_labels_matched.append(f'{obs2_plot_label} vs {obs1_plot_label}')
                _vcorr_data_matched.extend(np.asarray(d['vectcorr'], dtype=float) for d in matched_diff_dicts)
                _vcorr_labels_matched.extend(matched_labels_for_plot[:len(matched_diff_dicts)])
                if _vcorr_data_matched:
                    pf.plot_VectCorr_map(
                        _lon, _lat, np.asarray(_vcorr_data_matched, dtype=float), _vcorr_labels_matched, hemisphere,
                        data_cm=vectcorr_cmap, unit='Vector Correlation',
                        fig_name=str(fig_dir / 'VectCorr_map_matched.png'),
                    )

                _sid_heat_matched = np.array([[d.get(k, float('nan')) for k in _sid_keys] for d in matched_diffs_for_plot])
                with _PLOT_LOCK:
                    _fig, _ax = plt.subplots(
                        figsize=(
                            max(8, len(_sid_keys) * 0.8),
                            max(3, 1 + len(matched_diffs_for_plot)) * max(0.1, heat_main_height_scale),
                        )
                    )
                    if obs_diff_dict_matched is not None:
                        _obs2_row = np.array([abs(obs_diff_dict_matched.get(k, float('nan'))) for k in _sid_keys])
                        pf.plot_heat_map(
                            _sid_heat_matched, matched_labels_for_plot, _sid_keys, ax=_ax,
                            cbarlabel='Ratio to obs uncertainty',
                            obs_row=_obs2_row, obs_row_label=obs2_plot_label,
                            ratio_vmin=heat_ratio_vmin, ratio_vmax=heat_ratio_vmax,
                            cmap=heatmap_cmap,
                        )
                    else:
                        pf.plot_heat_map(
                            _sid_heat_matched, matched_labels_for_plot, _sid_keys,
                            ax=_ax, cbarlabel=r'$m^2/s^2$', cmap=heatmap_cmap,
                        )
                    _fig.tight_layout()
                    pf._save_fig(str(fig_dir / 'heat_map_matched.png'), close=False)
                    plt.close(_fig)

                _sid_heat_season_matched = []
                for _dct in matched_diffs_for_plot:
                    _row = []
                    for _season in _season_names:
                        md, sd, td, sm, tm, vc, cc, rr = _sid_extract_domain_extended_stats(
                            _dct, ('season', _season), hemisphere
                        )
                        _row.extend([md, sd, td, sm, tm, rr])
                    _sid_heat_season_matched.append(_row)
                _sid_heat_season_matched = np.asarray(_sid_heat_season_matched, dtype=float)
                if _sid_heat_season_matched.size > 0:
                    with _PLOT_LOCK:
                        _n_metric = len(_sid_season_core)
                        _n_rows = max(1, 1 + len(matched_diffs_for_plot))
                        _panel_w = max(8.0, _n_metric * 0.76)
                        _panel_h = max(3.2, _n_rows * 0.45 + 1.1)
                        _fig, _axes = plt.subplots(
                            2, 2,
                            figsize=(
                                _panel_w * 2.0 + 1.0,
                                (_panel_h * 2.0 + 0.4) * max(0.5, heat_season_height_scale),
                            ),
                            squeeze=False,
                        )
                        _axes_flat = _axes.ravel()
                        _used_axes = []
                        _first_im = None

                        if obs_diff_dict_matched is not None:
                            _obs2_row = []
                            for _season in _season_names:
                                md, sd, td, sm, tm, vc, cc, rr = _sid_extract_domain_extended_stats(
                                    obs_diff_dict_matched, ('season', _season), hemisphere
                                )
                                _obs2_row.extend([md, sd, td, sm, tm, rr])
                            _obs2_row = np.asarray(_obs2_row, dtype=float)
                        else:
                            _obs2_row = None

                        for _ii, _season in enumerate(_season_names):
                            _ax = _axes_flat[_ii]
                            _j0 = _ii * _n_metric
                            _j1 = _j0 + _n_metric
                            _im = pf.plot_heat_map(
                                _sid_heat_season_matched[:, _j0:_j1],
                                matched_labels_for_plot,
                                _sid_season_core,
                                ax=_ax,
                                cbarlabel='',
                                obs_row=(None if _obs2_row is None else _obs2_row[_j0:_j1]),
                                obs_row_label=obs2_plot_label,
                                ratio_vmin=heat_ratio_vmin,
                                ratio_vmax=heat_ratio_vmax,
                                cmap=heatmap_cmap,
                                add_colorbar=False,
                            )
                            _ax.set_title(str(_season), fontsize=17, pad=8)
                            _used_axes.append(_ax)
                            if _first_im is None:
                                _first_im = _im

                        for _jj in range(len(_season_names), len(_axes_flat)):
                            _fig.delaxes(_axes_flat[_jj])

                        if _first_im is not None and _used_axes:
                            _cb = _fig.colorbar(
                                _first_im,
                                ax=_used_axes,
                                orientation='vertical',
                                fraction=max(0.001, heat_season_cbar_fraction),
                                pad=max(0.0, heat_season_cbar_pad),
                                aspect=max(1.0, heat_season_cbar_aspect),
                                extend=pf._resolve_colorbar_extend(
                                    _first_im,
                                    data=np.abs(_sid_heat_season_matched),
                                    vmin=heat_ratio_vmin,
                                    vmax=heat_ratio_vmax,
                                    extend='auto',
                                ),
                            )
                            _cb.set_label('Ratio to obs uncertainty', fontsize=16)
                            _cb.ax.tick_params(labelsize=15)

                        _fig.subplots_adjust(
                            left=0.06,
                            right=min(0.98, max(0.70, heat_season_right)),
                            top=0.93,
                            bottom=0.10,
                            wspace=max(0.0, heat_season_wspace),
                            hspace=max(0.0, heat_season_hspace),
                        )
                        pf._save_fig(str(fig_dir / 'heat_map_seasonal_matched.png'), close=False)
                        plt.close(_fig)

                _sid_map_data_matched = np.array([[d['u_clim'], d['v_clim']] for d in _all_dicts_matched])
                pf.plot_SID_map(
                    _lon, _lat, _sid_map_data_matched, matched_panel_labels, hemisphere, _kk,
                    sid_range=[vector_speed_vmin, vector_speed_vmax],
                    rotate_flags=[True] * len(_all_dicts_matched),
                    quiver_scale=vector_quiver_scale,
                    quiver_width=vector_quiver_width,
                    quiver_min_speed=vector_quiver_min_speed,
                    fig_name=str(fig_dir / 'SID_map_matched.png')
                )
        try:
            pf.plot_sic_region_map(
                grid_nc_file=grid_file,
                hms=hemisphere,
                fig_name=str(fig_dir / 'SeaIceRegion_map.png'),
            )
        except Exception as exc:
            logger.warning("Failed to generate sea-ice region map (%s).", exc)

    logger.info('%s evaluation completed.', module)
    if not diff_dicts and not matched_diff_dicts:
        return None

    base_raw_table = _build_sid_month_period_raw_table(
        obs1_metric=obs_dict,
        obs2_metric=obs2_dict,
        model_metrics=model_dicts,
        model_labels=model_labels,
        hemisphere=hemisphere,
        obs1_label=f'{obs1_plot_label} (baseline)',
        obs2_label=obs2_plot_label,
    )
    base_diff_table = _build_sid_month_period_table(
        obs2_diff=obs_diff_dict,
        diff_dicts=diff_dicts,
        model_labels=model_labels,
        hemisphere=hemisphere,
        obs1_label=f'{obs1_plot_label} (baseline)',
        obs2_label=obs2_plot_label,
    )
    all_payload: Dict[str, Any] = {
        'type': 'dual_table',
        'sections': [
            {
                'id': 'base',
                'title': 'Original Coverage',
                'type': 'dual_table',
                'sections': [
                    {'id': 'raw', 'title': 'Raw Values', **base_raw_table},
                    {'id': 'diff', 'title': 'Differences', **base_diff_table},
                ],
            },
        ],
    }
    sid_extra_tables: List[Dict[str, Any]] = []

    def _fmt_sid_summary_scientific(v: float, sig_digits: int = 3) -> str:
        if not np.isfinite(v):
            return '--'
        if float(v) == 0.0:
            return '0'
        digits = max(2, int(sig_digits))
        txt = f'{float(v):.{digits - 1}e}'
        mantissa, exp = txt.split('e')
        mantissa = mantissa.rstrip('0').rstrip('.')
        return f'{mantissa}e{int(exp):+d}'

    def _sid_ano_summary_rows(metrics_list: List[dict], labels_list: List[str]) -> List[List[str]]:
        rows: List[List[str]] = []
        for label, metric_dict in zip(labels_list, metrics_list):
            if not isinstance(metric_dict, dict):
                continue
            series = np.asarray(metric_dict.get('MKE_ts_ano', np.array([])), dtype=float).reshape(-1)
            valid = np.isfinite(series)
            if int(np.sum(valid)) < 2:
                std_val = np.nan
                trend_val = np.nan
                dstd_val = np.nan
            else:
                series_v = series[valid]
                std_val = float(np.nanstd(series_v))
                x_fit = np.arange(series.size, dtype=float)
                coef = np.polyfit(x_fit[valid], series[valid], 1)
                slope = float(coef[0])
                intercept = float(coef[1])
                trend_val = float(slope * 12.0 * 10.0)
                detrended = series_v - (slope * x_fit[valid] + intercept)
                dstd_val = float(np.nanstd(detrended))

            rows.append([
                str(label),
                _fmt_sid_summary_scientific(std_val),
                _fmt_sid_summary_scientific(trend_val),
                _fmt_sid_summary_scientific(dstd_val),
            ])
        return rows

    _base_ano_rows = _sid_ano_summary_rows(
        [obs_dict] + ([obs2_dict] if isinstance(obs2_dict, dict) else []) + list(model_dicts),
        [obs1_plot_label] + ([obs2_plot_label] if isinstance(obs2_dict, dict) else []) + list(model_labels),
    )
    if _base_ano_rows:
        sid_extra_tables.append({
            'type': 'basic_table',
            'coverage_mode': 'base',
            'title': 'SID Anomaly Summary (Original Coverage)',
            'headers': ['Dataset', 'STD', 'Trend (per decade)', 'Detrended STD'],
            'units': ['', 'm²/s²', 'm²/s²/decade', 'm²/s²'],
            'rows': _base_ano_rows,
        })

    if matched_diff_dicts:
        matched_obs1_metric = None
        matched_model_metrics: List[dict] = []
        matched_model_labels: List[str] = []
        for ii, diff_payload in enumerate(matched_diff_dicts):
            if not isinstance(diff_payload, dict):
                continue
            if matched_obs1_metric is None and isinstance(diff_payload.get('sid1_metric'), dict):
                matched_obs1_metric = diff_payload.get('sid1_metric')
            sid2_metric = diff_payload.get('sid2_metric')
            if isinstance(sid2_metric, dict):
                matched_model_metrics.append(sid2_metric)
                matched_model_labels.append(
                    matched_model_labels_used[ii]
                    if ii < len(matched_model_labels_used)
                    else f'model{ii + 1}'
                )
        if matched_obs1_metric is None and isinstance(obs_diff_dict_matched, dict):
            sid1_metric = obs_diff_dict_matched.get('sid1_metric')
            if isinstance(sid1_metric, dict):
                matched_obs1_metric = sid1_metric
        matched_obs2_metric = None
        if isinstance(obs_diff_dict_matched, dict):
            sid2_metric = obs_diff_dict_matched.get('sid2_metric')
            if isinstance(sid2_metric, dict):
                matched_obs2_metric = sid2_metric

        matched_raw_table = _build_sid_month_period_raw_table(
            obs1_metric=matched_obs1_metric,
            obs2_metric=matched_obs2_metric,
            model_metrics=matched_model_metrics,
            model_labels=matched_model_labels,
            hemisphere=hemisphere,
            obs1_label=f'{obs1_plot_label} (baseline)',
            obs2_label=obs2_plot_label,
        )
        matched_diff_table = _build_sid_month_period_table(
            obs2_diff=obs_diff_dict_matched,
            diff_dicts=matched_diff_dicts,
            model_labels=matched_model_labels_used,
            hemisphere=hemisphere,
            obs1_label=f'{obs1_plot_label} (baseline)',
            obs2_label=obs2_plot_label,
        )
        all_payload['sections'].append(
            {
                'id': 'matched',
                'title': 'Obs-Matched Coverage',
                'type': 'dual_table',
                'sections': [
                    {'id': 'raw', 'title': 'Raw Values', **matched_raw_table},
                    {'id': 'diff', 'title': 'Differences', **matched_diff_table},
                ],
            }
        )
        _matched_ano_rows = _sid_ano_summary_rows(
            [matched_obs1_metric] + ([matched_obs2_metric] if isinstance(matched_obs2_metric, dict) else []) + list(matched_model_metrics),
            [obs1_plot_label] + ([obs2_plot_label] if isinstance(matched_obs2_metric, dict) else []) + list(matched_model_labels),
        )
        if _matched_ano_rows:
            sid_extra_tables.append({
                'type': 'basic_table',
                'coverage_mode': 'matched',
                'title': 'SID Anomaly Summary (Obs-Matched Coverage)',
                'headers': ['Dataset', 'STD', 'Trend (per decade)', 'Detrended STD'],
                'units': ['', 'm²/s²', 'm²/s²/decade', 'm²/s²'],
                'rows': _matched_ano_rows,
            })

    regional_tables: Dict[str, Any] = {'All': all_payload}
    # Keep SIdrift regional scalar tables available in report even when
    # figure payload is loaded from cache.
    skip_regional_table_expansion = False

    if (not skip_regional_table_expansion) and (
        (not obs_u_files) or (not model_u_files) or (len(model_u_files) != len(model_v_files))
    ):
        try:
            preprocessor = PP.DataPreprocessor(case_name, module, hemisphere=hemisphere)
            preprocessor.gen_eval_grid()
            file_groups = recipe.validate_module(module)
            obs_files = preprocessor.prep_obs(
                frequency='monthly',
                output_dir=data_dir,
                jobs=jobs,
            )
            obs_u_files = obs_v_files = obs_files

            if file_groups:
                if len(file_groups) % 2 != 0:
                    raise ValueError(
                        f"SIdrift model file groups should be paired u/v, got {len(file_groups)} groups."
                    )
                n_models = len(file_groups) // 2
                model_u_groups = file_groups[:n_models]
                model_v_groups = file_groups[n_models:]
            else:
                model_u_groups = []
                model_v_groups = []

            model_u_files, model_v_files, _ = preprocessor.prep_sidrift_models(
                u_file_groups=model_u_groups,
                v_file_groups=model_v_groups,
                model_direction=model_direction,
                model_angle=module_vars.get('model_angle', []),
                frequency='monthly',
                output_dir=data_dir,
                overwrite=False,
                prefer_metadata=True,
                jobs=jobs,
            )
        except Exception as exc:
            logger.warning("Failed to recover processed file list for regional %s tables (%s).", module, exc)
            obs_u_files, obs_v_files, model_u_files, model_v_files = [], [], [], []

    if (not skip_regional_table_expansion) and obs_u_files and model_u_files and len(model_u_files) == len(model_v_files):
        logger.info("Computing regional %s scalar tables ...", module)
        metric = _get_metric(module_vars)
        sid_metrics_region = SIM.SIDMetrics(
            grid_file=grid_file,
            hemisphere=hemisphere,
            time_sta=year_sta,
            time_end=year_end,
            metric=metric,
            projection=module_vars.get('projection', 'stere'),
        )
        obs_direction = 'xy'
        obs_u_path = os.path.join(data_dir, obs_u_files[0])
        obs_v_path = os.path.join(data_dir, obs_v_files[0])
        obs2_u_path = os.path.join(data_dir, obs_u_files[1]) if len(obs_u_files) > 1 else None
        obs2_v_path = os.path.join(data_dir, obs_v_files[1]) if len(obs_v_files) > 1 else None

        model_labels_full = _get_recipe_model_labels(module, module_vars, len(model_u_files))
        sectors = utils.get_hemisphere_sectors(hemisphere, include_all=False)

        for sector in sectors:
            try:
                obs2_diff_sec = None
                obs2_diff_sec_matched = None
                if obs2_u_path is not None and obs2_v_path is not None:
                    obs2_diff_sec = sid_metrics_region.SID_2M_metrics(
                        obs_u_path, u_key, obs_v_path, v_key,
                        obs2_u_path, u_key, obs2_v_path, v_key,
                        model_direction1=obs_direction,
                        model_direction2=obs_direction,
                        sector=sector,
                    )
                    obs2_diff_sec_matched = sid_metrics_region.SID_2M_metrics(
                        obs_u_path, u_key, obs_v_path, v_key,
                        obs2_u_path, u_key, obs2_v_path, v_key,
                        model_direction1=obs_direction,
                        model_direction2=obs_direction,
                        strict_obs_match=True,
                        obs_match_u_file=obs2_u_path,
                        obs_match_u_key=u_key,
                        obs_match_v_file=obs2_v_path,
                        obs_match_v_key=v_key,
                        obs_match_direction=obs_direction,
                        sector=sector,
                    )

                diff_sec: List[dict] = []
                diff_labels: List[str] = []
                matched_sec: List[dict] = []
                matched_labels: List[str] = []
                for ii, (mf_u, mf_v) in enumerate(zip(model_u_files, model_v_files)):
                    model_u_path = os.path.join(data_dir, mf_u)
                    model_v_path = os.path.join(data_dir, mf_v)
                    model_label = model_labels_full[ii] if ii < len(model_labels_full) else f'model{ii + 1}'
                    d_sec = sid_metrics_region.SID_2M_metrics(
                        obs_u_path, u_key, obs_v_path, v_key,
                        model_u_path, model_u_key, model_v_path, model_v_key,
                        model_direction1=obs_direction,
                        model_direction2='xy',
                        sector=sector,
                    )
                    if d_sec is not None:
                        diff_sec.append(d_sec)
                        diff_labels.append(model_label)
                    d_matched_sec = sid_metrics_region.SID_2M_metrics(
                        obs_u_path, u_key, obs_v_path, v_key,
                        model_u_path, model_u_key, model_v_path, model_v_key,
                        model_direction1=obs_direction,
                        model_direction2='xy',
                        strict_obs_match=True,
                        obs_match_u_file=obs2_u_path,
                        obs_match_u_key=u_key,
                        obs_match_v_file=obs2_v_path,
                        obs_match_v_key=v_key,
                        obs_match_direction=obs_direction,
                        sector=sector,
                    )
                    if d_matched_sec is not None:
                        matched_sec.append(d_matched_sec)
                        matched_labels.append(model_label)

                if not diff_sec:
                    continue

                base_obs1_metric = None
                if diff_sec and isinstance(diff_sec[0], dict):
                    sid1_metric = diff_sec[0].get('sid1_metric')
                    if isinstance(sid1_metric, dict):
                        base_obs1_metric = sid1_metric
                base_obs2_metric = None
                if isinstance(obs2_diff_sec, dict):
                    sid2_metric = obs2_diff_sec.get('sid2_metric')
                    if isinstance(sid2_metric, dict):
                        base_obs2_metric = sid2_metric
                base_model_metrics = [
                    d.get('sid2_metric') for d in diff_sec
                    if isinstance(d, dict) and isinstance(d.get('sid2_metric'), dict)
                ]
                base_raw_sec_tbl = _build_sid_month_period_raw_table(
                    obs1_metric=base_obs1_metric,
                    obs2_metric=base_obs2_metric,
                    model_metrics=base_model_metrics,
                    model_labels=diff_labels,
                    hemisphere=hemisphere,
                    obs1_label=f'{obs1_plot_label} (baseline)',
                    obs2_label=obs2_plot_label,
                )
                base_diff_sec_tbl = _build_sid_month_period_table(
                    obs2_diff=obs2_diff_sec,
                    diff_dicts=diff_sec,
                    model_labels=diff_labels,
                    hemisphere=hemisphere,
                    obs1_label=f'{obs1_plot_label} (baseline)',
                    obs2_label=obs2_plot_label,
                )
                sec_payload: Dict[str, Any] = {
                    'type': 'dual_table',
                    'sections': [
                        {
                            'id': 'base',
                            'title': 'Original Coverage',
                            'type': 'dual_table',
                            'sections': [
                                {'id': 'raw', 'title': 'Raw Values', **base_raw_sec_tbl},
                                {'id': 'diff', 'title': 'Differences', **base_diff_sec_tbl},
                            ],
                        },
                    ],
                }
                if matched_sec:
                    matched_obs1_metric = None
                    if matched_sec and isinstance(matched_sec[0], dict):
                        sid1_metric = matched_sec[0].get('sid1_metric')
                        if isinstance(sid1_metric, dict):
                            matched_obs1_metric = sid1_metric
                    matched_obs2_metric = None
                    if isinstance(obs2_diff_sec_matched, dict):
                        sid2_metric = obs2_diff_sec_matched.get('sid2_metric')
                        if isinstance(sid2_metric, dict):
                            matched_obs2_metric = sid2_metric
                    matched_model_metrics = [
                        d.get('sid2_metric') for d in matched_sec
                        if isinstance(d, dict) and isinstance(d.get('sid2_metric'), dict)
                    ]
                    matched_raw_sec_tbl = _build_sid_month_period_raw_table(
                        obs1_metric=matched_obs1_metric,
                        obs2_metric=matched_obs2_metric,
                        model_metrics=matched_model_metrics,
                        model_labels=matched_labels,
                        hemisphere=hemisphere,
                        obs1_label=f'{obs1_plot_label} (baseline)',
                        obs2_label=obs2_plot_label,
                    )
                    matched_diff_sec_tbl = _build_sid_month_period_table(
                        obs2_diff=obs2_diff_sec_matched,
                        diff_dicts=matched_sec,
                        model_labels=matched_labels,
                        hemisphere=hemisphere,
                        obs1_label=f'{obs1_plot_label} (baseline)',
                        obs2_label=obs2_plot_label,
                    )
                    sec_payload['sections'].append(
                        {
                            'id': 'matched',
                            'title': 'Obs-Matched Coverage',
                            'type': 'dual_table',
                            'sections': [
                                {'id': 'raw', 'title': 'Raw Values', **matched_raw_sec_tbl},
                                {'id': 'diff', 'title': 'Differences', **matched_diff_sec_tbl},
                            ],
                        }
                    )
                regional_tables[sector] = sec_payload
            except Exception as exc:
                logger.warning("Skipping %s regional table for sector '%s' (%s).", module, sector, exc)

    payload = _build_region_table_payload(
        hemisphere=hemisphere,
        regional_tables=regional_tables,
        payload_type='region_dual_table',
    )
    if sid_extra_tables:
        payload['extra_tables'] = sid_extra_tables
    return payload

__all__ = ["eval_sidrift"]
