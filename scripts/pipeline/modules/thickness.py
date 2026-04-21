# -*- coding: utf-8 -*-
"""Pipeline module evaluation helpers."""

from scripts.pipeline import app as _app

# Reuse runtime namespace (imports/constants/helpers) initialized in app.py.
globals().update({k: v for k, v in _app.__dict__.items() if k not in globals()})

from scripts.pipeline.modules.common import (
    _apply_plot_runtime_options,
    _check_outputs_exist,
    _load_plot_options,
    _plot_options_get_module,
)
from scripts.config import DAYS_PER_MONTH

def _compute_thickness_obs_model_metrics(
    thick_metrics, data_dir: str, obs_files: List[str], model_files: List[str],
    model_labels: list, obs_key: str, model_key: str, module_name: str,
    jobs: int = 1,
    stage_dir: Optional[Path] = None,
    sector: str = 'All',
):
    """Compute 1M/2M thickness metrics for obs and models (shared by SIthick & SNdepth).

    In parallel mode, each model metric payload is staged to disk first and then
    loaded back in order, which avoids building large in-memory result bundles.
    """
    obs_1m = None
    obs2_1m = None
    obs2_diff = None
    obs2_diff_matched = None
    diff_dicts = []
    model_labels_used = []
    matched_diff_dicts = []
    matched_model_labels_used = []

    if obs_files:
        obs_path = os.path.join(data_dir, obs_files[0])
        obs2_path = None
        with xr.open_dataset(obs_path) as ds:
            thickness = np.array(ds[obs_key])
            month_list = [int(t.dt.month) for t in ds['time']]
        obs_1m = thick_metrics.Thickness_1M_metrics(thickness, month_list)

        if len(obs_files) >= 2:
            obs2_path = os.path.join(data_dir, obs_files[1])
            with xr.open_dataset(obs2_path) as ds2:
                thickness2 = np.array(ds2[obs_key])
                month_list2 = [int(t.dt.month) for t in ds2['time']]
            obs2_1m = thick_metrics.Thickness_1M_metrics(thickness2, month_list2)
            try:
                obs2_diff = thick_metrics.Thickness_2M_metrics(
                    obs_path, obs_key, obs2_path, obs_key, sector=sector
                )
            except Exception as exc:
                logger.warning('%s obs2 diff metrics skipped (%s).', module_name, exc)
            try:
                obs2_diff_matched = thick_metrics.Thickness_2M_metrics(
                    obs_path, obs_key, obs2_path, obs_key,
                    strict_obs_match=True,
                    obs_match_file=obs2_path,
                    obs_match_key=obs_key,
                    sector=sector,
                )
            except Exception as exc:
                logger.warning('%s obs2 matched diff metrics skipped (%s).', module_name, exc)

        model_tasks = [
            (idx, mf, model_labels[idx] if idx < len(model_labels) else f'model{idx + 1}')
            for idx, mf in enumerate(model_files)
        ]
        if jobs > 1 and model_tasks:
            # Stage per-model outputs to disk first to avoid retaining all worker outputs in memory.
            safe_stage_dir = stage_dir or (Path(data_dir) / '_staging' / module_name)
            safe_stage_dir.mkdir(parents=True, exist_ok=True)

            def _worker(task: Tuple[int, str, str]) -> Dict[str, Any]:
                idx, mf, mlab = task
                diff_payload = None
                diff_matched_payload = None
                try:
                    diff_payload = thick_metrics.Thickness_2M_metrics(
                        obs_path, obs_key,
                        os.path.join(data_dir, mf), model_key,
                        sector=sector,
                    )
                except Exception as exc:
                    logger.warning('%s model metrics skipped for %s (%s).', module_name, mf, exc)
                try:
                    diff_matched_payload = thick_metrics.Thickness_2M_metrics(
                        obs_path, obs_key,
                        os.path.join(data_dir, mf), model_key,
                        strict_obs_match=True,
                        obs_match_file=obs2_path,
                        obs_match_key=obs_key,
                        sector=sector,
                    )
                except Exception as exc:
                    logger.warning('%s matched model metrics skipped for %s (%s).', module_name, mf, exc)
                if diff_payload is None and diff_matched_payload is None:
                    return {'skip': True, 'label': mlab}

                payload_file = safe_stage_dir / f'model_{idx:04d}.pkl'
                _save_pickle_atomic(
                    payload_file,
                    {'diff': diff_payload, 'diff_matched': diff_matched_payload, 'label': mlab},
                )
                return {'skip': False, 'label': mlab, 'payload_file': str(payload_file)}

            stage_refs = _parallel_map_ordered(
                items=model_tasks,
                worker_fn=_worker,
                max_workers=jobs,
                task_label=f'{module_name}/thickness-model-metrics',
            )
            for item in stage_refs:
                if item.get('skip'):
                    continue
                payload = _load_pickle(Path(str(item['payload_file'])))
                if payload.get('diff') is not None:
                    diff_dicts.append(payload['diff'])
                    model_labels_used.append(payload['label'])
                if payload.get('diff_matched') is not None:
                    matched_diff_dicts.append(payload['diff_matched'])
                    matched_model_labels_used.append(payload['label'])
        else:
            for _, mf, mlab in model_tasks:
                diff_payload = None
                diff_matched_payload = None
                try:
                    diff_payload = thick_metrics.Thickness_2M_metrics(
                        obs_path, obs_key,
                        os.path.join(data_dir, mf), model_key,
                        sector=sector,
                    )
                except Exception as exc:
                    logger.warning('%s model metrics skipped for %s (%s).', module_name, mf, exc)
                try:
                    diff_matched_payload = thick_metrics.Thickness_2M_metrics(
                        obs_path, obs_key,
                        os.path.join(data_dir, mf), model_key,
                        strict_obs_match=True,
                        obs_match_file=obs2_path,
                        obs_match_key=obs_key,
                        sector=sector,
                    )
                except Exception as exc:
                    logger.warning('%s matched model metrics skipped for %s (%s).', module_name, mf, exc)
                if diff_payload is not None:
                    diff_dicts.append(diff_payload)
                    model_labels_used.append(mlab)
                if diff_matched_payload is not None:
                    matched_diff_dicts.append(diff_matched_payload)
                    matched_model_labels_used.append(mlab)

    return (
        obs_1m,
        obs2_1m,
        obs2_diff,
        obs2_diff_matched,
        diff_dicts,
        model_labels_used,
        matched_diff_dicts,
        matched_model_labels_used,
    )

def _build_thickness_label_sets(obs_1m, obs2_1m, obs2_diff, diff_dicts, model_labels_used,
                                obs_labels: Optional[List[str]] = None):
    """Build label/dict lists for thickness-family figure generation."""
    obs1_label = (
        str(obs_labels[0]).strip()
        if isinstance(obs_labels, list) and len(obs_labels) >= 1 and str(obs_labels[0]).strip()
        else 'obs1'
    )
    obs2_label = (
        str(obs_labels[1]).strip()
        if isinstance(obs_labels, list) and len(obs_labels) >= 2 and str(obs_labels[1]).strip()
        else 'obs2'
    )
    obs_pair = ({'thick1_metric': obs_1m, 'thick2_metric': obs2_1m}
                if (obs_1m is not None and obs2_1m is not None) else None)
    all_diff_dicts_map = ([obs_pair] if obs_pair else []) + diff_dicts
    all_labels_map = ([obs1_label, obs2_label] if obs_pair else [obs1_label]) + model_labels_used
    all_diff_dicts = ([obs2_diff] if obs2_diff else []) + diff_dicts
    all_labels = ([obs1_label, obs2_label] if obs2_diff else [obs1_label]) + model_labels_used
    model_dicts = [d['thick2_metric'] for d in diff_dicts]
    return obs_pair, all_diff_dicts_map, all_labels_map, all_diff_dicts, all_labels, model_dicts


def _fmt_thickness_month_stat(value: float) -> str:
    return _format_min_sig(value, min_sig=3)


def _thickness_month_candidates(month: int) -> List[Tuple[str, str, int]]:
    """Return candidate (month_tag, month_label, month_number) tuples for one display month."""
    month = int(month)
    candidates = [
        (
            SIM.SeaIceMetricsBase.month_tag(month),
            SIM.SeaIceMetricsBase.month_label(month),
            month,
        )
    ]
    # Historical SH payloads use February key-month fields; allow a fallback so
    # March tabs can still show key-month diagnostics when March-specific keys
    # are not present.
    if month == 3:
        candidates.append(('feb', 'Feb', 2))
    return candidates


def _season_months(hemisphere: str) -> Dict[str, Tuple[int, int, int]]:
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


def _thickness_domain_specs(hemisphere: str) -> List[Tuple[str, Tuple[str, Optional[Union[int, str]]]]]:
    """Return ordered domain specs for scalar table tabs."""
    specs: List[Tuple[str, Tuple[str, Optional[Union[int, str]]]]] = [('Annual', ('annual', None))]
    for season_name in ('Spring', 'Summer', 'Autumn', 'Winter'):
        specs.append((season_name, ('season', season_name)))
    specs.extend([
        ('March', ('month', 3)),
        ('September', ('month', 9)),
    ])
    return specs


def _thickness_yearmon_series(metric_dict: Dict[str, Any], n_time: int) -> List[Tuple[int, int]]:
    """Infer (year, month) per timestep for one metric payload."""
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

    uni_mon = np.asarray(metric_dict.get('uni_mon', np.arange(1, 13)), dtype=int).reshape(-1)
    if uni_mon.size > 0:
        rep = int(np.ceil(float(n_time) / float(uni_mon.size)))
        mons = np.tile(uni_mon, rep)[:n_time]
    else:
        mons = np.tile(np.arange(1, 13, dtype=int), int(np.ceil(n_time / 12)))[:n_time]
    years = np.arange(n_time, dtype=int) // 12
    return [(int(years[ii]), int(mons[ii])) for ii in range(n_time)]


def _thickness_seasonal_diagnostics(
    metric_dict: Optional[dict], season: str, hemisphere: str
) -> Dict[str, Any]:
    """Build seasonal diagnostics (map climatology/std/trend + volume series)."""
    out: Dict[str, Any] = {
        'season': season,
        'months': tuple(),
        'clim_map': np.array([]),
        'std_map': np.array([]),
        'trend_map': np.array([]),
        'trend_p_map': np.array([]),
        'vol_years': np.array([], dtype=int),
        'vol_ano': np.array([], dtype=float),
        'vol_clim': np.nan,
    }
    if not isinstance(metric_dict, dict):
        return out

    months = _season_months(hemisphere).get(str(season), tuple())
    if not months:
        return out
    out['months'] = tuple(int(m) for m in months)

    thick_clim = np.asarray(metric_dict.get('thick_clim', np.array([])), dtype=float)
    uni_mon = np.asarray(metric_dict.get('uni_mon', np.arange(1, 13)), dtype=int).reshape(-1)
    if thick_clim.ndim == 3 and thick_clim.shape[0] == uni_mon.size and uni_mon.size > 0:
        maps = []
        weights = []
        for mm in months:
            idx = np.where(uni_mon == int(mm))[0]
            if idx.size <= 0:
                continue
            maps.append(np.asarray(thick_clim[int(idx[0])], dtype=float))
            weights.append(float(DAYS_PER_MONTH[int(mm) - 1]))
        if maps:
            maps_arr = np.asarray(maps, dtype=float)
            w = np.asarray(weights, dtype=float)
            if np.sum(w) > 0:
                out['clim_map'] = np.nansum(maps_arr * w[:, None, None], axis=0) / np.sum(w)

    vol_clim = np.asarray(metric_dict.get('Vol_clim', np.array([])), dtype=float).reshape(-1)
    if vol_clim.size == uni_mon.size and vol_clim.size > 0:
        vals = []
        weights = []
        for mm in months:
            idx = np.where(uni_mon == int(mm))[0]
            if idx.size <= 0:
                continue
            vv = float(vol_clim[int(idx[0])])
            if np.isfinite(vv):
                vals.append(vv)
                weights.append(float(DAYS_PER_MONTH[int(mm) - 1]))
        if vals:
            w = np.asarray(weights, dtype=float)
            v = np.asarray(vals, dtype=float)
            if np.sum(w) > 0:
                out['vol_clim'] = float(np.nansum(v * w) / np.sum(w))

    thick_ano = np.asarray(metric_dict.get('thick_ano', np.array([])), dtype=float)
    vol_ano = np.asarray(metric_dict.get('Vol_ano', np.array([])), dtype=float).reshape(-1)
    if thick_ano.ndim != 3 or vol_ano.ndim != 1 or thick_ano.shape[0] != vol_ano.size or vol_ano.size <= 0:
        return out

    ym = _thickness_yearmon_series(metric_dict, int(vol_ano.size))
    cross_year = (12 in months and 1 in months)
    grouped: Dict[int, List[int]] = {}
    for ii, (yy, mm) in enumerate(ym):
        if int(mm) not in months:
            continue
        sy = int(yy) + 1 if (cross_year and int(mm) == 12) else int(yy)
        grouped.setdefault(sy, []).append(ii)

    season_years = sorted(grouped.keys())
    if not season_years:
        return out

    map_list = []
    vol_list = []
    year_list = []
    for sy in season_years:
        idx = grouped.get(sy, [])
        if len(idx) < 2:
            continue
        map_list.append(np.nanmean(thick_ano[idx, :, :], axis=0))
        vol_list.append(float(np.nanmean(vol_ano[idx])))
        year_list.append(int(sy))

    if not map_list:
        return out

    map_ts = np.asarray(map_list, dtype=float)
    out['vol_years'] = np.asarray(year_list, dtype=int)
    out['vol_ano'] = np.asarray(vol_list, dtype=float)

    if map_ts.ndim == 3 and map_ts.shape[0] >= 2:
        std_map, tr_map, tr_p = SIM.ThicknessMetrics._detrended_std_and_trend_map(map_ts)
        out['std_map'] = std_map
        out['trend_map'] = tr_map
        out['trend_p_map'] = tr_p

    return out


def _thickness_extract_month_skill(diff_dict: Optional[dict], month: Optional[int]) -> Tuple[float, float]:
    """Return (Corr, RMSE) from one Thickness/SNdepth 2M payload for one month/annual."""
    if not isinstance(diff_dict, dict):
        return np.nan, np.nan

    corr = np.nan
    rmse = np.nan
    if month is None:
        s1 = diff_dict.get('thick1_metric') if isinstance(diff_dict.get('thick1_metric'), dict) else {}
        s2 = diff_dict.get('thick2_metric') if isinstance(diff_dict.get('thick2_metric'), dict) else {}
        y1 = np.asarray(s1.get('Vol_ano', np.array([])), dtype=float).squeeze()
        y2 = np.asarray(s2.get('Vol_ano', np.array([])), dtype=float).squeeze()
        if y1.ndim == 1 and y2.ndim == 1 and y1.size > 0 and y2.size > 0:
            n_use = min(y1.size, y2.size)
            y1 = y1[:n_use]
            y2 = y2[:n_use]
            valid = np.isfinite(y1) & np.isfinite(y2)
            if int(np.sum(valid)) >= 2:
                y1v = y1[valid]
                y2v = y2[valid]
                if np.nanstd(y1v) > 0 and np.nanstd(y2v) > 0:
                    corr = float(np.corrcoef(y1v, y2v)[0, 1])
                rmse = float(np.sqrt(np.nanmean((y1v - y2v) ** 2)))
                return (
                    float(corr) if np.isfinite(corr) else np.nan,
                    float(rmse) if np.isfinite(rmse) else np.nan,
                )
        return np.nan, np.nan

    candidates = _thickness_month_candidates(int(month))
    for _mtag, mlabel, _mnum in candidates:
        corr_key = f'{mlabel}_Corr'
        rmse_key = f'{mlabel}_RMSE'
        cc = diff_dict.get(corr_key, np.nan)
        rr = diff_dict.get(rmse_key, np.nan)
        if np.isfinite(cc) and np.isfinite(rr):
            return float(cc), float(rr)

    s1 = diff_dict.get('thick1_metric') if isinstance(diff_dict.get('thick1_metric'), dict) else {}
    s2 = diff_dict.get('thick2_metric') if isinstance(diff_dict.get('thick2_metric'), dict) else {}
    for mtag, _mlabel, _mnum in candidates:
        y1 = np.asarray(s1.get(f'Vol_ano_{mtag}', np.array([])), dtype=float).squeeze()
        y2 = np.asarray(s2.get(f'Vol_ano_{mtag}', np.array([])), dtype=float).squeeze()
        if y1.ndim == 1 and y2.ndim == 1 and y1.size > 0 and y2.size > 0:
            n_use = min(y1.size, y2.size)
            y1 = y1[:n_use]
            y2 = y2[:n_use]
            valid = np.isfinite(y1) & np.isfinite(y2)
            if int(np.sum(valid)) >= 2:
                y1v = y1[valid]
                y2v = y2[valid]
                if np.nanstd(y1v) > 0 and np.nanstd(y2v) > 0:
                    corr = float(np.corrcoef(y1v, y2v)[0, 1])
                rmse = float(np.sqrt(np.nanmean((y1v - y2v) ** 2)))
                return (
                    float(corr) if np.isfinite(corr) else np.nan,
                    float(rmse) if np.isfinite(rmse) else np.nan,
                )
    return float(corr) if np.isfinite(corr) else np.nan, float(rmse) if np.isfinite(rmse) else np.nan


def _thickness_extract_month_extended_stats(
    diff_dict: Optional[dict], month: Optional[int]
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return month-level diagnostics for one key month.

    Returns:
        (thick_mean_diff, thick_std_diff, thick_trend_diff,
         vol_mean_diff, vol_std_diff, vol_trend_diff, corr, rmse)
    """
    corr, rmse = _thickness_extract_month_skill(diff_dict, month)
    if not isinstance(diff_dict, dict):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, corr, rmse

    if month is None:
        return (
            float(diff_dict.get('thick_mean_diff', np.nan)) if np.isfinite(diff_dict.get('thick_mean_diff', np.nan)) else np.nan,
            float(diff_dict.get('thick_std_diff', np.nan)) if np.isfinite(diff_dict.get('thick_std_diff', np.nan)) else np.nan,
            float(diff_dict.get('thick_trend_diff', np.nan)) if np.isfinite(diff_dict.get('thick_trend_diff', np.nan)) else np.nan,
            float(diff_dict.get('Vol_mean_diff', np.nan)) if np.isfinite(diff_dict.get('Vol_mean_diff', np.nan)) else np.nan,
            float(diff_dict.get('Vol_std_diff', np.nan)) if np.isfinite(diff_dict.get('Vol_std_diff', np.nan)) else np.nan,
            float(diff_dict.get('Vol_trend_diff', np.nan)) if np.isfinite(diff_dict.get('Vol_trend_diff', np.nan)) else np.nan,
            float(corr) if np.isfinite(corr) else np.nan,
            float(rmse) if np.isfinite(rmse) else np.nan,
        )

    candidates = _thickness_month_candidates(int(month))

    def _pick_key(prefix: str) -> float:
        for mtag, _mlabel, _mnum in candidates:
            vv = diff_dict.get(f'{prefix}_{mtag}', np.nan)
            if np.isfinite(vv):
                return float(vv)
        return np.nan

    mean_diff = _pick_key('thick_mean_diff')
    std_diff = _pick_key('thick_std_diff')
    trend_diff = _pick_key('thick_trend_diff')

    if not (np.isfinite(mean_diff) and np.isfinite(std_diff) and np.isfinite(trend_diff)):
        s1 = diff_dict.get('thick1_metric') if isinstance(diff_dict.get('thick1_metric'), dict) else {}
        s2 = diff_dict.get('thick2_metric') if isinstance(diff_dict.get('thick2_metric'), dict) else {}
        for mtag, _mlabel, _mnum in candidates:
            if not np.isfinite(mean_diff):
                t1 = np.asarray(s1.get(f'thick_clim_{mtag}', np.array([])), dtype=float)
                t2 = np.asarray(s2.get(f'thick_clim_{mtag}', np.array([])), dtype=float)
                if t1.ndim == 2 and t2.ndim == 2 and t1.shape == t2.shape:
                    mean_diff = utils.MatrixDiff(t1, t2, metric='MAE', mask=True)
            if not np.isfinite(std_diff):
                s1m = np.asarray(s1.get(f'thick_ano_std_{mtag}', np.array([])), dtype=float)
                s2m = np.asarray(s2.get(f'thick_ano_std_{mtag}', np.array([])), dtype=float)
                if s1m.ndim == 2 and s2m.ndim == 2 and s1m.shape == s2m.shape:
                    std_diff = utils.MatrixDiff(s1m, s2m, metric='MAE', mask=True)
            if not np.isfinite(trend_diff):
                tr1 = np.asarray(s1.get(f'thick_ano_tr_{mtag}', np.array([])), dtype=float)
                tr2 = np.asarray(s2.get(f'thick_ano_tr_{mtag}', np.array([])), dtype=float)
                if tr1.ndim == 2 and tr2.ndim == 2 and tr1.shape == tr2.shape:
                    trend_diff = utils.MatrixDiff(tr1, tr2, metric='MAE', mask=True)
            if np.isfinite(mean_diff) and np.isfinite(std_diff) and np.isfinite(trend_diff):
                break

    vol_mean_diff = np.nan
    vol_std_diff = np.nan
    vol_trend_diff = np.nan
    s1 = diff_dict.get('thick1_metric') if isinstance(diff_dict.get('thick1_metric'), dict) else {}
    s2 = diff_dict.get('thick2_metric') if isinstance(diff_dict.get('thick2_metric'), dict) else {}
    uni1 = np.asarray(s1.get('uni_mon', np.array([])), dtype=int).reshape(-1)
    uni2 = np.asarray(s2.get('uni_mon', np.array([])), dtype=int).reshape(-1)
    vc1 = np.asarray(s1.get('Vol_clim', np.array([])), dtype=float).reshape(-1)
    vc2 = np.asarray(s2.get('Vol_clim', np.array([])), dtype=float).reshape(-1)
    for mtag, _mlabel, month_num in candidates:
        # Use monthly climatological volume difference for the month table's
        # "Vol Mean Diff" to represent absolute-state mismatch in that month.
        if not np.isfinite(vol_mean_diff) and uni1.size > 0 and uni2.size > 0 and vc1.size > 0 and vc2.size > 0:
            idx1 = np.where(uni1 == int(month_num))[0]
            idx2 = np.where(uni2 == int(month_num))[0]
            if idx1.size > 0 and idx2.size > 0:
                v1m = vc1[int(idx1[0])]
                v2m = vc2[int(idx2[0])]
                if np.isfinite(v1m) and np.isfinite(v2m):
                    vol_mean_diff = abs(float(v1m) - float(v2m))

        y1 = np.asarray(s1.get(f'Vol_ano_{mtag}', np.array([])), dtype=float).squeeze()
        y2 = np.asarray(s2.get(f'Vol_ano_{mtag}', np.array([])), dtype=float).squeeze()
        if y1.ndim == 1 and y2.ndim == 1 and y1.size > 0 and y2.size > 0:
            n_use = min(y1.size, y2.size)
            y1 = y1[:n_use]
            y2 = y2[:n_use]
            valid = np.isfinite(y1) & np.isfinite(y2)
            if int(np.sum(valid)) >= 2:
                y1v = y1[valid]
                y2v = y2[valid]
                if not np.isfinite(vol_mean_diff):
                    vol_mean_diff = abs(float(np.nanmean(y1v)) - float(np.nanmean(y2v)))
                vol_std_diff = abs(float(np.nanstd(y1v)) - float(np.nanstd(y2v)))
                if y1v.size >= 2 and y2v.size >= 2:
                    x = np.arange(y1v.size, dtype=float)
                    try:
                        slope1 = float(np.polyfit(x, y1v, 1)[0])
                        slope2 = float(np.polyfit(x, y2v, 1)[0])
                        vol_trend_diff = abs((slope1 - slope2) * 10.0)
                    except Exception:
                        vol_trend_diff = np.nan
                break
    if not np.isfinite(vol_mean_diff):
        vol_mean_diff = diff_dict.get('Vol_mean_diff', np.nan)
    if not np.isfinite(vol_std_diff):
        vol_std_diff = diff_dict.get('Vol_std_diff', np.nan)
    if not np.isfinite(vol_trend_diff):
        vol_trend_diff = diff_dict.get('Vol_trend_diff', np.nan)

    return (
        float(mean_diff) if np.isfinite(mean_diff) else np.nan,
        float(std_diff) if np.isfinite(std_diff) else np.nan,
        float(trend_diff) if np.isfinite(trend_diff) else np.nan,
        float(vol_mean_diff) if np.isfinite(vol_mean_diff) else np.nan,
        float(vol_std_diff) if np.isfinite(vol_std_diff) else np.nan,
        float(vol_trend_diff) if np.isfinite(vol_trend_diff) else np.nan,
        float(corr) if np.isfinite(corr) else np.nan,
        float(rmse) if np.isfinite(rmse) else np.nan,
    )


def _thickness_extract_month_absolute_stats(
    metric_dict: Optional[dict], month: Optional[int]
) -> Tuple[float, float, float, float, float, float]:
    """Return month-level absolute diagnostics for one key month (or annual mode).

    Returns:
        (thick_mean, thick_std, thick_trend, vol_mean, vol_std, vol_trend)
    """
    if not isinstance(metric_dict, dict):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    thick_mean = np.nan
    thick_std = np.nan
    thick_trend = np.nan
    vol_mean = np.nan
    vol_std = np.nan
    vol_trend = np.nan

    if month is None:
        thick_clim = np.asarray(metric_dict.get('thick_clim', np.array([])), dtype=float)
        if thick_clim.ndim == 3 and thick_clim.shape[0] > 0:
            n_use = min(int(thick_clim.shape[0]), 12)
            month_means = np.array([np.nanmean(thick_clim[ii]) for ii in range(n_use)], dtype=float)
            weights = np.asarray(DAYS_PER_MONTH[:n_use], dtype=float)
            valid = np.isfinite(month_means)
            if np.any(valid):
                w = weights[valid]
                if np.sum(w) > 0:
                    thick_mean = float(np.nansum(month_means[valid] * w) / np.sum(w))
        std_map = np.asarray(metric_dict.get('thick_ano_std', np.array([])), dtype=float)
        if std_map.ndim == 2 and np.any(np.isfinite(std_map)):
            thick_std = float(np.nanmean(std_map))
        tr_map = np.asarray(metric_dict.get('thick_ano_tr', np.array([])), dtype=float)
        if tr_map.ndim == 2 and np.any(np.isfinite(tr_map)):
            thick_trend = float(np.nanmean(tr_map))
    else:
        candidates = _thickness_month_candidates(int(month))
        m_idx = int(month) - 1
        thick_clim = np.asarray(metric_dict.get('thick_clim', np.array([])), dtype=float)
        if thick_clim.ndim == 3 and thick_clim.shape[0] > m_idx:
            vv = np.nanmean(thick_clim[m_idx])
            if np.isfinite(vv):
                thick_mean = float(vv)
        for mtag, _mlabel, _mnum in candidates:
            if not np.isfinite(thick_mean):
                mm = np.asarray(metric_dict.get(f'thick_clim_{mtag}', np.array([])), dtype=float)
                if mm.ndim == 2 and np.any(np.isfinite(mm)):
                    thick_mean = float(np.nanmean(mm))
            if not np.isfinite(thick_std):
                sm = np.asarray(metric_dict.get(f'thick_ano_std_{mtag}', np.array([])), dtype=float)
                if sm.ndim == 2 and np.any(np.isfinite(sm)):
                    thick_std = float(np.nanmean(sm))
            if not np.isfinite(thick_trend):
                tm = np.asarray(metric_dict.get(f'thick_ano_tr_{mtag}', np.array([])), dtype=float)
                if tm.ndim == 2 and np.any(np.isfinite(tm)):
                    thick_trend = float(np.nanmean(tm))
            if np.isfinite(thick_mean) and np.isfinite(thick_std) and np.isfinite(thick_trend):
                break

    vol_clim = np.asarray(metric_dict.get('Vol_clim', np.array([])), dtype=float).reshape(-1)
    uni_mon = np.asarray(metric_dict.get('uni_mon', np.array([])), dtype=int).reshape(-1)
    if month is None:
        if vol_clim.size > 0:
            n_use = min(vol_clim.size, len(DAYS_PER_MONTH))
            vals = vol_clim[:n_use]
            weights = np.asarray(DAYS_PER_MONTH[:n_use], dtype=float)
            valid = np.isfinite(vals)
            if np.any(valid):
                w = weights[valid]
                if np.sum(w) > 0:
                    vol_mean = float(np.nansum(vals[valid] * w) / np.sum(w))
        vol_ano = np.asarray(metric_dict.get('Vol_ano', np.array([])), dtype=float).squeeze()
        if vol_ano.ndim == 1 and vol_ano.size > 0:
            valid = np.isfinite(vol_ano)
            if int(np.sum(valid)) > 0:
                vv = vol_ano[valid]
                vol_std = float(np.nanstd(vv))
                if vv.size >= 2:
                    x = np.arange(vv.size, dtype=float)
                    try:
                        vol_trend = float(np.polyfit(x, vv, 1)[0]) * 10.0
                    except Exception:
                        vol_trend = np.nan
    else:
        candidates = _thickness_month_candidates(int(month))
        for mtag, _mlabel, mnum in candidates:
            if not np.isfinite(vol_mean) and vol_clim.size > 0 and uni_mon.size == vol_clim.size:
                idx = np.where(uni_mon == int(mnum))[0]
                if idx.size > 0:
                    vv = vol_clim[int(idx[0])]
                    if np.isfinite(vv):
                        vol_mean = float(vv)
            vol_ano_mon = np.asarray(metric_dict.get(f'Vol_ano_{mtag}', np.array([])), dtype=float).squeeze()
            if vol_ano_mon.ndim == 1 and vol_ano_mon.size > 0:
                valid = np.isfinite(vol_ano_mon)
                if int(np.sum(valid)) > 0:
                    vv = vol_ano_mon[valid]
                    vol_std = float(np.nanstd(vv))
                    if vv.size >= 2:
                        x = np.arange(vv.size, dtype=float)
                        try:
                            vol_trend = float(np.polyfit(x, vv, 1)[0]) * 10.0
                        except Exception:
                            vol_trend = np.nan
                    break

    return (
        float(thick_mean) if np.isfinite(thick_mean) else np.nan,
        float(thick_std) if np.isfinite(thick_std) else np.nan,
        float(thick_trend) if np.isfinite(thick_trend) else np.nan,
        float(vol_mean) if np.isfinite(vol_mean) else np.nan,
        float(vol_std) if np.isfinite(vol_std) else np.nan,
        float(vol_trend) if np.isfinite(vol_trend) else np.nan,
    )


def _thickness_extract_domain_absolute_stats(
    metric_dict: Optional[dict],
    domain: Tuple[str, Optional[Union[int, str]]],
    hemisphere: str,
) -> Tuple[float, float, float, float, float, float]:
    """Return absolute stats for one domain selector (annual/month/season)."""
    kind = str(domain[0]).lower() if domain else 'annual'
    val = domain[1] if domain else None
    if kind == 'annual':
        return _thickness_extract_month_absolute_stats(metric_dict, None)
    if kind == 'month':
        try:
            month = int(val) if val is not None else None
        except Exception:
            month = None
        return _thickness_extract_month_absolute_stats(metric_dict, month)
    if kind == 'season':
        season_name = str(val or '')
        sdiag = _thickness_seasonal_diagnostics(metric_dict, season_name, hemisphere)
        clim_map = np.asarray(sdiag.get('clim_map', np.array([])), dtype=float)
        std_map = np.asarray(sdiag.get('std_map', np.array([])), dtype=float)
        tr_map = np.asarray(sdiag.get('trend_map', np.array([])), dtype=float)
        vol_ano = np.asarray(sdiag.get('vol_ano', np.array([])), dtype=float).reshape(-1)
        thick_mean = float(np.nanmean(clim_map)) if clim_map.ndim == 2 and np.any(np.isfinite(clim_map)) else np.nan
        thick_std = float(np.nanmean(std_map)) if std_map.ndim == 2 and np.any(np.isfinite(std_map)) else np.nan
        thick_trend = float(np.nanmean(tr_map)) if tr_map.ndim == 2 and np.any(np.isfinite(tr_map)) else np.nan
        vol_mean = float(sdiag.get('vol_clim', np.nan))
        vol_std = float(np.nanstd(vol_ano)) if vol_ano.size > 0 and np.any(np.isfinite(vol_ano)) else np.nan
        vol_trend = np.nan
        if vol_ano.size >= 2:
            valid = np.isfinite(vol_ano)
            if int(np.sum(valid)) >= 2:
                y = vol_ano[valid]
                xx = np.arange(y.size, dtype=float)
                try:
                    vol_trend = float(np.polyfit(xx, y, 1)[0]) * 10.0
                except Exception:
                    vol_trend = np.nan
        return (
            float(thick_mean) if np.isfinite(thick_mean) else np.nan,
            float(thick_std) if np.isfinite(thick_std) else np.nan,
            float(thick_trend) if np.isfinite(thick_trend) else np.nan,
            float(vol_mean) if np.isfinite(vol_mean) else np.nan,
            float(vol_std) if np.isfinite(vol_std) else np.nan,
            float(vol_trend) if np.isfinite(vol_trend) else np.nan,
        )
    return _thickness_extract_month_absolute_stats(metric_dict, None)


def _thickness_extract_domain_extended_stats(
    diff_dict: Optional[dict],
    domain: Tuple[str, Optional[Union[int, str]]],
    hemisphere: str,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return diff stats for one domain selector (annual/month/season)."""
    kind = str(domain[0]).lower() if domain else 'annual'
    val = domain[1] if domain else None
    if kind == 'annual':
        return _thickness_extract_month_extended_stats(diff_dict, None)
    if kind == 'month':
        try:
            month = int(val) if val is not None else None
        except Exception:
            month = None
        return _thickness_extract_month_extended_stats(diff_dict, month)
    if kind == 'season':
        if not isinstance(diff_dict, dict):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        season_name = str(val or '')
        m1 = diff_dict.get('thick1_metric') if isinstance(diff_dict.get('thick1_metric'), dict) else {}
        m2 = diff_dict.get('thick2_metric') if isinstance(diff_dict.get('thick2_metric'), dict) else {}
        d1 = _thickness_seasonal_diagnostics(m1, season_name, hemisphere)
        d2 = _thickness_seasonal_diagnostics(m2, season_name, hemisphere)
        c1 = np.asarray(d1.get('clim_map', np.array([])), dtype=float)
        c2 = np.asarray(d2.get('clim_map', np.array([])), dtype=float)
        s1 = np.asarray(d1.get('std_map', np.array([])), dtype=float)
        s2 = np.asarray(d2.get('std_map', np.array([])), dtype=float)
        t1 = np.asarray(d1.get('trend_map', np.array([])), dtype=float)
        t2 = np.asarray(d2.get('trend_map', np.array([])), dtype=float)
        mean_diff = utils.MatrixDiff(c1, c2, metric='MAE', mask=True) if c1.ndim == 2 and c2.ndim == 2 and c1.shape == c2.shape else np.nan
        std_diff = utils.MatrixDiff(s1, s2, metric='MAE', mask=True) if s1.ndim == 2 and s2.ndim == 2 and s1.shape == s2.shape else np.nan
        trend_diff = utils.MatrixDiff(t1, t2, metric='MAE', mask=True) if t1.ndim == 2 and t2.ndim == 2 and t1.shape == t2.shape else np.nan
        vol_mean_diff = np.nan
        if np.isfinite(d1.get('vol_clim', np.nan)) and np.isfinite(d2.get('vol_clim', np.nan)):
            vol_mean_diff = abs(float(d1['vol_clim']) - float(d2['vol_clim']))
        v1 = np.asarray(d1.get('vol_ano', np.array([])), dtype=float).reshape(-1)
        v2 = np.asarray(d2.get('vol_ano', np.array([])), dtype=float).reshape(-1)
        n_use = min(v1.size, v2.size)
        corr = np.nan
        rmse = np.nan
        vol_std_diff = np.nan
        vol_trend_diff = np.nan
        if n_use >= 2:
            y1 = v1[:n_use]
            y2 = v2[:n_use]
            valid = np.isfinite(y1) & np.isfinite(y2)
            if int(np.sum(valid)) >= 2:
                y1v = y1[valid]
                y2v = y2[valid]
                if np.nanstd(y1v) > 0 and np.nanstd(y2v) > 0:
                    corr = float(np.corrcoef(y1v, y2v)[0, 1])
                rmse = float(np.sqrt(np.nanmean((y1v - y2v) ** 2)))
                vol_std_diff = abs(float(np.nanstd(y1v)) - float(np.nanstd(y2v)))
                xx = np.arange(y1v.size, dtype=float)
                try:
                    slope1 = float(np.polyfit(xx, y1v, 1)[0])
                    slope2 = float(np.polyfit(xx, y2v, 1)[0])
                    vol_trend_diff = abs((slope1 - slope2) * 10.0)
                except Exception:
                    vol_trend_diff = np.nan
        return (
            float(mean_diff) if np.isfinite(mean_diff) else np.nan,
            float(std_diff) if np.isfinite(std_diff) else np.nan,
            float(trend_diff) if np.isfinite(trend_diff) else np.nan,
            float(vol_mean_diff) if np.isfinite(vol_mean_diff) else np.nan,
            float(vol_std_diff) if np.isfinite(vol_std_diff) else np.nan,
            float(vol_trend_diff) if np.isfinite(vol_trend_diff) else np.nan,
            float(corr) if np.isfinite(corr) else np.nan,
            float(rmse) if np.isfinite(rmse) else np.nan,
        )
    return _thickness_extract_month_extended_stats(diff_dict, None)


def _build_thickness_month_period_raw_table(
    obs1_metric: Optional[dict],
    obs2_metric: Optional[dict],
    model_metrics: List[Optional[dict]],
    model_labels: List[str],
    hemisphere: str = 'nh',
    rmse_unit: str = '10^3 km^3',
    obs1_label: str = 'obs1 (baseline)',
    obs2_label: str = 'obs2',
) -> Dict[str, Any]:
    """Build annual/season/month raw (absolute) table for SIthick/SNdepth."""
    headers = [
        'Model/Obs Name',
        'Mean',
        'Std',
        'Trend',
        'Vol Mean',
        'Vol Std',
        'Vol Trend',
    ]
    units = ['', 'm', 'm', 'm/decade', rmse_unit, rmse_unit, f'{rmse_unit}/decade']
    specs = _thickness_domain_specs(hemisphere)
    season_order = [name for name, _ in specs]
    seasons: Dict[str, List[List[str]]] = {}

    for season_name, domain in specs:
        rows: List[List[str]] = []
        if isinstance(obs1_metric, dict):
            m0, s0, t0, vm0, vs0, vt0 = _thickness_extract_domain_absolute_stats(obs1_metric, domain, hemisphere)
            rows.append([
                str(obs1_label),
                _fmt_thickness_month_stat(m0),
                _fmt_thickness_month_stat(s0),
                _fmt_thickness_month_stat(t0),
                _fmt_thickness_month_stat(vm0),
                _fmt_thickness_month_stat(vs0),
                _fmt_thickness_month_stat(vt0),
            ])
        if isinstance(obs2_metric, dict):
            m0, s0, t0, vm0, vs0, vt0 = _thickness_extract_domain_absolute_stats(obs2_metric, domain, hemisphere)
            rows.append([
                str(obs2_label),
                _fmt_thickness_month_stat(m0),
                _fmt_thickness_month_stat(s0),
                _fmt_thickness_month_stat(t0),
                _fmt_thickness_month_stat(vm0),
                _fmt_thickness_month_stat(vs0),
                _fmt_thickness_month_stat(vt0),
            ])
        for ii, metric_dict in enumerate(model_metrics):
            label = model_labels[ii] if ii < len(model_labels) else f'model{ii + 1}'
            m1, s1, t1, vm1, vs1, vt1 = _thickness_extract_domain_absolute_stats(metric_dict, domain, hemisphere)
            rows.append([
                label,
                _fmt_thickness_month_stat(m1),
                _fmt_thickness_month_stat(s1),
                _fmt_thickness_month_stat(t1),
                _fmt_thickness_month_stat(vm1),
                _fmt_thickness_month_stat(vs1),
                _fmt_thickness_month_stat(vt1),
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


def _build_thickness_month_period_table(obs2_diff: Optional[dict],
                                        diff_dicts: List[dict],
                                        model_labels: List[str],
                                        hemisphere: str = 'nh',
                                        rmse_unit: str = '10^3 km^3',
                                        obs1_label: str = 'obs1 (baseline)',
                                        obs2_label: str = 'obs2') -> Dict[str, Any]:
    """Build annual/season/month period tabs for SIthick/SNdepth scalar table."""
    headers = [
        'Model/Obs Name',
        'Mean Diff',
        'Std Diff',
        'Trend Diff',
        'Vol Mean Diff',
        'Vol Std Diff',
        'Vol Trend Diff',
        'Corr',
        'RMSE',
    ]
    units = ['', 'm', 'm', 'm/decade', rmse_unit, rmse_unit, f'{rmse_unit}/decade', '', rmse_unit]
    specs = _thickness_domain_specs(hemisphere)
    season_order = [name for name, _ in specs]
    seasons: Dict[str, List[List[str]]] = {}

    for season_name, domain in specs:
        rows: List[List[str]] = [[
            str(obs1_label),
            _fmt_thickness_month_stat(0.0),
            _fmt_thickness_month_stat(0.0),
            _fmt_thickness_month_stat(0.0),
            _fmt_thickness_month_stat(0.0),
            _fmt_thickness_month_stat(0.0),
            _fmt_thickness_month_stat(0.0),
            _fmt_thickness_month_stat(0.0),
            _fmt_thickness_month_stat(0.0),
        ]]
        obs2_vals: Optional[List[float]] = None
        if isinstance(obs2_diff, dict):
            md0, sd0, td0, vmd0, vsd0, vtd0, c0, r0 = _thickness_extract_domain_extended_stats(
                obs2_diff, domain, hemisphere
            )
            obs2_vals = [md0, sd0, td0, vmd0, vsd0, vtd0, c0, r0]
            rows.append([
                str(obs2_label),
                _fmt_thickness_month_stat(_obs2_identity_ratio(md0)),
                _fmt_thickness_month_stat(_obs2_identity_ratio(sd0)),
                _fmt_thickness_month_stat(_obs2_identity_ratio(td0)),
                _fmt_thickness_month_stat(_obs2_identity_ratio(vmd0)),
                _fmt_thickness_month_stat(_obs2_identity_ratio(vsd0)),
                _fmt_thickness_month_stat(_obs2_identity_ratio(vtd0)),
                _fmt_thickness_month_stat(_obs2_identity_ratio(c0)),
                _fmt_thickness_month_stat(_obs2_identity_ratio(r0)),
            ])

        for ii, dct in enumerate(diff_dicts):
            label = model_labels[ii] if ii < len(model_labels) else f'model{ii + 1}'
            md, sd, td, vmd, vsd, vtd, cc, rr = _thickness_extract_domain_extended_stats(
                dct, domain, hemisphere
            )
            vals = [md, sd, td, vmd, vsd, vtd, cc, rr]
            if obs2_vals is not None:
                vals = [_calc_obs2_ratio(v, b) for v, b in zip(vals, obs2_vals)]
            rows.append([
                label,
                _fmt_thickness_month_stat(vals[0]),
                _fmt_thickness_month_stat(vals[1]),
                _fmt_thickness_month_stat(vals[2]),
                _fmt_thickness_month_stat(vals[3]),
                _fmt_thickness_month_stat(vals[4]),
                _fmt_thickness_month_stat(vals[5]),
                _fmt_thickness_month_stat(vals[6]),
                _fmt_thickness_month_stat(vals[7]),
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


def _adaptive_sit_map_ranges(
    diff_payloads: List[dict],
    month: int,
    raw_floor: float = 5.0,
    diff_floor: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """Return adaptive (raw_range, diff_range) for SIT monthly climatology maps."""
    try:
        month_i = int(month)
    except Exception:
        month_i = 3
    fields: List[np.ndarray] = []

    for idx, payload in enumerate(diff_payloads or []):
        if not isinstance(payload, dict):
            continue
        key = 'thick1_metric' if idx == 0 else 'thick2_metric'
        metric = payload.get(key) if isinstance(payload.get(key), dict) else {}
        if not isinstance(metric, dict):
            continue
        uni_mon = np.asarray(metric.get('uni_mon', np.array([])), dtype=int)
        clim = np.asarray(metric.get('thick_clim', np.array([])), dtype=float)
        if clim.ndim != 3 or uni_mon.size == 0 or month_i not in uni_mon:
            continue
        m_idx = int(np.where(uni_mon == month_i)[0][0])
        field = np.asarray(clim[m_idx], dtype=float)
        if field.ndim != 2:
            continue
        if fields and field.shape != fields[0].shape:
            continue
        fields.append(field)

    raw_floor_f = max(0.1, float(raw_floor))
    diff_floor_f = max(0.1, float(diff_floor))
    if not fields:
        return [0.0, raw_floor_f], [-2.0, 2.0]

    finite_raw = [f[np.isfinite(f)] for f in fields if np.any(np.isfinite(f))]
    raw_vals = np.concatenate(finite_raw) if finite_raw else np.array([], dtype=float)
    raw_p99 = float(np.nanpercentile(raw_vals, 99)) if raw_vals.size else np.nan
    raw_max = max(raw_floor_f, round(raw_p99 * 10) / 10) if np.isfinite(raw_p99) else raw_floor_f

    ref = np.asarray(fields[0], dtype=float)
    finite_diff: List[np.ndarray] = []
    for field in fields[1:]:
        if field.shape != ref.shape:
            continue
        delta = np.asarray(field - ref, dtype=float)
        if np.any(np.isfinite(delta)):
            finite_diff.append(delta[np.isfinite(delta)])
    diff_vals = np.concatenate(finite_diff) if finite_diff else np.array([], dtype=float)
    diff_p98 = float(np.nanpercentile(np.abs(diff_vals), 98)) if diff_vals.size else np.nan
    if np.isfinite(diff_p98):
        diff_max = max(diff_floor_f, round(diff_p98 * 10) / 10)
    else:
        diff_max = max(diff_floor_f, round(raw_max * 0.4 * 10) / 10)
    diff_max = min(max(diff_floor_f, diff_max), max(diff_floor_f, raw_max))
    return [0.0, float(raw_max)], [-float(diff_max), float(diff_max)]


def eval_sithick(case_name: str, recipe: RR.RecipeReader,
                 data_dir: str, output_dir: str,
                 recalculate: bool = False,
                 jobs: int = 1) -> Optional[dict]:
    """Evaluate sea ice thickness (SIthick)."""
    module = 'SIthick'
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
    raw_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'raw_cmap'], 'viridis')
    diff_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'diff_cmap'], 'RdBu_r')
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

    # Check if all expected outputs already exist
    fig_dir = Path(output_dir) / module
    if _check_outputs_exist(module, fig_dir, hemisphere, recalculate=recalculate):
        logger.info(f"{module} evaluation skipped — all outputs exist.")
        return None

    cache_file = _get_metrics_cache_file(case_name, output_dir, hemisphere, module)
    grid_file = _get_eval_grid(case_name, module, hemisphere)
    model_labels = _get_recipe_model_labels(module, module_vars, len(module_vars.get('model_file') or []))
    obs_labels = _get_reference_labels(module_vars, hemisphere)
    obs1_plot_label = obs_labels[0] if len(obs_labels) >= 1 else 'obs1'
    obs2_plot_label = obs_labels[1] if len(obs_labels) >= 2 else 'obs2'
    obs_files: List[str] = []
    model_files: List[str] = []

    obs_1m = None
    obs2_1m = None
    obs2_diff = None
    obs2_diff_matched = None
    diff_dicts: List[dict] = []
    model_labels_used: List[str] = []
    matched_diff_dicts: List[dict] = []
    matched_model_labels_used: List[str] = []
    cache_loaded = False

    if not recalculate:
        cached = _load_module_cache(cache_file, module, hemisphere)
        if cached is not None and cached.get('payload_kind') == module:
            try:
                records = cached.get('records', {})
                obs_1m = records.get('obs_1m')
                obs2_1m = records.get('obs2_1m')
                obs2_diff = records.get('obs2_diff_2m')
                obs2_diff_matched = records.get('obs2_diff_matched_2m')
                diff_records = cached.get('diff_records', [])
                matched_diff_records = cached.get('matched_diff_records', [])
                diff_dicts = [records[r] for r in diff_records if r in records]
                matched_diff_dicts = [records[r] for r in matched_diff_records if r in records]
                model_labels_used = list(cached.get('model_labels_used', []))
                matched_model_labels_used = list(cached.get('matched_model_labels_used', []))
                if not model_labels_used and diff_dicts:
                    model_labels_used = _get_recipe_model_labels(module, module_vars, len(diff_dicts))
                if not matched_model_labels_used and matched_diff_dicts:
                    matched_model_labels_used = _get_recipe_model_labels(
                        module, module_vars, len(matched_diff_dicts)
                    )
                cache_loaded = obs_1m is not None and bool(diff_dicts) and bool(matched_diff_dicts)
                if cache_loaded:
                    logger.info("Loaded %s metrics from cache: %s", module, cache_file)
            except Exception as exc:
                logger.warning("Cache payload for %s is incomplete (%s). Recalculating.", module, exc)
                cache_loaded = False

    if not cache_loaded:
        grid_file, obs_files, model_files = _run_preprocessing(
            case_name, module, recipe, data_dir, frequency='monthly', jobs=jobs
        )

        model_labels = _get_recipe_model_labels(module, module_vars, len(model_files))
        metric = _get_metric(module_vars)
        thick_metrics = SIM.ThicknessMetrics(
            grid_file=grid_file, hemisphere=hemisphere,
            year_sta=year_sta, year_end=year_end, metric=metric,
        )

        obs_key = module_vars.get('ref_var', 'sithick')
        (
            obs_1m,
            obs2_1m,
            obs2_diff,
            obs2_diff_matched,
            diff_dicts,
            model_labels_used,
            matched_diff_dicts,
            matched_model_labels_used,
        ) = _compute_thickness_obs_model_metrics(
            thick_metrics, data_dir, obs_files, model_files, model_labels,
            obs_key=obs_key,
            model_key=module_vars.get('model_var', 'sithick'),
            module_name=module,
            jobs=jobs,
            stage_dir=_get_stage_dir(case_name, output_dir, hemisphere, module) / 'model_metrics',
        )

        model_1m_dicts = [d.get('thick2_metric') for d in diff_dicts]
        model_records = [f'model_{i}_1m' for i in range(len(model_1m_dicts))]
        diff_records = [f'diff_{i}_2m' for i in range(len(diff_dicts))]
        matched_diff_records = [f'matched_diff_{i}_2m' for i in range(len(matched_diff_dicts))]
        records = {'obs_1m': obs_1m}
        if obs2_1m is not None:
            records['obs2_1m'] = obs2_1m
        if obs2_diff is not None:
            records['obs2_diff_2m'] = obs2_diff
        if obs2_diff_matched is not None:
            records['obs2_diff_matched_2m'] = obs2_diff_matched
        records.update({name: d for name, d in zip(model_records, model_1m_dicts)})
        records.update({name: d for name, d in zip(diff_records, diff_dicts)})
        records.update({name: d for name, d in zip(matched_diff_records, matched_diff_dicts)})

        used_entities: set = set()
        obs1_entity = _unique_entity_name(
            preferred=obs_labels[0] if len(obs_labels) >= 1 else 'Reference_1',
            fallback='Reference_1',
            used=used_entities,
        )
        entity_groups: Dict[str, str] = {'obs_1m': obs1_entity}
        if obs2_1m is not None:
            obs2_entity = _unique_entity_name(
                preferred=obs_labels[1] if len(obs_labels) >= 2 else 'Reference_2',
                fallback='Reference_2',
                used=used_entities,
            )
            entity_groups['obs2_1m'] = obs2_entity
            if obs2_diff is not None:
                entity_groups['obs2_diff_2m'] = _unique_entity_name(
                    preferred=f'{obs1_entity}_vs_{obs2_entity}',
                    fallback='Reference_1_vs_Reference_2',
                    used=used_entities,
                )

        for i, rec_name in enumerate(model_records):
            model_name = model_labels_used[i] if i < len(model_labels_used) else f'{module}_dataset_{i + 1}'
            entity_groups[rec_name] = _unique_entity_name(
                preferred=model_name,
                fallback=model_name,
                used=used_entities,
            )

        for i, rec_name in enumerate(diff_records):
            model_name = model_labels_used[i] if i < len(model_labels_used) else f'{module}_dataset_{i + 1}'
            entity_groups[rec_name] = _unique_entity_name(
                preferred=f'{obs1_entity}_vs_{model_name}',
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
                    'model_labels_used': model_labels_used,
                    'matched_model_labels_used': matched_model_labels_used,
                    'model_records': model_records,
                    'diff_records': diff_records,
                    'matched_diff_records': matched_diff_records,
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
            obs_1m_plot = records.get('obs_1m')
            diff_records_plot = cached_for_plot.get('diff_records', [])
            matched_diff_records_plot = cached_for_plot.get('matched_diff_records', [])
            diff_dicts_plot = [records[r] for r in diff_records_plot if r in records]
            matched_diff_dicts_plot = [records[r] for r in matched_diff_records_plot if r in records]
            if obs_1m_plot is not None and diff_dicts_plot and matched_diff_dicts_plot:
                obs_1m = obs_1m_plot
                obs2_1m = records.get('obs2_1m')
                obs2_diff = records.get('obs2_diff_2m')
                obs2_diff_matched = records.get('obs2_diff_matched_2m')
                diff_dicts = diff_dicts_plot
                matched_diff_dicts = matched_diff_dicts_plot
                model_labels_used = list(cached_for_plot.get('model_labels_used', model_labels_used))
                matched_model_labels_used = list(
                    cached_for_plot.get('matched_model_labels_used', matched_model_labels_used)
                )
                if not model_labels_used and diff_dicts:
                    model_labels_used = _get_recipe_model_labels(module, module_vars, len(diff_dicts))
                if not matched_model_labels_used and matched_diff_dicts:
                    matched_model_labels_used = _get_recipe_model_labels(
                        module, module_vars, len(matched_diff_dicts)
                    )
                logger.info("Using cache-backed %s payload for plotting: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to reload %s cache for plotting (%s). Using in-memory payload.", module, exc)

    # Metric-level multi-model group means (computed after per-model metrics exist).
    base_diff_dicts_for_ts = list(diff_dicts)
    base_model_labels_for_ts = list(model_labels_used)
    base_model_metric_payloads_for_ts = [
        d.get('thick2_metric') if isinstance(d, dict) else None
        for d in base_diff_dicts_for_ts
    ]
    base_matched_diff_dicts_for_ts = list(matched_diff_dicts)
    base_matched_labels_for_ts = list(matched_model_labels_used)
    base_matched_model_metric_payloads_for_ts = [
        d.get('thick2_metric') if isinstance(d, dict) else None
        for d in base_matched_diff_dicts_for_ts
    ]

    group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=list(model_labels_used),
    )
    group_labels: List[str] = []
    group_model_metric_means_for_ts: List[Any] = []
    group_model_metric_stds_for_ts: List[Any] = []
    group_diff_means_for_ts: List[Any] = []
    group_diff_stds_for_ts: List[Any] = []
    if group_specs and base_model_metric_payloads_for_ts:
        group_model_metric_means_for_ts, group_model_metric_stds_for_ts, _g_labels = _build_group_mean_std_payloads(
            model_payloads=base_model_metric_payloads_for_ts,
            model_labels=list(base_model_labels_for_ts),
            group_specs=group_specs,
        )
    if group_specs and base_diff_dicts_for_ts:
        group_diff_means_for_ts, group_diff_stds_for_ts, _gd_labels = _build_group_mean_std_payloads(
            model_payloads=base_diff_dicts_for_ts,
            model_labels=list(base_model_labels_for_ts),
            group_specs=group_specs,
        )
    if group_specs and diff_dicts:
        model_metric_payloads = [
            d.get('thick2_metric') if isinstance(d, dict) else None
            for d in diff_dicts
        ]
        _grouped_models, grouped_diffs, grouped_labels, group_labels = _build_group_mean_payloads(
            model_payloads=model_metric_payloads,
            diff_payloads=diff_dicts,
            model_labels=list(model_labels_used),
            group_specs=group_specs,
        )
        diff_dicts = list(grouped_diffs or [])
        model_labels_used = list(grouped_labels)
        if group_labels:
            logger.info(
                "Enabled metric-level group means for %s [%s]: %s",
                module, hemisphere.upper(), ', '.join(group_labels),
            )

    matched_group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=list(base_matched_labels_for_ts),
    )
    matched_group_labels: List[str] = []
    matched_group_model_metric_means_for_ts: List[Any] = []
    matched_group_model_metric_stds_for_ts: List[Any] = []
    matched_group_diff_means_for_ts: List[Any] = []
    matched_group_diff_stds_for_ts: List[Any] = []
    if matched_group_specs and base_matched_model_metric_payloads_for_ts:
        matched_group_model_metric_means_for_ts, matched_group_model_metric_stds_for_ts, _mg_labels = _build_group_mean_std_payloads(
            model_payloads=base_matched_model_metric_payloads_for_ts,
            model_labels=list(base_matched_labels_for_ts),
            group_specs=matched_group_specs,
        )
    if matched_group_specs and base_matched_diff_dicts_for_ts:
        matched_group_diff_means_for_ts, matched_group_diff_stds_for_ts, _mgd_labels = _build_group_mean_std_payloads(
            model_payloads=base_matched_diff_dicts_for_ts,
            model_labels=list(base_matched_labels_for_ts),
            group_specs=matched_group_specs,
        )
    if matched_group_specs and matched_diff_dicts:
        matched_model_payloads = [
            d.get('thick2_metric') if isinstance(d, dict) else None
            for d in matched_diff_dicts
        ]
        _matched_grouped_models, grouped_matched_diffs, grouped_matched_labels, matched_group_labels = _build_group_mean_payloads(
            model_payloads=matched_model_payloads,
            diff_payloads=matched_diff_dicts,
            model_labels=list(matched_model_labels_used),
            group_specs=matched_group_specs,
        )
        matched_diff_dicts = list(grouped_matched_diffs or [])
        matched_model_labels_used = list(grouped_matched_labels)
        if matched_group_labels:
            logger.info(
                "Enabled metric-level matched group means for %s [%s]: %s",
                module, hemisphere.upper(), ', '.join(matched_group_labels),
            )

    if diff_dicts:
        logger.info('Generating figures ...')
        fig_dir = Path(output_dir) / module
        fig_dir.mkdir(parents=True, exist_ok=True)
        key_months = SIM.SeaIceMetricsBase.resolve_key_months(hemisphere)

        obs_pair_base, all_diff_dicts_map_base, all_labels_map_base, all_diff_dicts_base, all_labels_base, model_dicts_base = \
            _build_thickness_label_sets(
                obs_1m, obs2_1m, obs2_diff, base_diff_dicts_for_ts, base_model_labels_for_ts, obs_labels=obs_labels
            )
        obs_pair, all_diff_dicts_map, all_labels_map, all_diff_dicts, all_labels, model_dicts = \
            _build_thickness_label_sets(
                obs_1m, obs2_1m, obs2_diff, diff_dicts, model_labels_used, obs_labels=obs_labels
            )
        obs_pair_base, all_diff_dicts_map_base, all_labels_map_base, all_diff_dicts_base, all_labels_base, model_dicts_base = \
            _build_thickness_label_sets(
                obs_1m, obs2_1m, obs2_diff, base_diff_dicts_for_ts, base_model_labels_for_ts, obs_labels=obs_labels
            )

        with xr.open_dataset(grid_file) as ds:
            _lon, _lat = np.array(ds['lon']), np.array(ds['lat'])

        pf.plot_sit_ts(
            model_dicts_base, all_diff_dicts_map_base, all_labels_map_base, str(fig_dir / 'SIT_ts.png'),
            line_style=line_styles, color=line_colors,
        )
        pf.plot_sit_ano(
            model_dicts_base, all_diff_dicts_base, all_labels_base, module_vars['year_range'],
            hms=hemisphere, fig_name=str(fig_dir / 'SIT_ano.png'),
            line_style=line_styles, color=line_colors,
        )

        _sit_metric_all = [obs_1m] + ([obs2_1m] if obs2_1m is not None else []) + [d['thick2_metric'] for d in base_diff_dicts_for_ts]
        _sit_labels_all = [obs1_plot_label] + ([obs2_plot_label] if obs2_1m is not None else []) + base_model_labels_for_ts
        for month in key_months:
            mtag = SIM.SeaIceMetricsBase.month_tag(month)
            pf.plot_sit_key_month_ano(
                _sit_metric_all, _sit_labels_all, module_vars['year_range'], month,
                hms=hemisphere, fig_name=str(fig_dir / f'sithick_ano_timeseries_{mtag}.png'),
                line_style=line_styles, color=line_colors,
            )
            if group_labels and group_model_metric_means_for_ts:
                pf.plot_sit_key_month_ano(
                    [obs_1m] + ([obs2_1m] if obs2_1m is not None else []) + group_model_metric_means_for_ts,
                    [obs1_plot_label] + ([obs2_plot_label] if obs2_1m is not None else []) + group_labels,
                    module_vars['year_range'], month,
                    hms=hemisphere,
                    fig_name=str(fig_dir / f'sithick_ano_timeseries_{mtag}_groupmean.png'),
                    line_style=line_styles, color=line_colors,
                    model_spread_payloads=group_model_metric_stds_for_ts,
                )

        if group_labels and group_model_metric_means_for_ts and group_diff_means_for_ts:
            pf.plot_sit_ts(
                group_model_metric_means_for_ts,
                ([obs_pair_base] if obs_pair_base else []) + group_diff_means_for_ts,
                [obs1_plot_label, obs2_plot_label][: (2 if obs_pair_base else 1)] + group_labels,
                str(fig_dir / 'SIT_ts_groupmean.png'),
                line_style=line_styles, color=line_colors,
                model_spread_payloads=group_model_metric_stds_for_ts,
            )
            pf.plot_sit_ano(
                group_model_metric_means_for_ts,
                ([obs2_diff] if obs2_diff else []) + group_diff_means_for_ts,
                [obs1_plot_label] + ([obs2_plot_label] if obs2_diff else []) + group_labels,
                module_vars['year_range'],
                hms=hemisphere,
                fig_name=str(fig_dir / 'SIT_ano_groupmean.png'),
                line_style=line_styles, color=line_colors,
                model_spread_payloads=group_model_metric_stds_for_ts,
            )

        _season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
        _sit_metric_maps = [obs_1m] + ([obs2_1m] if obs2_1m is not None else []) + [d['thick2_metric'] for d in diff_dicts]
        _sit_labels_maps = [obs1_plot_label] + ([obs2_plot_label] if obs2_1m is not None else []) + model_labels_used

        def _stack_season_maps(diag_list: List[Dict[str, Any]], key: str) -> Optional[np.ndarray]:
            ref = None
            for dd in diag_list:
                arr = np.asarray(dd.get(key, np.array([])), dtype=float)
                if arr.ndim == 2:
                    ref = arr.shape
                    break
            if ref is None:
                return None
            rows = []
            for dd in diag_list:
                arr = np.asarray(dd.get(key, np.array([])), dtype=float)
                if arr.ndim != 2 or arr.shape != ref:
                    arr = np.full(ref, np.nan, dtype=float)
                rows.append(arr)
            return np.asarray(rows, dtype=float)

        for _season in _season_names:
            _diags = [_thickness_seasonal_diagnostics(mdict, _season, hemisphere) for mdict in _sit_metric_maps]
            _clim_stack = _stack_season_maps(_diags, 'clim_map')
            if _clim_stack is not None:
                _clim_p95 = np.nanpercentile(np.abs(_clim_stack[0]), 95) if np.any(np.isfinite(_clim_stack[0])) else np.nan
                _clim_max = max(0.1, round(float(_clim_p95) * 10) / 10) if np.isfinite(_clim_p95) else 5.0
                pf.plot_SIC_map(
                    grid_file, _clim_stack, _sit_labels_maps, hemisphere,
                    sic_range=[0, _clim_max], diff_range=[-_clim_max * 0.5, _clim_max * 0.5],
                    sic_cm=raw_cmap, diff_cm=diff_cmap, unit='m',
                    plot_mode='raw', fig_name=str(fig_dir / f'SIT_seasonal_clim_{_season}_raw.png'),
                )
                pf.plot_SIC_map(
                    grid_file, _clim_stack, _sit_labels_maps, hemisphere,
                    sic_range=[0, _clim_max], diff_range=[-_clim_max * 0.5, _clim_max * 0.5],
                    sic_cm=raw_cmap, diff_cm=diff_cmap, unit='m',
                    plot_mode='diff', fig_name=str(fig_dir / f'SIT_seasonal_clim_{_season}_diff.png'),
                )

            _std_stack = _stack_season_maps(_diags, 'std_map')
            if _std_stack is not None:
                _std_p95 = np.nanpercentile(_std_stack[0], 95) if np.any(np.isfinite(_std_stack[0])) else np.nan
                _std_max = max(0.05, round(float(_std_p95) * 20) / 20) if np.isfinite(_std_p95) else 0.5
                pf.plot_SIC_map(
                    grid_file, _std_stack, _sit_labels_maps, hemisphere,
                    sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                    sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                    plot_mode='raw', fig_name=str(fig_dir / f'SIT_seasonal_std_{_season}_raw.png'),
                )
                pf.plot_SIC_map(
                    grid_file, _std_stack, _sit_labels_maps, hemisphere,
                    sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                    sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                    plot_mode='diff', fig_name=str(fig_dir / f'SIT_seasonal_std_{_season}_diff.png'),
                )

            _tr_stack = _stack_season_maps(_diags, 'trend_map')
            _trp_stack = _stack_season_maps(_diags, 'trend_p_map')
            if _tr_stack is not None and _trp_stack is not None:
                _tr_p95 = np.nanpercentile(np.abs(_tr_stack[0]), 95) if np.any(np.isfinite(_tr_stack[0])) else np.nan
                _tr_max = max(0.05, round(float(_tr_p95) * 20) / 20) if np.isfinite(_tr_p95) else 0.5
                pf.plot_trend_map(
                    grid_file, _tr_stack, _trp_stack, _sit_labels_maps, hemisphere,
                    trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/decade',
                    plot_mode='raw', fig_name=str(fig_dir / f'SIT_seasonal_trend_{_season}_raw.png'),
                )
                pf.plot_trend_map(
                    grid_file, _tr_stack, _trp_stack, _sit_labels_maps, hemisphere,
                    trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/decade',
                    plot_mode='diff', fig_name=str(fig_dir / f'SIT_seasonal_trend_{_season}_diff.png'),
                )

        _mon = 3 if hemisphere == 'nh' else 9
        _sit_raw_range, _sit_diff_range = _adaptive_sit_map_ranges(all_diff_dicts_map, _mon, raw_floor=5.0)
        pf.plot_sit_map(_lon, _lat, all_diff_dicts_map, _mon, hemisphere,
                        data_range=_sit_raw_range, diff_range=_sit_diff_range, model_labels=all_labels_map,
                        cm=raw_cmap, diff_cm=diff_cmap,
                        plot_mode='raw', fig_name=str(fig_dir / 'SIT_map_raw.png'))
        pf.plot_sit_map(_lon, _lat, all_diff_dicts_map, _mon, hemisphere,
                        data_range=_sit_raw_range, diff_range=_sit_diff_range, model_labels=all_labels_map,
                        cm=raw_cmap, diff_cm=diff_cmap,
                        plot_mode='diff', fig_name=str(fig_dir / 'SIT_map_diff.png'))

        _obs_1m = obs_1m if obs_1m is not None else diff_dicts[0]['thick1_metric']
        _obs2_1m = obs2_1m if obs2_1m is not None else (obs2_diff['thick2_metric'] if obs2_diff is not None else None)
        _thick_std_maps = np.array([_obs_1m['thick_ano_std']] + ([_obs2_1m['thick_ano_std']] if _obs2_1m is not None else []) + [d['thick2_metric']['thick_ano_std'] for d in diff_dicts])
        _std_max = max(0.1, round(float(np.nanpercentile(_thick_std_maps[0], 95)) * 10) / 10)
        pf.plot_SIC_map(grid_file, _thick_std_maps, all_labels_map, hemisphere,
                        sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                        sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                        plot_mode='raw', fig_name=str(fig_dir / 'SIT_std_map_raw.png'))
        pf.plot_SIC_map(grid_file, _thick_std_maps, all_labels_map, hemisphere,
                        sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                        sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                        plot_mode='diff', fig_name=str(fig_dir / 'SIT_std_map_diff.png'))

        _thick_tr_maps = np.array([_obs_1m['thick_ano_tr']] + ([_obs2_1m['thick_ano_tr']] if _obs2_1m is not None else []) + [d['thick2_metric']['thick_ano_tr'] for d in diff_dicts])
        _thick_tr_pval = np.array([_obs_1m['thick_ano_tr_p']] + ([_obs2_1m['thick_ano_tr_p']] if _obs2_1m is not None else []) + [d['thick2_metric']['thick_ano_tr_p'] for d in diff_dicts])
        _tr_max = max(0.1, round(float(np.nanpercentile(np.abs(_thick_tr_maps[0]), 95)) * 10) / 10)
        pf.plot_trend_map(grid_file, _thick_tr_maps, _thick_tr_pval, all_labels_map, hemisphere,
                          trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/decade',
                          plot_mode='raw', fig_name=str(fig_dir / 'SIT_trend_map_raw.png'))
        pf.plot_trend_map(grid_file, _thick_tr_maps, _thick_tr_pval, all_labels_map, hemisphere,
                          trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/decade',
                          plot_mode='diff', fig_name=str(fig_dir / 'SIT_trend_map_diff.png'))

        _sit_keys = ['thick_mean_diff', 'thick_std_diff', 'thick_trend_diff', 'Vol_mean_diff', 'Vol_std_diff', 'Vol_trend_diff']
        for month in key_months:
            mlabel = SIM.SeaIceMetricsBase.month_label(month)
            _sit_keys.extend([f'{mlabel}_RMSE'])
        _sit_heat = np.array([[d.get(k, float('nan')) for k in _sit_keys] for d in diff_dicts])
        with _PLOT_LOCK:
            _fig, _ax = plt.subplots(
                figsize=(
                    max(8, len(_sit_keys) * 0.8),
                    max(3, 1 + len(diff_dicts)) * max(0.1, heat_main_height_scale),
                )
            )
            if obs2_diff is not None:
                _obs2_row = np.array([abs(obs2_diff.get(k, float('nan'))) for k in _sit_keys])
                pf.plot_heat_map(
                    _sit_heat, model_labels_used, _sit_keys, ax=_ax,
                    cbarlabel='Ratio to obs uncertainty', obs_row=_obs2_row,
                    obs_row_label=obs2_plot_label,
                    ratio_vmin=heat_ratio_vmin, ratio_vmax=heat_ratio_vmax,
                    cmap=heatmap_cmap,
                )
            else:
                pf.plot_heat_map(_sit_heat, model_labels_used, _sit_keys, ax=_ax, cbarlabel='m', cmap=heatmap_cmap)
            _fig.tight_layout()
            pf._save_fig(str(fig_dir / 'heat_map.png'), close=False)
            plt.close(_fig)

        _sit_season_core = ['MeanDiff', 'StdDiff', 'TrendDiff', 'RMSE']
        _sit_season_keys: List[str] = []
        for _season in _season_names:
            _sit_season_keys.extend([
                f'{_season}_{_kk}' for _kk in _sit_season_core
            ])
        _sit_heat_season = []
        for _dct in diff_dicts:
            _row: List[float] = []
            for _season in _season_names:
                _md, _sd, _td, _vmd, _vsd, _vtd, _cc, _rr = _thickness_extract_domain_extended_stats(
                    _dct, ('season', _season), hemisphere
                )
                _row.extend([_md, _sd, _td, _rr])
            _sit_heat_season.append(_row)
        _sit_heat_season = np.asarray(_sit_heat_season, dtype=float)
        if _sit_heat_season.size > 0:
            with _PLOT_LOCK:
                _n_metric = len(_sit_season_core)
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
                if obs2_diff is not None:
                    _obs2_row = []
                    for _season in _season_names:
                        _md, _sd, _td, _vmd, _vsd, _vtd, _cc, _rr = _thickness_extract_domain_extended_stats(
                            obs2_diff, ('season', _season), hemisphere
                        )
                        _obs2_row.extend([_md, _sd, _td, _rr])
                    _obs2_row = np.asarray(_obs2_row, dtype=float)
                else:
                    _obs2_row = None

                for _ii, _season in enumerate(_season_names):
                    _ax = _axes_flat[_ii]
                    _j0 = _ii * _n_metric
                    _j1 = _j0 + _n_metric
                    _im = pf.plot_heat_map(
                        _sit_heat_season[:, _j0:_j1],
                        model_labels_used,
                        _sit_season_core,
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
                            data=np.abs(_sit_heat_season),
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

        if matched_diff_dicts:
            if obs2_diff_matched is not None:
                _obs_pair_matched = {
                    'thick1_metric': obs2_diff_matched.get('thick1_metric'),
                    'thick2_metric': obs2_diff_matched.get('thick2_metric'),
                }
                _all_diff_dicts_map_matched = [_obs_pair_matched] + base_matched_diff_dicts_for_ts
                _all_labels_map_matched = [obs1_plot_label, obs2_plot_label] + base_matched_labels_for_ts
                _all_diff_dicts_matched = [obs2_diff_matched] + base_matched_diff_dicts_for_ts
                _all_labels_matched = [obs1_plot_label, obs2_plot_label] + base_matched_labels_for_ts
                _model_dicts_matched = [d['thick2_metric'] for d in base_matched_diff_dicts_for_ts]
            else:
                (
                    _obs_pair_matched,
                    _all_diff_dicts_map_matched,
                    _all_labels_map_matched,
                    _all_diff_dicts_matched,
                    _all_labels_matched,
                    _model_dicts_matched,
                ) = _build_thickness_label_sets(
                    obs_1m, None, None, base_matched_diff_dicts_for_ts, base_matched_labels_for_ts, obs_labels=obs_labels
                )

            pf.plot_sit_ts(
                _model_dicts_matched,
                _all_diff_dicts_map_matched,
                _all_labels_map_matched,
                str(fig_dir / 'SIT_ts_matched.png'),
                line_style=line_styles, color=line_colors,
            )
            pf.plot_sit_ano(
                _model_dicts_matched,
                _all_diff_dicts_matched,
                _all_labels_matched,
                module_vars['year_range'],
                hms=hemisphere,
                fig_name=str(fig_dir / 'SIT_ano_matched.png'),
                line_style=line_styles, color=line_colors,
            )

            if obs2_diff_matched is not None:
                _sit_metric_all_matched = [obs2_diff_matched['thick1_metric'], obs2_diff_matched['thick2_metric']] + [d['thick2_metric'] for d in base_matched_diff_dicts_for_ts]
                _sit_labels_all_matched = [obs1_plot_label, obs2_plot_label] + base_matched_labels_for_ts
            else:
                _sit_metric_all_matched = [obs_1m] + [d['thick2_metric'] for d in base_matched_diff_dicts_for_ts]
                _sit_labels_all_matched = [obs1_plot_label] + base_matched_labels_for_ts
            for month in key_months:
                mtag = SIM.SeaIceMetricsBase.month_tag(month)
                pf.plot_sit_key_month_ano(
                    _sit_metric_all_matched,
                    _sit_labels_all_matched,
                    module_vars['year_range'],
                    month,
                    hms=hemisphere,
                    fig_name=str(fig_dir / f'sithick_ano_timeseries_{mtag}_matched.png'),
                    line_style=line_styles, color=line_colors,
                )
                if matched_group_labels and matched_group_model_metric_means_for_ts:
                    pf.plot_sit_key_month_ano(
                        [obs2_diff_matched['thick1_metric'], obs2_diff_matched['thick2_metric']] + matched_group_model_metric_means_for_ts
                        if obs2_diff_matched is not None else
                        [obs_1m] + matched_group_model_metric_means_for_ts,
                        [obs1_plot_label, obs2_plot_label][: (2 if obs2_diff_matched is not None else 1)] + matched_group_labels,
                        module_vars['year_range'],
                        month,
                        hms=hemisphere,
                        fig_name=str(fig_dir / f'sithick_ano_timeseries_{mtag}_matched_groupmean.png'),
                        line_style=line_styles, color=line_colors,
                        model_spread_payloads=matched_group_model_metric_stds_for_ts,
                    )

            if matched_group_labels and matched_group_model_metric_means_for_ts and matched_group_diff_means_for_ts:
                pf.plot_sit_ts(
                    matched_group_model_metric_means_for_ts,
                    ([_obs_pair_matched] if _obs_pair_matched is not None else []) + matched_group_diff_means_for_ts,
                    [obs1_plot_label, obs2_plot_label][: (2 if _obs_pair_matched is not None else 1)] + matched_group_labels,
                    str(fig_dir / 'SIT_ts_matched_groupmean.png'),
                    line_style=line_styles, color=line_colors,
                    model_spread_payloads=matched_group_model_metric_stds_for_ts,
                )
                pf.plot_sit_ano(
                    matched_group_model_metric_means_for_ts,
                    ([obs2_diff_matched] if obs2_diff_matched is not None else []) + matched_group_diff_means_for_ts,
                    [obs1_plot_label] + ([obs2_plot_label] if obs2_diff_matched is not None else []) + matched_group_labels,
                    module_vars['year_range'],
                    hms=hemisphere,
                    fig_name=str(fig_dir / 'SIT_ano_matched_groupmean.png'),
                    line_style=line_styles, color=line_colors,
                    model_spread_payloads=matched_group_model_metric_stds_for_ts,
                )

            _mon = 3 if hemisphere == 'nh' else 9
            _sit_raw_range_m, _sit_diff_range_m = _adaptive_sit_map_ranges(_all_diff_dicts_map_matched, _mon, raw_floor=5.0)
            pf.plot_sit_map(
                _lon, _lat, _all_diff_dicts_map_matched, _mon, hemisphere,
                data_range=_sit_raw_range_m, diff_range=_sit_diff_range_m, model_labels=_all_labels_map_matched,
                cm=raw_cmap, diff_cm=diff_cmap,
                plot_mode='raw', fig_name=str(fig_dir / 'SIT_map_matched_raw.png'),
            )
            pf.plot_sit_map(
                _lon, _lat, _all_diff_dicts_map_matched, _mon, hemisphere,
                data_range=_sit_raw_range_m, diff_range=_sit_diff_range_m, model_labels=_all_labels_map_matched,
                cm=raw_cmap, diff_cm=diff_cmap,
                plot_mode='diff', fig_name=str(fig_dir / 'SIT_map_matched_diff.png'),
            )

            _obs_1m_matched = (
                obs2_diff_matched['thick1_metric']
                if obs2_diff_matched is not None else matched_diff_dicts[0]['thick1_metric']
            )
            _thick_std_maps_matched = np.array(
                ([_obs_1m_matched['thick_ano_std'], obs2_diff_matched['thick2_metric']['thick_ano_std']]
                 if obs2_diff_matched is not None else [_obs_1m_matched['thick_ano_std']])
                + [d['thick2_metric']['thick_ano_std'] for d in matched_diff_dicts]
            )
            _std_max_matched = max(0.1, round(float(np.nanpercentile(_thick_std_maps_matched[0], 95)) * 10) / 10)
            pf.plot_SIC_map(
                grid_file, _thick_std_maps_matched, _all_labels_map_matched, hemisphere,
                sic_range=[0, _std_max_matched], diff_range=[-_std_max_matched * 0.5, _std_max_matched * 0.5],
                sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                plot_mode='raw', fig_name=str(fig_dir / 'SIT_std_map_matched_raw.png'),
            )
            pf.plot_SIC_map(
                grid_file, _thick_std_maps_matched, _all_labels_map_matched, hemisphere,
                sic_range=[0, _std_max_matched], diff_range=[-_std_max_matched * 0.5, _std_max_matched * 0.5],
                sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                plot_mode='diff', fig_name=str(fig_dir / 'SIT_std_map_matched_diff.png'),
            )

            _thick_tr_maps_matched = np.array(
                ([_obs_1m_matched['thick_ano_tr'], obs2_diff_matched['thick2_metric']['thick_ano_tr']]
                 if obs2_diff_matched is not None else [_obs_1m_matched['thick_ano_tr']])
                + [d['thick2_metric']['thick_ano_tr'] for d in matched_diff_dicts]
            )
            _thick_tr_pval_matched = np.array(
                ([_obs_1m_matched['thick_ano_tr_p'], obs2_diff_matched['thick2_metric']['thick_ano_tr_p']]
                 if obs2_diff_matched is not None else [_obs_1m_matched['thick_ano_tr_p']])
                + [d['thick2_metric']['thick_ano_tr_p'] for d in matched_diff_dicts]
            )
            _tr_max_matched = max(0.1, round(float(np.nanpercentile(np.abs(_thick_tr_maps_matched[0]), 95)) * 10) / 10)
            pf.plot_trend_map(
                grid_file, _thick_tr_maps_matched, _thick_tr_pval_matched, _all_labels_map_matched, hemisphere,
                trend_range=[-_tr_max_matched, _tr_max_matched], cmap=trend_cmap, unit='m/decade',
                plot_mode='raw', fig_name=str(fig_dir / 'SIT_trend_map_matched_raw.png'),
            )
            pf.plot_trend_map(
                grid_file, _thick_tr_maps_matched, _thick_tr_pval_matched, _all_labels_map_matched, hemisphere,
                trend_range=[-_tr_max_matched, _tr_max_matched], cmap=trend_cmap, unit='m/decade',
                plot_mode='diff', fig_name=str(fig_dir / 'SIT_trend_map_matched_diff.png'),
            )

            _sit_heat_matched = np.array([[d.get(k, float('nan')) for k in _sit_keys] for d in matched_diff_dicts])
            with _PLOT_LOCK:
                _fig, _ax = plt.subplots(
                    figsize=(
                        max(8, len(_sit_keys) * 0.8),
                        max(3, 1 + len(matched_diff_dicts)) * max(0.1, heat_main_height_scale),
                    )
                )
                if obs2_diff_matched is not None:
                    _obs2_row = np.array([abs(obs2_diff_matched.get(k, float('nan'))) for k in _sit_keys])
                    pf.plot_heat_map(
                        _sit_heat_matched, matched_model_labels_used, _sit_keys, ax=_ax,
                        cbarlabel='Ratio to obs uncertainty',
                        obs_row=_obs2_row, obs_row_label=obs2_plot_label,
                        ratio_vmin=heat_ratio_vmin, ratio_vmax=heat_ratio_vmax,
                        cmap=heatmap_cmap,
                    )
                else:
                    pf.plot_heat_map(
                        _sit_heat_matched, matched_model_labels_used, _sit_keys,
                        ax=_ax, cbarlabel='m', cmap=heatmap_cmap,
                    )
                _fig.tight_layout()
                pf._save_fig(str(fig_dir / 'heat_map_matched.png'), close=False)
                plt.close(_fig)

        try:
            pf.plot_sic_region_map(
                grid_nc_file=grid_file,
                hms=hemisphere,
                fig_name=str(fig_dir / 'SeaIceRegion_map.png'),
            )
        except Exception as exc:
            logger.warning("Failed to generate sea-ice region map (%s).", exc)

    logger.info('%s evaluation completed.', module)
    if not diff_dicts:
        return None

    base_raw_table = _build_thickness_month_period_raw_table(
        obs1_metric=obs_1m,
        obs2_metric=obs2_1m,
        model_metrics=[
            d.get('thick2_metric')
            for d in diff_dicts
            if isinstance(d, dict) and isinstance(d.get('thick2_metric'), dict)
        ],
        model_labels=model_labels_used,
        hemisphere=hemisphere,
        rmse_unit='10^3 km^3',
        obs1_label=f'{obs1_plot_label} (baseline)',
        obs2_label=obs2_plot_label,
    )
    base_diff_table = _build_thickness_month_period_table(
        obs2_diff=obs2_diff,
        diff_dicts=diff_dicts,
        model_labels=model_labels_used,
        hemisphere=hemisphere,
        rmse_unit='10^3 km^3',
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
    if matched_diff_dicts:
        matched_obs1_metric = None
        if isinstance(obs2_diff_matched, dict):
            m1 = obs2_diff_matched.get('thick1_metric')
            if isinstance(m1, dict):
                matched_obs1_metric = m1
        if matched_obs1_metric is None:
            for diff_payload in matched_diff_dicts:
                if isinstance(diff_payload, dict) and isinstance(diff_payload.get('thick1_metric'), dict):
                    matched_obs1_metric = diff_payload.get('thick1_metric')
                    break
        matched_obs2_metric = None
        if isinstance(obs2_diff_matched, dict):
            m2 = obs2_diff_matched.get('thick2_metric')
            if isinstance(m2, dict):
                matched_obs2_metric = m2
        matched_model_metrics: List[dict] = []
        matched_model_labels: List[str] = []
        for ii, diff_payload in enumerate(matched_diff_dicts):
            if not isinstance(diff_payload, dict):
                continue
            m2 = diff_payload.get('thick2_metric')
            if not isinstance(m2, dict):
                continue
            matched_model_metrics.append(m2)
            matched_model_labels.append(
                matched_model_labels_used[ii]
                if ii < len(matched_model_labels_used)
                else f'model{ii + 1}'
            )
        matched_raw_table = _build_thickness_month_period_raw_table(
            obs1_metric=matched_obs1_metric,
            obs2_metric=matched_obs2_metric,
            model_metrics=matched_model_metrics,
            model_labels=matched_model_labels,
            hemisphere=hemisphere,
            rmse_unit='10^3 km^3',
            obs1_label=f'{obs1_plot_label} (baseline)',
            obs2_label=obs2_plot_label,
        )
        matched_diff_table = _build_thickness_month_period_table(
            obs2_diff=obs2_diff_matched,
            diff_dicts=matched_diff_dicts,
            model_labels=matched_model_labels_used,
            hemisphere=hemisphere,
            rmse_unit='10^3 km^3',
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

    regional_tables: Dict[str, Any] = {'All': all_payload}
    skip_regional_table_expansion = bool(cache_loaded)
    if skip_regional_table_expansion:
        # Keep cache-backed runs responsive: per-sector expansion is expensive
        # and not required for figure generation.
        logger.info("Skip regional %s table expansion in cache-backed run.", module)

    if (not skip_regional_table_expansion) and (not obs_files or not model_files):
        try:
            _, obs_files, model_files = _run_preprocessing(
                case_name, module, recipe, data_dir, frequency='monthly', jobs=jobs
            )
        except Exception as exc:
            logger.warning("Failed to recover processed file list for regional %s tables (%s).", module, exc)
            obs_files, model_files = [], []

    if (not skip_regional_table_expansion) and obs_files and model_files:
        logger.info("Computing regional %s scalar tables ...", module)
        metric = _get_metric(module_vars)
        thick_metrics_region = SIM.ThicknessMetrics(
            grid_file=grid_file, hemisphere=hemisphere,
            year_sta=year_sta, year_end=year_end, metric=metric,
        )
        obs_path = os.path.join(data_dir, obs_files[0])
        obs2_path = os.path.join(data_dir, obs_files[1]) if len(obs_files) > 1 else None
        model_labels_full = _get_recipe_model_labels(module, module_vars, len(model_files))
        sectors = utils.get_hemisphere_sectors(hemisphere, include_all=False)
        obs_key = module_vars.get('ref_var', 'sithick')
        model_key = module_vars.get('model_var', 'sithick')

        for sector in sectors:
            try:
                obs2_diff_sec = None
                obs2_diff_sec_matched = None
                if obs2_path is not None:
                    obs2_diff_sec = thick_metrics_region.Thickness_2M_metrics(
                        obs_path, obs_key,
                        obs2_path, obs_key,
                        sector=sector,
                    )
                    obs2_diff_sec_matched = thick_metrics_region.Thickness_2M_metrics(
                        obs_path, obs_key,
                        obs2_path, obs_key,
                        strict_obs_match=True,
                        obs_match_file=obs2_path,
                        obs_match_key=obs_key,
                        sector=sector,
                    )

                diff_sec: List[dict] = []
                diff_labels: List[str] = []
                matched_sec: List[dict] = []
                matched_labels: List[str] = []
                for ii, mf in enumerate(model_files):
                    model_path = os.path.join(data_dir, mf)
                    model_label = model_labels_full[ii] if ii < len(model_labels_full) else f'model{ii + 1}'
                    d_raw = thick_metrics_region.Thickness_2M_metrics(
                        obs_path, obs_key,
                        model_path, model_key,
                        sector=sector,
                    )
                    d_matched = thick_metrics_region.Thickness_2M_metrics(
                        obs_path, obs_key,
                        model_path, model_key,
                        strict_obs_match=True,
                        obs_match_file=obs2_path,
                        obs_match_key=obs_key,
                        sector=sector,
                    )
                    if d_raw is not None:
                        diff_sec.append(d_raw)
                        diff_labels.append(model_label)
                    if d_matched is not None:
                        matched_sec.append(d_matched)
                        matched_labels.append(model_label)

                if not diff_sec:
                    continue

                base_obs1_metric = None
                if diff_sec and isinstance(diff_sec[0], dict):
                    m1 = diff_sec[0].get('thick1_metric')
                    if isinstance(m1, dict):
                        base_obs1_metric = m1
                base_obs2_metric = None
                if isinstance(obs2_diff_sec, dict):
                    m2 = obs2_diff_sec.get('thick2_metric')
                    if isinstance(m2, dict):
                        base_obs2_metric = m2
                base_model_metrics = [
                    d.get('thick2_metric') for d in diff_sec
                    if isinstance(d, dict) and isinstance(d.get('thick2_metric'), dict)
                ]
                base_raw_sec = _build_thickness_month_period_raw_table(
                    obs1_metric=base_obs1_metric,
                    obs2_metric=base_obs2_metric,
                    model_metrics=base_model_metrics,
                    model_labels=diff_labels,
                    hemisphere=hemisphere,
                    rmse_unit='10^3 km^3',
                    obs1_label=f'{obs1_plot_label} (baseline)',
                    obs2_label=obs2_plot_label,
                )
                base_diff_sec = _build_thickness_month_period_table(
                    obs2_diff=obs2_diff_sec,
                    diff_dicts=diff_sec,
                    model_labels=diff_labels,
                    hemisphere=hemisphere,
                    rmse_unit='10^3 km^3',
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
                                {'id': 'raw', 'title': 'Raw Values', **base_raw_sec},
                                {'id': 'diff', 'title': 'Differences', **base_diff_sec},
                            ],
                        },
                    ],
                }
                if matched_sec:
                    matched_obs1_metric = None
                    if isinstance(obs2_diff_sec_matched, dict):
                        m1 = obs2_diff_sec_matched.get('thick1_metric')
                        if isinstance(m1, dict):
                            matched_obs1_metric = m1
                    if matched_obs1_metric is None and isinstance(matched_sec[0], dict):
                        m1 = matched_sec[0].get('thick1_metric')
                        if isinstance(m1, dict):
                            matched_obs1_metric = m1
                    matched_obs2_metric = None
                    if isinstance(obs2_diff_sec_matched, dict):
                        m2 = obs2_diff_sec_matched.get('thick2_metric')
                        if isinstance(m2, dict):
                            matched_obs2_metric = m2
                    matched_model_metrics = [
                        d.get('thick2_metric') for d in matched_sec
                        if isinstance(d, dict) and isinstance(d.get('thick2_metric'), dict)
                    ]
                    matched_raw_sec = _build_thickness_month_period_raw_table(
                        obs1_metric=matched_obs1_metric,
                        obs2_metric=matched_obs2_metric,
                        model_metrics=matched_model_metrics,
                        model_labels=matched_labels,
                        hemisphere=hemisphere,
                        rmse_unit='10^3 km^3',
                        obs1_label=f'{obs1_plot_label} (baseline)',
                        obs2_label=obs2_plot_label,
                    )
                    matched_diff_sec = _build_thickness_month_period_table(
                        obs2_diff=obs2_diff_sec_matched,
                        diff_dicts=matched_sec,
                        model_labels=matched_labels,
                        hemisphere=hemisphere,
                        rmse_unit='10^3 km^3',
                        obs1_label=f'{obs1_plot_label} (baseline)',
                        obs2_label=obs2_plot_label,
                    )
                    sec_payload['sections'].append(
                        {
                            'id': 'matched',
                            'title': 'Obs-Matched Coverage',
                            'type': 'dual_table',
                            'sections': [
                                {'id': 'raw', 'title': 'Raw Values', **matched_raw_sec},
                                {'id': 'diff', 'title': 'Differences', **matched_diff_sec},
                            ],
                        }
                    )
                regional_tables[sector] = sec_payload
            except Exception as exc:
                logger.warning("Skipping %s regional table for sector '%s' (%s).", module, sector, exc)

    return _build_region_table_payload(
        hemisphere=hemisphere,
        regional_tables=regional_tables,
        payload_type='region_dual_table',
    )

def eval_sndepth(case_name: str, recipe: RR.RecipeReader,
                 data_dir: str, output_dir: str,
                 recalculate: bool = False,
                 jobs: int = 1) -> Optional[dict]:
    """Evaluate snow depth (SNdepth)."""
    module = 'SNdepth'
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
    raw_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'raw_cmap'], 'Blues')
    diff_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'diff_cmap'], 'RdBu_r')
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

    # Check if all expected outputs already exist
    fig_dir = Path(output_dir) / module
    if _check_outputs_exist(module, fig_dir, hemisphere, recalculate=recalculate):
        logger.info(f"{module} evaluation skipped — all outputs exist.")
        return None

    cache_file = _get_metrics_cache_file(case_name, output_dir, hemisphere, module)
    grid_file = _get_eval_grid(case_name, module, hemisphere)
    model_labels = _get_recipe_model_labels(module, module_vars, len(module_vars.get('model_file') or []))
    obs_labels = _get_reference_labels(module_vars, hemisphere)
    obs1_plot_label = obs_labels[0] if len(obs_labels) >= 1 else 'obs1'
    obs2_plot_label = obs_labels[1] if len(obs_labels) >= 2 else 'obs2'
    obs_files: List[str] = []
    model_files: List[str] = []

    obs_1m = None
    obs2_1m = None
    obs2_diff = None
    obs2_diff_matched = None
    diff_dicts: List[dict] = []
    model_labels_used: List[str] = []
    matched_diff_dicts: List[dict] = []
    matched_model_labels_used: List[str] = []
    cache_loaded = False

    if not recalculate:
        cached = _load_module_cache(cache_file, module, hemisphere)
        if cached is not None and cached.get('payload_kind') == module:
            try:
                records = cached.get('records', {})
                obs_1m = records.get('obs_1m')
                obs2_1m = records.get('obs2_1m')
                obs2_diff = records.get('obs2_diff_2m')
                obs2_diff_matched = records.get('obs2_diff_matched_2m')
                diff_records = cached.get('diff_records', [])
                matched_diff_records = cached.get('matched_diff_records', [])
                diff_dicts = [records[r] for r in diff_records if r in records]
                matched_diff_dicts = [records[r] for r in matched_diff_records if r in records]
                model_labels_used = list(cached.get('model_labels_used', []))
                matched_model_labels_used = list(cached.get('matched_model_labels_used', []))
                if not model_labels_used and diff_dicts:
                    model_labels_used = _get_recipe_model_labels(module, module_vars, len(diff_dicts))
                if not matched_model_labels_used and matched_diff_dicts:
                    matched_model_labels_used = _get_recipe_model_labels(
                        module, module_vars, len(matched_diff_dicts)
                    )
                cache_loaded = obs_1m is not None and bool(diff_dicts) and bool(matched_diff_dicts)
                if cache_loaded:
                    logger.info("Loaded %s metrics from cache: %s", module, cache_file)
            except Exception as exc:
                logger.warning("Cache payload for %s is incomplete (%s). Recalculating.", module, exc)
                cache_loaded = False

    if not cache_loaded:
        grid_file, obs_files, model_files = _run_preprocessing(
            case_name, module, recipe, data_dir, frequency='monthly', jobs=jobs
        )

        model_labels = _get_recipe_model_labels(module, module_vars, len(model_files))
        metric = _get_metric(module_vars)
        thick_metrics = SIM.ThicknessMetrics(
            grid_file=grid_file, hemisphere=hemisphere,
            year_sta=year_sta, year_end=year_end, metric=metric,
        )

        obs_key = module_vars.get('ref_var', 'snow_depth')
        (
            obs_1m,
            obs2_1m,
            obs2_diff,
            obs2_diff_matched,
            diff_dicts,
            model_labels_used,
            matched_diff_dicts,
            matched_model_labels_used,
        ) = _compute_thickness_obs_model_metrics(
            thick_metrics, data_dir, obs_files, model_files, model_labels,
            obs_key=obs_key,
            model_key=module_vars.get('model_var', 'sisnthick'),
            module_name=module,
            jobs=jobs,
            stage_dir=_get_stage_dir(case_name, output_dir, hemisphere, module) / 'model_metrics',
        )

        model_1m_dicts = [d.get('thick2_metric') for d in diff_dicts]
        model_records = [f'model_{i}_1m' for i in range(len(model_1m_dicts))]
        diff_records = [f'diff_{i}_2m' for i in range(len(diff_dicts))]
        matched_diff_records = [f'matched_diff_{i}_2m' for i in range(len(matched_diff_dicts))]
        records = {'obs_1m': obs_1m}
        if obs2_1m is not None:
            records['obs2_1m'] = obs2_1m
        if obs2_diff is not None:
            records['obs2_diff_2m'] = obs2_diff
        if obs2_diff_matched is not None:
            records['obs2_diff_matched_2m'] = obs2_diff_matched
        records.update({name: d for name, d in zip(model_records, model_1m_dicts)})
        records.update({name: d for name, d in zip(diff_records, diff_dicts)})
        records.update({name: d for name, d in zip(matched_diff_records, matched_diff_dicts)})

        used_entities: set = set()
        obs1_entity = _unique_entity_name(
            preferred=obs_labels[0] if len(obs_labels) >= 1 else 'Reference_1',
            fallback='Reference_1',
            used=used_entities,
        )
        entity_groups: Dict[str, str] = {'obs_1m': obs1_entity}
        if obs2_1m is not None:
            obs2_entity = _unique_entity_name(
                preferred=obs_labels[1] if len(obs_labels) >= 2 else 'Reference_2',
                fallback='Reference_2',
                used=used_entities,
            )
            entity_groups['obs2_1m'] = obs2_entity
            if obs2_diff is not None:
                entity_groups['obs2_diff_2m'] = _unique_entity_name(
                    preferred=f'{obs1_entity}_vs_{obs2_entity}',
                    fallback='Reference_1_vs_Reference_2',
                    used=used_entities,
                )

        for i, rec_name in enumerate(model_records):
            model_name = model_labels_used[i] if i < len(model_labels_used) else f'{module}_dataset_{i + 1}'
            entity_groups[rec_name] = _unique_entity_name(
                preferred=model_name,
                fallback=model_name,
                used=used_entities,
            )

        for i, rec_name in enumerate(diff_records):
            model_name = model_labels_used[i] if i < len(model_labels_used) else f'{module}_dataset_{i + 1}'
            entity_groups[rec_name] = _unique_entity_name(
                preferred=f'{obs1_entity}_vs_{model_name}',
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
                    'model_labels_used': model_labels_used,
                    'matched_model_labels_used': matched_model_labels_used,
                    'model_records': model_records,
                    'diff_records': diff_records,
                    'matched_diff_records': matched_diff_records,
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
            obs_1m_plot = records.get('obs_1m')
            diff_records_plot = cached_for_plot.get('diff_records', [])
            matched_diff_records_plot = cached_for_plot.get('matched_diff_records', [])
            diff_dicts_plot = [records[r] for r in diff_records_plot if r in records]
            matched_diff_dicts_plot = [records[r] for r in matched_diff_records_plot if r in records]
            if obs_1m_plot is not None and diff_dicts_plot and matched_diff_dicts_plot:
                obs_1m = obs_1m_plot
                obs2_1m = records.get('obs2_1m')
                obs2_diff = records.get('obs2_diff_2m')
                obs2_diff_matched = records.get('obs2_diff_matched_2m')
                diff_dicts = diff_dicts_plot
                matched_diff_dicts = matched_diff_dicts_plot
                model_labels_used = list(cached_for_plot.get('model_labels_used', model_labels_used))
                matched_model_labels_used = list(
                    cached_for_plot.get('matched_model_labels_used', matched_model_labels_used)
                )
                if not model_labels_used and diff_dicts:
                    model_labels_used = _get_recipe_model_labels(module, module_vars, len(diff_dicts))
                if not matched_model_labels_used and matched_diff_dicts:
                    matched_model_labels_used = _get_recipe_model_labels(
                        module, module_vars, len(matched_diff_dicts)
                    )
                logger.info("Using cache-backed %s payload for plotting: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to reload %s cache for plotting (%s). Using in-memory payload.", module, exc)

    # Group-only payloads for dedicated time-series panels (obs + group mean±std).
    base_diff_dicts_for_ts = list(diff_dicts)
    base_model_labels_for_ts = list(model_labels_used)
    base_model_metric_payloads_for_ts = [
        d.get('thick2_metric') if isinstance(d, dict) else None
        for d in base_diff_dicts_for_ts
    ]
    group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=list(base_model_labels_for_ts),
    )
    group_labels_for_ts: List[str] = []
    group_model_metric_means_for_ts: List[Any] = []
    group_model_metric_stds_for_ts: List[Any] = []
    group_diff_means_for_ts: List[Any] = []
    group_diff_stds_for_ts: List[Any] = []
    if group_specs and base_model_metric_payloads_for_ts:
        group_model_metric_means_for_ts, group_model_metric_stds_for_ts, group_labels_for_ts = _build_group_mean_std_payloads(
            model_payloads=base_model_metric_payloads_for_ts,
            model_labels=list(base_model_labels_for_ts),
            group_specs=group_specs,
        )
    if group_specs and base_diff_dicts_for_ts:
        group_diff_means_for_ts, group_diff_stds_for_ts, _gd_labels = _build_group_mean_std_payloads(
            model_payloads=base_diff_dicts_for_ts,
            model_labels=list(base_model_labels_for_ts),
            group_specs=group_specs,
        )

    base_matched_diff_dicts_for_ts = list(matched_diff_dicts)
    base_matched_labels_for_ts = list(matched_model_labels_used)
    base_matched_model_metric_payloads_for_ts = [
        d.get('thick2_metric') if isinstance(d, dict) else None
        for d in base_matched_diff_dicts_for_ts
    ]
    matched_group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=list(base_matched_labels_for_ts),
    )
    matched_group_labels_for_ts: List[str] = []
    matched_group_model_metric_means_for_ts: List[Any] = []
    matched_group_model_metric_stds_for_ts: List[Any] = []
    matched_group_diff_means_for_ts: List[Any] = []
    matched_group_diff_stds_for_ts: List[Any] = []
    if matched_group_specs and base_matched_model_metric_payloads_for_ts:
        matched_group_model_metric_means_for_ts, matched_group_model_metric_stds_for_ts, matched_group_labels_for_ts = _build_group_mean_std_payloads(
            model_payloads=base_matched_model_metric_payloads_for_ts,
            model_labels=list(base_matched_labels_for_ts),
            group_specs=matched_group_specs,
        )
    if matched_group_specs and base_matched_diff_dicts_for_ts:
        matched_group_diff_means_for_ts, matched_group_diff_stds_for_ts, _mgd_labels = _build_group_mean_std_payloads(
            model_payloads=base_matched_diff_dicts_for_ts,
            model_labels=list(base_matched_labels_for_ts),
            group_specs=matched_group_specs,
        )

    if diff_dicts:
        logger.info('Generating figures ...')
        fig_dir = Path(output_dir) / module
        fig_dir.mkdir(parents=True, exist_ok=True)
        key_months = SIM.SeaIceMetricsBase.resolve_key_months(hemisphere)

        obs_pair_base, all_diff_dicts_map_base, all_labels_map_base, all_diff_dicts_base, all_labels_base, model_dicts_base = \
            _build_thickness_label_sets(
                obs_1m, obs2_1m, obs2_diff, base_diff_dicts_for_ts, base_model_labels_for_ts, obs_labels=obs_labels
            )
        obs_pair, all_diff_dicts_map, all_labels_map, all_diff_dicts, all_labels, model_dicts = \
            _build_thickness_label_sets(
                obs_1m, obs2_1m, obs2_diff, diff_dicts, model_labels_used, obs_labels=obs_labels
            )

        with xr.open_dataset(grid_file) as ds:
            _lon, _lat = np.array(ds['lon']), np.array(ds['lat'])

        pf.plot_snd_ts(
            model_dicts_base, all_diff_dicts_map_base, all_labels_map_base, str(fig_dir / 'SND_ts.png'),
            line_style=line_styles, color=line_colors,
        )
        pf.plot_snd_ano(
            model_dicts_base, all_diff_dicts_base, all_labels_base, module_vars['year_range'],
            fig_name=str(fig_dir / 'SND_ano.png'),
            line_style=line_styles, color=line_colors,
        )
        _snd_metric_all = [obs_1m] + ([obs2_1m] if obs2_1m is not None else []) + [d['thick2_metric'] for d in base_diff_dicts_for_ts]
        _snd_labels_all = [obs1_plot_label] + ([obs2_plot_label] if obs2_1m is not None else []) + base_model_labels_for_ts
        for month in key_months:
            mtag = SIM.SeaIceMetricsBase.month_tag(month)
            pf.plot_snd_key_month_ano(
                _snd_metric_all, _snd_labels_all, module_vars['year_range'], month,
                hms=hemisphere, fig_name=str(fig_dir / f'sndepth_ano_timeseries_{mtag}.png'),
                line_style=line_styles, color=line_colors,
            )
            if group_labels_for_ts and group_model_metric_means_for_ts:
                pf.plot_snd_key_month_ano(
                    [obs_1m] + ([obs2_1m] if obs2_1m is not None else []) + group_model_metric_means_for_ts,
                    [obs1_plot_label] + ([obs2_plot_label] if obs2_1m is not None else []) + group_labels_for_ts,
                    module_vars['year_range'], month,
                    hms=hemisphere, fig_name=str(fig_dir / f'sndepth_ano_timeseries_{mtag}_groupmean.png'),
                    line_style=line_styles, color=line_colors,
                    model_spread_payloads=group_model_metric_stds_for_ts,
                )

        if group_labels_for_ts and group_model_metric_means_for_ts and group_diff_means_for_ts:
            pf.plot_snd_ts(
                group_model_metric_means_for_ts,
                ([obs_pair_base] if obs_pair_base else []) + group_diff_means_for_ts,
                [obs1_plot_label, obs2_plot_label][: (2 if obs_pair_base else 1)] + group_labels_for_ts,
                str(fig_dir / 'SND_ts_groupmean.png'),
                line_style=line_styles, color=line_colors,
                model_spread_payloads=group_model_metric_stds_for_ts,
            )
            pf.plot_snd_ano(
                group_model_metric_means_for_ts,
                ([obs2_diff] if obs2_diff else []) + group_diff_means_for_ts,
                [obs1_plot_label] + ([obs2_plot_label] if obs2_diff else []) + group_labels_for_ts,
                module_vars['year_range'],
                fig_name=str(fig_dir / 'SND_ano_groupmean.png'),
                line_style=line_styles, color=line_colors,
                model_spread_payloads=group_model_metric_stds_for_ts,
            )

        _season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
        _snd_metric_maps = [obs_1m] + ([obs2_1m] if obs2_1m is not None else []) + [d['thick2_metric'] for d in diff_dicts]
        _snd_labels_maps = [obs1_plot_label] + ([obs2_plot_label] if obs2_1m is not None else []) + model_labels_used

        def _stack_snd_season_maps(diag_list: List[Dict[str, Any]], key: str) -> Optional[np.ndarray]:
            ref = None
            for dd in diag_list:
                arr = np.asarray(dd.get(key, np.array([])), dtype=float)
                if arr.ndim == 2:
                    ref = arr.shape
                    break
            if ref is None:
                return None
            rows = []
            for dd in diag_list:
                arr = np.asarray(dd.get(key, np.array([])), dtype=float)
                if arr.ndim != 2 or arr.shape != ref:
                    arr = np.full(ref, np.nan, dtype=float)
                rows.append(arr)
            return np.asarray(rows, dtype=float)

        for _season in _season_names:
            _diags = [_thickness_seasonal_diagnostics(mdict, _season, hemisphere) for mdict in _snd_metric_maps]
            _clim_stack = _stack_snd_season_maps(_diags, 'clim_map')
            if _clim_stack is not None:
                _clim_p95 = np.nanpercentile(np.abs(_clim_stack[0]), 95) if np.any(np.isfinite(_clim_stack[0])) else np.nan
                _clim_max = max(0.02, round(float(_clim_p95) * 20) / 20) if np.isfinite(_clim_p95) else 0.5
                pf.plot_SIC_map(
                    grid_file, _clim_stack, _snd_labels_maps, hemisphere,
                    sic_range=[0, _clim_max], diff_range=[-_clim_max * 0.5, _clim_max * 0.5],
                    sic_cm=raw_cmap, diff_cm=diff_cmap, unit='m',
                    plot_mode='raw', fig_name=str(fig_dir / f'SND_seasonal_clim_{_season}_raw.png'),
                )
                pf.plot_SIC_map(
                    grid_file, _clim_stack, _snd_labels_maps, hemisphere,
                    sic_range=[0, _clim_max], diff_range=[-_clim_max * 0.5, _clim_max * 0.5],
                    sic_cm=raw_cmap, diff_cm=diff_cmap, unit='m',
                    plot_mode='diff', fig_name=str(fig_dir / f'SND_seasonal_clim_{_season}_diff.png'),
                )

            _std_stack = _stack_snd_season_maps(_diags, 'std_map')
            if _std_stack is not None:
                _std_p95 = np.nanpercentile(_std_stack[0], 95) if np.any(np.isfinite(_std_stack[0])) else np.nan
                _std_max = max(0.01, round(float(_std_p95) * 100) / 100) if np.isfinite(_std_p95) else 0.2
                pf.plot_SIC_map(
                    grid_file, _std_stack, _snd_labels_maps, hemisphere,
                    sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                    sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                    plot_mode='raw', fig_name=str(fig_dir / f'SND_seasonal_std_{_season}_raw.png'),
                )
                pf.plot_SIC_map(
                    grid_file, _std_stack, _snd_labels_maps, hemisphere,
                    sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                    sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                    plot_mode='diff', fig_name=str(fig_dir / f'SND_seasonal_std_{_season}_diff.png'),
                )

            _tr_stack = _stack_snd_season_maps(_diags, 'trend_map')
            _trp_stack = _stack_snd_season_maps(_diags, 'trend_p_map')
            if _tr_stack is not None and _trp_stack is not None:
                _tr_p95 = np.nanpercentile(np.abs(_tr_stack[0]), 95) if np.any(np.isfinite(_tr_stack[0])) else np.nan
                _tr_max = max(0.01, round(float(_tr_p95) * 100) / 100) if np.isfinite(_tr_p95) else 0.2
                pf.plot_trend_map(
                    grid_file, _tr_stack, _trp_stack, _snd_labels_maps, hemisphere,
                    trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/decade',
                    plot_mode='raw', fig_name=str(fig_dir / f'SND_seasonal_trend_{_season}_raw.png'),
                )
                pf.plot_trend_map(
                    grid_file, _tr_stack, _trp_stack, _snd_labels_maps, hemisphere,
                    trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/decade',
                    plot_mode='diff', fig_name=str(fig_dir / f'SND_seasonal_trend_{_season}_diff.png'),
                )

        _ref_mons = obs_1m.get('uni_mon') if obs_1m is not None else None
        for _mon, _mon_str in [(2, 'Feb'), (9, 'Sep')]:
            if _ref_mons is not None and _mon not in _ref_mons:
                continue
            pf.plot_snd_map(_lon, _lat, all_diff_dicts_map, _mon, hemisphere,
                            data_range=[0, 0.5], diff_range=[-0.3, 0.3], model_labels=all_labels_map,
                            cm=raw_cmap, diff_cm=diff_cmap,
                            plot_mode='raw', fig_name=str(fig_dir / f'SND_map_{_mon_str}_raw.png'))
            pf.plot_snd_map(_lon, _lat, all_diff_dicts_map, _mon, hemisphere,
                            data_range=[0, 0.5], diff_range=[-0.3, 0.3], model_labels=all_labels_map,
                            cm=raw_cmap, diff_cm=diff_cmap,
                            plot_mode='diff', fig_name=str(fig_dir / f'SND_map_{_mon_str}_diff.png'))

        _obs_1m = obs_1m if obs_1m is not None else diff_dicts[0]['thick1_metric']
        _obs2_1m = obs2_1m if obs2_1m is not None else (obs2_diff['thick2_metric'] if obs2_diff is not None else None)
        _snd_std_maps = np.array([_obs_1m['thick_ano_std']] + ([_obs2_1m['thick_ano_std']] if _obs2_1m is not None else []) + [d['thick2_metric']['thick_ano_std'] for d in diff_dicts])
        _std_max = max(0.02, round(float(np.nanpercentile(_snd_std_maps[0], 95)) * 100) / 100)
        pf.plot_SIC_map(grid_file, _snd_std_maps, all_labels_map, hemisphere,
                        sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                        sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                        plot_mode='raw', fig_name=str(fig_dir / 'SND_std_map_raw.png'))
        pf.plot_SIC_map(grid_file, _snd_std_maps, all_labels_map, hemisphere,
                        sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                        sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                        plot_mode='diff', fig_name=str(fig_dir / 'SND_std_map_diff.png'))

        _snd_tr_maps = np.array([_obs_1m['thick_ano_tr']] + ([_obs2_1m['thick_ano_tr']] if _obs2_1m is not None else []) + [d['thick2_metric']['thick_ano_tr'] for d in diff_dicts])
        _snd_tr_pval = np.array([_obs_1m['thick_ano_tr_p']] + ([_obs2_1m['thick_ano_tr_p']] if _obs2_1m is not None else []) + [d['thick2_metric']['thick_ano_tr_p'] for d in diff_dicts])
        _tr_max = max(0.01, round(float(np.nanpercentile(np.abs(_snd_tr_maps[0]), 95)) * 100) / 100)
        pf.plot_trend_map(grid_file, _snd_tr_maps, _snd_tr_pval, all_labels_map, hemisphere,
                          trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/decade',
                          plot_mode='raw', fig_name=str(fig_dir / 'SND_trend_map_raw.png'))
        pf.plot_trend_map(grid_file, _snd_tr_maps, _snd_tr_pval, all_labels_map, hemisphere,
                          trend_range=[-_tr_max, _tr_max], cmap=trend_cmap, unit='m/decade',
                          plot_mode='diff', fig_name=str(fig_dir / 'SND_trend_map_diff.png'))

        _snd_keys = ['thick_mean_diff', 'thick_std_diff', 'thick_trend_diff', 'Vol_mean_diff', 'Vol_std_diff', 'Vol_trend_diff']
        _snd_heat = np.array([[d.get(k, float('nan')) for k in _snd_keys] for d in diff_dicts])
        with _PLOT_LOCK:
            _fig, _ax = plt.subplots(
                figsize=(
                    max(8, len(_snd_keys) * 0.8),
                    max(3, 1 + len(diff_dicts)) * max(0.1, heat_main_height_scale),
                )
            )
            if obs2_diff is not None:
                _obs2_row = np.array([abs(obs2_diff.get(k, float('nan'))) for k in _snd_keys])
                pf.plot_heat_map(
                    _snd_heat, model_labels_used, _snd_keys, ax=_ax,
                    cbarlabel='Ratio to obs uncertainty', obs_row=_obs2_row,
                    obs_row_label=obs2_plot_label,
                    ratio_vmin=heat_ratio_vmin, ratio_vmax=heat_ratio_vmax,
                    cmap=heatmap_cmap,
                )
            else:
                pf.plot_heat_map(_snd_heat, model_labels_used, _snd_keys, ax=_ax, cbarlabel='m', cmap=heatmap_cmap)
            _fig.tight_layout()
            pf._save_fig(str(fig_dir / 'heat_map.png'), close=False)
            plt.close(_fig)

        _snd_season_core = ['MeanDiff', 'StdDiff', 'TrendDiff', 'RMSE']
        _snd_season_keys: List[str] = []
        for _season in _season_names:
            _snd_season_keys.extend([
                f'{_season}_{_kk}' for _kk in _snd_season_core
            ])
        _snd_heat_season = []
        for _dct in diff_dicts:
            _row: List[float] = []
            for _season in _season_names:
                _md, _sd, _td, _vmd, _vsd, _vtd, _cc, _rr = _thickness_extract_domain_extended_stats(
                    _dct, ('season', _season), hemisphere
                )
                _row.extend([_md, _sd, _td, _rr])
            _snd_heat_season.append(_row)
        _snd_heat_season = np.asarray(_snd_heat_season, dtype=float)
        if _snd_heat_season.size > 0:
            with _PLOT_LOCK:
                _n_metric = len(_snd_season_core)
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
                if obs2_diff is not None:
                    _obs2_row = []
                    for _season in _season_names:
                        _md, _sd, _td, _vmd, _vsd, _vtd, _cc, _rr = _thickness_extract_domain_extended_stats(
                            obs2_diff, ('season', _season), hemisphere
                        )
                        _obs2_row.extend([_md, _sd, _td, _rr])
                    _obs2_row = np.asarray(_obs2_row, dtype=float)
                else:
                    _obs2_row = None

                for _ii, _season in enumerate(_season_names):
                    _ax = _axes_flat[_ii]
                    _j0 = _ii * _n_metric
                    _j1 = _j0 + _n_metric
                    _im = pf.plot_heat_map(
                        _snd_heat_season[:, _j0:_j1],
                        model_labels_used,
                        _snd_season_core,
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
                    _snd_extend = pf._resolve_colorbar_extend(
                        _first_im,
                        data=np.abs(_snd_heat_season),
                        vmin=heat_ratio_vmin,
                        vmax=heat_ratio_vmax,
                        extend='auto',
                    )
                    try:
                        _snd_max = float(np.nanmax(np.abs(_snd_heat_season)))
                    except Exception:
                        _snd_max = np.nan
                    _tol = max(1e-12, 1e-8 * max(1.0, abs(float(heat_ratio_vmax))))
                    if np.isfinite(_snd_max) and _snd_max >= (float(heat_ratio_vmax) - _tol):
                        if _snd_extend == 'neither':
                            _snd_extend = 'max'
                        elif _snd_extend == 'min':
                            _snd_extend = 'both'
                    _cb = _fig.colorbar(
                        _first_im,
                        ax=_used_axes,
                        orientation='vertical',
                        fraction=max(0.001, heat_season_cbar_fraction),
                        pad=max(0.0, heat_season_cbar_pad),
                        aspect=max(1.0, heat_season_cbar_aspect),
                        extend=_snd_extend,
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

        if matched_diff_dicts:
            if obs2_diff_matched is not None:
                _obs_pair_matched = {
                    'thick1_metric': obs2_diff_matched.get('thick1_metric'),
                    'thick2_metric': obs2_diff_matched.get('thick2_metric'),
                }
                _all_diff_dicts_map_matched = [_obs_pair_matched] + base_matched_diff_dicts_for_ts
                _all_labels_map_matched = [obs1_plot_label, obs2_plot_label] + base_matched_labels_for_ts
                _all_diff_dicts_matched = [obs2_diff_matched] + base_matched_diff_dicts_for_ts
                _all_labels_matched = [obs1_plot_label, obs2_plot_label] + base_matched_labels_for_ts
                _model_dicts_matched = [d['thick2_metric'] for d in base_matched_diff_dicts_for_ts]
            else:
                (
                    _obs_pair_matched,
                    _all_diff_dicts_map_matched,
                    _all_labels_map_matched,
                    _all_diff_dicts_matched,
                    _all_labels_matched,
                    _model_dicts_matched,
                ) = _build_thickness_label_sets(
                    obs_1m, None, None, base_matched_diff_dicts_for_ts, base_matched_labels_for_ts, obs_labels=obs_labels
                )

            pf.plot_snd_ts(
                _model_dicts_matched,
                _all_diff_dicts_map_matched,
                _all_labels_map_matched,
                str(fig_dir / 'SND_ts_matched.png'),
                line_style=line_styles, color=line_colors,
            )
            pf.plot_snd_ano(
                _model_dicts_matched,
                _all_diff_dicts_matched,
                _all_labels_matched,
                module_vars['year_range'],
                fig_name=str(fig_dir / 'SND_ano_matched.png'),
                line_style=line_styles, color=line_colors,
            )
            if obs2_diff_matched is not None:
                _snd_metric_all_matched = [obs2_diff_matched['thick1_metric'], obs2_diff_matched['thick2_metric']] + [d['thick2_metric'] for d in base_matched_diff_dicts_for_ts]
                _snd_labels_all_matched = [obs1_plot_label, obs2_plot_label] + base_matched_labels_for_ts
            else:
                _snd_metric_all_matched = [obs_1m] + [d['thick2_metric'] for d in base_matched_diff_dicts_for_ts]
                _snd_labels_all_matched = [obs1_plot_label] + base_matched_labels_for_ts
            for month in key_months:
                mtag = SIM.SeaIceMetricsBase.month_tag(month)
                pf.plot_snd_key_month_ano(
                    _snd_metric_all_matched,
                    _snd_labels_all_matched,
                    module_vars['year_range'],
                    month,
                    hms=hemisphere,
                    fig_name=str(fig_dir / f'sndepth_ano_timeseries_{mtag}_matched.png'),
                    line_style=line_styles, color=line_colors,
                )
                if matched_group_labels_for_ts and matched_group_model_metric_means_for_ts:
                    pf.plot_snd_key_month_ano(
                        [obs2_diff_matched['thick1_metric'], obs2_diff_matched['thick2_metric']] + matched_group_model_metric_means_for_ts
                        if obs2_diff_matched is not None else
                        [obs_1m] + matched_group_model_metric_means_for_ts,
                        [obs1_plot_label, obs2_plot_label][: (2 if obs2_diff_matched is not None else 1)] + matched_group_labels_for_ts,
                        module_vars['year_range'],
                        month,
                        hms=hemisphere,
                        fig_name=str(fig_dir / f'sndepth_ano_timeseries_{mtag}_matched_groupmean.png'),
                        line_style=line_styles, color=line_colors,
                        model_spread_payloads=matched_group_model_metric_stds_for_ts,
                    )

            if matched_group_labels_for_ts and matched_group_model_metric_means_for_ts and matched_group_diff_means_for_ts:
                pf.plot_snd_ts(
                    matched_group_model_metric_means_for_ts,
                    ([_obs_pair_matched] if _obs_pair_matched is not None else []) + matched_group_diff_means_for_ts,
                    [obs1_plot_label, obs2_plot_label][: (2 if _obs_pair_matched is not None else 1)] + matched_group_labels_for_ts,
                    str(fig_dir / 'SND_ts_matched_groupmean.png'),
                    line_style=line_styles, color=line_colors,
                    model_spread_payloads=matched_group_model_metric_stds_for_ts,
                )
                pf.plot_snd_ano(
                    matched_group_model_metric_means_for_ts,
                    ([obs2_diff_matched] if obs2_diff_matched is not None else []) + matched_group_diff_means_for_ts,
                    [obs1_plot_label] + ([obs2_plot_label] if obs2_diff_matched is not None else []) + matched_group_labels_for_ts,
                    module_vars['year_range'],
                    fig_name=str(fig_dir / 'SND_ano_matched_groupmean.png'),
                    line_style=line_styles, color=line_colors,
                    model_spread_payloads=matched_group_model_metric_stds_for_ts,
                )

            _obs_ref_for_months = (
                obs2_diff_matched['thick1_metric']
                if obs2_diff_matched is not None else obs_1m
            )
            _ref_mons_matched = _obs_ref_for_months.get('uni_mon') if _obs_ref_for_months is not None else None
            for _mon, _mon_str in [(2, 'Feb'), (9, 'Sep')]:
                if _ref_mons_matched is not None and _mon not in _ref_mons_matched:
                    continue
                pf.plot_snd_map(
                    _lon, _lat, _all_diff_dicts_map_matched, _mon, hemisphere,
                    data_range=[0, 0.5], diff_range=[-0.3, 0.3], model_labels=_all_labels_map_matched,
                    cm=raw_cmap, diff_cm=diff_cmap,
                    plot_mode='raw', fig_name=str(fig_dir / f'SND_map_{_mon_str}_matched_raw.png'),
                )
                pf.plot_snd_map(
                    _lon, _lat, _all_diff_dicts_map_matched, _mon, hemisphere,
                    data_range=[0, 0.5], diff_range=[-0.3, 0.3], model_labels=_all_labels_map_matched,
                    cm=raw_cmap, diff_cm=diff_cmap,
                    plot_mode='diff', fig_name=str(fig_dir / f'SND_map_{_mon_str}_matched_diff.png'),
                )

            _obs_1m_matched = (
                obs2_diff_matched['thick1_metric']
                if obs2_diff_matched is not None else matched_diff_dicts[0]['thick1_metric']
            )
            _snd_std_maps_matched = np.array(
                ([_obs_1m_matched['thick_ano_std'], obs2_diff_matched['thick2_metric']['thick_ano_std']]
                 if obs2_diff_matched is not None else [_obs_1m_matched['thick_ano_std']])
                + [d['thick2_metric']['thick_ano_std'] for d in matched_diff_dicts]
            )
            _std_max_matched = max(0.02, round(float(np.nanpercentile(_snd_std_maps_matched[0], 95)) * 100) / 100)
            pf.plot_SIC_map(
                grid_file, _snd_std_maps_matched, _all_labels_map_matched, hemisphere,
                sic_range=[0, _std_max_matched], diff_range=[-_std_max_matched * 0.5, _std_max_matched * 0.5],
                sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                plot_mode='raw', fig_name=str(fig_dir / 'SND_std_map_matched_raw.png'),
            )
            pf.plot_SIC_map(
                grid_file, _snd_std_maps_matched, _all_labels_map_matched, hemisphere,
                sic_range=[0, _std_max_matched], diff_range=[-_std_max_matched * 0.5, _std_max_matched * 0.5],
                sic_cm=std_raw_cmap, diff_cm=std_diff_cmap, unit='m',
                plot_mode='diff', fig_name=str(fig_dir / 'SND_std_map_matched_diff.png'),
            )

            _snd_tr_maps_matched = np.array(
                ([_obs_1m_matched['thick_ano_tr'], obs2_diff_matched['thick2_metric']['thick_ano_tr']]
                 if obs2_diff_matched is not None else [_obs_1m_matched['thick_ano_tr']])
                + [d['thick2_metric']['thick_ano_tr'] for d in matched_diff_dicts]
            )
            _snd_tr_pval_matched = np.array(
                ([_obs_1m_matched['thick_ano_tr_p'], obs2_diff_matched['thick2_metric']['thick_ano_tr_p']]
                 if obs2_diff_matched is not None else [_obs_1m_matched['thick_ano_tr_p']])
                + [d['thick2_metric']['thick_ano_tr_p'] for d in matched_diff_dicts]
            )
            _tr_max_matched = max(0.01, round(float(np.nanpercentile(np.abs(_snd_tr_maps_matched[0]), 95)) * 100) / 100)
            pf.plot_trend_map(
                grid_file, _snd_tr_maps_matched, _snd_tr_pval_matched, _all_labels_map_matched, hemisphere,
                trend_range=[-_tr_max_matched, _tr_max_matched], cmap=trend_cmap, unit='m/decade',
                plot_mode='raw', fig_name=str(fig_dir / 'SND_trend_map_matched_raw.png'),
            )
            pf.plot_trend_map(
                grid_file, _snd_tr_maps_matched, _snd_tr_pval_matched, _all_labels_map_matched, hemisphere,
                trend_range=[-_tr_max_matched, _tr_max_matched], cmap=trend_cmap, unit='m/decade',
                plot_mode='diff', fig_name=str(fig_dir / 'SND_trend_map_matched_diff.png'),
            )

            _snd_heat_matched = np.array([[d.get(k, float('nan')) for k in _snd_keys] for d in matched_diff_dicts])
            with _PLOT_LOCK:
                _fig, _ax = plt.subplots(
                    figsize=(
                        max(8, len(_snd_keys) * 0.8),
                        max(3, 1 + len(matched_diff_dicts)) * max(0.1, heat_main_height_scale),
                    )
                )
                if obs2_diff_matched is not None:
                    _obs2_row = np.array([abs(obs2_diff_matched.get(k, float('nan'))) for k in _snd_keys])
                    pf.plot_heat_map(
                        _snd_heat_matched, matched_model_labels_used, _snd_keys, ax=_ax,
                        cbarlabel='Ratio to obs uncertainty',
                        obs_row=_obs2_row, obs_row_label=obs2_plot_label,
                        ratio_vmin=heat_ratio_vmin, ratio_vmax=heat_ratio_vmax,
                        cmap=heatmap_cmap,
                    )
                else:
                    pf.plot_heat_map(
                        _snd_heat_matched, matched_model_labels_used, _snd_keys,
                        ax=_ax, cbarlabel='m', cmap=heatmap_cmap,
                    )
                _fig.tight_layout()
                pf._save_fig(str(fig_dir / 'heat_map_matched.png'), close=False)
                plt.close(_fig)

        try:
            pf.plot_sic_region_map(
                grid_nc_file=grid_file,
                hms=hemisphere,
                fig_name=str(fig_dir / 'SeaIceRegion_map.png'),
            )
        except Exception as exc:
            logger.warning("Failed to generate sea-ice region map (%s).", exc)

    logger.info('%s evaluation completed.', module)
    if not diff_dicts:
        return None

    base_raw_table = _build_thickness_month_period_raw_table(
        obs1_metric=obs_1m,
        obs2_metric=obs2_1m,
        model_metrics=[
            d.get('thick2_metric')
            for d in diff_dicts
            if isinstance(d, dict) and isinstance(d.get('thick2_metric'), dict)
        ],
        model_labels=model_labels_used,
        hemisphere=hemisphere,
        rmse_unit='10^3 km^3',
        obs1_label=f'{obs1_plot_label} (baseline)',
        obs2_label=obs2_plot_label,
    )
    base_diff_table = _build_thickness_month_period_table(
        obs2_diff=obs2_diff,
        diff_dicts=diff_dicts,
        model_labels=model_labels_used,
        hemisphere=hemisphere,
        rmse_unit='10^3 km^3',
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
    if matched_diff_dicts:
        matched_obs1_metric = None
        if isinstance(obs2_diff_matched, dict):
            m1 = obs2_diff_matched.get('thick1_metric')
            if isinstance(m1, dict):
                matched_obs1_metric = m1
        if matched_obs1_metric is None:
            for diff_payload in matched_diff_dicts:
                if isinstance(diff_payload, dict) and isinstance(diff_payload.get('thick1_metric'), dict):
                    matched_obs1_metric = diff_payload.get('thick1_metric')
                    break
        matched_obs2_metric = None
        if isinstance(obs2_diff_matched, dict):
            m2 = obs2_diff_matched.get('thick2_metric')
            if isinstance(m2, dict):
                matched_obs2_metric = m2
        matched_model_metrics: List[dict] = []
        matched_model_labels: List[str] = []
        for ii, diff_payload in enumerate(matched_diff_dicts):
            if not isinstance(diff_payload, dict):
                continue
            m2 = diff_payload.get('thick2_metric')
            if not isinstance(m2, dict):
                continue
            matched_model_metrics.append(m2)
            matched_model_labels.append(
                matched_model_labels_used[ii]
                if ii < len(matched_model_labels_used)
                else f'model{ii + 1}'
            )
        matched_raw_table = _build_thickness_month_period_raw_table(
            obs1_metric=matched_obs1_metric,
            obs2_metric=matched_obs2_metric,
            model_metrics=matched_model_metrics,
            model_labels=matched_model_labels,
            hemisphere=hemisphere,
            rmse_unit='10^3 km^3',
            obs1_label=f'{obs1_plot_label} (baseline)',
            obs2_label=obs2_plot_label,
        )
        matched_diff_table = _build_thickness_month_period_table(
            obs2_diff=obs2_diff_matched,
            diff_dicts=matched_diff_dicts,
            model_labels=matched_model_labels_used,
            hemisphere=hemisphere,
            rmse_unit='10^3 km^3',
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

    regional_tables: Dict[str, Any] = {'All': all_payload}
    skip_regional_table_expansion = bool(cache_loaded)
    if skip_regional_table_expansion:
        # Keep cache-backed runs responsive: per-sector expansion is expensive
        # and not required for figure generation.
        logger.info("Skip regional %s table expansion in cache-backed run.", module)

    if (not skip_regional_table_expansion) and (not obs_files or not model_files):
        try:
            _, obs_files, model_files = _run_preprocessing(
                case_name, module, recipe, data_dir, frequency='monthly', jobs=jobs
            )
        except Exception as exc:
            logger.warning("Failed to recover processed file list for regional %s tables (%s).", module, exc)
            obs_files, model_files = [], []

    if (not skip_regional_table_expansion) and obs_files and model_files:
        logger.info("Computing regional %s scalar tables ...", module)
        metric = _get_metric(module_vars)
        thick_metrics_region = SIM.ThicknessMetrics(
            grid_file=grid_file, hemisphere=hemisphere,
            year_sta=year_sta, year_end=year_end, metric=metric,
        )
        obs_path = os.path.join(data_dir, obs_files[0])
        obs2_path = os.path.join(data_dir, obs_files[1]) if len(obs_files) > 1 else None
        model_labels_full = _get_recipe_model_labels(module, module_vars, len(model_files))
        sectors = utils.get_hemisphere_sectors(hemisphere, include_all=False)
        obs_key = module_vars.get('ref_var', 'sndepth')
        model_key = module_vars.get('model_var', 'sndepth')

        for sector in sectors:
            try:
                obs2_diff_sec = None
                obs2_diff_sec_matched = None
                if obs2_path is not None:
                    obs2_diff_sec = thick_metrics_region.Thickness_2M_metrics(
                        obs_path, obs_key,
                        obs2_path, obs_key,
                        sector=sector,
                    )
                    obs2_diff_sec_matched = thick_metrics_region.Thickness_2M_metrics(
                        obs_path, obs_key,
                        obs2_path, obs_key,
                        strict_obs_match=True,
                        obs_match_file=obs2_path,
                        obs_match_key=obs_key,
                        sector=sector,
                    )

                diff_sec: List[dict] = []
                diff_labels: List[str] = []
                matched_sec: List[dict] = []
                matched_labels: List[str] = []
                for ii, mf in enumerate(model_files):
                    model_path = os.path.join(data_dir, mf)
                    model_label = model_labels_full[ii] if ii < len(model_labels_full) else f'model{ii + 1}'
                    d_raw = thick_metrics_region.Thickness_2M_metrics(
                        obs_path, obs_key,
                        model_path, model_key,
                        sector=sector,
                    )
                    d_matched = thick_metrics_region.Thickness_2M_metrics(
                        obs_path, obs_key,
                        model_path, model_key,
                        strict_obs_match=True,
                        obs_match_file=obs2_path,
                        obs_match_key=obs_key,
                        sector=sector,
                    )
                    if d_raw is not None:
                        diff_sec.append(d_raw)
                        diff_labels.append(model_label)
                    if d_matched is not None:
                        matched_sec.append(d_matched)
                        matched_labels.append(model_label)

                if not diff_sec:
                    continue

                base_obs1_metric = None
                if diff_sec and isinstance(diff_sec[0], dict):
                    m1 = diff_sec[0].get('thick1_metric')
                    if isinstance(m1, dict):
                        base_obs1_metric = m1
                base_obs2_metric = None
                if isinstance(obs2_diff_sec, dict):
                    m2 = obs2_diff_sec.get('thick2_metric')
                    if isinstance(m2, dict):
                        base_obs2_metric = m2
                base_model_metrics = [
                    d.get('thick2_metric') for d in diff_sec
                    if isinstance(d, dict) and isinstance(d.get('thick2_metric'), dict)
                ]
                base_raw_sec = _build_thickness_month_period_raw_table(
                    obs1_metric=base_obs1_metric,
                    obs2_metric=base_obs2_metric,
                    model_metrics=base_model_metrics,
                    model_labels=diff_labels,
                    hemisphere=hemisphere,
                    rmse_unit='10^3 km^3',
                    obs1_label=f'{obs1_plot_label} (baseline)',
                    obs2_label=obs2_plot_label,
                )
                base_diff_sec = _build_thickness_month_period_table(
                    obs2_diff=obs2_diff_sec,
                    diff_dicts=diff_sec,
                    model_labels=diff_labels,
                    hemisphere=hemisphere,
                    rmse_unit='10^3 km^3',
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
                                {'id': 'raw', 'title': 'Raw Values', **base_raw_sec},
                                {'id': 'diff', 'title': 'Differences', **base_diff_sec},
                            ],
                        },
                    ],
                }
                if matched_sec:
                    matched_obs1_metric = None
                    if isinstance(obs2_diff_sec_matched, dict):
                        m1 = obs2_diff_sec_matched.get('thick1_metric')
                        if isinstance(m1, dict):
                            matched_obs1_metric = m1
                    if matched_obs1_metric is None and isinstance(matched_sec[0], dict):
                        m1 = matched_sec[0].get('thick1_metric')
                        if isinstance(m1, dict):
                            matched_obs1_metric = m1
                    matched_obs2_metric = None
                    if isinstance(obs2_diff_sec_matched, dict):
                        m2 = obs2_diff_sec_matched.get('thick2_metric')
                        if isinstance(m2, dict):
                            matched_obs2_metric = m2
                    matched_model_metrics = [
                        d.get('thick2_metric') for d in matched_sec
                        if isinstance(d, dict) and isinstance(d.get('thick2_metric'), dict)
                    ]
                    matched_raw_sec = _build_thickness_month_period_raw_table(
                        obs1_metric=matched_obs1_metric,
                        obs2_metric=matched_obs2_metric,
                        model_metrics=matched_model_metrics,
                        model_labels=matched_labels,
                        hemisphere=hemisphere,
                        rmse_unit='10^3 km^3',
                        obs1_label=f'{obs1_plot_label} (baseline)',
                        obs2_label=obs2_plot_label,
                    )
                    matched_diff_sec = _build_thickness_month_period_table(
                        obs2_diff=obs2_diff_sec_matched,
                        diff_dicts=matched_sec,
                        model_labels=matched_labels,
                        hemisphere=hemisphere,
                        rmse_unit='10^3 km^3',
                        obs1_label=f'{obs1_plot_label} (baseline)',
                        obs2_label=obs2_plot_label,
                    )
                    sec_payload['sections'].append(
                        {
                            'id': 'matched',
                            'title': 'Obs-Matched Coverage',
                            'type': 'dual_table',
                            'sections': [
                                {'id': 'raw', 'title': 'Raw Values', **matched_raw_sec},
                                {'id': 'diff', 'title': 'Differences', **matched_diff_sec},
                            ],
                        }
                    )
                regional_tables[sector] = sec_payload
            except Exception as exc:
                logger.warning("Skipping %s regional table for sector '%s' (%s).", module, sector, exc)

    return _build_region_table_payload(
        hemisphere=hemisphere,
        regional_tables=regional_tables,
        payload_type='region_dual_table',
    )

__all__ = [
    "_compute_thickness_obs_model_metrics",
    "_build_thickness_label_sets",
    "eval_sithick",
    "eval_sndepth",
]
