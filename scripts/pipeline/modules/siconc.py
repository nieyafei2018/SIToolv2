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


def _siconc_season_months(hemisphere: str) -> Dict[str, Tuple[int, int, int]]:
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


def _siconc_month_weights(months: Tuple[int, ...]) -> np.ndarray:
    return np.asarray([float(DAYS_PER_MONTH[int(m) - 1]) for m in months], dtype=float)


def _siconc_matrix_seasonal_mean(field_mon: np.ndarray, months: Tuple[int, ...]) -> np.ndarray:
    """Weighted seasonal mean map from 12-month climatology fields."""
    arr = np.asarray(field_mon, dtype=float)
    if arr.ndim != 3 or arr.shape[0] < 12:
        return np.array([])
    idx = [int(m) - 1 for m in months if 1 <= int(m) <= arr.shape[0]]
    if not idx:
        return np.array([])
    vals = np.asarray([arr[i] for i in idx], dtype=float)
    w = _siconc_month_weights(tuple(i + 1 for i in idx))
    if np.sum(w) <= 0:
        return np.array([])
    return np.nansum(vals * w[:, None, None], axis=0) / np.sum(w)


def _siconc_season_group_indices(n_time: int, months: Tuple[int, ...]) -> Dict[int, List[int]]:
    """Group monthly indices into season-years."""
    month_tuple = tuple(int(m) for m in months)
    cross_year = (12 in month_tuple and 1 in month_tuple)
    grouped: Dict[int, List[int]] = {}
    for ii in range(int(max(0, n_time))):
        mm = int(ii % 12) + 1
        if mm not in month_tuple:
            continue
        yy = int(ii // 12)
        sy = yy + 1 if (cross_year and mm == 12) else yy
        grouped.setdefault(sy, []).append(ii)
    return grouped


def _siconc_seasonal_mean_series(series: np.ndarray, months: Tuple[int, ...]) -> np.ndarray:
    """Return seasonal mean series (weighted by month lengths)."""
    arr = np.asarray(series, dtype=float).reshape(-1)
    if arr.size <= 0:
        return np.array([], dtype=float)
    grouped = _siconc_season_group_indices(int(arr.size), months)
    out: List[float] = []
    for sy in sorted(grouped.keys()):
        idx = grouped[sy]
        if len(idx) < 2:
            continue
        vals = arr[idx]
        mnums = tuple((int(i) % 12) + 1 for i in idx)
        weights = _siconc_month_weights(mnums)
        valid = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if int(np.sum(valid)) <= 0:
            continue
        vv = vals[valid]
        ww = weights[valid]
        if np.sum(ww) <= 0:
            continue
        out.append(float(np.nansum(vv * ww) / np.sum(ww)))
    return np.asarray(out, dtype=float)


def _siconc_series_skill(ref: np.ndarray, cmp: np.ndarray) -> Tuple[float, float]:
    """Return (Corr, RMSE) between two paired 1-D series."""
    y1 = np.asarray(ref, dtype=float).reshape(-1)
    y2 = np.asarray(cmp, dtype=float).reshape(-1)
    n_use = min(y1.size, y2.size)
    if n_use < 2:
        return np.nan, np.nan
    y1 = y1[:n_use]
    y2 = y2[:n_use]
    valid = np.isfinite(y1) & np.isfinite(y2)
    if int(np.sum(valid)) < 2:
        return np.nan, np.nan
    y1v = y1[valid]
    y2v = y2[valid]
    corr = np.nan
    if np.nanstd(y1v) > 0 and np.nanstd(y2v) > 0:
        corr = float(np.corrcoef(y1v, y2v)[0, 1])
    rmse = float(np.sqrt(np.nanmean((y2v - y1v) ** 2)))
    return float(corr) if np.isfinite(corr) else np.nan, float(rmse) if np.isfinite(rmse) else np.nan


def _siconc_series_trend_per_decade(series: np.ndarray) -> float:
    """Linear trend (per decade) for one annual/seasonal time series."""
    yy = np.asarray(series, dtype=float).reshape(-1)
    if yy.size < 2:
        return np.nan
    xx = np.arange(yy.size, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy)
    if int(np.sum(valid)) < 2:
        return np.nan
    reg = stats.linregress(xx[valid], yy[valid])
    slope = float(reg.slope) if np.isfinite(reg.slope) else np.nan
    return slope * 10.0 if np.isfinite(slope) else np.nan


def _siconc_weighted_monthly_clim_value(monthly_vals: np.ndarray, months: Tuple[int, ...]) -> float:
    """Weighted mean over selected months from a 12-month climatology series."""
    arr = np.asarray(monthly_vals, dtype=float).reshape(-1)
    if arr.size < 12:
        return np.nan
    idx = [int(mm) - 1 for mm in months if 1 <= int(mm) <= 12]
    if not idx:
        return np.nan
    vals = np.asarray([arr[ii] for ii in idx], dtype=float)
    wts = _siconc_month_weights(tuple(int(ii + 1) for ii in idx))
    valid = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if int(np.sum(valid)) <= 0:
        return np.nan
    vals = vals[valid]
    wts = wts[valid]
    if np.sum(wts) <= 0:
        return np.nan
    return float(np.nansum(vals * wts) / np.sum(wts))


def _siconc_seasonal_products(metric_dict: Optional[dict], season: str, hemisphere: str) -> Dict[str, Any]:
    """Build seasonal diagnostics (clim/std/trend maps + SIE/SIA series)."""
    out: Dict[str, Any] = {
        'season': str(season),
        'months': tuple(),
        'clim_map': np.array([]),
        'std_map': np.array([]),
        'trend_map': np.array([]),
        'trend_p_map': np.array([]),
        'SIE_season': np.array([], dtype=float),
        'SIA_season': np.array([], dtype=float),
        'SIE_ano': np.array([], dtype=float),
        'SIA_ano': np.array([], dtype=float),
    }
    if not isinstance(metric_dict, dict):
        return out

    months = _siconc_season_months(hemisphere).get(str(season), tuple())
    if not months:
        return out
    out['months'] = tuple(int(m) for m in months)

    clim_mon = np.asarray(metric_dict.get('siconc_clim', np.array([])), dtype=float)
    out['clim_map'] = _siconc_matrix_seasonal_mean(clim_mon, out['months'])

    ano_field = np.asarray(metric_dict.get('siconc_ano', np.array([])), dtype=float)
    if ano_field.ndim == 3 and ano_field.shape[0] > 0:
        grouped = _siconc_season_group_indices(int(ano_field.shape[0]), out['months'])
        map_ts = []
        for sy in sorted(grouped.keys()):
            idx = grouped.get(sy, [])
            if len(idx) < 2:
                continue
            map_ts.append(np.nanmean(ano_field[idx, :, :], axis=0))
        if map_ts:
            map_ts_arr = np.asarray(map_ts, dtype=float)
            if map_ts_arr.ndim == 3 and map_ts_arr.shape[0] >= 2:
                std_map, tr_map, tr_p = SIM.ThicknessMetrics._detrended_std_and_trend_map(map_ts_arr)
                out['std_map'] = std_map
                out['trend_map'] = tr_map
                out['trend_p_map'] = tr_p

    out['SIE_season'] = _siconc_seasonal_mean_series(
        np.asarray(metric_dict.get('SIE_ts', np.array([])), dtype=float),
        out['months'],
    )
    out['SIA_season'] = _siconc_seasonal_mean_series(
        np.asarray(metric_dict.get('SIA_ts', np.array([])), dtype=float),
        out['months'],
    )
    out['SIE_ano'] = _siconc_seasonal_mean_series(
        np.asarray(metric_dict.get('SIE_ano', np.array([])), dtype=float),
        out['months'],
    )
    out['SIA_ano'] = _siconc_seasonal_mean_series(
        np.asarray(metric_dict.get('SIA_ano', np.array([])), dtype=float),
        out['months'],
    )
    return out


def _siconc_extract_season_pair_stats(
    ref_metric: Optional[dict],
    cmp_metric: Optional[dict],
    pair_diff: Optional[dict],
    season: str,
    hemisphere: str,
) -> Dict[str, float]:
    """Return Lin et al.-style seasonal diff stats for one pair."""
    d1 = _siconc_seasonal_products(ref_metric, season, hemisphere)
    d2 = _siconc_seasonal_products(cmp_metric, season, hemisphere)
    c1 = np.asarray(d1.get('clim_map', np.array([])), dtype=float)
    c2 = np.asarray(d2.get('clim_map', np.array([])), dtype=float)
    s1 = np.asarray(d1.get('std_map', np.array([])), dtype=float)
    s2 = np.asarray(d2.get('std_map', np.array([])), dtype=float)
    t1 = np.asarray(d1.get('trend_map', np.array([])), dtype=float)
    t2 = np.asarray(d2.get('trend_map', np.array([])), dtype=float)
    sie1 = np.asarray(d1.get('SIE_season', np.array([])), dtype=float).reshape(-1)
    sie2 = np.asarray(d2.get('SIE_season', np.array([])), dtype=float).reshape(-1)
    sia1 = np.asarray(d1.get('SIA_season', np.array([])), dtype=float).reshape(-1)
    sia2 = np.asarray(d2.get('SIA_season', np.array([])), dtype=float).reshape(-1)
    sie_ano1 = np.asarray(d1.get('SIE_ano', np.array([])), dtype=float).reshape(-1)
    sie_ano2 = np.asarray(d2.get('SIE_ano', np.array([])), dtype=float).reshape(-1)
    sia_ano1 = np.asarray(d1.get('SIA_ano', np.array([])), dtype=float).reshape(-1)
    sia_ano2 = np.asarray(d2.get('SIA_ano', np.array([])), dtype=float).reshape(-1)

    out = {
        'SIC_MeanDiff': np.nan,
        'SIC_AnoMeanDiff': np.nan,
        'SIC_TrendDiff': np.nan,
        'SIE_MeanDiff': np.nan,
        'SIE_StdDiff': np.nan,
        'SIE_TrendDiff': np.nan,
        'SIA_MeanDiff': np.nan,
        'SIA_StdDiff': np.nan,
        'SIA_TrendDiff': np.nan,
        'IIEE_MeanDiff': np.nan,
    }
    if c1.ndim == 2 and c2.ndim == 2 and c1.shape == c2.shape:
        out['SIC_MeanDiff'] = utils.MatrixDiff(c1, c2, metric='MAE', mask=True)
    if s1.ndim == 2 and s2.ndim == 2 and s1.shape == s2.shape:
        out['SIC_AnoMeanDiff'] = utils.MatrixDiff(s1, s2, metric='MAE', mask=True)
    if t1.ndim == 2 and t2.ndim == 2 and t1.shape == t2.shape:
        out['SIC_TrendDiff'] = utils.MatrixDiff(t1, t2, metric='MAE', mask=True)

    if sie1.size > 0 and sie2.size > 0:
        out['SIE_MeanDiff'] = abs(float(np.nanmean(sie1)) - float(np.nanmean(sie2)))
    if sie_ano1.size > 0 and sie_ano2.size > 0:
        out['SIE_StdDiff'] = abs(float(np.nanstd(sie_ano1)) - float(np.nanstd(sie_ano2)))
        _tr1 = _siconc_series_trend_per_decade(sie_ano1)
        _tr2 = _siconc_series_trend_per_decade(sie_ano2)
        if np.isfinite(_tr1) and np.isfinite(_tr2):
            out['SIE_TrendDiff'] = abs(_tr1 - _tr2)
    if sia1.size > 0 and sia2.size > 0:
        out['SIA_MeanDiff'] = abs(float(np.nanmean(sia1)) - float(np.nanmean(sia2)))
    if sia_ano1.size > 0 and sia_ano2.size > 0:
        out['SIA_StdDiff'] = abs(float(np.nanstd(sia_ano1)) - float(np.nanstd(sia_ano2)))
        _tr1 = _siconc_series_trend_per_decade(sia_ano1)
        _tr2 = _siconc_series_trend_per_decade(sia_ano2)
        if np.isfinite(_tr1) and np.isfinite(_tr2):
            out['SIA_TrendDiff'] = abs(_tr1 - _tr2)

    if isinstance(pair_diff, dict):
        iiee_clim = np.asarray(pair_diff.get('IIEE_clim_diff', np.array([])), dtype=float).reshape(-1)
        months = tuple(int(mm) for mm in d1.get('months', tuple()))
        if iiee_clim.size >= 12 and months:
            out['IIEE_MeanDiff'] = _siconc_weighted_monthly_clim_value(iiee_clim, months)
        else:
            try:
                vv = float(pair_diff.get('IIEE_mean_diff', np.nan))
            except (TypeError, ValueError):
                vv = np.nan
            out['IIEE_MeanDiff'] = vv if np.isfinite(vv) else np.nan
    return out


def eval_sic(case_name: str, recipe: RR.RecipeReader,
             data_dir: str, output_dir: str,
             recalculate: bool = False,
             jobs: int = 1) -> Optional[dict]:
    """Evaluate sea ice concentration (SIconc)."""
    module = 'SIconc'
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

    sic_raw_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'raw_cmap'], 'YlGnBu')
    sic_diff_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'diff_cmap'], 'RdBu_r')
    sic_std_raw_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'std_raw_cmap'], 'Purples')
    sic_std_diff_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'std_diff_cmap'], 'RdBu_r')
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
        plot_opts, module, ['heatmap', 'seasonal_subplot_hspace'], 0.22,
    ))
    heat_season_cbar_fraction = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_colorbar_fraction'], 0.018,
    ))
    heat_season_cbar_pad = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'seasonal_colorbar_pad'], 0.070,
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
        plot_opts, module, ['line', 'key_month_ano_legend_y_gap'], 0.036,
    ))
    iiee_legend_gap = float(_plot_options_get_module(
        plot_opts, module, ['line', 'iiee_ts_legend_y_gap'], 0.040,
    ))

    fig_dir = Path(output_dir) / module
    if _check_outputs_exist(module, fig_dir, hemisphere, recalculate=recalculate):
        logger.info(f"{module} evaluation skipped - all outputs exist.")
        return None

    cache_file = _get_metrics_cache_file(case_name, output_dir, hemisphere, module)
    grid_file = _get_eval_grid(case_name, module, hemisphere)
    year_range = module_vars['year_range']
    key_months = SIM.SeaIceMetricsBase.resolve_key_months(hemisphere)
    obs_key = module_vars.get('ref_var', 'siconc')
    model_key = module_vars.get('model_var', 'siconc')

    obs_dict: Optional[dict] = None
    obs_dict2: Optional[dict] = None
    obs_diff_dict: Optional[dict] = None
    model_dicts: List[dict] = []
    diff_dicts: List[dict] = []
    obs_files: List[str] = []
    model_files: List[str] = []
    obs_paths: List[str] = []
    model_paths: List[str] = []
    cached_obs_files: List[str] = []
    cached_model_files: List[str] = []
    model_labels = _get_recipe_model_labels(module, module_vars, len(module_vars.get('model_file') or []))
    obs_labels = _get_reference_labels(module_vars, hemisphere)
    obs1_plot_label = obs_labels[0] if len(obs_labels) >= 1 else 'obs1'
    obs2_plot_label = obs_labels[1] if len(obs_labels) >= 2 else 'obs2'

    cache_loaded = False
    if not recalculate:
        cached = _load_module_cache(cache_file, module, hemisphere)
        if cached is not None and cached.get('payload_kind') == module:
            try:
                records = cached.get('records', {})
                obs_dict = records['obs1_1m']
                if cached.get('has_obs2'):
                    obs_dict2 = records.get('obs2_1m')
                    obs_diff_dict = records.get('obs2_vs_obs1_2m')
                model_records = cached.get('model_records', [])
                diff_records = cached.get('diff_records', [])
                model_dicts = [records[r] for r in model_records if r in records]
                diff_dicts = [records[r] for r in diff_records if r in records]
                model_labels = cached.get('model_labels', model_labels)
                cached_obs_files = [
                    str(f).strip()
                    for f in (cached.get('obs_files', []) or [])
                    if str(f).strip()
                ]
                cached_model_files = [
                    str(f).strip()
                    for f in (cached.get('model_files', []) or [])
                    if str(f).strip()
                ]
                if not model_labels and model_dicts:
                    model_labels = _get_recipe_model_labels(module, module_vars, len(model_dicts))
                cache_loaded = obs_dict is not None and len(model_dicts) == len(diff_dicts)
                if cache_loaded:
                    logger.info("Loaded %s metrics from cache: %s", module, cache_file)
            except Exception as exc:
                logger.warning("Cache payload for %s is incomplete (%s). Recalculating.", module, exc)
                cache_loaded = False

    if not cache_loaded:
        grid_file, obs_files, model_files = _run_preprocessing(
            case_name, module, recipe, data_dir, frequency='monthly', jobs=jobs
        )
        if not obs_files:
            logger.error("No observation files available for %s - skipping.", module)
            return None

        metric = _get_metric(module_vars)
        model_labels = _get_recipe_model_labels(module, module_vars, len(model_files))
        sic_metrics = SIM.SIconcMetrics(grid_file=grid_file, hemisphere=hemisphere, metric=metric)

        obs_paths = [os.path.join(data_dir, f) for f in obs_files[:2]]
        model_paths = [os.path.join(data_dir, f) for f in model_files]
        obs_dict = sic_metrics.SIC_1M_metrics(obs_paths[0], obs_key)
        obs_dict2 = sic_metrics.SIC_1M_metrics(obs_paths[1], obs_key) if len(obs_paths) > 1 else None

        logger.info("Computing observational uncertainty (obs2 vs obs1) ...")
        obs_diff_dict = sic_metrics.SIC_2M_metrics(
            sic1_file=obs_paths[0], sic1_key=obs_key,
            sic2_file=obs_paths[1], sic2_key=obs_key,
        ) if len(obs_paths) > 1 else None

        # In parallel mode each model payload is staged to disk first, then loaded
        # in order. This keeps worker memory bounded and avoids large shared objects.
        if jobs > 1 and model_files:
            stage_dir = _get_stage_dir(case_name, output_dir, hemisphere, module)

            def _worker(task: Tuple[int, str]) -> Dict[str, Any]:
                idx, mf = task
                local_metrics = SIM.SIconcMetrics(grid_file=grid_file, hemisphere=hemisphere, metric=metric)
                model_path = os.path.join(data_dir, mf)
                payload = {
                    'model_index': idx,
                    'model_1m': local_metrics.SIC_1M_metrics(model_path, model_key),
                    'diff_2m': local_metrics.SIC_2M_metrics(
                        sic1_file=obs_paths[0], sic1_key=obs_key,
                        sic2_file=model_path, sic2_key=model_key,
                    ),
                }
                payload_file = stage_dir / f'model_{idx:04d}.pkl'
                _save_pickle_atomic(payload_file, payload)
                logger.info(
                    "[%s/%s/model-%02d] Staged metrics payload: %s",
                    hemisphere.upper(), module, idx + 1, payload_file.name,
                )
                return {'model_index': idx, 'payload_file': str(payload_file)}

            stage_refs = _parallel_map_ordered(
                items=list(enumerate(model_files)),
                worker_fn=_worker,
                max_workers=jobs,
                task_label=f'{hemisphere}/{module}/model-metrics',
            )
            staged_payloads = _load_staged_payloads(stage_refs)
            model_dicts = [p['model_1m'] for p in staged_payloads]
            diff_dicts = [p['diff_2m'] for p in staged_payloads]
        else:
            model_dicts = [
                sic_metrics.SIC_1M_metrics(os.path.join(data_dir, mf), model_key)
                for mf in model_files
            ]
            diff_dicts = []
            for mf in model_files:
                model_path = os.path.join(data_dir, mf)
                d_dict = sic_metrics.SIC_2M_metrics(
                    sic1_file=obs_paths[0], sic1_key=obs_key,
                    sic2_file=model_path, sic2_key=model_key,
                )
                diff_dicts.append(d_dict)

        model_records = [f'model{i}_1m' for i in range(len(model_dicts))]
        diff_records = [f'model{i}_vs_obs1_2m' for i in range(len(diff_dicts))]
        records = {'obs1_1m': obs_dict}
        records.update({name: d for name, d in zip(model_records, model_dicts)})
        records.update({name: d for name, d in zip(diff_records, diff_dicts)})
        if obs_dict2 is not None:
            records['obs2_1m'] = obs_dict2
        if obs_diff_dict is not None:
            records['obs2_vs_obs1_2m'] = obs_diff_dict

        used_entities: set = set()
        obs1_entity = _unique_entity_name(
            preferred=obs_labels[0] if len(obs_labels) >= 1 else 'Reference_1',
            fallback='Reference_1',
            used=used_entities,
        )
        entity_groups: Dict[str, str] = {'obs1_1m': obs1_entity}
        if obs_dict2 is not None:
            obs2_entity = _unique_entity_name(
                preferred=obs_labels[1] if len(obs_labels) >= 2 else 'Reference_2',
                fallback='Reference_2',
                used=used_entities,
            )
            entity_groups['obs2_1m'] = obs2_entity
            if obs_diff_dict is not None:
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

        for i, rec_name in enumerate(diff_records):
            model_entity = entity_groups.get(model_records[i], f'{module}_dataset_{i + 1}')
            entity_groups[rec_name] = _unique_entity_name(
                preferred=f'{obs1_entity}_vs_{model_entity}',
                fallback=f'Reference_1_vs_{module}_dataset_{i + 1}',
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
                    'has_obs2': obs_dict2 is not None,
                    'model_labels': model_labels,
                    'model_records': model_records,
                    'diff_records': diff_records,
                    'obs_files': list(obs_files[:2]),
                    'model_files': list(model_files),
                },
                records=records,
                entity_groups=entity_groups,
                grid_file=grid_file,
            )
            logger.info("Saved %s metrics to cache: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to save %s cache (%s).", module, exc)

    # Always render figures from the on-disk cache payload.
    cached_for_plot = _load_module_cache(cache_file, module, hemisphere)
    if cached_for_plot is not None and cached_for_plot.get('payload_kind') == module:
        try:
            records = cached_for_plot.get('records', {})
            obs_dict_plot = records.get('obs1_1m')
            model_records_plot = cached_for_plot.get('model_records', [])
            diff_records_plot = cached_for_plot.get('diff_records', [])
            model_dicts_plot = [records[r] for r in model_records_plot if r in records]
            diff_dicts_plot = [records[r] for r in diff_records_plot if r in records]
            has_obs2_plot = bool(cached_for_plot.get('has_obs2'))
            if obs_dict_plot is not None and len(model_dicts_plot) == len(diff_dicts_plot):
                obs_dict = obs_dict_plot
                obs_dict2 = records.get('obs2_1m') if has_obs2_plot else None
                obs_diff_dict = records.get('obs2_vs_obs1_2m') if has_obs2_plot else None
                model_dicts = model_dicts_plot
                diff_dicts = diff_dicts_plot
                model_labels = cached_for_plot.get('model_labels', model_labels) or model_labels
                logger.info("Using cache-backed %s payload for plotting: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to reload %s cache for plotting (%s). Using in-memory payload.", module, exc)

    # Metric-level multi-model group means (computed after per-model metrics exist).
    base_model_dicts_for_ts = list(model_dicts)
    base_diff_dicts_for_ts = list(diff_dicts)
    base_model_labels_for_group = list(model_labels)
    group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=base_model_labels_for_group,
    )
    group_labels: List[str] = []
    group_model_means_for_ts: List[Any] = []
    group_model_stds_for_ts: List[Any] = []
    group_diff_means_for_ts: List[Any] = []
    group_diff_stds_for_ts: List[Any] = []
    group_labels_for_ts: List[str] = []
    if group_specs and base_model_dicts_for_ts:
        group_model_means_for_ts, group_model_stds_for_ts, group_labels_for_ts = _build_group_mean_std_payloads(
            model_payloads=base_model_dicts_for_ts,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
        )
    if group_specs and base_diff_dicts_for_ts:
        group_diff_means_for_ts, group_diff_stds_for_ts, _group_diff_labels = _build_group_mean_std_payloads(
            model_payloads=base_diff_dicts_for_ts,
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
            logger.info(
                "Enabled metric-level group means for %s [%s]: %s",
                module, hemisphere.upper(), ', '.join(group_labels),
            )

    if obs_dict is None:
        return None
    if obs_dict2 is None:
        logger.warning("SIconc plotting currently expects two observation datasets. Skipping figure generation.")
        return None

    if not obs_paths or (not model_paths and model_dicts):
        if cached_obs_files and cached_model_files:
            obs_paths = [os.path.join(data_dir, f) for f in cached_obs_files[:2]]
            model_paths = [os.path.join(data_dir, f) for f in cached_model_files]
            if not all(Path(p).exists() for p in obs_paths + model_paths):
                logger.info(
                    "Cached %s source-file list is stale; recovering from preprocessing.",
                    module,
                )
                obs_paths = []
                model_paths = []

        if not obs_paths or (not model_paths and model_dicts):
            try:
                _, obs_files_reg, model_files_reg = _run_preprocessing(
                    case_name, module, recipe, data_dir, frequency='monthly', jobs=jobs
                )
                obs_paths = [os.path.join(data_dir, f) for f in obs_files_reg[:2]]
                model_paths = [os.path.join(data_dir, f) for f in model_files_reg]
            except Exception as exc:
                logger.warning(
                    "Failed to recover processed file list for regional SIconc stats (%s).",
                    exc,
                )
                obs_paths = []
                model_paths = []

    logger.info("Generating figures ...")
    fig_dir.mkdir(parents=True, exist_ok=True)

    obs_list_for_ts = [obs_dict] + ([obs_dict2] if obs_dict2 is not None else [])
    base_labels_for_ts = [obs1_plot_label] + ([obs2_plot_label] if obs_dict2 is not None else []) + base_model_labels_for_group
    pf.plot_SIC_ts(obs_list_for_ts, base_model_dicts_for_ts,
                   base_labels_for_ts,
                   str(fig_dir / 'SIC_ts.png'),
                   line_style=line_styles, color=line_colors)

    pf.plot_SIC_ano(
        obs_list_for_ts,
        base_model_dicts_for_ts,
        base_labels_for_ts,
        year_range,
        str(fig_dir / 'SIC_ano.png'),
        hms=hemisphere,
        line_style=line_styles,
        color=line_colors,
    )

    for month in key_months:
        mtag = SIM.SeaIceMetricsBase.month_tag(month)
        pf.plot_siconc_key_month_ano(
            obs_list_for_ts,
            base_model_dicts_for_ts,
            base_labels_for_ts,
            year_range=year_range, month=month, hms=hemisphere,
            fig_name=str(fig_dir / f'siconc_ano_timeseries_{mtag}.png'),
            line_style=line_styles,
            color=line_colors,
            legend_y_gap=key_month_legend_gap,
        )

    pf.plot_IIEE_ts(
        base_diff_dicts_for_ts, base_model_labels_for_group,
        str(fig_dir / 'IIEE_ts.png'),
        obs_uncertainty=obs_diff_dict,
        obs_uncertainty_label=f'{obs2_plot_label}-{obs1_plot_label}',
        obs_count=(2 if obs_dict2 is not None else 1),
        line_style=line_styles,
        color=line_colors,
        legend_y_gap=iiee_legend_gap,
    )

    if group_labels_for_ts and group_model_means_for_ts:
        group_labels_obs = [obs1_plot_label] + ([obs2_plot_label] if obs_dict2 is not None else []) + group_labels_for_ts
        pf.plot_SIC_ts(
            obs_list_for_ts,
            group_model_means_for_ts,
            group_labels_obs,
            str(fig_dir / 'SIC_ts_groupmean.png'),
            line_style=line_styles,
            color=line_colors,
            model_spread_payloads=group_model_stds_for_ts,
        )
        pf.plot_SIC_ano(
            obs_list_for_ts,
            group_model_means_for_ts,
            group_labels_obs,
            year_range,
            str(fig_dir / 'SIC_ano_groupmean.png'),
            hms=hemisphere,
            line_style=line_styles,
            color=line_colors,
            model_spread_payloads=group_model_stds_for_ts,
        )
        for month in key_months:
            mtag = SIM.SeaIceMetricsBase.month_tag(month)
            pf.plot_siconc_key_month_ano(
                obs_list_for_ts,
                group_model_means_for_ts,
                group_labels_obs,
                year_range=year_range,
                month=month,
                hms=hemisphere,
                fig_name=str(fig_dir / f'siconc_ano_timeseries_{mtag}_groupmean.png'),
                line_style=line_styles,
                color=line_colors,
                legend_y_gap=key_month_legend_gap,
                model_spread_payloads=group_model_stds_for_ts,
            )
        if group_diff_means_for_ts:
            pf.plot_IIEE_ts(
                group_diff_means_for_ts,
                group_labels_for_ts,
                str(fig_dir / 'IIEE_ts_groupmean.png'),
                obs_uncertainty=obs_diff_dict,
                obs_uncertainty_label=f'{obs2_plot_label}-{obs1_plot_label}',
                obs_count=(2 if obs_dict2 is not None else 1),
                line_style=line_styles,
                color=line_colors,
                legend_y_gap=iiee_legend_gap,
                spread_payloads=group_diff_stds_for_ts,
            )

    sic_maps = np.array(
        [np.nanmean(obs_dict['siconc_clim'], axis=0),
         np.nanmean(obs_dict2['siconc_clim'], axis=0)] +
        [np.nanmean(d['siconc_clim'], axis=0) for d in model_dicts]
    )
    _sic_labels = [obs1_plot_label] + ([obs2_plot_label] if obs_dict2 is not None else []) + model_labels
    pf.plot_SIC_map(
        grid_file, sic_maps, _sic_labels, hemisphere,
        sic_range=[0, 100], diff_range=[-50, 50], unit='Sea Ice Concentration (%)',
        sic_cm=sic_raw_cmap, diff_cm=sic_diff_cmap,
        plot_mode='raw', fig_name=str(fig_dir / 'SIC_map_raw.png'),
    )
    pf.plot_SIC_map(
        grid_file, sic_maps, _sic_labels, hemisphere,
        sic_range=[0, 100], diff_range=[-50, 50], unit='Sea Ice Concentration (%)',
        sic_cm=sic_raw_cmap, diff_cm=sic_diff_cmap,
        plot_mode='diff', fig_name=str(fig_dir / 'SIC_map_diff.png'),
    )

    logger.info("Generating anomaly standard deviation maps ...")
    sic_std_maps = np.array(
        [obs_dict['siconc_ano_std'],
         obs_dict2['siconc_ano_std']] +
        [d['siconc_ano_std'] for d in model_dicts]
    )
    pf.plot_SIC_map(
        grid_file, sic_std_maps, _sic_labels, hemisphere,
        sic_range=[0, 30], diff_range=[-15, 15],
        sic_cm=sic_std_raw_cmap, diff_cm=sic_std_diff_cmap, unit='Std Dev (%)',
        plot_mode='raw', fig_name=str(fig_dir / 'SIC_std_map_raw.png'),
    )
    pf.plot_SIC_map(
        grid_file, sic_std_maps, _sic_labels, hemisphere,
        sic_range=[0, 30], diff_range=[-15, 15],
        sic_cm=sic_std_raw_cmap, diff_cm=sic_std_diff_cmap, unit='Std Dev (%)',
        plot_mode='diff', fig_name=str(fig_dir / 'SIC_std_map_diff.png'),
    )

    logger.info("Generating trend maps ...")
    sic_trend_maps = np.array(
        [obs_dict['siconc_ano_tr'],
         obs_dict2['siconc_ano_tr']] +
        [d['siconc_ano_tr'] for d in model_dicts]
    )
    sic_pvalue_maps = np.array(
        [obs_dict['siconc_ano_tr_p'],
         obs_dict2['siconc_ano_tr_p']] +
        [d['siconc_ano_tr_p'] for d in model_dicts]
    )
    _abs_max = float(np.nanpercentile(np.abs(sic_trend_maps[0]), 95))
    _trend_range = max(5.0, round(_abs_max / 5) * 5)
    pf.plot_trend_map(
        grid_file, sic_trend_maps, sic_pvalue_maps,
        _sic_labels, hemisphere,
        trend_range=[-_trend_range, _trend_range],
        cmap=trend_cmap, unit='%/decade',
        plot_mode='raw', fig_name=str(fig_dir / 'SIC_trend_map_raw.png'),
    )
    pf.plot_trend_map(
        grid_file, sic_trend_maps, sic_pvalue_maps,
        _sic_labels, hemisphere,
        trend_range=[-_trend_range, _trend_range],
        cmap=trend_cmap, unit='%/decade',
        plot_mode='diff', fig_name=str(fig_dir / 'SIC_trend_map_diff.png'),
    )

    for month in key_months:
        mtag = SIM.SeaIceMetricsBase.month_tag(month)

        _clim_maps = np.array(
            [obs_dict[f'siconc_clim_{mtag}'], obs_dict2[f'siconc_clim_{mtag}']] +
            [d[f'siconc_clim_{mtag}'] for d in model_dicts]
        )
        pf.plot_SIC_map(
            grid_file, _clim_maps, _sic_labels, hemisphere,
            sic_range=[0, 100], diff_range=[-50, 50], unit='Sea Ice Concentration (%)',
            sic_cm=sic_raw_cmap, diff_cm=sic_diff_cmap,
            plot_mode='raw', fig_name=str(fig_dir / f'siconc_spatial_clim_{mtag}_{hemisphere}_raw.png'),
        )
        pf.plot_SIC_map(
            grid_file, _clim_maps, _sic_labels, hemisphere,
            sic_range=[0, 100], diff_range=[-50, 50], unit='Sea Ice Concentration (%)',
            sic_cm=sic_raw_cmap, diff_cm=sic_diff_cmap,
            plot_mode='diff', fig_name=str(fig_dir / f'siconc_spatial_clim_{mtag}_{hemisphere}_diff.png'),
        )

        _std_maps = np.array(
            [obs_dict[f'siconc_ano_std_{mtag}'], obs_dict2[f'siconc_ano_std_{mtag}']] +
            [d[f'siconc_ano_std_{mtag}'] for d in model_dicts]
        )
        _std_max = max(1.0, round(float(np.nanpercentile(_std_maps[0], 95)) / 5) * 5)
        pf.plot_SIC_map(
            grid_file, _std_maps, _sic_labels, hemisphere,
            sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
            sic_cm=sic_std_raw_cmap, diff_cm=sic_std_diff_cmap, unit='Std Dev (%)',
            plot_mode='raw', fig_name=str(fig_dir / f'siconc_spatial_std_{mtag}_{hemisphere}_raw.png'),
        )
        pf.plot_SIC_map(
            grid_file, _std_maps, _sic_labels, hemisphere,
            sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
            sic_cm=sic_std_raw_cmap, diff_cm=sic_std_diff_cmap, unit='Std Dev (%)',
            plot_mode='diff', fig_name=str(fig_dir / f'siconc_spatial_std_{mtag}_{hemisphere}_diff.png'),
        )

        _tr_maps = np.array(
            [obs_dict[f'siconc_ano_tr_{mtag}'], obs_dict2[f'siconc_ano_tr_{mtag}']] +
            [d[f'siconc_ano_tr_{mtag}'] for d in model_dicts]
        )
        _tr_pval_maps = np.array(
            [obs_dict[f'siconc_ano_tr_p_{mtag}'], obs_dict2[f'siconc_ano_tr_p_{mtag}']] +
            [d[f'siconc_ano_tr_p_{mtag}'] for d in model_dicts]
        )
        _tr_max = max(1.0, round(float(np.nanpercentile(np.abs(_tr_maps[0]), 95)) / 5) * 5)
        pf.plot_trend_map(
            grid_file, _tr_maps, _tr_pval_maps,
            _sic_labels, hemisphere,
            trend_range=[-_tr_max, _tr_max],
            cmap=trend_cmap, unit='%/decade',
            plot_mode='raw', fig_name=str(fig_dir / f'siconc_spatial_trend_{mtag}_{hemisphere}_raw.png'),
        )
        pf.plot_trend_map(
            grid_file, _tr_maps, _tr_pval_maps,
            _sic_labels, hemisphere,
            trend_range=[-_tr_max, _tr_max],
            cmap=trend_cmap, unit='%/decade',
            plot_mode='diff', fig_name=str(fig_dir / f'siconc_spatial_trend_{mtag}_{hemisphere}_diff.png'),
        )

    _season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
    _sic_metric_maps = [obs_dict] + ([obs_dict2] if obs_dict2 is not None else []) + model_dicts

    def _stack_siconc_season_maps(diag_list: List[Dict[str, Any]], key: str) -> Optional[np.ndarray]:
        arrs = [np.asarray(d.get(key, np.array([])), dtype=float) for d in diag_list]
        if not arrs or any(a.ndim != 2 for a in arrs):
            return None
        shp = arrs[0].shape
        if any(a.shape != shp for a in arrs):
            return None
        return np.asarray(arrs, dtype=float)

    for _season in _season_names:
        _season_diags = [
            _siconc_seasonal_products(mdict, _season, hemisphere)
            for mdict in _sic_metric_maps
        ]
        _clim_maps = _stack_siconc_season_maps(_season_diags, 'clim_map')
        if _clim_maps is not None:
            pf.plot_SIC_map(
                grid_file, _clim_maps, _sic_labels, hemisphere,
                sic_range=[0, 100], diff_range=[-50, 50], unit='Sea Ice Concentration (%)',
                sic_cm=sic_raw_cmap, diff_cm=sic_diff_cmap,
                plot_mode='raw', fig_name=str(fig_dir / f'siconc_spatial_clim_{_season}_{hemisphere}_raw.png'),
            )
            pf.plot_SIC_map(
                grid_file, _clim_maps, _sic_labels, hemisphere,
                sic_range=[0, 100], diff_range=[-50, 50], unit='Sea Ice Concentration (%)',
                sic_cm=sic_raw_cmap, diff_cm=sic_diff_cmap,
                plot_mode='diff', fig_name=str(fig_dir / f'siconc_spatial_clim_{_season}_{hemisphere}_diff.png'),
            )

        _std_maps = _stack_siconc_season_maps(_season_diags, 'std_map')
        if _std_maps is not None:
            _std_max = max(1.0, round(float(np.nanpercentile(_std_maps[0], 95)) / 5) * 5)
            pf.plot_SIC_map(
                grid_file, _std_maps, _sic_labels, hemisphere,
                sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                sic_cm=sic_std_raw_cmap, diff_cm=sic_std_diff_cmap, unit='Std Dev (%)',
                plot_mode='raw', fig_name=str(fig_dir / f'siconc_spatial_std_{_season}_{hemisphere}_raw.png'),
            )
            pf.plot_SIC_map(
                grid_file, _std_maps, _sic_labels, hemisphere,
                sic_range=[0, _std_max], diff_range=[-_std_max * 0.5, _std_max * 0.5],
                sic_cm=sic_std_raw_cmap, diff_cm=sic_std_diff_cmap, unit='Std Dev (%)',
                plot_mode='diff', fig_name=str(fig_dir / f'siconc_spatial_std_{_season}_{hemisphere}_diff.png'),
            )

        _tr_maps = _stack_siconc_season_maps(_season_diags, 'trend_map')
        _tr_pmaps = _stack_siconc_season_maps(_season_diags, 'trend_p_map')
        if _tr_maps is not None and _tr_pmaps is not None:
            _tr_max = max(1.0, round(float(np.nanpercentile(np.abs(_tr_maps[0]), 95)) / 5) * 5)
            pf.plot_trend_map(
                grid_file, _tr_maps, _tr_pmaps,
                _sic_labels, hemisphere,
                trend_range=[-_tr_max, _tr_max],
                cmap=trend_cmap, unit='%/decade',
                plot_mode='raw', fig_name=str(fig_dir / f'siconc_spatial_trend_{_season}_{hemisphere}_raw.png'),
            )
            pf.plot_trend_map(
                grid_file, _tr_maps, _tr_pmaps,
                _sic_labels, hemisphere,
                trend_range=[-_tr_max, _tr_max],
                cmap=trend_cmap, unit='%/decade',
                plot_mode='diff', fig_name=str(fig_dir / f'siconc_spatial_trend_{_season}_{hemisphere}_diff.png'),
            )

    if len(diff_dicts) >= 1 and obs_diff_dict is not None:
        _metric_specs = [
            ('(a) SIC mean diff', 'siconc_mean_diff'),
            ('(b) SIC ano mean diff', 'siconc_std_diff'),
            ('(c) SIC trend mean diff', 'siconc_trend_diff'),
            ('(d) SIE mean diff', 'SIE_mean_diff'),
            ('(e) SIE std diff', 'SIE_std_diff'),
            ('(f) SIE trend diff', 'SIE_trend_diff'),
            ('(g) SIA mean diff', 'SIA_mean_diff'),
            ('(h) SIA std diff', 'SIA_std_diff'),
            ('(i) SIA trend diff', 'SIA_trend_diff'),
            ('(j) IIEE mean diff', 'IIEE_mean_diff'),
        ]
        _metric_labels = [lab for lab, _ in _metric_specs]
        _metric_keys = [key for _, key in _metric_specs]
        _obs2_row = np.array([obs_diff_dict.get(k, float('nan')) for k in _metric_keys], dtype=float)
        _heat_data = np.array([[d.get(k, float('nan')) for k in _metric_keys] for d in diff_dicts], dtype=float)
        _n_rows = 1 + len(diff_dicts)
        with _PLOT_LOCK:
            _fig, _ax = plt.subplots(
                figsize=(
                    max(13.5, len(_metric_keys) * 1.05),
                    max(3, _n_rows) * max(0.1, heat_main_height_scale),
                )
            )
            pf.plot_heat_map(
                _heat_data, model_labels, _metric_labels, ax=_ax,
                cbarlabel='Ratio to obs uncertainty',
                obs_row=_obs2_row, obs_row_label=obs2_plot_label,
                ratio_vmin=heat_ratio_vmin, ratio_vmax=heat_ratio_vmax,
                cmap=heatmap_cmap,
            )
            _fig.tight_layout()
            pf._save_fig(str(fig_dir / 'heat_map.png'), close=False)
            plt.close(_fig)

    if model_dicts and obs_dict2 is not None:
        _season_core = [
            '(a) SIC mean diff',
            '(b) SIC ano mean diff',
            '(c) SIC trend mean diff',
            '(d) SIE mean diff',
            '(e) SIE std diff',
            '(f) SIE trend diff',
            '(g) SIA mean diff',
            '(h) SIA std diff',
            '(i) SIA trend diff',
            '(j) IIEE mean diff',
        ]
        _season_core_keys = [
            'SIC_MeanDiff',
            'SIC_AnoMeanDiff',
            'SIC_TrendDiff',
            'SIE_MeanDiff',
            'SIE_StdDiff',
            'SIE_TrendDiff',
            'SIA_MeanDiff',
            'SIA_StdDiff',
            'SIA_TrendDiff',
            'IIEE_MeanDiff',
        ]

        _heat_data_season = []
        for _ii, _mdict in enumerate(model_dicts):
            _pair_diff = diff_dicts[_ii] if _ii < len(diff_dicts) else None
            _row = []
            for _season in _season_names:
                _st = _siconc_extract_season_pair_stats(
                    obs_dict, _mdict, _pair_diff, _season, hemisphere,
                )
                _row.extend([_st.get(kk, np.nan) for kk in _season_core_keys])
            _heat_data_season.append(_row)
        _heat_data_season = np.asarray(_heat_data_season, dtype=float)

        if _heat_data_season.size > 0:
            _obs2_row = []
            for _season in _season_names:
                _st = _siconc_extract_season_pair_stats(
                    obs_dict, obs_dict2, obs_diff_dict, _season, hemisphere,
                )
                _obs2_row.extend([_st.get(kk, np.nan) for kk in _season_core_keys])
            _obs2_row = np.asarray(_obs2_row, dtype=float)
            with _PLOT_LOCK:
                _n_metric = len(_season_core)
                _n_rows = max(1, 1 + len(model_dicts))
                _panel_w = max(12.2, _n_metric * 0.95)
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
                for _ii, _season in enumerate(_season_names):
                    _ax = _axes_flat[_ii]
                    _j0 = _ii * _n_metric
                    _j1 = _j0 + _n_metric
                    _im = pf.plot_heat_map(
                        _heat_data_season[:, _j0:_j1],
                        model_labels,
                        _season_core,
                        ax=_ax,
                        cbarlabel='',
                        obs_row=_obs2_row[_j0:_j1],
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
                            data=np.abs(_heat_data_season),
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

    try:
        pf.plot_sic_region_map(
            grid_nc_file=grid_file,
            hms=hemisphere,
            fig_name=str(fig_dir / 'SeaIceRegion_map.png'),
        )
    except Exception as exc:
        logger.warning("Failed to generate sea-ice region map (%s).", exc)

    logger.info("%s evaluation completed.", module)
    # Build region-aware scalar summary tables (All + per-sea-sector).
    metric = _get_metric(module_vars)
    sic_metrics_region = SIM.SIconcMetrics(grid_file=grid_file, hemisphere=hemisphere, metric=metric)
    regional_stats: Dict[str, Dict[str, Any]] = {
        'All': {
            'obs1_stats': obs_dict.get('siconc_stats', {}),
            'obs2_stats': obs_dict2.get('siconc_stats', {}) if obs_dict2 is not None else {},
            'model_stats_list': [d.get('siconc_stats', {}) for d in model_dicts],
            'obs2_iiee_period': _build_siconc_iiee_period_stats(obs_diff_dict, hemisphere)
            if isinstance(obs_diff_dict, dict) and obs_diff_dict else {},
            'model_iiee_period_list': [
                _build_siconc_iiee_period_stats(d, hemisphere) if isinstance(d, dict) else {}
                for d in (diff_dicts or [])
            ],
        }
    }

    sectors = utils.get_hemisphere_sectors(hemisphere, include_all=False)
    if obs_paths:
        logger.info(
            "Computing regional SIconc scalar stats for %d sectors ...",
            len(sectors),
        )
        for sector in sectors:
            try:
                obs1_stats = sic_metrics_region.SIC_period_stats(
                    obs_paths[0], obs_key, sector=sector,
                )
                obs2_stats = (
                    sic_metrics_region.SIC_period_stats(obs_paths[1], obs_key, sector=sector)
                    if len(obs_paths) > 1 else {}
                )
                model_stats = [
                    sic_metrics_region.SIC_period_stats(model_path, model_key, sector=sector)
                    for model_path in model_paths
                ]
                obs2_iiee_period: Dict[str, float] = {}
                if len(obs_paths) > 1:
                    try:
                        obs2_diff = sic_metrics_region.SIC_2M_metrics(
                            obs_paths[0], obs_key,
                            obs_paths[1], obs_key,
                            sector=sector,
                        )
                        obs2_iiee_period = _build_siconc_iiee_period_stats(obs2_diff, hemisphere)
                    except Exception as exc:
                        logger.warning(
                            "Failed to compute regional obs2 IIEE for sector '%s' (%s).",
                            sector, exc,
                        )
                        obs2_iiee_period = {}

                model_iiee_period_list: List[Dict[str, float]] = []
                for model_path in model_paths:
                    try:
                        model_diff = sic_metrics_region.SIC_2M_metrics(
                            obs_paths[0], obs_key,
                            model_path, model_key,
                            sector=sector,
                        )
                        model_iiee_period_list.append(_build_siconc_iiee_period_stats(model_diff, hemisphere))
                    except Exception as exc:
                        logger.warning(
                            "Failed to compute regional model IIEE for sector '%s' and '%s' (%s).",
                            sector, model_path, exc,
                        )
                        model_iiee_period_list.append({})

                sector_model_stats = list(model_stats)
                sector_model_iiee = list(model_iiee_period_list)
                if group_specs and sector_model_stats:
                    grouped_stats, grouped_iiee, _grouped_labels, _grouped_names = _build_group_mean_payloads(
                        model_payloads=sector_model_stats,
                        diff_payloads=sector_model_iiee,
                        model_labels=base_model_labels_for_group,
                        group_specs=group_specs,
                    )
                    sector_model_stats = list(grouped_stats)
                    sector_model_iiee = list(grouped_iiee or [])

                regional_stats[sector] = {
                    'obs1_stats': obs1_stats,
                    'obs2_stats': obs2_stats,
                    'model_stats_list': sector_model_stats,
                    'obs2_iiee_period': obs2_iiee_period,
                    'model_iiee_period_list': sector_model_iiee,
                }
            except Exception as exc:
                logger.warning(
                    "Skipping regional scalar stats for sector '%s' (%s).",
                    sector, exc,
                )
    else:
        logger.warning(
            "Regional SIconc scalar stats skipped because processed source files were not available."
        )

    siconc_payload = _build_siconc_region_period_table(
        hemisphere=hemisphere,
        regional_stats=regional_stats,
        model_labels=model_labels,
        obs_labels=obs_labels,
    )
    try:
        ano_keys = ('SIE', 'SIA', 'PIA', 'MIZ')

        def _series_mean(metric_dict: dict, key: str) -> float:
            arr = np.asarray(
                metric_dict.get(f'{key}_ts', metric_dict.get(f'{key}_clim', np.array([]))),
                dtype=float,
            ).reshape(-1)
            if arr.size <= 0:
                return np.nan
            out = float(np.nanmean(arr))
            return out if np.isfinite(out) else np.nan

        def _series_trend(metric_dict: dict, key: str) -> Tuple[float, float]:
            payload = metric_dict.get(f'{key}_ano_tr')
            slope, pval = pf._extract_trend_slope_pvalue(payload)
            slope = float(slope) if np.isfinite(slope) else np.nan
            pval = float(pval) if np.isfinite(pval) else np.nan
            if not np.isfinite(slope):
                # GroupMean payloads are built via numeric NaN-mean on nested dicts.
                # Linregress-like objects (e.g. ``SIE_ano_tr``) are non-numeric and may
                # be dropped during group aggregation, which previously led to NaN trend.
                # Fallback: regress directly from the available anomaly series.
                ano = np.asarray(metric_dict.get(f'{key}_ano', np.array([])), dtype=float).reshape(-1)
                if ano.size >= 2:
                    valid = np.isfinite(ano)
                    if np.sum(valid) >= 2:
                        xx = np.arange(ano.size, dtype=float)
                        reg = stats.linregress(xx[valid], ano[valid])
                        slope = float(reg.slope) if np.isfinite(reg.slope) else np.nan
                        pval = float(reg.pvalue) if np.isfinite(reg.pvalue) else np.nan
            if not np.isfinite(slope):
                return np.nan, pval
            return float(slope) * 12.0 * 10.0, pval

        def _fmt_with_sig(value: float, pval: float) -> str:
            base = _format_siconc_value(value, digits=3)
            if base == 'nan':
                return base
            return f'{base}*' if np.isfinite(pval) and pval < 0.05 else base

        summary_rows: List[List[str]] = []
        all_metrics = [obs_dict] + ([obs_dict2] if obs_dict2 is not None else []) + model_dicts
        all_labels = [obs1_plot_label] + ([obs2_plot_label] if obs_dict2 is not None else []) + model_labels
        for label, metric_dict in zip(all_labels, all_metrics):
            if not isinstance(metric_dict, dict):
                continue
            row = [str(label)]
            for key in ano_keys:
                mean_v = _series_mean(metric_dict, key)
                trend_v, trend_p = _series_trend(metric_dict, key)
                row.extend([
                    _format_siconc_value(mean_v, digits=3),
                    _fmt_with_sig(trend_v, trend_p),
                ])
            summary_rows.append(row)

        if summary_rows:
            siconc_payload.setdefault('extra_tables', []).append({
                'type': 'basic_table',
                'title': 'SIC Anomaly Summary',
                'headers': [
                    'Dataset',
                    'SIE Mean', 'SIE Trend',
                    'SIA Mean', 'SIA Trend',
                    'PIA Mean', 'PIA Trend',
                    'MIZ Mean', 'MIZ Trend',
                ],
                'units': [
                    '',
                    '10⁶ km²', '10⁶ km²/decade',
                    '10⁶ km²', '10⁶ km²/decade',
                    '10⁶ km²', '10⁶ km²/decade',
                    '10⁶ km²', '10⁶ km²/decade',
                ],
                'rows': summary_rows,
            })
    except Exception as exc:
        logger.warning("Failed to build SIconc anomaly summary table (%s).", exc)

    return siconc_payload

__all__ = ["eval_sic"]
