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

def eval_sitrans(case_name: str, recipe: RR.RecipeReader,
                 data_dir: str, output_dir: str,
                 recalculate: bool = False,
                 jobs: int = 1) -> Optional[dict]:
    """Evaluate sea ice transition dates (SItrans)."""
    module = 'SItrans'
    _log_module_header(module)

    module_vars = recipe.variables[module]
    hemisphere = recipe.hemisphere
    year_sta, year_end = _get_year_range(module_vars)
    date_start = f"{year_sta}-01-01"
    date_end = f"{year_end}-12-31"
    plot_opts = _load_plot_options(case_name)
    _apply_plot_runtime_options(plot_opts, module)
    climatology_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'climatology_cmap'], 'viridis')
    std_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'std_cmap'], 'Purples')
    std_diff_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'std_diff_cmap'], 'RdBu_r')
    trend_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'trend_cmap'], 'RdBu_r')
    bias_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'bias_cmap'], 'RdBu_r')
    skill_heatmap_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'skill_heatmap_cmap'], 'RdBu_r')
    heat_ratio_vmin = float(_plot_options_get_module(plot_opts, module, ['heatmap', 'ratio_vmin'], 0.5))
    heat_ratio_vmax = float(_plot_options_get_module(plot_opts, module, ['heatmap', 'ratio_vmax'], 2.0))
    heat_main_height_scale = float(_plot_options_get_module(
        plot_opts, module, ['heatmap', 'main_height_scale'], 0.50,
    ))

    def _to_float_opt(value, default: float) -> float:
        try:
            out = float(value)
            return out if np.isfinite(out) else float(default)
        except Exception:
            return float(default)

    def _to_int_opt(value, default: int, min_val: int) -> int:
        try:
            out = int(round(float(value)))
            return max(int(min_val), out)
        except Exception:
            return max(int(min_val), int(default))

    sitrans_threshold = _to_float_opt(module_vars.get('threshold', 15.0), 15.0)
    sitrans_smooth_window_days = _to_int_opt(module_vars.get('smooth_window_days', 15), 15, 1)
    sitrans_persistence_days = _to_int_opt(module_vars.get('persistence_days', 7), 7, 2)
    sitrans_trend_sig_p = _to_float_opt(module_vars.get('trend_significance_p', 0.05), 0.05)
    if not (0.0 < sitrans_trend_sig_p < 1.0):
        sitrans_trend_sig_p = 0.05

    fig_dir = Path(output_dir) / module
    if _check_outputs_exist(module, fig_dir, hemisphere, recalculate=recalculate):
        logger.info(f"{module} evaluation skipped - all outputs exist.")
        return None

    cache_file = _get_metrics_cache_file(case_name, output_dir, hemisphere, module)
    grid_file = _get_eval_grid(case_name, module, hemisphere)

    metric = _get_metric(module_vars)
    sitr_metrics = SIM.SItransMetrics(
        grid_file=grid_file, date_sta=date_start, date_end=date_end,
        hemisphere=hemisphere, metric=metric,
        threshold=sitrans_threshold,
        smooth_window_days=sitrans_smooth_window_days,
        persistence_days=sitrans_persistence_days,
        trend_sig_p=sitrans_trend_sig_p,
    )

    model_labels = _get_recipe_model_labels(module, module_vars, len(module_vars.get('model_file') or []))
    obs_files: List[str] = []
    model_files: List[str] = []
    obs1_metrics_for_maps = None
    obs2_metrics = None
    model_metrics: List[dict] = []
    model_scalar_rows: List[dict] = []
    cache_loaded = False
    obs_labels = _get_reference_labels(module_vars, hemisphere)
    obs1_plot_label = obs_labels[0] if len(obs_labels) >= 1 else 'Obs1'
    obs2_plot_label = obs_labels[1] if len(obs_labels) >= 2 else 'Obs2'

    def _extract_sitrans_obs_payload(pair_metrics: Optional[dict]) -> Optional[dict]:
        if not isinstance(pair_metrics, dict):
            return None
        required = [
            'advance_clim_obs', 'retreat_clim_obs',
            'advance_std_obs', 'retreat_std_obs',
            'advance_trend_obs', 'retreat_trend_obs',
        ]
        if not all(k in pair_metrics for k in required):
            return None
        payload = {
            'advance_clim': np.array(pair_metrics['advance_clim_obs'], dtype=float),
            'retreat_clim': np.array(pair_metrics['retreat_clim_obs'], dtype=float),
            'advance_std': np.array(pair_metrics['advance_std_obs'], dtype=float),
            'retreat_std': np.array(pair_metrics['retreat_std_obs'], dtype=float),
            'advance_trend': np.array(pair_metrics['advance_trend_obs'], dtype=float),
            'retreat_trend': np.array(pair_metrics['retreat_trend_obs'], dtype=float),
        }
        optional_map_keys = {
            'advance_valid_year_count': 'advance_valid_year_count_obs',
            'retreat_valid_year_count': 'retreat_valid_year_count_obs',
            'advance_trend_sigmask': 'advance_trend_sigmask_obs',
            'retreat_trend_sigmask': 'retreat_trend_sigmask_obs',
            'ice_season_clim': 'ice_season_clim_obs',
            'open_water_clim': 'open_water_clim_obs',
        }
        for out_key, src_key in optional_map_keys.items():
            if src_key in pair_metrics:
                payload[out_key] = np.array(pair_metrics[src_key], dtype=float)

        optional_scalar_keys = {
            'retreat_to_next_advance_corr': 'retreat_to_next_advance_corr_obs',
            'retreat_advance_corr': 'retreat_advance_corr_obs',
        }
        for out_key, src_key in optional_scalar_keys.items():
            if src_key in pair_metrics:
                payload[out_key] = float(pair_metrics[src_key])

        optional_series_keys = {
            'year': 'year',
            'advance_series': 'advance_series_obs',
            'retreat_series': 'retreat_series_obs',
            'ice_season_series': 'ice_season_series_obs',
            'open_water_series': 'open_water_series_obs',
        }
        for out_key, src_key in optional_series_keys.items():
            if src_key in pair_metrics:
                payload[out_key] = np.array(pair_metrics[src_key], dtype=float)
        return payload

    def _extract_sitrans_mod_payload(pair_metrics: Optional[dict]) -> Optional[dict]:
        if not isinstance(pair_metrics, dict):
            return None
        required = [
            'advance_clim_mod', 'retreat_clim_mod',
            'advance_std_mod', 'retreat_std_mod',
            'advance_trend_mod', 'retreat_trend_mod',
        ]
        if not all(k in pair_metrics for k in required):
            return None
        payload = {
            'advance_clim': np.array(pair_metrics['advance_clim_mod'], dtype=float),
            'retreat_clim': np.array(pair_metrics['retreat_clim_mod'], dtype=float),
            'advance_std': np.array(pair_metrics['advance_std_mod'], dtype=float),
            'retreat_std': np.array(pair_metrics['retreat_std_mod'], dtype=float),
            'advance_trend': np.array(pair_metrics['advance_trend_mod'], dtype=float),
            'retreat_trend': np.array(pair_metrics['retreat_trend_mod'], dtype=float),
        }
        optional_map_keys = {
            'advance_valid_year_count': 'advance_valid_year_count_mod',
            'retreat_valid_year_count': 'retreat_valid_year_count_mod',
            'advance_trend_sigmask': 'advance_trend_sigmask_mod',
            'retreat_trend_sigmask': 'retreat_trend_sigmask_mod',
            'ice_season_clim': 'ice_season_clim_mod',
            'open_water_clim': 'open_water_clim_mod',
        }
        for out_key, src_key in optional_map_keys.items():
            if src_key in pair_metrics:
                payload[out_key] = np.array(pair_metrics[src_key], dtype=float)

        optional_scalar_keys = {
            'retreat_to_next_advance_corr': 'retreat_to_next_advance_corr_mod',
            'retreat_advance_corr': 'retreat_advance_corr_mod',
        }
        for out_key, src_key in optional_scalar_keys.items():
            if src_key in pair_metrics:
                payload[out_key] = float(pair_metrics[src_key])

        optional_series_keys = {
            'year': 'year',
            'advance_series': 'advance_series_mod',
            'retreat_series': 'retreat_series_mod',
            'ice_season_series': 'ice_season_series_mod',
            'open_water_series': 'open_water_series_mod',
        }
        for out_key, src_key in optional_series_keys.items():
            if src_key in pair_metrics:
                payload[out_key] = np.array(pair_metrics[src_key], dtype=float)
        return payload

    def _obs_payload_to_plot_dict(payload: Optional[dict]) -> Optional[dict]:
        if not isinstance(payload, dict):
            return None
        required = ['advance_clim', 'retreat_clim', 'advance_std', 'retreat_std', 'advance_trend', 'retreat_trend']
        if not all(k in payload for k in required):
            return None
        return {
            'advance_clim_obs': np.array(payload['advance_clim'], dtype=float),
            'retreat_clim_obs': np.array(payload['retreat_clim'], dtype=float),
            'advance_std_obs': np.array(payload['advance_std'], dtype=float),
            'retreat_std_obs': np.array(payload['retreat_std'], dtype=float),
            'advance_trend_obs': np.array(payload['advance_trend'], dtype=float),
            'retreat_trend_obs': np.array(payload['retreat_trend'], dtype=float),
        }

    def _build_scalar_row(metrics: Dict[str, Any]) -> Dict[str, float]:
        def _g(key: str) -> float:
            val = metrics.get(key, np.nan)
            try:
                vv = float(val)
                return vv if np.isfinite(vv) else np.nan
            except Exception:
                return np.nan

        return {
            'advance_corr': _g('advance_corr'),
            'advance_rmse': _g('advance_rmse'),
            'advance_mean_bias': _g('advance_mean_bias'),
            'advance_trend_bias': _g('advance_trend_bias'),
            'advance_trend_sig_fraction_mod': _g('advance_trend_sig_fraction_mod'),
            'advance_valid_year_mean_mod': _g('advance_valid_year_mean_mod'),
            'retreat_corr': _g('retreat_corr'),
            'retreat_rmse': _g('retreat_rmse'),
            'retreat_mean_bias': _g('retreat_mean_bias'),
            'retreat_trend_bias': _g('retreat_trend_bias'),
            'retreat_trend_sig_fraction_mod': _g('retreat_trend_sig_fraction_mod'),
            'retreat_valid_year_mean_mod': _g('retreat_valid_year_mean_mod'),
            'ice_season_mean_bias': _g('ice_season_mean_bias'),
            'ice_season_rmse': _g('ice_season_rmse'),
            'open_water_mean_bias': _g('open_water_mean_bias'),
            'open_water_rmse': _g('open_water_rmse'),
            'retreat_to_next_advance_corr_mod': _g('retreat_to_next_advance_corr_mod'),
            'retreat_to_next_advance_corr_bias': _g('retreat_to_next_advance_corr_bias'),
        }

    if not recalculate:
        cached = _load_module_cache(cache_file, module, hemisphere)
        if cached is not None and cached.get('payload_kind') == module:
            try:
                records = cached.get('records', {})
                obs1_metrics_for_maps = _obs_payload_to_plot_dict(records.get('obs1_1m'))
                if obs1_metrics_for_maps is None:
                    obs1_metrics_for_maps = records.get('obs1_metrics')
                obs2_metrics = records.get('obs2_metrics')
                model_records = cached.get('model_records', [])
                model_metrics = [records[r] for r in model_records if r in records]
                model_scalar_rows = list(cached.get('model_scalar_rows', []) or [])
                model_labels = cached.get('model_labels', model_labels)
                if not model_labels and model_metrics:
                    model_labels = _get_recipe_model_labels(module, module_vars, len(model_metrics))
                cache_loaded = obs1_metrics_for_maps is not None and bool(model_metrics)
                if cache_loaded:
                    logger.info("Loaded %s metrics from cache: %s", module, cache_file)
            except Exception as exc:
                logger.warning("Cache payload for %s is incomplete (%s). Recalculating.", module, exc)
                cache_loaded = False

    if not cache_loaded:
        grid_file, obs_files, model_files = _run_preprocessing(
            case_name, module, recipe, data_dir, frequency='daily', jobs=jobs
        )
        sitr_metrics = SIM.SItransMetrics(
            grid_file=grid_file, date_sta=date_start, date_end=date_end,
            hemisphere=hemisphere, metric=metric,
            threshold=sitrans_threshold,
            smooth_window_days=sitrans_smooth_window_days,
            persistence_days=sitrans_persistence_days,
            trend_sig_p=sitrans_trend_sig_p,
        )

        model_labels = _get_recipe_model_labels(module, module_vars, len(model_files))

        obs1_metrics_for_maps = None
        obs2_metrics = None
        model_metrics = []
        model_scalar_rows = []

        if obs_files and model_files:
            obs_path = os.path.join(data_dir, obs_files[0])

            if jobs > 1:
                stage_dir = _get_stage_dir(case_name, output_dir, hemisphere, module) / 'model_metrics'
                stage_dir.mkdir(parents=True, exist_ok=True)

                def _worker(task: Tuple[int, str]) -> Dict[str, Any]:
                    idx, mf = task
                    model_path = os.path.join(data_dir, mf)
                    metrics = sitr_metrics.SItrans_2M_metrics(
                        sic_file1=obs_path,
                        sic_file2=model_path,
                        sic_name1=module_vars.get('ref_var', 'siconc'),
                        sic_name2=module_vars.get('model_var', 'siconc'),
                    )
                    payload_file = stage_dir / f'model_{idx:04d}.pkl'
                    _save_pickle_atomic(payload_file, {'metrics': metrics})
                    return {'payload_file': str(payload_file)}

                stage_refs = _parallel_map_ordered(
                    items=list(enumerate(model_files)),
                    worker_fn=_worker,
                    max_workers=jobs,
                    task_label=f'{hemisphere}/{module}/model-metrics',
                )
                staged_payloads = _load_staged_payloads(stage_refs)
                for payload in staged_payloads:
                    metrics = payload['metrics']
                    model_metrics.append(metrics)
                    model_scalar_rows.append(_build_scalar_row(metrics))
            else:
                for mf in model_files:
                    model_path = os.path.join(data_dir, mf)
                    metrics = sitr_metrics.SItrans_2M_metrics(
                        sic_file1=obs_path,
                        sic_file2=model_path,
                        sic_name1=module_vars.get('ref_var', 'siconc'),
                        sic_name2=module_vars.get('model_var', 'siconc'),
                    )
                    model_metrics.append(metrics)
                    model_scalar_rows.append(_build_scalar_row(metrics))

            if model_metrics:
                obs1_metrics_for_maps = _obs_payload_to_plot_dict(
                    _extract_sitrans_obs_payload(model_metrics[0])
                )

            if len(obs_files) >= 2:
                obs2_path = os.path.join(data_dir, obs_files[1])
                obs2_metrics = sitr_metrics.SItrans_2M_metrics(
                    sic_file1=obs_path,
                    sic_file2=obs2_path,
                    sic_name1=module_vars.get('ref_var', 'siconc'),
                    sic_name2=module_vars.get('ref_var', 'siconc'),
                )

        obs1_payload = _extract_sitrans_obs_payload(model_metrics[0]) if model_metrics else None
        obs2_payload = _extract_sitrans_mod_payload(obs2_metrics) if obs2_metrics is not None else None
        model_1m_payloads = [_extract_sitrans_mod_payload(m) for m in model_metrics]
        model_1m_records = [f'model_{i}_1m' for i in range(len(model_1m_payloads))]
        model_records = [f'model_{i}_metrics' for i in range(len(model_metrics))]
        records: Dict[str, Dict[str, Any]] = {}
        if obs1_payload is not None:
            records['obs1_1m'] = obs1_payload
        if obs1_metrics_for_maps is not None:
            records['obs1_metrics'] = obs1_metrics_for_maps
        if obs2_payload is not None:
            records['obs2_1m'] = obs2_payload
        if obs2_metrics is not None:
            records['obs2_metrics'] = obs2_metrics
        records.update({name: d for name, d in zip(model_1m_records, model_1m_payloads) if d is not None})
        records.update({name: d for name, d in zip(model_records, model_metrics)})
        used_entities: set = set()
        obs1_entity = _unique_entity_name(
            preferred=obs_labels[0] if len(obs_labels) >= 1 else 'Reference_1',
            fallback='Reference_1',
            used=used_entities,
        )
        entity_groups: Dict[str, str] = {}
        if 'obs1_1m' in records:
            entity_groups['obs1_1m'] = obs1_entity
        if 'obs1_metrics' in records:
            entity_groups['obs1_metrics'] = _unique_entity_name(
                preferred=f'{obs1_entity}_maps',
                fallback=f'{obs1_entity}_maps',
                used=used_entities,
            )
        if obs2_metrics is not None:
            obs2_entity = _unique_entity_name(
                preferred=obs_labels[1] if len(obs_labels) >= 2 else 'Reference_2',
                fallback='Reference_2',
                used=used_entities,
            )
            if 'obs2_1m' in records:
                entity_groups['obs2_1m'] = obs2_entity
            entity_groups['obs2_metrics'] = _unique_entity_name(
                preferred=f'{obs1_entity}_vs_{obs2_entity}',
                fallback='Reference_1_vs_Reference_2',
                used=used_entities,
            )

        for i, rec_name in enumerate(model_1m_records):
            model_name = model_labels[i] if i < len(model_labels) else f'{module}_dataset_{i + 1}'
            entity_groups[rec_name] = _unique_entity_name(
                preferred=model_name,
                fallback=model_name,
                used=used_entities,
            )

        for i, rec_name in enumerate(model_records):
            model_name = model_labels[i] if i < len(model_labels) else f'{module}_dataset_{i + 1}'
            entity_groups[rec_name] = _unique_entity_name(
                preferred=f'{obs1_entity}_vs_{model_name}',
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
                    'model_labels': model_labels,
                    'model_records': model_records,
                    'model_1m_records': model_1m_records,
                    'model_scalar_rows': model_scalar_rows,
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
            obs1_plot = _obs_payload_to_plot_dict(records.get('obs1_1m'))
            if obs1_plot is None:
                obs1_plot = records.get('obs1_metrics')
            model_records_plot = cached_for_plot.get('model_records', [])
            model_metrics_plot = [records[r] for r in model_records_plot if r in records]
            if obs1_plot is not None and model_metrics_plot:
                obs1_metrics_for_maps = obs1_plot
                obs2_metrics = records.get('obs2_metrics')
                model_metrics = model_metrics_plot
                model_scalar_rows = list(cached_for_plot.get('model_scalar_rows', []) or [])
                model_labels = cached_for_plot.get('model_labels', model_labels) or model_labels
                logger.info("Using cache-backed %s payload for plotting: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to reload %s cache for plotting (%s). Using in-memory payload.", module, exc)

    # Metric-level multi-model group means (computed after per-model metrics exist).
    base_model_labels_for_group = list(model_labels)
    base_model_metrics_for_ts = list(model_metrics)
    group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=base_model_labels_for_group,
    )
    group_labels_for_ts: List[str] = []
    group_model_metric_means_for_ts: List[Any] = []
    group_model_metric_stds_for_ts: List[Any] = []
    if group_specs and base_model_metrics_for_ts:
        group_model_metric_means_for_ts, group_model_metric_stds_for_ts, group_labels_for_ts = _build_group_mean_std_payloads(
            model_payloads=base_model_metrics_for_ts,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
        )

    group_labels_for_maps: List[str] = []
    if group_specs and model_metrics:
        scalar_payloads: Optional[List[dict]]
        if isinstance(model_scalar_rows, list) and len(model_scalar_rows) == len(model_metrics):
            scalar_payloads = model_scalar_rows
        else:
            scalar_payloads = None
        grouped_models, grouped_scalar_rows, grouped_labels, group_labels_for_maps = _build_group_mean_payloads(
            model_payloads=model_metrics,
            diff_payloads=scalar_payloads,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
        )
        model_metrics = list(grouped_models)
        if grouped_scalar_rows is not None:
            model_scalar_rows = list(grouped_scalar_rows)
        model_labels = list(grouped_labels)
        if group_labels_for_maps:
            logger.info(
                "Enabled metric-level group means for %s [%s]: %s",
                module, hemisphere.upper(), ', '.join(group_labels_for_maps),
            )

    if obs1_metrics_for_maps is not None:
        logger.info("Generating figures ...")
        fig_dir = Path(output_dir) / module
        fig_dir.mkdir(parents=True, exist_ok=True)
        stale_pngs = [
            'SeaIceRegion_map_raw.png',
            'SItrans_retreat_advance_relationship_scatter_diff.png',
            'SItrans_model_skill_ranking_bar_diff.png',
            'SItrans_skill_heat_map_diff.png',
            'SItrans_IIEE_all_diff.png',
        ]
        for stale_name in stale_pngs:
            stale_path = fig_dir / stale_name
            if stale_path.exists():
                try:
                    stale_path.unlink()
                except Exception:
                    pass

        participant_labels = [obs1_plot_label]
        advance_clim_fields = [np.array(obs1_metrics_for_maps['advance_clim_obs'], dtype=float)]
        retreat_clim_fields = [np.array(obs1_metrics_for_maps['retreat_clim_obs'], dtype=float)]
        advance_std_fields = [np.array(obs1_metrics_for_maps['advance_std_obs'], dtype=float)]
        retreat_std_fields = [np.array(obs1_metrics_for_maps['retreat_std_obs'], dtype=float)]
        advance_trend_fields = [np.array(obs1_metrics_for_maps['advance_trend_obs'], dtype=float)]
        retreat_trend_fields = [np.array(obs1_metrics_for_maps['retreat_trend_obs'], dtype=float)]
        obs1_adv_sig = obs1_metrics_for_maps.get('advance_trend_sigmask_obs')
        obs1_ret_sig = obs1_metrics_for_maps.get('retreat_trend_sigmask_obs')
        if obs1_adv_sig is None and model_metrics:
            obs1_adv_sig = model_metrics[0].get('advance_trend_sigmask_obs')
        if obs1_ret_sig is None and model_metrics:
            obs1_ret_sig = model_metrics[0].get('retreat_trend_sigmask_obs')
        advance_trend_sigmasks = [None if obs1_adv_sig is None else np.array(obs1_adv_sig, dtype=float)]
        retreat_trend_sigmasks = [None if obs1_ret_sig is None else np.array(obs1_ret_sig, dtype=float)]

        if obs2_metrics is not None:
            participant_labels.append(obs2_plot_label)
            advance_clim_fields.append(np.array(obs2_metrics['advance_clim_mod'], dtype=float))
            retreat_clim_fields.append(np.array(obs2_metrics['retreat_clim_mod'], dtype=float))
            advance_std_fields.append(np.array(obs2_metrics['advance_std_mod'], dtype=float))
            retreat_std_fields.append(np.array(obs2_metrics['retreat_std_mod'], dtype=float))
            advance_trend_fields.append(np.array(obs2_metrics['advance_trend_mod'], dtype=float))
            retreat_trend_fields.append(np.array(obs2_metrics['retreat_trend_mod'], dtype=float))
            advance_trend_sigmasks.append(
                np.array(obs2_metrics['advance_trend_sigmask_mod'], dtype=float)
                if 'advance_trend_sigmask_mod' in obs2_metrics else None
            )
            retreat_trend_sigmasks.append(
                np.array(obs2_metrics['retreat_trend_sigmask_mod'], dtype=float)
                if 'retreat_trend_sigmask_mod' in obs2_metrics else None
            )

        for ii, metrics in enumerate(model_metrics):
            label = model_labels[ii] if ii < len(model_labels) else f'{module}_dataset_{ii + 1}'
            participant_labels.append(label)
            advance_clim_fields.append(np.array(metrics['advance_clim_mod'], dtype=float))
            retreat_clim_fields.append(np.array(metrics['retreat_clim_mod'], dtype=float))
            advance_std_fields.append(np.array(metrics['advance_std_mod'], dtype=float))
            retreat_std_fields.append(np.array(metrics['retreat_std_mod'], dtype=float))
            advance_trend_fields.append(np.array(metrics['advance_trend_mod'], dtype=float))
            retreat_trend_fields.append(np.array(metrics['retreat_trend_mod'], dtype=float))
            advance_trend_sigmasks.append(
                np.array(metrics['advance_trend_sigmask_mod'], dtype=float)
                if 'advance_trend_sigmask_mod' in metrics else None
            )
            retreat_trend_sigmasks.append(
                np.array(metrics['retreat_trend_sigmask_mod'], dtype=float)
                if 'retreat_trend_sigmask_mod' in metrics else None
            )

        pf.advance_climatology_map(
            grid_file, advance_clim_fields, participant_labels, hemisphere,
            cmap=climatology_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='raw',
            fig_name=str(fig_dir / 'SItrans_advance_climatology_all_raw.png'),
        )
        pf.advance_climatology_map(
            grid_file, advance_clim_fields, participant_labels, hemisphere,
            cmap=climatology_cmap, diff_cmap=bias_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='diff',
            fig_name=str(fig_dir / 'SItrans_advance_bias_all_models_diff.png'),
        )
        pf.advance_std_map(
            grid_file, advance_std_fields, participant_labels, hemisphere,
            cmap=std_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='raw',
            fig_name=str(fig_dir / 'SItrans_advance_std_all_raw.png'),
        )
        pf.advance_std_map(
            grid_file, advance_std_fields, participant_labels, hemisphere,
            cmap=std_cmap, diff_cmap=std_diff_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='diff',
            fig_name=str(fig_dir / 'SItrans_advance_std_all_diff.png'),
        )
        pf.advance_trend_map(
            grid_file, advance_trend_fields, participant_labels, hemisphere,
            cmap=trend_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='raw',
            sigmask_list=advance_trend_sigmasks,
            fig_name=str(fig_dir / 'SItrans_advance_trend_all_raw.png'),
        )
        pf.advance_trend_map(
            grid_file, advance_trend_fields, participant_labels, hemisphere,
            cmap=trend_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='diff',
            sigmask_list=advance_trend_sigmasks,
            fig_name=str(fig_dir / 'SItrans_advance_trend_all_diff.png'),
        )
        pf.retreat_climatology_map(
            grid_file, retreat_clim_fields, participant_labels, hemisphere,
            cmap=climatology_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='raw',
            fig_name=str(fig_dir / 'SItrans_retreat_climatology_all_raw.png'),
        )
        pf.retreat_climatology_map(
            grid_file, retreat_clim_fields, participant_labels, hemisphere,
            cmap=climatology_cmap, diff_cmap=bias_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='diff',
            fig_name=str(fig_dir / 'SItrans_retreat_bias_all_models_diff.png'),
        )
        pf.retreat_std_map(
            grid_file, retreat_std_fields, participant_labels, hemisphere,
            cmap=std_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='raw',
            fig_name=str(fig_dir / 'SItrans_retreat_std_all_raw.png'),
        )
        pf.retreat_std_map(
            grid_file, retreat_std_fields, participant_labels, hemisphere,
            cmap=std_cmap, diff_cmap=std_diff_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='diff',
            fig_name=str(fig_dir / 'SItrans_retreat_std_all_diff.png'),
        )
        pf.retreat_trend_map(
            grid_file, retreat_trend_fields, participant_labels, hemisphere,
            cmap=trend_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='raw',
            sigmask_list=retreat_trend_sigmasks,
            fig_name=str(fig_dir / 'SItrans_retreat_trend_all_raw.png'),
        )
        pf.retreat_trend_map(
            grid_file, retreat_trend_fields, participant_labels, hemisphere,
            cmap=trend_cmap,
            n_obs=(2 if obs2_metrics is not None else 1),
            plot_mode='diff',
            sigmask_list=retreat_trend_sigmasks,
            fig_name=str(fig_dir / 'SItrans_retreat_trend_all_diff.png'),
        )

        if model_metrics:
            labels_for_models = [
                model_labels[ii] if ii < len(model_labels) else f'model{ii + 1}'
                for ii in range(len(model_metrics))
            ]

            skill_cols = [
                ('advance_rmse', 'Adv RMSE'),
                ('retreat_rmse', 'Ret RMSE'),
                ('advance_mean_bias', 'Adv Bias'),
                ('retreat_mean_bias', 'Ret Bias'),
                ('advance_trend_bias', 'Adv TrendBias'),
                ('retreat_trend_bias', 'Ret TrendBias'),
                ('ice_season_mean_bias', 'IceSeason Bias'),
                ('open_water_mean_bias', 'OpenWater Bias'),
            ]
            skill_mat = np.array([
                [float(row.get(k, np.nan)) for k, _ in skill_cols]
                for row in model_scalar_rows
            ], dtype=float) if model_scalar_rows else np.array([])
            if isinstance(skill_mat, np.ndarray) and skill_mat.size:
                skill_keys = [k for k, _ in skill_cols]
                with _PLOT_LOCK:
                    _fig, _ax = plt.subplots(
                        figsize=(
                            max(8, len(skill_cols) * 0.95),
                            max(3, 1 + len(labels_for_models)) * max(0.1, heat_main_height_scale),
                        )
                    )
                    if isinstance(obs2_metrics, dict):
                        _obs2_row = np.array(
                            [abs(float(obs2_metrics.get(k, np.nan))) for k in skill_keys],
                            dtype=float,
                        )
                        pf.plot_heat_map(
                            skill_mat,
                            labels_for_models,
                            [name for _, name in skill_cols],
                            ax=_ax,
                            cbarlabel='Ratio to obs uncertainty',
                            obs_row=_obs2_row,
                            obs_row_label=obs2_plot_label,
                            ratio_vmin=heat_ratio_vmin,
                            ratio_vmax=heat_ratio_vmax,
                            cmap=skill_heatmap_cmap,
                        )
                    else:
                        pf.plot_heat_map(
                            skill_mat,
                            labels_for_models,
                            [name for _, name in skill_cols],
                            ax=_ax,
                            cbarlabel='Metric value',
                            cmap=skill_heatmap_cmap,
                        )
                    _fig.tight_layout()
                    pf._save_fig(str(fig_dir / 'SItrans_skill_heat_map.png'), close=False)
                    plt.close(_fig)

        iiee_curves = []
        iiee_labels = []
        days = np.arange(1, 366, 5)
        obs1_adv = np.array(obs1_metrics_for_maps['advance_clim_obs'], dtype=float)
        obs1_ret = np.array(obs1_metrics_for_maps['retreat_clim_obs'], dtype=float)

        def _safe_slug(text: str, fallback: str = 'obs1') -> str:
            raw = str(text or '').strip()
            slug = ''.join(ch.lower() if ch.isalnum() else '_' for ch in raw)
            slug = '_'.join(part for part in slug.split('_') if part)
            return slug or fallback

        iiee_ref_slug = _safe_slug(obs1_plot_label, fallback='obs1')
        iiee_fig_name = f'SItrans_IIEE_all_vs_{iiee_ref_slug}_diff.png'

        if obs2_metrics is not None:
            obs2_adv = np.array(obs2_metrics['advance_clim_mod'], dtype=float)
            obs2_ret = np.array(obs2_metrics['retreat_clim_mod'], dtype=float)
            aO, aU = sitr_metrics.calculate_edge_area_difference(
                obs1_adv, obs2_adv, min_day=1, max_day=365, step=5, is_advance=True,
            )
            rO, rU = sitr_metrics.calculate_edge_area_difference(
                obs1_ret, obs2_ret, min_day=1, max_day=365, step=5, is_advance=False,
            )
            iiee_curves.append({
                'days_range': days,
                'advance_over': aO,
                'advance_under': aU,
                'retreat_over': rO,
                'retreat_under': rU,
            })
            iiee_labels.append(str(obs2_plot_label))

        for ii, metrics in enumerate(model_metrics):
            label = model_labels[ii] if ii < len(model_labels) else f'model{ii + 1}'
            mod_adv = np.array(metrics['advance_clim_mod'], dtype=float)
            mod_ret = np.array(metrics['retreat_clim_mod'], dtype=float)
            aO, aU = sitr_metrics.calculate_edge_area_difference(
                obs1_adv, mod_adv, min_day=1, max_day=365, step=5, is_advance=True,
            )
            rO, rU = sitr_metrics.calculate_edge_area_difference(
                obs1_ret, mod_ret, min_day=1, max_day=365, step=5, is_advance=False,
            )
            iiee_curves.append({
                'days_range': days,
                'advance_over': aO,
                'advance_under': aU,
                'retreat_over': rO,
                'retreat_under': rU,
            })
            iiee_labels.append(str(label))

        if iiee_curves:
            pf.plot_sitrans_iiee_all(
                iiee_curves,
                iiee_labels,
                fig_name=str(fig_dir / iiee_fig_name),
                hms=hemisphere,
            )

    logger.info("%s evaluation completed.", module)

    if model_scalar_rows:
        def _fmt(v, nd=3):
            return 'nan' if v is None or not np.isfinite(v) else f'{float(v):.{nd}f}'

        def _mean_field(payload: Optional[dict], key: str) -> float:
            if not isinstance(payload, dict):
                return np.nan
            arr = np.asarray(payload.get(key, np.array([])), dtype=float)
            if arr.size == 0:
                return np.nan
            vv = float(np.nanmean(arr))
            return vv if np.isfinite(vv) else np.nan

        def _build_phase_raw_payload(obs1_payload: Optional[dict],
                                     obs2_payload: Optional[dict],
                                     model_payloads: List[Optional[dict]],
                                     labels: List[str],
                                     obs1_label: str,
                                     obs2_label: str) -> Dict[str, Any]:
            def _phase_rows(phase_prefix: str) -> List[List[str]]:
                valid_key = f'{phase_prefix}_valid_year_count'
                sig_key = f'{phase_prefix}_trend_sigmask'
                rows: List[List[str]] = []
                rows.append([
                    f'{obs1_label} (baseline)',
                    _fmt(_mean_field(obs1_payload, f'{phase_prefix}_clim'), 2),
                    _fmt(_mean_field(obs1_payload, f'{phase_prefix}_std'), 2),
                    _fmt(_mean_field(obs1_payload, f'{phase_prefix}_trend'), 3),
                    _fmt(_mean_field(obs1_payload, valid_key), 2),
                    _fmt(100.0 * _mean_field(obs1_payload, sig_key), 1),
                    _fmt(_mean_field(obs1_payload, 'ice_season_clim'), 2),
                    _fmt(_mean_field(obs1_payload, 'open_water_clim'), 2),
                    _fmt(_mean_field(obs1_payload, 'retreat_to_next_advance_corr'), 3),
                ])
                if isinstance(obs2_payload, dict):
                    rows.append([
                        str(obs2_label),
                        _fmt(_mean_field(obs2_payload, f'{phase_prefix}_clim'), 2),
                        _fmt(_mean_field(obs2_payload, f'{phase_prefix}_std'), 2),
                        _fmt(_mean_field(obs2_payload, f'{phase_prefix}_trend'), 3),
                        _fmt(_mean_field(obs2_payload, valid_key), 2),
                        _fmt(100.0 * _mean_field(obs2_payload, sig_key), 1),
                        _fmt(_mean_field(obs2_payload, 'ice_season_clim'), 2),
                        _fmt(_mean_field(obs2_payload, 'open_water_clim'), 2),
                        _fmt(_mean_field(obs2_payload, 'retreat_to_next_advance_corr'), 3),
                    ])
                for ii, payload in enumerate(model_payloads):
                    label = labels[ii] if ii < len(labels) else f'model{ii + 1}'
                    rows.append([
                        label,
                        _fmt(_mean_field(payload, f'{phase_prefix}_clim'), 2),
                        _fmt(_mean_field(payload, f'{phase_prefix}_std'), 2),
                        _fmt(_mean_field(payload, f'{phase_prefix}_trend'), 3),
                        _fmt(_mean_field(payload, valid_key), 2),
                        _fmt(100.0 * _mean_field(payload, sig_key), 1),
                        _fmt(_mean_field(payload, 'ice_season_clim'), 2),
                        _fmt(_mean_field(payload, 'open_water_clim'), 2),
                        _fmt(_mean_field(payload, 'retreat_to_next_advance_corr'), 3),
                    ])
                return rows

            return {
                'type': 'phase_table',
                'headers': [
                    'Model/Obs Name', 'Climatology Mean', 'Std Mean', 'Trend Mean',
                    'Valid Years', 'Trend Sig(%)', 'Ice Season Mean', 'Open Water Mean', 'Lag1 Corr',
                ],
                'units': ['', 'days', 'days', 'days/year', 'years', '%', 'days', 'days', '1'],
                'phases': {
                    'Advance': _phase_rows('advance'),
                    'Retreat': _phase_rows('retreat'),
                },
            }

        def _build_phase_diff_payload(obs2_pair_metrics: Optional[dict], scalar_rows: List[dict],
                                      labels: List[str], obs1_label: str, obs2_label: str) -> Dict[str, Any]:
            phase_nd = [3, 2, 2, 3, 1, 2, 2, 2, 2, 3]

            def _fmt_phase_vals(vals: List[float]) -> List[str]:
                return [_fmt(vals[ii], phase_nd[ii]) for ii in range(len(phase_nd))]

            advance_rows: List[List[str]] = [[
                f'{obs1_label} (baseline)',
                _fmt(0.0, 3),
                _fmt(0.0, 2),
                _fmt(0.0, 2),
                _fmt(0.0, 3),
                _fmt(0.0, 1),
                _fmt(0.0, 2),
                _fmt(0.0, 2),
                _fmt(0.0, 2),
                _fmt(0.0, 3),
            ]]
            retreat_rows: List[List[str]] = [[
                f'{obs1_label} (baseline)',
                _fmt(0.0, 3),
                _fmt(0.0, 2),
                _fmt(0.0, 2),
                _fmt(0.0, 3),
                _fmt(0.0, 1),
                _fmt(0.0, 2),
                _fmt(0.0, 2),
                _fmt(0.0, 2),
                _fmt(0.0, 3),
            ]]
            obs2_adv_vals: Optional[List[float]] = None
            obs2_ret_vals: Optional[List[float]] = None

            if obs2_pair_metrics is not None:
                obs2_adv_vals = [
                    obs2_pair_metrics.get('advance_corr', np.nan),
                    obs2_pair_metrics.get('advance_rmse', np.nan),
                    obs2_pair_metrics.get('advance_mean_bias', np.nan),
                    obs2_pair_metrics.get('advance_trend_bias', np.nan),
                    obs2_pair_metrics.get('advance_trend_sig_fraction_mod', np.nan),
                    obs2_pair_metrics.get('advance_valid_year_mean_mod', np.nan),
                    obs2_pair_metrics.get('ice_season_mean_bias', np.nan),
                    obs2_pair_metrics.get('open_water_mean_bias', np.nan),
                    obs2_pair_metrics.get('ice_season_rmse', np.nan),
                    obs2_pair_metrics.get('retreat_to_next_advance_corr_bias', np.nan),
                ]
                obs2_ret_vals = [
                    obs2_pair_metrics.get('retreat_corr', np.nan),
                    obs2_pair_metrics.get('retreat_rmse', np.nan),
                    obs2_pair_metrics.get('retreat_mean_bias', np.nan),
                    obs2_pair_metrics.get('retreat_trend_bias', np.nan),
                    obs2_pair_metrics.get('retreat_trend_sig_fraction_mod', np.nan),
                    obs2_pair_metrics.get('retreat_valid_year_mean_mod', np.nan),
                    obs2_pair_metrics.get('ice_season_mean_bias', np.nan),
                    obs2_pair_metrics.get('open_water_mean_bias', np.nan),
                    obs2_pair_metrics.get('open_water_rmse', np.nan),
                    obs2_pair_metrics.get('retreat_to_next_advance_corr_bias', np.nan),
                ]
                advance_rows.append([
                    str(obs2_label),
                    *_fmt_phase_vals([_obs2_identity_ratio(vv) for vv in obs2_adv_vals]),
                ])
                retreat_rows.append([
                    str(obs2_label),
                    *_fmt_phase_vals([_obs2_identity_ratio(vv) for vv in obs2_ret_vals]),
                ])

            for ii, row in enumerate(scalar_rows):
                label = labels[ii] if ii < len(labels) else f'model{ii + 1}'
                adv_vals = [
                    row.get('advance_corr', np.nan),
                    row.get('advance_rmse', np.nan),
                    row.get('advance_mean_bias', np.nan),
                    row.get('advance_trend_bias', np.nan),
                    row.get('advance_trend_sig_fraction_mod', np.nan),
                    row.get('advance_valid_year_mean_mod', np.nan),
                    row.get('ice_season_mean_bias', np.nan),
                    row.get('open_water_mean_bias', np.nan),
                    row.get('ice_season_rmse', np.nan),
                    row.get('retreat_to_next_advance_corr_bias', np.nan),
                ]
                ret_vals = [
                    row.get('retreat_corr', np.nan),
                    row.get('retreat_rmse', np.nan),
                    row.get('retreat_mean_bias', np.nan),
                    row.get('retreat_trend_bias', np.nan),
                    row.get('retreat_trend_sig_fraction_mod', np.nan),
                    row.get('retreat_valid_year_mean_mod', np.nan),
                    row.get('ice_season_mean_bias', np.nan),
                    row.get('open_water_mean_bias', np.nan),
                    row.get('open_water_rmse', np.nan),
                    row.get('retreat_to_next_advance_corr_bias', np.nan),
                ]
                if obs2_adv_vals is not None:
                    adv_vals = [_calc_obs2_ratio(v, b) for v, b in zip(adv_vals, obs2_adv_vals)]
                if obs2_ret_vals is not None:
                    ret_vals = [_calc_obs2_ratio(v, b) for v, b in zip(ret_vals, obs2_ret_vals)]
                advance_rows.append([
                    label,
                    *_fmt_phase_vals(adv_vals),
                ])
                retreat_rows.append([
                    label,
                    *_fmt_phase_vals(ret_vals),
                ])

            return {
                'type': 'phase_table',
                'headers': [
                    'Model/Obs Name', 'Correlation', 'RMSE', 'Mean Bias', 'Trend Bias',
                    'Trend Sig(%)', 'Valid Years', 'Ice Season Bias', 'Open Water Bias',
                    'Duration RMSE', 'Lag1 Corr Bias',
                ],
                'units': ['', '1', 'days', 'days', 'days/year', '%', 'years', 'days', 'days', 'days', '1'],
                'phases': {
                    'Advance': advance_rows,
                    'Retreat': retreat_rows,
                },
            }

        obs1_raw_payload = _extract_sitrans_obs_payload(model_metrics[0]) if model_metrics else None
        obs2_raw_payload = _extract_sitrans_mod_payload(obs2_metrics) if isinstance(obs2_metrics, dict) else None
        model_raw_payloads = [_extract_sitrans_mod_payload(metrics) for metrics in model_metrics]

        all_payload = {
            'type': 'dual_table',
            'sections': [
                {
                    'id': 'raw',
                    'title': 'Raw Values',
                    **_build_phase_raw_payload(
                        obs1_payload=obs1_raw_payload,
                        obs2_payload=obs2_raw_payload,
                        model_payloads=model_raw_payloads,
                        labels=model_labels,
                        obs1_label=obs1_plot_label,
                        obs2_label=obs2_plot_label,
                    ),
                },
                {
                    'id': 'diff',
                    'title': 'Differences',
                    **_build_phase_diff_payload(
                        obs2_metrics, model_scalar_rows, model_labels,
                        obs1_label=obs1_plot_label,
                        obs2_label=obs2_plot_label,
                    ),
                },
            ],
        }
        regional_tables: Dict[str, Any] = {'All': all_payload}

        if not obs_files or not model_files:
            try:
                _, obs_files, model_files = _run_preprocessing(
                    case_name, module, recipe, data_dir, frequency='daily', jobs=jobs
                )
            except Exception as exc:
                logger.warning("Failed to recover processed file list for regional %s tables (%s).", module, exc)
                obs_files, model_files = [], []

        if obs_files and model_files:
            logger.info("Computing regional %s scalar tables ...", module)
            sitr_metrics_region = SIM.SItransMetrics(
                grid_file=grid_file, date_sta=date_start, date_end=date_end,
                hemisphere=hemisphere, metric=metric,
                threshold=sitrans_threshold,
                smooth_window_days=sitrans_smooth_window_days,
                persistence_days=sitrans_persistence_days,
                trend_sig_p=sitrans_trend_sig_p,
            )
            obs_path = os.path.join(data_dir, obs_files[0])
            obs2_path = os.path.join(data_dir, obs_files[1]) if len(obs_files) > 1 else None
            model_labels_full = _get_recipe_model_labels(module, module_vars, len(model_files))
            regional_group_specs = _resolve_group_mean_specs(
                module=module,
                module_vars=module_vars,
                common_config=recipe.common_config,
                model_labels=model_labels_full,
            )
            regional_group_labels = [
                str(spec.get('label') or _format_group_mean_label(spec.get('name')))
                for spec in regional_group_specs
            ]
            regional_labels_with_groups = regional_group_labels + list(model_labels_full)
            sectors = utils.get_hemisphere_sectors(hemisphere, include_all=False)
            region_timeseries_payloads_base: List[Dict[str, Any]] = []
            region_timeseries_payloads_group: List[Dict[str, Any]] = []

            def _align_model_labels(labels_in: List[str], n_models: int) -> List[str]:
                labels_out = list(labels_in or [])
                if len(labels_out) < int(n_models):
                    for idx in range(len(labels_out), int(n_models)):
                        labels_out.append(f'{module}_dataset_{idx + 1}')
                return labels_out[:int(n_models)]

            def _extract_spread_series(spread_payload: Any, key: str) -> np.ndarray:
                if not isinstance(spread_payload, dict):
                    return np.array([], dtype=float)
                try:
                    return np.array(spread_payload.get(key, np.array([])), dtype=float)
                except Exception:
                    return np.array([], dtype=float)

            base_labels_all = _align_model_labels(base_model_labels_for_group, len(base_model_metrics_for_ts))
            all_source = base_model_metrics_for_ts[0] if base_model_metrics_for_ts else None
            if all_source is not None:
                region_timeseries_payloads_base.append({
                    'region': utils.get_sector_label(hemisphere, 'All'),
                    'years': np.array(all_source.get('year', np.array([])), dtype=float),
                    'advance_obs1': np.array(all_source.get('advance_series_obs', np.array([])), dtype=float),
                    'retreat_obs1': np.array(all_source.get('retreat_series_obs', np.array([])), dtype=float),
                    'advance_obs2': np.array(obs2_metrics.get('advance_series_mod', np.array([])), dtype=float) if isinstance(obs2_metrics, dict) else np.array([]),
                    'retreat_obs2': np.array(obs2_metrics.get('retreat_series_mod', np.array([])), dtype=float) if isinstance(obs2_metrics, dict) else np.array([]),
                    'advance_models': [np.array(mm.get('advance_series_mod', np.array([])), dtype=float) for mm in base_model_metrics_for_ts],
                    'retreat_models': [np.array(mm.get('retreat_series_mod', np.array([])), dtype=float) for mm in base_model_metrics_for_ts],
                    'model_labels': base_labels_all,
                    'obs1_label': obs1_plot_label,
                    'obs2_label': obs2_plot_label,
                })
            if group_labels_for_ts and group_model_metric_means_for_ts and all_source is not None:
                region_timeseries_payloads_group.append({
                    'region': utils.get_sector_label(hemisphere, 'All'),
                    'years': np.array(all_source.get('year', np.array([])), dtype=float),
                    'advance_obs1': np.array(all_source.get('advance_series_obs', np.array([])), dtype=float),
                    'retreat_obs1': np.array(all_source.get('retreat_series_obs', np.array([])), dtype=float),
                    'advance_obs2': np.array(obs2_metrics.get('advance_series_mod', np.array([])), dtype=float) if isinstance(obs2_metrics, dict) else np.array([]),
                    'retreat_obs2': np.array(obs2_metrics.get('retreat_series_mod', np.array([])), dtype=float) if isinstance(obs2_metrics, dict) else np.array([]),
                    'advance_models': [
                        np.array(mm.get('advance_series_mod', np.array([])), dtype=float)
                        if isinstance(mm, dict) else np.array([], dtype=float)
                        for mm in group_model_metric_means_for_ts
                    ],
                    'retreat_models': [
                        np.array(mm.get('retreat_series_mod', np.array([])), dtype=float)
                        if isinstance(mm, dict) else np.array([], dtype=float)
                        for mm in group_model_metric_means_for_ts
                    ],
                    'advance_model_spreads': [
                        _extract_spread_series(std_payload, 'advance_series_mod')
                        for std_payload in group_model_metric_stds_for_ts
                    ],
                    'retreat_model_spreads': [
                        _extract_spread_series(std_payload, 'retreat_series_mod')
                        for std_payload in group_model_metric_stds_for_ts
                    ],
                    'model_labels': list(group_labels_for_ts),
                    'obs1_label': obs1_plot_label,
                    'obs2_label': obs2_plot_label,
                })

            years_ref = np.array(all_source.get('year', np.array([])), dtype=float) if all_source is not None else np.array([])

            def _new_sitr_metrics_region() -> SIM.SItransMetrics:
                return SIM.SItransMetrics(
                    grid_file=grid_file, date_sta=date_start, date_end=date_end,
                    hemisphere=hemisphere, metric=metric,
                    threshold=sitrans_threshold,
                    smooth_window_days=sitrans_smooth_window_days,
                    persistence_days=sitrans_persistence_days,
                    trend_sig_p=sitrans_trend_sig_p,
                )

            def _compute_sector_outputs(
                metric_engine: SIM.SItransMetrics,
                sector_key: str,
            ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
                obs2_metrics_sec = None
                if obs2_path is not None:
                    obs2_metrics_sec = metric_engine.SItrans_2M_metrics(
                        sic_file1=obs_path,
                        sic_file2=obs2_path,
                        sic_name1=module_vars.get('ref_var', 'siconc'),
                        sic_name2=module_vars.get('ref_var', 'siconc'),
                        sector=sector_key,
                    )

                model_rows_sec: List[dict] = []
                model_metrics_sec: List[dict] = []
                for mf in model_files:
                    model_path = os.path.join(data_dir, mf)
                    metrics_sec = metric_engine.SItrans_2M_metrics(
                        sic_file1=obs_path,
                        sic_file2=model_path,
                        sic_name1=module_vars.get('ref_var', 'siconc'),
                        sic_name2=module_vars.get('model_var', 'siconc'),
                        sector=sector_key,
                    )
                    model_metrics_sec.append(metrics_sec)
                    model_rows_sec.append(_build_scalar_row(metrics_sec))

                base_model_metrics_sec = list(model_metrics_sec)
                base_model_rows_sec = list(model_rows_sec)
                base_labels_for_sector = _align_model_labels(model_labels_full, len(base_model_metrics_sec))

                group_metric_means_sec: List[Any] = []
                group_metric_stds_sec: List[Any] = []
                group_labels_sec: List[str] = []
                if regional_group_specs and base_model_metrics_sec:
                    group_metric_means_sec, group_metric_stds_sec, group_labels_sec = _build_group_mean_std_payloads(
                        model_payloads=base_model_metrics_sec,
                        model_labels=base_labels_for_sector,
                        group_specs=regional_group_specs,
                    )

                labels_for_sector = list(base_labels_for_sector)
                if regional_group_specs and base_model_metrics_sec:
                    grouped_metrics_sec, grouped_rows_sec, grouped_labels_sec, _group_names_sec = _build_group_mean_payloads(
                        model_payloads=base_model_metrics_sec,
                        diff_payloads=base_model_rows_sec,
                        model_labels=base_labels_for_sector,
                        group_specs=regional_group_specs,
                    )
                    model_metrics_sec = list(grouped_metrics_sec)
                    model_rows_sec = list(grouped_rows_sec or [])
                    labels_for_sector = list(grouped_labels_sec)

                if not model_rows_sec:
                    raise RuntimeError(f"No valid SItrans regional model rows for sector '{sector_key}'.")
                if not base_model_metrics_sec:
                    raise RuntimeError(f"No valid SItrans regional model metrics for sector '{sector_key}'.")

                obs1_raw_sec = _extract_sitrans_obs_payload(base_model_metrics_sec[0])
                obs2_raw_sec = _extract_sitrans_mod_payload(obs2_metrics_sec) if obs2_metrics_sec is not None else None
                model_raw_sec = [_extract_sitrans_mod_payload(metrics_sec) for metrics_sec in model_metrics_sec]

                years_sec = np.array(base_model_metrics_sec[0].get('year', np.array([])), dtype=float)
                ts_payload_base = {
                    'region': utils.get_sector_label(hemisphere, sector_key),
                    'years': years_sec,
                    'advance_obs1': np.array(base_model_metrics_sec[0].get('advance_series_obs', np.array([])), dtype=float),
                    'retreat_obs1': np.array(base_model_metrics_sec[0].get('retreat_series_obs', np.array([])), dtype=float),
                    'advance_obs2': np.array(obs2_metrics_sec.get('advance_series_mod', np.array([])), dtype=float) if isinstance(obs2_metrics_sec, dict) else np.array([]),
                    'retreat_obs2': np.array(obs2_metrics_sec.get('retreat_series_mod', np.array([])), dtype=float) if isinstance(obs2_metrics_sec, dict) else np.array([]),
                    'advance_models': [np.array(mm.get('advance_series_mod', np.array([])), dtype=float) for mm in base_model_metrics_sec],
                    'retreat_models': [np.array(mm.get('retreat_series_mod', np.array([])), dtype=float) for mm in base_model_metrics_sec],
                    'model_labels': list(base_labels_for_sector),
                    'obs1_label': obs1_plot_label,
                    'obs2_label': obs2_plot_label,
                }
                ts_payload_group: Optional[Dict[str, Any]] = None
                if group_labels_sec and group_metric_means_sec:
                    ts_payload_group = {
                        'region': utils.get_sector_label(hemisphere, sector_key),
                        'years': years_sec,
                        'advance_obs1': np.array(base_model_metrics_sec[0].get('advance_series_obs', np.array([])), dtype=float),
                        'retreat_obs1': np.array(base_model_metrics_sec[0].get('retreat_series_obs', np.array([])), dtype=float),
                        'advance_obs2': np.array(obs2_metrics_sec.get('advance_series_mod', np.array([])), dtype=float) if isinstance(obs2_metrics_sec, dict) else np.array([]),
                        'retreat_obs2': np.array(obs2_metrics_sec.get('retreat_series_mod', np.array([])), dtype=float) if isinstance(obs2_metrics_sec, dict) else np.array([]),
                        'advance_models': [
                            np.array(mm.get('advance_series_mod', np.array([])), dtype=float)
                            if isinstance(mm, dict) else np.array([], dtype=float)
                            for mm in group_metric_means_sec
                        ],
                        'retreat_models': [
                            np.array(mm.get('retreat_series_mod', np.array([])), dtype=float)
                            if isinstance(mm, dict) else np.array([], dtype=float)
                            for mm in group_metric_means_sec
                        ],
                        'advance_model_spreads': [
                            _extract_spread_series(std_payload, 'advance_series_mod')
                            for std_payload in group_metric_stds_sec
                        ],
                        'retreat_model_spreads': [
                            _extract_spread_series(std_payload, 'retreat_series_mod')
                            for std_payload in group_metric_stds_sec
                        ],
                        'model_labels': list(group_labels_sec),
                        'obs1_label': obs1_plot_label,
                        'obs2_label': obs2_plot_label,
                    }

                table_payload = {
                    'type': 'dual_table',
                    'sections': [
                        {
                            'id': 'raw',
                            'title': 'Raw Values',
                            **_build_phase_raw_payload(
                                obs1_payload=obs1_raw_sec,
                                obs2_payload=obs2_raw_sec,
                                model_payloads=model_raw_sec,
                                labels=labels_for_sector,
                                obs1_label=obs1_plot_label,
                                obs2_label=obs2_plot_label,
                            ),
                        },
                        {
                            'id': 'diff',
                            'title': 'Differences',
                            **_build_phase_diff_payload(
                                obs2_metrics_sec,
                                model_rows_sec,
                                labels_for_sector,
                                obs1_label=obs1_plot_label,
                                obs2_label=obs2_plot_label,
                            ),
                        },
                    ],
                }
                return ts_payload_base, ts_payload_group, table_payload

            def _build_empty_sector_outputs(
                sector_key: str,
            ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
                n_years = int(years_ref.size)
                nan_series = np.full((n_years,), np.nan, dtype=float)
                ts_payload_base = {
                    'region': utils.get_sector_label(hemisphere, sector_key),
                    'years': np.array(years_ref, dtype=float),
                    'advance_obs1': np.array(nan_series, dtype=float),
                    'retreat_obs1': np.array(nan_series, dtype=float),
                    'advance_obs2': np.array(nan_series, dtype=float),
                    'retreat_obs2': np.array(nan_series, dtype=float),
                    'advance_models': [np.array(nan_series, dtype=float) for _ in model_labels_full],
                    'retreat_models': [np.array(nan_series, dtype=float) for _ in model_labels_full],
                    'model_labels': list(model_labels_full),
                    'obs1_label': obs1_plot_label,
                    'obs2_label': obs2_plot_label,
                }
                ts_payload_group: Optional[Dict[str, Any]] = None
                if regional_group_labels:
                    ts_payload_group = {
                        'region': utils.get_sector_label(hemisphere, sector_key),
                        'years': np.array(years_ref, dtype=float),
                        'advance_obs1': np.array(nan_series, dtype=float),
                        'retreat_obs1': np.array(nan_series, dtype=float),
                        'advance_obs2': np.array(nan_series, dtype=float),
                        'retreat_obs2': np.array(nan_series, dtype=float),
                        'advance_models': [np.array(nan_series, dtype=float) for _ in regional_group_labels],
                        'retreat_models': [np.array(nan_series, dtype=float) for _ in regional_group_labels],
                        'advance_model_spreads': [np.array(nan_series, dtype=float) for _ in regional_group_labels],
                        'retreat_model_spreads': [np.array(nan_series, dtype=float) for _ in regional_group_labels],
                        'model_labels': list(regional_group_labels),
                        'obs1_label': obs1_plot_label,
                        'obs2_label': obs2_plot_label,
                    }
                table_payload = {
                    'type': 'dual_table',
                    'sections': [
                        {
                            'id': 'raw',
                            'title': 'Raw Values',
                            **_build_phase_raw_payload(
                                obs1_payload=None,
                                obs2_payload=None,
                                model_payloads=[None for _ in regional_labels_with_groups],
                                labels=regional_labels_with_groups,
                                obs1_label=obs1_plot_label,
                                obs2_label=obs2_plot_label,
                            ),
                        },
                        {
                            'id': 'diff',
                            'title': 'Differences',
                            **_build_phase_diff_payload(
                                None,
                                [{} for _ in regional_labels_with_groups],
                                regional_labels_with_groups,
                                obs1_label=obs1_plot_label,
                                obs2_label=obs2_plot_label,
                            ),
                        },
                    ],
                }
                return ts_payload_base, ts_payload_group, table_payload

            for sector in sectors:
                ts_payload_base = None
                ts_payload_group = None
                table_payload = None
                sector_error = None
                for attempt in range(2):
                    metric_engine = sitr_metrics_region if attempt == 0 else _new_sitr_metrics_region()
                    try:
                        ts_payload_base, ts_payload_group, table_payload = _compute_sector_outputs(metric_engine, sector)
                        if attempt == 1:
                            logger.info(
                                "Recovered %s regional table for sector '%s' after retry.",
                                module,
                                sector,
                            )
                        break
                    except Exception as exc:
                        sector_error = exc
                        if attempt == 0:
                            logger.warning(
                                "Regional %s table failed for sector '%s' (%s); retrying with fresh metric context.",
                                module,
                                sector,
                                exc,
                            )
                if ts_payload_base is None or table_payload is None:
                    logger.warning("Falling back to empty %s regional payload for sector '%s' (%s).", module, sector, sector_error)
                    ts_payload_base, ts_payload_group, table_payload = _build_empty_sector_outputs(sector)

                region_timeseries_payloads_base.append(ts_payload_base)
                if ts_payload_group is not None:
                    region_timeseries_payloads_group.append(ts_payload_group)
                regional_tables[sector] = table_payload

            if region_timeseries_payloads_base:
                try:
                    fig_dir = Path(output_dir) / module
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    pf.plot_sitrans_region_timeseries(
                        region_timeseries_payloads_base,
                        phase='advance',
                        hms=hemisphere,
                        fig_name=str(fig_dir / 'SItrans_advance_region_timeseries_raw.png'),
                    )
                    pf.plot_sitrans_region_timeseries(
                        region_timeseries_payloads_base,
                        phase='retreat',
                        hms=hemisphere,
                        fig_name=str(fig_dir / 'SItrans_retreat_region_timeseries_raw.png'),
                    )
                    if region_timeseries_payloads_group:
                        pf.plot_sitrans_region_timeseries(
                            region_timeseries_payloads_group,
                            phase='advance',
                            hms=hemisphere,
                            fig_name=str(fig_dir / 'SItrans_advance_region_timeseries_raw_groupmean.png'),
                        )
                        pf.plot_sitrans_region_timeseries(
                            region_timeseries_payloads_group,
                            phase='retreat',
                            hms=hemisphere,
                            fig_name=str(fig_dir / 'SItrans_retreat_region_timeseries_raw_groupmean.png'),
                        )
                except Exception as exc:
                    logger.warning("Failed to generate regional SItrans time-series figures (%s).", exc)

        return _build_region_table_payload(
            hemisphere=hemisphere,
            regional_tables=regional_tables,
            payload_type='region_dual_table',
        )

    return None

__all__ = ["eval_sitrans"]
