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


def eval_sicb(case_name: str, recipe: RR.RecipeReader,
              data_dir: str, output_dir: str,
              recalculate: bool = False,
              jobs: int = 1) -> Optional[dict]:
    """Evaluate sea ice concentration budget (SICB).

    Args:
        case_name: Evaluation experiment name.
        recipe: Validated RecipeReader instance.
        data_dir: Directory for processed data files.
        output_dir: Base Output/ directory for figures.

    Returns:
        Scalar metric table dict for the HTML report, or None if no data.
    """
    module = 'SICB'
    _log_module_header(module)

    module_vars = recipe.variables[module]
    hemisphere = recipe.hemisphere
    year_sta, year_end = _get_year_range(module_vars)
    date_start = f"{year_sta}-01-01"
    date_end = f"{year_end}-12-31"
    plot_opts = _load_plot_options(case_name)
    _apply_plot_runtime_options(plot_opts, module)
    line_colors = _plot_options_get_module(plot_opts, module, ['line', 'model_colors'], None)
    if not isinstance(line_colors, (list, tuple)) or len(line_colors) == 0:
        line_colors = None
    line_styles = _plot_options_get_module(plot_opts, module, ['line', 'model_linestyles'], None)
    if not isinstance(line_styles, (list, tuple)) or len(line_styles) == 0:
        line_styles = None
    budget_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'budget_cmap'], 'RdBu_r')
    budget_vmin = float(_plot_options_get_module(plot_opts, module, ['maps', 'vmin'], -100.0))
    budget_vmax = float(_plot_options_get_module(plot_opts, module, ['maps', 'vmax'], 100.0))
    budget_tick_bin = float(_plot_options_get_module(plot_opts, module, ['maps', 'cbtick_bin'], 20.0))

    requested_jobs = max(1, int(jobs))
    default_sicb_cap = requested_jobs
    try:
        sicb_jobs_cap = max(1, int(os.environ.get('SITOOL_SICB_MAX_JOBS', str(default_sicb_cap))))
    except Exception:
        sicb_jobs_cap = default_sicb_cap
    jobs = min(requested_jobs, sicb_jobs_cap)
    if jobs < requested_jobs:
        logger.info(
            "Capping %s internal jobs via SITOOL_SICB_MAX_JOBS: requested=%d -> effective=%d",
            module, requested_jobs, jobs,
        )

    # Check if all expected outputs already exist
    fig_dir = Path(output_dir) / module
    if _check_outputs_exist(module, fig_dir, hemisphere, recalculate=recalculate):
        logger.info(f"{module} evaluation skipped — all outputs exist.")
        return None

    cache_file = _get_metrics_cache_file(case_name, output_dir, hemisphere, module)
    preprocessor = PP.DataPreprocessor(case_name, module, hemisphere=recipe.hemisphere)
    grid_file = preprocessor.gen_eval_grid()

    metric = _get_metric(module_vars)

    def _new_sicb_metrics() -> SIM.SICBMetrics:
        """Build one SICBMetrics instance.

        We create a factory so parallel workers can allocate thread-local
        metric objects instead of sharing one mutable instance.
        """
        return SIM.SICBMetrics(
            grid_file=grid_file, date_start=date_start, date_end=date_end,
            hemisphere=hemisphere, metric=metric,
        )

    sicb_metrics = _new_sicb_metrics()
    obs_labels = _get_reference_labels(module_vars, hemisphere, suffix='_sic')
    sidrift_obs_labels = _get_reference_labels(module_vars, hemisphere, suffix='_sidrift')
    obs1_plot_label = obs_labels[0] if len(obs_labels) >= 1 else 'obs1'
    obs2_plot_label = obs_labels[1] if len(obs_labels) >= 2 else 'obs2'

    def _merge_obs_pair_name(index: int) -> str:
        sic_name = (
            str(obs_labels[index]).strip()
            if index < len(obs_labels) and str(obs_labels[index]).strip()
            else f'obs{index + 1}'
        )
        sid_name = (
            str(sidrift_obs_labels[index]).strip()
            if index < len(sidrift_obs_labels) and str(sidrift_obs_labels[index]).strip()
            else ''
        )
        if sid_name and sid_name.lower() != sic_name.lower():
            return f'{sic_name}+{sid_name}'
        return sic_name

    obs1_plot_label_sicb2 = _merge_obs_pair_name(0)
    obs2_plot_label_sicb2 = _merge_obs_pair_name(1)

    obs_seas_file: Optional[str] = None
    obs_mon_clim: Optional[str] = None
    obs_daily_file: Optional[str] = None
    obs2_seas_file: Optional[str] = None
    obs2_mon_clim: Optional[str] = None
    obs2_daily_file: Optional[str] = None
    model_seas_files: List[str] = []
    model_mon_clim_files: List[str] = []
    model_daily_files: List[str] = []
    model_labels: List[str] = _get_recipe_model_labels(module, module_vars, len(module_vars.get('model_file_sic') or []))
    cache_loaded = False

    def _derive_daily_budget_path(seas_path: Optional[str]) -> Optional[str]:
        if not seas_path:
            return None
        p = Path(str(seas_path))
        name = p.name
        suffix = '_SeasClim.nc'
        if not name.endswith(suffix):
            return None
        daily_name = name[:-len(suffix)] + '_daily.nc'
        daily_path = str(p.with_name(daily_name))
        return daily_path if Path(daily_path).exists() else None

    def _sicb_dataset_has_ridging(path: Optional[str], required_vars: List[str]) -> bool:
        if not path or (not Path(path).exists()):
            return False
        try:
            with xr.open_dataset(path) as ds:
                for vname in required_vars:
                    if vname not in ds.variables:
                        return False
                return int(ds.attrs.get('sicb_ridging_enabled', 0)) == 1
        except Exception:
            return False

    def _sicb_payload_has_ridging(seas_path: Optional[str], daily_path: Optional[str]) -> bool:
        return (
            _sicb_dataset_has_ridging(seas_path, ['ridging_mask', 'sic_mean'])
            and _sicb_dataset_has_ridging(daily_path, ['ridging_ratio', 'sic'])
        )

    def _budget_payload_to_paths(payload: Optional[dict]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not isinstance(payload, dict):
            return None, None, None
        seas = payload.get('seas_file')
        mon = payload.get('mon_file')
        daily = payload.get('daily_file')
        seas_path = str(seas) if seas is not None else None
        mon_path = str(mon) if mon is not None else None
        daily_path = str(daily) if daily is not None else None
        if isinstance(seas_path, str) and seas_path.strip() == '':
            seas_path = None
        if isinstance(mon_path, str) and mon_path.strip() == '':
            mon_path = None
        if isinstance(daily_path, str) and daily_path.strip() == '':
            daily_path = None
        if seas_path and not Path(seas_path).exists():
            seas_path = None
        if mon_path and not Path(mon_path).exists():
            mon_path = None
        if daily_path and not Path(daily_path).exists():
            daily_path = None
        if daily_path is None:
            daily_path = _derive_daily_budget_path(seas_path)
        return seas_path, mon_path, daily_path

    if not recalculate:
        cached = _load_module_cache(cache_file, module, hemisphere)
        if cached is not None and cached.get('payload_kind') == module:
            try:
                records = cached.get('records', {})
                obs_seas_file, obs_mon_clim, obs_daily_file = _budget_payload_to_paths(records.get('obs1_budget'))
                obs2_seas_file, obs2_mon_clim, obs2_daily_file = _budget_payload_to_paths(records.get('obs2_budget'))

                model_records = cached.get('model_records', [])
                for rec_name in model_records:
                    seas_file, mon_file, daily_file = _budget_payload_to_paths(records.get(rec_name))
                    if seas_file:
                        model_seas_files.append(seas_file)
                    if mon_file:
                        model_mon_clim_files.append(mon_file)
                    if daily_file:
                        model_daily_files.append(daily_file)

                model_labels = cached.get('model_labels', model_labels)
                if not model_labels and model_records:
                    model_labels = _get_recipe_model_labels(module, module_vars, len(model_records))

                cache_loaded = (
                    obs_seas_file is not None
                    and bool(model_seas_files)
                    and len(model_seas_files) == len(model_daily_files)
                    and _sicb_payload_has_ridging(obs_seas_file, obs_daily_file)
                    and all(
                        _sicb_payload_has_ridging(seas_file, daily_file)
                        for seas_file, daily_file in zip(model_seas_files, model_daily_files)
                    )
                )
                if cache_loaded:
                    logger.info("Loaded %s metrics from cache: %s", module, cache_file)
                else:
                    logger.info("SICB cache is stale (ridging variables missing). Recalculating.")
            except Exception as exc:
                logger.warning("Cache payload for %s is incomplete (%s). Recalculating.", module, exc)
                cache_loaded = False

    if not cache_loaded:
        file_groups = recipe.validate_module(module)

        # SICB preprocessing parallelization plan:
        # - keep dedicated lanes for SIC and drift observations;
        # - reserve most worker budget for the model preprocess long tail;
        # - keep full fallback compatibility via one feature flag.
        case_name_norm = str(case_name or '').strip().lower()
        prep_pipeline_v2_raw = str(
            os.environ.get('SITOOL_SICB_PREP_PIPELINE_V2', 'auto')
        ).strip().lower()
        if prep_pipeline_v2_raw in {'0', 'false', 'no', 'off'}:
            enable_prep_pipeline_v2 = False
        elif prep_pipeline_v2_raw in {'1', 'true', 'yes', 'on'}:
            enable_prep_pipeline_v2 = True
        else:
            enable_prep_pipeline_v2 = case_name_norm.startswith('highres')

        obs_parallel_workers = 1
        obs_jobs_each = max(1, int(jobs))
        model_prep_jobs = max(1, int(jobs))
        if enable_prep_pipeline_v2 and int(jobs) >= 4:
            try:
                obs_jobs_each = max(
                    1,
                    min(
                        int(jobs) - 1,
                        int(os.environ.get('SITOOL_SICB_OBS_JOBS', '4')),
                    ),
                )
            except Exception:
                obs_jobs_each = max(1, min(int(jobs) - 1, 4))
            obs_parallel_workers = 2
            # Observation and model preprocess stages are still sequential,
            # so keep full budget for the model long-tail stage.
            model_prep_jobs = max(1, int(jobs))
        logger.info(
            "SICB preprocess layout [%s]: pipeline_v2=%s, obs_workers=%d, "
            "obs_jobs_each=%d, model_prep_jobs=%d (module_jobs=%d).",
            hemisphere.upper(), enable_prep_pipeline_v2,
            obs_parallel_workers, obs_jobs_each, model_prep_jobs, jobs,
        )

        obs_sic: List[str] = []
        obs_sid: List[str] = []
        if obs_parallel_workers > 1:
            obs_tasks = [('sic', '_sic'), ('sid', '_sidrift')]

            def _obs_worker(obs_task: Tuple[str, str]) -> Dict[str, Any]:
                obs_key, obs_flag = obs_task
                out_files = preprocessor.prep_obs(
                    frequency='daily',
                    output_dir=data_dir,
                    flag_name=obs_flag,
                    overwrite=bool(recalculate),
                    jobs=max(1, int(obs_jobs_each)),
                )
                return {'obs_key': obs_key, 'files': out_files}

            obs_stage = _parallel_map_ordered(
                items=obs_tasks,
                worker_fn=_obs_worker,
                max_workers=obs_parallel_workers,
                task_label=f'{hemisphere}/{module}/obs-preprocess',
            )
            for item in obs_stage:
                if item.get('obs_key') == 'sic':
                    obs_sic = list(item.get('files') or [])
                elif item.get('obs_key') == 'sid':
                    obs_sid = list(item.get('files') or [])
        else:
            # SICB requires separate processing of SIC and drift obs files
            obs_sic = preprocessor.prep_obs(
                frequency='daily',
                output_dir=data_dir,
                flag_name='_sic',
                overwrite=bool(recalculate),
                jobs=jobs,
            )
            obs_sid = preprocessor.prep_obs(
                frequency='daily',
                output_dir=data_dir,
                flag_name='_sidrift',
                overwrite=bool(recalculate),
                jobs=jobs,
            )
        all_model_files = preprocessor.prep_models(
            file_groups, frequency='daily', output_dir=data_dir,
            overwrite=bool(recalculate), jobs=model_prep_jobs
        ) if file_groups else []
        n_models = len(all_model_files) // 3
        model_sic_files = all_model_files[:n_models]
        model_u_files = all_model_files[n_models:2 * n_models]
        model_v_files = all_model_files[2 * n_models:]
        model_labels = _get_recipe_model_labels(module, module_vars, n_models)

        # Calculate obs1 budget
        if obs_sic and obs_sid:
            obs_seas_file, obs_mon_clim = sicb_metrics.Cal_SIC_budget(
                sic_file=os.path.join(data_dir, obs_sic[0]),
                sic_key=module_vars.get('ref_var_sic', 'siconc'),
                u_file=os.path.join(data_dir, obs_sid[0]),
                u_key=module_vars.get('ref_var_u', 'u'),
                v_file=os.path.join(data_dir, obs_sid[0]),
                v_key=module_vars.get('ref_var_v', 'v'),
                hemisphere=hemisphere,
                rotate=False,
                output_folder=str(data_dir) + '/',
                jobs=max(1, int(jobs)),
                output_label=module_vars.get('ref_label', _sanitize_group_name(obs1_plot_label) or 'obs'),
                reuse_existing=not bool(recalculate),
            )
            obs_daily_file = _derive_daily_budget_path(obs_seas_file)

        # Calculate obs2 budget (if second obs product available)
        if obs_sic and len(obs_sic) >= 2 and obs_sid and len(obs_sid) >= 2:
            obs2_seas_file, obs2_mon_clim = sicb_metrics.Cal_SIC_budget(
                sic_file=os.path.join(data_dir, obs_sic[1]),
                sic_key=module_vars.get('ref_var_sic', 'siconc'),
                u_file=os.path.join(data_dir, obs_sid[1]),
                u_key=module_vars.get('ref_var_u', 'u'),
                v_file=os.path.join(data_dir, obs_sid[1]),
                v_key=module_vars.get('ref_var_v', 'v'),
                hemisphere=hemisphere,
                rotate=False,
                output_folder=str(data_dir) + '/',
                jobs=max(1, int(jobs)),
                output_label=_sanitize_group_name(obs2_plot_label) or 'obs2',
                reuse_existing=not bool(recalculate),
            )
            obs2_daily_file = _derive_daily_budget_path(obs2_seas_file)

        # Calculate model budgets
        model_seas_files = []
        model_mon_clim_files = []
        model_daily_files = []
        model_tasks = list(enumerate(zip(model_sic_files, model_u_files, model_v_files)))

        if jobs > 1 and model_tasks:
            stage_dir = _get_stage_dir(case_name, output_dir, hemisphere, module) / 'model_budget'
            stage_dir.mkdir(parents=True, exist_ok=True)
            model_workers = max(1, min(int(jobs), len(model_tasks)))
            try:
                inner_jobs_override_raw = str(
                    os.environ.get('SITOOL_SICB_BUDGET_INNER_JOBS', '')
                ).strip()
                if inner_jobs_override_raw:
                    inner_budget_jobs = max(1, int(inner_jobs_override_raw))
                else:
                    inner_budget_jobs = max(1, int(jobs) // max(1, model_workers))
            except Exception:
                inner_budget_jobs = max(1, int(jobs) // max(1, model_workers))
            inner_budget_jobs = max(1, min(int(jobs), int(inner_budget_jobs)))
            logger.info(
                "SICB model-budget nested parallel layout [%s]: model_workers=%d, inner_budget_jobs=%d (module_jobs=%d).",
                hemisphere.upper(), model_workers, inner_budget_jobs, jobs,
            )

            def _worker(task: Tuple[int, Tuple[str, str, str]]) -> Dict[str, Any]:
                ii, (mf_sic, mf_u, mf_v) = task
                label = model_labels[ii] if ii < len(model_labels) else f'model{ii + 1}'

                # IMPORTANT:
                # Create a dedicated SICBMetrics object per worker thread.
                # This avoids cross-thread mutation on shared state such as
                # cached coordinates and temporary computation members.
                worker_metrics = _new_sicb_metrics()
                seas_file, mon_clim = worker_metrics.Cal_SIC_budget(
                    sic_file=os.path.join(data_dir, mf_sic),
                    sic_key=module_vars.get('model_var_sic', 'siconc'),
                    u_file=os.path.join(data_dir, mf_u),
                    u_key=module_vars.get('model_var_u', 'u'),
                    v_file=os.path.join(data_dir, mf_v),
                    v_key=module_vars.get('model_var_v', 'v'),
                    hemisphere=hemisphere,
                    rotate=module_vars.get('rotate_model', True),
                    output_folder=str(data_dir) + '/',
                    jobs=max(1, int(inner_budget_jobs)),
                    output_label=label,
                    reuse_existing=not bool(recalculate),
                )
                payload_file = stage_dir / f'model_{ii:04d}.pkl'
                _save_pickle_atomic(payload_file, {
                    'seas_file': seas_file,
                    'mon_file': mon_clim,
                    'daily_file': _derive_daily_budget_path(seas_file),
                })
                return {'payload_file': str(payload_file)}

            stage_refs = _parallel_map_ordered(
                items=model_tasks,
                worker_fn=_worker,
                max_workers=model_workers,
                task_label=f'{hemisphere}/{module}/model-budget',
            )
            staged_payloads = _load_staged_payloads(stage_refs)
            for payload in staged_payloads:
                model_seas_files.append(payload['seas_file'])
                model_mon_clim_files.append(payload['mon_file'])
                model_daily_files.append(payload.get('daily_file'))
        else:
            for ii, (mf_sic, mf_u, mf_v) in model_tasks:
                label = model_labels[ii] if ii < len(model_labels) else f'model{ii + 1}'
                seas_file, mon_clim = sicb_metrics.Cal_SIC_budget(
                    sic_file=os.path.join(data_dir, mf_sic),
                    sic_key=module_vars.get('model_var_sic', 'siconc'),
                    u_file=os.path.join(data_dir, mf_u),
                    u_key=module_vars.get('model_var_u', 'u'),
                    v_file=os.path.join(data_dir, mf_v),
                    v_key=module_vars.get('model_var_v', 'v'),
                    hemisphere=hemisphere,
                    rotate=module_vars.get('rotate_model', True),
                    output_folder=str(data_dir) + '/',
                    jobs=max(1, int(jobs)),
                    output_label=label,
                    reuse_existing=not bool(recalculate),
                )
                model_seas_files.append(seas_file)
                model_mon_clim_files.append(mon_clim)
                model_daily_files.append(_derive_daily_budget_path(seas_file))

        model_records = [f'model_{i}_budget' for i in range(len(model_seas_files))]
        records: Dict[str, Dict[str, Any]] = {}
        if obs_seas_file and obs_mon_clim:
            records['obs1_budget'] = {
                'seas_file': str(Path(obs_seas_file).resolve()),
                'mon_file': str(Path(obs_mon_clim).resolve()),
                'daily_file': str(Path(obs_daily_file).resolve()) if obs_daily_file else '',
            }
        if obs2_seas_file and obs2_mon_clim:
            records['obs2_budget'] = {
                'seas_file': str(Path(obs2_seas_file).resolve()),
                'mon_file': str(Path(obs2_mon_clim).resolve()),
                'daily_file': str(Path(obs2_daily_file).resolve()) if obs2_daily_file else '',
            }
        for rec_name, seas_file, mon_file, daily_file in zip(
            model_records, model_seas_files, model_mon_clim_files, model_daily_files,
        ):
            records[rec_name] = {
                'seas_file': str(Path(seas_file).resolve()),
                'mon_file': str(Path(mon_file).resolve()),
                'daily_file': str(Path(daily_file).resolve()) if daily_file else '',
            }

        used_entities: set = set()
        entity_groups: Dict[str, str] = {}
        if 'obs1_budget' in records:
            entity_groups['obs1_budget'] = _unique_entity_name(
                preferred=obs_labels[0] if len(obs_labels) >= 1 else 'Reference_1',
                fallback='Reference_1',
                used=used_entities,
            )
        if 'obs2_budget' in records:
            entity_groups['obs2_budget'] = _unique_entity_name(
                preferred=obs_labels[1] if len(obs_labels) >= 2 else 'Reference_2',
                fallback='Reference_2',
                used=used_entities,
            )
        for ii, rec_name in enumerate(model_records):
            entity_groups[rec_name] = _unique_entity_name(
                preferred=model_labels[ii] if ii < len(model_labels) else f'{module}_dataset_{ii + 1}',
                fallback=f'{module}_dataset_{ii + 1}',
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
            obs_seas_plot, obs_mon_plot, obs_daily_plot = _budget_payload_to_paths(records.get('obs1_budget'))
            obs2_seas_plot, obs2_mon_plot, obs2_daily_plot = _budget_payload_to_paths(records.get('obs2_budget'))
            model_records_plot = cached_for_plot.get('model_records', [])
            model_seas_plot: List[str] = []
            model_mon_plot: List[str] = []
            model_daily_plot: List[str] = []
            for rec_name in model_records_plot:
                seas_file, mon_file, daily_file = _budget_payload_to_paths(records.get(rec_name))
                if seas_file:
                    model_seas_plot.append(seas_file)
                if mon_file:
                    model_mon_plot.append(mon_file)
                if daily_file:
                    model_daily_plot.append(daily_file)
            if obs_seas_plot is not None and model_seas_plot:
                obs_seas_file = obs_seas_plot
                obs_mon_clim = obs_mon_plot
                obs_daily_file = obs_daily_plot
                obs2_seas_file = obs2_seas_plot
                obs2_mon_clim = obs2_mon_plot
                obs2_daily_file = obs2_daily_plot
                model_seas_files = model_seas_plot
                model_mon_clim_files = model_mon_plot
                model_daily_files = model_daily_plot
                model_labels = cached_for_plot.get('model_labels', model_labels) or model_labels
                logger.info("Using cache-backed %s payload for plotting: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to reload %s cache for plotting (%s). Using in-memory payload.", module, exc)

    # Metric-level (post-diagnostic) group means built from SICB diagnostic files.
    base_model_seas_files = list(model_seas_files)
    base_model_mon_clim_files = list(model_mon_clim_files)
    base_model_daily_files = list(model_daily_files)
    base_model_labels_for_group = list(model_labels)
    model_labels_seas = list(base_model_labels_for_group)
    model_labels_mon = list(base_model_labels_for_group)
    model_labels_daily = list(base_model_labels_for_group)
    group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=base_model_labels_for_group,
    )
    group_labels_seas: List[str] = []
    group_labels_mon: List[str] = []
    group_labels_daily: List[str] = []
    group_member_map_mon: Dict[str, List[str]] = {}
    group_member_map_daily: Dict[str, List[str]] = {}
    if group_specs and model_seas_files:
        group_dir = Path(output_dir) / module / '_groupmean'
        seas_with_groups, labels_with_groups, group_labels_seas = _build_group_mean_file_payloads(
            file_paths=model_seas_files,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
            output_dir=group_dir / 'seasonal',
            file_tag='sicb_seasonal_groupmean',
        )
        model_seas_files = list(seas_with_groups)
        model_labels_seas = list(labels_with_groups)
        if group_labels_seas:
            logger.info(
                "Enabled file-level group means for %s [%s] seasonal panels/tables: %s",
                module, hemisphere.upper(), ', '.join(group_labels_seas),
            )
    if group_specs and model_mon_clim_files:
        group_member_map_mon = _build_group_member_file_map(
            file_paths=base_model_mon_clim_files,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
        )
        mon_with_groups, labels_mon_groups, _mon_groups = _build_group_mean_file_payloads(
            file_paths=model_mon_clim_files,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
            output_dir=Path(output_dir) / module / '_groupmean' / 'monthly',
            file_tag='sicb_monthly_groupmean',
        )
        model_mon_clim_files = list(mon_with_groups)
        model_labels_mon = list(labels_mon_groups)
        group_labels_mon = list(_mon_groups or [])
    if group_specs and model_daily_files:
        group_member_map_daily = _build_group_member_file_map(
            file_paths=base_model_daily_files,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
        )
        daily_with_groups, labels_daily_groups, _daily_groups = _build_group_mean_file_payloads(
            file_paths=model_daily_files,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
            output_dir=Path(output_dir) / module / '_groupmean' / 'daily',
            file_tag='sicb_daily_groupmean',
        )
        model_daily_files = list(daily_with_groups)
        model_labels_daily = list(labels_daily_groups)
        group_labels_daily = list(_daily_groups or [])

    if not model_labels_seas:
        model_labels_seas = list(base_model_labels_for_group)
    if not model_labels_mon:
        model_labels_mon = list(base_model_labels_for_group)
    if not model_labels_daily:
        model_labels_daily = list(base_model_labels_for_group)
    model_labels = list(model_labels_seas)

    # --- Figures ---
    if obs_seas_file:
        logger.info("Generating figures ...")
        fig_dir = Path(output_dir) / module
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Remove legacy matched plots so HTML no longer exposes Obs-Matched
        # coverage controls for SICB.
        for stale_png in fig_dir.glob('*matched*.png'):
            try:
                stale_png.unlink()
            except Exception as exc:
                logger.warning("Failed to remove stale SICB matched plot %s (%s).", stale_png, exc)

        # Observation-only budget panel
        pf.plot_SICB(
            obs_seas_file, hemisphere,
            vmin=budget_vmin, vmax=budget_vmax, cbtick_bin=budget_tick_bin,
            cm=budget_cmap,
            fig_name=str(fig_dir / 'SICB_obs.png'),
        )

        # Build combined lists including obs2 if available
        all_seas_files = (
            [obs_seas_file] +
            ([obs2_seas_file] if obs2_seas_file else []) +
            model_seas_files
        )
        model_labels_used_seas = [
            model_labels_seas[ii] if ii < len(model_labels_seas) else f'model{ii + 1}'
            for ii in range(len(model_seas_files))
        ]
        all_labels = (
            [obs1_plot_label] +
            ([obs2_plot_label] if obs2_seas_file else []) +
            model_labels_used_seas
        )
        all_labels_sicb2 = (
            [obs1_plot_label_sicb2] +
            ([obs2_plot_label_sicb2] if obs2_seas_file else []) +
            model_labels_used_seas
        )

        # Multi-dataset comparison panels (one figure per season)
        if model_seas_files:
            pf.plot_SICB2(
                all_seas_files, all_labels_sicb2, hemisphere,
                vmin=budget_vmin, vmax=budget_vmax, cbtick_bin=budget_tick_bin,
                cm=budget_cmap,
                fig_label=str(fig_dir / 'SICB2'),
            )

        # Monthly climatology time series of budget terms
        all_mon_clim = (
            [obs_mon_clim] +
            ([obs2_mon_clim] if obs2_mon_clim else []) +
            base_model_mon_clim_files
        )
        model_labels_used_mon = list(base_model_labels_for_group)
        all_labels_mon = (
            [obs1_plot_label] +
            ([obs2_plot_label] if obs2_mon_clim else []) +
            model_labels_used_mon
        )
        if obs_mon_clim and base_model_mon_clim_files:
            pf.plot_SICB_ts(
                all_mon_clim, all_labels_mon, hemisphere,
                line_style=line_styles, color=line_colors,
                n_obs=(2 if obs2_mon_clim else 1),
                fig_name=str(fig_dir / 'SICB_ts.png'),
            )
            if group_labels_mon:
                group_mon_files = model_mon_clim_files[:len(group_labels_mon)]
                group_labels_mon_only = model_labels_mon[:len(group_labels_mon)]
                pf.plot_SICB_ts(
                    [obs_mon_clim] + ([obs2_mon_clim] if obs2_mon_clim else []) + list(group_mon_files),
                    [obs1_plot_label] + ([obs2_plot_label] if obs2_mon_clim else []) + list(group_labels_mon_only),
                    hemisphere,
                    line_style=line_styles, color=line_colors,
                    n_obs=(2 if obs2_mon_clim else 1),
                    fig_name=str(fig_dir / 'SICB_ts_groupmean.png'),
                    group_member_files=group_member_map_mon,
                )

        model_labels_used_daily = [
            model_labels_daily[ii] if ii < len(model_labels_daily) else f'model{ii + 1}'
            for ii in range(len(base_model_daily_files))
        ]
        all_labels_daily = (
            [obs1_plot_label] +
            ([obs2_plot_label] if obs2_daily_file else []) +
            model_labels_used_daily
        )
        all_daily = [obs_daily_file] + ([obs2_daily_file] if obs2_daily_file else []) + base_model_daily_files
        daily_pairs = [
            (fpath, label)
            for fpath, label in zip(all_daily, all_labels_daily)
            if isinstance(fpath, str) and Path(fpath).exists()
        ]
        obs_daily_ready = isinstance(obs_daily_file, str) and Path(obs_daily_file).exists()
        valid_model_daily_count = sum(1 for fpath in base_model_daily_files if isinstance(fpath, str) and Path(fpath).exists())
        if obs_daily_ready and valid_model_daily_count > 0 and len(daily_pairs) >= 2:
            pf.plot_SICB_ridging_ts(
                [p[0] for p in daily_pairs], [p[1] for p in daily_pairs], hemisphere,
                line_style=line_styles, color=line_colors,
                n_obs=(2 if obs2_daily_file and Path(obs2_daily_file).exists() else 1),
                fig_name=str(fig_dir / 'SICB_ridging_ts.png'),
            )
            # Ridging group-mean figure:
            # - Preferred: direct daily group-mean NetCDF files (if build succeeded).
            # - Fallback: compute group means on-the-fly from member daily files.
            ridging_group_labels = list(group_labels_daily)
            if (not ridging_group_labels) and isinstance(group_member_map_daily, dict):
                ridging_group_labels = [
                    str(lb) for lb in group_member_map_daily.keys()
                    if str(lb).strip()
                ]

            if ridging_group_labels:
                group_daily_file_by_label: Dict[str, str] = {}
                if group_labels_daily:
                    group_daily_files = model_daily_files[:len(group_labels_daily)]
                    group_daily_labels = model_labels_daily[:len(group_labels_daily)]
                    for gp_file, gp_label in zip(group_daily_files, group_daily_labels):
                        gp_label_txt = str(gp_label).strip()
                        if gp_label_txt and isinstance(gp_file, str) and gp_file:
                            group_daily_file_by_label[gp_label_txt] = gp_file

                group_daily_all = [obs_daily_file] + ([obs2_daily_file] if obs2_daily_file else [])
                group_labels_all = [obs1_plot_label] + ([obs2_plot_label] if obs2_daily_file else [])
                for gp_label in ridging_group_labels:
                    group_daily_all.append(group_daily_file_by_label.get(gp_label, ''))
                    group_labels_all.append(gp_label)

                group_pairs = []
                for fpath, label in zip(group_daily_all, group_labels_all):
                    if _is_group_mean_label(label):
                        has_members = bool(
                            isinstance(group_member_map_daily, dict)
                            and group_member_map_daily.get(str(label), [])
                        )
                        if (isinstance(fpath, str) and Path(fpath).exists()) or has_members:
                            group_pairs.append((fpath, label))
                    else:
                        if isinstance(fpath, str) and Path(fpath).exists():
                            group_pairs.append((fpath, label))

                if len(group_pairs) >= 2 and any(_is_group_mean_label(lb) for _, lb in group_pairs):
                    pf.plot_SICB_ridging_ts(
                        [p[0] for p in group_pairs], [p[1] for p in group_pairs], hemisphere,
                        line_style=line_styles, color=line_colors,
                        n_obs=(2 if obs2_daily_file and Path(obs2_daily_file).exists() else 1),
                        fig_name=str(fig_dir / 'SICB_ridging_ts_groupmean.png'),
                        group_member_files=group_member_map_daily,
                    )

        try:
            pf.plot_sic_region_map(
                grid_nc_file=grid_file,
                hms=hemisphere,
                fig_name=str(fig_dir / 'SeaIceRegion_map.png'),
            )
        except Exception as exc:
            logger.warning("Failed to generate sea-ice region map (%s).", exc)

    logger.info("%s evaluation completed.", module)

    # Seasonal scalar table from SeasClim files (one section per season)
    if obs_seas_file or obs2_seas_file or model_seas_files:
        _season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        _headers = [
            'Model/Obs Name',
            'dadt',
            'adv',
            'adv/dadt (%)',
            'div',
            'div/dadt (%)',
            'res',
            'res/dadt (%)',
        ]
        _units = [
            '',
            '10^6 km^2/month',
            '10^6 km^2/month',
            '%',
            '10^6 km^2/month',
            '%',
            '10^6 km^2/month',
            '%',
        ]

        def _fmt_total(v: float) -> str:
            return f"{v:.3f}" if np.isfinite(v) else 'NaN'

        def _fmt_pct(v: float) -> str:
            return f"{v:.1f}%" if np.isfinite(v) else 'NaN'

        def _build_seasonal_rows_for_sector(
            sector: str,
            obs1_file: Optional[str],
            baseline_file: Optional[str],
            model_files: List[str],
            model_names: List[str],
        ) -> Dict[str, List[List[str]]]:
            season_rows = {s: [] for s in _season_order}

            def _append_dataset_rows(label: str, seas_file: str) -> None:
                summary = sicb_metrics.summarize_seasonal_budget_table(
                    seas_clim_file=seas_file,
                    dataset_name=label,
                    sector=sector,
                )
                for season in _season_order:
                    s = summary[season]
                    season_rows[season].append([
                        label,
                        _fmt_total(s['dadt']),
                        _fmt_total(s['adv']),
                        _fmt_pct(s['adv_dadt_pct']),
                        _fmt_total(s['div']),
                        _fmt_pct(s['div_dadt_pct']),
                        _fmt_total(s['res']),
                        _fmt_pct(s['res_dadt_pct']),
                    ])

            if obs1_file:
                _append_dataset_rows(f'{obs1_plot_label} (baseline)', obs1_file)

            if baseline_file:
                _append_dataset_rows(obs2_plot_label, baseline_file)

            for ii, seas_file in enumerate(model_files):
                label = model_names[ii] if ii < len(model_names) else f'model{ii + 1}'
                _append_dataset_rows(label, seas_file)
            return season_rows

        model_labels_used = [
            model_labels_seas[ii] if ii < len(model_labels_seas) else f'model{ii + 1}'
            for ii in range(len(model_seas_files))
        ]

        all_payload = {
            'type': 'seasonal_table',
            'headers': _headers,
                'rows': [],
                'units': _units,
                'seasons': _build_seasonal_rows_for_sector(
                    'All', obs_seas_file, obs2_seas_file, model_seas_files, model_labels_used
                ),
            }

        regional_tables: Dict[str, Any] = {'All': all_payload}
        sectors = utils.get_hemisphere_sectors(hemisphere, include_all=False)
        for sector in sectors:
            try:
                regional_tables[sector] = {
                    'type': 'seasonal_table',
                    'headers': _headers,
                        'rows': [],
                        'units': _units,
                        'seasons': _build_seasonal_rows_for_sector(
                        sector, obs_seas_file, obs2_seas_file, model_seas_files, model_labels_used
                        ),
                    }
            except Exception as exc:
                logger.warning("Skipping %s regional table for sector '%s' (%s).", module, sector, exc)

        return _build_region_table_payload(
            hemisphere=hemisphere,
            regional_tables=regional_tables,
            payload_type='region_seasonal_table',
        )

    return None

__all__ = ["eval_sicb"]
