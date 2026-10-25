# -*- coding: utf-8 -*-
"""Split preprocessing mixins."""

import multiprocessing as mp
import queue as queue_mod

from scripts.preprocess import base as _base

# Reuse shared namespace (imports/constants/helpers) from base module.
globals().update({k: v for k, v in _base.__dict__.items() if k not in globals()})


class ObsModelPrepMixin:
    """Mixin with grouped preprocessing methods."""

    # -----------------------------------------------------------------------------
    # Function: `prep_obs`.
    # Purpose: Handle prep obs workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_obs(self, frequency: str = None, output_dir: str = None,
                 flag_name: str = '',
                 # backward-compatible aliases
                 freq: str = None, out_dir: str = None,
                 overwrite: bool = False,
                 jobs: int = 1) -> List[str]:
        """Preprocess observation data for evaluation.

        Args:
            frequency: Temporal frequency ('daily' or 'monthly').
            output_dir: Output directory for processed files.
            flag_name: Additional flag for file naming (e.g., '_sic').
            freq: Alias for *frequency* (backward compatibility).
            out_dir: Alias for *output_dir* (backward compatibility).
            overwrite: Rebuild output files even when existing files are valid.
            jobs: Maximum number of worker threads for per-observation preprocessing.

        Returns:
            List of processed filenames (basenames only).

        Raises:
            ValueError: If frequency is invalid or a required file is missing.
        """
        frequency = frequency or freq
        output_dir = output_dir or out_dir

        # Common setup: validate frequency, generate grid, build date range, create CDO
        grid_txt, start_date, end_date, cdo, module_vars = \
            self._prepare_common(frequency, output_dir)

        processed_files: List[str] = []
        safe_jobs = max(1, int(jobs))
        if safe_jobs > 1:
            force_serial = str(os.environ.get('SITOOL_PREP_FORCE_SERIAL', '0')).strip().lower() in {
                '1', 'true', 'yes', 'on',
            }
            max_parallel = max(1, int(os.environ.get('SITOOL_PREP_MAX_PARALLEL', str(safe_jobs))))
            if force_serial:
                logger.info(
                    "Observation preprocessing forced to serial mode by SITOOL_PREP_FORCE_SERIAL "
                    "(requested=%d -> 1).",
                    safe_jobs,
                )
                safe_jobs = 1
            else:
                if safe_jobs > max_parallel:
                    logger.info(
                        "Capping observation preprocessing jobs via SITOOL_PREP_MAX_PARALLEL "
                        "(requested=%d -> %d).",
                        safe_jobs, max_parallel,
                    )
                safe_jobs = min(safe_jobs, max_parallel)
        # Build the recipe observation file key from hemisphere and optional suffix, e.g. "ref_sh_sic"
        ref_key = f"ref_{self.hemisphere}{flag_name}"

        if ref_key not in module_vars:
            return processed_files

        obs_list = list(module_vars[ref_key])

        # -----------------------------------------------------------------------------
        # Function: `_resolve_obs_output`.
        # Purpose: Handle resolve obs output workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _resolve_obs_output(obs_file: str) -> Tuple[str, str]:
            stem = os.path.splitext(os.path.basename(obs_file))[0]
            out_name = (
                f"{stem}_{self.hemisphere}_{start_date.replace('-', '')}-"
                f"{end_date.replace('-', '')}_{frequency}_i.nc"
            )
            out_filepath = os.path.join(output_dir, out_name)
            return out_name, out_filepath

        expected_outputs: List[Tuple[str, str]] = [_resolve_obs_output(obs_file) for obs_file in obs_list]
        if (not overwrite) and expected_outputs:
            all_ready = True
            for _, out_filepath in expected_outputs:
                if (not os.path.exists(out_filepath)) or (not self._is_nonempty_time_file(out_filepath)):
                    all_ready = False
                    break
            if all_ready:
                logger.info(
                    "Observation preprocessing skipped - reusing %d processed files.",
                    len(expected_outputs),
                )
                return [out_name for out_name, _ in expected_outputs]

        logger.info("-" * 20)
        logger.info("Processing observation data ...")
        logger.info("-" * 20)

        # -----------------------------------------------------------------------------
        # Function: `_preferred_obs_vars`.
        # Purpose: Handle preferred obs vars workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _preferred_obs_vars() -> List[str]:
            """Resolve preferred observation variable names to keep before interpolation."""
            preferred: List[str] = []
            if self.module_name == 'SICB':
                if flag_name == '_sic':
                    preferred.append(str(module_vars.get('ref_var_sic', 'siconc')).strip())
                elif flag_name == '_sidrift':
                    preferred.append(str(module_vars.get('ref_var_u', 'u')).strip())
                    preferred.append(str(module_vars.get('ref_var_v', 'v')).strip())
            elif self.module_name == 'SIconc' or flag_name == '_sic':
                preferred.append(str(module_vars.get('ref_var', 'siconc')).strip())
            else:
                ref_var = str(module_vars.get('ref_var', '')).strip()
                if ref_var:
                    preferred.append(ref_var)

            deduped: List[str] = []
            for var in preferred:
                if var and var not in deduped:
                    deduped.append(var)
            return deduped

        keep_obs_vars = _preferred_obs_vars()

        # -----------------------------------------------------------------------------
        # Function: `_process_one_obs`.
        # Purpose: Handle process one obs workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _process_one_obs(task: Tuple[int, str]) -> Tuple[int, Optional[str]]:
            obs_idx, obs_file = task
            out_name, out_filepath = _resolve_obs_output(obs_file)
            worker_cdo = cdo if safe_jobs == 1 else self._new_cdo_instance()
            # Fast path: skip lock wait when a valid output is already available.
            if (not overwrite) and os.path.exists(out_filepath):
                if self._is_nonempty_time_file(out_filepath):
                    return obs_idx, out_name

            logger.info("  %s", obs_file)

            ref_dir = self._get_reference_directory(flag_name)
            in_filepath = os.path.join(self.ref_data_path, ref_dir, obs_file)

            if not os.path.exists(in_filepath):
                raise ValueError(f"File not found: {in_filepath}")

            # Build staged output outside path lock. Keep lock only for final publish.
            with self._tempdir(prefix='sitool_') as tmp:
                if frequency == 'monthly':
                    t0 = os.path.join(tmp, 't0.nc')
                    t1 = os.path.join(tmp, 't1.nc')
                    # Slice first to avoid running monmean on the full historical span.
                    self._safe_seldate(worker_cdo, start_date, end_date, os.path.abspath(in_filepath), t0)
                    self._safe_monmean(worker_cdo, t0, t1)
                    current = t1
                else:
                    current = os.path.abspath(in_filepath)

                t2 = os.path.join(tmp, 't2.nc')
                self._safe_seldate(worker_cdo, start_date, end_date, current, t2)
                interp_input = t2
                if keep_obs_vars:
                    t3 = os.path.join(tmp, 't3.nc')
                    sel_expr = ','.join(keep_obs_vars)
                    try:
                        worker_cdo.selvar(sel_expr, input=t2, output=t3)
                        interp_input = t3
                    except Exception as sel_err:
                        logger.warning(
                            "Observation selvar(%s) failed for %s (%s).",
                            sel_expr, obs_file, sel_err,
                        )
                        try:
                            with utils.open_netcdf_dataset(t2) as ds_sel:
                                existing = [v for v in keep_obs_vars if v in ds_sel.data_vars]
                                if existing:
                                    ds_keep = ds_sel[existing]
                                else:
                                    ds_keep = ds_sel
                                utils.write_netcdf_compressed(ds_keep, t3)
                                interp_input = t3
                        except Exception:
                            interp_input = t2

                staged_output = os.path.join(tmp, 'out_interp.nc')
                utils.stable_interpolation(grid_txt, interp_input, staged_output, worker_cdo)
                if not os.path.exists(staged_output):
                    raise RuntimeError(f"Failed to create staged observation output: {staged_output}")

                with _acquire_path_locks([out_filepath]):
                    if os.path.exists(out_filepath):
                        if overwrite:
                            logger.info("  Overwriting existing processed observation: %s", out_name)
                            try:
                                os.remove(out_filepath)
                            except OSError:
                                pass
                        else:
                            if self._is_nonempty_time_file(out_filepath):
                                return obs_idx, out_name
                            logger.warning(
                                "Processed observation has empty time axis; rebuilding: %s",
                                out_name,
                            )
                            try:
                                os.remove(out_filepath)
                            except OSError:
                                pass

                    os.replace(staged_output, os.path.abspath(out_filepath))

            return obs_idx, out_name

        tasks = list(enumerate(obs_list))
        if safe_jobs > 1 and len(tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results: Dict[int, Optional[str]] = {}
            with ThreadPoolExecutor(max_workers=safe_jobs, thread_name_prefix='prep-obs') as pool:
                fut_map = {
                    pool.submit(rte.run_tracked_task, _process_one_obs, task): task[0]
                    for task in tasks
                }
                for fut in as_completed(fut_map):
                    idx, out_name = fut.result()
                    results[idx] = out_name
            for idx in sorted(results.keys()):
                if results[idx]:
                    processed_files.append(results[idx])
        else:
            for task in tasks:
                _, out_name = _process_one_obs(task)
                if out_name:
                    processed_files.append(out_name)

        logger.info("Observation preprocessing completed.")
        return processed_files

    # -----------------------------------------------------------------------------
    # Function: `prep_models`.
    # Purpose: Handle prep models workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_models(self, file_groups: List[List[str]], frequency: str = None,
                    output_dir: str = None, overwrite: bool = False,
                    freq: str = None, out_dir: str = None, jobs: int = 1) -> List[str]:
        """Preprocess model data for evaluation.

        Args:
            file_groups: Groups of model files to process.
            frequency: Temporal frequency ('daily' or 'monthly').
            output_dir: Output directory for processed files.
            overwrite: Whether to rebuild model output files even if they already exist.
            freq: Alias for *frequency* (backward compatibility).
            out_dir: Alias for *output_dir* (backward compatibility).
            jobs: Maximum number of worker threads for per-group preprocessing.

        Returns:
            List of processed filenames (basenames only).
        """
        frequency = frequency or freq
        output_dir = output_dir or out_dir

        # Common setup: validate frequency, generate grid, build date range, create CDO
        # (the shared CDO instance is only used in serial mode).
        grid_txt, start_date, end_date, cdo, module_vars = \
            self._prepare_common(frequency, output_dir)
        year_range = module_vars['year_range']
        model_target_var = str(module_vars.get('model_var', '')).strip()

        processed_files: List[str] = []

        base_stems_by_group: Dict[int, str] = {}
        stem_counts: Dict[str, int] = {}
        for gidx, gfiles in enumerate(file_groups):
            if not gfiles:
                continue
            last_file = gfiles[-1]
            base_stem = re.sub(r'_\d{6,8}-\d{6,8}$', '', os.path.splitext(os.path.basename(last_file))[0])
            base_stems_by_group[gidx] = base_stem
            stem_counts[base_stem] = stem_counts.get(base_stem, 0) + 1

        def _safe_output_token(text: str, fallback: str) -> str:
            token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text or '')).strip('._-')
            return token or fallback

        # -----------------------------------------------------------------------------
        # Function: `_resolve_output`.
        # Purpose: Handle resolve output workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _resolve_output(group_idx: int, group_files: List[str]) -> Tuple[Optional[str], Optional[str]]:
            if not group_files:
                return None, None
            last_file = group_files[-1]
            base_stem = base_stems_by_group.get(group_idx)
            if not base_stem:
                base_stem = re.sub(r'_\d{6,8}-\d{6,8}$', '', os.path.splitext(os.path.basename(last_file))[0])
            if stem_counts.get(base_stem, 0) > 1:
                parent_token = _safe_output_token(
                    os.path.basename(os.path.dirname(last_file)),
                    f'model_{group_idx + 1}',
                )
                base_stem = f'{parent_token}_{base_stem}'
            out_name = (
                f"{base_stem}_{self.hemisphere}_"
                f"{year_range[0]}01-{year_range[1]}12_{frequency}_i.nc"
            )
            out_path = os.path.join(output_dir, out_name)
            return out_name, out_path

        # Detect output-name collisions before running any parallel worker.
        out_registry: Dict[str, int] = {}
        for gidx, gfiles in enumerate(file_groups):
            out_name, _ = _resolve_output(gidx, gfiles)
            if not out_name:
                continue
            if out_name in out_registry:
                raise ValueError(
                    f"prep_models output name collision: group {out_registry[out_name] + 1} and "
                    f"group {gidx + 1} both map to '{out_name}'."
                )
            out_registry[out_name] = gidx

        expected_outputs: List[Tuple[str, str]] = []
        for gidx, gfiles in enumerate(file_groups):
            out_name, out_path = _resolve_output(gidx, gfiles)
            if out_name and out_path:
                expected_outputs.append((out_name, out_path))
        if (not overwrite) and expected_outputs:
            all_ready = True
            for _, out_path in expected_outputs:
                if (not os.path.exists(out_path)) or (not self._is_nonempty_time_file(out_path)):
                    all_ready = False
                    break
            if all_ready:
                logger.info(
                    "Model preprocessing skipped - reusing %d processed files.",
                    len(expected_outputs),
                )
                return [out_name for out_name, _ in expected_outputs]

        logger.info("-" * 20)
        logger.info("Processing model data ...")
        logger.info("-" * 20)

        safe_jobs = max(1, int(jobs))
        if safe_jobs > 1:
            force_serial = str(os.environ.get('SITOOL_PREP_FORCE_SERIAL', '0')).strip().lower() in {
                '1', 'true', 'yes', 'on',
            }
            max_parallel = max(1, int(os.environ.get('SITOOL_PREP_MAX_PARALLEL', str(safe_jobs))))
            if force_serial:
                logger.info(
                    "Model preprocessing forced to serial mode by SITOOL_PREP_FORCE_SERIAL "
                    "(requested=%d -> 1).",
                    safe_jobs,
                )
                safe_jobs = 1
            else:
                if safe_jobs > max_parallel:
                    logger.info(
                        "Capping model preprocessing jobs via SITOOL_PREP_MAX_PARALLEL "
                        "(requested=%d -> %d).",
                        safe_jobs, max_parallel,
                    )
                safe_jobs = min(safe_jobs, max_parallel)

        # Hierarchical worker layout:
        # - group_workers control how many model groups run at the same time,
        # - segment_workers control per-group segment fan-out.
        # This lets `jobs` remain fully utilized even when group count is small
        # but each group contains many time-split files.
        group_workers, segment_workers = _plan_nested_workers(safe_jobs, len(file_groups))
        is_highres_case = str(self.eval_ex or '').strip().lower().startswith('highres')
        balance_segments_raw = str(os.environ.get('SITOOL_PREP_BALANCE_SEGMENTS', 'auto')).strip().lower()
        if balance_segments_raw in {'0', 'false', 'no', 'off'}:
            enable_segment_rebalance = False
        elif balance_segments_raw in {'1', 'true', 'yes', 'on'}:
            enable_segment_rebalance = True
        else:
            enable_segment_rebalance = is_highres_case

        try:
            target_segment_workers = max(
                2,
                int(os.environ.get('SITOOL_PREP_TARGET_SEGMENT_WORKERS', '4')),
            )
        except Exception:
            target_segment_workers = 4

        # Segment-rebalance heuristic: only apply when the number of model groups is
        # *fewer* than the available job budget.  In that case _plan_nested_workers
        # already assigns group_workers = n_groups < safe_jobs, leaving idle slots
        # that can be filled by giving each group extra segment workers.
        #
        # When groups >> jobs (e.g. SICB with 42 groups and 5 jobs) the optimal
        # layout is already group_workers=safe_jobs, segment_workers=1 — all workers
        # stay busy processing different groups.  Rebalancing in that scenario
        # wrongly reduces group_workers (e.g. 5→2) and hurts overall throughput.
        #
        # Previous condition used `len(file_groups) >= safe_jobs` which triggered for
        # ANY case where groups ≥ jobs (the common SICB case), cutting concurrency in
        # half.  Corrected to `len(file_groups) < safe_jobs` so rebalance only fires
        # when there really are too few groups to keep all workers occupied.
        if (
            enable_segment_rebalance
            and safe_jobs >= 4
            and len(file_groups) < safe_jobs  # only rebalance when groups are scarce
            and segment_workers == 1
        ):
            desired_inner = min(target_segment_workers, max(2, safe_jobs // 2))
            rebalance_group_workers = max(1, min(len(file_groups), safe_jobs // desired_inner))
            rebalance_segment_workers = max(1, safe_jobs // max(1, rebalance_group_workers))
            while rebalance_group_workers * rebalance_segment_workers > safe_jobs and rebalance_segment_workers > 1:
                rebalance_segment_workers -= 1
            if rebalance_group_workers >= 2 and rebalance_segment_workers >= 2:
                logger.info(
                    "Rebalanced model preprocess workers for segment fan-out: "
                    "group_workers=%d->%d, segment_workers=%d->%d, jobs=%d, groups=%d.",
                    group_workers, rebalance_group_workers,
                    segment_workers, rebalance_segment_workers,
                    safe_jobs, len(file_groups),
                )
                group_workers = rebalance_group_workers
                segment_workers = rebalance_segment_workers

        segment_backend_raw = str(os.environ.get('SITOOL_PREP_SEGMENT_BACKEND', 'auto')).strip().lower()
        try:
            segment_process_min_files = max(
                2,
                int(os.environ.get('SITOOL_PREP_SEGMENT_PROCESS_MIN_FILES', '4')),
            )
        except Exception:
            segment_process_min_files = 4
        try:
            segment_process_max_workers = max(
                1,
                int(os.environ.get('SITOOL_PREP_SEGMENT_PROCESS_MAX_WORKERS', '6')),
            )
        except Exception:
            segment_process_max_workers = 6
        # Prevent oversubscription: when all job slots are already consumed by group_workers
        # (the normal SICB layout: 42 groups, 9 jobs → group_workers=9, segment_workers=1),
        # allowing 6 segment procs per group thread would spawn 9×6=54 CDO processes on a
        # 9-job budget, causing 3× CPU oversubscription and scheduler thrashing.
        # Cap so that group_workers × segment_process_max_workers ≤ safe_jobs.
        if group_workers > 1:
            segment_process_max_workers = max(
                1, min(segment_process_max_workers, safe_jobs // group_workers)
            )
        try:
            segment_process_timeout_seconds = float(
                os.environ.get('SITOOL_PREP_SEGMENT_PROCESS_TIMEOUT_SECONDS', '1800')
            )
        except Exception:
            segment_process_timeout_seconds = 1800.0
        segment_process_timeout_seconds = max(60.0, float(segment_process_timeout_seconds))
        module_name_norm = str(self.module_name or '').strip().upper()
        is_sicb_module = module_name_norm == 'SICB'

        prep_pipeline_v2_raw = str(
            os.environ.get('SITOOL_SICB_PREP_PIPELINE_V2', 'auto')
        ).strip().lower()
        if prep_pipeline_v2_raw in {'0', 'false', 'no', 'off'}:
            enable_prep_pipeline_v2 = False
        elif prep_pipeline_v2_raw in {'1', 'true', 'yes', 'on'}:
            enable_prep_pipeline_v2 = True
        else:
            enable_prep_pipeline_v2 = bool(is_sicb_module and is_highres_case)

        sanitize_bounds_raw = str(
            os.environ.get('SITOOL_PREP_SANITIZE_BOUNDS', 'auto')
        ).strip().lower()
        if sanitize_bounds_raw in {'0', 'false', 'no', 'off'}:
            enable_bounds_sanitize = False
        elif sanitize_bounds_raw in {'1', 'true', 'yes', 'on'}:
            enable_bounds_sanitize = True
        else:
            # Safety-first default:
            # keep OFF in auto mode because some unstructured source grids rely on
            # spatial bounds for conservative remapping.
            enable_bounds_sanitize = False

        segment_pool_mode_raw = str(
            os.environ.get('SITOOL_PREP_SEGMENT_POOL_MODE', 'auto')
        ).strip().lower()
        if segment_pool_mode_raw in {'legacy', 'per_segment', 'old'}:
            use_segment_pool_backend = False
        elif segment_pool_mode_raw in {'pool', 'worker_pool', 'new'}:
            use_segment_pool_backend = True
        else:
            use_segment_pool_backend = bool(enable_prep_pipeline_v2)

        try:
            process_disable_fail_ratio = float(
                os.environ.get('SITOOL_PREP_SEGMENT_DISABLE_FAIL_RATIO', '0.25')
            )
        except Exception:
            process_disable_fail_ratio = 0.25
        process_disable_fail_ratio = max(0.0, min(1.0, process_disable_fail_ratio))
        try:
            process_disable_min_groups = max(
                1, int(os.environ.get('SITOOL_PREP_SEGMENT_DISABLE_MIN_GROUPS', '2'))
            )
        except Exception:
            process_disable_min_groups = 2

        _stats_lock = threading.Lock()
        _segment_process_disabled = threading.Event()
        prep_stats: Dict[str, int] = {
            'group_retries': 0,
            'segment_xarray_fallback': 0,
            'segment_sanitize_success': 0,
            'segment_sanitize_failed': 0,
            'segment_process_groups': 0,
            'segment_process_failed_groups': 0,
            'segment_process_total_segments': 0,
            'segment_process_failed_segments': 0,
        }

        # -----------------------------------------------------------------------------
        # Function: `_stat_inc`.
        # Purpose: Handle stat inc workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _stat_inc(key: str, delta: int = 1) -> None:
            with _stats_lock:
                prep_stats[key] = int(prep_stats.get(key, 0)) + int(delta)

        # -----------------------------------------------------------------------------
        # Function: `_maybe_disable_process_segments`.
        # Purpose: Handle maybe disable process segments workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _maybe_disable_process_segments() -> None:
            with _stats_lock:
                groups = int(prep_stats.get('segment_process_groups', 0))
                failed = int(prep_stats.get('segment_process_failed_segments', 0))
                total = int(prep_stats.get('segment_process_total_segments', 0))
            if groups < process_disable_min_groups or total <= 0:
                return
            fail_ratio = float(failed) / float(total)
            if fail_ratio >= process_disable_fail_ratio and (not _segment_process_disabled.is_set()):
                _segment_process_disabled.set()
                logger.warning(
                    "Disabling segment process backend for remaining groups due to high failure ratio: "
                    "failed_segments=%d total_segments=%d ratio=%.1f%% (threshold=%.1f%%).",
                    failed, total, fail_ratio * 100.0, process_disable_fail_ratio * 100.0,
                )

        # -----------------------------------------------------------------------------
        # Function: `_sanitize_ready_for_merge`.
        # Purpose: Handle sanitize ready for merge workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _sanitize_ready_for_merge(src_path: str, dst_path: str, preferred_var: str) -> bool:
            """Normalize one prepared segment for stable merge/remap.

            Removes known problematic bounds helper variables/attrs that can cause
            inconsistent variable-definition failures in CDO mergetime/remap.
            """
            if not os.path.exists(src_path):
                return False
            with utils.open_netcdf_dataset(src_path) as ds_src:
                ds_work = ds_src
                if preferred_var and preferred_var in ds_work.data_vars:
                    ds_work = ds_work[[preferred_var]]
                elif preferred_var and preferred_var in ds_work.variables and preferred_var not in ds_work.coords:
                    ds_work = ds_work[[preferred_var]]
                elif len(ds_work.data_vars) == 1:
                    only_var = list(ds_work.data_vars)[0]
                    ds_work = ds_work[[only_var]]
                elif len(ds_work.data_vars) > 1:
                    # Keep all data vars when target variable is ambiguous.
                    ds_work = ds_work[list(ds_work.data_vars)]

                # Keep spatial bounds variables (e.g. lon_bnds/lat_bnds) because
                # conservative remapping may require them. Only strip time bounds.
                drop_bounds = [
                    str(vn) for vn in ds_work.variables
                    if str(vn).lower() in {'time_bnds', 'time_bounds'}
                ]
                if drop_bounds:
                    ds_work = ds_work.drop_vars(drop_bounds, errors='ignore')

                for var_name in list(ds_work.variables):
                    try:
                        if str(var_name).lower() == 'time':
                            ds_work[var_name].attrs.pop('bounds', None)
                    except Exception:
                        pass
                for coord_name in list(ds_work.coords):
                    try:
                        if str(coord_name).lower() == 'time':
                            ds_work[coord_name].attrs.pop('bounds', None)
                    except Exception:
                        pass

                if 'time' in ds_work.coords:
                    try:
                        ds_work = ds_work.sortby('time')
                    except Exception:
                        pass

                if 'time' in ds_work.sizes and int(ds_work.sizes.get('time', 0)) <= 0:
                    return False
                utils.write_netcdf_compressed(ds_work, dst_path)
            return os.path.exists(dst_path)
        logger.info(
            "Model preprocess runtime flags [%s/%s]: pipeline_v2=%s, segment_pool=%s, "
            "sanitize_bounds=%s, process_disable_ratio=%.2f, process_disable_min_groups=%d, "
            "segment_process_timeout_s=%.0f.",
            str(self.eval_ex), str(self.module_name),
            enable_prep_pipeline_v2,
            ('pool' if use_segment_pool_backend else 'legacy'),
            enable_bounds_sanitize,
            process_disable_fail_ratio,
            process_disable_min_groups,
            segment_process_timeout_seconds,
        )

        # -----------------------------------------------------------------------------
        # Function: `_prefer_segment_process_backend`.
        # Purpose: Handle prefer segment process backend workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _prefer_segment_process_backend(seg_count: int, local_workers: int) -> bool:
            if seg_count <= 1:
                return False
            if _segment_process_disabled.is_set():
                return False
            if segment_backend_raw in {'process', 'proc', 'multiprocess', 'mp'}:
                return True
            if segment_backend_raw in {'thread', 'threads'}:
                return False
            # Auto mode: enable for highres with enough segments.
            # When group_workers == safe_jobs (groups >> jobs, the normal SICB layout),
            # _plan_nested_workers returns segment_workers=1, so local_workers==1.
            # In that case, process-level segment parallelism via
            # segment_process_max_workers is still beneficial and must not be blocked
            # by the local_workers guard — the two levels are independent.
            is_highres = str(self.eval_ex or '').strip().lower().startswith('highres')
            if not (is_highres and seg_count >= segment_process_min_files):
                # Non-highres or too few segments: require thread-level workers > 1.
                return local_workers > 1
            # Highres with enough segments: allow if either thread workers or dedicated
            # segment process workers can provide parallelism.
            return local_workers > 1 or int(segment_process_max_workers) > 1

        # -----------------------------------------------------------------------------
        # Function: `_process_one_group`.
        # Purpose: Handle process one group workflow for the current execution path.
        # Maintenance: Keep input/output schema stable for upstream and downstream callers.
        # -----------------------------------------------------------------------------
        def _process_one_group(task: Tuple[int, List[str]], force_serial: bool = False) -> Optional[str]:
            group_idx, file_group = task
            if not file_group:
                return None

            out_name, out_filepath = _resolve_output(group_idx, file_group)
            if out_name is None or out_filepath is None:
                return None
            local_segment_workers = 1 if force_serial else segment_workers

            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                if attempt == 1:
                    logger.info("  Processing group %d", group_idx + 1)
                else:
                    _stat_inc('group_retries', 1)
                    logger.warning(
                        "  Retrying group %d (%d/%d) after transient failure.",
                        group_idx + 1, attempt, max_attempts,
                    )
                try:
                    if (not overwrite) and os.path.exists(out_filepath):
                        if self._is_nonempty_time_file(out_filepath):
                            return out_name
                        logger.warning(
                            "Existing processed model has empty time axis; rebuilding: %s",
                            out_name,
                        )

                    # Build staged output outside lock to minimize cross-module lock hold time.
                    with nullcontext():
                        if overwrite and os.path.exists(out_filepath):
                            logger.info(
                                "    Existing processed file will be overwritten after staged build: %s",
                                out_name,
                            )

                        # Each group uses one temporary root directory. Every segment
                        # gets its own child directory so segment-level workers never
                        # collide on temporary file names.
                        with self._tempdir(prefix=f'sitool_g{group_idx}_') as tmp:
                            # -----------------------------------------------------------------------------
                            # Function: `_process_one_segment`.
                            # Purpose: Handle process one segment workflow for the current execution path.
                            # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                            # -----------------------------------------------------------------------------
                            def _process_one_segment(seg_task: Tuple[int, str]) -> Tuple[int, Optional[str]]:
                                fi, input_file = seg_task
                                in_path = input_file
                                logger.info("    %s", os.path.basename(in_path))

                                if not os.path.exists(in_path):
                                    logger.warning("Input file not found: %s", input_file)
                                    return fi, None

                                seg_dir = os.path.join(tmp, f'seg_{fi:04d}')
                                os.makedirs(seg_dir, exist_ok=True)
                                tmp_freq = os.path.join(seg_dir, 'f.nc')
                                tmp_miss = os.path.join(seg_dir, 'm.nc')
                                tmp_ready = os.path.join(seg_dir, 'ready.nc')
                                worker_cdo = cdo if (safe_jobs == 1 and local_segment_workers == 1) else self._new_cdo_instance()
                                target_var = model_target_var
                                if not target_var:
                                    guess = os.path.basename(in_path).split('_')[0].strip()
                                    if re.match(r'^[A-Za-z][A-Za-z0-9_]*$', guess):
                                        target_var = guess

                                # -----------------------------------------------------------------------------
                                # Function: `_xarray_segment_fallback`.
                                # Purpose: Handle xarray segment fallback workflow for the current execution path.
                                # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                                # -----------------------------------------------------------------------------
                                def _xarray_segment_fallback(src_path: str, dst_path: str) -> bool:
                                    """Fallback segment preparation path when CDO fails.

                                    This path is intentionally conservative:
                                    - temporal subsetting uses xarray;
                                    - optional monthly aggregation mirrors CDO monmean behavior;
                                    - target variable is selected when available;
                                    - non-finite values are mapped to a stable fill value.
                                    """
                                    if not os.path.exists(src_path):
                                        return False
                                    _stat_inc('segment_xarray_fallback', 1)
                                    with utils.open_netcdf_dataset(src_path) as ds_src:
                                        ds_work = ds_src
                                        if 'time' in ds_work.coords or 'time' in ds_work.dims:
                                            ds_work = ds_work.sel(time=slice(start_date, end_date))
                                            if frequency == 'monthly':
                                                ds_work = ds_work.resample(time='MS').mean(keep_attrs=True)

                                        if target_var and target_var in ds_work.data_vars:
                                            ds_work = ds_work[[target_var]]
                                        elif not list(ds_work.data_vars):
                                            return False

                                        if 'time' in ds_work.sizes and int(ds_work.sizes.get('time', 0)) <= 0:
                                            return False

                                        for var_name in list(ds_work.data_vars):
                                            da = ds_work[var_name]
                                            fill_raw = da.attrs.get('_FillValue', da.attrs.get('missing_value', -9999.0))
                                            try:
                                                fill_val = float(fill_raw)
                                            except Exception:
                                                fill_val = -9999.0
                                            ds_work[var_name] = da.where(np.isfinite(da), other=fill_val)
                                            ds_work[var_name].attrs.setdefault('_FillValue', fill_val)
                                            ds_work[var_name].attrs.setdefault('missing_value', fill_val)

                                        utils.write_netcdf_compressed(ds_work, dst_path)
                                    return os.path.exists(dst_path)

                                try:
                                    if frequency == 'monthly':
                                        tmp_selraw = os.path.join(seg_dir, 'selraw.nc')
                                        # Slice first to avoid running monmean on the full historical span.
                                        self._safe_seldate(
                                            worker_cdo, start_date, end_date,
                                            os.path.abspath(in_path), tmp_selraw,
                                        )
                                        self._safe_monmean(worker_cdo, tmp_selraw, tmp_freq)
                                        current = tmp_freq
                                    else:
                                        tmp_selraw = os.path.join(seg_dir, 'selraw.nc')
                                        self._safe_seldate(
                                            worker_cdo, start_date, end_date,
                                            os.path.abspath(in_path), tmp_selraw,
                                        )
                                        current = tmp_selraw

                                    if frequency == 'monthly' and not os.path.exists(tmp_freq):
                                        logger.error("Failed to create monthly mean: %s", tmp_freq)
                                        return fi, None
                                    if frequency != 'monthly' and not os.path.exists(current):
                                        logger.error(
                                            "Failed to slice daily date range for segment: %s",
                                            os.path.basename(in_path),
                                        )
                                        return fi, None

                                    if target_var:
                                        tmp_var = os.path.join(seg_dir, 'var.nc')
                                        try:
                                            worker_cdo.selvar(target_var, input=current, output=tmp_var)
                                            current = tmp_var
                                        except Exception as sel_err:
                                            logger.warning(
                                                "selvar(%s) failed for segment %s (%s) — using original variables.",
                                                target_var, os.path.basename(in_path), sel_err,
                                            )

                                    try:
                                        worker_cdo.setmissval(-9999, input=current, output=tmp_miss)
                                        if not os.path.exists(tmp_miss):
                                            raise RuntimeError(f"Failed to set missing value for segment: {tmp_miss}")
                                        os.replace(tmp_miss, tmp_ready)
                                    except Exception as miss_err:
                                        logger.warning(
                                            "setmissval failed for segment %d in group %d (%s): %s. "
                                            "Trying xarray fallback.",
                                            fi + 1, group_idx + 1, os.path.basename(in_path), miss_err,
                                        )
                                        fallback_ok = False
                                        for fallback_src in (current, tmp_selraw, os.path.abspath(in_path)):
                                            try:
                                                if _xarray_segment_fallback(str(fallback_src), tmp_ready):
                                                    fallback_ok = True
                                                    break
                                            except Exception:
                                                continue
                                        if not fallback_ok:
                                            return fi, None

                                    if not self._is_nonempty_time_file(tmp_ready):
                                        logger.warning(
                                            "Segment produced empty time axis; skipping segment %d in group %d (%s).",
                                            fi + 1, group_idx + 1, os.path.basename(in_path),
                                        )
                                        fallback_ok = False
                                        for fallback_src in (tmp_selraw, os.path.abspath(in_path)):
                                            try:
                                                if _xarray_segment_fallback(str(fallback_src), tmp_ready):
                                                    fallback_ok = self._is_nonempty_time_file(tmp_ready)
                                                    if fallback_ok:
                                                        break
                                            except Exception:
                                                continue
                                        if not fallback_ok:
                                            return fi, None
                                    if enable_bounds_sanitize:
                                        tmp_clean = os.path.join(seg_dir, 'ready_clean.nc')
                                        try:
                                            if _sanitize_ready_for_merge(tmp_ready, tmp_clean, target_var):
                                                tmp_ready = tmp_clean
                                                _stat_inc('segment_sanitize_success', 1)
                                            else:
                                                logger.warning(
                                                    "Segment sanitize generated empty output; keeping original: "
                                                    "group=%d segment=%d (%s).",
                                                    group_idx + 1, fi + 1, os.path.basename(in_path),
                                                )
                                        except Exception as clean_err:
                                            _stat_inc('segment_sanitize_failed', 1)
                                            logger.warning(
                                                "Segment sanitize failed for group %d segment %d (%s): %s. "
                                                "Continue with unsanitized segment.",
                                                group_idx + 1, fi + 1, os.path.basename(in_path), clean_err,
                                            )

                                    # P8 — Apply per-segment remap when weight cache is
                                    # available.  `_group_weight_file` is built once per
                                    # group (outer scope) before the segment loop starts,
                                    # and is reused here by every segment in this group.
                                    # When the weight file is absent (cache disabled or
                                    # weight build failed) this block is skipped and the
                                    # remap falls back to the original single-remap-after-
                                    # merge path below.
                                    if _enable_remap_per_seg and _group_weight_file:
                                        tmp_seg_interp = os.path.join(seg_dir, 'seg_interp.nc')
                                        try:
                                            utils.stable_interpolation(
                                                grid_txt,
                                                tmp_ready,
                                                tmp_seg_interp,
                                                worker_cdo,
                                                weight_file=_group_weight_file,
                                            )
                                            if os.path.exists(tmp_seg_interp) and os.path.getsize(tmp_seg_interp) > 0:
                                                return fi, tmp_seg_interp
                                            # Empty output: fall back to returning the
                                            # un-remapped file for the merge-then-remap path.
                                            logger.warning(
                                                "Per-segment remap produced empty output for "
                                                "group %d segment %d; using fallback path.",
                                                group_idx + 1, fi + 1,
                                            )
                                        except Exception as remap_err:
                                            logger.warning(
                                                "Per-segment remap failed for group %d segment %d (%s): %s. "
                                                "Using fallback merge-then-remap path.",
                                                group_idx + 1, fi + 1,
                                                os.path.basename(in_path), remap_err,
                                            )

                                    return fi, tmp_ready
                                except Exception as exc:
                                    logger.error(
                                        "Segment preprocessing failed for group %d, segment %d (%s): %s",
                                        group_idx + 1, fi + 1, os.path.basename(in_path), exc,
                                    )
                                    return fi, None

                            # ------------------------------------------------------------------
                            # P8 — Remap-per-segment: build a shared remap weight cache for
                            # this group so every segment can call `cdo remap <weights>`
                            # instead of recomputing the full weight matrix each time.
                            #
                            # Weight computation is the dominant cost for high-resolution
                            # regular grids (e.g. EC-Earth3P-HR ~127 min, ECMWF-IFS-HR
                            # ~55 min when done on the merged 96-month file).  By moving
                            # the remap into _process_one_segment we:
                            #   1. Compute weights once and reuse across all N segments.
                            #   2. Run N small-file remaps in parallel (segment_workers).
                            #   3. Final mergetime operates on the (small) target grid
                            #      instead of the (large) source grid.
                            #
                            # IMPORTANT — side-effect guard: weight building uses `cdo
                            # griddes` which takes 30-120 s on high-resolution files.
                            # When many group_workers run concurrently each one blocks on
                            # the per-key lock while the first builder runs griddes, causing
                            # implicit serialisation of the build phase.  For monthly data
                            # (SIconc, SIthick, SItrans, SNdepth) the groups have only 2-8
                            # short segments; the build overhead far exceeds the remap
                            # savings, making total time *worse*.
                            #
                            # Rules:
                            #   - Only enable for daily data (monthly = too few segments).
                            #   - Only enable when the group has ≥ SITOOL_PREP_REMAP_MIN_SEGS
                            #     segments (default 4) so the one-time build is amortised.
                            #   - Disabled via SITOOL_REMAP_WEIGHT_CACHE=0 or
                            #     SITOOL_PREP_REMAP_PER_SEG=0.
                            # ------------------------------------------------------------------
                            _remap_per_seg_raw = str(
                                os.environ.get('SITOOL_PREP_REMAP_PER_SEG', 'auto')
                            ).strip().lower()
                            try:
                                _remap_min_segs = max(
                                    1, int(os.environ.get('SITOOL_PREP_REMAP_MIN_SEGS', '4'))
                                )
                            except Exception:
                                _remap_min_segs = 4

                            _enable_remap_per_seg = (
                                _remap_per_seg_raw not in {'0', 'false', 'no', 'off'}
                                and len(file_group) >= _remap_min_segs
                                and os.path.exists(grid_txt)
                                # Daily data only: monthly modules (SIconc, SIthick, etc.)
                                # have too few short segments to amortise weight build cost.
                                and frequency == 'daily'
                            )

                            # Build weight cache directory under the case temp root so it
                            # persists for the duration of the run and is shared across
                            # hemispheres (the target grid is the same for nh/sh).
                            _weight_cache_dir = os.path.join(
                                self.temp_root, 'remap_weight_cache'
                            )
                            _group_weight_file: Optional[str] = None

                            if _enable_remap_per_seg and file_group:
                                # Use the first available segment file as the grid probe —
                                # all files in the group share the same source grid.
                                _probe_file = next(
                                    (f for f in file_group if os.path.exists(f)), None
                                )
                                if _probe_file:
                                    # Build/retrieve weight file; a new CDO instance is used
                                    # here so the griddes call does not interfere with the
                                    # shared CDO instance used for the group.
                                    _probe_cdo = self._new_cdo_instance()
                                    _group_weight_file = utils.build_remap_weights(
                                        grid_file=grid_txt,
                                        source_file=_probe_file,
                                        weight_dir=_weight_cache_dir,
                                        cdo_instance=_probe_cdo,
                                        method='bilinear',
                                    )
                                    if _group_weight_file:
                                        logger.info(
                                            "  Group %d: using cached remap weights (%s) "
                                            "for %d segment(s).",
                                            group_idx + 1,
                                            os.path.basename(_group_weight_file),
                                            len(file_group),
                                        )
                                    else:
                                        # Weight cache unavailable; fall back to the
                                        # original single-remap-after-merge strategy.
                                        _enable_remap_per_seg = False

                            # ------------------------------------------------------------------
                            # P10 — Large single-file year splitting (AWI-HR and similar).
                            #
                            # Models like AWI-CM-1-1-HR store 10 years of daily data in a
                            # single file (11 GB, 3652 time steps × 1.3 M unstructured grid
                            # cells).  CDO monmean on such a file is sequential and cannot
                            # be parallelised; it dominates the group wall time (~218 min).
                            #
                            # Fix: when a group contains exactly one file and that file has
                            # more than SITOOL_PREP_SPLIT_YEAR_THRESH time steps, split it
                            # into annual pseudo-segments using CDO selyear.  These smaller
                            # files (365 steps each) then flow through the normal parallel
                            # segment pipeline (including the per-segment remap in P8).
                            #
                            # The split produces temporary files under the group's tmp dir
                            # so nothing is written to the permanent processed directory.
                            # Disabled via SITOOL_PREP_SPLIT_LARGE_FILE=0.
                            # ------------------------------------------------------------------
                            _split_raw = str(
                                os.environ.get('SITOOL_PREP_SPLIT_LARGE_FILE', 'auto')
                            ).strip().lower()
                            _enable_split = (
                                _split_raw not in {'0', 'false', 'no', 'off'}
                                and frequency == 'daily'
                                and len(file_group) == 1
                            )
                            try:
                                _split_thresh = int(
                                    os.environ.get('SITOOL_PREP_SPLIT_YEAR_THRESH', '730')
                                )
                            except Exception:
                                _split_thresh = 730

                            if _enable_split and len(file_group) == 1:
                                _single_file = file_group[0]
                                _tsteps = self._dataset_time_size(_single_file)
                                if _tsteps is not None and _tsteps > _split_thresh:
                                    # Determine which years overlap the evaluation range.
                                    _eval_start_yr = int(str(start_date)[:4])
                                    _eval_end_yr   = int(str(end_date)[:4])
                                    _split_dir = os.path.join(tmp, 'split_years')
                                    os.makedirs(_split_dir, exist_ok=True)
                                    _split_files: List[str] = []
                                    _split_cdo = self._new_cdo_instance()
                                    logger.info(
                                        "  Group %d: splitting large single file (%d time steps)"
                                        " into annual segments (years %d–%d).",
                                        group_idx + 1, _tsteps,
                                        _eval_start_yr, _eval_end_yr,
                                    )
                                    _split_ok = True
                                    for _yr in range(_eval_start_yr, _eval_end_yr + 1):
                                        _yr_out = os.path.join(
                                            _split_dir,
                                            f'{os.path.splitext(os.path.basename(_single_file))[0]}'
                                            f'_yr{_yr}.nc'
                                        )
                                        try:
                                            _split_cdo.selyear(
                                                str(_yr),
                                                input=os.path.abspath(_single_file),
                                                output=_yr_out,
                                            )
                                            if os.path.exists(_yr_out) and os.path.getsize(_yr_out) > 0:
                                                _split_files.append(_yr_out)
                                            else:
                                                logger.warning(
                                                    "  Group %d: selyear(%d) produced empty output; "
                                                    "abandoning split.",
                                                    group_idx + 1, _yr,
                                                )
                                                _split_ok = False
                                                break
                                        except Exception as _split_exc:
                                            logger.warning(
                                                "  Group %d: selyear(%d) failed (%s); "
                                                "abandoning split.",
                                                group_idx + 1, _yr, _split_exc,
                                            )
                                            _split_ok = False
                                            break

                                    if _split_ok and _split_files:
                                        logger.info(
                                            "  Group %d: split into %d annual segments.",
                                            group_idx + 1, len(_split_files),
                                        )
                                        # Replace the single-file group with the annual segments.
                                        # The weight cache probe was built from the original file
                                        # whose grid is identical to the annual splits, so the
                                        # cached weights remain valid.
                                        file_group = _split_files
                                    else:
                                        logger.warning(
                                            "  Group %d: year split failed; processing as single file.",
                                            group_idx + 1,
                                        )

                            seg_tasks = list(enumerate(file_group))
                            # -----------------------------------------------------------------------------
                            # Function: `_run_segments_with_threads`.
                            # Purpose: Handle run segments with threads workflow for the current execution path.
                            # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                            # -----------------------------------------------------------------------------
                            def _run_segments_with_threads(seg_task_items: List[Tuple[int, str]]) -> Dict[int, Optional[str]]:
                                segment_out: Dict[int, Optional[str]] = {}
                                if local_segment_workers > 1 and len(seg_task_items) > 1:
                                    from concurrent.futures import ThreadPoolExecutor, as_completed
                                    inner_workers = min(local_segment_workers, len(seg_task_items))
                                    with ThreadPoolExecutor(max_workers=inner_workers, thread_name_prefix=f'prep-model-g{group_idx}') as seg_pool:
                                        fut_map = {
                                            seg_pool.submit(rte.run_tracked_task, _process_one_segment, seg_task): seg_task[0]
                                            for seg_task in seg_task_items
                                        }
                                        for fut in as_completed(fut_map):
                                            fi, interp_path = fut.result()
                                            segment_out[fi] = interp_path
                                else:
                                    for seg_task in seg_task_items:
                                        fi, interp_path = _process_one_segment(seg_task)
                                        segment_out[fi] = interp_path
                                return segment_out

                            # -----------------------------------------------------------------------------
                            # Function: `_run_segments_with_processes_legacy`.
                            # Purpose: Handle run segments with processes legacy workflow for the current execution path.
                            # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                            # -----------------------------------------------------------------------------
                            def _run_segments_with_processes_legacy(
                                seg_task_items: List[Tuple[int, str]]
                            ) -> Dict[int, Optional[str]]:
                                segment_out: Dict[int, Optional[str]] = {}
                                if not seg_task_items:
                                    return segment_out

                                # _seg_proc_entry is a nested closure: forkserver cannot pickle it.
                                # Detect this upfront and fall back to threads immediately rather
                                # than failing per-group after spawn attempts.
                                import pickle as _pickle
                                try:
                                    _pickle.dumps(_seg_proc_entry)
                                except Exception:
                                    return _run_segments_with_threads(seg_task_items)

                                try:
                                    seg_ctx = mp.get_context('forkserver')
                                except Exception as exc:
                                    logger.warning(
                                        "Segment process backend unavailable for group %d (%s); fallback to threads.",
                                        group_idx + 1, exc,
                                    )
                                    return _run_segments_with_threads(seg_task_items)

                                from collections import deque as _deque
                                # When segment_workers==1 (groups >> jobs layout) the
                                # process backend is still beneficial; use
                                # segment_process_max_workers as the concurrency budget
                                # rather than being capped at 1 by local_segment_workers.
                                _proc_budget = max(
                                    int(local_segment_workers),
                                    int(segment_process_max_workers),
                                )
                                max_seg_workers = max(
                                    1,
                                    min(
                                        _proc_budget,
                                        len(seg_task_items),
                                    ),
                                )
                                pending = _deque(seg_task_items)
                                running: Dict[int, Any] = {}
                                reported: set = set()
                                result_q = seg_ctx.Queue()

                                # -----------------------------------------------------------------------------
                                # Function: `_seg_proc_entry`.
                                # Purpose: Handle seg proc entry workflow for the current execution path.
                                # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                                # -----------------------------------------------------------------------------
                                def _seg_proc_entry(seg_payload: Tuple[int, str], queue_obj) -> None:
                                    fi = int(seg_payload[0])
                                    try:
                                        fi_ret, interp_path = _process_one_segment(seg_payload)
                                        queue_obj.put({
                                            'fi': fi_ret, 'ok': True, 'interp_path': interp_path, 'error': None,
                                        })
                                    except Exception as exc:
                                        queue_obj.put({
                                            'fi': fi, 'ok': False, 'interp_path': None, 'error': str(exc),
                                        })

                                try:
                                    loop_started = time.monotonic()
                                    while pending or running:
                                        if (time.monotonic() - loop_started) >= segment_process_timeout_seconds:
                                            active_segments = sorted([
                                                int(fi) + 1 for fi, proc in running.items()
                                                if proc.is_alive()
                                            ])
                                            active_preview = ','.join(str(x) for x in active_segments[:8])
                                            if len(active_segments) > 8:
                                                active_preview = f"{active_preview},..."
                                            logger.warning(
                                                "Segment process backend timeout in group %d after %.1fs; "
                                                "pending=%d running=%d active_segments=[%s]. "
                                                "Terminating stuck workers and falling back to thread backend "
                                                "for unfinished segments. "
                                                "You can tune SITOOL_PREP_SEGMENT_PROCESS_TIMEOUT_SECONDS.",
                                                group_idx + 1,
                                                float(time.monotonic() - loop_started),
                                                len(pending),
                                                len(running),
                                                active_preview,
                                            )
                                            break

                                        while pending and len(running) < max_seg_workers:
                                            seg_payload = pending.popleft()
                                            fi = int(seg_payload[0])
                                            proc = seg_ctx.Process(
                                                target=_seg_proc_entry,
                                                args=(seg_payload, result_q),
                                                name=f'prep-seg-g{group_idx:03d}-{fi:04d}',
                                            )
                                            proc.daemon = False
                                            try:
                                                proc.start()
                                            except AttributeError as _pickle_err:
                                                # _seg_proc_entry is a nested closure and cannot be
                                                # pickled by forkserver. Fallback all remaining work
                                                # to the thread backend immediately.
                                                logger.warning(
                                                    "Segment process backend unavailable for group %d "
                                                    "(pickle error: %s); falling back to threads.",
                                                    group_idx + 1, _pickle_err,
                                                )
                                                pending.appendleft(seg_payload)
                                                raise
                                            running[fi] = proc

                                        try:
                                            msg = result_q.get(timeout=0.5)
                                            fi = int(msg.get('fi'))
                                            segment_out[fi] = msg.get('interp_path')
                                            reported.add(fi)
                                            if not bool(msg.get('ok', True)):
                                                logger.warning(
                                                    "Segment process failed in group %d segment %d (%s).",
                                                    group_idx + 1, fi + 1,
                                                    msg.get('error') or 'unknown error',
                                                )
                                        except queue_mod.Empty:
                                            pass

                                        finished: List[int] = []
                                        for fi, proc in running.items():
                                            if not proc.is_alive():
                                                proc.join(timeout=0.1)
                                                finished.append(fi)
                                        for fi in finished:
                                            proc = running.pop(fi)
                                            if fi not in reported and fi not in segment_out:
                                                segment_out[fi] = None
                                                if proc.exitcode not in (0, None):
                                                    logger.warning(
                                                        "Segment process exited with non-zero code "
                                                        "for group %d segment %d: %s.",
                                                        group_idx + 1, fi + 1, proc.exitcode,
                                                    )
                                finally:
                                    for proc in running.values():
                                        try:
                                            if proc.is_alive():
                                                proc.terminate()
                                            proc.join(timeout=0.2)
                                        except Exception:
                                            pass
                                    try:
                                        result_q.close()
                                        result_q.join_thread()
                                    except Exception:
                                        pass

                                return segment_out

                            # -----------------------------------------------------------------------------
                            # Function: `_run_segments_with_process_pool`.
                            # Purpose: Handle run segments with process pool workflow for the current execution path.
                            # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                            # -----------------------------------------------------------------------------
                            def _run_segments_with_process_pool(
                                seg_task_items: List[Tuple[int, str]]
                            ) -> Dict[int, Optional[str]]:
                                segment_out: Dict[int, Optional[str]] = {}
                                if not seg_task_items:
                                    return segment_out

                                try:
                                    seg_ctx = mp.get_context('forkserver')
                                except Exception as exc:
                                    logger.warning(
                                        "Segment process backend unavailable for group %d (%s); fallback to threads.",
                                        group_idx + 1, exc,
                                    )
                                    return _run_segments_with_threads(seg_task_items)

                                # Mirror the legacy backend fix: use segment_process_max_workers
                                # as the concurrency budget so that segment_workers==1 layouts
                                # (groups >> jobs) still get process-level parallelism.
                                _proc_budget_pool = max(
                                    int(local_segment_workers),
                                    int(segment_process_max_workers),
                                )
                                max_seg_workers = max(
                                    1,
                                    min(
                                        _proc_budget_pool,
                                        len(seg_task_items),
                                    ),
                                )
                                if max_seg_workers <= 1:
                                    return _run_segments_with_threads(seg_task_items)

                                task_q = seg_ctx.Queue()
                                result_q = seg_ctx.Queue()
                                workers: List[Any] = []
                                expected = len(seg_task_items)
                                received = 0
                                reported: set = set()

                                # -----------------------------------------------------------------------------
                                # Function: `_seg_proc_loop`.
                                # Purpose: Handle seg proc loop workflow for the current execution path.
                                # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                                # -----------------------------------------------------------------------------
                                def _seg_proc_loop(task_queue, result_queue) -> None:
                                    while True:
                                        payload = task_queue.get()
                                        if payload is None:
                                            break
                                        fi = int(payload[0])
                                        try:
                                            fi_ret, interp_path = _process_one_segment(payload)
                                            result_queue.put({
                                                'fi': fi_ret,
                                                'ok': True,
                                                'interp_path': interp_path,
                                                'error': None,
                                            })
                                        except Exception as exc:
                                            result_queue.put({
                                                'fi': fi,
                                                'ok': False,
                                                'interp_path': None,
                                                'error': str(exc),
                                            })

                                try:
                                    loop_started = time.monotonic()
                                    for wi in range(max_seg_workers):
                                        proc = seg_ctx.Process(
                                            target=_seg_proc_loop,
                                            args=(task_q, result_q),
                                            name=f'prep-segpool-g{group_idx:03d}-w{wi:02d}',
                                        )
                                        proc.daemon = False
                                        proc.start()
                                        workers.append(proc)

                                    for seg_payload in seg_task_items:
                                        task_q.put(seg_payload)
                                    for _ in workers:
                                        task_q.put(None)

                                    while received < expected:
                                        if (time.monotonic() - loop_started) >= segment_process_timeout_seconds:
                                            live_workers = sum(1 for p in workers if p.is_alive())
                                            logger.warning(
                                                "Segment process pool timeout in group %d after %.1fs; "
                                                "received=%d/%d live_workers=%d. "
                                                "Terminating pool workers and falling back to thread backend "
                                                "for unfinished segments. "
                                                "You can tune SITOOL_PREP_SEGMENT_PROCESS_TIMEOUT_SECONDS.",
                                                group_idx + 1,
                                                float(time.monotonic() - loop_started),
                                                int(received),
                                                int(expected),
                                                int(live_workers),
                                            )
                                            break

                                        try:
                                            msg = result_q.get(timeout=0.5)
                                            fi = int(msg.get('fi'))
                                            segment_out[fi] = msg.get('interp_path')
                                            reported.add(fi)
                                            received += 1
                                            if not bool(msg.get('ok', True)):
                                                logger.warning(
                                                    "Segment process failed in group %d segment %d (%s).",
                                                    group_idx + 1, fi + 1,
                                                    msg.get('error') or 'unknown error',
                                                )
                                        except queue_mod.Empty:
                                            if workers and all((not p.is_alive()) for p in workers):
                                                break

                                    for proc in workers:
                                        try:
                                            proc.join(timeout=0.2)
                                        except Exception:
                                            pass

                                    for seg_payload in seg_task_items:
                                        fi = int(seg_payload[0])
                                        if fi not in segment_out:
                                            segment_out[fi] = None
                                finally:
                                    for proc in workers:
                                        try:
                                            if proc.is_alive():
                                                proc.terminate()
                                            proc.join(timeout=0.2)
                                        except Exception:
                                            pass
                                    try:
                                        task_q.close()
                                        task_q.join_thread()
                                    except Exception:
                                        pass
                                    try:
                                        result_q.close()
                                        result_q.join_thread()
                                    except Exception:
                                        pass

                                return segment_out

                            # -----------------------------------------------------------------------------
                            # Function: `_run_segments_with_processes`.
                            # Purpose: Handle run segments with processes workflow for the current execution path.
                            # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                            # -----------------------------------------------------------------------------
                            def _run_segments_with_processes(
                                seg_task_items: List[Tuple[int, str]]
                            ) -> Dict[int, Optional[str]]:
                                if use_segment_pool_backend:
                                    try:
                                        return _run_segments_with_process_pool(seg_task_items)
                                    except Exception as exc:
                                        logger.warning(
                                            "Segment process pool backend failed for group %d (%s); "
                                            "fallback to legacy process backend.",
                                            group_idx + 1, exc,
                                        )
                                return _run_segments_with_processes_legacy(seg_task_items)

                            if _prefer_segment_process_backend(len(seg_tasks), local_segment_workers):
                                logger.info(
                                    "  Group %d uses process segment backend: segments=%d, workers=%d.",
                                    group_idx + 1, len(seg_tasks),
                                    max(
                                        1,
                                        min(
                                            local_segment_workers,
                                            len(seg_tasks),
                                            int(segment_process_max_workers),
                                        ),
                                    ),
                                )
                                interp_results = _run_segments_with_processes(seg_tasks)
                                failed_seg_tasks = [
                                    seg_task for seg_task in seg_tasks
                                    if not interp_results.get(seg_task[0])
                                ]
                                _stat_inc('segment_process_groups', 1)
                                _stat_inc('segment_process_total_segments', len(seg_tasks))
                                if failed_seg_tasks:
                                    _stat_inc('segment_process_failed_groups', 1)
                                    _stat_inc('segment_process_failed_segments', len(failed_seg_tasks))
                                if failed_seg_tasks:
                                    logger.warning(
                                        "  Group %d process segment backend failed on %d/%d segments; "
                                        "retrying failed segments with thread backend.",
                                        group_idx + 1, len(failed_seg_tasks), len(seg_tasks),
                                    )
                                    fallback_results = _run_segments_with_threads(failed_seg_tasks)
                                    interp_results.update(fallback_results)
                                if segment_backend_raw not in {'process', 'proc', 'multiprocess', 'mp'}:
                                    _maybe_disable_process_segments()
                            else:
                                interp_results = _run_segments_with_threads(seg_tasks)

                            ready_files = [
                                interp_results[fi]
                                for fi in sorted(interp_results.keys())
                                if interp_results[fi]
                            ]

                            if not ready_files:
                                logger.error("No valid prepared files for group %d", group_idx)
                                return None

                            merge_cdo = cdo if (safe_jobs == 1 and local_segment_workers == 1) else self._new_cdo_instance()
                            merged = os.path.join(tmp, 'merged.nc')
                            merge_cdo.mergetime(
                                input=' '.join(os.path.abspath(f) for f in ready_files),
                                output=merged,
                            )

                            if _enable_remap_per_seg and _group_weight_file:
                                # Segments were already remapped to the target grid.
                                # Only a date-range trim is needed; no remap step.
                                # Check that the first ready file lives on the target grid
                                # by testing for the weight file presence (proxy).
                                staged_output = os.path.join(tmp, 'out_interp.nc')
                                self._safe_seldate(merge_cdo, start_date, end_date, merged, staged_output)
                                logger.info(
                                    "  Group %d: skipped post-merge remap (per-segment remap used).",
                                    group_idx + 1,
                                )
                            else:
                                # Original path: remap the merged full-time-series file.
                                selected = os.path.join(tmp, 'selected.nc')
                                self._safe_seldate(merge_cdo, start_date, end_date, merged, selected)
                                staged_output = os.path.join(tmp, 'out_interp.nc')
                                utils.stable_interpolation(grid_txt, selected, staged_output, merge_cdo)

                            if not os.path.exists(staged_output):
                                raise RuntimeError(f"Failed to create staged output: {staged_output}")

                            # Apply model_var_nan_fill if set in the recipe.
                            # Intended for datasets that store NaN for physically valid zero
                            # (e.g. siconc with cell_methods: area mean where sea_ice).
                            nan_fill_raw = module_vars.get('model_var_nan_fill')
                            if nan_fill_raw is not None:
                                try:
                                    nan_fill_v = float(nan_fill_raw)
                                    nan_fill_tmp = staged_output + '.nanfill.nc'
                                    with utils.open_netcdf_dataset(staged_output) as _ds:
                                        _ds_out = _ds.copy()
                                        if model_target_var and model_target_var in _ds_out.data_vars:
                                            _ds_out[model_target_var] = _ds_out[model_target_var].fillna(nan_fill_v)
                                        utils.write_netcdf_compressed(_ds_out, nan_fill_tmp)
                                    os.replace(nan_fill_tmp, staged_output)
                                    logger.info(
                                        "  Group %d: NaN→%g in '%s' (model_var_nan_fill).",
                                        group_idx + 1, nan_fill_v, model_target_var,
                                    )
                                except Exception as _nf_exc:
                                    logger.warning(
                                        "  model_var_nan_fill step failed for group %d (skipped): %s",
                                        group_idx + 1, _nf_exc,
                                    )

                            with _acquire_path_locks([out_filepath]):
                                if os.path.exists(out_filepath):
                                    if overwrite:
                                        logger.info("    Overwriting existing processed file: %s", out_name)
                                        try:
                                            os.remove(out_filepath)
                                        except OSError:
                                            pass
                                    else:
                                        if self._is_nonempty_time_file(out_filepath):
                                            return out_name
                                        logger.warning(
                                            "Existing processed model has empty time axis; rebuilding: %s",
                                            out_name,
                                        )
                                        try:
                                            os.remove(out_filepath)
                                        except OSError:
                                            pass

                                os.replace(staged_output, os.path.abspath(out_filepath))

                            if os.path.exists(out_filepath):
                                logger.info("Created: %s", out_name)
                                return out_name
                            raise RuntimeError(f"Failed to publish: {out_filepath}")
                except Exception as exc:
                    if attempt < max_attempts:
                        logger.warning(
                            "Group %d preprocessing attempt %d/%d failed: %s",
                            group_idx + 1, attempt, max_attempts, exc,
                        )
                        time.sleep(float(attempt))
                        continue
                    logger.error(
                        "Group %d preprocessing failed after %d attempts: %s",
                        group_idx + 1, max_attempts, exc, exc_info=True,
                    )
                    return None

        tasks = [(idx, grp) for idx, grp in enumerate(file_groups)]

        # ------------------------------------------------------------------
        # Longest-Processing-Time (LPT) scheduling: sort groups by estimated
        # cost in *descending* order before submitting to the thread/process
        # pool.  Starting the slowest groups first minimises tail latency
        # because workers pick up smaller groups as the long ones drain.
        # Without this, a single slow group (e.g. AWI unstructured-grid or
        # large HadGEM3 files) can sit idle until a worker frees up late in
        # the run, extending total wall time by 30-60 minutes.
        #
        # Cost proxy: total on-disk file size of the group, with an extra
        # multiplier for unstructured-grid models (AWI, FESOM, ICON) that
        # cannot use bilinear interpolation and fall back to conservative
        # remapping — empirically 3-5x slower per MB than regular models.
        #
        # The sort is applied only in parallel mode and can be disabled via
        # SITOOL_PREP_LPT_SORT=0.  The original group indices (stored in
        # task[0]) are preserved, so downstream result collection is unchanged.
        # ------------------------------------------------------------------
        _lpt_sort_raw = str(
            os.environ.get('SITOOL_PREP_LPT_SORT', 'auto')
        ).strip().lower()
        enable_lpt_sort = (
            _lpt_sort_raw not in {'0', 'false', 'no', 'off'}
            and safe_jobs > 1
            and len(tasks) > 1
        )

        if enable_lpt_sort:
            # Models known to use unstructured grids where remapbil fails and
            # CDO must fall back to remapnn/remapcon (significantly slower).
            _UNSTRUCTURED_HINTS: frozenset = frozenset({
                'awi', 'fesom', 'icon', 'nemo', 'fvcom',
            })

            def _group_cost(task: tuple) -> float:
                """Estimate processing cost for a group of model files.

                Uses total on-disk file size as the primary signal, scaled by
                an unstructured-grid penalty when the model name matches known
                non-regular grids.  Returns a float for descending sort.
                """
                _, grp_files = task
                if not grp_files:
                    return 0.0
                total_bytes = 0
                for fpath in grp_files:
                    try:
                        total_bytes += os.path.getsize(fpath)
                    except OSError:
                        # Path unavailable (e.g. symlink not resolved); treat
                        # as 1 byte so the group still sorts after known sizes.
                        total_bytes += 1
                # Detect unstructured grids from the first file's basename.
                # CMIP6 DRS format: <var>_<table>_<model>_<exp>_...nc, so the
                # model name appears early and contains the hint token.
                first_name = os.path.basename(grp_files[0]).lower()
                is_unstructured = any(hint in first_name for hint in _UNSTRUCTURED_HINTS)
                # Empirical penalty: unstructured models require conservative
                # remapping (~4x slower per byte than bilinear interpolation).
                unstructured_penalty = 4.0 if is_unstructured else 1.0
                return float(total_bytes) * unstructured_penalty

            tasks_sorted = sorted(tasks, key=_group_cost, reverse=True)

            # Log the reordering for diagnosability in worker logs.
            if tasks_sorted != tasks:
                def _group_label(task: tuple) -> str:
                    _, grp_files = task
                    return os.path.basename(grp_files[0]) if grp_files else '(empty)'

                top_labels = [_group_label(t) for t in tasks_sorted[:3]]
                bot_labels = [_group_label(t) for t in tasks_sorted[-3:]]
                logger.info(
                    "LPT group scheduling: reordered %d groups by estimated cost "
                    "(slowest-first). Top-3: [%s] ... Bottom-3: [%s]",
                    len(tasks_sorted),
                    ', '.join(top_labels),
                    ', '.join(bot_labels),
                )
            tasks = tasks_sorted

        try:
            group_process_timeout_seconds = float(
                os.environ.get('SITOOL_PREP_GROUP_PROCESS_TIMEOUT_SECONDS', '3600')
            )
        except Exception:
            group_process_timeout_seconds = 3600.0
        group_process_timeout_seconds = max(120.0, float(group_process_timeout_seconds))
        if safe_jobs > 1 and len(tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from collections import deque

            # -----------------------------------------------------------------------------
            # Function: `_run_groups_with_threads`.
            # Purpose: Handle run groups with threads workflow for the current execution path.
            # Maintenance: Keep input/output schema stable for upstream and downstream callers.
            # -----------------------------------------------------------------------------
            def _run_groups_with_threads(task_items: List[Tuple[int, List[str]]]) -> Dict[int, Optional[str]]:
                out: Dict[int, Optional[str]] = {}
                with ThreadPoolExecutor(max_workers=group_workers, thread_name_prefix='prep-model') as pool:
                    if segment_workers > 1:
                        fut_map = {pool.submit(_process_one_group, task, False): task[0] for task in task_items}
                    else:
                        fut_map = {
                            pool.submit(rte.run_tracked_task, _process_one_group, task, False): task[0]
                            for task in task_items
                        }
                    for fut in as_completed(fut_map):
                        gidx = fut_map[fut]
                        out[gidx] = fut.result()
                return out

            # -----------------------------------------------------------------------------
            # Function: `_run_groups_with_processes`.
            # Purpose: Handle run groups with processes workflow for the current execution path.
            # Maintenance: Keep input/output schema stable for upstream and downstream callers.
            # -----------------------------------------------------------------------------
            def _run_groups_with_processes(task_items: List[Tuple[int, List[str]]]) -> Dict[int, Optional[str]]:
                out: Dict[int, Optional[str]] = {}
                if not task_items:
                    return out

                try:
                    ctx = mp.get_context('forkserver')
                except Exception as exc:
                    logger.warning(
                        "Process backend unavailable for model preprocessing (%s); fallback to threads.",
                        exc,
                    )
                    return _run_groups_with_threads(task_items)

                max_proc_workers = max(1, min(group_workers, len(task_items)))
                pending = deque(task_items)
                running: Dict[int, Any] = {}
                reported: set = set()
                result_q = ctx.Queue()

                # -----------------------------------------------------------------------------
                # Function: `_proc_entry`.
                # Purpose: Handle proc entry workflow for the current execution path.
                # Maintenance: Keep input/output schema stable for upstream and downstream callers.
                # -----------------------------------------------------------------------------
                def _proc_entry(task_payload: Tuple[int, List[str]], queue_obj) -> None:
                    gidx = int(task_payload[0])
                    try:
                        out_name = _process_one_group(task_payload, False)
                        queue_obj.put({'gidx': gidx, 'ok': True, 'out_name': out_name, 'error': None})
                    except Exception as exc:
                        queue_obj.put({'gidx': gidx, 'ok': False, 'out_name': None, 'error': str(exc)})

                try:
                    loop_started = time.monotonic()
                    while pending or running:
                        if (time.monotonic() - loop_started) >= group_process_timeout_seconds:
                            active_groups = sorted([
                                int(gidx) + 1 for gidx, proc in running.items()
                                if proc.is_alive()
                            ])
                            active_preview = ','.join(str(x) for x in active_groups[:8])
                            if len(active_groups) > 8:
                                active_preview = f"{active_preview},..."
                            logger.warning(
                                "Grouped process backend timeout after %.1fs; pending=%d running=%d "
                                "active_groups=[%s]. Terminating stuck group workers and falling back to serial "
                                "retry for unfinished groups. "
                                "You can tune SITOOL_PREP_GROUP_PROCESS_TIMEOUT_SECONDS.",
                                float(time.monotonic() - loop_started),
                                len(pending),
                                len(running),
                                active_preview,
                            )
                            break

                        while pending and len(running) < max_proc_workers:
                            task_payload = pending.popleft()
                            gidx = int(task_payload[0])
                            proc = ctx.Process(
                                target=_proc_entry,
                                args=(task_payload, result_q),
                                name=f'prep-model-g{gidx:03d}',
                            )
                            proc.daemon = False
                            proc.start()
                            running[gidx] = proc

                        try:
                            msg = result_q.get(timeout=0.5)
                            gidx = int(msg.get('gidx'))
                            out[gidx] = msg.get('out_name')
                            reported.add(gidx)
                            if not bool(msg.get('ok', True)):
                                logger.warning(
                                    "Group %d process task failed (%s).",
                                    gidx + 1, msg.get('error') or 'unknown error',
                                )
                        except queue_mod.Empty:
                            pass

                        finished: List[int] = []
                        for gidx, proc in running.items():
                            if not proc.is_alive():
                                proc.join(timeout=0.1)
                                finished.append(gidx)
                        for gidx in finished:
                            proc = running.pop(gidx)
                            if gidx not in reported and gidx not in out:
                                out[gidx] = None
                                if proc.exitcode not in (0, None):
                                    logger.warning(
                                        "Group %d process exited with non-zero code %s.",
                                        gidx + 1, proc.exitcode,
                                    )
                finally:
                    for proc in running.values():
                        try:
                            if proc.is_alive():
                                proc.terminate()
                            proc.join(timeout=0.2)
                        except Exception:
                            pass
                    try:
                        result_q.close()
                        result_q.join_thread()
                    except Exception:
                        pass

                return out

            group_backend_raw = str(os.environ.get('SITOOL_PREP_GROUP_BACKEND', 'auto')).strip().lower()
            if group_backend_raw in {'process', 'proc', 'multiprocess', 'mp'}:
                use_process_backend = True
            elif group_backend_raw in {'thread', 'threads'}:
                use_process_backend = False
            else:
                # Keep grouped backend conservative in auto mode.
                # Process backend is still available via explicit env var.
                use_process_backend = False

            if use_process_backend:
                logger.info(
                    "Using process backend for grouped model preprocessing: groups=%d, workers=%d.",
                    len(tasks), max(1, min(group_workers, len(tasks))),
                )
                results = _run_groups_with_processes(tasks)
            else:
                results = _run_groups_with_threads(tasks)

            missing_after_parallel = [task for task in tasks if not results.get(task[0])]
            if missing_after_parallel:
                logger.warning(
                    "Retrying %d failed/empty model groups in serial fallback mode.",
                    len(missing_after_parallel),
                )
            for task in missing_after_parallel:
                gidx = task[0]
                results[gidx] = _process_one_group(task, True)
            failed_groups = [gidx + 1 for gidx in sorted(results.keys()) if not results[gidx]]
            if failed_groups:
                raise RuntimeError(
                    f"Model preprocessing failed for group indices: {failed_groups}. "
                    "Please retry with fewer jobs or check source files."
                )
            for gidx in sorted(results.keys()):
                if results[gidx]:
                    processed_files.append(results[gidx])
        else:
            failed_serial: List[int] = []
            for task in tasks:
                gidx = task[0]
                out_name = _process_one_group(task, True)
                if out_name:
                    processed_files.append(out_name)
                else:
                    failed_serial.append(gidx + 1)
            if failed_serial:
                raise RuntimeError(
                    f"Model preprocessing failed in serial mode. Missing groups: {failed_serial}."
                )

        logger.info(
            "Model preprocessing completed. runtime_stats: "
            "group_retries=%d, xarray_fallback=%d, sanitize_ok=%d, sanitize_failed=%d, "
            "segment_process_groups=%d, segment_process_failed_groups=%d, "
            "segment_process_failed_segments=%d/%d, segment_pool_mode=%s",
            int(prep_stats.get('group_retries', 0)),
            int(prep_stats.get('segment_xarray_fallback', 0)),
            int(prep_stats.get('segment_sanitize_success', 0)),
            int(prep_stats.get('segment_sanitize_failed', 0)),
            int(prep_stats.get('segment_process_groups', 0)),
            int(prep_stats.get('segment_process_failed_groups', 0)),
            int(prep_stats.get('segment_process_failed_segments', 0)),
            int(prep_stats.get('segment_process_total_segments', 0)),
            ('pool' if use_segment_pool_backend else 'legacy'),
        )
        return processed_files

    # -----------------------------------------------------------------------------
    # Function: `_extract_date_suffix`.
    # Purpose: Handle extract date suffix workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _extract_date_suffix(path: str) -> Optional[str]:
        """Extract date suffix token like 20000101-20091231 from filename."""
        name = os.path.basename(path)
        m = re.search(r'_(\d{6,8}-\d{6,8})\.nc$', name)
        return m.group(1) if m else None

    # -----------------------------------------------------------------------------
    # Function: `_pair_model_segments`.
    # Purpose: Handle pair model segments workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def _pair_model_segments(self, u_group: List[str], v_group: List[str]) -> List[Tuple[str, str]]:
        """Pair u/v segment files by date suffix when available."""
        if not u_group or not v_group:
            return []

        u_map = {self._extract_date_suffix(p): p for p in u_group}
        v_map = {self._extract_date_suffix(p): p for p in v_group}
        keys = [k for k in u_map if k is not None]

        if keys and all(k in v_map for k in keys) and len(keys) == len(u_group) == len(v_group):
            ordered = sorted(keys)
            return [(u_map[k], v_map[k]) for k in ordered]

        if len(u_group) != len(v_group):
            raise ValueError(
                f"u/v segment count mismatch: {len(u_group)} vs {len(v_group)}."
            )

        return list(zip(sorted(u_group), sorted(v_group)))

    # -----------------------------------------------------------------------------
    # Function: `_mask_missing`.
    # Purpose: Handle mask missing workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _mask_missing(arr: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
        """Convert common fill/missing sentinels to NaN."""
        out = np.asarray(arr, dtype=float)
        fill_candidates = [
            attrs.get('_FillValue'),
            attrs.get('missing_value'),
            1.0e20,
            1.0e30,
            -9999.0,
        ]
        for fv in fill_candidates:
            try:
                if fv is None:
                    continue
                fv_val = float(fv)
            except Exception:
                continue
            out[np.isclose(out, fv_val)] = np.nan
        out[np.abs(out) > 1.0e19] = np.nan
        return out

    # -----------------------------------------------------------------------------
    # Function: `_extract_lon_lat_arrays`.
    # Purpose: Handle extract lon lat arrays workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _extract_lon_lat_arrays(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract longitude/latitude arrays from common variable naming conventions."""
        lon_candidates = ['lon', 'longitude', 'nav_lon', 'glamt', 'xlon']
        lat_candidates = ['lat', 'latitude', 'nav_lat', 'gphit', 'xlat']

        lon_name = next((k for k in lon_candidates if k in ds.variables), None)
        lat_name = next((k for k in lat_candidates if k in ds.variables), None)

        if lon_name is None or lat_name is None:
            for var_name, da in ds.variables.items():
                attrs = {str(k).lower(): str(v).lower() for k, v in da.attrs.items()}
                std_name = attrs.get('standard_name', '')
                units = attrs.get('units', '')
                if lon_name is None and ('longitude' in std_name or units == 'degrees_east'):
                    lon_name = var_name
                if lat_name is None and ('latitude' in std_name or units == 'degrees_north'):
                    lat_name = var_name
                if lon_name is not None and lat_name is not None:
                    break

        if lon_name is None or lat_name is None:
            raise KeyError(
                f"Cannot find lon/lat variables in dataset. Variables: {list(ds.variables)[:20]}"
            )

        return np.array(ds[lon_name]), np.array(ds[lat_name])

    # -----------------------------------------------------------------------------
    # Function: `_detect_theta_unit`.
    # Purpose: Handle detect theta unit workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    @staticmethod
    def _detect_theta_unit(theta_attrs: Dict[str, Any]) -> str:
        """Infer theta unit from metadata."""
        txt = " ".join([str(k).lower() + " " + str(v).lower() for k, v in theta_attrs.items()])
        if 'degree' in txt:
            return 'degree'
        if 'radian' in txt:
            return 'radian'
        return 'auto'

    # -----------------------------------------------------------------------------
    # Function: `_get_reference_directory`.
    # Purpose: Handle get reference directory workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def _get_reference_directory(self, flag_name: str = '') -> str:
        """Return the reference data sub-directory for the current module.

        Args:
            flag_name: Optional suffix (e.g., '_sic', '_sidrift').

        Returns:
            Sub-directory name under the reference data root.
        """
        if self.module_name == 'SItrans':
            return 'SIconc'
        if self.module_name == 'SICB':
            return 'SIconc' if flag_name == '_sic' else 'SIdrift'
        return self.module_name

__all__ = [
    "ObsModelPrepMixin",
]
