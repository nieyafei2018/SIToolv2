# -*- coding: utf-8 -*-
"""Split preprocessing mixins."""

import multiprocessing as mp
import queue as queue_mod

from scripts.preprocess import base as _base

# Reuse shared namespace (imports/constants/helpers) from base module.
globals().update({k: v for k, v in _base.__dict__.items() if k not in globals()})


class ObsModelPrepMixin:
    """Mixin with grouped preprocessing methods."""

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

        logger.info("-" * 20)
        logger.info("Processing observation data ...")
        logger.info("-" * 20)

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

        def _process_one_obs(task: Tuple[int, str]) -> Tuple[int, Optional[str]]:
            obs_idx, obs_file = task
            stem = os.path.splitext(os.path.basename(obs_file))[0]
            # Output filename encodes hemisphere, date range, and frequency for traceability
            out_name = (
                f"{stem}_{self.hemisphere}_{start_date.replace('-', '')}-"
                f"{end_date.replace('-', '')}_{frequency}_i.nc"
            )
            out_filepath = os.path.join(output_dir, out_name)
            worker_cdo = cdo if safe_jobs == 1 else self._new_cdo_instance()
            with _acquire_path_locks([out_filepath]):
                # Skip if already processed (idempotent)
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

                logger.info("  %s", obs_file)

                ref_dir = self._get_reference_directory(flag_name)
                in_filepath = os.path.join(self.ref_data_path, ref_dir, obs_file)

                if not os.path.exists(in_filepath):
                    raise ValueError(f"File not found: {in_filepath}")

                # CDO pipeline inside a temp dir to avoid leftover intermediate files:
                #   1. (monthly only) compute monthly means from sub-monthly input
                #   2. select the requested date range
                #   3. interpolate to the evaluation grid
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
                                with xr.open_dataset(t2) as ds_sel:
                                    existing = [v for v in keep_obs_vars if v in ds_sel.data_vars]
                                    if existing:
                                        ds_keep = ds_sel[existing]
                                    else:
                                        ds_keep = ds_sel
                                    utils.write_netcdf_compressed(ds_keep, t3)
                                    interp_input = t3
                            except Exception:
                                interp_input = t2
                    utils.stable_interpolation(grid_txt, interp_input, os.path.abspath(out_filepath), worker_cdo)

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

        logger.info("-" * 20)
        logger.info("Processing model data ...")
        logger.info("-" * 20)

        processed_files: List[str] = []
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

        def _resolve_output(group_idx: int, group_files: List[str]) -> Tuple[Optional[str], Optional[str]]:
            if not group_files:
                return None, None
            last_file = group_files[-1]
            base_stem = re.sub(r'_\d{6,8}-\d{6,8}$', '', os.path.splitext(os.path.basename(last_file))[0])
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

        if (
            enable_segment_rebalance
            and safe_jobs >= 4
            and len(file_groups) >= safe_jobs
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

        def _stat_inc(key: str, delta: int = 1) -> None:
            with _stats_lock:
                prep_stats[key] = int(prep_stats.get(key, 0)) + int(delta)

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

        def _sanitize_ready_for_merge(src_path: str, dst_path: str, preferred_var: str) -> bool:
            """Normalize one prepared segment for stable merge/remap.

            Removes known problematic bounds helper variables/attrs that can cause
            inconsistent variable-definition failures in CDO mergetime/remap.
            """
            if not os.path.exists(src_path):
                return False
            with xr.open_dataset(src_path) as ds_src:
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
            "sanitize_bounds=%s, process_disable_ratio=%.2f, process_disable_min_groups=%d.",
            str(self.eval_ex), str(self.module_name),
            enable_prep_pipeline_v2,
            ('pool' if use_segment_pool_backend else 'legacy'),
            enable_bounds_sanitize,
            process_disable_fail_ratio,
            process_disable_min_groups,
        )

        def _prefer_segment_process_backend(seg_count: int, local_workers: int) -> bool:
            if local_workers <= 1 or seg_count <= 1:
                return False
            if _segment_process_disabled.is_set():
                return False
            if segment_backend_raw in {'process', 'proc', 'multiprocess', 'mp'}:
                return True
            if segment_backend_raw in {'thread', 'threads'}:
                return False
            # Auto mode:
            # - enable for highres-like runs with enough file segments;
            # - allow nested process segments inside worker processes to avoid
            #   thread-unsafe NetCDF/HDF access while still exposing parallelism.
            return (
                str(self.eval_ex or '').strip().lower().startswith('highres')
                and seg_count >= segment_process_min_files
            )

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
                    with _acquire_path_locks([out_filepath]):
                        if os.path.exists(out_filepath):
                            if overwrite:
                                logger.info("    Overwriting existing processed file: %s", out_name)
                                os.remove(out_filepath)
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

                        # Each group uses one temporary root directory. Every segment
                        # gets its own child directory so segment-level workers never
                        # collide on temporary file names.
                        with self._tempdir(prefix=f'sitool_g{group_idx}_') as tmp:
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
                                target_var = str(module_vars.get('model_var', '')).strip()
                                if not target_var:
                                    guess = os.path.basename(in_path).split('_')[0].strip()
                                    if re.match(r'^[A-Za-z][A-Za-z0-9_]*$', guess):
                                        target_var = guess

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
                                    with xr.open_dataset(src_path) as ds_src:
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
                                    return fi, tmp_ready
                                except Exception as exc:
                                    logger.error(
                                        "Segment preprocessing failed for group %d, segment %d (%s): %s",
                                        group_idx + 1, fi + 1, os.path.basename(in_path), exc,
                                    )
                                    return fi, None

                            seg_tasks = list(enumerate(file_group))
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

                            def _run_segments_with_processes_legacy(
                                seg_task_items: List[Tuple[int, str]]
                            ) -> Dict[int, Optional[str]]:
                                segment_out: Dict[int, Optional[str]] = {}
                                if not seg_task_items:
                                    return segment_out

                                try:
                                    seg_ctx = mp.get_context('fork')
                                except Exception as exc:
                                    logger.warning(
                                        "Segment process backend unavailable for group %d (%s); fallback to threads.",
                                        group_idx + 1, exc,
                                    )
                                    return _run_segments_with_threads(seg_task_items)

                                from collections import deque as _deque
                                max_seg_workers = max(
                                    1,
                                    min(
                                        local_segment_workers,
                                        len(seg_task_items),
                                        int(segment_process_max_workers),
                                    ),
                                )
                                pending = _deque(seg_task_items)
                                running: Dict[int, Any] = {}
                                reported: set = set()
                                result_q = seg_ctx.Queue()

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
                                    while pending or running:
                                        while pending and len(running) < max_seg_workers:
                                            seg_payload = pending.popleft()
                                            fi = int(seg_payload[0])
                                            proc = seg_ctx.Process(
                                                target=_seg_proc_entry,
                                                args=(seg_payload, result_q),
                                                name=f'prep-seg-g{group_idx:03d}-{fi:04d}',
                                            )
                                            proc.daemon = False
                                            proc.start()
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

                            def _run_segments_with_process_pool(
                                seg_task_items: List[Tuple[int, str]]
                            ) -> Dict[int, Optional[str]]:
                                segment_out: Dict[int, Optional[str]] = {}
                                if not seg_task_items:
                                    return segment_out

                                try:
                                    seg_ctx = mp.get_context('fork')
                                except Exception as exc:
                                    logger.warning(
                                        "Segment process backend unavailable for group %d (%s); fallback to threads.",
                                        group_idx + 1, exc,
                                    )
                                    return _run_segments_with_threads(seg_task_items)

                                max_seg_workers = max(
                                    1,
                                    min(
                                        local_segment_workers,
                                        len(seg_task_items),
                                        int(segment_process_max_workers),
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
                            selected = os.path.join(tmp, 'selected.nc')
                            merge_cdo.mergetime(
                                input=' '.join(os.path.abspath(f) for f in ready_files),
                                output=merged,
                            )
                            self._safe_seldate(merge_cdo, start_date, end_date, merged, selected)
                            utils.stable_interpolation(grid_txt, selected, os.path.abspath(out_filepath), merge_cdo)
                            if os.path.exists(out_filepath):
                                logger.info("Created: %s", out_name)
                                return out_name
                            raise RuntimeError(f"Failed to create: {out_filepath}")
                except Exception as exc:
                    if os.path.exists(out_filepath):
                        try:
                            os.remove(out_filepath)
                        except OSError:
                            pass
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
        if safe_jobs > 1 and len(tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from collections import deque

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

            def _run_groups_with_processes(task_items: List[Tuple[int, List[str]]]) -> Dict[int, Optional[str]]:
                out: Dict[int, Optional[str]] = {}
                if not task_items:
                    return out

                try:
                    ctx = mp.get_context('fork')
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

                def _proc_entry(task_payload: Tuple[int, List[str]], queue_obj) -> None:
                    gidx = int(task_payload[0])
                    try:
                        out_name = _process_one_group(task_payload, False)
                        queue_obj.put({'gidx': gidx, 'ok': True, 'out_name': out_name, 'error': None})
                    except Exception as exc:
                        queue_obj.put({'gidx': gidx, 'ok': False, 'out_name': None, 'error': str(exc)})

                try:
                    while pending or running:
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

    @staticmethod
    def _extract_date_suffix(path: str) -> Optional[str]:
        """Extract date suffix token like 20000101-20091231 from filename."""
        name = os.path.basename(path)
        m = re.search(r'_(\d{6,8}-\d{6,8})\.nc$', name)
        return m.group(1) if m else None

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

    @staticmethod
    def _detect_theta_unit(theta_attrs: Dict[str, Any]) -> str:
        """Infer theta unit from metadata."""
        txt = " ".join([str(k).lower() + " " + str(v).lower() for k, v in theta_attrs.items()])
        if 'degree' in txt:
            return 'degree'
        if 'radian' in txt:
            return 'radian'
        return 'auto'

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
