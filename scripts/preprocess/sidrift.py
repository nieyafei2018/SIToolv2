# -*- coding: utf-8 -*-
"""Split preprocessing mixins."""

from scripts.preprocess import base as _base

# Reuse shared namespace (imports/constants/helpers) from base module.
globals().update({k: v for k, v in _base.__dict__.items() if k not in globals()})


class SidriftPrepMixin:
    """Mixin with grouped preprocessing methods."""

    def _load_angle_field(self, angle_spec: Any, spatial_shape: Tuple[int, ...]) -> Tuple[Optional[np.ndarray], str]:
        """Load optional model angle field from a recipe entry.

        Accepted formats:
          - \"path/to/file.nc,var_name\"
          - [\"path/to/file.nc\", \"var_name\"]
          - \"path/to/file.nc\" (auto-select first data variable)
        """
        if angle_spec is None:
            return None, 'none'
        if isinstance(angle_spec, str) and not angle_spec.strip():
            return None, 'none'

        angle_file = None
        angle_var = None
        if isinstance(angle_spec, (list, tuple)) and len(angle_spec) >= 2:
            angle_file = str(angle_spec[0]).strip()
            angle_var = str(angle_spec[1]).strip()
        elif isinstance(angle_spec, str):
            if ',' in angle_spec:
                angle_file, angle_var = [s.strip() for s in angle_spec.split(',', 1)]
            else:
                angle_file = angle_spec.strip()
                angle_var = None
        else:
            return None, 'invalid_spec'

        if not angle_file:
            return None, 'invalid_spec'

        if not os.path.exists(angle_file):
            angle_file_try = os.path.join(self.model_data_path, angle_file)
            if os.path.exists(angle_file_try):
                angle_file = angle_file_try
            else:
                logger.warning("Angle file not found: %s", angle_file)
                return None, 'file_missing'

        with xr.open_dataset(angle_file) as ds_ang:
            if angle_var is None:
                data_vars = list(ds_ang.data_vars)
                if not data_vars:
                    return None, 'empty_angle_file'
                angle_var = data_vars[0]
            if angle_var not in ds_ang.variables:
                logger.warning("Angle var '%s' not found in %s", angle_var, angle_file)
                return None, 'var_missing'

            theta = np.array(ds_ang[angle_var])
            if theta.ndim > len(spatial_shape):
                # Allow singleton time-like leading dimensions
                while theta.ndim > len(spatial_shape) and theta.shape[0] == 1:
                    theta = theta[0]
            if tuple(theta.shape) != tuple(spatial_shape):
                logger.warning(
                    "Angle shape mismatch: expected %s, got %s from %s:%s",
                    spatial_shape, theta.shape, angle_file, angle_var,
                )
                return None, 'shape_mismatch'

            unit = self._detect_theta_unit(dict(ds_ang[angle_var].attrs))
            if unit == 'degree':
                theta = np.deg2rad(theta)
                return theta, 'loaded_degree'
            return theta, f'loaded_{unit}'

    def prep_sidrift_models(
        self,
        u_file_groups: List[List[str]],
        v_file_groups: List[List[str]],
        model_direction: List[str],
        model_angle: Optional[List[Any]] = None,
        frequency: str = None,
        output_dir: str = None,
        overwrite: bool = False,
        prefer_metadata: bool = True,
        freq: str = None,
        out_dir: str = None,
        jobs: int = 1,
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Preprocess SIdrift model u/v pairs with rotation before interpolation.

        Pipeline per u/v segment:
          1) monthly/daily aggregation and date selection on source grid
          2) rotate native x/y -> east/north on source grid (when needed)
          3) interpolate east/north to evaluation grid
          4) rotate east/north -> evaluation-grid x/y on target grid

        Args:
            jobs: Maximum number of worker threads for per-model preprocessing.
                Each worker handles one model end-to-end and writes to dedicated
                output files to avoid write collisions.
        """
        frequency = frequency or freq
        output_dir = output_dir or out_dir

        grid_txt, start_date, end_date, cdo, module_vars = self._prepare_common(frequency, output_dir)
        year_range = module_vars['year_range']
        model_angle = model_angle or []

        if len(u_file_groups) != len(v_file_groups):
            raise ValueError(
                f"SIdrift u/v group count mismatch: {len(u_file_groups)} vs {len(v_file_groups)}."
            )

        logger.info("-" * 20)
        logger.info("Processing SIdrift model pairs (rotate-before-remap) ...")
        logger.info("-" * 20)

        model_u_key = module_vars.get('model_var_u', 'u')
        model_v_key = module_vars.get('model_var_v', 'v')

        out_u_files: List[str] = []
        out_v_files: List[str] = []
        audit_rows: List[Dict[str, Any]] = []
        safe_jobs = max(1, int(jobs))
        if safe_jobs > 1:
            force_serial = str(os.environ.get('SITOOL_PREP_FORCE_SERIAL', '0')).strip().lower() in {
                '1', 'true', 'yes', 'on',
            }
            max_parallel = max(1, int(os.environ.get('SITOOL_PREP_MAX_PARALLEL', str(safe_jobs))))
            if force_serial:
                logger.info(
                    "SIdrift preprocessing forced to serial mode by SITOOL_PREP_FORCE_SERIAL "
                    "(requested=%d -> 1).",
                    safe_jobs,
                )
                safe_jobs = 1
            else:
                if safe_jobs > max_parallel:
                    logger.info(
                        "Capping SIdrift preprocessing jobs via SITOOL_PREP_MAX_PARALLEL "
                        "(requested=%d -> %d).",
                        safe_jobs, max_parallel,
                    )
                safe_jobs = min(safe_jobs, max_parallel)
        model_workers, segment_workers = _plan_nested_workers(safe_jobs, len(u_file_groups))

        def _resolve_output_names(midx: int, u_group: List[str], v_group: List[str]) -> Tuple[str, str, str, str, str]:
            model_name = os.path.basename(os.path.dirname(u_group[0])) or f'model_{midx + 1}'
            u_base = re.sub(r'_\d{6,8}-\d{6,8}$', '', os.path.splitext(os.path.basename(u_group[-1]))[0])
            v_base = re.sub(r'_\d{6,8}-\d{6,8}$', '', os.path.splitext(os.path.basename(v_group[-1]))[0])
            out_u_name = f"{u_base}_{self.hemisphere}_{year_range[0]}01-{year_range[1]}12_{frequency}_i.nc"
            out_v_name = f"{v_base}_{self.hemisphere}_{year_range[0]}01-{year_range[1]}12_{frequency}_i.nc"
            out_u_path = os.path.join(output_dir, out_u_name)
            out_v_path = os.path.join(output_dir, out_v_name)
            return model_name, out_u_name, out_v_name, out_u_path, out_v_path

        # Detect output-name collisions before launching parallel workers.
        name_registry: Dict[str, int] = {}
        for midx, (u_group, v_group) in enumerate(zip(u_file_groups, v_file_groups)):
            if not u_group or not v_group:
                continue
            _model_name, out_u_name, out_v_name, _out_u_path, _out_v_path = _resolve_output_names(midx, u_group, v_group)
            for fname in (out_u_name, out_v_name):
                if fname in name_registry:
                    raise ValueError(
                        f"SIdrift output name collision: model #{name_registry[fname] + 1} and "
                        f"model #{midx + 1} both map to '{fname}'."
                    )
                name_registry[fname] = midx

        def _process_one_model(task: Tuple[int, List[str], List[str]]) -> Dict[str, Any]:
            midx, u_group, v_group = task
            if not u_group or not v_group:
                return {
                    'model_index': midx,
                    'model_name': f'model_{midx + 1}',
                    'status': 'skipped_empty',
                }

            model_name, out_u_name, out_v_name, out_u_path, out_v_path = _resolve_output_names(midx, u_group, v_group)
            logger.info("  Model %d: %s", midx + 1, model_name)

            lock_ctx = _acquire_path_locks([out_u_path, out_v_path])
            lock_ctx.__enter__()
            try:
                if os.path.exists(out_u_path) and os.path.exists(out_v_path):
                    if overwrite:
                        logger.info("    Overwriting existing processed files for %s", model_name)
                        os.remove(out_u_path)
                        os.remove(out_v_path)
                    else:
                        return {
                            'model_index': midx,
                            'model_name': model_name,
                            'status': 'cached',
                            'out_u_name': out_u_name,
                            'out_v_name': out_v_name,
                        }

                seg_pairs = self._pair_model_segments(u_group, v_group)
                if not seg_pairs:
                    logger.warning("    No valid u/v segment pair for %s", model_name)
                    return {
                        'model_index': midx,
                        'model_name': model_name,
                        'status': 'skipped_no_pair',
                    }

                req_dir = model_direction[midx] if midx < len(model_direction) else 'auto'
                angle_spec = model_angle[midx] if midx < len(model_angle) else None
                infer_set = set()
                eff_set = set()
                resolve_set = set()
                grid_set = set()
                src_rot_set = set()
                override_count = 0

                with self._tempdir(prefix=f'sitool_sid_{midx}_') as tmp:
                    u_chunks: List[str] = []
                    v_chunks: List[str] = []

                    def _process_one_segment(seg_task: Tuple[int, str, str]) -> Dict[str, Any]:
                        sidx, u_in, v_in = seg_task
                        logger.info("    Segment %d: %s | %s", sidx + 1, os.path.basename(u_in), os.path.basename(v_in))
                        seg_dir = os.path.join(tmp, f'seg_{sidx:04d}')
                        os.makedirs(seg_dir, exist_ok=True)
                        worker_cdo = cdo if (safe_jobs == 1 and segment_workers == 1) else self._new_cdo_instance()

                        u_freq = os.path.join(seg_dir, 'u_freq.nc')
                        v_freq = os.path.join(seg_dir, 'v_freq.nc')
                        u_selraw = os.path.join(seg_dir, 'u_selraw.nc')
                        v_selraw = os.path.join(seg_dir, 'v_selraw.nc')
                        u_sel = os.path.join(seg_dir, 'u_sel.nc')
                        v_sel = os.path.join(seg_dir, 'v_sel.nc')

                        if frequency == 'monthly':
                            # Slice first to avoid running monmean on the full historical span.
                            self._safe_seldate(worker_cdo, start_date, end_date, os.path.abspath(u_in), u_selraw)
                            self._safe_seldate(worker_cdo, start_date, end_date, os.path.abspath(v_in), v_selraw)
                            self._safe_monmean(worker_cdo, u_selraw, u_freq)
                            self._safe_monmean(worker_cdo, v_selraw, v_freq)
                            u_current, v_current = u_freq, v_freq
                        else:
                            u_current, v_current = os.path.abspath(u_in), os.path.abspath(v_in)

                        self._safe_seldate(worker_cdo, start_date, end_date, u_current, u_sel)
                        self._safe_seldate(worker_cdo, start_date, end_date, v_current, v_sel)

                        with xr.open_dataset(u_sel) as ds_u, xr.open_dataset(v_sel) as ds_v:
                            if model_u_key not in ds_u.variables or model_v_key not in ds_v.variables:
                                raise ValueError(
                                    f"Missing SIdrift variables in segment files: {u_sel}, {v_sel}"
                                )

                            lon, lat = self._extract_lon_lat_arrays(ds_u)
                            u_attrs = dict(ds_u[model_u_key].attrs)
                            v_attrs = dict(ds_v[model_v_key].attrs)
                            inferred, infer_reason = vr.infer_source_frame_from_attrs(u_attrs, v_attrs)
                            effective, resolve_reason = vr.resolve_source_frame(
                                req_dir, inferred, prefer_inferred=prefer_metadata
                            )
                            gkind = vr.grid_kind_from_lonlat(lon, lat)
                            override_flag = (
                                inferred is not None
                                and inferred != str(req_dir).lower()
                                and resolve_reason == 'metadata_override'
                            )

                            u_en = os.path.join(seg_dir, 'u_en.nc')
                            v_en = os.path.join(seg_dir, 'v_en.nc')
                            theta, theta_state = self._load_angle_field(angle_spec, tuple(lon.shape))
                            use_angle = theta is not None

                            if effective == 'lonlat' and not use_angle:
                                src_rot = 'input_eastnorth'
                                worker_cdo.setmissval(-9999, input=u_sel, output=u_en)
                                worker_cdo.setmissval(-9999, input=v_sel, output=v_en)
                            elif use_angle:
                                src_rot = f'angle:{theta_state}'
                                u_raw = self._mask_missing(np.array(ds_u[model_u_key]), u_attrs)
                                v_raw = self._mask_missing(np.array(ds_v[model_v_key]), v_attrs)
                                u_rot, v_rot = vr.rotate_by_angle(u_raw, v_raw, theta, theta_unit='radian')

                                ds_u_out = ds_u.copy(deep=False)
                                ds_v_out = ds_v.copy(deep=False)
                                ds_u_out[model_u_key] = (ds_u[model_u_key].dims, u_rot.astype(np.float32))
                                ds_v_out[model_v_key] = (ds_v[model_v_key].dims, v_rot.astype(np.float32))
                                ds_u_out[model_u_key].attrs = dict(ds_u[model_u_key].attrs)
                                ds_v_out[model_v_key].attrs = dict(ds_v[model_v_key].attrs)
                                utils.write_netcdf_compressed(ds_u_out, u_en)
                                utils.write_netcdf_compressed(ds_v_out, v_en)
                            elif effective == 'xy' and gkind == 'structured':
                                src_rot = 'structured_basis_lonlat_gradient'
                                basis, _qc = vr.estimate_structured_basis_from_lonlat(lon, lat)
                                u_raw = self._mask_missing(np.array(ds_u[model_u_key]), u_attrs)
                                v_raw = self._mask_missing(np.array(ds_v[model_v_key]), v_attrs)
                                u_east, v_north = vr.rotate_native_xy_to_eastnorth(u_raw, v_raw, basis)

                                ds_u_out = ds_u.copy(deep=False)
                                ds_v_out = ds_v.copy(deep=False)
                                ds_u_out[model_u_key] = (ds_u[model_u_key].dims, u_east.astype(np.float32))
                                ds_v_out[model_v_key] = (ds_v[model_v_key].dims, v_north.astype(np.float32))
                                ds_u_out[model_u_key].attrs = dict(ds_u[model_u_key].attrs)
                                ds_v_out[model_v_key].attrs = dict(ds_v[model_v_key].attrs)
                                utils.write_netcdf_compressed(ds_u_out, u_en)
                                utils.write_netcdf_compressed(ds_v_out, v_en)
                            else:
                                src_rot = 'assume_eastnorth_unstructured'
                                worker_cdo.setmissval(-9999, input=u_sel, output=u_en)
                                worker_cdo.setmissval(-9999, input=v_sel, output=v_en)

                        u_en_std = os.path.join(seg_dir, 'u_en_std.nc')
                        v_en_std = os.path.join(seg_dir, 'v_en_std.nc')
                        worker_cdo.setmissval(-9999, input=u_en, output=u_en_std)
                        worker_cdo.setmissval(-9999, input=v_en, output=v_en_std)
                        u_en, v_en = u_en_std, v_en_std

                        u_interp = os.path.join(seg_dir, 'u_interp.nc')
                        v_interp = os.path.join(seg_dir, 'v_interp.nc')
                        utils.stable_interpolation(grid_txt, u_en, u_interp, worker_cdo)
                        utils.stable_interpolation(grid_txt, v_en, v_interp, worker_cdo)

                        u_xy = os.path.join(seg_dir, 'u_xy.nc')
                        v_xy = os.path.join(seg_dir, 'v_xy.nc')
                        with xr.open_dataset(u_interp) as ds_ui, xr.open_dataset(v_interp) as ds_vi:
                            u_i = self._mask_missing(np.array(ds_ui[model_u_key]), dict(ds_ui[model_u_key].attrs))
                            v_i = self._mask_missing(np.array(ds_vi[model_v_key]), dict(ds_vi[model_v_key].attrs))
                            lon_i, _lat_i = self._extract_lon_lat_arrays(ds_ui)
                            u_rot, v_rot = utils.rotate_vector_formula(
                                u=u_i, v=v_i, hemisphere=self.hemisphere, lons=lon_i
                            )

                            ds_u_xy = ds_ui.copy(deep=False)
                            ds_v_xy = ds_vi.copy(deep=False)
                            ds_u_xy[model_u_key] = (ds_ui[model_u_key].dims, u_rot.astype(np.float32))
                            ds_v_xy[model_v_key] = (ds_vi[model_v_key].dims, v_rot.astype(np.float32))
                            ds_u_xy[model_u_key].attrs = dict(ds_ui[model_u_key].attrs)
                            ds_v_xy[model_v_key].attrs = dict(ds_vi[model_v_key].attrs)
                            ds_u_xy.attrs['sitool_vector_frame'] = 'target_xy'
                            ds_v_xy.attrs['sitool_vector_frame'] = 'target_xy'
                            ds_u_xy.attrs['sitool_source_rotation'] = src_rot
                            ds_v_xy.attrs['sitool_source_rotation'] = src_rot
                            ds_u_xy.attrs['sitool_target_rotation'] = 'eastnorth_to_stere_xy_formula'
                            ds_v_xy.attrs['sitool_target_rotation'] = 'eastnorth_to_stere_xy_formula'
                            utils.write_netcdf_compressed(ds_u_xy, u_xy)
                            utils.write_netcdf_compressed(ds_v_xy, v_xy)

                        return {
                            'segment_index': sidx,
                            'u_xy': u_xy,
                            'v_xy': v_xy,
                            'inferred_signal': f"{inferred or 'none'}:{infer_reason}",
                            'effective_direction': effective,
                            'resolve_mode': resolve_reason,
                            'grid_kind': gkind,
                            'source_rotation': src_rot,
                            'metadata_override': bool(override_flag),
                        }

                    seg_tasks = [(sidx, u_in, v_in) for sidx, (u_in, v_in) in enumerate(seg_pairs)]
                    seg_results: Dict[int, Dict[str, Any]] = {}
                    if segment_workers > 1 and len(seg_tasks) > 1:
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        inner_workers = min(segment_workers, len(seg_tasks))
                        with ThreadPoolExecutor(max_workers=inner_workers, thread_name_prefix=f'prep-sidrift-m{midx}') as seg_pool:
                            fut_map = {
                                seg_pool.submit(rte.run_tracked_task, _process_one_segment, seg_task): seg_task[0]
                                for seg_task in seg_tasks
                            }
                            for fut in as_completed(fut_map):
                                seg_idx = fut_map[fut]
                                seg_results[seg_idx] = fut.result()
                    else:
                        for seg_task in seg_tasks:
                            seg_idx, _, _ = seg_task
                            seg_results[seg_idx] = _process_one_segment(seg_task)

                    for seg_idx in sorted(seg_results.keys()):
                        seg_payload = seg_results[seg_idx]
                        u_chunks.append(seg_payload['u_xy'])
                        v_chunks.append(seg_payload['v_xy'])
                        infer_set.add(seg_payload['inferred_signal'])
                        eff_set.add(seg_payload['effective_direction'])
                        resolve_set.add(seg_payload['resolve_mode'])
                        grid_set.add(seg_payload['grid_kind'])
                        src_rot_set.add(seg_payload['source_rotation'])
                        if seg_payload['metadata_override']:
                            override_count += 1

                    if not u_chunks or not v_chunks:
                        logger.warning("    No valid processed chunks for %s", model_name)
                        return {
                            'model_index': midx,
                            'model_name': model_name,
                            'status': 'failed_no_chunks',
                        }

                    merge_cdo = cdo if (safe_jobs == 1 and segment_workers == 1) else self._new_cdo_instance()
                    merged_u = os.path.join(tmp, 'u_merged.nc')
                    merged_v = os.path.join(tmp, 'v_merged.nc')
                    merge_cdo.mergetime(input=' '.join(os.path.abspath(p) for p in u_chunks), output=merged_u)
                    merge_cdo.mergetime(input=' '.join(os.path.abspath(p) for p in v_chunks), output=merged_v)
                    self._safe_seldate(merge_cdo, start_date, end_date, merged_u, os.path.abspath(out_u_path))
                    self._safe_seldate(merge_cdo, start_date, end_date, merged_v, os.path.abspath(out_v_path))

                if not (os.path.exists(out_u_path) and os.path.exists(out_v_path)):
                    logger.error("Failed to create SIdrift outputs for %s", model_name)
                    return {
                        'model_index': midx,
                        'model_name': model_name,
                        'status': 'failed_output_missing',
                    }

                logger.info("Created: %s / %s", out_u_name, out_v_name)
                return {
                    'model_index': midx,
                    'model_name': model_name,
                    'status': 'created',
                    'out_u_name': out_u_name,
                    'out_v_name': out_v_name,
                    'requested_direction': req_dir,
                    'inferred_signals': sorted(infer_set),
                    'effective_directions': sorted(eff_set),
                    'resolve_modes': sorted(resolve_set),
                    'grid_kinds': sorted(grid_set),
                    'source_rotation_methods': sorted(src_rot_set),
                    'metadata_override_count': int(override_count),
                }
            except Exception as exc:
                logger.error("Failed to preprocess SIdrift model %s (%s).", model_name, exc, exc_info=True)
                return {
                    'model_index': midx,
                    'model_name': model_name,
                    'status': 'failed_exception',
                    'error': str(exc),
                }
            finally:
                lock_ctx.__exit__(None, None, None)

        tasks = [(midx, u_group, v_group) for midx, (u_group, v_group) in enumerate(zip(u_file_groups, v_file_groups))]
        if model_workers > 1 and len(tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            result_map: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=model_workers, thread_name_prefix='prep-sidrift') as pool:
                if segment_workers > 1:
                    fut_map = {pool.submit(_process_one_model, task): task[0] for task in tasks}
                else:
                    fut_map = {
                        pool.submit(rte.run_tracked_task, _process_one_model, task): task[0]
                        for task in tasks
                    }
                for fut in as_completed(fut_map):
                    result_map[fut_map[fut]] = fut.result()
            ordered_rows = [result_map[k] for k in sorted(result_map.keys())]
        else:
            ordered_rows = [_process_one_model(task) for task in tasks]

        for row in ordered_rows:
            status = str(row.get('status', 'unknown'))
            if 'out_u_name' in row and 'out_v_name' in row and status in {'cached', 'created'}:
                out_u_files.append(str(row['out_u_name']))
                out_v_files.append(str(row['out_v_name']))
            audit_rows.append(row)

        audit_file = os.path.join(output_dir, f'sidrift_rotation_audit_{self.hemisphere}.json')
        try:
            payload = {
                'case': self.eval_ex,
                'module': self.module_name,
                'hemisphere': self.hemisphere,
                'created_at': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'models': audit_rows,
            }
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            logger.info("Saved SIdrift rotation audit: %s", audit_file)
        except Exception as exc:
            logger.warning("Failed to write SIdrift audit file (%s).", exc)

        return out_u_files, out_v_files, audit_rows

__all__ = [
    "SidriftPrepMixin",
]
