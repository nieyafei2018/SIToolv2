# -*- coding: utf-8 -*-
"""Cache, staging, and metric persistence helpers for pipeline runtime."""

from scripts import SITOOL_NC_COMPRESS_LEVEL, SITOOL_NC_SHUFFLE
from scripts.pipeline import app as _app

# Reuse runtime namespace (imports/constants/helpers) initialized in app.py.
globals().update({k: v for k, v in _app.__dict__.items() if k not in globals()})


def _parallel_map_ordered(
    items: List[Any],
    worker_fn,
    max_workers: int,
    task_label: str,
) -> List[Any]:
    """Run one worker function on multiple items in parallel and keep original order.

    Args:
        items: Ordered list of input items.
        worker_fn: Callable that accepts one item and returns one result.
        max_workers: Number of thread workers.
        task_label: Human-readable task label for diagnostics.
    """
    if not items:
        return []

    requested_workers = max(1, int(max_workers))
    n_workers = requested_workers

    # Safety gate:
    # NetCDF/HDF5 access through xarray/netCDF4 is not thread-safe under
    # threads in this runtime. For metric-heavy stages, prefer process-based
    # parallelism to keep safety while improving utilization.
    metric_thread_labels = ('model-metrics', 'model-budget', 'model-transition')
    metric_stage = bool(task_label and any(token in str(task_label) for token in metric_thread_labels))

    def _run_metric_stage_with_processes(process_workers: int) -> Optional[List[Any]]:
        import queue as queue_mod
        from collections import deque as _deque

        try:
            ctx = mp.get_context('fork')
        except Exception as exc:
            logger.warning(
                "Process backend unavailable for metric stage %s (%s); falling back to serial.",
                task_label, exc,
            )
            return None

        safe_workers = max(1, min(int(process_workers), len(items)))
        if safe_workers <= 1:
            return None

        logger.info(
            "Using process backend for metric stage: task=%s (workers=%d).",
            task_label, safe_workers,
        )

        result_q = ctx.Queue()
        marker_missing = object()
        results: List[Any] = [marker_missing] * len(items)
        failures: Dict[int, str] = {}
        pending = _deque(list(enumerate(items)))
        running: Dict[int, Any] = {}
        reported: set = set()

        def _proc_entry(item_idx: int, payload: Any, queue_obj) -> None:
            try:
                out_val = worker_fn(payload)
                msg = {'idx': item_idx, 'ok': True, 'result': out_val, 'error': None}
            except Exception as exc:
                msg = {
                    'idx': item_idx,
                    'ok': False,
                    'result': None,
                    'error': f"{type(exc).__name__}: {exc}",
                }
            try:
                queue_obj.put(msg)
            except Exception as q_exc:
                try:
                    queue_obj.put({
                        'idx': item_idx,
                        'ok': False,
                        'result': None,
                        'error': f'queue_put_failed: {q_exc}',
                    })
                except Exception:
                    pass

        try:
            while pending or running:
                while pending and len(running) < safe_workers:
                    item_idx, payload = pending.popleft()
                    proc = ctx.Process(
                        target=_proc_entry,
                        args=(int(item_idx), payload, result_q),
                        name=f'sitool-stage-{int(item_idx):04d}',
                    )
                    proc.daemon = False
                    proc.start()
                    running[int(item_idx)] = proc

                try:
                    msg = result_q.get(timeout=0.5)
                    item_idx = int(msg.get('idx'))
                    reported.add(item_idx)
                    if bool(msg.get('ok', True)):
                        results[item_idx] = msg.get('result')
                    else:
                        failures[item_idx] = str(msg.get('error') or 'unknown error')
                except queue_mod.Empty:
                    pass

                finished: List[int] = []
                for item_idx, proc in running.items():
                    if not proc.is_alive():
                        proc.join(timeout=0.1)
                        finished.append(item_idx)
                for item_idx in finished:
                    proc = running.pop(item_idx)
                    if item_idx not in reported and item_idx not in failures:
                        failures[item_idx] = (
                            f'process exited without result (exit_code={proc.exitcode})'
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

        failed_indices = [idx for idx, val in enumerate(results) if val is marker_missing]
        if failed_indices:
            logger.warning(
                "Metric stage process backend had %d failed items in task=%s; "
                "retrying failed items serially.",
                len(failed_indices), task_label,
            )
            for idx in failed_indices:
                try:
                    results[idx] = worker_fn(items[idx])
                    failures.pop(idx, None)
                except Exception as exc:
                    hint = failures.get(idx, '')
                    raise RuntimeError(
                        f"Parallel task failed in {task_label} (item #{idx}): {exc}"
                        + (f" | process_error={hint}" if hint else '')
                    ) from exc
        return results

    if metric_stage and requested_workers > 1:
        backend_raw = str(os.environ.get('SITOOL_METRIC_STAGE_BACKEND', 'process')).strip().lower()
        use_process_backend = backend_raw in {'process', 'proc', 'multiprocess', 'mp', 'auto'}
        if use_process_backend:
            try:
                proc_limit_raw = os.environ.get('SITOOL_METRIC_STAGE_PROCESS_MAX_WORKERS', '')
                if str(proc_limit_raw).strip():
                    proc_limit = max(1, int(str(proc_limit_raw).strip()))
                else:
                    proc_limit = requested_workers
            except Exception:
                proc_limit = requested_workers
            process_results = _run_metric_stage_with_processes(
                process_workers=min(requested_workers, proc_limit),
            )
            if process_results is not None:
                return process_results
            n_workers = 1
        else:
            n_workers = 1
        if n_workers == 1:
            logger.info(
                "Serializing threaded metric stage for safety: task=%s (requested_workers=%d -> 1)",
                task_label, requested_workers,
            )

    if n_workers == 1 or len(items) == 1:
        return [worker_fn(item) for item in items]

    results: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix='sitool') as pool:
        future_to_idx = {
            pool.submit(rte.run_tracked_task, worker_fn, item): idx
            for idx, item in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                raise RuntimeError(f"Parallel task failed in {task_label} (item #{idx}): {exc}") from exc
    return results


def _case_dir_from_output_dir(case_name: str, output_dir: str) -> Path:
    """Resolve the case directory from an output path."""
    out = Path(output_dir)
    return out.parent.parent if out.parent.name == 'Output' else _resolve_case_run_dir(case_name, log_warning=False)


def _get_stage_dir(case_name: str, output_dir: str, hemisphere: str, module: str) -> Path:
    """Return the per-run staging directory for parallel worker payloads."""
    case_dir = _case_dir_from_output_dir(case_name, output_dir)
    run_id = os.environ.get('SITOOL_RUN_ID', 'run')
    stage_dir = case_dir / 'metrics' / '_staging' / run_id / hemisphere / module
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def _save_pickle_atomic(path: Path, payload: Any) -> None:
    """Persist one Python payload atomically to avoid torn writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f'.{path.stem}_tmp_',
        suffix='.pkl',
        dir=str(path.parent),
        delete=False,
    ) as tf:
        tmp_path = Path(tf.name)
        pickle.dump(payload, tf, protocol=pickle.HIGHEST_PROTOCOL)
    _safe_replace(tmp_path, path)


def _load_pickle(path: Path) -> Any:
    """Load one staged Python payload from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def _cleanup_stage_dir(case_dir: Path, run_id: Optional[str] = None) -> None:
    """Remove staging artifacts for one run to reclaim disk space."""
    metrics_dir = case_dir / 'metrics' / '_staging'
    target = metrics_dir / run_id if run_id else metrics_dir
    if not target.exists():
        return
    try:
        shutil.rmtree(target)
        logger.info("Removed staging directory: %s", target)
    except Exception as exc:
        logger.warning("Failed to remove staging directory %s (%s).", target, exc)


def _get_metrics_cache_file(case_name: str, output_dir: str,
                            hemisphere: str, module: str) -> Path:
    """Return module-specific cache path: cases/<case>/metrics/<hms>_<module>_metrics.nc."""
    out = Path(output_dir)
    case_dir = out.parent.parent if out.parent.name == 'Output' else _resolve_case_run_dir(
        case_name, log_warning=False
    )
    metrics_dir = case_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    hms = str(hemisphere or '').lower()
    mod = str(module or '').strip()
    return metrics_dir / f'{hms}_{mod}_metrics.nc'


def _open_dataset_with_engines(path: Path, group: Optional[str] = None) -> xr.Dataset:
    """Open one NetCDF group with engine fallback."""
    errors: List[str] = []
    for attempt in range(_NC_IO_RETRIES):
        attempt_errors: List[str] = []
        retryable_hit = False
        for engine in _CACHE_ENGINES:
            try:
                kwargs: Dict[str, Any] = {'decode_times': False}
                if group is not None:
                    kwargs['group'] = group
                if engine is not None:
                    kwargs['engine'] = engine
                return xr.open_dataset(path, **kwargs)
            except Exception as exc:
                attempt_errors.append(f'{engine or "default"}: {exc}')
                retryable_hit = retryable_hit or _is_retryable_nc_error(exc)
        errors.extend(attempt_errors)
        if (not retryable_hit) or attempt >= _NC_IO_RETRIES - 1:
            break
        time.sleep(_NC_IO_BASE_DELAY_SEC * (attempt + 1))
    raise OSError(f'Unable to open cache file/group {path}::{group}. Tried: {" | ".join(errors)}')


def _write_dataset_with_engines(path: Path, ds: xr.Dataset, mode: str = 'a',
                                group: Optional[str] = None) -> None:
    """Write one NetCDF group with engine fallback."""
    ds_out = ds.copy(deep=False)
    encoding: Dict[str, Dict[str, Any]] = {}
    for var_name in ds_out.data_vars:
        da = ds_out[var_name]
        kind = da.dtype.kind
        # Remove _FillValue from attrs to avoid xarray conflict with encoding.
        fv = da.attrs.pop('_FillValue', None)
        if kind in ('U', 'S', 'O'):
            encoding[var_name] = {'_FillValue': None}
        else:
            var_enc: Dict[str, Any] = {
                'zlib': True,
                'complevel': int(SITOOL_NC_COMPRESS_LEVEL),
                'shuffle': bool(SITOOL_NC_SHUFFLE),
            }
            if isinstance(fv, (str, bytes)):
                pass
            elif fv is not None:
                var_enc['_FillValue'] = fv
            encoding[var_name] = var_enc

    modes = [mode]
    if mode == 'a' and not path.exists():
        modes = ['w', 'a']

    errors: List[str] = []
    for write_mode in modes:
        for attempt in range(_NC_IO_RETRIES):
            attempt_errors: List[str] = []
            retryable_hit = False
            for engine in _CACHE_ENGINES:
                try:
                    kwargs: Dict[str, Any] = {'mode': write_mode}
                    if group is not None:
                        kwargs['group'] = group
                    if engine is not None:
                        kwargs['engine'] = engine
                    if encoding:
                        kwargs['encoding'] = encoding
                    ds_out.to_netcdf(path, **kwargs)
                    return
                except Exception as exc:
                    attempt_errors.append(f'mode={write_mode}, {engine or "default"}: {exc}')
                    retryable_hit = retryable_hit or _is_retryable_nc_error(exc)
            errors.extend(attempt_errors)
            if (not retryable_hit) or attempt >= _NC_IO_RETRIES - 1:
                break
            time.sleep(_NC_IO_BASE_DELAY_SEC * (attempt + 1))
    raise OSError(f'Unable to write cache file/group {path}::{group}. Tried: {" | ".join(errors)}')


def _read_cache_root_attrs(cache_file: Path) -> Dict[str, Any]:
    """Read root-level cache attributes."""
    if not cache_file.exists():
        return {}
    try:
        ds = _open_dataset_with_engines(cache_file)
    except Exception:
        return {}
    try:
        return dict(ds.attrs)
    finally:
        ds.close()


def _update_cache_root_attrs(cache_file: Path, attrs: Dict[str, Any]) -> None:
    """Merge and persist root-level cache attributes."""
    merged = _read_cache_root_attrs(cache_file) if cache_file.exists() else {}
    merged.update(attrs)
    root_ds = xr.Dataset()
    root_ds.attrs.update(merged)
    mode = 'a' if cache_file.exists() else 'w'
    _write_dataset_with_engines(cache_file, root_ds, mode=mode, group=None)


def _sanitize_group_name(name: str) -> str:
    """Sanitize one group-name segment for NetCDF hierarchy."""
    text = re.sub(r'\s+', '_', str(name or '').strip())
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_.')
    return text or 'unnamed'


def _derive_label_from_ref_name(ref_entry: str, hemisphere: str) -> str:
    """Derive a short dataset label from one recipe reference filename."""
    stem = Path(str(ref_entry)).stem
    stem = re.sub(rf'_{hemisphere}(?:_|$)', '_', stem, flags=re.IGNORECASE)
    stem = re.sub(r'_[0-9]{6,8}(?:-[0-9]{6,8})?(?=_)', '', stem)
    stem = re.sub(r'_[0-9]{6,8}(?:-[0-9]{6,8})?$', '', stem)
    for token in (
        '_siconc', '_sidrift', '_sithick', '_sisnthick', '_snow_depth', '_sndepth',
        '_sndepth', '_snod', '_siu', '_siv', '_sivol',
        '_daily', '_day', '_monthly', '_monmean', '_mon',
    ):
        hit = re.search(re.escape(token), stem, flags=re.IGNORECASE)
        if hit:
            stem = stem[:hit.start()]
            break
    stem = stem.strip('_')
    return stem or Path(str(ref_entry)).stem


def _obs_label_is_placeholder(value: Any, obs_key: str) -> bool:
    """Return True when one obs label is blank or generic placeholder."""
    txt = str(value if value is not None else '').strip().lower()
    if not txt:
        return True
    compact = re.sub(r'\s+', '', txt)
    key = str(obs_key or '').strip().lower()
    if not key:
        return compact in {'obs1', 'obs2', 'observation1', 'observation2'}
    placeholders = {
        key,
        key.replace('obs', 'observation'),
        'obs1',
        'obs2',
        'observation1',
        'observation2',
    }
    return compact in placeholders


def _get_obs_name_from_recipe(module_vars: Dict[str, Any], hemisphere: str, obs_key: str) -> Optional[str]:
    """Read one configured obs name from module-scoped obs_names in recipe."""
    raw = module_vars.get('obs_names')
    if not isinstance(raw, dict):
        return None
    hms = str(hemisphere or '').strip().lower()
    hms_cfg = raw.get(hms) if isinstance(raw.get(hms), dict) else {}
    if isinstance(hms_cfg, dict) and obs_key in hms_cfg:
        value = hms_cfg.get(obs_key)
    else:
        value = raw.get(obs_key)
    if value is None:
        return None
    txt = str(value).strip()
    return txt if txt else None


def _get_reference_labels(module_vars: Dict[str, Any], hemisphere: str,
                          suffix: str = '') -> List[str]:
    """Get observation labels for one module/hemisphere.

    Priority:
      1) module ``obs_names`` (supports per-hemisphere ``nh/sh`` blocks),
      2) legacy ``ref_labels*`` fields,
      3) derived labels from reference filenames.

    Placeholder values such as ``Obs1`` / ``Obs2`` are treated as missing and
    replaced by inferred labels.
    """
    ref_key = f'ref_{hemisphere}{suffix}'
    refs = module_vars.get(ref_key) or []
    if not isinstance(refs, list):
        refs = [refs]

    label_keys = [
        f'ref_labels_{hemisphere}{suffix}',
        f'ref_labels{suffix}',
        f'ref_labels_{hemisphere}',
        'ref_labels',
    ]
    labels: List[str] = []
    for key in label_keys:
        raw = module_vars.get(key)
        if isinstance(raw, list) and raw:
            labels = [str(v).strip() for v in raw if str(v).strip()]
            break

    if not labels:
        single_label = module_vars.get('ref_label')
        if isinstance(single_label, str) and single_label.strip():
            labels.append(single_label.strip())

    derived = [_derive_label_from_ref_name(r, hemisphere) for r in refs]
    n_out = max(len(derived), len(labels), len(refs))
    if n_out <= 0:
        n_out = 2

    out: List[str] = []
    for idx in range(n_out):
        obs_key = 'obs1' if idx == 0 else ('obs2' if idx == 1 else f'obs{idx + 1}')
        configured = _get_obs_name_from_recipe(module_vars, hemisphere, obs_key)
        legacy = labels[idx] if idx < len(labels) else None
        inferred = derived[idx] if idx < len(derived) else None

        chosen: Optional[str] = None
        if configured and (not _obs_label_is_placeholder(configured, obs_key)):
            chosen = configured
        elif legacy and (not _obs_label_is_placeholder(legacy, obs_key)):
            chosen = str(legacy).strip()
        elif inferred and str(inferred).strip():
            chosen = str(inferred).strip()
        else:
            chosen = 'Obs1' if idx == 0 else ('Obs2' if idx == 1 else f'Obs{idx + 1}')
        out.append(chosen)
    return out


def _derive_label_from_model_pattern(pattern: str) -> str:
    """Derive one model label from one recipe model_file pattern."""
    text = str(pattern or '').strip()
    if not text:
        return ''
    cleaned = text.replace('\\', '/').strip()
    first_part = cleaned.split('/', 1)[0].strip()
    if first_part and not any(ch in first_part for ch in '*?[]'):
        return first_part

    stem = Path(cleaned).stem
    stem = re.sub(r'[\*\?\[\]]+', '', stem)
    for token in ('_SImon', '_day', '_mon', '_monthly', '_daily', '_siconc', '_sivol', '_siu', '_siv', '_sisnthick'):
        if token in stem:
            stem = stem.split(token, 1)[0]
            break
    return stem.strip('_') or cleaned


def _get_recipe_model_labels(module: str, module_vars: Dict[str, Any], n_models: int) -> List[str]:
    """Get model labels from recipe labels or recipe model_file patterns."""
    if n_models <= 0:
        return []

    explicit = module_vars.get('model_labels') or []
    if isinstance(explicit, list) and explicit:
        labels = [str(v).strip() for v in explicit if str(v).strip()]
    else:
        labels = []

    pattern_keys = {
        'SIdrift': ['model_file_u', 'model_file_v'],
        'SICB': ['model_file_sic', 'model_file_u', 'model_file_v'],
        'SIMbudget': ['model_file_sidmassdyn', 'model_file_sidmasssi'],
        'SNMbudget': ['model_file_sndmassdyn', 'model_file_sndmasssi'],
    }.get(module, ['model_file'])

    patterns: List[str] = []
    for key in pattern_keys:
        vals = module_vars.get(key) or []
        if isinstance(vals, list) and vals:
            patterns = [str(v) for v in vals]
            break

    derived = [_derive_label_from_model_pattern(p) for p in patterns]
    if labels:
        out = labels[:n_models]
        if len(out) < n_models:
            out.extend(derived[len(out):n_models])
    else:
        out = derived[:n_models]

    used: set = set()
    final: List[str] = []
    for ii in range(n_models):
        base = out[ii] if ii < len(out) and str(out[ii]).strip() else (patterns[ii] if ii < len(patterns) else f'{module}_dataset_{ii + 1}')
        candidate = _sanitize_group_name(base)
        if candidate in used:
            jj = 2
            while f'{candidate}_{jj}' in used:
                jj += 1
            candidate = f'{candidate}_{jj}'
        used.add(candidate)
        final.append(candidate)
    return final


def _unique_entity_name(preferred: str, fallback: str, used: set) -> str:
    """Return a unique level-3 group name for one cached record."""
    base = _sanitize_group_name(preferred if preferred else fallback)
    candidate = base
    idx = 2
    while candidate in used:
        candidate = f'{base}_{idx}'
        idx += 1
    used.add(candidate)
    return candidate


def _infer_units(var_name: str) -> str:
    """Best-effort unit inference for cached metric variables."""
    key = var_name.lower()
    if 'corr' in key or key.endswith('_p') or 'pvalue' in key:
        return '1'
    if 'siconc' in key:
        return '%/decade' if 'trend' in key or '_tr' in key else '%'
    if key.startswith(('sie', 'sia', 'miz', 'pia')) or '_sie_' in key or '_sia_' in key:
        return '10^6 km^2/decade' if 'trend' in key else '10^6 km^2'
    if key.startswith('iiee') or key.startswith('o_u'):
        return '10^6 km^2'
    if 'thick' in key:
        return 'm/decade' if 'trend' in key or '_tr' in key else 'm'
    if key.startswith('vol') or '_vol_' in key:
        return '10^3 km^3/decade' if 'trend' in key else '10^3 km^3'
    if 'mke' in key:
        return 'm^2/s^2/decade' if 'trend' in key else 'm^2/s^2'
    if key.startswith('speed'):
        return 'm/s'
    if key.startswith(('advance', 'retreat')):
        if 'sig_fraction' in key:
            return '%'
        if 'valid_year' in key:
            return 'years'
        if 'trend' in key:
            return 'days/year'
        if 'rmse' in key or 'bias' in key or 'clim' in key or 'std' in key:
            return 'days'
    if key.startswith(('ice_season', 'open_water')):
        if 'trend' in key:
            return 'days/year'
        if 'rmse' in key or 'bias' in key or 'clim' in key or 'std' in key:
            return 'days'
        if 'corr' in key or 'r2' in key or 'pvalue' in key:
            return '1'
    if 'sigmask' in key:
        return '1'
    if key == 'year':
        return 'year'
    if key.startswith('aipe') or key.endswith('_pct') or key.endswith('_p'):
        return '%'
    if key.startswith(('dadt', 'adv', 'div', 'res')):
        return '10^6 km^2/month'
    return 'unknown'


def _build_units_map(metric_dict: Dict[str, Any]) -> Dict[str, str]:
    """Create a variable->unit map for a metric dictionary."""
    return {k: _infer_units(k) for k in metric_dict.keys()}


def _load_grid_coords(grid_file: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    """Load optional 2-D latitude/longitude coordinates from one grid file."""
    if not grid_file:
        return None
    try:
        with xr.open_dataset(grid_file, decode_times=False) as ds:
            lon = ds['lon'] if 'lon' in ds.variables else (ds['longitude'] if 'longitude' in ds.variables else None)
            lat = ds['lat'] if 'lat' in ds.variables else (ds['latitude'] if 'latitude' in ds.variables else None)
            if lon is None or lat is None:
                return None
            lon_vals = np.asarray(lon.values)
            lat_vals = np.asarray(lat.values)
            if lon_vals.ndim != 2 or lat_vals.ndim != 2:
                return None
            if lon_vals.shape != lat_vals.shape:
                return None
            return {'lon': lon_vals, 'lat': lat_vals}
    except Exception:
        return None


def _write_metric_record(cache_file: Path, group_path: str, metric_dict: Dict[str, Any],
                         grid_coords: Optional[Dict[str, np.ndarray]] = None,
                         start_year: Optional[int] = None,
                         end_year: Optional[int] = None) -> None:
    """Write one metric dictionary into one NetCDF group."""
    if not metric_dict:
        return
    units_map = _build_units_map(metric_dict)
    SIM.SeaIceMetricsBase.save_to_nc(
        output_file=str(cache_file),
        metric_dict=metric_dict,
        group=group_path,
        mode='a',
        engines=_CACHE_ENGINES,
        units_map=units_map,
        grid_coords=grid_coords,
        start_year=start_year,
        end_year=end_year,
    )
    # Quick post-write check: ensure group can be reopened and contains variables.
    check_ds = _open_dataset_with_engines(cache_file, group=group_path)
    try:
        if not check_ds.data_vars:
            raise OSError(f'NetCDF group {group_path} is empty after write.')
    finally:
        check_ds.close()


def _read_metric_record(cache_file: Path, group_path: str) -> Dict[str, Any]:
    """Read one metric dictionary from one level-3 group."""
    ds = _open_dataset_with_engines(cache_file, group=group_path)
    try:
        return SIM.SeaIceMetricsBase.from_xarray(ds)
    finally:
        ds.close()


def _write_module_metadata(cache_file: Path, meta: Dict[str, Any]) -> None:
    """Store module payload metadata in a dedicated metadata group."""
    meta_ds = xr.Dataset({
        'payload_json': xr.DataArray(np.asarray(json.dumps(meta), dtype=str))
    })
    meta_ds['payload_json'].attrs.update({
        'units': 'json',
        'long_name': 'module_cache_metadata',
        'sitool_encoding': 'json',
    })
    _write_dataset_with_engines(cache_file, meta_ds, mode='a', group=_CACHE_META_GROUP)


def _read_module_metadata(cache_file: Path) -> Optional[Dict[str, Any]]:
    """Read module payload metadata from the metadata group."""
    try:
        ds = _open_dataset_with_engines(cache_file, group=_CACHE_META_GROUP)
    except Exception:
        return None
    try:
        if 'payload_json' not in ds.data_vars:
            return None
        text = str(ds['payload_json'].values.item())
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None
    finally:
        ds.close()


def _save_module_cache(cache_file: Path, case_name: str, module: str, hemisphere: str,
                       start_year: int, end_year: int,
                       payload_meta: Dict[str, Any],
                       records: Dict[str, Dict[str, Any]],
                       entity_groups: Optional[Dict[str, str]] = None,
                       grid_file: Optional[str] = None) -> None:
    """Persist one module payload with groups at root level: /<Dataset_Name> or /<Model_vs_Obs>."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{cache_file.stem}_tmp_',
        suffix='.nc',
        dir=str(cache_file.parent),
    )
    os.close(fd)
    tmp_cache = Path(tmp_name)
    _safe_unlink(tmp_cache)

    try:
        _update_cache_root_attrs(tmp_cache, {
            'cache_schema_version': CACHE_SCHEMA_VERSION,
            'case_name': case_name,
            'module': module,
            'hemisphere': hemisphere,
            'start_year': int(start_year),
            'end_year': int(end_year),
        })

        used_names: set = set()
        record_entity_map: Dict[str, str] = {}
        record_group_map: Dict[str, str] = {}
        grid_coords = _load_grid_coords(grid_file)
        for idx, (record_name, metric_dict) in enumerate(records.items(), start=1):
            if not metric_dict:
                continue
            preferred = (entity_groups or {}).get(record_name, record_name)
            entity_name = _unique_entity_name(preferred=preferred, fallback=f'Entity_{idx}', used=used_names)
            group_path = f'/{entity_name}'
            try:
                _write_metric_record(
                    tmp_cache, group_path, metric_dict,
                    grid_coords=grid_coords, start_year=start_year, end_year=end_year,
                )
                record_entity_map[record_name] = entity_name
                record_group_map[record_name] = group_path
            except Exception as exc:
                logger.warning(
                    "Skipping cache record %s for [%s/%s] (%s).",
                    record_name, hemisphere, module, exc,
                )

        meta = dict(payload_meta)
        meta.update({
            'module': module,
            'hemisphere': hemisphere,
            'record_names': list(record_entity_map.keys()),
            'record_entity_map': record_entity_map,
            'record_group_map': record_group_map,
        })
        _write_module_metadata(tmp_cache, meta=meta)
        _safe_replace(tmp_cache, cache_file)
    finally:
        if tmp_cache.exists():
            try:
                _safe_unlink(tmp_cache)
            except Exception:
                pass


def _load_module_cache(cache_file: Path, module: str, hemisphere: str) -> Optional[Dict[str, Any]]:
    """Load one module cache payload from module-specific cache groups."""
    if not cache_file.exists():
        return None

    attrs = _read_cache_root_attrs(cache_file)
    if attrs.get('cache_schema_version') != CACHE_SCHEMA_VERSION:
        logger.warning("Cache schema mismatch for %s. Expected %s, got %s. Recalculating.",
                       cache_file, CACHE_SCHEMA_VERSION, attrs.get('cache_schema_version'))
        return None
    if attrs.get('module') not in (None, '', module):
        logger.warning("Cache module mismatch for %s. Expected %s, got %s. Recalculating.",
                       cache_file, module, attrs.get('module'))
        return None
    if attrs.get('hemisphere') not in (None, '', hemisphere):
        logger.warning("Cache hemisphere mismatch for %s. Expected %s, got %s. Recalculating.",
                       cache_file, hemisphere, attrs.get('hemisphere'))
        return None

    meta = _read_module_metadata(cache_file)
    if meta is None:
        return None

    try:
        record_entity_map = meta.get('record_entity_map', {})
        record_group_map = meta.get('record_group_map', {})
        records: Dict[str, Dict[str, Any]] = {}
        for record_name in meta.get('record_names', []):
            entity_name = record_entity_map.get(record_name, record_name)
            group_path = record_group_map.get(record_name, f'/{entity_name}')
            records[record_name] = _read_metric_record(cache_file, group_path)
        meta['records'] = records
        return meta
    except Exception as exc:
        logger.warning("Failed to read cache for [%s/%s] (%s). Recalculating.", hemisphere, module, exc)
        return None


def _write_json_payload_group(cache_file: Path, group: str, payload: Dict[str, Any]) -> None:
    """Write one JSON payload into a dedicated NetCDF group."""
    ds = xr.Dataset({
        'payload_json': xr.DataArray(np.asarray(json.dumps(payload), dtype=str))
    })
    ds['payload_json'].attrs.update({
        'units': 'json',
        'long_name': 'cache_index_payload',
        'sitool_encoding': 'json',
    })
    _write_dataset_with_engines(cache_file, ds, mode='a', group=group)


def _build_unified_metrics_cache(case_name: str, case_dir: Path,
                                 eval_hms: List[str], modules_to_run: List[str]) -> Optional[Path]:
    """Build one unified cache file by serially merging all module cache files.

    This function is intentionally single-writer because NetCDF group writes are
    not safe to perform concurrently into the same destination file.
    """
    metrics_dir = case_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    unified_file = metrics_dir / f'{case_name}_all_metrics.nc'

    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{unified_file.stem}_tmp_',
        suffix='.nc',
        dir=str(metrics_dir),
    )
    os.close(fd)
    tmp_file = Path(tmp_name)
    _safe_unlink(tmp_file)

    merged_count = 0
    try:
        _update_cache_root_attrs(tmp_file, {
            'cache_schema_version': UNIFIED_CACHE_SCHEMA_VERSION,
            'case_name': case_name,
            'run_id': os.environ.get('SITOOL_RUN_ID', ''),
            'created_at': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        })

        for hms in eval_hms:
            for module in modules_to_run:
                module_cache = metrics_dir / f'{hms}_{module}_metrics.nc'
                if not module_cache.exists():
                    continue
                payload = _load_module_cache(module_cache, module, hms)
                if payload is None:
                    continue

                src_attrs = _read_cache_root_attrs(module_cache)
                start_year = src_attrs.get('start_year')
                end_year = src_attrs.get('end_year')
                try:
                    start_year = int(start_year) if start_year is not None else None
                except Exception:
                    start_year = None
                try:
                    end_year = int(end_year) if end_year is not None else None
                except Exception:
                    end_year = None

                records = payload.get('records', {})
                record_entity_map = payload.get('record_entity_map', {})
                record_names = payload.get('record_names', list(records.keys()))
                written_map: Dict[str, str] = {}

                for rec_name in record_names:
                    metric_dict = records.get(rec_name)
                    if not isinstance(metric_dict, dict) or not metric_dict:
                        continue
                    entity = _sanitize_group_name(record_entity_map.get(rec_name, rec_name))
                    group_path = f'/{hms}/{module}/{entity}'
                    _write_metric_record(
                        cache_file=tmp_file,
                        group_path=group_path,
                        metric_dict=metric_dict,
                        grid_coords=None,
                        start_year=start_year,
                        end_year=end_year,
                    )
                    written_map[rec_name] = group_path
                    merged_count += 1

                module_meta = {
                    'case_name': case_name,
                    'hemisphere': hms,
                    'module': module,
                    'record_names': [k for k in record_names if k in written_map],
                    'record_group_map': written_map,
                    'model_records': payload.get('model_records', []),
                    'diff_records': payload.get('diff_records', []),
                    'model_labels': payload.get('model_labels', []),
                    'payload_kind': payload.get('payload_kind', module),
                }
                _write_json_payload_group(
                    cache_file=tmp_file,
                    group=f'/_index/{hms}/{module}',
                    payload=module_meta,
                )

        if merged_count == 0:
            logger.warning("No module cache records available to build unified cache.")
            return None

        _safe_replace(tmp_file, unified_file)
        logger.info("Unified cache built: %s (records=%d)", unified_file, merged_count)
        return unified_file
    finally:
        if tmp_file.exists():
            try:
                _safe_unlink(tmp_file)
            except Exception:
                pass


__all__ = [
    "_parallel_map_ordered",
    "_case_dir_from_output_dir",
    "_get_stage_dir",
    "_save_pickle_atomic",
    "_load_pickle",
    "_cleanup_stage_dir",
    "_get_metrics_cache_file",
    "_open_dataset_with_engines",
    "_write_dataset_with_engines",
    "_read_cache_root_attrs",
    "_update_cache_root_attrs",
    "_sanitize_group_name",
    "_derive_label_from_ref_name",
    "_get_reference_labels",
    "_derive_label_from_model_pattern",
    "_get_recipe_model_labels",
    "_unique_entity_name",
    "_infer_units",
    "_build_units_map",
    "_load_grid_coords",
    "_write_metric_record",
    "_read_metric_record",
    "_write_module_metadata",
    "_read_module_metadata",
    "_save_module_cache",
    "_load_module_cache",
    "_write_json_payload_group",
    "_build_unified_metrics_cache",
]
