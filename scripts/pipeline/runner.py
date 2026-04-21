# -*- coding: utf-8 -*-
"""CLI runner for SIToolv2 pipeline orchestration."""

from scripts.pipeline import app as _app

# Reuse runtime namespace (imports/constants/helpers) initialized in app.py.
globals().update({k: v for k, v in _app.__dict__.items() if k not in globals()})


def _configure_case_tmpdir(case_dir: Path) -> Path:
    """Bind runtime temporary directories to case-local `_tmp` folder."""
    tmp_dir = case_dir / '_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = str(tmp_dir.resolve())

    os.environ['SITOOL_TMPDIR'] = tmp_path
    os.environ['SITOOL_CASE_TMPDIR'] = tmp_path
    for key in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ[key] = tmp_path
    tempfile.tempdir = tmp_path
    return tmp_dir


def _cleanup_case_tmpdir(case_tmp_dir: Path) -> None:
    """Remove case-local temporary directory unless explicitly preserved."""
    keep_tmp = str(os.environ.get('SITOOL_KEEP_TMP', '0')).strip().lower() in {
        '1', 'true', 'yes', 'on',
    }
    if keep_tmp:
        logger.info("Keeping case temporary directory by SITOOL_KEEP_TMP: %s", case_tmp_dir)
        return
    if not case_tmp_dir.exists():
        return
    try:
        shutil.rmtree(case_tmp_dir)
        logger.info("Removed case temporary directory: %s", case_tmp_dir)
    except Exception as exc:
        logger.warning("Failed to remove case temporary directory %s (%s).", case_tmp_dir, exc)


def _resolve_eval_hms_with_override(recipe_hms: List[str]) -> List[str]:
    """Resolve effective eval_hms with optional environment override.

    Environment
    -----------
    SITOOL_EVAL_HMS_OVERRIDE:
        Comma-separated hemispheres, e.g. ``nh`` or ``sh`` or ``nh,sh``.
    """
    base = [str(h).lower() for h in (recipe_hms or []) if str(h).lower() in {'nh', 'sh'}]
    if not base:
        base = ['nh']

    raw = str(os.environ.get('SITOOL_EVAL_HMS_OVERRIDE', '') or '').strip()
    if not raw:
        return base

    req = []
    for tok in raw.replace(';', ',').split(','):
        t = tok.strip().lower()
        if t in {'nh', 'sh'} and t not in req:
            req.append(t)
    if not req:
        logger.warning(
            "Ignoring invalid SITOOL_EVAL_HMS_OVERRIDE='%s'; keep recipe eval_hms=%s.",
            raw, base,
        )
        return base

    selected = [h for h in base if h in req]
    if not selected:
        logger.warning(
            "SITOOL_EVAL_HMS_OVERRIDE='%s' does not intersect recipe eval_hms=%s; keep recipe values.",
            raw, base,
        )
        return base

    logger.info(
        "Applying eval_hms override from env: requested=%s, effective=%s (recipe=%s).",
        req, selected, base,
    )
    return selected


def main() -> None:
    """Parse CLI arguments and run the requested evaluation modules."""
    parser = argparse.ArgumentParser(
        description='SIToolv2 — Sea Ice Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('case_name', help='Evaluation case name (matches recipe_<case_name>.yml)')
    parser.add_argument(
        '--modules', nargs='+', choices=ALL_MODULES, metavar='MODULE',
        help='Modules to evaluate (default: all enabled in recipe)',
    )
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity (default: INFO)',
    )
    parser.add_argument(
        '-r', '--rotate', action='append', default=[], metavar='FILE,VAR',
        help=(
            "Deprecated placeholder for SIdrift custom angle rotation. "
            "Accepted for compatibility but currently ignored."
        ),
    )
    parser.add_argument(
        '--recalculate', action='store_true',
        help='Ignore cached metrics and recompute all requested module diagnostics.',
    )
    parser.add_argument(
        '-j', '--jobs', type=int, default=1, metavar='N',
        help='Total parallel worker threads. Safe options are logged at runtime (default: 1).',
    )
    parser.add_argument(
        '--keep-staging', action='store_true',
        help='Keep per-run staging artifacts under cases/<case>/metrics/_staging.',
    )
    args = parser.parse_args()

    case_dir = _resolve_case_run_dir(args.case_name, log_warning=False)
    os.environ['SITOOL_CASE_RUN_DIR'] = str(case_dir)
    case_tmp_dir = _configure_case_tmpdir(case_dir)
    os.environ['SITOOL_LOG_LEVEL'] = str(args.log_level).upper()
    log_file = str(case_dir / f'{args.case_name}.log')
    setup_logging(getattr(logging, args.log_level), log_file=log_file)
    preferred_case_dir = Path('cases') / args.case_name
    if case_dir != preferred_case_dir:
        logger.warning(
            "Using fallback case run directory '%s' because '%s' is not usable.",
            case_dir, preferred_case_dir
        )

    _install_thread_safe_plot_wrappers()
    effective_jobs = _resolve_worker_count(args.jobs, max_cpu_fraction=_DEFAULT_MAX_CPU_FRACTION)
    logger.info("Parallel workers (system gate): requested=%d, effective=%d", int(args.jobs), int(effective_jobs))

    logger.info("SIToolv2 starting — case: %s", args.case_name)
    if args.rotate:
        logger.warning(
            "--rotate is currently ignored. SIdrift rotation now follows recipe "
            "model_direction during metric calculation."
        )

    # Load and validate recipe
    try:
        recipe = RR.RecipeReader(args.case_name)
    except ValueError as exc:
        logger.error("Failed to load recipe: %s", exc)
        sys.exit(1)

    # Determine which modules to run
    if args.modules:
        modules_to_run = args.modules
    else:
        # Run all modules whose recipe flag is True (default True if key absent)
        modules_to_run = [
            m for m in ALL_MODULES
            if m in recipe.variables
            and recipe.common_config.get(_MODULE_FLAG.get(m, ''), True)
        ]

    if not modules_to_run:
        logger.warning("No modules to run. Check recipe flags or --modules argument.")
        sys.exit(0)

    logger.info("Modules to evaluate: %s", ', '.join(modules_to_run))

    # Resolve the list of hemispheres to evaluate from the recipe.
    eval_hms: List[str] = _resolve_eval_hms_with_override(recipe.hemispheres)

    selected_jobs, safe_options, per_hemi_cap = _resolve_safe_job_plan(
        requested_jobs=effective_jobs,
        recipe=recipe,
        modules_to_run=modules_to_run,
        eval_hms=eval_hms,
    )
    if selected_jobs != effective_jobs:
        logger.warning(
            "Requested effective jobs=%d adjusted to safe jobs=%d. Safe options: %s",
            effective_jobs, selected_jobs, safe_options,
        )
    else:
        logger.info("Safe thread options for this case: %s", safe_options)
    logger.info(
        "Parallel profile: hemispheres=%s, per_hemisphere_cap=%d, selected_jobs=%d",
        eval_hms, per_hemi_cap, selected_jobs,
    )

    os.environ['SITOOL_DISABLE_TQDM'] = '1' if selected_jobs > 1 else '0'
    run_id = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S') + f'_{uuid.uuid4().hex[:8]}'
    os.environ['SITOOL_RUN_ID'] = run_id

    enable_cpu_trace = selected_jobs > 1
    cpu_trace_file: Optional[Path] = None
    if enable_cpu_trace:
        cpu_trace_file = case_tmp_dir / f'.cpu_usage_trace_{run_id}.jsonl'
        rte.initialize_trace(
            trace_file=cpu_trace_file,
            total_threads=selected_jobs,
            case_name=args.case_name,
            run_id=run_id,
        )
        logger.info("Real CPU utilization trace enabled: %s", cpu_trace_file)
    else:
        rte.disable_trace()

    eff_png = case_dir / 'cpu_realtime_efficiency_latest.png'
    legacy_eff_csv = case_dir / 'cpu_realtime_efficiency_latest.csv'
    legacy_snapshots_dir = case_dir / 'cpu_efficiency_snapshots'
    if legacy_snapshots_dir.exists():
        try:
            shutil.rmtree(legacy_snapshots_dir)
            logger.info("Removed legacy CPU efficiency snapshot directory: %s", legacy_snapshots_dir)
        except Exception as exc:
            logger.warning(
                "Failed to remove legacy CPU efficiency snapshot directory %s (%s).",
                legacy_snapshots_dir, exc,
            )
    cpu_fig_run_start_ts = time.monotonic()
    try:
        _cpu_fig_interval_env = float(os.environ.get('SITOOL_CPU_EFF_FIG_INTERVAL_SECONDS', '300'))
    except Exception:
        _cpu_fig_interval_env = 300.0
    cpu_fig_interval_seconds = max(60.0, float(_cpu_fig_interval_env))
    next_cpu_fig_export_ts = cpu_fig_run_start_ts + cpu_fig_interval_seconds
    cpu_fig_export_index = 0
    if enable_cpu_trace:
        logger.info(
            "Periodic CPU utilization figure export enabled: every %.0fs (overwrite latest figure).",
            cpu_fig_interval_seconds,
        )

    def _maybe_export_cpu_figure(force: bool = False, reason: str = 'periodic') -> bool:
        nonlocal next_cpu_fig_export_ts, cpu_fig_export_index
        if not enable_cpu_trace or cpu_trace_file is None:
            return False
        now_mono = time.monotonic()
        if not force and now_mono < next_cpu_fig_export_ts:
            return False
        if not cpu_trace_file.exists():
            return False
        try:
            rte.export_trace_outputs(
                trace_file=cpu_trace_file,
                output_png=eff_png,
                output_csv=None,
                chart_title=f'Real CPU Utilization ({args.case_name})',
            )
            cpu_fig_export_index += 1
            logger.info(
                "Real CPU utilization figure (%s #%d): %s",
                reason, cpu_fig_export_index, eff_png,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to export %s real CPU utilization figure (%s).", reason, exc)
            return False
        finally:
            while next_cpu_fig_export_ts <= now_mono:
                next_cpu_fig_export_ts += cpu_fig_interval_seconds

    # Create all directories up-front (including per-module output dirs).
    data_dir, output_dir = _init_case_dirs(case_dir, eval_hms, modules_to_run)

    # metric_tables is nested: {hms: {module: table_dict}}
    # Load persisted snapshot first so partial reruns can refresh one module
    # without dropping previously rendered modules from summary_report.html.
    metric_tables: dict = _load_metric_table_store(output_dir=output_dir, hemispheres=eval_hms)

    enabled_modules = [m for m in modules_to_run if m in recipe.variables]
    if not enabled_modules:
        logger.warning("No valid modules found in recipe for this run.")
        sys.exit(0)

    try:
        bootstrap_module = enabled_modules[0]
        _prepare_eval_grids_serial(
            case_name=args.case_name,
            bootstrap_module=bootstrap_module,
            eval_hms=eval_hms,
        )
    except Exception as exc:
        logger.error("Failed to prepare evaluation grids before scheduling (%s).", exc)
        sys.exit(1)

    # Scheduler policy:
    # 1) selected_jobs == 1: hemisphere-major order (NH all modules -> SH all modules),
    #    matching the single-thread requirement.
    # 2) selected_jobs > 1: dynamic task dispatcher over all (hemisphere, module)
    #    tasks. Real CPU utilization is sampled at runtime and used as feedback
    #    to increase task concurrency when the machine appears under-utilized.
    if selected_jobs == 1:
        logger.info(
            "Single-thread scheduler enabled (hemisphere-major): hemispheres=%s, modules=%s",
            eval_hms, enabled_modules,
        )
        for hms in eval_hms:
            for module in enabled_modules:
                logger.info("[%s/%s] Started ...", hms.upper(), module)
                try:
                    payload = _run_one_module_task(
                        case_name=args.case_name,
                        hemisphere=hms,
                        module=module,
                        data_dir=data_dir / hms,
                        output_dir=output_dir / hms,
                        recalculate=args.recalculate,
                        jobs_for_module=1,
                        isolated_worker_logging=False,
                    )
                    result = payload.get('result')
                    logger.info("[%s/%s] Success", hms.upper(), module)
                    if result is not None:
                        metric_tables[hms][module] = result
                    _refresh_html_report(
                        case_name=args.case_name,
                        output_dir=output_dir,
                        modules_to_run=modules_to_run,
                        metric_tables=metric_tables,
                        context=f'{hms.upper()}/{module}',
                    )
                except Exception:
                    logger.error("[%s/%s] FAILED (see log)", hms.upper(), module, exc_info=True)
    else:
        all_tasks: List[Tuple[str, str]] = []
        for module in enabled_modules:
            for hms in eval_hms:
                all_tasks.append((hms, module))

        case_name_norm = str(args.case_name or '').strip().lower()
        aggressive_parallel = (
            case_name_norm.startswith('highres')
            or str(os.environ.get('SITOOL_AGGRESSIVE_PARALLEL', '')).strip().lower() in {
                '1', 'true', 'yes', 'on',
            }
        )
        if aggressive_parallel:
            priority = {
                'SICB': 0,
                'SIdrift': 1,
                'SItrans': 2,
                'SIMbudget': 2,
                'SNMbudget': 2,
                'SIthick': 3,
                'SNdepth': 3,
                'SIconc': 4,
            }
            all_tasks = sorted(
                all_tasks,
                key=lambda task: (
                    int(priority.get(str(task[1]), 100)),
                    str(task[0]),
                    str(task[1]),
                ),
            )
            logger.info(
                "Aggressive parallel policy enabled for case '%s' "
                "(priority dispatch + higher per-task thread budgets).",
                args.case_name,
            )

        max_parallel_tasks = min(len(all_tasks), selected_jobs)
        try:
            _default_initial_ratio = 1.00 if aggressive_parallel else 0.75
            _initial_ratio = float(
                os.environ.get('SITOOL_INITIAL_PARALLEL_RATIO', str(_default_initial_ratio))
            )
        except Exception:
            _initial_ratio = 0.75
        initial_ratio = min(1.0, max(0.25, _initial_ratio))
        target_parallel_tasks = max(
            1,
            min(max_parallel_tasks, int(math.ceil(float(max_parallel_tasks) * initial_ratio))),
        )
        logger.info(
            "Dynamic task scheduler enabled: total_tasks=%d, max_parallel_tasks=%d, "
            "initial_target_parallel=%d, selected_jobs=%d, initial_ratio=%.2f",
            len(all_tasks), max_parallel_tasks, target_parallel_tasks, selected_jobs, initial_ratio,
        )

        pending_tasks: deque = deque(all_tasks)
        # SICB is the longest-running module. Allow dual-hemisphere SICB tasks
        # to overlap on larger thread budgets to avoid long low-utilization tails.
        max_sicb_parallel = 2 if (len(eval_hms) >= 2 and selected_jobs >= 24) else 1
        if aggressive_parallel:
            default_sicb_cap = max(
                8,
                int(math.ceil(float(selected_jobs) / float(max(1, max_sicb_parallel)))),
            )
        else:
            default_sicb_cap = max(
                2,
                min(
                    6,
                    int(math.ceil(selected_jobs / max(1, max_sicb_parallel * 4))),
                ),
            )
        try:
            _sicb_env_cap = int(os.environ.get('SITOOL_SICB_MAX_JOBS', str(default_sicb_cap)))
        except Exception:
            _sicb_env_cap = default_sicb_cap
        sicb_jobs_cap = max(1, min(int(selected_jobs), int(_sicb_env_cap)))
        if aggressive_parallel:
            default_sidrift_cap = max(
                4,
                int(math.ceil(float(selected_jobs) / float(max(2, len(eval_hms) * 2)))),
            )
        else:
            default_sidrift_cap = max(2, min(4, int(math.ceil(selected_jobs / 12.0))))
        try:
            _sidrift_env_cap = int(os.environ.get('SITOOL_SIDRIFT_MAX_JOBS', str(default_sidrift_cap)))
        except Exception:
            _sidrift_env_cap = default_sidrift_cap
        sidrift_jobs_cap = max(1, min(int(selected_jobs), int(_sidrift_env_cap)))
        default_simbudget_cap = max(2, int(math.ceil(selected_jobs / 4.0)))
        try:
            _simbudget_env_cap = int(os.environ.get('SITOOL_SIMBUDGET_MAX_JOBS', str(default_simbudget_cap)))
        except Exception:
            _simbudget_env_cap = default_simbudget_cap
        simbudget_jobs_cap = max(1, min(int(selected_jobs), int(_simbudget_env_cap)))
        if aggressive_parallel:
            default_core_cap = max(
                2,
                int(math.ceil(float(selected_jobs) / float(max(2, len(eval_hms) * 3)))),
            )
        else:
            default_core_cap = 1
            if selected_jobs >= 4:
                default_core_cap = 2
            if selected_jobs >= 24:
                default_core_cap = 3
        try:
            _core_env_cap = int(os.environ.get('SITOOL_CORE_MODULE_MAX_JOBS', str(default_core_cap)))
        except Exception:
            _core_env_cap = default_core_cap
        core_jobs_cap = max(1, min(int(selected_jobs), int(_core_env_cap)))
        logger.info(
            "Dynamic task guard: max concurrent SICB tasks = %d "
            "(SICB jobs cap=%d, SIdrift jobs cap=%d, SIMbudget jobs cap=%d, core-module jobs cap=%d)",
            max_sicb_parallel, sicb_jobs_cap, sidrift_jobs_cap, simbudget_jobs_cap, core_jobs_cap,
        )

        def _can_dispatch(task: Tuple[str, str], running_info: Dict[Any, Dict[str, Any]]) -> bool:
            _hms, module_name = task
            if module_name == 'SICB':
                running_sicb = sum(
                    1 for info in running_info.values()
                    if str(info.get('module')) == 'SICB'
                )
                if running_sicb >= max_sicb_parallel:
                    return False
            return True

        def _has_dispatchable_pending(running_info: Dict[Any, Dict[str, Any]]) -> bool:
            for task in list(pending_tasks):
                if _can_dispatch(task, running_info):
                    return True
            return False

        mp_ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=max_parallel_tasks,
            mp_context=mp_ctx,
            max_tasks_per_child=1,
        ) as pool:
            running: Dict[Any, Dict[str, Any]] = {}
            last_wait_log_ts = 0.0

            while running or pending_tasks:
                launched_in_round = False
                dispatch_budget = len(pending_tasks)
                while len(running) < target_parallel_tasks and dispatch_budget > 0 and pending_tasks:
                    task = pending_tasks.popleft()
                    dispatch_budget -= 1
                    if not _can_dispatch(task, running):
                        pending_tasks.append(task)
                        continue

                    hms, module_name = task
                    requested_by_running = sum(
                        max(1, int(info.get('module_jobs', 1)))
                        for info in running.values()
                    )
                    available_jobs = max(1, int(selected_jobs) - int(requested_by_running))
                    remaining_slots = max(1, int(target_parallel_tasks) - int(len(running)))
                    reserve_for_future = max(0, remaining_slots - 1)
                    allocatable_jobs = max(1, available_jobs - reserve_for_future)
                    base_jobs = max(
                        1,
                        int(math.ceil(float(selected_jobs) / float(max(1, int(target_parallel_tasks))))),
                    )
                    weighted_base_jobs = base_jobs
                    if aggressive_parallel:
                        weight_map = {
                            'SICB': 1.50,
                            'SIdrift': 1.25,
                            'SItrans': 1.25,
                            'SIMbudget': 1.20,
                            'SNMbudget': 1.20,
                            'SIconc': 1.00,
                            'SIthick': 1.00,
                            'SNdepth': 1.00,
                        }
                        module_weight = float(weight_map.get(module_name, 1.0))
                        weighted_base_jobs = max(1, int(math.ceil(float(base_jobs) * module_weight)))
                    if module_name == 'SICB':
                        module_jobs = min(max(weighted_base_jobs, 2), sicb_jobs_cap)
                    elif module_name == 'SIdrift':
                        module_jobs = min(max(weighted_base_jobs, 2), sidrift_jobs_cap)
                    elif module_name in {'SIMbudget', 'SNMbudget'}:
                        module_jobs = min(max(weighted_base_jobs, 2), simbudget_jobs_cap)
                    elif module_name in {'SIconc', 'SIthick', 'SNdepth', 'SItrans'}:
                        module_jobs = min(max(weighted_base_jobs, 1), core_jobs_cap)
                    else:
                        module_jobs = weighted_base_jobs
                    module_jobs = max(1, min(int(module_jobs), int(allocatable_jobs)))
                    min_dispatch_jobs = 1
                    if aggressive_parallel:
                        if module_name == 'SICB':
                            min_dispatch_jobs = 2
                        elif module_name in {'SIdrift', 'SItrans', 'SIMbudget', 'SNMbudget'}:
                            min_dispatch_jobs = 2
                        elif module_name in {'SIconc', 'SIthick', 'SNdepth'}:
                            min_dispatch_jobs = 1
                    if module_jobs < min_dispatch_jobs and running:
                        pending_tasks.append(task)
                        continue
                    logger.info(
                        "[%s/%s] Queued by dynamic scheduler: running_tasks=%d, target_parallel=%d, "
                        "module_jobs=%d (available=%d, running_requested=%d), pending=%d",
                        hms.upper(), module_name, len(running) + 1, target_parallel_tasks,
                        module_jobs, available_jobs, requested_by_running, len(pending_tasks),
                    )
                    fut = pool.submit(
                        _run_one_module_task,
                        case_name=args.case_name,
                        hemisphere=hms,
                        module=module_name,
                        data_dir=data_dir / hms,
                        output_dir=output_dir / hms,
                        recalculate=args.recalculate,
                        jobs_for_module=module_jobs,
                        isolated_worker_logging=True,
                    )
                    running[fut] = {
                        'hemisphere': hms,
                        'module': module_name,
                        'module_jobs': module_jobs,
                        'started_at': time.time(),
                    }
                    launched_in_round = True

                if not running:
                    if not pending_tasks:
                        break
                    if not launched_in_round:
                        time.sleep(0.2)
                    continue

                done, _pending = wait(
                    list(running.keys()),
                    timeout=3.0,
                    return_when=FIRST_COMPLETED,
                )
                _maybe_export_cpu_figure(force=False, reason='periodic')
                if not done:
                    now_ts = time.time()
                    if now_ts - last_wait_log_ts >= 30.0:
                        running_desc = []
                        for info in sorted(
                            running.values(),
                            key=lambda item: (str(item.get('hemisphere')), str(item.get('module'))),
                        ):
                            elapsed = int(now_ts - float(info.get('started_at', now_ts)))
                            running_desc.append(
                                f"{str(info.get('hemisphere')).upper()}/{info.get('module')}({elapsed}s)"
                            )
                        logger.info(
                            "Waiting for running tasks (%d): %s; pending=%d",
                            len(running),
                            ', '.join(running_desc),
                            len(pending_tasks),
                        )
                        last_wait_log_ts = now_ts
                    if (
                        enable_cpu_trace
                        and cpu_trace_file is not None
                        and target_parallel_tasks < max_parallel_tasks
                        and _has_dispatchable_pending(running)
                    ):
                        cpu_util = rte.latest_cpu_utilization_percent(
                            trace_file=cpu_trace_file,
                            fallback_percent=0.0,
                        )
                        try:
                            _default_low_watermark = 55.0 if aggressive_parallel else 80.0
                            _low_cpu_env = float(
                                os.environ.get(
                                    'SITOOL_CPU_LOW_WATERMARK',
                                    str(_default_low_watermark),
                                )
                            )
                        except Exception:
                            _low_cpu_env = 55.0 if aggressive_parallel else 80.0
                        low_bound = 25.0 if aggressive_parallel else 20.0
                        low_cpu_watermark = min(99.0, max(low_bound, _low_cpu_env))
                        if cpu_util < low_cpu_watermark:
                            step = 2 if cpu_util < max(20.0, low_cpu_watermark - 15.0) else 1
                            desired = min(
                                max_parallel_tasks,
                                max(target_parallel_tasks + int(step), target_parallel_tasks + 1),
                            )
                            old_target = target_parallel_tasks
                            target_parallel_tasks = desired
                            logger.info(
                                "Real CPU utilization %.1f%% < %.1f%%. Increasing parallel tasks: %d -> %d",
                                cpu_util, low_cpu_watermark, old_target, target_parallel_tasks,
                            )
                    continue

                for fut in done:
                    info = running.pop(fut)
                    hms = str(info.get('hemisphere'))
                    module_name = str(info.get('module'))
                    try:
                        payload = fut.result()
                        result = payload.get('result')
                        logger.info("[%s/%s] Success", hms.upper(), module_name)
                        worker_log = payload.get('worker_log')
                        if worker_log:
                            logger.info("[%s/%s] Worker log: %s", hms.upper(), module_name, worker_log)
                        if result is not None:
                            metric_tables[hms][module_name] = result
                        _refresh_html_report(
                            case_name=args.case_name,
                            output_dir=output_dir,
                            modules_to_run=modules_to_run,
                            metric_tables=metric_tables,
                            context=f'{hms.upper()}/{module_name}',
                        )
                    except Exception:
                        logger.error("[%s/%s] FAILED (see log)", hms.upper(), module_name, exc_info=True)

    build_unified_cache = str(os.environ.get('SITOOL_BUILD_UNIFIED_CACHE', '0')).strip().lower() in {
        '1', 'true', 'yes', 'on',
    }
    if build_unified_cache:
        logger.info("Building unified metrics cache ...")
        cache_build_t0 = time.time()
        unified_cache = _build_unified_metrics_cache(
            case_name=args.case_name,
            case_dir=case_dir,
            eval_hms=eval_hms,
            modules_to_run=modules_to_run,
        )
        logger.info(
            "Unified cache build finished in %.1fs.",
            max(0.0, time.time() - cache_build_t0),
        )
        if unified_cache is not None:
            logger.info("Unified metrics cache ready: %s", unified_cache)
    else:
        logger.info(
            "Skip unified metrics cache build (set SITOOL_BUILD_UNIFIED_CACHE=1 to enable)."
        )

    _refresh_html_report(
        case_name=args.case_name,
        output_dir=output_dir,
        modules_to_run=modules_to_run,
        metric_tables=metric_tables,
        context='finalize',
    )

    if args.keep_staging:
        logger.info("Keeping staging directory for this run: %s", case_dir / 'metrics' / '_staging' / run_id)
    else:
        _cleanup_stage_dir(case_dir=case_dir, run_id=run_id)

    if enable_cpu_trace and cpu_trace_file is not None:
        try:
            _maybe_export_cpu_figure(force=True, reason='final')
            try:
                if legacy_eff_csv.exists():
                    legacy_eff_csv.unlink()
                    logger.info("Removed legacy CPU efficiency CSV: %s", legacy_eff_csv)
            except Exception as exc:
                logger.warning("Failed to remove legacy CPU efficiency CSV (%s).", exc)
        finally:
            try:
                cpu_trace_file.unlink(missing_ok=True)
            except Exception:
                pass
            rte.disable_trace()
    else:
        logger.info("Skip real CPU utilization figure export because selected_jobs=%d.", selected_jobs)
        rte.disable_trace()

    _cleanup_case_tmpdir(case_tmp_dir)

    logger.info("SIToolv2 finished.")


__all__ = ["main"]
