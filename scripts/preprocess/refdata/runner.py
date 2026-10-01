# -*- coding: utf-8 -*-
"""YAML-driven runner for offline SITool reference-data utilities.

Usage
-----
Run from repository root:

    # Environment-variable-based config (original workflow):
    python -m scripts.preprocess.refdata.runner --config cases/refdata_prep_default.yml

    # User-friendly config with inline path variables (recommended):
    python -m scripts.preprocess.refdata.runner --config cases/refdata_prep_user.yml

    # Abort on any validation failure instead of skipping bad datasets:
    python -m scripts.preprocess.refdata.runner --config cases/refdata_prep_user.yml --strict

Configuration schema
--------------------
Optional top-level variables (resolved before environment-variable expansion):

    raw_data_root: /path/to/raw/data    # expanded as ${raw_data_root}
    ref_data_dir:  /path/to/RefData     # expanded as ${ref_data_dir}

Required when prep_tasks is non-empty:

    ref_data_dir: <processed reference output root>

Optional enable-flag block (used by refdata_prep_user.yml):

    enable:
      NSIDC_SIconc: true
      PIOMAS_SIthick: false   # skip this dataset

Task lists:

    prep_tasks:
      - method: prep_NSIDC_SIconc
        enable_key: NSIDC_SIconc   # optional; matched against enable block above
        kwargs:
          data_path: ${raw_data_root}/NSIDC/G02202_V5
          hemisphere: nh
    quick_check_tasks:
      - function: plot_check_miss_general
        kwargs: {...}
    quick_look_tasks:
      - function: ql_SIdrift_trend
        kwargs: {...}

Notes
-----
- String values support ``$VAR``/``${VAR}`` environment-variable expansion
  (applied *after* inline ``${raw_data_root}`` / ``${ref_data_dir}`` expansion).
- Tasks with ``enable_key`` set to false in the ``enable`` block are skipped.
- Filesystem validation runs before any processing; failed datasets are skipped
  unless ``--strict`` is passed.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from scripts.preprocess.refdata.refdata_prep import ReferenceDataManager
from scripts.preprocess.refdata import quick_check
from scripts.preprocess.refdata import quick_look
from scripts.preprocess.refdata.validator import (
    ValidationResult,
    print_validation_report,
    validate_tasks,
)

logger = logging.getLogger(__name__)

_INFILE_VARS = ("raw_data_root", "ref_data_dir")


# ---------------------------------------------------------------------------
# Parallel worker — must be at module level for pickle compatibility
# ---------------------------------------------------------------------------

def _prep_worker_init(log_queue: "multiprocessing.Queue", log_level: int = logging.INFO) -> None:
    """Initializer called once per worker process to configure logging.

    With the ``spawn`` start method, Queue objects cannot be pickled through
    ``executor.submit()`` arguments.  The ``initializer`` / ``initargs``
    mechanism is the correct way to inject a shared queue into worker processes.
    """
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(log_level)


def _prep_worker(
    ref_data_dir: str,
    method_name: str,
    kwargs: dict,
    inner_jobs: int = 1,
) -> str:
    """Run one prep task (logging is already configured by *_prep_worker_init*)."""
    manager = ReferenceDataManager(ref_data_dir=ref_data_dir, inner_jobs=inner_jobs)
    getattr(manager, method_name)(**kwargs)
    return method_name


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def _expand_env_values(value: Any) -> Any:
    """Recursively expand environment variables and '~' in string values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_env_values(v) for k, v in value.items()}
    return value


def _expand_infile_vars(data: Dict[str, Any]) -> Dict[str, Any]:
    """Expand ``${raw_data_root}`` and ``${ref_data_dir}`` using values
    defined in the same YAML mapping.

    This is applied *before* environment-variable expansion so that users can
    write ``${raw_data_root}/NSIDC/G02202_V5`` in ``refdata_prep_user.yml``
    without having to set a shell environment variable.
    """
    replacements: Dict[str, str] = {}
    for var in _INFILE_VARS:
        raw = str(data.get(var) or "").strip()
        if raw:
            replacements[var] = raw

    if not replacements:
        return data

    pattern = re.compile(
        r"\$\{(" + "|".join(re.escape(k) for k in replacements) + r")\}"
    )

    def _sub(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(lambda m: replacements[m.group(1)], value)
        if isinstance(value, list):
            return [_sub(item) for item in value]
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        return value

    return _sub(data)


def _load_config(config_path: str) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    # Resolve in-file variables first, then shell environment variables.
    data = _expand_infile_vars(data)
    return _expand_env_values(data)


# ---------------------------------------------------------------------------
# Enable-flag filtering
# ---------------------------------------------------------------------------

def _filter_enabled_tasks(tasks: List[Any], enable: Dict[str, bool]) -> Tuple[List[Any], List[str]]:
    """Return (enabled_tasks, skipped_labels) based on the ``enable`` mapping.

    Tasks that have no ``enable_key`` field are always included.
    """
    if not enable:
        return list(tasks), []

    enabled: List[Any] = []
    skipped: List[str] = []

    for task in tasks:
        if not isinstance(task, dict):
            enabled.append(task)
            continue
        key = task.get("enable_key")
        if key is None:
            enabled.append(task)
        elif enable.get(key, True):
            enabled.append(task)
        else:
            method = task.get("method", key)
            hemi = (task.get("kwargs") or {}).get("hemisphere", "")
            label = f"{method}({hemi})" if hemi else method
            skipped.append(label)

    return enabled, skipped


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def _run_prep_tasks(
    config: Dict[str, Any],
    tasks: List[Any],
    skip_methods: Optional[set] = None,
    jobs: int = 1,
    inner_jobs: int = 1,
) -> None:
    """Execute prep tasks using :class:`ReferenceDataManager`.

    Parameters
    ----------
    config:
        Full parsed config dict (used to resolve ``ref_data_dir``).
    tasks:
        Already-filtered list of task dicts to execute.
    skip_methods:
        Set of (method, data_path) tuples to skip due to validation failure.
    jobs:
        Number of parallel worker processes (task-level).  ``1`` runs sequentially.
    inner_jobs:
        Number of threads each worker uses for intra-task loops (e.g. OSI SIconc
        year loop, MPMR year loop).  ``1`` disables intra-task parallelism.
    """
    if not tasks:
        return

    ref_data_dir = str(config.get("ref_data_dir") or "").strip()
    if not ref_data_dir:
        raise ValueError("`ref_data_dir` is required when `prep_tasks` is provided")

    skip_methods = skip_methods or set()

    # --- Validate task specs and build work list (always done in main process) ---
    # Use a bare instance (inner_jobs=1) just for attribute existence checks.
    _checker = ReferenceDataManager(ref_data_dir=ref_data_dir)
    work_items: List[Tuple[int, str, dict]] = []
    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"`prep_tasks[{idx}]` must be a mapping")
        method_name = str(task.get("method") or "").strip()
        kwargs = task.get("kwargs") or {}
        if not method_name:
            raise ValueError(f"`prep_tasks[{idx}].method` is required")
        if not isinstance(kwargs, dict):
            raise ValueError(f"`prep_tasks[{idx}].kwargs` must be a mapping")
        if not hasattr(_checker, method_name):
            raise AttributeError(f"ReferenceDataManager has no method: {method_name}")

        key = (method_name, kwargs.get("data_path", ""))
        if key in skip_methods:
            logger.warning(
                "[prep %d/%d] SKIPPED (validation failed): %s data_path=%s",
                idx, len(tasks), method_name, kwargs.get("data_path", ""),
            )
            continue
        work_items.append((idx, method_name, kwargs))

    if not work_items:
        return

    # --- Sequential path ---
    if jobs <= 1:
        manager = ReferenceDataManager(ref_data_dir=ref_data_dir, inner_jobs=inner_jobs)
        for idx, method_name, kwargs in work_items:
            logger.info("[prep %d/%d] %s(%s)", idx, len(tasks), method_name, kwargs)
            getattr(manager, method_name)(**kwargs)
        return

    # --- Parallel path ---
    logger.info(
        "Parallel mode: %d tasks, %d outer workers, %d inner threads (spawn).",
        len(work_items), jobs, inner_jobs,
    )
    mp_ctx = multiprocessing.get_context("spawn")
    log_queue: multiprocessing.Queue = mp_ctx.Queue(-1)
    # Replicate the root logger's handlers inside the listener so that worker
    # log records reach the same destinations (stderr, file, …) as the main process.
    root_handlers = logging.getLogger().handlers or [logging.StreamHandler()]
    listener = logging.handlers.QueueListener(
        log_queue, *root_handlers, respect_handler_level=True
    )
    listener.start()
    try:
        _root_level = logging.getLogger().level or logging.INFO
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=mp_ctx,
            initializer=_prep_worker_init,
            initargs=(log_queue, _root_level),
        ) as executor:
            futures = {
                executor.submit(_prep_worker, ref_data_dir, method_name, kwargs, inner_jobs):
                (idx, method_name, kwargs)
                for idx, method_name, kwargs in work_items
            }
            logger.info("Submitted %d tasks — waiting for completion ...", len(futures))

            first_exc: Optional[Exception] = None
            completed = 0
            for future in as_completed(futures):
                idx, method_name, kwargs = futures[future]
                try:
                    future.result()
                    completed += 1
                    logger.info(
                        "[prep %d/%d done] %s",
                        completed, len(work_items), method_name,
                    )
                except Exception as exc:
                    logger.error(
                        "[prep FAILED] %s(%s): %s",
                        method_name, kwargs, exc, exc_info=True,
                    )
                    if first_exc is None:
                        first_exc = exc
                        for f in futures:
                            f.cancel()

            if first_exc is not None:
                raise first_exc
    finally:
        listener.stop()


def _run_module_tasks(config: Dict[str, Any], *, key: str, module_obj: Any) -> None:
    tasks = config.get(key) or []
    if not tasks:
        return
    if not isinstance(tasks, list):
        raise ValueError(f"`{key}` must be a list")

    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"`{key}[{idx}]` must be a mapping")
        func_name = str(task.get("function") or "").strip()
        kwargs = task.get("kwargs") or {}
        if not func_name:
            raise ValueError(f"`{key}[{idx}].function` is required")
        if not isinstance(kwargs, dict):
            raise ValueError(f"`{key}[{idx}].kwargs` must be a mapping")
        if not hasattr(module_obj, func_name):
            raise AttributeError(
                f"Function not found in module `{module_obj.__name__}`: {func_name}"
            )
        logger.info("[%s %d/%d] %s(%s)", key, idx, len(tasks), func_name, kwargs)
        getattr(module_obj, func_name)(**kwargs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_from_config(config_path: str, strict: bool = False,
                    jobs: int = 1, inner_jobs: int = 1) -> None:
    """Execute reference-data tasks from one YAML config file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    strict:
        If True, abort with a non-zero exit code when any dataset fails
        validation instead of skipping it.
    jobs:
        Number of parallel worker processes for prep tasks (default: 1).
    inner_jobs:
        Number of threads each worker uses for intra-task loops (default: 1).
    """
    config = _load_config(config_path)

    all_tasks = config.get("prep_tasks") or []
    if not isinstance(all_tasks, list):
        raise ValueError("`prep_tasks` must be a list")

    enable: Dict[str, bool] = config.get("enable") or {}
    enabled_tasks, skipped_labels = _filter_enabled_tasks(all_tasks, enable)

    if skipped_labels:
        logger.info(
            "Skipping disabled datasets (%d tasks): %s",
            len(skipped_labels), ", ".join(skipped_labels),
        )

    # Validate filesystem paths for enabled tasks.
    skip_methods: set = set()
    if enabled_tasks:
        results: List[ValidationResult] = validate_tasks(enabled_tasks)
        print_validation_report(results)

        failed = [r for r in results if not r.ok]
        if failed:
            if strict:
                logger.error(
                    "Validation failed for %d dataset(s) and --strict is set. Aborting.",
                    len(failed),
                )
                sys.exit(1)
            # Build skip set: (method, data_path) pairs that failed validation.
            failed_paths = {r.message.split(" — ")[0].strip() for r in failed}
            for task in enabled_tasks:
                kwargs = task.get("kwargs") or {}
                dp = kwargs.get("data_path", "")
                if any(dp.startswith(fp) or fp in dp for fp in failed_paths):
                    skip_methods.add((str(task.get("method", "")), dp))
            if skip_methods:
                logger.warning(
                    "Skipping %d task(s) due to validation failures. "
                    "Use --strict to abort instead.",
                    len(skip_methods),
                )

    _run_prep_tasks(config, enabled_tasks, skip_methods=skip_methods,
                    jobs=jobs, inner_jobs=inner_jobs)
    _run_module_tasks(config, key="quick_check_tasks", module_obj=quick_check)
    _run_module_tasks(config, key="quick_look_tasks", module_obj=quick_look)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SITool reference-data preparation/check/quick-look tasks from YAML.",
    )
    parser.add_argument(
        "--config",
        default="cases/refdata_prep_user.yml",
        help=(
            "YAML config path "
            "(default: cases/refdata_prep_user.yml)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Abort with a non-zero exit code when any dataset fails path validation "
            "instead of skipping it and continuing."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: INFO)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int, default=1, metavar="N",
        help="Number of parallel worker processes for prep tasks (default: 1).",
    )
    parser.add_argument(
        "-J", "--inner-jobs",
        type=int, default=1, metavar="M",
        dest="inner_jobs",
        help=(
            "Number of threads each worker uses for intra-task parallelism "
            "(e.g. year loops in OSI SIconc and MPMR).  Default: 1."
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_from_config(args.config, strict=args.strict, jobs=args.jobs, inner_jobs=args.inner_jobs)


if __name__ == "__main__":
    main()
