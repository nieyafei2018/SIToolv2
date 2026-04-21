# -*- coding: utf-8 -*-
"""YAML-driven runner for offline SITool reference-data utilities.

Usage
-----
Run from repository root:

    python -m scripts.preprocess.refdata.runner --config cases/refdata_prep_default.yml

Configuration schema (minimal)
------------------------------
ref_data_dir: <processed reference output root, required when prep_tasks is non-empty>
prep_tasks:
  - method: prep_NSIDC_SIconc
    kwargs:
      data_path: /path/to/raw/input
      hemisphere: nh
quick_check_tasks:
  - function: plot_check_miss_general
    kwargs: {...}
quick_look_tasks:
  - function: ql_SIdrift_trend
    kwargs: {...}

Notes
-----
- String values support ``$VAR``/``${VAR}`` environment-variable expansion.
- This runner intentionally keeps the original callable names so legacy scripts,
  notebooks, and procedure notes remain easy to map.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from scripts.preprocess.refdata.refdata_prep import ReferenceDataManager
from scripts.preprocess.refdata import quick_check
from scripts.preprocess.refdata import quick_look

logger = logging.getLogger(__name__)


def _expand_env_values(value: Any) -> Any:
    """Recursively expand environment variables and '~' in string values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_env_values(v) for k, v in value.items()}
    return value


def _load_config(config_path: str) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    return _expand_env_values(data)


def _run_prep_tasks(config: Dict[str, Any]) -> None:
    tasks = config.get("prep_tasks") or []
    if not tasks:
        return
    if not isinstance(tasks, list):
        raise ValueError("`prep_tasks` must be a list")

    ref_data_dir = str(config.get("ref_data_dir") or "").strip()
    if not ref_data_dir:
        raise ValueError("`ref_data_dir` is required when `prep_tasks` is provided")

    manager = ReferenceDataManager(ref_data_dir=ref_data_dir)
    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"`prep_tasks[{idx}]` must be a mapping")
        method_name = str(task.get("method") or "").strip()
        kwargs = task.get("kwargs") or {}
        if not method_name:
            raise ValueError(f"`prep_tasks[{idx}].method` is required")
        if not isinstance(kwargs, dict):
            raise ValueError(f"`prep_tasks[{idx}].kwargs` must be a mapping")
        if not hasattr(manager, method_name):
            raise AttributeError(f"ReferenceDataManager has no method: {method_name}")
        logger.info("[prep %d/%d] %s(%s)", idx, len(tasks), method_name, kwargs)
        getattr(manager, method_name)(**kwargs)


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
            raise AttributeError(f"Function not found in module `{module_obj.__name__}`: {func_name}")
        logger.info("[%s %d/%d] %s(%s)", key, idx, len(tasks), func_name, kwargs)
        getattr(module_obj, func_name)(**kwargs)


def run_from_config(config_path: str) -> None:
    """Execute reference-data tasks from one YAML config file."""
    config = _load_config(config_path)
    _run_prep_tasks(config)
    _run_module_tasks(config, key="quick_check_tasks", module_obj=quick_check)
    _run_module_tasks(config, key="quick_look_tasks", module_obj=quick_look)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SITool reference-data preparation/check/quick-look tasks from YAML.",
    )
    parser.add_argument(
        "--config",
        default="cases/refdata_prep_default.yml",
        help="YAML config path (default: cases/refdata_prep_default.yml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_from_config(args.config)


if __name__ == "__main__":
    main()

