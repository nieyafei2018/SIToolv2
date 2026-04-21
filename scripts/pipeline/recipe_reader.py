# -*- coding: utf-8 -*-
"""
Recipe file reader and validator for SIToolv2 evaluation system.

This module provides functionality to read and validate recipe configuration
files for sea ice model evaluation experiments.
"""

import glob
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml

logger = logging.getLogger(__name__)

# Absolute path to the project-level `cases/` directory.
CASES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cases')
)


class RecipeReader:
    """Read and validate recipe configuration files for sea ice model evaluation."""

    SUPPORTED_MODULES = [
        "SIconc", "SIthick", "SNdepth", "SIdrift", "SICB", "SItrans",
        "SIMbudget", "SNMbudget",
    ]
    TIME_FREQUENCY_REQUIREMENTS = {'SICB': 1, 'SItrans': 1, 'default': 31}
    MODULE_FILE_KEYS = {
        'SIconc': ['model_file'],
        'SIthick': ['model_file'],
        'SNdepth': ['model_file'],
        'SItrans': ['model_file'],
        'SIdrift': ['model_file_u', 'model_file_v'],
        'SICB': ['model_file_sic', 'model_file_u', 'model_file_v'],
        'SIMbudget': [
            'model_file_sidmassdyn',
            'model_file_sidmassevapsubl',
            'model_file_sidmassgrowthbot',
            'model_file_sidmassgrowthwat',
            'model_file_sidmasslat',
            'model_file_sidmassmeltbot',
            'model_file_sidmassmelttop',
            'model_file_sidmasssi',
            'model_file_areacello',
        ],
        'SNMbudget': [
            'model_file_sndmassdyn',
            'model_file_sndmassmelt',
            'model_file_sndmasssi',
            'model_file_sndmasssnf',
            'model_file_sndmasssubl',
            'model_file_areacello',
        ],
    }

    def __init__(self, eval_ex: str):
        """Initialize the recipe reader.

        Args:
            eval_ex: Evaluation experiment name (matches recipe_<eval_ex>.yml).

        Raises:
            ValueError: If the recipe file or required data paths are not found.
        """
        self.eval_ex = eval_ex
        self.recipe_file = os.path.join(CASES_DIR, f"recipe_{eval_ex}.yml")

        if not os.path.exists(self.recipe_file):
            raise ValueError(f"Recipe file not found: {self.recipe_file}")

        with open(self.recipe_file, 'r', encoding='utf-8') as fh:
            recipe_config = yaml.safe_load(fh)

        self._expand_compact_recipe(recipe_config)
        self.variables = recipe_config['variables']
        self.common_config = self.variables['common']
        hms_raw = self.common_config.get('eval_hms', [])
        if isinstance(hms_raw, list):
            self.hemispheres = [h for h in hms_raw if h in ('nh', 'sh')]
        else:
            self.hemispheres = [hms_raw] if hms_raw in ('nh', 'sh') else []
        self.hemisphere = self.hemispheres[0] if self.hemispheres else 'nh'
        self.ref_data_path = self.common_config['SIToolv2_RefData_path']
        self.model_data_path = self.common_config['model_data_path']

        if not os.path.exists(self.ref_data_path):
            raise ValueError(f"Reference data path not found: {self.ref_data_path}")
        if not os.path.exists(self.model_data_path):
            raise ValueError(f"Model data path not found: {self.model_data_path}")

    # ------------------------------------------------------------------
    # Compact-recipe expansion
    # ------------------------------------------------------------------

    def _expand_compact_recipe(self, recipe_config: Dict[str, Any]) -> None:
        """Expand compact recipe declarations into canonical model_file fields.

        Supported compact keys (all optional and backward-compatible):
        - variables.common.model_groups: Dict[str, List[str]]
        - variables.<Module>.models: List[str] or str (single model or group name)
        - variables.<Module>.model_group: str or List[str]
        - variables.<Module>.model_exclude: List[str]
        - variables.<Module>.model_file_template: str (for single-file modules)
        - variables.<Module>.model_file_templates: Dict[file_key, template]

        Template strings must include ``{model}``, e.g. ``"{model}/siconc_*.nc"``.
        """
        if not isinstance(recipe_config, dict):
            raise ValueError("Recipe root must be a mapping.")
        variables = recipe_config.get('variables')
        if not isinstance(variables, dict):
            raise ValueError("Recipe must contain a 'variables' mapping.")

        common = variables.get('common')
        if not isinstance(common, dict):
            raise ValueError("Recipe must contain variables.common mapping.")

        model_groups = self._normalize_model_groups(common.get('model_groups'))

        for module_name in self.SUPPORTED_MODULES:
            module_cfg = variables.get(module_name)
            if not isinstance(module_cfg, dict):
                continue
            self._expand_module_compact_fields(
                module_name=module_name,
                module_cfg=module_cfg,
                model_groups=model_groups,
            )

    @staticmethod
    def _normalize_model_groups(raw_groups: Any) -> Dict[str, List[str]]:
        """Normalize common.model_groups into Dict[str, List[str]]."""
        if raw_groups is None:
            return {}
        if not isinstance(raw_groups, dict):
            raise ValueError("variables.common.model_groups must be a mapping.")

        groups: Dict[str, List[str]] = {}
        for group_name, members in raw_groups.items():
            if isinstance(members, str):
                groups[str(group_name)] = [members]
                continue
            if not isinstance(members, list):
                raise ValueError(
                    f"model_groups['{group_name}'] must be a list of model names."
                )
            groups[str(group_name)] = [str(m) for m in members]
        return groups

    def _expand_module_compact_fields(
        self,
        module_name: str,
        module_cfg: Dict[str, Any],
        model_groups: Dict[str, List[str]],
    ) -> None:
        """Expand one module's compact fields into canonical file lists."""
        models = self._resolve_module_models(module_cfg, model_groups)
        if not models:
            return

        models = self._dedup_preserve_order(models)
        if not models:
            return

        if not module_cfg.get('model_labels'):
            module_cfg['model_labels'] = list(models)

        single_template = module_cfg.get('model_file_template')
        if isinstance(single_template, str):
            file_keys = self.MODULE_FILE_KEYS.get(module_name, ['model_file'])
            if len(file_keys) == 1:
                file_key = file_keys[0]
                if not module_cfg.get(file_key):
                    module_cfg[file_key] = [
                        self._render_model_template(single_template, model, module_name, file_key)
                        for model in models
                    ]
            else:
                raise ValueError(
                    f"{module_name} uses multiple model_file keys; please use "
                    f"'model_file_templates' instead of 'model_file_template'."
                )
        elif single_template is not None:
            raise ValueError(f"{module_name}.model_file_template must be a string.")

        template_map = module_cfg.get('model_file_templates')
        if isinstance(template_map, dict):
            for raw_key, tpl in template_map.items():
                if not isinstance(tpl, str):
                    raise ValueError(
                        f"{module_name}.model_file_templates['{raw_key}'] must be a string."
                    )
                file_key = str(raw_key)
                if not file_key.startswith('model_file'):
                    file_key = f"model_file_{file_key}"
                if module_cfg.get(file_key):
                    continue
                module_cfg[file_key] = [
                    self._render_model_template(tpl, model, module_name, file_key)
                    for model in models
                ]
        elif template_map is not None:
            raise ValueError(f"{module_name}.model_file_templates must be a mapping.")

    @staticmethod
    def _dedup_preserve_order(items: List[str]) -> List[str]:
        """Deduplicate while preserving the first occurrence order."""
        seen = set()
        deduped: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _resolve_module_models(
        self,
        module_cfg: Dict[str, Any],
        model_groups: Dict[str, List[str]],
    ) -> List[str]:
        """Resolve module models from models/model_group/model_exclude fields."""
        models: List[str] = []

        group_ref = module_cfg.get('model_group')
        group_names: List[str] = []
        if isinstance(group_ref, str):
            group_names = [group_ref]
        elif isinstance(group_ref, list):
            group_names = [str(g) for g in group_ref]
        elif group_ref is not None:
            raise ValueError("'model_group' must be a string or list of strings.")

        models_ref = module_cfg.get('models')
        if isinstance(models_ref, str):
            if models_ref in model_groups:
                group_names.append(models_ref)
            else:
                models.append(models_ref)
        elif isinstance(models_ref, list):
            models.extend([str(m) for m in models_ref])
        elif models_ref is not None:
            raise ValueError("'models' must be a string or list of strings.")

        for group_name in group_names:
            if group_name not in model_groups:
                raise ValueError(
                    f"Unknown model group '{group_name}' in module config."
                )
            models.extend(model_groups[group_name])

        model_exclude = module_cfg.get('model_exclude')
        if model_exclude is None:
            return models
        if isinstance(model_exclude, str):
            exclude_set = {model_exclude}
        elif isinstance(model_exclude, list):
            exclude_set = {str(m) for m in model_exclude}
        else:
            raise ValueError("'model_exclude' must be a string or list of strings.")

        return [m for m in models if m not in exclude_set]

    @staticmethod
    def _render_model_template(
        template: str,
        model: str,
        module_name: str,
        file_key: str,
    ) -> str:
        """Render one template item with current model name."""
        try:
            rendered = template.format(model=model)
        except Exception as exc:
            raise ValueError(
                f"Invalid template for {module_name}.{file_key}: {template!r}"
            ) from exc
        if '{' in rendered or '}' in rendered:
            raise ValueError(
                f"Unresolved template braces in {module_name}.{file_key}: {template!r}"
            )
        return rendered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_module(self, module_name: str) -> Optional[List[List[str]]]:
        """Validate all files and configurations for a specific evaluation module.

        Args:
            module_name: One of the supported module names.

        Returns:
            List of validated model file groups, or None.

        Raises:
            ValueError: If module name is unsupported or validation fails.
        """
        if module_name not in self.SUPPORTED_MODULES:
            raise ValueError(
                f"Module must be one of: {', '.join(self.SUPPORTED_MODULES)}"
            )

        logger.info("=" * 20)
        logger.info("Validating %s ...", module_name)
        logger.info("=" * 20)

        logger.info("Checking observation data ...")
        self._validate_observations(module_name)

        logger.info("Checking model files ...")
        file_groups = self._validate_model_files(module_name)
        if module_name == 'SItrans':
            self._validate_sitrans_options(self.variables[module_name])

        logger.info("%s validation completed.", module_name)
        return file_groups

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_observations(self, module_name: str) -> None:
        """Validate observation data files for the specified module.

        Args:
            module_name: Module whose observations are to be validated.

        Raises:
            ValueError: If required observation files are missing or time coverage
                is insufficient.
        """
        module_config = self.variables[module_name]

        # SItrans borrows SIconc obs; SICB needs both SIconc and SIdrift obs
        if module_name == 'SItrans':
            reference_mappings = {'SIconc': ''}
        elif module_name == 'SICB':
            reference_mappings = {'SIconc': '_sic', 'SIdrift': '_sidrift'}
        else:
            reference_mappings = {module_name: ''}

        for data_module, key_suffix in reference_mappings.items():
            # Recipe key for this hemisphere + optional suffix, e.g. "ref_sh_sic"
            ref_key = f"ref_{self.hemisphere}{key_suffix}"
            if ref_key not in module_config:
                continue

            # Scan the reference data directory for available files
            pattern = os.path.join(self.ref_data_path, data_module, f"*{self.hemisphere}*.nc")
            available_files = glob.glob(pattern)
            if not available_files:
                raise ValueError(f"No observation files found: {pattern}")

            available_obs = [os.path.basename(f) for f in available_files]

            for ref_file in module_config[ref_key]:
                if ref_file not in available_obs:
                    raise ValueError(
                        f"Reference file {ref_file} not found in {data_module} observations"
                    )

                # Open the file and verify it covers the requested year range
                ref_file_path = os.path.join(self.ref_data_path, data_module, ref_file)
                try:
                    with xr.open_dataset(ref_file_path) as ds:
                        time_var = ds['time']
                        start_year = pd.to_datetime(time_var[0].values).year
                        end_year = pd.to_datetime(time_var[-1].values).year
                        yr = module_config['year_range']
                        if yr[0] < start_year or yr[1] > end_year:
                            raise ValueError(
                                f"Year range {yr} not covered by {os.path.basename(ref_file_path)} "
                                f"({start_year}–{end_year})"
                            )
                except Exception as exc:
                    raise ValueError(
                        f"Error reading time coverage from {ref_file_path}: {exc}"
                    ) from exc

    def _validate_model_files(self, module_name: str) -> Optional[List[List[str]]]:
        """Validate model data files for the specified module.

        Args:
            module_name: Module whose model files are to be validated.

        Returns:
            List of validated file groups, or None.

        Raises:
            ValueError: If required model files are missing or invalid.
        """
        module_config = self.variables[module_name]
        data_root = module_config.get('model_data_path', self.model_data_path)
        if not isinstance(data_root, str):
            data_root = str(data_root)
        data_root = data_root.strip() or self.model_data_path
        if not os.path.exists(data_root):
            raise ValueError(f"Model data path not found for {module_name}: {data_root}")
        file_groups: List[List[str]] = []

        # Determine which variable types this module needs
        # SIdrift and SICB require separate u/v (and sic) file groups
        if module_name in ['SIconc', 'SIthick', 'SNdepth', 'SItrans']:
            file_types = ['']
        elif module_name == 'SIdrift':
            file_types = ['_u', '_v']
        elif module_name == 'SICB':
            file_types = ['_sic', '_u', '_v']
        elif module_name == 'SIMbudget':
            file_types = [
                '_sidmassdyn',
                '_sidmassevapsubl',
                '_sidmassgrowthbot',
                '_sidmassgrowthwat',
                '_sidmasslat',
                '_sidmassmeltbot',
                '_sidmassmelttop',
                '_sidmasssi',
            ]
        elif module_name == 'SNMbudget':
            file_types = [
                '_sndmassdyn',
                '_sndmassmelt',
                '_sndmasssi',
                '_sndmasssnf',
                '_sndmasssubl',
            ]
        else:
            file_types = ['']

        # Maximum allowed time step (days) — daily modules need ≤1 day intervals
        min_freq = self.TIME_FREQUENCY_REQUIREMENTS.get(module_name, 31)

        for file_type in file_types:
            file_key = f"model_file{file_type}"
            var_key = f"model_var{file_type}"

            if file_key not in module_config or not module_config[file_key]:
                if module_name in {'SIMbudget', 'SNMbudget'}:
                    logger.info("'%s' not specified — skipping.", file_key)
                else:
                    logger.warning("'%s' not specified — skipping.", file_key)
                continue

            for file_pattern in module_config[file_key]:
                # Expand glob patterns relative to the model data root
                full_pattern = os.path.join(data_root, file_pattern)
                matched_files = sorted(glob.glob(str(full_pattern)))

                if not matched_files:
                    raise ValueError(f"No files found matching: {full_pattern}")

                matched_paths = matched_files
                # Check that the matched files collectively span the requested period
                covering_files, time_var_name = self._validate_time_coverage_multiple(
                    matched_paths, module_config['year_range']
                )

                valid_files: List[str] = []
                for file_path in covering_files:
                    try:
                        with xr.open_dataset(file_path) as ds:
                            # Confirm the expected variable exists in the file
                            if module_config[var_key] not in ds.variables:
                                raise ValueError(
                                    f"Variable '{module_config[var_key]}' not found in {file_path}"
                                )
                            # Check temporal resolution does not exceed the module's requirement
                            time_var = ds[time_var_name]
                            if len(time_var) > 1:
                                days_diff = float(
                                    (time_var[1] - time_var[0]).values / np.timedelta64(1, 'D')
                                )
                                if days_diff > min_freq:
                                    raise ValueError(
                                        f"Data interval ({days_diff:.1f} d) exceeds "
                                        f"maximum ({min_freq} d) for {file_path}"
                                    )
                    except Exception as exc:
                        raise ValueError(
                            f"Error validating {file_path}: {exc}"
                        ) from exc

                    valid_files.append(str(file_path))

                if valid_files:
                    file_groups.append(valid_files)

        return file_groups if file_groups else None

    def _validate_sitrans_options(self, module_config: dict) -> None:
        """Validate optional SItrans algorithm parameters."""
        if not isinstance(module_config, dict):
            return

        if 'threshold' in module_config:
            try:
                threshold = float(module_config['threshold'])
            except Exception as exc:
                raise ValueError(f"SItrans threshold must be numeric, got {module_config['threshold']!r}") from exc
            if (not np.isfinite(threshold)) or threshold < 0.0 or threshold > 100.0:
                raise ValueError(f"SItrans threshold out of range [0, 100]: {threshold}")

        if 'smooth_window_days' in module_config:
            try:
                smooth_window_days = int(round(float(module_config['smooth_window_days'])))
            except Exception as exc:
                raise ValueError(
                    f"SItrans smooth_window_days must be an integer >= 1, got {module_config['smooth_window_days']!r}"
                ) from exc
            if smooth_window_days < 1:
                raise ValueError(f"SItrans smooth_window_days must be >= 1, got {smooth_window_days}")

        if 'persistence_days' in module_config:
            try:
                persistence_days = int(round(float(module_config['persistence_days'])))
            except Exception as exc:
                raise ValueError(
                    f"SItrans persistence_days must be an integer >= 2, got {module_config['persistence_days']!r}"
                ) from exc
            if persistence_days < 2:
                raise ValueError(f"SItrans persistence_days must be >= 2, got {persistence_days}")

        if 'trend_significance_p' in module_config:
            try:
                p_sig = float(module_config['trend_significance_p'])
            except Exception as exc:
                raise ValueError(
                    f"SItrans trend_significance_p must be numeric in (0, 1), got {module_config['trend_significance_p']!r}"
                ) from exc
            if (not np.isfinite(p_sig)) or p_sig <= 0.0 or p_sig >= 1.0:
                raise ValueError(f"SItrans trend_significance_p out of range (0, 1): {p_sig}")

    def _validate_time_coverage_multiple(
        self,
        file_paths: List[str],
        date_range: List[int],
    ) -> Tuple[List[str], str]:
        """Validate that a set of files collectively covers the required date range.

        Args:
            file_paths: List of candidate file paths.
            date_range: [start_year, end_year] or [start_date, end_date].

        Returns:
            Tuple of (covering_files, time_dimension_name).

        Raises:
            ValueError: If the files do not cover the target date range.
        """
        all_months: set = set()
        covering_files: List[str] = []
        time_var_name: Optional[str] = None

        if len(str(date_range[0])) == 4:
            start_date, end_date = f"{date_range[0]}-01", f"{date_range[1]}-12"
        else:
            start_date, end_date = date_range

        target_months = {
            f"{d.year}-{d.month:02d}"
            for d in pd.date_range(start=start_date, end=end_date, freq="MS")
        }

        for file_path in file_paths:
            if time_var_name is None:
                time_var_name = self._get_time_dimension_name(file_path)

            try:
                with xr.open_dataset(file_path, decode_cf=True) as ds:
                    time_index = ds[time_var_name].to_index()
                    file_months = {f"{t.year}-{t.month:02d}" for t in time_index}
            except Exception as exc:
                raise ValueError(
                    f"Error extracting months from {file_path}: {exc}"
                ) from exc

            if file_months & target_months:
                covering_files.append(file_path)
                all_months.update(file_months)

        missing = target_months - all_months
        if missing:
            raise ValueError(
                f"Files do not cover target date range {date_range}. "
                f"Missing months: {sorted(missing)}"
            )

        return covering_files, time_var_name

    @staticmethod
    def _get_time_dimension_name(file_path: str) -> str:
        """Identify the time dimension name in a NetCDF file.

        Args:
            file_path: Path to the NetCDF file.

        Returns:
            Name of the time dimension.

        Raises:
            ValueError: If no time dimension can be identified.
        """
        with xr.open_dataset(file_path, decode_cf=True) as ds:
            for dim in ds.dims:
                dim_lower = dim.lower()
                if 'bnds' in dim_lower or 'bounds' in dim_lower:
                    continue
                if dim_lower in ('time', 't', 'time_counter', 'date', 'datetime'):
                    return dim
                if dim in ds.variables:
                    units = ds[dim].attrs.get('units', '')
                    if 'since' in units:
                        return dim
            if 'time' in ds.dims:
                return 'time'
        raise ValueError(f"Cannot identify time dimension in {file_path}")
