# -*- coding: utf-8 -*-
"""Pipeline module evaluation helpers for SIMbudget/SNMbudget."""

from scripts.pipeline import app as _app

# Reuse runtime namespace (imports/constants/helpers) initialized in app.py.
globals().update({k: v for k, v in _app.__dict__.items() if k not in globals()})

import glob
import re

from scripts.pipeline.modules.common import (
    _apply_plot_runtime_options,
    _check_outputs_exist,
    _load_plot_options,
    _plot_options_get_module,
)


_MASS_BUDGET_SPECS: Dict[str, Dict[str, Any]] = {
    'SIMbudget': {
        'var_order': [
            'sidmassdyn',
            'sidmassevapsubl',
            'sidmassgrowthbot',
            'sidmassgrowthwat',
            'sidmasslat',
            'sidmassmeltbot',
            'sidmassmelttop',
            'sidmasssi',
        ],
        'dynamic_terms': ['sidmassdyn'],
        'source_terms': ['sidmassgrowthbot', 'sidmassgrowthwat', 'sidmasssi'],
        'sink_terms': ['sidmassmeltbot', 'sidmassmelttop', 'sidmasslat', 'sidmassevapsubl'],
        'map_unit': r'$\mathrm{kg\ m^{-2}/season}$',
    },
    'SNMbudget': {
        'var_order': [
            'sndmassdyn',
            'sndmassmelt',
            'sndmasssi',
            'sndmasssnf',
            'sndmasssubl',
        ],
        'dynamic_terms': ['sndmassdyn'],
        'source_terms': ['sndmasssnf'],
        'sink_terms': ['sndmassmelt', 'sndmasssi', 'sndmasssubl'],
        'map_unit': r'$\mathrm{kg\ m^{-2}/season}$',
    },
}


def _mass_budget_spec(module: str) -> Dict[str, Any]:
    spec = _MASS_BUDGET_SPECS.get(str(module))
    if spec is None:
        raise ValueError(f'Unsupported mass budget module: {module}')
    return spec


def _ordered_available_terms(module: str, term_names: List[str]) -> List[str]:
    """Return module terms in canonical order with unknown terms appended."""
    spec = _mass_budget_spec(module)
    req = [str(t) for t in (term_names or []) if str(t).strip()]
    if not req:
        return []
    ordered = [t for t in spec['var_order'] if t in req]
    extras = [t for t in req if t not in ordered]
    return ordered + extras


def _infer_terms_from_seas_file(module: str, seas_file: str) -> List[str]:
    """Infer available term names from one seasonal climatology file."""
    if not seas_file or (not Path(seas_file).exists()):
        return []
    try:
        with xr.open_dataset(seas_file) as ds:
            return _ordered_available_terms(module, list(ds.data_vars))
    except Exception:
        return []


def _season_months(hemisphere: str) -> Dict[str, List[int]]:
    if str(hemisphere).lower() == 'sh':
        return {
            'Spring': [9, 10, 11],
            'Summer': [12, 1, 2],
            'Autumn': [3, 4, 5],
            'Winter': [6, 7, 8],
        }
    return {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2],
    }


def _fmt_total(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else 'NaN'


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%" if np.isfinite(v) else 'NaN'


def _season_display_label(hemisphere: str, season: str) -> str:
    base = str(season)
    if str(hemisphere).lower() == 'sh':
        mapping = {
            'Spring': 'Spring (SON)',
            'Summer': 'Summer (DJF)',
            'Autumn': 'Autumn (MAM)',
            'Winter': 'Winter (JJA)',
        }
    else:
        mapping = {
            'Spring': 'Spring (MAM)',
            'Summer': 'Summer (JJA)',
            'Autumn': 'Autumn (SON)',
            'Winter': 'Winter (DJF)',
        }
    return mapping.get(base, base)


def _nice_symmetric_range(vmax_raw: float) -> float:
    vmax = float(np.nan_to_num(vmax_raw, nan=0.0))
    vmax = abs(vmax)
    if vmax <= 0:
        return 1.0
    exponent = 10 ** np.floor(np.log10(vmax))
    fraction = vmax / exponent
    if fraction <= 1:
        nice = 1
    elif fraction <= 2:
        nice = 2
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    return float(nice * exponent)


def _auto_budget_map_range(seas_files: List[str]) -> Tuple[float, float, float]:
    vals = []
    for path in seas_files:
        if not path or (not Path(path).exists()):
            continue
        try:
            with xr.open_dataset(path) as ds:
                for key in list(ds.data_vars):
                    if key not in ds.variables:
                        continue
                    arr = np.asarray(ds[key], dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        vals.append(arr)
        except Exception:
            continue

    if not vals:
        return -1.0, 1.0, 0.2

    all_vals = np.concatenate(vals)
    p95 = float(np.nanpercentile(np.abs(all_vals), 95))
    vmax = _nice_symmetric_range(p95)
    vmin = -vmax
    cbtick_bin = vmax / 5.0 if vmax > 0 else 0.2
    if cbtick_bin <= 0:
        cbtick_bin = 0.2
    return float(vmin), float(vmax), float(cbtick_bin)


def _find_lon_lat_names(ds: xr.Dataset) -> Tuple[str, str]:
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
    return lon_name, lat_name


def _estimate_area_rectilinear(lon_1d: np.ndarray, lat_1d: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_1d, dtype=float).ravel()
    lat = np.asarray(lat_1d, dtype=float).ravel()
    if lon.size < 2 or lat.size < 2:
        raise ValueError("Rectilinear lon/lat size is too small for area estimation.")

    lon_rad = np.unwrap(np.deg2rad(lon))
    lat_rad = np.deg2rad(lat)

    lon_b = np.empty(lon.size + 1, dtype=float)
    lat_b = np.empty(lat.size + 1, dtype=float)
    lon_b[1:-1] = 0.5 * (lon_rad[:-1] + lon_rad[1:])
    lat_b[1:-1] = 0.5 * (lat_rad[:-1] + lat_rad[1:])
    lon_b[0] = lon_rad[0] - (lon_b[1] - lon_rad[0])
    lon_b[-1] = lon_rad[-1] + (lon_rad[-1] - lon_b[-2])
    lat_b[0] = max(-np.pi / 2, lat_rad[0] - (lat_b[1] - lat_rad[0]))
    lat_b[-1] = min(np.pi / 2, lat_rad[-1] + (lat_rad[-1] - lat_b[-2]))

    earth_r = 6371000.0
    dlon = np.diff(lon_b)
    dphi = np.diff(np.sin(lat_b))
    area = (earth_r ** 2) * np.abs(dphi)[:, None] * np.abs(dlon)[None, :]
    return area


def _estimate_area_curvilinear(lon_2d: np.ndarray, lat_2d: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_2d, dtype=float)
    lat = np.asarray(lat_2d, dtype=float)
    if lon.shape != lat.shape or lon.ndim != 2:
        raise ValueError("Curvilinear lon/lat must be 2D arrays with the same shape.")

    lat_rad = np.deg2rad(lat)
    lon_rad = np.unwrap(np.deg2rad(lon), axis=1)
    dlat = np.gradient(lat_rad, axis=0)
    dlon = np.gradient(lon_rad, axis=1)
    earth_r = 6371000.0
    area = (earth_r ** 2) * np.abs(np.cos(lat_rad) * dlat * dlon)
    area[~np.isfinite(area)] = np.nan
    return area


def _estimate_area_from_dataset(ds_ref: xr.Dataset, var_name: str) -> np.ndarray:
    ref_da = ds_ref[var_name]
    spatial_dims = ref_da.dims[-2:]
    spatial_shape = tuple(int(ref_da.sizes[d]) for d in spatial_dims)

    lon_name, lat_name = _find_lon_lat_names(ds_ref)
    lon = np.asarray(ds_ref[lon_name], dtype=float)
    lat = np.asarray(ds_ref[lat_name], dtype=float)

    # First try explicit vertices-based bounds (shape: [..., 4]).
    for lon_b_name, lat_b_name in (
        ('vertices_longitude', 'vertices_latitude'),
        ('lon_vertices', 'lat_vertices'),
        ('longitude_vertices', 'latitude_vertices'),
    ):
        if lon_b_name in ds_ref.variables and lat_b_name in ds_ref.variables:
            lon_b = np.asarray(ds_ref[lon_b_name], dtype=float)
            lat_b = np.asarray(ds_ref[lat_b_name], dtype=float)
            if lon_b.ndim >= 3 and lat_b.ndim >= 3 and lon_b.shape[-1] == 4 and lat_b.shape[-1] == 4:
                lon_b = np.squeeze(lon_b)
                lat_b = np.squeeze(lat_b)
                if lon_b.shape[:2] == spatial_shape and lat_b.shape[:2] == spatial_shape:
                    area = utils._calculate_cell_areas(
                        lon_b, lat_b, int(spatial_shape[0]), int(spatial_shape[1])
                    )
                    if area.shape != spatial_shape and area.T.shape == spatial_shape:
                        area = area.T
                    area = np.asarray(area, dtype=float)
                    area[~np.isfinite(area)] = np.nan
                    if np.any(area > 0.0):
                        return area
                    logger.warning(
                        "Vertex-based area estimation produced no positive values for '%s'. "
                        "Falling back to lon/lat spacing estimate.",
                        var_name,
                    )

    if lon.ndim == 1 and lat.ndim == 1:
        area = _estimate_area_rectilinear(lon, lat)
    elif lon.ndim == 2 and lat.ndim == 2:
        area = _estimate_area_curvilinear(lon, lat)
    else:
        raise ValueError(
            f"Unsupported lon/lat dimensions for area estimation: lon={lon.shape}, lat={lat.shape}"
        )

    if area.shape != spatial_shape:
        if area.T.shape == spatial_shape:
            area = area.T
        else:
            raise ValueError(
                f"Estimated source area shape mismatch: estimated={area.shape}, expected={spatial_shape}."
            )
    return np.asarray(area, dtype=float)


def _read_area_array(area_file: str, target_shape: Tuple[int, int]) -> np.ndarray:
    with xr.open_dataset(area_file) as ds_area:
        area_var = None
        if 'areacello' in ds_area.variables:
            area_var = 'areacello'
        else:
            for name, da in ds_area.data_vars.items():
                attrs = {str(k).lower(): str(v).lower() for k, v in da.attrs.items()}
                if attrs.get('standard_name') == 'cell_area':
                    area_var = name
                    break
            if area_var is None:
                for name, da in ds_area.data_vars.items():
                    units = str(da.attrs.get('units', '')).lower().replace(' ', '')
                    if units in {'m2', 'm^2', 'm**2'}:
                        area_var = name
                        break
        if area_var is None:
            raise ValueError(f"Cannot find area variable in {area_file}.")

        area_arr = np.asarray(ds_area[area_var], dtype=float)
        while area_arr.ndim > 2:
            area_arr = np.asarray(area_arr[0], dtype=float)
        if area_arr.shape != target_shape:
            if area_arr.T.shape == target_shape:
                area_arr = area_arr.T
            else:
                raise ValueError(
                    f"Area shape mismatch in {area_file}: got {area_arr.shape}, expected {target_shape}."
                )
        area_arr[np.abs(area_arr) > 1e19] = np.nan
        area_arr[area_arr <= 0.0] = np.nan
        return area_arr


def _build_source_area_file(input_file: str, var_name: str,
                            area_file: Optional[str], output_file: str) -> str:
    with xr.open_dataset(input_file) as ds_ref:
        ref_da = ds_ref[var_name]
        spatial_dims = ref_da.dims[-2:]
        spatial_shape = tuple(int(ref_da.sizes[d]) for d in spatial_dims)

        area_source = 'estimated_from_lonlat'
        if area_file and Path(area_file).exists():
            try:
                area_arr = _read_area_array(area_file, spatial_shape)
                area_source = 'areacello'
            except Exception as exc:
                logger.warning(
                    "Failed to use areacello file %s (%s). Falling back to lon/lat area estimation.",
                    area_file, exc,
                )
                area_arr = _estimate_area_from_dataset(ds_ref, var_name)
        else:
            area_arr = _estimate_area_from_dataset(ds_ref, var_name)

        coords: Dict[str, Any] = {}
        for dim in spatial_dims:
            if dim in ds_ref.coords:
                coords[dim] = ds_ref.coords[dim]
            else:
                coords[dim] = np.arange(int(ds_ref.sizes[dim]), dtype=np.int32)

        ds_area_out = xr.Dataset({
            'src_area': xr.DataArray(
                np.asarray(area_arr, dtype=np.float32),
                dims=spatial_dims,
                coords=coords,
                attrs={
                    'long_name': 'source_grid_cell_area',
                    'units': 'm2',
                    'sitool_area_source': area_source,
                },
            )
        })
        utils.write_netcdf_compressed(ds_area_out, output_file)
        return area_source


def _build_target_area_file(grid_nc_file: str, output_file: str) -> None:
    with xr.open_dataset(grid_nc_file) as ds_grid:
        if 'cell_area' not in ds_grid.variables:
            raise ValueError(f"cell_area is missing in evaluation grid file: {grid_nc_file}")
        da = ds_grid['cell_area']
        coords: Dict[str, Any] = {}
        for dim in da.dims:
            if dim in ds_grid.coords:
                coords[dim] = ds_grid.coords[dim]
            else:
                coords[dim] = np.arange(int(ds_grid.sizes[dim]), dtype=np.int32)
        ds_tgt = xr.Dataset({
            'tgt_area': xr.DataArray(
                np.asarray(da, dtype=np.float32),
                dims=da.dims,
                coords=coords,
                attrs={'long_name': 'target_grid_cell_area', 'units': 'm2'},
            )
        })
        utils.write_netcdf_compressed(ds_tgt, output_file)


def _conservative_remap_with_area(preprocessor: PP.DataPreprocessor,
                                  input_file: str,
                                  var_name: str,
                                  grid_txt: str,
                                  output_file: str,
                                  area_file: Optional[str]) -> str:
    grid_nc = str(grid_txt).replace('.txt', '.nc')
    cdo = preprocessor._new_cdo_instance()
    area_source = 'conservative_no_area'
    with preprocessor._tempdir(prefix='sitool_budget_remap_') as tmp:
        src_area_file = os.path.join(tmp, 'src_area.nc')
        tgt_area_file = os.path.join(tmp, 'tgt_area.nc')
        extensive_file = os.path.join(tmp, 'extensive.nc')
        remap_extensive_file = os.path.join(tmp, 'extensive_i.nc')
        remap_flux_file = os.path.join(tmp, 'flux_i.nc')

        try:
            area_source = _build_source_area_file(
                input_file=input_file,
                var_name=var_name,
                area_file=area_file,
                output_file=src_area_file,
            )
            _build_target_area_file(grid_nc_file=grid_nc, output_file=tgt_area_file)

            cdo.mul(input=f'{os.path.abspath(input_file)} {os.path.abspath(src_area_file)}',
                    output=extensive_file)
            cdo.remapcon(grid_txt, input=extensive_file, output=remap_extensive_file)
            cdo.div(input=f'{os.path.abspath(remap_extensive_file)} {os.path.abspath(tgt_area_file)}',
                    output=remap_flux_file)
            # Keep only the target variable when possible.
            try:
                cdo.selvar(var_name, input=remap_flux_file, output=output_file)
            except Exception:
                cdo.copy(input=remap_flux_file, output=output_file)
            return area_source
        except Exception as exc:
            exc_text = str(exc).lower()
            logger.warning(
                "Area-aware conservative remap failed for %s (%s). "
                "Fallback to standard remap path.",
                input_file, exc,
            )
            # When remapcon itself is unstable for the source grid, retrying the
            # same operator adds noise and latency; go directly to bilinear.
            if ('remapcon' in exc_text) or ('aborted' in exc_text) or ('core dumped' in exc_text):
                utils.stable_interpolation(
                    grid_txt,
                    os.path.abspath(input_file),
                    os.path.abspath(output_file),
                    cdo,
                    method='bilinear',
                    allow_fallback=True,
                )
                return 'fallback_bilinear'

            utils.stable_interpolation(
                grid_txt,
                os.path.abspath(input_file),
                os.path.abspath(output_file),
                cdo,
                method='conservative',
                allow_fallback=True,
            )
            return 'fallback_conservative'


_SEGMENT_COPY_RE = re.compile(r"\s+\(\d+\)\.nc$")
_SEGMENT_DATE_RE = re.compile(r"_(\d{6,8}-\d{6,8})(?:\s+\(\d+\))?\.nc$")


def _segment_group_key(path: str) -> str:
    """Build a stable key for one time-split segment file."""
    name = os.path.basename(path)
    m = _SEGMENT_DATE_RE.search(name)
    if m:
        return m.group(1)
    return name


def _is_copy_segment_name(path: str) -> bool:
    """Return True for duplicate copy names like ``foo (1).nc``."""
    return _SEGMENT_COPY_RE.search(os.path.basename(path)) is not None


def _dedupe_segment_candidates(paths: List[str], *, context: str) -> List[str]:
    """Remove duplicated segment copies while preserving chronological keys.

    Some archives include duplicate files like ``..._195001-199912.nc`` and
    ``..._195001-199912 (1).nc`` with identical coverage. Keep one file per
    segment key and prefer the non-copy filename.
    """
    if len(paths) <= 1:
        return [str(p) for p in paths]

    key_order: List[str] = []
    best_by_key: Dict[str, str] = {}
    all_by_key: Dict[str, List[str]] = {}

    for raw_path in paths:
        path = str(raw_path)
        key = _segment_group_key(path)
        all_by_key.setdefault(key, []).append(path)
        if key not in best_by_key:
            key_order.append(key)
            best_by_key[key] = path
            continue

        current = best_by_key[key]
        current_is_copy = _is_copy_segment_name(current)
        candidate_is_copy = _is_copy_segment_name(path)
        if current_is_copy and not candidate_is_copy:
            best_by_key[key] = path
        elif current_is_copy == candidate_is_copy:
            if os.path.basename(path) < os.path.basename(current):
                best_by_key[key] = path

    deduped = [best_by_key[key] for key in key_order]
    for key in key_order:
        cands = all_by_key.get(key, [])
        if len(cands) <= 1:
            continue
        chosen = best_by_key[key]
        dropped = [os.path.basename(p) for p in cands if p != chosen]
        logger.warning(
            "%s duplicate segment files for %s; using %s, dropping %s",
            context, key, os.path.basename(chosen), dropped,
        )
    return deduped


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return bool(default)


def _allow_incomplete_budget_models(recipe: RR.RecipeReader, module: str,
                                    module_vars: Dict[str, Any]) -> bool:
    case_name = str(getattr(recipe, 'eval_ex', '') or '').strip().lower()
    default = (
        (case_name == 'snmbudget' and str(module) == 'SNMbudget')
        or (case_name == 'simbudget' and str(module) == 'SIMbudget')
    )
    for key in ('allow_incomplete_models', 'allow_missing_model_terms', 'skip_incomplete_models'):
        if key in module_vars:
            return _as_bool(module_vars.get(key), default=default)
    return default


def _collect_budget_model_inputs(recipe: RR.RecipeReader, module: str,
                                 module_vars: Dict[str, Any],
                                 allow_incomplete_models: bool = False) -> Tuple[List[str], List[Dict[str, List[str]]], List[Optional[str]], List[str]]:
    spec = _mass_budget_spec(module)
    data_root = str(module_vars.get('model_data_path', recipe.model_data_path)).strip()
    if not data_root:
        data_root = str(recipe.model_data_path)
    if not os.path.exists(data_root):
        raise ValueError(f"{module} model_data_path not found: {data_root}")
    year_range = module_vars['year_range']

    available_vars: List[str] = []
    pattern_map: Dict[str, List[str]] = {}
    model_count: Optional[int] = None

    for var_name in spec['var_order']:
        file_key = f'model_file_{var_name}'
        patterns = module_vars.get(file_key) or []
        if not patterns:
            continue
        if model_count is None:
            model_count = len(patterns)
        elif len(patterns) != model_count:
            raise ValueError(
                f"{module} expects the same model count across all budget terms. "
                f"'{file_key}' has {len(patterns)}, expected {model_count}."
            )
        available_vars.append(var_name)
        pattern_map[var_name] = [str(p) for p in patterns]

    if model_count is None or model_count <= 0:
        raise ValueError(f"No model_file_* entries found for {module}.")
    if 'dynamic_terms' in spec and len([v for v in spec['dynamic_terms'] if v in available_vars]) == 0:
        raise ValueError(f"{module} requires at least one dynamic term file (e.g., model_file_{spec['dynamic_terms'][0]}).")

    var_expected: Dict[str, str] = {}
    for var_name in available_vars:
        var_key = f'model_var_{var_name}'
        if not str(module_vars.get(var_key, '')).strip():
            raise ValueError(f"Missing required recipe key: {var_key}")
        var_expected[var_name] = str(module_vars[var_key]).strip()

    declared_labels = _get_recipe_model_labels(module, module_vars, model_count)
    kept_indices: List[int] = []
    kept_labels: List[str] = []
    model_term_groups: List[Dict[str, List[str]]] = []

    for midx in range(model_count):
        model_label = (
            declared_labels[midx]
            if midx < len(declared_labels)
            else f'{module}_dataset_{midx + 1}'
        )
        term_group: Dict[str, List[str]] = {}
        skip_reason: Optional[str] = None

        for var_name in available_vars:
            pattern = pattern_map[var_name][midx]
            full_pattern = os.path.join(data_root, pattern)
            matched = sorted(glob.glob(full_pattern))
            if not matched:
                if allow_incomplete_models:
                    skip_reason = f"missing files for {var_name}: {full_pattern}"
                    break
                raise ValueError(f"No files found matching: {full_pattern}")
            matched = _dedupe_segment_candidates(
                matched,
                context=f"{module} model[{midx + 1}] {var_name}",
            )

            try:
                covering_files, time_var_name = recipe._validate_time_coverage_multiple(matched, year_range)
                covering_files = _dedupe_segment_candidates(
                    covering_files,
                    context=f"{module} model[{midx + 1}] {var_name} (coverage)",
                )
                checked: List[str] = []
                expected_var = var_expected[var_name]
                for path in covering_files:
                    with xr.open_dataset(path) as ds:
                        if expected_var not in ds.variables:
                            raise ValueError(
                                f"Variable '{expected_var}' not found in {path}"
                            )
                        time_var = ds[time_var_name]
                        if len(time_var) > 1:
                            days_diff = float((time_var[1] - time_var[0]).values / np.timedelta64(1, 'D'))
                            if days_diff > 31:
                                raise ValueError(
                                    f"Data interval ({days_diff:.1f} d) exceeds monthly requirement (<=31 d) for {path}"
                                )
                    checked.append(str(path))
                term_group[var_name] = checked
            except Exception as exc:
                if allow_incomplete_models:
                    skip_reason = f"invalid files for {var_name}: {exc}"
                    break
                raise

        if skip_reason is not None:
            logger.warning(
                "%s skipping model '%s' (index=%d): %s",
                module, model_label, midx, skip_reason,
            )
            continue

        kept_indices.append(midx)
        kept_labels.append(model_label)
        model_term_groups.append(term_group)

    if not kept_indices:
        raise ValueError(f"{module} has no valid model after filtering incomplete inputs.")
    if allow_incomplete_models and len(kept_indices) < model_count:
        logger.warning(
            "%s kept %d/%d models after filtering incomplete-model inputs.",
            module, len(kept_indices), model_count,
        )

    # Resolve optional areacello files (one per model).
    area_patterns = module_vars.get('model_file_areacello') or []
    area_patterns = [str(p) for p in area_patterns]
    if area_patterns and len(area_patterns) != model_count:
        raise ValueError(
            f"{module} 'model_file_areacello' length ({len(area_patterns)}) "
            f"must equal number of models ({model_count})."
        )

    first_var = available_vars[0]
    first_patterns = pattern_map[first_var]
    area_files: List[Optional[str]] = []
    for kept_pos, midx in enumerate(kept_indices):
        area_path: Optional[str] = None
        if area_patterns:
            candidates = sorted(glob.glob(os.path.join(data_root, area_patterns[midx])))
            if candidates:
                area_path = candidates[0]
            else:
                logger.info(
                    "%s model[%d] areacello not found for pattern %s; "
                    "will estimate source cell area from lon/lat.",
                    module, kept_pos, area_patterns[midx],
                )
        else:
            model_dir = str(Path(first_patterns[midx]).parent)
            auto_pattern = os.path.join(data_root, model_dir, 'areacello_*.nc')
            candidates = sorted(glob.glob(auto_pattern))
            if candidates:
                area_path = candidates[0]
            else:
                logger.info(
                    "%s model[%d] no areacello_*.nc under %s; "
                    "will estimate source cell area from lon/lat.",
                    module, kept_pos, os.path.join(data_root, model_dir),
                )
        area_files.append(area_path)

    return kept_labels, model_term_groups, area_files, available_vars


def _prep_one_budget_variable(preprocessor: PP.DataPreprocessor,
                              file_group: List[str], var_name: str,
                              output_file: str, start_date: str, end_date: str,
                              grid_txt: str, area_file: Optional[str]) -> str:
    out_path = str(output_file)
    if preprocessor._is_nonempty_time_file(out_path):
        return out_path

    cdo = preprocessor._new_cdo_instance()
    with preprocessor._tempdir(prefix='sitool_budget_prep_') as tmp:
        segment_ready: List[str] = []
        for sidx, in_file in enumerate(file_group):
            seg_dir = os.path.join(tmp, f'seg_{sidx:04d}')
            os.makedirs(seg_dir, exist_ok=True)
            t_selraw = os.path.join(seg_dir, 'selraw.nc')
            t_mon = os.path.join(seg_dir, 'mon.nc')
            t_sel = os.path.join(seg_dir, 'sel.nc')
            t_var = os.path.join(seg_dir, 'var.nc')
            t_miss = os.path.join(seg_dir, 'miss.nc')

            # Slice first to avoid running monmean on the full historical span.
            preprocessor._safe_seldate(cdo, start_date, end_date, os.path.abspath(in_file), t_selraw)
            preprocessor._safe_monmean(cdo, t_selraw, t_mon)
            preprocessor._safe_seldate(cdo, start_date, end_date, t_mon, t_sel)
            try:
                cdo.selvar(var_name, input=t_sel, output=t_var)
            except Exception:
                # Keep compatibility with files where selvar fails unexpectedly.
                with xr.open_dataset(t_sel) as ds_sel:
                    if var_name not in ds_sel.variables:
                        raise
                    utils.write_netcdf_compressed(ds_sel[[var_name]], t_var)
            cdo.setmissval(-9999, input=t_var, output=t_miss)
            if preprocessor._is_nonempty_time_file(t_miss):
                segment_ready.append(t_miss)

        if not segment_ready:
            raise RuntimeError(f'No valid segments for variable {var_name}.')

        t_merge = os.path.join(tmp, 'merge.nc')
        t_range = os.path.join(tmp, 'range.nc')
        cdo.mergetime(input=' '.join(os.path.abspath(p) for p in segment_ready), output=t_merge)
        preprocessor._safe_seldate(cdo, start_date, end_date, t_merge, t_range)
        _conservative_remap_with_area(
            preprocessor=preprocessor,
            input_file=t_range,
            var_name=var_name,
            grid_txt=grid_txt,
            output_file=out_path,
            area_file=area_file,
        )
    return out_path


def _compute_budget_climatology(module: str,
                                hemisphere: str,
                                year_sta: int,
                                year_end: int,
                                model_label: str,
                                var_files: Dict[str, str],
                                output_dir: str) -> Tuple[str, str]:
    term_order = _ordered_available_terms(module, list(var_files.keys()))
    if not term_order:
        raise ValueError(f'No processed variables available for {module} / {model_label}.')

    arrays = []
    term_attrs: Dict[str, Dict[str, Any]] = {}
    for var_name in term_order:
        path = var_files[var_name]
        with xr.open_dataset(path) as ds:
            if var_name not in ds.variables:
                raise ValueError(f"Variable '{var_name}' not found in processed file: {path}")
            da = ds[var_name].load().rename(var_name)
            if 'time' in da.dims:
                try:
                    time_index = da.indexes.get('time')
                    if time_index is None:
                        time_index = pd.Index(np.asarray(da['time'].values))
                    dup_mask = np.asarray(time_index.duplicated(keep='first'), dtype=bool)
                    if np.any(dup_mask):
                        keep_idx = np.flatnonzero(~dup_mask)
                        logger.warning(
                            "%s/%s variable %s contains %d duplicate time stamps; keeping first occurrences.",
                            module, model_label, var_name, int(np.sum(dup_mask)),
                        )
                        da = da.isel(time=keep_idx)
                    da = da.sortby('time')
                except Exception as exc:
                    logger.warning(
                        "Time-index normalization failed for %s/%s variable %s (%s); using raw order.",
                        module, model_label, var_name, exc,
                    )
            term_attrs[var_name] = dict(getattr(ds[var_name], 'attrs', {}) or {})
            arrays.append(da)

    ds_all = xr.merge(arrays, join='inner', compat='override')
    if 'time' not in ds_all.coords:
        raise ValueError(f"{module} processed files do not contain time coordinate for {model_label}.")

    month_seconds = ds_all['time'].dt.days_in_month.astype(np.float64) * 86400.0
    month_numbers = np.arange(1, 13, dtype=np.int16)

    mon_clim: Dict[str, xr.DataArray] = {}
    for var_name in term_order:
        if var_name not in ds_all.data_vars:
            continue
        m = (ds_all[var_name] * month_seconds).groupby('time.month').mean('time', skipna=True)
        mon_clim[var_name] = m.reindex(month=month_numbers)
    term_order = [v for v in term_order if v in mon_clim]
    if not term_order:
        raise ValueError(f'No valid monthly climatology terms for {module} / {model_label}.')

    season_map = _season_months(hemisphere)
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

    def _to_season(da_month: xr.DataArray) -> xr.DataArray:
        parts = []
        for season in season_order:
            mon_list = season_map[season]
            parts.append(
                da_month.sel(month=mon_list).sum(dim='month', skipna=True, min_count=1)
            )
        out = xr.concat(parts, dim='season')
        out = out.assign_coords(season=season_order)
        return out

    ds_mon = xr.Dataset({name: mon_clim[name] for name in term_order})
    ds_seas = xr.Dataset({name: _to_season(mon_clim[name]) for name in term_order})

    for name in term_order:
        src_attr = term_attrs.get(name, {})
        long_name = str(src_attr.get('long_name') or name)
        ds_mon[name].attrs.update({'long_name': long_name, 'units': 'kg m-2 month-1'})
        ds_seas[name].attrs.update({'long_name': long_name, 'units': 'kg m-2 season-1'})

    common_attrs = {
        'module': module,
        'hemisphere': hemisphere,
        'model_label': model_label,
        'start_year': int(year_sta),
        'end_year': int(year_end),
        'sitool_mass_budget': 1,
        'budget_terms': ','.join(term_order),
    }
    ds_mon.attrs.update(common_attrs)
    ds_seas.attrs.update(common_attrs)

    safe_label = _sanitize_group_name(model_label)
    mon_path = os.path.join(output_dir, f'{module}_{hemisphere}_{safe_label}_{year_sta}-{year_end}_MonClim.nc')
    seas_path = os.path.join(output_dir, f'{module}_{hemisphere}_{safe_label}_{year_sta}-{year_end}_SeasClim.nc')
    utils.write_netcdf_compressed(ds_mon, mon_path)
    utils.write_netcdf_compressed(ds_seas, seas_path)
    return seas_path, mon_path


def _summarize_mass_budget_table(module: str, seas_file: str, grid_file: str,
                                 hemisphere: str, sector: str,
                                 term_names: List[str]) -> Dict[str, Dict[str, float]]:
    sec_mask = utils.region_index(grid_file=grid_file, hms=hemisphere, sector=sector)
    with xr.open_dataset(grid_file) as ds_grid:
        area = np.asarray(ds_grid['cell_area'], dtype=float)
    with xr.open_dataset(seas_file) as ds:
        terms = _ordered_available_terms(
            module=module,
            term_names=[t for t in term_names if t in ds.data_vars],
        ) if term_names else _ordered_available_terms(module, list(ds.data_vars))
        if not terms:
            terms = list(ds.data_vars)
        term_data = {name: np.asarray(ds[name], dtype=float) for name in terms}
        if 'season' in ds.coords:
            season_labels = [str(v) for v in ds['season'].values]
        else:
            season_labels = ['Spring', 'Summer', 'Autumn', 'Winter']

    out: Dict[str, Dict[str, float]] = {}
    for idx, season in enumerate(season_labels[:4]):
        season_vals: Dict[str, float] = {}
        for term_name, arr in term_data.items():
            if arr.ndim < 3:
                season_vals[term_name] = np.nan
                continue
            valid = sec_mask & np.isfinite(area) & np.isfinite(arr[idx])
            value_gt = (
                np.nansum(arr[idx][valid] * area[valid]) / 1e12
                if np.any(valid) else np.nan
            )
            season_vals[term_name] = float(value_gt) if np.isfinite(value_gt) else np.nan
        out[str(season)] = season_vals
    return out


def _build_mass_budget_region_table(module: str, hemisphere: str, grid_file: str,
                                    model_labels: List[str], model_seas_files: List[str],
                                    term_names: List[str]) -> dict:
    terms = _ordered_available_terms(module, term_names)
    if (not terms) and model_seas_files:
        terms = _infer_terms_from_seas_file(module, model_seas_files[0])
    if not terms:
        raise ValueError(f'Cannot resolve budget terms for {module} regional table.')
    headers = ['Model Name'] + terms
    units = [''] + ['Gt/season'] * len(terms)
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

    def _rows_for_sector(sector: str) -> Dict[str, List[List[str]]]:
        season_rows: Dict[str, List[List[str]]] = {s: [] for s in season_order}
        for idx, seas_file in enumerate(model_seas_files):
            label = model_labels[idx] if idx < len(model_labels) else f'model{idx + 1}'
            summary = _summarize_mass_budget_table(
                module=module,
                seas_file=seas_file,
                grid_file=grid_file,
                hemisphere=hemisphere,
                sector=sector,
                term_names=terms,
            )
            for season in season_order:
                s = summary.get(season, {})
                row = [label]
                for term_name in terms:
                    row.append(_fmt_total(s.get(term_name, np.nan)))
                season_rows[season].append(row)
        return season_rows

    regional_tables: Dict[str, Any] = {}
    all_region_keys = utils.get_hemisphere_sectors(hemisphere, include_all=True)
    for sector in all_region_keys:
        try:
            regional_tables[sector] = {
                'type': 'seasonal_table',
                'season_order': season_order,
                'headers': headers,
                'rows': [],
                'units': units,
                'seasons': _rows_for_sector(sector),
            }
        except Exception as exc:
            logger.warning(
                "Skipping %s regional table for sector '%s' (%s).",
                module, sector, exc,
            )
    return _build_region_table_payload(
        hemisphere=hemisphere,
        regional_tables=regional_tables,
        payload_type='region_seasonal_table',
    )


def _resolve_mass_budget_terms(module: str,
                               term_names: List[str],
                               model_seas_files: List[str]) -> List[str]:
    terms = _ordered_available_terms(module, term_names)
    if (not terms) and model_seas_files:
        terms = _infer_terms_from_seas_file(module, model_seas_files[0])
    return terms


def _extract_monthly_term_series(mon_file: str, term_name: str) -> Tuple[np.ndarray, Optional[str]]:
    with xr.open_dataset(mon_file) as ds:
        if term_name not in ds.data_vars:
            return np.array([], dtype=float), None
        da = ds[term_name]
        arr = np.asarray(da, dtype=float)
        if arr.ndim >= 3:
            vals = np.nanmean(arr, axis=tuple(range(1, arr.ndim)))
        else:
            vals = arr.astype(float)
        vals = np.asarray(vals, dtype=float).ravel()
        if vals.size == 12:
            vals = np.concatenate([vals, vals[:1]])
        elif vals.size > 13:
            vals = vals[:13]
        units = str(da.attrs.get('units', '')).strip() or None
        return vals, units


def _plot_mass_budget_timeseries(module: str,
                                 hemisphere: str,
                                 mon_files: List[str],
                                 model_labels: List[str],
                                 term_names: List[str],
                                 fig_name: str,
                                 line_style: Optional[List[str]] = None,
                                 color: Optional[List[str]] = None,
                                 group_member_files: Optional[Dict[str, List[str]]] = None) -> None:
    terms = _ordered_available_terms(module, term_names)
    if not terms:
        logger.warning("Skip %s time-series plot: no terms available.", module)
        return

    n_terms = len(terms)
    ncols = 2 if n_terms <= 6 else 3
    nrows = int(math.ceil(float(n_terms) / float(ncols)))
    fig_w = max(7.0, 5.0 * ncols)
    fig_h = max(4.0, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=pf._line_figsize(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()
    x_idx = np.arange(13, dtype=int)
    term_units: Dict[str, str] = {}
    group_rank: Dict[str, int] = {}
    for label in model_labels:
        if pf._is_group_label(label) and label not in group_rank:
            group_rank[label] = len(group_rank)

    for tidx, term_name in enumerate(terms):
        ax = axes_flat[tidx]
        ax.axhline(0, **pf._line_zero_style(alpha=0.5))
        for midx, mon_file in enumerate(mon_files):
            yvals, unit_name = _extract_monthly_term_series(mon_file, term_name)
            if yvals.size == 0:
                continue
            if unit_name and term_name not in term_units:
                term_units[term_name] = unit_name
            label = model_labels[midx] if midx < len(model_labels) else f'model{midx + 1}'
            if pf._is_group_label(label):
                style = pf._group_style(group_rank.get(label, 0))
            else:
                ls, cr = pf._get_style(midx, 0, line_style, color)
                style = {
                    'linestyle': ls,
                    'color': cr,
                    'lw': pf._line_model_width(),
                }
                alpha = pf._line_model_alpha(default=0.95)
                if alpha is not None:
                    style['alpha'] = alpha
            ax.plot(
                x_idx[:yvals.size],
                yvals,
                label=label,
                **style,
            )
            if (
                isinstance(group_member_files, dict)
                and pf._is_group_label(label)
                and label in group_member_files
            ):
                member_vals = []
                for member_file in group_member_files.get(label, []):
                    ys_member, _ = _extract_monthly_term_series(member_file, term_name)
                    if ys_member.size > 0:
                        member_vals.append(np.asarray(ys_member, dtype=float))
                if member_vals:
                    arr = np.asarray(member_vals, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 1:
                        std_vals = np.nanstd(arr, axis=0)
                        n_use = min(yvals.size, std_vals.size)
                        pf._plot_group_std_band(
                            ax,
                            x_idx[:n_use],
                            np.asarray(yvals[:n_use], dtype=float),
                            np.asarray(std_vals[:n_use], dtype=float),
                            style,
                        )

        ax.set_title(term_name, fontsize=14, weight='bold')
        pf._apply_month_ticks(ax, interval=2, rotation=30, use_datetime=False)
        pf._apply_light_grid(ax)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_xlim([-0.25, 12.25])
        ax.set_ylabel(term_units.get(term_name, 'kg m-2 month-1'), fontsize=10)

    for idx in range(n_terms, len(axes_flat)):
        axes_flat[idx].axis('off')

    handles, labels = [], []
    for ax in axes_flat[:max(1, n_terms)]:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            ncol=max(1, min(len(labels), 4)),
            fontsize=11,
            frameon=True,
        )
    fig.suptitle(f'{module} Monthly Climatology ({str(hemisphere).upper()})', fontsize=15, y=0.995)
    pf._finalize_line_layout(fig, legend_location='bottom' if handles else 'none', pad=0.9)
    pf._save_fig(fig_name)


def _align_field_to_grid(field_2d: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    field = np.asarray(field_2d, dtype=float)
    if lon.ndim == 1 and lat.ndim == 1:
        target_shape = (lat.size, lon.size)
    elif lon.ndim == 2 and lat.ndim == 2:
        target_shape = lon.shape
    else:
        return field

    if field.shape == target_shape:
        return field
    if field.T.shape == target_shape:
        return field.T
    return field


def _collect_mass_budget_spatial_payload(module: str,
                                         model_seas_files: List[str],
                                         model_labels: List[str],
                                         term_names: List[str]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    terms = _ordered_available_terms(module, term_names)
    if not terms and model_seas_files:
        terms = _infer_terms_from_seas_file(module, model_seas_files[0])

    payloads: List[Dict[str, Any]] = []
    season_order_ref: List[str] = []
    for midx, seas_file in enumerate(model_seas_files):
        with xr.open_dataset(seas_file) as ds:
            if not terms:
                terms = _ordered_available_terms(module, list(ds.data_vars)) or list(ds.data_vars)
            lon_name, lat_name = _find_lon_lat_names(ds)
            lon = np.asarray(ds[lon_name], dtype=float)
            lat = np.asarray(ds[lat_name], dtype=float)
            seasons = [str(v) for v in ds.coords.get('season', xr.DataArray(
                ['Spring', 'Summer', 'Autumn', 'Winter'], dims=('season',)
            )).values]
            if not season_order_ref:
                preferred = ['Spring', 'Summer', 'Autumn', 'Winter']
                season_order_ref = [s for s in preferred if s in seasons]
                season_order_ref.extend([s for s in seasons if s not in season_order_ref])

            term_arrays: Dict[str, np.ndarray] = {}
            for term_name in terms:
                if term_name in ds.data_vars:
                    term_arrays[term_name] = np.asarray(ds[term_name], dtype=float)

            payloads.append({
                'label': model_labels[midx] if midx < len(model_labels) else f'model{midx + 1}',
                'seasons': seasons,
                'lon': lon,
                'lat': lat,
                'terms': term_arrays,
            })

    return terms, season_order_ref, payloads


def _plot_mass_budget_season_panels(module: str,
                                    hemisphere: str,
                                    model_seas_files: List[str],
                                    model_labels: List[str],
                                    term_names: List[str],
                                    fig_dir: Path,
                                    *,
                                    vmin: float,
                                    vmax: float,
                                    cbtick_bin: float,
                                    cmap: str,
                                    unit: str) -> None:
    terms, seasons, payloads = _collect_mass_budget_spatial_payload(
        module=module,
        model_seas_files=model_seas_files,
        model_labels=model_labels,
        term_names=term_names,
    )
    if not payloads or not terms or not seasons:
        logger.warning("Skip %s spatial maps: insufficient payload.", module)
        return

    n_rows = len(payloads)
    n_cols = len(terms)
    proj = ccrs.Stereographic(central_latitude=-90 if str(hemisphere).lower() == 'sh' else 90, central_longitude=0)

    tick_vals = np.linspace(vmin, vmax, 11)
    if cbtick_bin is not None and np.isfinite(cbtick_bin) and float(cbtick_bin) > 0:
        cand = np.arange(vmin, vmax + 0.5 * float(cbtick_bin), float(cbtick_bin))
        if 2 <= cand.size <= 17:
            tick_vals = cand

    for season_name in seasons:
        fig_w = max(6.0, 3.4 * n_cols + 1.0)
        fig_h = max(4.0, 3.2 * n_rows + 1.2)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=pf._map_abs_figsize(fig_w, fig_h),
            subplot_kw={'projection': proj},
            squeeze=False,
        )
        last_im = None
        season_vals: List[np.ndarray] = []

        for ridx, pack in enumerate(payloads):
            label = str(pack.get('label', f'model{ridx + 1}'))
            seasons_this = [str(s) for s in pack.get('seasons', [])]
            sidx = seasons_this.index(season_name) if season_name in seasons_this else None
            lon = np.asarray(pack.get('lon'), dtype=float)
            lat = np.asarray(pack.get('lat'), dtype=float)
            term_arrays = pack.get('terms', {})
            for cidx, term_name in enumerate(terms):
                ax = axes[ridx, cidx]
                pf.polar_map(str(hemisphere).lower(), ax)
                plotted = False
                arr3 = term_arrays.get(term_name)
                if (arr3 is not None) and (sidx is not None) and (arr3.ndim >= 3) and (0 <= sidx < arr3.shape[0]):
                    field = _align_field_to_grid(arr3[sidx, :, :], lon, lat)
                    if np.any(np.isfinite(field)):
                        finite = np.asarray(field, dtype=float)
                        finite = finite[np.isfinite(finite)]
                        if finite.size > 0:
                            season_vals.append(finite)
                        last_im = ax.pcolormesh(
                            lon,
                            lat,
                            field,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=plt.get_cmap(cmap),
                            transform=ccrs.PlateCarree(),
                        )
                        plotted = True
                if not plotted:
                    nan_field = np.full((lat.shape[0], lon.shape[0]), np.nan) if (lon.ndim == 1 and lat.ndim == 1) else np.full(lon.shape, np.nan)
                    last_im = ax.pcolormesh(
                        lon,
                        lat,
                        _align_field_to_grid(nan_field, lon, lat),
                        vmin=vmin,
                        vmax=vmax,
                        cmap=plt.get_cmap(cmap),
                        transform=ccrs.PlateCarree(),
                    )

                if ridx == 0:
                    ax.set_title(term_name, fontsize=12, weight='bold')
                if cidx == 0:
                    ax.text(
                        -0.10,
                        0.5,
                        label,
                        transform=ax.transAxes,
                        rotation=90,
                        va='center',
                        ha='center',
                        fontsize=11,
                        weight='bold',
                    )

        if last_im is not None:
            all_data = np.concatenate(season_vals) if season_vals else np.array([], dtype=float)
            cb = fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                orientation='horizontal',
                fraction=0.035,
                pad=0.06,
                ticks=tick_vals,
                extend=pf._resolve_colorbar_extend(
                    last_im, data=all_data, vmin=vmin, vmax=vmax, extend='auto'
                ),
            )
            cb.ax.tick_params(labelsize=10)
            cb.ax.set_title(unit, fontsize=11, weight='bold')

        fig.suptitle(
            f'{module} Spatial Budget Terms — {_season_display_label(hemisphere, season_name)}',
            fontsize=15,
            y=0.992,
        )
        pf._finalize_map_layout(fig, has_bottom_cbar=(last_im is not None), pad=0.8)
        fig_name = fig_dir / f'{module}_map_{_sanitize_group_name(season_name)}.png'
        pf._save_fig(str(fig_name))


def _cleanup_module_pngs(fig_dir: Path) -> None:
    if not fig_dir.exists():
        return
    for png_file in fig_dir.glob('*.png'):
        try:
            png_file.unlink()
        except Exception:
            continue


def _budget_payload_to_paths(payload: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(payload, dict):
        return None, None
    seas = payload.get('seas_file')
    mon = payload.get('mon_file')
    seas_path = str(seas) if seas else None
    mon_path = str(mon) if mon else None
    if seas_path and not Path(seas_path).exists():
        seas_path = None
    if mon_path and not Path(mon_path).exists():
        mon_path = None
    return seas_path, mon_path


def _eval_mass_budget_module(case_name: str, module: str, recipe: RR.RecipeReader,
                             data_dir: str, output_dir: str,
                             recalculate: bool = False,
                             jobs: int = 1) -> Optional[dict]:
    _log_module_header(module)
    module_vars = recipe.variables[module]
    hemisphere = recipe.hemisphere
    year_sta, year_end = _get_year_range(module_vars)
    plot_opts = _load_plot_options(case_name)
    _apply_plot_runtime_options(plot_opts, module)
    line_colors = _plot_options_get_module(plot_opts, module, ['line', 'model_colors'], None)
    if not isinstance(line_colors, (list, tuple)) or len(line_colors) == 0:
        line_colors = None
    line_styles = _plot_options_get_module(plot_opts, module, ['line', 'model_linestyles'], None)
    if not isinstance(line_styles, (list, tuple)) or len(line_styles) == 0:
        line_styles = None

    budget_cmap = _plot_options_get_module(plot_opts, module, ['maps', 'budget_cmap'], 'RdBu_r')
    budget_vmin_raw = _plot_options_get_module(plot_opts, module, ['maps', 'vmin'], None)
    budget_vmax_raw = _plot_options_get_module(plot_opts, module, ['maps', 'vmax'], None)
    budget_tick_raw = _plot_options_get_module(plot_opts, module, ['maps', 'cbtick_bin'], None)

    fig_dir = Path(output_dir) / module
    if _check_outputs_exist(module, fig_dir, hemisphere, recalculate=recalculate):
        logger.info("%s evaluation skipped — all outputs exist.", module)
        return None

    cache_file = _get_metrics_cache_file(case_name, output_dir, hemisphere, module)
    preprocessor = PP.DataPreprocessor(case_name, module, hemisphere=hemisphere)
    grid_file = preprocessor.gen_eval_grid()
    grid_txt = str(grid_file).replace('.nc', '.txt')
    start_date = f"{year_sta}-01-01"
    end_date = f"{year_end}-12-31"
    requested_jobs = max(1, int(jobs))
    max_parallel_env_name = f'SITOOL_{module.upper()}_MAX_JOBS'
    try:
        max_parallel_cap = max(1, int(os.environ.get(max_parallel_env_name, str(requested_jobs))))
    except Exception:
        max_parallel_cap = requested_jobs
    effective_jobs = max(1, min(requested_jobs, max_parallel_cap))
    if effective_jobs != requested_jobs:
        logger.info(
            "Capping %s internal jobs via %s: requested=%d -> effective=%d",
            module, max_parallel_env_name, requested_jobs, effective_jobs,
        )
    logger.info(
        "%s internal parallel workers: requested=%d, effective=%d",
        module, requested_jobs, effective_jobs,
    )
    py_thread_parallel = str(os.environ.get('SITOOL_MASSBUDGET_PY_THREADS', '0')).strip().lower() in {
        '1', 'true', 'yes', 'on',
    }
    thread_pool_jobs = effective_jobs if py_thread_parallel else 1
    if effective_jobs > 1 and thread_pool_jobs == 1:
        logger.info(
            "%s Python-thread fan-out is disabled by default for NetCDF stability; "
            "using CDO-thread parallelism instead (set SITOOL_MASSBUDGET_PY_THREADS=1 to enable).",
            module,
        )

    model_labels: List[str] = []
    model_mon_files: List[str] = []
    model_seas_files: List[str] = []
    term_names: List[str] = []
    cache_loaded = False

    if not recalculate:
        cached = _load_module_cache(cache_file, module, hemisphere)
        if cached is not None and cached.get('payload_kind') == module:
            try:
                records = cached.get('records', {})
                model_records = cached.get('model_records', [])
                model_labels = list(cached.get('model_labels', []))
                term_names = _ordered_available_terms(module, list(cached.get('term_names', [])))
                for rec in model_records:
                    seas_file, mon_file = _budget_payload_to_paths(records.get(rec))
                    if seas_file and mon_file:
                        model_seas_files.append(seas_file)
                        model_mon_files.append(mon_file)
                if model_seas_files and len(model_seas_files) == len(model_mon_files):
                    if not term_names:
                        term_names = _infer_terms_from_seas_file(module, model_seas_files[0])
                    cache_loaded = True
                    logger.info("Loaded %s metrics from cache: %s", module, cache_file)
            except Exception as exc:
                logger.warning("Cache payload for %s is incomplete (%s). Recalculating.", module, exc)
                cache_loaded = False

    if not cache_loaded:
        allow_incomplete_models = _allow_incomplete_budget_models(recipe, module, module_vars)
        if allow_incomplete_models:
            logger.warning(
                "%s runs in tolerant mode: incomplete model terms are skipped with warnings.",
                module,
            )
        else:
            recipe.validate_module(module)
        model_labels, model_term_groups, area_files, available_vars = _collect_budget_model_inputs(
            recipe=recipe,
            module=module,
            module_vars=module_vars,
            allow_incomplete_models=allow_incomplete_models,
        )
        term_names = _ordered_available_terms(module, available_vars)

        processed_by_model: List[Dict[str, str]] = [dict() for _ in range(len(model_labels))]
        prep_tasks: List[Tuple[int, str, str, List[str], Optional[str], str]] = []
        for var_name in available_vars:
            for midx, model_label in enumerate(model_labels):
                safe_label = _sanitize_group_name(model_label)
                out_name = (
                    f"{var_name}_{safe_label}_{hemisphere}_"
                    f"{year_sta}01-{year_end}12_monthly_i.nc"
                )
                out_path = str(Path(data_dir) / out_name)
                src_group = model_term_groups[midx][var_name]
                area_file = area_files[midx] if midx < len(area_files) else None
                prep_tasks.append((midx, var_name, model_label, src_group, area_file, out_path))

        def _run_one_prep_task(task: Tuple[int, str, str, List[str], Optional[str], str]) -> Tuple[int, str, str]:
            midx, var_name, model_label, src_group, area_file, out_path = task
            logger.info(
                "[%s/%s] Preprocessing %s for %s (area=%s)",
                hemisphere.upper(), module, var_name, model_label,
                area_file if area_file else 'estimated',
            )
            processed = _prep_one_budget_variable(
                preprocessor=preprocessor,
                file_group=src_group,
                var_name=var_name,
                output_file=out_path,
                start_date=start_date,
                end_date=end_date,
                grid_txt=grid_txt,
                area_file=area_file,
            )
            return midx, var_name, processed

        if thread_pool_jobs > 1 and len(prep_tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            n_workers = min(thread_pool_jobs, len(prep_tasks))
            with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix=f'{module.lower()}-prep') as pool:
                fut_map = {
                    pool.submit(rte.run_tracked_task, _run_one_prep_task, task): task
                    for task in prep_tasks
                }
                for fut in as_completed(fut_map):
                    task = fut_map[fut]
                    task_var = str(task[1])
                    task_model = str(task[2])
                    try:
                        midx, var_name, processed = fut.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"{module} preprocessing failed for variable={task_var}, model={task_model}: {exc}"
                        ) from exc
                    processed_by_model[midx][var_name] = processed
        else:
            for task in prep_tasks:
                midx, var_name, processed = _run_one_prep_task(task)
                processed_by_model[midx][var_name] = processed

        model_seas_files = []
        model_mon_files = []
        clim_tasks: List[Tuple[int, str, Dict[str, str]]] = [
            (midx, model_label, processed_by_model[midx])
            for midx, model_label in enumerate(model_labels)
        ]

        def _run_one_climatology_task(task: Tuple[int, str, Dict[str, str]]) -> Tuple[int, str, str]:
            midx, model_label, var_files = task
            seas_file, mon_file = _compute_budget_climatology(
                module=module,
                hemisphere=hemisphere,
                year_sta=year_sta,
                year_end=year_end,
                model_label=model_label,
                var_files=var_files,
                output_dir=str(data_dir),
            )
            return midx, seas_file, mon_file

        if thread_pool_jobs > 1 and len(clim_tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            n_workers = min(thread_pool_jobs, len(clim_tasks))
            seas_map: Dict[int, str] = {}
            mon_map: Dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix=f'{module.lower()}-clim') as pool:
                fut_map = {
                    pool.submit(rte.run_tracked_task, _run_one_climatology_task, task): task
                    for task in clim_tasks
                }
                for fut in as_completed(fut_map):
                    task = fut_map[fut]
                    task_model = str(task[1])
                    try:
                        midx, seas_file, mon_file = fut.result()
                    except Exception as exc:
                        raise RuntimeError(f"{module} climatology failed for model={task_model}: {exc}") from exc
                    seas_map[midx] = seas_file
                    mon_map[midx] = mon_file
            for midx in range(len(model_labels)):
                model_seas_files.append(seas_map[midx])
                model_mon_files.append(mon_map[midx])
        else:
            for task in clim_tasks:
                _midx, seas_file, mon_file = _run_one_climatology_task(task)
                model_seas_files.append(seas_file)
                model_mon_files.append(mon_file)

        model_records = [f'model_{i}_budget' for i in range(len(model_seas_files))]
        records = {}
        for rec_name, seas_file, mon_file in zip(model_records, model_seas_files, model_mon_files):
            records[rec_name] = {
                'seas_file': str(Path(seas_file).resolve()),
                'mon_file': str(Path(mon_file).resolve()),
            }

        used_entities: set = set()
        entity_groups: Dict[str, str] = {}
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
                    'term_names': term_names,
                },
                records=records,
                entity_groups=entity_groups,
                grid_file=grid_file,
            )
            logger.info("Saved %s metrics to cache: %s", module, cache_file)
        except Exception as exc:
            logger.warning("Failed to save %s cache (%s).", module, exc)

    # Re-read cache payload for plotting robustness.
    cached_for_plot = _load_module_cache(cache_file, module, hemisphere)
    if cached_for_plot is not None and cached_for_plot.get('payload_kind') == module:
        try:
            records = cached_for_plot.get('records', {})
            model_records = cached_for_plot.get('model_records', [])
            seas_loaded: List[str] = []
            mon_loaded: List[str] = []
            for rec in model_records:
                seas_file, mon_file = _budget_payload_to_paths(records.get(rec))
                if seas_file and mon_file:
                    seas_loaded.append(seas_file)
                    mon_loaded.append(mon_file)
            if seas_loaded and len(seas_loaded) == len(mon_loaded):
                model_seas_files = seas_loaded
                model_mon_files = mon_loaded
                model_labels = list(cached_for_plot.get('model_labels', model_labels)) or model_labels
                cached_terms = _ordered_available_terms(module, list(cached_for_plot.get('term_names', [])))
                if cached_terms:
                    term_names = cached_terms
                logger.info("Using cache-backed %s payload for plotting: %s", module, cache_file)
        except Exception as exc:
            logger.warning(
                "Failed to reload %s cache for plotting (%s). Using in-memory payload.",
                module, exc,
            )

    # Metric-level (post-diagnostic) group means built from per-model budget files.
    base_model_seas_files = list(model_seas_files)
    base_model_mon_files = list(model_mon_files)
    base_model_labels_for_group = list(model_labels)
    model_labels_for_maps = list(base_model_labels_for_group)
    model_labels_for_ts = list(base_model_labels_for_group)
    group_specs = _resolve_group_mean_specs(
        module=module,
        module_vars=module_vars,
        common_config=recipe.common_config,
        model_labels=base_model_labels_for_group,
    )
    group_labels: List[str] = []
    group_member_map_mon: Dict[str, List[str]] = {}
    if group_specs and model_seas_files:
        group_dir = Path(output_dir) / module / '_groupmean'
        seas_with_groups, labels_with_groups, group_labels = _build_group_mean_file_payloads(
            file_paths=model_seas_files,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
            output_dir=group_dir / 'seas',
            file_tag=f'{module.lower()}_seas_groupmean',
        )
        model_seas_files = list(seas_with_groups)
        model_labels_for_maps = list(labels_with_groups)
        if group_labels:
            logger.info(
                "Enabled file-level group means for %s [%s] seasonal maps/tables: %s",
                module, hemisphere.upper(), ', '.join(group_labels),
            )
    if group_specs and model_mon_files:
        group_member_map_mon = _build_group_member_file_map(
            file_paths=base_model_mon_files,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
        )
        mon_with_groups, labels_mon_with_groups, _mon_groups = _build_group_mean_file_payloads(
            file_paths=model_mon_files,
            model_labels=base_model_labels_for_group,
            group_specs=group_specs,
            output_dir=Path(output_dir) / module / '_groupmean' / 'monthly',
            file_tag=f'{module.lower()}_monthly_groupmean',
        )
        model_mon_files = list(mon_with_groups)
        model_labels_for_ts = list(labels_mon_with_groups)

    if not model_labels_for_maps:
        model_labels_for_maps = list(base_model_labels_for_group)
    if not model_labels_for_ts:
        model_labels_for_ts = list(base_model_labels_for_group)
    model_labels = list(model_labels_for_maps)

    if not model_seas_files or not model_mon_files:
        logger.error("No processed model climatology files available for %s.", module)
        return None
    term_names = _resolve_mass_budget_terms(module, term_names, model_seas_files)
    if not term_names:
        logger.error("No valid term names resolved for %s.", module)
        return None

    # Plotting
    fig_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_module_pngs(fig_dir)
    spec = _mass_budget_spec(module)
    if budget_vmin_raw is None or budget_vmax_raw is None or budget_tick_raw is None:
        auto_vmin, auto_vmax, auto_tick = _auto_budget_map_range(model_seas_files)
    else:
        auto_vmin, auto_vmax, auto_tick = float(budget_vmin_raw), float(budget_vmax_raw), float(budget_tick_raw)

    _plot_mass_budget_season_panels(
        module=module,
        hemisphere=hemisphere,
        model_seas_files=model_seas_files,
        model_labels=model_labels,
        term_names=term_names,
        fig_dir=fig_dir,
        vmin=auto_vmin,
        vmax=auto_vmax,
        cbtick_bin=auto_tick,
        cmap=budget_cmap,
        unit=spec['map_unit'],
    )

    _plot_mass_budget_timeseries(
        module=module,
        hemisphere=hemisphere,
        mon_files=base_model_mon_files,
        model_labels=base_model_labels_for_group,
        term_names=term_names,
        fig_name=str(fig_dir / f'{module}_ts.png'),
        line_style=list(line_styles) if line_styles is not None else None,
        color=list(line_colors) if line_colors is not None else None,
    )
    if group_labels:
        _plot_mass_budget_timeseries(
            module=module,
            hemisphere=hemisphere,
            mon_files=model_mon_files[:len(group_labels)],
            model_labels=model_labels_for_ts[:len(group_labels)],
            term_names=term_names,
            fig_name=str(fig_dir / f'{module}_ts_groupmean.png'),
            line_style=list(line_styles) if line_styles is not None else None,
            color=list(line_colors) if line_colors is not None else None,
            group_member_files=group_member_map_mon,
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
    return _build_mass_budget_region_table(
        module=module,
        hemisphere=hemisphere,
        grid_file=grid_file,
        model_labels=model_labels,
        model_seas_files=model_seas_files,
        term_names=term_names,
    )


def eval_simbudget(case_name: str, recipe: RR.RecipeReader,
                   data_dir: str, output_dir: str,
                   recalculate: bool = False,
                   jobs: int = 1) -> Optional[dict]:
    """Evaluate SIMbudget (sea-ice mass budget) in model-only mode."""
    return _eval_mass_budget_module(
        case_name=case_name,
        module='SIMbudget',
        recipe=recipe,
        data_dir=data_dir,
        output_dir=output_dir,
        recalculate=recalculate,
        jobs=jobs,
    )


def eval_snmbudget(case_name: str, recipe: RR.RecipeReader,
                   data_dir: str, output_dir: str,
                   recalculate: bool = False,
                   jobs: int = 1) -> Optional[dict]:
    """Evaluate SNMbudget (snow mass budget) in model-only mode."""
    return _eval_mass_budget_module(
        case_name=case_name,
        module='SNMbudget',
        recipe=recipe,
        data_dir=data_dir,
        output_dir=output_dir,
        recalculate=recalculate,
        jobs=jobs,
    )


__all__ = ["eval_simbudget", "eval_snmbudget"]
