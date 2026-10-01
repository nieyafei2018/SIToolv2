# -*- coding: utf-8 -*-
"""Pipeline module evaluation helpers."""

from scripts.pipeline import app as _app

# Reuse runtime namespace (imports/constants/helpers) initialized in app.py.
globals().update({k: v for k, v in _app.__dict__.items() if k not in globals()})

import copy
from typing import Sequence, Tuple

import yaml


_PLOT_OPTIONS_CACHE: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Function: `_default_plot_options`.
# Purpose: Handle default plot options workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _default_plot_options() -> Dict[str, Any]:
    """Return built-in plotting defaults for all modules.

    User overrides are loaded from:
      1) ``cases/plot_options.yml`` (global)
      2) ``cases/<case_name>/plot_options.yml`` (case-specific)
    """
    return {
        'global': {
            'line': {
                # Model-only style arrays used by plot helper functions.
                # Keep empty to preserve existing module default style cycles.
                'model_colors': [],
                'model_linestyles': [],
                # Runtime line-plot controls
                'figure_width_scale': 1.0,
                'figure_height_scale': 1.0,
                'min_figure_width': 3.0,
                'min_figure_height': 2.0,
                'model_linewidth': 0.9,
                # Optional: set to null to keep per-plot alpha defaults
                'model_alpha': None,
                'obs1_linewidth': 0.9,
                'obs2_linewidth': 0.9,
                # Optional: set to null to keep per-plot subplot defaults
                'subplot_wspace': None,
                'subplot_hspace': None,
                'zero_line_color': 'grey',
                'zero_line_style': '--',
                'zero_line_width': 0.5,
                'zero_line_alpha': 0.7,
                'grid_enabled': True,
                'grid_color': '0.8',
                'grid_style': '--',
                'grid_linewidth': 0.35,
                'grid_alpha': 0.7,
                'date_tick_minticks': 4,
                'date_tick_maxticks': 8,
                'month_tick_interval': 2,
                'month_tick_rotation': 30.0,
                # Optional: set to null to keep builtin legend fontsize
                'legend_fontsize': None,
                'legend_frameon': True,
                # Unified line-legend placement (top/bottom)
                'legend_position': 'top',
                # Fixed legend columns for all line plots in one run.
                # Use ``None`` for automatic column inference from figure width
                # and label lengths.
                'legend_ncol_fixed': None,
                # Maximum legend rows when labels are many.
                # Use ``None`` to enable automatic row-cap heuristics.
                'legend_max_rows': None,
                # Extra gap from subplot tight-bbox to top legend
                'legend_top_gap': 0.012,
                # Extra reserved top margin for top legends
                'legend_top_margin_extra': 0.0,
                # Increase bottom legend spacing from axes (in figure coords)
                'legend_y_gap_extra': 0.0,
                # Increase reserved bottom margin for bottom legends
                'legend_bottom_margin_extra': 0.0,
                'symmetric_ylim_pad_ratio': 0.08,
            },
            'typography': {
                'main_title': 20,
                'subplot_title': 20,
                'axis_label': 14,
                'tick': 12,
                'legend': 12,
                'colorbar_label': 16,
                'colorbar_tick': 16,
            },
            'style': {
                # Layout engine: manual | tight | constrained | compressed
                'layout_engine': 'manual',
            },
            'heatmap': {
                'cmap': 'RdBu_r',
                'ratio_vmin': 0.5,
                'ratio_vmax': 2.0,
                # Main heat_map figure height scale (width unchanged)
                'main_height_scale': 0.50,
                # Seasonal 2x2 heatmap layout (currently used by SIconc)
                'seasonal_subplot_wspace': 0.30,
                'seasonal_subplot_hspace': 0.18,
                'seasonal_colorbar_fraction': 0.018,
                'seasonal_colorbar_pad': 0.055,
                'seasonal_colorbar_aspect': 40,
                'seasonal_height_scale': 0.84,
            },
            'maps': {
                'diff_cmap': 'RdBu_r',
                'std_cmap': 'Purples',
                'trend_cmap': 'RdBu_r',
                # Shared map-panel spacing (smaller horizontal, larger vertical)
                'subplot_wspace': 0.01,
                'subplot_hspace': 0.22,
                # Shared map panel size coefficients (inches per panel)
                'panel_width_factor': 5.0,
                'panel_height_factor': 4.5,
                # Absolute map-size scaling for fixed-size map figures
                'figure_width_scale': 1.0,
                'figure_height_scale': 1.0,
                'min_figure_width': 3.0,
                'min_figure_height': 3.0,
                # Treat near-zero model values as missing in spatial maps
                'model_zero_mask_eps': 1e-8,
                # Polar basemap styling
                'coastline_linewidth': 0.5,
                'land_facecolor': 'grey',
                'gridline_linewidth': 0.2,
                'gridline_color': 'grey',
                'gridline_linestyle': '--',
                'ice_shelf_facecolor': 'lightgrey',
                # Contour overlays (e.g. 15% ice edge)
                'contour_linewidth': 1.0,
                # Shared bottom colorbar geometry
                'colorbar_fraction': 0.035,
                'colorbar_pad': 0.075,
                'colorbar_aspect': 42,
                # Optional map-colorbar text sizes (None => typography defaults)
                'colorbar_label_fontsize': None,
                'colorbar_tick_fontsize': None,
            },
            'output': {
                'dpi': 200,
                'bbox_inches': 'tight',
            },
        },
        'SIconc': {
            'maps': {
                'raw_cmap': 'YlGnBu',
                'diff_cmap': 'RdBu_r',
                'std_raw_cmap': 'Purples',
                'std_diff_cmap': 'RdBu_r',
                'trend_cmap': 'RdBu_r',
            },
        },
        'SIdrift': {
            'maps': {
                'mke_raw_cmap': 'viridis',
                'mke_diff_cmap': 'RdBu_r',
                'vectcorr_cmap': 'viridis',
                'std_raw_cmap': 'Purples',
                'std_diff_cmap': 'RdBu_r',
                'trend_cmap': 'RdBu_r',
            },
            'vector': {
                'quiver_skip': 3,
                'speed_vmin': 0.0,
                'speed_vmax': 0.3,
                'quiver_scale': 1.0,
                'quiver_width': 0.004,
                'quiver_min_speed': 0.0,
            },
        },
        'SIthick': {
            'maps': {
                'raw_cmap': 'viridis',
                'diff_cmap': 'RdBu_r',
                'std_raw_cmap': 'Purples',
                'std_diff_cmap': 'RdBu_r',
                'trend_cmap': 'RdBu_r',
            },
        },
        'SNdepth': {
            'maps': {
                'raw_cmap': 'Blues',
                'diff_cmap': 'RdBu_r',
                'std_raw_cmap': 'Purples',
                'std_diff_cmap': 'RdBu_r',
                'trend_cmap': 'RdBu_r',
            },
        },
        'SICB': {
            'maps': {
                'budget_cmap': 'RdBu_r',
                'vmin': -100.0,
                'vmax': 100.0,
                'cbtick_bin': 20.0,
            },
        },
        'SIMbudget': {
            'maps': {
                'budget_cmap': 'RdBu_r',
                'vmin': None,
                'vmax': None,
                'cbtick_bin': None,
            },
        },
        'SNMbudget': {
            'maps': {
                'budget_cmap': 'RdBu_r',
                'vmin': None,
                'vmax': None,
                'cbtick_bin': None,
            },
        },
        'SItrans': {
            'maps': {
                'climatology_cmap': 'viridis',
                'std_cmap': 'Purples',
                'trend_cmap': 'RdBu_r',
                'bias_cmap': 'RdBu_r',
                'std_diff_cmap': 'RdBu_r',
                'skill_heatmap_cmap': 'RdBu_r',
            },
        },
    }


# -----------------------------------------------------------------------------
# Function: `_deep_update`.
# Purpose: Handle deep update workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge mapping ``updates`` into ``base``."""
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# -----------------------------------------------------------------------------
# Function: `_load_plot_options`.
# Purpose: Handle load plot options workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _load_plot_options(case_name: str) -> Dict[str, Any]:
    """Load plotting options with global + case-level overrides."""
    case_key = str(case_name or '').strip() or '__default__'
    if case_key in _PLOT_OPTIONS_CACHE:
        return _PLOT_OPTIONS_CACHE[case_key]

    options = copy.deepcopy(_default_plot_options())
    root_file = Path('cases') / 'plot_options.yml'
    case_file = Path('cases') / str(case_name) / 'plot_options.yml'
    for cfg_file in (root_file, case_file):
        if not cfg_file.exists():
            continue
        try:
            with cfg_file.open('r', encoding='utf-8') as fh:
                payload = yaml.safe_load(fh) or {}
            if not isinstance(payload, dict):
                logger.warning(
                    "Ignoring plot options file with invalid root type: %s",
                    cfg_file,
                )
                continue
            _deep_update(options, payload)
        except Exception as exc:
            logger.warning("Failed to read plot options file %s (%s).", cfg_file, exc)

    _PLOT_OPTIONS_CACHE[case_key] = options
    return options


# -----------------------------------------------------------------------------
# Function: `_plot_options_get`.
# Purpose: Handle plot options get workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _plot_options_get(options: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    """Read nested plotting option value by path."""
    cur: Any = options
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# -----------------------------------------------------------------------------
# Function: `_plot_options_get_module`.
# Purpose: Handle plot options get module workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _plot_options_get_module(
    options: Dict[str, Any],
    module: str,
    path: Sequence[str],
    default: Any = None,
) -> Any:
    """Read module option and fall back to global branch with same path."""
    mod_val = _plot_options_get(options, [str(module)] + list(path), None)
    if mod_val is not None:
        return mod_val
    glob_val = _plot_options_get(options, ['global'] + list(path), None)
    if glob_val is not None:
        return glob_val
    return default


# -----------------------------------------------------------------------------
# Function: `_clamp_float`.
# Purpose: Handle clamp float workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _clamp_float(value: Any, lower: float, upper: float) -> float:
    """Return ``value`` clamped to ``[lower, upper]`` with float casting."""
    lo = float(lower)
    hi = float(upper)
    if hi < lo:
        lo, hi = hi, lo
    try:
        vv = float(value)
    except Exception:
        vv = lo
    return max(lo, min(hi, vv))


# -----------------------------------------------------------------------------
# Function: `_heatmap_main_figsize`.
# Purpose: Handle heatmap main figsize workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _heatmap_main_figsize(n_metrics: int, n_models: int, height_scale: float) -> Tuple[float, float]:
    """Adaptive main heatmap figure size shared across modules."""
    n_cols = max(1, int(n_metrics))
    model_count = max(1, int(n_models))
    n_rows = max(1, 1 + model_count)
    width = max(8.4, n_cols * 0.98 + 2.3)
    scale = max(0.70, float(height_scale))
    compact_boost = 1.16 if model_count <= 2 else (1.08 if model_count <= 4 else 1.0)
    height = max(4.8, (2.6 + n_rows * 0.76) * scale * compact_boost)
    return float(width), float(height)


# -----------------------------------------------------------------------------
# Function: `_heatmap_main_cbar_kwargs`.
# Purpose: Handle heatmap main cbar kwargs workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _heatmap_main_cbar_kwargs(n_metrics: int, n_models: int) -> Dict[str, float]:
    """Adaptive horizontal colorbar geometry for main heatmaps."""
    n_cols = max(1, int(n_metrics))
    model_count = max(1, int(n_models))
    n_rows = max(2, 1 + model_count)
    fraction = 0.029 + 0.004 * min(1.0, 6.0 / float(n_rows))
    if model_count <= 3:
        fraction += 0.002
    if model_count <= 2:
        pad = 0.140
    elif model_count <= 4:
        pad = 0.134
    elif model_count <= 8:
        pad = 0.127
    else:
        pad = 0.122
    if n_cols >= 8:
        pad += 0.018
    if n_cols >= 10:
        pad += 0.008
    aspect = 26.0 + 1.6 * float(n_cols)
    return {
        'orientation': 'horizontal',
        'fraction': _clamp_float(fraction, 0.028, 0.038),
        'pad': _clamp_float(pad, 0.122, 0.190),
        'aspect': _clamp_float(aspect, 26.0, 64.0),
    }


# -----------------------------------------------------------------------------
# Function: `_heatmap_main_adjust`.
# Purpose: Handle heatmap main adjust workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _heatmap_main_adjust(*, n_models: int, n_metrics: int = 0) -> Dict[str, float]:
    """Return consistent subplots-adjust parameters for main heatmaps."""
    model_count = max(1, int(n_models))
    metric_count = max(1, int(n_metrics))
    if model_count <= 2:
        bottom = 0.246
    elif model_count <= 4:
        bottom = 0.238
    elif model_count <= 8:
        bottom = 0.228
    else:
        bottom = 0.220
    if metric_count >= 8:
        bottom += 0.036
    elif metric_count >= 6:
        bottom += 0.024
    return {
        'left': 0.06,
        'right': 0.99,
        'top': 0.95,
        'bottom': _clamp_float(bottom, 0.220, 0.320),
    }


# -----------------------------------------------------------------------------
# Function: `_heatmap_seasonal_figsize`.
# Purpose: Handle heatmap seasonal figsize workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _heatmap_seasonal_figsize(n_metrics: int, n_models: int, height_scale: float) -> Tuple[float, float]:
    """Adaptive 2×2 seasonal heatmap figure size shared across modules."""
    n_cols = max(1, int(n_metrics))
    model_count = max(1, int(n_models))
    n_rows = max(1, 1 + model_count)
    panel_w = max(8.2, n_cols * 0.90 + 0.8)
    panel_h = max(4.0, 1.90 + n_rows * 0.62)
    scale = max(0.82, float(height_scale))
    compact_boost = (
        1.22 if model_count <= 2 else
        1.14 if model_count <= 4 else
        1.06 if model_count <= 6 else
        1.0
    )
    label_room = 1.20 + 0.16 * max(0, n_cols - 6)
    fig_w = panel_w * 2.0 + 1.2
    fig_h = max(8.8, (panel_h * 2.0 + label_room) * scale * compact_boost)
    return float(fig_w), float(fig_h)


# -----------------------------------------------------------------------------
# Function: `_heatmap_seasonal_spacing`.
# Purpose: Handle heatmap seasonal spacing workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _heatmap_seasonal_spacing(wspace: float, hspace: float, *, n_models: int = -1) -> Tuple[float, float]:
    """Clamp seasonal subplot spacing to avoid overlap and excessive gaps."""
    model_count = max(1, int(n_models)) if int(n_models) > 0 else 4
    if model_count <= 3:
        ws_floor = 0.24
    elif model_count <= 6:
        ws_floor = 0.22
    else:
        ws_floor = 0.20
    if model_count <= 2:
        hs_floor = 0.36
    elif model_count <= 4:
        hs_floor = 0.33
    elif model_count <= 8:
        hs_floor = 0.30
    else:
        hs_floor = 0.28
    return _clamp_float(wspace, ws_floor, 0.48), _clamp_float(hspace, hs_floor, 0.50)


# -----------------------------------------------------------------------------
# Function: `_heatmap_seasonal_cbar_kwargs`.
# Purpose: Handle heatmap seasonal cbar kwargs workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _heatmap_seasonal_cbar_kwargs(
    *,
    n_metrics: int,
    n_models: int,
    fraction: float,
    pad: float,
    aspect: float,
) -> Dict[str, float]:
    """Adaptive bottom colorbar geometry for seasonal heatmaps."""
    n_cols = max(1, int(n_metrics))
    model_count = max(1, int(n_models))
    n_rows = max(2, 1 + model_count)
    auto_fraction = 0.022 + 0.004 * min(1.0, 6.0 / float(n_rows))
    if model_count <= 3:
        auto_fraction += 0.002
    if model_count <= 2:
        auto_pad = 0.165
    elif model_count <= 4:
        auto_pad = 0.156
    elif model_count <= 6:
        auto_pad = 0.142
    elif model_count <= 10:
        auto_pad = 0.128
    else:
        auto_pad = 0.118
    auto_aspect = max(30.0, 34.0 + 3.8 * float(n_cols))
    return {
        'fraction': _clamp_float(0.45 * float(fraction) + 0.55 * auto_fraction, 0.019, 0.036),
        'pad': _clamp_float(0.35 * float(pad) + 0.65 * auto_pad, 0.112, 0.185),
        'aspect': _clamp_float(0.55 * float(aspect) + 0.45 * auto_aspect, 20.0, 84.0),
    }


# -----------------------------------------------------------------------------
# Function: `_heatmap_seasonal_adjust`.
# Purpose: Handle heatmap seasonal adjust workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _heatmap_seasonal_adjust(
    *,
    n_models: int,
    n_metrics: int = 0,
    wspace: float,
    hspace: float,
    right_margin: float,
) -> Dict[str, float]:
    """Return consistent subplots-adjust parameters for 2×2 seasonal heatmaps."""
    model_count = max(1, int(n_models))
    metric_count = max(1, int(n_metrics))
    ws, hs = _heatmap_seasonal_spacing(wspace, hspace, n_models=model_count)
    if model_count <= 2:
        base_bottom = 0.245
    elif model_count <= 4:
        base_bottom = 0.230
    elif model_count <= 6:
        base_bottom = 0.215
    elif model_count <= 10:
        base_bottom = 0.198
    else:
        base_bottom = 0.182
    if metric_count >= 8:
        base_bottom += 0.028
    return {
        'left': 0.06,
        'right': _clamp_float(float(right_margin) + 0.06, 0.90, 0.98),
        'top': 0.94,
        'bottom': _clamp_float(base_bottom, 0.17, 0.27),
        'wspace': ws,
        'hspace': hs,
    }

# -----------------------------------------------------------------------------
# Function: `_apply_plot_runtime_options`.
# Purpose: Handle apply plot runtime options workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _apply_plot_runtime_options(options: Dict[str, Any], module: str) -> None:
    """Apply runtime plotting knobs (line + spatial) to plot backend."""
    try:
        import scripts.plot_figs as pf  # local import to avoid circular side effects

        kw: Dict[str, Any] = {}
        map_wspace = _plot_options_get_module(options, module, ['maps', 'subplot_wspace'], None)
        map_hspace = _plot_options_get_module(options, module, ['maps', 'subplot_hspace'], None)
        panel_w = _plot_options_get_module(options, module, ['maps', 'panel_width_factor'], None)
        panel_h = _plot_options_get_module(options, module, ['maps', 'panel_height_factor'], None)
        zero_eps = _plot_options_get_module(options, module, ['maps', 'model_zero_mask_eps'], None)
        map_fig_w_scale = _plot_options_get_module(options, module, ['maps', 'figure_width_scale'], None)
        map_fig_h_scale = _plot_options_get_module(options, module, ['maps', 'figure_height_scale'], None)
        map_fig_min_w = _plot_options_get_module(options, module, ['maps', 'min_figure_width'], None)
        map_fig_min_h = _plot_options_get_module(options, module, ['maps', 'min_figure_height'], None)
        map_coast_lw = _plot_options_get_module(options, module, ['maps', 'coastline_linewidth'], None)
        map_land_color = _plot_options_get_module(options, module, ['maps', 'land_facecolor'], None)
        map_grid_lw = _plot_options_get_module(options, module, ['maps', 'gridline_linewidth'], None)
        map_grid_color = _plot_options_get_module(options, module, ['maps', 'gridline_color'], None)
        map_grid_style = _plot_options_get_module(options, module, ['maps', 'gridline_linestyle'], None)
        map_shelf_color = _plot_options_get_module(options, module, ['maps', 'ice_shelf_facecolor'], None)
        map_contour_lw = _plot_options_get_module(options, module, ['maps', 'contour_linewidth'], None)
        map_cbar_fraction = _plot_options_get_module(options, module, ['maps', 'colorbar_fraction'], None)
        map_cbar_pad = _plot_options_get_module(options, module, ['maps', 'colorbar_pad'], None)
        map_cbar_aspect = _plot_options_get_module(options, module, ['maps', 'colorbar_aspect'], None)
        map_cbar_label_fs = _plot_options_get_module(options, module, ['maps', 'colorbar_label_fontsize'], None)
        map_cbar_tick_fs = _plot_options_get_module(options, module, ['maps', 'colorbar_tick_fontsize'], None)

        line_fig_w_scale = _plot_options_get_module(options, module, ['line', 'figure_width_scale'], None)
        line_fig_h_scale = _plot_options_get_module(options, module, ['line', 'figure_height_scale'], None)
        line_fig_min_w = _plot_options_get_module(options, module, ['line', 'min_figure_width'], None)
        line_fig_min_h = _plot_options_get_module(options, module, ['line', 'min_figure_height'], None)
        line_model_lw = _plot_options_get_module(options, module, ['line', 'model_linewidth'], None)
        line_model_alpha = _plot_options_get_module(options, module, ['line', 'model_alpha'], None)
        line_obs1_lw = _plot_options_get_module(options, module, ['line', 'obs1_linewidth'], None)
        line_obs2_lw = _plot_options_get_module(options, module, ['line', 'obs2_linewidth'], None)
        line_subplot_wspace = _plot_options_get_module(options, module, ['line', 'subplot_wspace'], None)
        line_subplot_hspace = _plot_options_get_module(options, module, ['line', 'subplot_hspace'], None)
        line_zero_color = _plot_options_get_module(options, module, ['line', 'zero_line_color'], None)
        line_zero_style = _plot_options_get_module(options, module, ['line', 'zero_line_style'], None)
        line_zero_width = _plot_options_get_module(options, module, ['line', 'zero_line_width'], None)
        line_zero_alpha = _plot_options_get_module(options, module, ['line', 'zero_line_alpha'], None)
        line_grid_enabled = _plot_options_get_module(options, module, ['line', 'grid_enabled'], None)
        line_grid_color = _plot_options_get_module(options, module, ['line', 'grid_color'], None)
        line_grid_style = _plot_options_get_module(options, module, ['line', 'grid_style'], None)
        line_grid_width = _plot_options_get_module(options, module, ['line', 'grid_linewidth'], None)
        line_grid_alpha = _plot_options_get_module(options, module, ['line', 'grid_alpha'], None)
        line_dt_min = _plot_options_get_module(options, module, ['line', 'date_tick_minticks'], None)
        line_dt_max = _plot_options_get_module(options, module, ['line', 'date_tick_maxticks'], None)
        line_month_int = _plot_options_get_module(options, module, ['line', 'month_tick_interval'], None)
        line_month_rot = _plot_options_get_module(options, module, ['line', 'month_tick_rotation'], None)
        line_legend_fs = _plot_options_get_module(options, module, ['line', 'legend_fontsize'], None)
        line_legend_frame = _plot_options_get_module(options, module, ['line', 'legend_frameon'], None)
        line_legend_position = _plot_options_get_module(options, module, ['line', 'legend_position'], None)
        line_legend_ncol_fixed = _plot_options_get_module(options, module, ['line', 'legend_ncol_fixed'], None)
        line_legend_max_rows = _plot_options_get_module(options, module, ['line', 'legend_max_rows'], None)
        line_legend_top_gap = _plot_options_get_module(options, module, ['line', 'legend_top_gap'], None)
        line_legend_top_margin_extra = _plot_options_get_module(options, module, ['line', 'legend_top_margin_extra'], None)
        line_legend_gap_extra = _plot_options_get_module(options, module, ['line', 'legend_y_gap_extra'], None)
        line_legend_margin_extra = _plot_options_get_module(options, module, ['line', 'legend_bottom_margin_extra'], None)
        line_sym_pad = _plot_options_get_module(options, module, ['line', 'symmetric_ylim_pad_ratio'], None)
        fs_main_title = _plot_options_get_module(options, module, ['typography', 'main_title'], None)
        fs_subplot_title = _plot_options_get_module(options, module, ['typography', 'subplot_title'], None)
        fs_axis_label = _plot_options_get_module(options, module, ['typography', 'axis_label'], None)
        fs_tick = _plot_options_get_module(options, module, ['typography', 'tick'], None)
        fs_legend = _plot_options_get_module(options, module, ['typography', 'legend'], None)
        fs_cbar_label = _plot_options_get_module(options, module, ['typography', 'colorbar_label'], None)
        fs_cbar_tick = _plot_options_get_module(options, module, ['typography', 'colorbar_tick'], None)
        style_layout_engine = _plot_options_get_module(options, module, ['style', 'layout_engine'], None)
        save_dpi = _plot_options_get_module(options, module, ['output', 'dpi'], None)
        save_bbox = _plot_options_get_module(options, module, ['output', 'bbox_inches'], None)

        if map_wspace is not None:
            kw['map_subplot_wspace'] = float(map_wspace)
        if map_hspace is not None:
            kw['map_subplot_hspace'] = float(map_hspace)
        if panel_w is not None:
            kw['map_panel_width'] = float(panel_w)
        if panel_h is not None:
            kw['map_panel_height'] = float(panel_h)
        if zero_eps is not None:
            kw['model_zero_mask_eps'] = float(zero_eps)
        if map_fig_w_scale is not None:
            kw['map_fig_width_scale'] = float(map_fig_w_scale)
        if map_fig_h_scale is not None:
            kw['map_fig_height_scale'] = float(map_fig_h_scale)
        if map_fig_min_w is not None:
            kw['map_fig_min_width'] = float(map_fig_min_w)
        if map_fig_min_h is not None:
            kw['map_fig_min_height'] = float(map_fig_min_h)
        if map_coast_lw is not None:
            kw['map_coastline_linewidth'] = float(map_coast_lw)
        if map_land_color is not None:
            kw['map_land_facecolor'] = str(map_land_color)
        if map_grid_lw is not None:
            kw['map_gridline_linewidth'] = float(map_grid_lw)
        if map_grid_color is not None:
            kw['map_gridline_color'] = str(map_grid_color)
        if map_grid_style is not None:
            kw['map_gridline_linestyle'] = str(map_grid_style)
        if map_shelf_color is not None:
            kw['map_ice_shelf_facecolor'] = str(map_shelf_color)
        if map_contour_lw is not None:
            kw['map_contour_linewidth'] = float(map_contour_lw)
        if map_cbar_fraction is not None:
            kw['map_cbar_fraction'] = float(map_cbar_fraction)
        if map_cbar_pad is not None:
            kw['map_cbar_pad'] = float(map_cbar_pad)
        if map_cbar_aspect is not None:
            kw['map_cbar_aspect'] = float(map_cbar_aspect)
        if map_cbar_label_fs is not None:
            kw['map_cbar_label_fontsize'] = float(map_cbar_label_fs)
        if map_cbar_tick_fs is not None:
            kw['map_cbar_tick_fontsize'] = float(map_cbar_tick_fs)

        if line_fig_w_scale is not None:
            kw['line_fig_width_scale'] = float(line_fig_w_scale)
        if line_fig_h_scale is not None:
            kw['line_fig_height_scale'] = float(line_fig_h_scale)
        if line_fig_min_w is not None:
            kw['line_fig_min_width'] = float(line_fig_min_w)
        if line_fig_min_h is not None:
            kw['line_fig_min_height'] = float(line_fig_min_h)
        if line_model_lw is not None:
            kw['line_model_linewidth'] = float(line_model_lw)
        if line_model_alpha is not None:
            kw['line_model_alpha'] = float(line_model_alpha)
        if line_obs1_lw is not None:
            kw['line_obs1_linewidth'] = float(line_obs1_lw)
        if line_obs2_lw is not None:
            kw['line_obs2_linewidth'] = float(line_obs2_lw)
        if line_subplot_wspace is not None:
            kw['line_subplot_wspace'] = float(line_subplot_wspace)
        if line_subplot_hspace is not None:
            kw['line_subplot_hspace'] = float(line_subplot_hspace)
        if line_zero_color is not None:
            kw['line_zero_color'] = str(line_zero_color)
        if line_zero_style is not None:
            kw['line_zero_style'] = str(line_zero_style)
        if line_zero_width is not None:
            kw['line_zero_width'] = float(line_zero_width)
        if line_zero_alpha is not None:
            kw['line_zero_alpha'] = float(line_zero_alpha)
        if line_grid_enabled is not None:
            kw['line_grid_enabled'] = bool(line_grid_enabled)
        if line_grid_color is not None:
            kw['line_grid_color'] = str(line_grid_color)
        if line_grid_style is not None:
            kw['line_grid_style'] = str(line_grid_style)
        if line_grid_width is not None:
            kw['line_grid_linewidth'] = float(line_grid_width)
        if line_grid_alpha is not None:
            kw['line_grid_alpha'] = float(line_grid_alpha)
        if line_dt_min is not None:
            kw['line_date_minticks'] = int(line_dt_min)
        if line_dt_max is not None:
            kw['line_date_maxticks'] = int(line_dt_max)
        if line_month_int is not None:
            kw['line_month_tick_interval'] = int(line_month_int)
        if line_month_rot is not None:
            kw['line_month_tick_rotation'] = float(line_month_rot)
        if line_legend_fs is not None:
            kw['line_legend_fontsize'] = float(line_legend_fs)
        if line_legend_frame is not None:
            kw['line_legend_frameon'] = bool(line_legend_frame)
        if line_legend_position is not None:
            kw['line_legend_position'] = str(line_legend_position)
        if line_legend_ncol_fixed is not None:
            kw['line_legend_ncol_fixed'] = int(line_legend_ncol_fixed)
        if line_legend_max_rows is not None:
            kw['line_legend_max_rows'] = int(line_legend_max_rows)
        if line_legend_top_gap is not None:
            kw['line_legend_top_gap'] = float(line_legend_top_gap)
        if line_legend_top_margin_extra is not None:
            kw['line_legend_top_margin_extra'] = float(line_legend_top_margin_extra)
        if line_legend_gap_extra is not None:
            kw['line_legend_y_gap_extra'] = float(line_legend_gap_extra)
        if line_legend_margin_extra is not None:
            kw['line_legend_bottom_margin_extra'] = float(line_legend_margin_extra)
        if line_sym_pad is not None:
            kw['line_sym_ylim_pad_ratio'] = float(line_sym_pad)
        if fs_main_title is not None:
            kw['fs_main_title'] = float(fs_main_title)
        if fs_subplot_title is not None:
            kw['fs_subplot_title'] = float(fs_subplot_title)
        if fs_axis_label is not None:
            kw['fs_axis_label'] = float(fs_axis_label)
        if fs_tick is not None:
            kw['fs_tick'] = float(fs_tick)
        if fs_legend is not None:
            kw['fs_legend'] = float(fs_legend)
        if fs_cbar_label is not None:
            kw['fs_cbar_label'] = float(fs_cbar_label)
        if fs_cbar_tick is not None:
            kw['fs_cbar_tick'] = float(fs_cbar_tick)
        if style_layout_engine is not None:
            kw['style_layout_engine'] = str(style_layout_engine)

        if save_dpi is not None:
            kw['save_dpi'] = int(save_dpi)
        if save_bbox is not None:
            kw['save_bbox_inches'] = str(save_bbox)

        if kw:
            pf.configure_plot_runtime(**kw)
    except Exception as exc:
        logger.warning(
            "Failed to apply runtime plot options for module %s (%s).",
            module,
            exc,
        )

# -----------------------------------------------------------------------------
# Function: `_check_outputs_exist`.
# Purpose: Handle check outputs exist workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _check_outputs_exist(module: str, fig_dir: Path, hemisphere: str,
                         recalculate: bool = False) -> bool:
    """Always re-render module outputs so plotting stays decoupled from metric calculation."""
    logger.info(
        "Full-refresh plotting enabled for [%s/%s] — existing figures will be overwritten.",
        str(hemisphere).upper(),
        module,
    )
    return False


# -----------------------------------------------------------------------------
# Function: `_reuse_processed_inputs_for_recalculate`.
# Purpose: Handle reuse processed inputs for recalculate workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _reuse_processed_inputs_for_recalculate(case_name: str, recalculate: bool) -> bool:
    """Return whether `--recalculate` should reuse existing Processed files.

    Control via env var:
      - `SITOOL_RECALCULATE_REUSE_PROCESSED=1/true/yes/on` -> force reuse
      - `SITOOL_RECALCULATE_REUSE_PROCESSED=0/false/no/off` -> force rebuild
      - unset/other -> default False (keep legacy behavior)
    """
    if not bool(recalculate):
        return False

    raw = str(os.environ.get('SITOOL_RECALCULATE_REUSE_PROCESSED', '')).strip().lower()
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    return False

__all__ = [
    "_check_outputs_exist",
    "_reuse_processed_inputs_for_recalculate",
    "_load_plot_options",
    "_plot_options_get",
    "_plot_options_get_module",
    "_heatmap_main_figsize",
    "_heatmap_main_cbar_kwargs",
    "_heatmap_main_adjust",
    "_heatmap_seasonal_figsize",
    "_heatmap_seasonal_spacing",
    "_heatmap_seasonal_cbar_kwargs",
    "_heatmap_seasonal_adjust",
    "_apply_plot_runtime_options",
]
