# -*- coding: utf-8 -*-
"""
Plotting utilities for SIToolv2 evaluation figures.

Provides polar map helpers and figure-generation functions for sea-ice
concentration (SIC), drift (SID), thickness (SIT), snow depth (SND),
concentration budget (SICB), and transition-date (SItrans) diagnostics.

All plot functions follow a common pattern:
  1. Assemble data from metric dicts returned by SeaIceMetrics classes.
  2. Create a matplotlib figure with Cartopy polar-stereographic axes.
  3. Plot observation(s) first, then model(s), using consistent styling.
  4. Save the figure to *fig_name* and close it to free memory.

Created on 2022/8/24
"""

import matplotlib
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import datetime
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scripts.config import LINE_STYLES, COLORS, DPI, MONTH_TICKS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typography constants (publication quality)
# ---------------------------------------------------------------------------
_FS_MAIN_TITLE = 20     # Main / super title
_FS_SUBPLOT_TITLE = 20  # Per-panel subplot titles
_FS_AXIS_LABEL = 14     # X / Y axis labels
_FS_TICK = 12           # Tick label sizes
_FS_LEGEND = 12         # Legend text
_FS_CBAR_LABEL = 16     # Colorbar label
_FS_CBAR_TICK = 16      # Colorbar tick labels

# ---------------------------------------------------------------------------
# Variable unit strings used in colorbar / axis labels
# ---------------------------------------------------------------------------
_UNIT_LABELS = {
    'SIconc':  'Sea Ice Concentration (%)',
    'SIthick': 'Sea Ice Thickness (m)',
    'SNdepth': 'Snow Depth (m)',
    'SIdrift': 'Sea Ice Drift Speed (m/s)',
    'SItrans': 'Transition Date (MM-DD)',
}

# ---------------------------------------------------------------------------
# Runtime-configurable plotting controls (overridable via plot_options.yml)
# ---------------------------------------------------------------------------
_MAP_SUBPLOT_WSPACE = 0.02
_MAP_SUBPLOT_HSPACE = 0.12
_MODEL_ZERO_MASK_EPS = 1e-8
_MAP_PANEL_WIDTH = 4.0
_MAP_PANEL_HEIGHT = 5.4
_MAP_FIG_WIDTH_SCALE = 1.0
_MAP_FIG_HEIGHT_SCALE = 1.0
_MAP_FIG_MIN_WIDTH = 3.0
_MAP_FIG_MIN_HEIGHT = 3.0

_LINE_FIG_WIDTH_SCALE = 1.0
_LINE_FIG_HEIGHT_SCALE = 1.0
_LINE_FIG_MIN_WIDTH = 3.0
_LINE_FIG_MIN_HEIGHT = 2.0

_LINE_MODEL_LINEWIDTH = 0.9
_LINE_MODEL_ALPHA = None
_LINE_OBS1_LINEWIDTH = 0.9
_LINE_OBS2_LINEWIDTH = 0.9
_LINE_SUBPLOT_WSPACE = None
_LINE_SUBPLOT_HSPACE = None
_LINE_ZERO_COLOR = 'grey'
_LINE_ZERO_STYLE = '--'
_LINE_ZERO_WIDTH = 0.5
_LINE_ZERO_ALPHA = 0.7
_LINE_GRID_ENABLED = True
_LINE_GRID_STYLE = '--'
_LINE_GRID_COLOR = '0.8'
_LINE_GRID_WIDTH = 0.35
_LINE_GRID_ALPHA = 0.7
_LINE_DATE_MINTICKS = 4
_LINE_DATE_MAXTICKS = 8
_LINE_MONTH_TICK_INTERVAL = 2
_LINE_MONTH_TICK_ROTATION = 30.0
_LINE_LEGEND_FONTSIZE = None
_LINE_LEGEND_FRAMEON = True
_LINE_LEGEND_Y_GAP_EXTRA = 0.0
_LINE_LEGEND_BOTTOM_MARGIN_EXTRA = 0.0
_LINE_LEGEND_POSITION = 'top'
# ``None`` means adaptive (auto) layout instead of fixed value.
_LINE_LEGEND_NCOL_FIXED = None
_LINE_LEGEND_MAX_ROWS = None
_LINE_LEGEND_TOP_GAP = 0.012
_LINE_LEGEND_TOP_MARGIN_EXTRA = 0.0
_LINE_SYM_Y_PAD_RATIO = 0.08

_MAP_COASTLINE_LINEWIDTH = 0.5
_MAP_LAND_FACECOLOR = 'grey'
_MAP_GRIDLINE_LINEWIDTH = 0.2
_MAP_GRIDLINE_COLOR = 'grey'
_MAP_GRIDLINE_STYLE = '--'
_MAP_ICE_SHELF_FACECOLOR = 'lightgrey'
_MAP_CONTOUR_LINEWIDTH = 1.0
_MAP_CBAR_FRACTION = 0.035
_MAP_CBAR_PAD = 0.055
_MAP_CBAR_ASPECT = 42
_MAP_CBAR_LABEL_FONTSIZE = None
_MAP_CBAR_TICK_FONTSIZE = None

_SAVE_DPI = DPI
_SAVE_BBOX_INCHES = 'tight'

_GROUP_LINE_COLORS = ['#7b3294', '#1b9e77', '#d95f02', '#2b8cbe']
_GROUP_LINE_STYLE = '-.'
_GROUP_LINE_WIDTH = 1.35

# Optional adaptive layout engine (runtime-configurable)
_STYLE_LAYOUT_ENGINE = 'manual'

# -----------------------------------------------------------------------------
# Developer notes
# -----------------------------------------------------------------------------
# This module is intentionally organized by diagnostic family rather than by
# plotting primitive.  In other words, each public function corresponds to a
# scientific product that appears in the SITool workflow (time series, anomaly
# curves, climatology maps, heatmaps, etc.), not merely to a low-level Matplotlib
# chart type.
#
# A few conventions are used throughout the file:
#   1. Observation products are always plotted before model products so the
#      caller can rely on stable ordering in legends and panel layouts.
#   2. Most functions accept metric dictionaries produced by SeaIceMetrics.  The
#      plotting code therefore spends a lot of effort translating those metric
#      dictionaries into simple 1-D, 2-D, or 3-D NumPy arrays suitable for
#      plotting.
#   3. Figures are saved and closed immediately unless a function explicitly
#      returns the figure object.  This is important because SITool can generate
#      many large polar map panels in one run.
#
# When modifying this file, prefer to preserve the current data contracts
# instead of introducing new plotting-only data structures.  The surrounding
# workflow in main.py and SeaIceMetrics.py depends on these conventions.


# ---------------------------------------------------------------------------
# Private helper: resolve line style & color for dataset index *idx*
# ---------------------------------------------------------------------------

def _get_style(idx: int, n_obs: int,
               line_style=None, color=None):
    """Return (linestyle, colour) for dataset *idx*."""
    model_idx = max(0, int(idx) - int(n_obs))
    if line_style:
        ls = line_style[model_idx % len(line_style)]
    else:
        ls = LINE_STYLES[model_idx % len(LINE_STYLES)]
    if color:
        cr = color[model_idx % len(color)]
    else:
        cr = COLORS[model_idx % len(COLORS)]
    return ls, cr

# -----------------------------------------------------------------------------
# Function: `_obs_style`.
# Purpose: Handle obs style workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _obs_style(obs_idx: int):
    """Return canonical observation line style by observation index."""
    if obs_idx <= 0:
        return {'color': 'k', 'linestyle': '-', 'linewidth': _LINE_OBS1_LINEWIDTH}
    return {'color': '0.5', 'linestyle': '-', 'linewidth': _LINE_OBS2_LINEWIDTH}

# -----------------------------------------------------------------------------
# Function: `_line_model_width`.
# Purpose: Handle line model width workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_model_width(default: Optional[float] = None) -> float:
    """Return runtime model linewidth (or fallback default)."""
    base = _LINE_MODEL_LINEWIDTH if default is None else float(default)
    return max(0.05, float(base))

# -----------------------------------------------------------------------------
# Function: `_line_model_alpha`.
# Purpose: Handle line model alpha workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_model_alpha(default: Optional[float] = None) -> Optional[float]:
    """Return runtime model alpha (or fallback default)."""
    if _LINE_MODEL_ALPHA is None:
        return default
    try:
        return max(0.0, min(1.0, float(_LINE_MODEL_ALPHA)))
    except Exception:
        return default

# -----------------------------------------------------------------------------
# Function: `_line_subplot_wspace`.
# Purpose: Handle line subplot wspace workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_subplot_wspace(default: float) -> float:
    """Return line-chart subplot horizontal spacing."""
    base = default if _LINE_SUBPLOT_WSPACE is None else _LINE_SUBPLOT_WSPACE
    try:
        return max(0.0, float(base))
    except Exception:
        return max(0.0, float(default))

# -----------------------------------------------------------------------------
# Function: `_line_subplot_hspace`.
# Purpose: Handle line subplot hspace workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_subplot_hspace(default: float) -> float:
    """Return line-chart subplot vertical spacing."""
    base = default if _LINE_SUBPLOT_HSPACE is None else _LINE_SUBPLOT_HSPACE
    try:
        return max(0.0, float(base))
    except Exception:
        return max(0.0, float(default))

# -----------------------------------------------------------------------------
# Function: `_line_bottom_legend_y`.
# Purpose: Handle line bottom legend y workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_bottom_legend_y(default: float = 0.0) -> float:
    """Return runtime-adjusted legend y-anchor for bottom legends."""
    base = float(default)
    try:
        return base - max(0.0, float(_LINE_LEGEND_Y_GAP_EXTRA))
    except Exception:
        return base

# -----------------------------------------------------------------------------
# Function: `_line_bottom_margin`.
# Purpose: Handle line bottom margin workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_bottom_margin(default: float = 0.18) -> float:
    """Return runtime-adjusted bottom margin for bottom legends."""
    base = max(0.0, float(default))
    try:
        return base + max(0.0, float(_LINE_LEGEND_BOTTOM_MARGIN_EXTRA))
    except Exception:
        return base

# -----------------------------------------------------------------------------
# Function: `_line_top_legend_ncol`.
# Purpose: Handle line top legend ncol workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_top_legend_ncol(default: Optional[int] = None) -> Optional[int]:
    """Return runtime fixed legend-column target (or ``None`` for auto)."""
    base = _LINE_LEGEND_NCOL_FIXED if _LINE_LEGEND_NCOL_FIXED is not None else default
    if base is None:
        return None
    try:
        return max(1, int(base))
    except Exception:
        if default is None:
            return None
        try:
            return max(1, int(default))
        except Exception:
            return None

# -----------------------------------------------------------------------------
# Function: `_line_top_legend_max_rows`.
# Purpose: Handle line top legend max rows workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_top_legend_max_rows(default: Optional[int] = None) -> Optional[int]:
    """Return runtime legend-row cap (or ``None`` for auto heuristic)."""
    base = _LINE_LEGEND_MAX_ROWS if _LINE_LEGEND_MAX_ROWS is not None else default
    if base is None:
        return None
    try:
        return max(1, int(base))
    except Exception:
        if default is None:
            return None
        try:
            return max(1, int(default))
        except Exception:
            return None

# -----------------------------------------------------------------------------
# Function: `_legend_axes_list`.
# Purpose: Handle legend axes list workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _legend_axes_list(fig, axes=None):
    """Return a flat list of axes that belong to *fig*."""
    if axes is None:
        axes_iter = list(getattr(fig, 'axes', []) or [])
    elif isinstance(axes, (list, tuple, np.ndarray)):
        axes_iter = [ax for ax in np.ravel(np.asarray(axes, dtype=object)).tolist()]
    else:
        axes_iter = [axes]

    out = []
    for ax in axes_iter:
        if ax is None:
            continue
        if getattr(ax, 'figure', None) is not fig:
            continue
        out.append(ax)
    return out

# -----------------------------------------------------------------------------
# Function: `_legend_axes_span`.
# Purpose: Handle legend axes span workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _legend_axes_span(axes_list):
    """Return axes span (x0, x1, y0, y1) in figure coordinates."""
    if not axes_list:
        return 0.08, 0.98, 0.10, 0.90
    boxes = [ax.get_position() for ax in axes_list]
    x0 = float(min(bb.x0 for bb in boxes))
    x1 = float(max(bb.x1 for bb in boxes))
    y0 = float(min(bb.y0 for bb in boxes))
    y1 = float(max(bb.y1 for bb in boxes))
    return x0, x1, y0, y1

# -----------------------------------------------------------------------------
# Function: `_legend_axes_tight_top`.
# Purpose: Handle legend axes tight top workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _legend_axes_tight_top(fig, axes_list, renderer):
    """Return max tight-bbox top for axes in figure coordinates."""
    tops = []
    for ax in axes_list:
        try:
            bb = ax.get_tightbbox(renderer)
        except Exception:
            bb = None
        if bb is None:
            continue
        bbf = bb.transformed(fig.transFigure.inverted())
        yt = float(bbf.y1)
        if np.isfinite(yt):
            tops.append(yt)
    if tops:
        return float(max(tops))
    return _legend_axes_span(axes_list)[3]

# -----------------------------------------------------------------------------
# Function: `_dedup_legend_items`.
# Purpose: Handle dedup legend items workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _dedup_legend_items(handles, labels):
    """Deduplicate legend items while preserving first occurrence order."""
    out_h = []
    out_l = []
    seen = set()
    for h, l in zip(handles or [], labels or []):
        txt = '' if l is None else str(l)
        key = txt.strip()
        if (not key) or key.startswith('_') or key in seen:
            continue
        seen.add(key)
        out_h.append(h)
        out_l.append(key)
    return out_h, out_l

# -----------------------------------------------------------------------------
# Function: `_auto_line_legend_rows_cap`.
# Purpose: Handle auto line legend rows cap workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _auto_line_legend_rows_cap(n_items: int) -> int:
    """Heuristic maximum rows for auto legend layout."""
    total = max(0, int(n_items))
    if total <= 8:
        return 2
    if total <= 12:
        return 3
    if total <= 18:
        return 4
    if total <= 24:
        return 5
    return 6

# -----------------------------------------------------------------------------
# Function: `_legend_max_fit_ncol_by_width`.
# Purpose: Handle legend max fit ncol by width workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _legend_max_fit_ncol_by_width(fig, labels, width_frac: float, fontsize: float) -> int:
    """Estimate maximum legend columns that fit within available width."""
    if fig is None:
        return max(1, len(labels or []))
    try:
        fig_width_px = float(fig.get_figwidth()) * float(fig.dpi) * max(0.10, float(width_frac))
    except Exception:
        fig_width_px = 1200.0
    fig_width_px = max(1.0, float(fig_width_px))

    lbs = [str(lb) for lb in (labels or []) if str(lb).strip()]
    if not lbs:
        return 1
    lens = np.asarray([max(1, len(lb.strip())) for lb in lbs], dtype=float)
    try:
        p80 = float(np.nanpercentile(lens, 80))
    except Exception:
        p80 = float(np.nanmax(lens))
    lmean = float(np.nanmean(lens)) if lens.size else 8.0
    longest = float(np.nanmax(lens)) if lens.size else 8.0
    ref_chars = max(6.0, min(longest, max(p80, lmean + 2.0)))

    fs = max(1.0, float(fontsize))
    char_px = fs * float(fig.dpi) / 72.0 * 0.60
    # Approximate one legend item = label text + marker/handle + padding.
    item_px = max(72.0, char_px * (ref_chars + 8.0))
    return max(1, int(fig_width_px // item_px))

# -----------------------------------------------------------------------------
# Function: `_resolve_line_legend_ncol`.
# Purpose: Handle resolve line legend ncol workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _resolve_line_legend_ncol(
    n_items: int,
    *,
    fig=None,
    axes=None,
    labels=None,
    fontsize=None,
    ncol=None,
    max_rows=None,
) -> int:
    """Resolve shared legend ncol from explicit settings or adaptive heuristics."""
    total = max(0, int(n_items))
    if total <= 0:
        return 1

    ncol_fixed = None
    if ncol is not None:
        try:
            ncol_fixed = max(1, int(ncol))
        except Exception:
            ncol_fixed = None
    if ncol_fixed is None:
        ncol_fixed = _line_top_legend_ncol(None)

    rows_cap = None
    if max_rows is not None:
        try:
            rows_cap = max(1, int(max_rows))
        except Exception:
            rows_cap = None
    if rows_cap is None:
        rows_cap = _line_top_legend_max_rows(None)
    if rows_cap is None:
        rows_cap = _auto_line_legend_rows_cap(total)

    if ncol_fixed is not None:
        ncol_use = max(1, min(total, int(ncol_fixed)))
    else:
        axes_list = _legend_axes_list(fig, axes=axes) if fig is not None else []
        if not axes_list and fig is not None:
            axes_list = _legend_axes_list(fig, axes=None)
        if axes_list:
            x0, x1, _, _ = _legend_axes_span(axes_list)
            width_frac = max(0.10, float(x1 - x0))
        else:
            width_frac = 0.90
        try:
            fs = _FS_LEGEND if fontsize is None else float(fontsize)
        except Exception:
            fs = float(_FS_LEGEND)
        max_fit = _legend_max_fit_ncol_by_width(fig, labels, width_frac, fs)
        ncol_use = max(1, min(total, int(max_fit)))

    rows_now = int(math.ceil(total / float(max(1, ncol_use))))
    if rows_now > rows_cap:
        ncol_need = int(math.ceil(total / float(max(1, rows_cap))))
        ncol_use = max(1, min(total, max(ncol_use, ncol_need)))
    return max(1, min(total, int(ncol_use)))

# -----------------------------------------------------------------------------
# Function: `_place_unified_line_legend`.
# Purpose: Handle place unified line legend workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _place_unified_line_legend(fig, axes, handles, labels, *,
                               fontsize=None, ncol=None, max_rows=None,
                               top_gap=None, top_margin_extra=None):
    """Place one figure-level legend with unified runtime-configured layout."""
    handles_use, labels_use = _dedup_legend_items(handles, labels)
    if not handles_use or not labels_use:
        return None

    fs = _LINE_LEGEND_FONTSIZE if (fontsize is None and _LINE_LEGEND_FONTSIZE is not None) else (
        _FS_LEGEND if fontsize is None else float(fontsize)
    )
    ncol_use = _resolve_line_legend_ncol(
        len(labels_use),
        fig=fig,
        axes=axes,
        labels=labels_use,
        fontsize=fs,
        ncol=ncol,
        max_rows=max_rows,
    )
    axes_list = _legend_axes_list(fig, axes=axes)
    if not axes_list:
        axes_list = _legend_axes_list(fig, axes=None)
    x0, x1, y0, _ = _legend_axes_span(axes_list)
    width = max(0.15, float(x1 - x0))

    legend_pos = str(_LINE_LEGEND_POSITION or 'top').strip().lower()
    if legend_pos != 'top':
        n_rows = int(math.ceil(len(labels_use) / float(max(1, ncol_use))))
        bottom_need = min(
            0.48,
            max(0.12, 0.05 + 0.052 * n_rows) + max(0.0, float(_LINE_LEGEND_BOTTOM_MARGIN_EXTRA)),
        )
        fig.subplots_adjust(bottom=max(float(fig.subplotpars.bottom), bottom_need))
        x0, x1, y0, _ = _legend_axes_span(axes_list)
        width = max(0.15, float(x1 - x0))
        legend_top = max(
            0.008,
            float(y0) - max(0.010, 0.012 + max(0.0, float(_LINE_LEGEND_Y_GAP_EXTRA))),
        )
        return fig.legend(
            handles_use,
            labels_use,
            loc='upper left',
            bbox_to_anchor=(x0, legend_top, width, 0.01),
            bbox_transform=fig.transFigure,
            ncol=ncol_use,
            mode='expand',
            fontsize=fs,
            frameon=_LINE_LEGEND_FRAMEON,
            borderaxespad=0.0,
            labelspacing=0.35,
            columnspacing=0.90,
            handletextpad=0.42,
        )

    gap = max(0.0, float(_LINE_LEGEND_TOP_GAP if top_gap is None else top_gap))
    top_extra = max(
        0.0,
        float(_LINE_LEGEND_TOP_MARGIN_EXTRA if top_margin_extra is None else top_margin_extra),
    )
    top_limit = max(0.68, 0.995 - top_extra)
    legend = None

    for _ in range(4):
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:
            renderer = None

        x0, x1, _, _ = _legend_axes_span(axes_list)
        width = max(0.15, float(x1 - x0))
        if renderer is not None:
            tight_top = _legend_axes_tight_top(fig, axes_list, renderer)
        else:
            tight_top = _legend_axes_span(axes_list)[3]
        if not np.isfinite(tight_top):
            tight_top = _legend_axes_span(axes_list)[3]

        legend_bottom = max(0.0, float(tight_top) + gap)
        if legend is None:
            legend = fig.legend(
                handles_use,
                labels_use,
                loc='lower left',
                bbox_to_anchor=(x0, legend_bottom, width, 0.01),
                bbox_transform=fig.transFigure,
                mode='expand',
                ncol=ncol_use,
                fontsize=fs,
                frameon=_LINE_LEGEND_FRAMEON,
                borderaxespad=0.0,
                labelspacing=0.35,
                columnspacing=0.90,
                handletextpad=0.42,
            )
        else:
            legend.set_bbox_to_anchor((x0, legend_bottom, width, 0.01), transform=fig.transFigure)

        if renderer is None:
            continue
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            leg_bb = legend.get_window_extent(renderer)
            leg_fig = leg_bb.transformed(fig.transFigure.inverted()) if leg_bb is not None else None
        except Exception:
            leg_fig = None

        if leg_fig is None:
            continue

        if leg_fig.y1 <= top_limit + 1e-4:
            desired_gap = max(0.006, min(0.030, gap + 0.006))
            actual_gap = float(leg_fig.y0 - tight_top)
            if actual_gap > desired_gap + 0.012:
                legend_bottom -= (actual_gap - desired_gap)
                legend.set_bbox_to_anchor((x0, legend_bottom, width, 0.01), transform=fig.transFigure)
            return legend

        overflow = float(leg_fig.y1 - top_limit)
        new_top = max(0.45, float(fig.subplotpars.top) - overflow - 0.004)
        if new_top >= float(fig.subplotpars.top) - 1e-4:
            legend_bottom -= overflow + 0.002
            legend.set_bbox_to_anchor((x0, legend_bottom, width, 0.01), transform=fig.transFigure)
            return legend
        fig.subplots_adjust(top=new_top)

    return legend

# -----------------------------------------------------------------------------
# Function: `_line_zero_style`.
# Purpose: Handle line zero style workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_zero_style(*,
                     color: str = 'grey',
                     linestyle: str = '--',
                     linewidth: float = 0.5,
                     alpha: float = 0.7) -> dict:
    """Return runtime-aware style for baseline zero-reference lines."""
    return {
        'color': _LINE_ZERO_COLOR if _LINE_ZERO_COLOR is not None else color,
        'linestyle': _LINE_ZERO_STYLE if _LINE_ZERO_STYLE is not None else linestyle,
        'linewidth': max(0.05, float(_LINE_ZERO_WIDTH if _LINE_ZERO_WIDTH is not None else linewidth)),
        'alpha': max(0.0, min(1.0, float(_LINE_ZERO_ALPHA if _LINE_ZERO_ALPHA is not None else alpha))),
    }

# -----------------------------------------------------------------------------
# Function: `_line_figsize`.
# Purpose: Handle line figsize workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _line_figsize(width: float, height: float) -> tuple:
    """Return runtime-scaled absolute figure size for line charts."""
    w = max(_LINE_FIG_MIN_WIDTH, float(width) * _LINE_FIG_WIDTH_SCALE)
    h = max(_LINE_FIG_MIN_HEIGHT, float(height) * _LINE_FIG_HEIGHT_SCALE)
    return (w, h)

# -----------------------------------------------------------------------------
# Function: `_map_abs_figsize`.
# Purpose: Handle map abs figsize workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _map_abs_figsize(width: float, height: float) -> tuple:
    """Return runtime-scaled absolute figure size for map charts."""
    w = max(_MAP_FIG_MIN_WIDTH, float(width) * _MAP_FIG_WIDTH_SCALE)
    h = max(_MAP_FIG_MIN_HEIGHT, float(height) * _MAP_FIG_HEIGHT_SCALE)
    return (w, h)

# -----------------------------------------------------------------------------
# Function: `_map_contour_linewidth`.
# Purpose: Handle map contour linewidth workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _map_contour_linewidth(default: float = 1.0) -> float:
    """Return runtime contour line width for spatial maps."""
    try:
        return max(0.05, float(_MAP_CONTOUR_LINEWIDTH))
    except Exception:
        return max(0.05, float(default))

# -----------------------------------------------------------------------------
# Function: `_is_obs_label`.
# Purpose: Handle is obs label workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _is_obs_label(label) -> bool:
    """Return True when a panel/legend label corresponds to observations."""
    txt = '' if label is None else str(label).strip().lower()
    return txt.startswith('obs')


# -----------------------------------------------------------------------------
# Function: `_is_group_label`.
# Purpose: Handle is group label workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _is_group_label(label) -> bool:
    """Return True when a label represents a synthetic group-mean dataset."""
    txt = '' if label is None else str(label).strip().lower()
    return (
        txt.startswith('groupmean[')
        or txt.startswith('groupmean(')
        or txt.startswith('group mean[')
        or txt.startswith('group mean(')
        or txt.startswith('gm[')
    )


# -----------------------------------------------------------------------------
# Function: `_group_style`.
# Purpose: Handle group style workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _group_style(group_idx: int = 0) -> dict:
    """Return a dedicated style for group-mean curves."""
    idx = max(0, int(group_idx))
    color = _GROUP_LINE_COLORS[idx % len(_GROUP_LINE_COLORS)]
    return {
        'color': color,
        'linestyle': _GROUP_LINE_STYLE,
        'linewidth': max(_GROUP_LINE_WIDTH, _line_model_width(1.0)),
    }

# -----------------------------------------------------------------------------
# Function: `_sync_runtime_globals_to_plot_modules`.
# Purpose: Handle sync runtime globals to plot modules workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _sync_runtime_globals_to_plot_modules() -> None:
    """Propagate runtime-updated globals into plotting submodules.

    Plot-family modules import core globals via ``globals().update(...)`` at import
    time, which copies scalar values into each module namespace. This helper keeps
    those copied values in sync after runtime configuration updates.
    """
    runtime_names = (
        '_FS_MAIN_TITLE',
        '_FS_SUBPLOT_TITLE',
        '_FS_AXIS_LABEL',
        '_FS_TICK',
        '_FS_LEGEND',
        '_FS_CBAR_LABEL',
        '_FS_CBAR_TICK',
        '_MAP_SUBPLOT_WSPACE',
        '_MAP_SUBPLOT_HSPACE',
        '_MODEL_ZERO_MASK_EPS',
        '_MAP_PANEL_WIDTH',
        '_MAP_PANEL_HEIGHT',
        '_MAP_FIG_WIDTH_SCALE',
        '_MAP_FIG_HEIGHT_SCALE',
        '_MAP_FIG_MIN_WIDTH',
        '_MAP_FIG_MIN_HEIGHT',
        '_LINE_FIG_WIDTH_SCALE',
        '_LINE_FIG_HEIGHT_SCALE',
        '_LINE_FIG_MIN_WIDTH',
        '_LINE_FIG_MIN_HEIGHT',
        '_LINE_MODEL_LINEWIDTH',
        '_LINE_MODEL_ALPHA',
        '_LINE_OBS1_LINEWIDTH',
        '_LINE_OBS2_LINEWIDTH',
        '_LINE_SUBPLOT_WSPACE',
        '_LINE_SUBPLOT_HSPACE',
        '_LINE_ZERO_COLOR',
        '_LINE_ZERO_STYLE',
        '_LINE_ZERO_WIDTH',
        '_LINE_ZERO_ALPHA',
        '_LINE_GRID_ENABLED',
        '_LINE_GRID_STYLE',
        '_LINE_GRID_COLOR',
        '_LINE_GRID_WIDTH',
        '_LINE_GRID_ALPHA',
        '_LINE_DATE_MINTICKS',
        '_LINE_DATE_MAXTICKS',
        '_LINE_MONTH_TICK_INTERVAL',
        '_LINE_MONTH_TICK_ROTATION',
        '_LINE_LEGEND_FONTSIZE',
        '_LINE_LEGEND_FRAMEON',
        '_LINE_LEGEND_Y_GAP_EXTRA',
        '_LINE_LEGEND_BOTTOM_MARGIN_EXTRA',
        '_LINE_LEGEND_POSITION',
        '_LINE_LEGEND_NCOL_FIXED',
        '_LINE_LEGEND_MAX_ROWS',
        '_LINE_LEGEND_TOP_GAP',
        '_LINE_LEGEND_TOP_MARGIN_EXTRA',
        '_LINE_SYM_Y_PAD_RATIO',
        '_MAP_COASTLINE_LINEWIDTH',
        '_MAP_LAND_FACECOLOR',
        '_MAP_GRIDLINE_LINEWIDTH',
        '_MAP_GRIDLINE_COLOR',
        '_MAP_GRIDLINE_STYLE',
        '_MAP_ICE_SHELF_FACECOLOR',
        '_MAP_CONTOUR_LINEWIDTH',
        '_MAP_CBAR_FRACTION',
        '_MAP_CBAR_PAD',
        '_MAP_CBAR_ASPECT',
        '_MAP_CBAR_LABEL_FONTSIZE',
        '_MAP_CBAR_TICK_FONTSIZE',
        '_SAVE_DPI',
        '_SAVE_BBOX_INCHES',
    )
    target_modules = (
        'scripts.plot_figs',
        'scripts.plot_figs.siconc',
        'scripts.plot_figs.sidrift',
        'scripts.plot_figs.thickness',
        'scripts.plot_figs.sicb',
        'scripts.plot_figs.sitrans',
    )
    src = globals()
    for mod_name in target_modules:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        mod_dict = getattr(mod, '__dict__', {})
        for key in runtime_names:
            if key in mod_dict and key in src:
                mod_dict[key] = src[key]

# -----------------------------------------------------------------------------
# Function: `_normalize_layout_engine`.
# Purpose: Handle normalize layout engine workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _normalize_layout_engine(value) -> str:
    """Normalize layout-engine option to one of known runtime modes."""
    txt = '' if value is None else str(value).strip().lower()
    if txt in ('', 'none', 'manual', 'off', 'default'):
        return 'manual'
    if txt in ('constrained', 'compressed', 'tight'):
        return txt
    return 'manual'

# -----------------------------------------------------------------------------
# Function: `configure_plot_runtime`.
# Purpose: Handle configure plot runtime workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def configure_plot_runtime(*,
                           map_subplot_wspace=None,
                           map_subplot_hspace=None,
                           model_zero_mask_eps=None,
                           map_panel_width=None,
                           map_panel_height=None,
                           map_fig_width_scale=None,
                           map_fig_height_scale=None,
                           map_fig_min_width=None,
                           map_fig_min_height=None,
                           map_coastline_linewidth=None,
                           map_land_facecolor=None,
                           map_gridline_linewidth=None,
                           map_gridline_color=None,
                           map_gridline_linestyle=None,
                           map_ice_shelf_facecolor=None,
                           map_contour_linewidth=None,
                           map_cbar_fraction=None,
                           map_cbar_pad=None,
                           map_cbar_aspect=None,
                           map_cbar_label_fontsize=None,
                           map_cbar_tick_fontsize=None,
                           line_fig_width_scale=None,
                           line_fig_height_scale=None,
                           line_fig_min_width=None,
                           line_fig_min_height=None,
                           line_model_linewidth=None,
                           line_model_alpha=None,
                           line_obs1_linewidth=None,
                           line_obs2_linewidth=None,
                           line_subplot_wspace=None,
                           line_subplot_hspace=None,
                           line_zero_color=None,
                           line_zero_style=None,
                           line_zero_width=None,
                           line_zero_alpha=None,
                           line_grid_enabled=None,
                           line_grid_style=None,
                           line_grid_color=None,
                           line_grid_linewidth=None,
                           line_grid_alpha=None,
                           line_date_minticks=None,
                           line_date_maxticks=None,
                           line_month_tick_interval=None,
                           line_month_tick_rotation=None,
                           line_legend_fontsize=None,
                           line_legend_frameon=None,
                           line_legend_y_gap_extra=None,
                           line_legend_bottom_margin_extra=None,
                           line_legend_position=None,
                           line_legend_ncol_fixed=None,
                           line_legend_max_rows=None,
                           line_legend_top_gap=None,
                           line_legend_top_margin_extra=None,
                           line_sym_ylim_pad_ratio=None,
                           fs_main_title=None,
                           fs_subplot_title=None,
                           fs_axis_label=None,
                           fs_tick=None,
                           fs_legend=None,
                           fs_cbar_label=None,
                           fs_cbar_tick=None,
                           style_layout_engine=None,
                           save_dpi=None,
                           save_bbox_inches=None):
    """Configure global plotting runtime knobs from pipeline options."""
    global _MAP_SUBPLOT_WSPACE, _MAP_SUBPLOT_HSPACE, _MODEL_ZERO_MASK_EPS
    global _MAP_PANEL_WIDTH, _MAP_PANEL_HEIGHT
    global _MAP_FIG_WIDTH_SCALE, _MAP_FIG_HEIGHT_SCALE, _MAP_FIG_MIN_WIDTH, _MAP_FIG_MIN_HEIGHT
    global _MAP_COASTLINE_LINEWIDTH, _MAP_LAND_FACECOLOR, _MAP_GRIDLINE_LINEWIDTH
    global _MAP_GRIDLINE_COLOR, _MAP_GRIDLINE_STYLE, _MAP_ICE_SHELF_FACECOLOR
    global _MAP_CONTOUR_LINEWIDTH, _MAP_CBAR_FRACTION, _MAP_CBAR_PAD, _MAP_CBAR_ASPECT
    global _MAP_CBAR_LABEL_FONTSIZE, _MAP_CBAR_TICK_FONTSIZE
    global _LINE_FIG_WIDTH_SCALE, _LINE_FIG_HEIGHT_SCALE, _LINE_FIG_MIN_WIDTH, _LINE_FIG_MIN_HEIGHT
    global _LINE_MODEL_LINEWIDTH, _LINE_MODEL_ALPHA, _LINE_OBS1_LINEWIDTH, _LINE_OBS2_LINEWIDTH
    global _LINE_SUBPLOT_WSPACE, _LINE_SUBPLOT_HSPACE
    global _LINE_ZERO_COLOR, _LINE_ZERO_STYLE, _LINE_ZERO_WIDTH, _LINE_ZERO_ALPHA
    global _LINE_GRID_ENABLED, _LINE_GRID_STYLE, _LINE_GRID_COLOR, _LINE_GRID_WIDTH, _LINE_GRID_ALPHA
    global _LINE_DATE_MINTICKS, _LINE_DATE_MAXTICKS, _LINE_MONTH_TICK_INTERVAL, _LINE_MONTH_TICK_ROTATION
    global _LINE_LEGEND_FONTSIZE, _LINE_LEGEND_FRAMEON
    global _LINE_LEGEND_Y_GAP_EXTRA, _LINE_LEGEND_BOTTOM_MARGIN_EXTRA
    global _LINE_LEGEND_POSITION, _LINE_LEGEND_NCOL_FIXED, _LINE_LEGEND_MAX_ROWS
    global _LINE_LEGEND_TOP_GAP, _LINE_LEGEND_TOP_MARGIN_EXTRA, _LINE_SYM_Y_PAD_RATIO
    global _FS_MAIN_TITLE, _FS_SUBPLOT_TITLE, _FS_AXIS_LABEL, _FS_TICK
    global _FS_LEGEND, _FS_CBAR_LABEL, _FS_CBAR_TICK
    global _SAVE_DPI, _SAVE_BBOX_INCHES
    global _STYLE_LAYOUT_ENGINE

    if map_subplot_wspace is not None:
        try:
            _MAP_SUBPLOT_WSPACE = max(0.0, float(map_subplot_wspace))
        except Exception:
            pass

    if map_subplot_hspace is not None:
        try:
            _MAP_SUBPLOT_HSPACE = max(0.0, float(map_subplot_hspace))
        except Exception:
            pass

    if model_zero_mask_eps is not None:
        try:
            _MODEL_ZERO_MASK_EPS = max(0.0, float(model_zero_mask_eps))
        except Exception:
            pass

    if map_panel_width is not None:
        try:
            _MAP_PANEL_WIDTH = max(0.1, float(map_panel_width))
        except Exception:
            pass

    if map_panel_height is not None:
        try:
            _MAP_PANEL_HEIGHT = max(0.1, float(map_panel_height))
        except Exception:
            pass

    if map_fig_width_scale is not None:
        try:
            _MAP_FIG_WIDTH_SCALE = max(0.1, float(map_fig_width_scale))
        except Exception:
            pass

    if map_fig_height_scale is not None:
        try:
            _MAP_FIG_HEIGHT_SCALE = max(0.1, float(map_fig_height_scale))
        except Exception:
            pass

    if map_fig_min_width is not None:
        try:
            _MAP_FIG_MIN_WIDTH = max(0.1, float(map_fig_min_width))
        except Exception:
            pass

    if map_fig_min_height is not None:
        try:
            _MAP_FIG_MIN_HEIGHT = max(0.1, float(map_fig_min_height))
        except Exception:
            pass

    if map_coastline_linewidth is not None:
        try:
            _MAP_COASTLINE_LINEWIDTH = max(0.01, float(map_coastline_linewidth))
        except Exception:
            pass

    if map_land_facecolor is not None:
        _MAP_LAND_FACECOLOR = str(map_land_facecolor)

    if map_gridline_linewidth is not None:
        try:
            _MAP_GRIDLINE_LINEWIDTH = max(0.01, float(map_gridline_linewidth))
        except Exception:
            pass

    if map_gridline_color is not None:
        _MAP_GRIDLINE_COLOR = str(map_gridline_color)

    if map_gridline_linestyle is not None:
        _MAP_GRIDLINE_STYLE = str(map_gridline_linestyle)

    if map_ice_shelf_facecolor is not None:
        _MAP_ICE_SHELF_FACECOLOR = str(map_ice_shelf_facecolor)

    if map_contour_linewidth is not None:
        try:
            _MAP_CONTOUR_LINEWIDTH = max(0.01, float(map_contour_linewidth))
        except Exception:
            pass

    if map_cbar_fraction is not None:
        try:
            _MAP_CBAR_FRACTION = max(0.001, float(map_cbar_fraction))
        except Exception:
            pass

    if map_cbar_pad is not None:
        try:
            _MAP_CBAR_PAD = max(0.0, float(map_cbar_pad))
        except Exception:
            pass

    if map_cbar_aspect is not None:
        try:
            _MAP_CBAR_ASPECT = max(1.0, float(map_cbar_aspect))
        except Exception:
            pass

    if map_cbar_label_fontsize is not None:
        try:
            _MAP_CBAR_LABEL_FONTSIZE = max(1.0, float(map_cbar_label_fontsize))
        except Exception:
            pass

    if map_cbar_tick_fontsize is not None:
        try:
            _MAP_CBAR_TICK_FONTSIZE = max(1.0, float(map_cbar_tick_fontsize))
        except Exception:
            pass

    if line_fig_width_scale is not None:
        try:
            _LINE_FIG_WIDTH_SCALE = max(0.1, float(line_fig_width_scale))
        except Exception:
            pass

    if line_fig_height_scale is not None:
        try:
            _LINE_FIG_HEIGHT_SCALE = max(0.1, float(line_fig_height_scale))
        except Exception:
            pass

    if line_fig_min_width is not None:
        try:
            _LINE_FIG_MIN_WIDTH = max(0.1, float(line_fig_min_width))
        except Exception:
            pass

    if line_fig_min_height is not None:
        try:
            _LINE_FIG_MIN_HEIGHT = max(0.1, float(line_fig_min_height))
        except Exception:
            pass

    if line_model_linewidth is not None:
        try:
            _LINE_MODEL_LINEWIDTH = max(0.05, float(line_model_linewidth))
        except Exception:
            pass

    if line_model_alpha is not None:
        try:
            _LINE_MODEL_ALPHA = max(0.0, min(1.0, float(line_model_alpha)))
        except Exception:
            pass

    if line_obs1_linewidth is not None:
        try:
            _LINE_OBS1_LINEWIDTH = max(0.05, float(line_obs1_linewidth))
        except Exception:
            pass

    if line_obs2_linewidth is not None:
        try:
            _LINE_OBS2_LINEWIDTH = max(0.05, float(line_obs2_linewidth))
        except Exception:
            pass

    if line_subplot_wspace is not None:
        try:
            _LINE_SUBPLOT_WSPACE = max(0.0, float(line_subplot_wspace))
        except Exception:
            pass

    if line_subplot_hspace is not None:
        try:
            _LINE_SUBPLOT_HSPACE = max(0.0, float(line_subplot_hspace))
        except Exception:
            pass

    if line_zero_color is not None:
        _LINE_ZERO_COLOR = str(line_zero_color)

    if line_zero_style is not None:
        _LINE_ZERO_STYLE = str(line_zero_style)

    if line_zero_width is not None:
        try:
            _LINE_ZERO_WIDTH = max(0.01, float(line_zero_width))
        except Exception:
            pass

    if line_zero_alpha is not None:
        try:
            _LINE_ZERO_ALPHA = max(0.0, min(1.0, float(line_zero_alpha)))
        except Exception:
            pass

    if line_grid_enabled is not None:
        _LINE_GRID_ENABLED = bool(line_grid_enabled)

    if line_grid_style is not None:
        _LINE_GRID_STYLE = str(line_grid_style)

    if line_grid_color is not None:
        _LINE_GRID_COLOR = str(line_grid_color)

    if line_grid_linewidth is not None:
        try:
            _LINE_GRID_WIDTH = max(0.01, float(line_grid_linewidth))
        except Exception:
            pass

    if line_grid_alpha is not None:
        try:
            _LINE_GRID_ALPHA = max(0.0, min(1.0, float(line_grid_alpha)))
        except Exception:
            pass

    if line_date_minticks is not None:
        try:
            _LINE_DATE_MINTICKS = max(1, int(line_date_minticks))
        except Exception:
            pass

    if line_date_maxticks is not None:
        try:
            _LINE_DATE_MAXTICKS = max(1, int(line_date_maxticks))
        except Exception:
            pass

    if line_month_tick_interval is not None:
        try:
            _LINE_MONTH_TICK_INTERVAL = max(1, int(line_month_tick_interval))
        except Exception:
            pass

    if line_month_tick_rotation is not None:
        try:
            _LINE_MONTH_TICK_ROTATION = float(line_month_tick_rotation)
        except Exception:
            pass

    if line_legend_fontsize is not None:
        try:
            _LINE_LEGEND_FONTSIZE = max(1.0, float(line_legend_fontsize))
        except Exception:
            pass

    if line_legend_frameon is not None:
        _LINE_LEGEND_FRAMEON = bool(line_legend_frameon)

    if line_legend_y_gap_extra is not None:
        try:
            _LINE_LEGEND_Y_GAP_EXTRA = max(0.0, float(line_legend_y_gap_extra))
        except Exception:
            pass

    if line_legend_bottom_margin_extra is not None:
        try:
            _LINE_LEGEND_BOTTOM_MARGIN_EXTRA = max(0.0, float(line_legend_bottom_margin_extra))
        except Exception:
            pass

    if line_legend_position is not None:
        txt = str(line_legend_position).strip().lower()
        if txt in ('top', 'bottom'):
            _LINE_LEGEND_POSITION = txt

    if line_legend_ncol_fixed is not None:
        try:
            _LINE_LEGEND_NCOL_FIXED = max(1, int(line_legend_ncol_fixed))
        except Exception:
            pass

    if line_legend_max_rows is not None:
        try:
            _LINE_LEGEND_MAX_ROWS = max(1, int(line_legend_max_rows))
        except Exception:
            pass

    if line_legend_top_gap is not None:
        try:
            _LINE_LEGEND_TOP_GAP = max(0.0, float(line_legend_top_gap))
        except Exception:
            pass

    if line_legend_top_margin_extra is not None:
        try:
            _LINE_LEGEND_TOP_MARGIN_EXTRA = max(0.0, float(line_legend_top_margin_extra))
        except Exception:
            pass

    if line_sym_ylim_pad_ratio is not None:
        try:
            _LINE_SYM_Y_PAD_RATIO = max(0.0, float(line_sym_ylim_pad_ratio))
        except Exception:
            pass

    if fs_main_title is not None:
        try:
            _FS_MAIN_TITLE = max(1.0, float(fs_main_title))
        except Exception:
            pass

    if fs_subplot_title is not None:
        try:
            _FS_SUBPLOT_TITLE = max(1.0, float(fs_subplot_title))
        except Exception:
            pass

    if fs_axis_label is not None:
        try:
            _FS_AXIS_LABEL = max(1.0, float(fs_axis_label))
        except Exception:
            pass

    if fs_tick is not None:
        try:
            _FS_TICK = max(1.0, float(fs_tick))
        except Exception:
            pass

    if fs_legend is not None:
        try:
            _FS_LEGEND = max(1.0, float(fs_legend))
        except Exception:
            pass

    if fs_cbar_label is not None:
        try:
            _FS_CBAR_LABEL = max(1.0, float(fs_cbar_label))
        except Exception:
            pass

    if fs_cbar_tick is not None:
        try:
            _FS_CBAR_TICK = max(1.0, float(fs_cbar_tick))
        except Exception:
            pass

    if style_layout_engine is not None:
        _STYLE_LAYOUT_ENGINE = _normalize_layout_engine(style_layout_engine)

    if save_dpi is not None:
        try:
            _SAVE_DPI = max(1, int(save_dpi))
        except Exception:
            pass

    if save_bbox_inches is not None:
        _SAVE_BBOX_INCHES = str(save_bbox_inches)
    _sync_runtime_globals_to_plot_modules()

# -----------------------------------------------------------------------------
# Function: `_mask_model_zeros`.
# Purpose: Handle mask model zeros workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _mask_model_zeros(field, eps: Optional[float] = None):
    """Mask near-zero values as NaN for model spatial fields."""
    out = np.array(field, dtype=float, copy=True)
    tol = _MODEL_ZERO_MASK_EPS if eps is None else max(0.0, float(eps))
    out[np.isfinite(out) & (np.abs(out) <= tol)] = np.nan
    return out

# -----------------------------------------------------------------------------
# Function: `_map_figsize`.
# Purpose: Handle map figsize workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _map_figsize(ncols: int, nrows: int):
    """Return runtime-configurable figsize for map panel grids."""
    c = max(1, int(ncols))
    r = max(1, int(nrows))
    return _map_abs_figsize(c * _MAP_PANEL_WIDTH, r * _MAP_PANEL_HEIGHT)

# -----------------------------------------------------------------------------
# Function: `_adaptive_grid`.
# Purpose: Handle adaptive grid workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _adaptive_grid(N: int) -> tuple:
    """Return (nrows, ncols) with a portrait-first layout preference.

    Spatial maps in the HTML report read better when the figure height is
    comparable to (or larger than) the width.  This helper therefore favors
    row-dominant panel grids instead of wide, row-sparse layouts.
    """
    if N <= 0:
        return 1, 1
    if N == 1:
        return 1, 1
    if N == 2:
        return 2, 1
    if N == 3:
        return 2, 2

    target = 0.85  # preferred ncols/nrows ratio (portrait leaning)
    best_nrows, best_ncols = 1, N
    best_score = float('inf')
    for nrows in range(1, N + 1):
        ncols = math.ceil(N / nrows)
        ratio = ncols / nrows
        ratio_err = abs(ratio - target)
        landscape_penalty = max(0.0, ratio - 1.0) * 0.25
        empty_penalty = 0.015 * (nrows * ncols - N)
        score = ratio_err + landscape_penalty + empty_penalty
        if score < best_score:
            best_score = score
            best_nrows, best_ncols = nrows, ncols
    return best_nrows, best_ncols

# -----------------------------------------------------------------------------
# Function: `_model_color`.
# Purpose: Handle model color workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _model_color(model_idx: int, color=None) -> str:
    """Return model colour from user override or module cycle."""
    if color:
        return color[model_idx % len(color)]
    return COLORS[model_idx % len(COLORS)]

# -----------------------------------------------------------------------------
# Function: `_plot_monthly_cycle`.
# Purpose: Handle plot monthly cycle workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _plot_monthly_cycle(ax, values, style, label: str, x=None):
    """Plot a 12-month seasonal cycle with repeated January endpoint."""
    if values is None or np.all(np.isnan(values)):
        return
    x = np.arange(13) if x is None else x
    y = np.append(values, values[0])
    ax.plot(x, y, label=label, **style)

# -----------------------------------------------------------------------------
# Function: `_extract_monthly_cycle`.
# Purpose: Handle extract monthly cycle workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _extract_monthly_cycle(metric_dict: dict, key_primary: str, key_fallback: str = None):
    """Extract a 12-month climatological cycle using ``metric['uni_mon']``.

    The SeaIceMetrics layer stores monthly climatologies in a compact form:
    values are saved only for the months that actually exist in the matched
    analysis period, and the corresponding month numbers are stored in the
    ``uni_mon`` array.  This helper expands that sparse representation into a
    fixed-length 12-element vector suitable for plotting January–December cycles.

    Behavior:
      - ``key_primary`` is tried first.
      - ``key_fallback`` is used only when the primary key is missing.
      - Missing calendar months are filled with ``NaN`` so that plotting code
        can leave gaps instead of silently inventing values.

    Args:
        metric_dict:   Metrics dictionary returned by SeaIceMetrics.
        key_primary:   Preferred key holding a 1-D monthly climatology.
        key_fallback:  Optional compatibility key used when the preferred key is
                       unavailable.

    Returns:
        A NumPy array of shape ``(12,)`` or ``None`` when the input payload does
        not contain a valid monthly cycle.
    """
    if not metric_dict:
        return None
    vals = metric_dict.get(key_primary)
    if vals is None and key_fallback is not None:
        vals = metric_dict.get(key_fallback)
    if vals is None:
        return None

    vals = np.asarray(vals, dtype=float).squeeze()
    if vals.ndim != 1 or vals.size == 0:
        return None
    # Robust path for cache payloads where the monthly climatology is already
    # expanded to Jan-Dec while ``uni_mon`` may be padded or malformed.
    if vals.size == 12:
        return vals.copy()

    mons = metric_dict.get('uni_mon')
    if mons is None:
        return None
    mons = np.asarray(mons, dtype=float).squeeze()
    if mons.ndim != 1:
        return None
    if mons.size != vals.size:
        finite_mons = mons[np.isfinite(mons)]
        if finite_mons.size < vals.size:
            return None
        mons = finite_mons[:vals.size]

    out = np.full(12, np.nan, dtype=float)
    for mm, vv in zip(mons, vals):
        m_int = int(round(float(mm)))
        if 1 <= m_int <= 12:
            out[m_int - 1] = float(vv)
    return out

# -----------------------------------------------------------------------------
# Function: `_extract_ano_series`.
# Purpose: Handle extract ano series workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _extract_ano_series(metric_dict: dict):
    """Extract a 1-D anomaly time series from a metric dictionary."""
    if not metric_dict:
        return None
    y = metric_dict.get('Vol_ano')
    return np.asarray(y, dtype=float) if y is not None else None

# -----------------------------------------------------------------------------
# Function: `_extract_trend_slope_pvalue`.
# Purpose: Handle extract trend slope pvalue workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _extract_trend_slope_pvalue(trend_payload):
    """Return (slope, pvalue) from multiple trend payload formats.

    Supports:
      - scipy linregress-like objects with attributes (`slope`, `pvalue`)
      - dict payloads containing `slope`/`pvalue`
      - JSON strings (including {"type":"linregress","value":{...}})
      - legacy tuple/array payloads with linregress ordering:
        [slope, intercept, rvalue, pvalue, stderr, ...]
    """
    if trend_payload is None:
        return np.nan, np.nan

    # -----------------------------------------------------------------------------
    # Function: `_to_float`.
    # Purpose: Handle to float workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def _to_float(value):
        try:
            return float(np.asarray(value).squeeze())
        except Exception:
            return np.nan

    # -----------------------------------------------------------------------------
    # Function: `_extract_from_mapping`.
    # Purpose: Handle extract from mapping workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def _extract_from_mapping(mapping):
        if not isinstance(mapping, dict):
            return np.nan, np.nan

        slope = np.nan
        pvalue = np.nan

        nested = mapping.get('value')
        if isinstance(nested, dict):
            slope, pvalue = _extract_from_mapping(nested)

        if not np.isfinite(slope):
            for key in ('slope', 'trend', 'coef', 'coefficient'):
                if key in mapping:
                    slope = _to_float(mapping.get(key))
                    if np.isfinite(slope):
                        break
        if not np.isfinite(pvalue):
            for key in ('pvalue', 'p', 'pval'):
                if key in mapping:
                    pvalue = _to_float(mapping.get(key))
                    if np.isfinite(pvalue):
                        break
        return slope, pvalue

    if isinstance(trend_payload, (str, bytes, np.str_, np.bytes_)):
        if isinstance(trend_payload, (bytes, np.bytes_)):
            try:
                raw_txt = trend_payload.decode('utf-8')
            except Exception:
                raw_txt = str(trend_payload)
        else:
            raw_txt = str(trend_payload)
        txt = raw_txt.strip()
        if not txt:
            return np.nan, np.nan
        try:
            decoded = json.loads(txt)
        except Exception:
            return _to_float(txt), np.nan
        if isinstance(decoded, dict):
            return _extract_from_mapping(decoded)
        arr = np.asarray(decoded)
        if arr.ndim == 0:
            return _to_float(arr), np.nan
        try:
            flat = np.asarray(arr, dtype=float).ravel()
        except Exception:
            return np.nan, np.nan
        if flat.size >= 4:
            return float(flat[0]), float(flat[3])
        if flat.size >= 2:
            return float(flat[0]), float(flat[1])
        return np.nan, np.nan

    if isinstance(trend_payload, dict):
        return _extract_from_mapping(trend_payload)

    if hasattr(trend_payload, '_asdict'):
        try:
            return _extract_from_mapping(dict(trend_payload._asdict()))
        except Exception:
            pass

    if hasattr(trend_payload, 'slope') or hasattr(trend_payload, 'pvalue'):
        slope = getattr(trend_payload, 'slope', np.nan)
        pvalue = getattr(
            trend_payload,
            'pvalue',
            getattr(trend_payload, 'p', getattr(trend_payload, 'pval', np.nan)),
        )
        return _to_float(slope), _to_float(pvalue)

    arr = np.asarray(trend_payload)
    if arr.ndim == 0:
        scalar = arr.item()
        if isinstance(scalar, (str, bytes, np.str_, np.bytes_)):
            return _extract_trend_slope_pvalue(scalar)
        return _to_float(scalar), np.nan
    if arr.size >= 4:
        try:
            flat = np.asarray(arr, dtype=float).ravel()
            return float(flat[0]), float(flat[3])
        except Exception:
            return np.nan, np.nan
    if arr.size >= 2:
        try:
            flat = np.asarray(arr, dtype=float).ravel()
            return float(flat[0]), float(flat[1])
        except Exception:
            return np.nan, np.nan
    return np.nan, np.nan

# -----------------------------------------------------------------------------
# Function: `_as_yearmon_pairs`.
# Purpose: Handle as yearmon pairs workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _as_yearmon_pairs(yearmon_payload):
    """Normalize year/month payloads into a list of (year, month) tuples."""
    if yearmon_payload is None:
        return []
    if isinstance(yearmon_payload, np.ndarray):
        raw = yearmon_payload.tolist()
    else:
        raw = yearmon_payload
    if not isinstance(raw, (list, tuple)):
        return []
    pairs = []
    for item in raw:
        if not isinstance(item, (list, tuple, np.ndarray)) or len(item) < 2:
            continue
        try:
            year = int(item[0])
            month = int(item[1])
        except Exception:
            continue
        if 1 <= month <= 12:
            pairs.append((year, month))
    return pairs

# -----------------------------------------------------------------------------
# Function: `_metric_time_coord`.
# Purpose: Handle metric time coord workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _metric_time_coord(metric_dict: dict):
    """Return decoded datetime coordinate array from one metric dict."""
    if not metric_dict:
        return None
    raw = metric_dict.get('time_coord')
    if raw is None:
        return None
    try:
        t = pd.to_datetime(np.asarray(raw).ravel(), errors='coerce')
    except Exception:
        return None
    if t.size == 0:
        return None
    return np.asarray(t)

# -----------------------------------------------------------------------------
# Function: `_extract_series_with_dates`.
# Purpose: Handle extract series with dates workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _extract_series_with_dates(metric_dict: dict, value_key: str,
                               year_range=None, yearmon_list=None):
    """Extract one 1-D series with a best-effort datetime axis."""
    if not metric_dict:
        return np.array([], dtype='datetime64[ns]'), np.array([], dtype=float)
    y = np.asarray(metric_dict.get(value_key, np.array([])), dtype=float).squeeze()
    if y.ndim != 1 or y.size == 0:
        return np.array([], dtype='datetime64[ns]'), np.array([], dtype=float)

    dates = None
    pairs = _as_yearmon_pairs(yearmon_list) if yearmon_list is not None else []
    if pairs and len(pairs) == y.size:
        dates = np.array(
            [datetime.datetime(int(yy), int(mm), 1) for yy, mm in pairs],
            dtype='datetime64[ns]',
        )
    else:
        tcoord = _metric_time_coord(metric_dict)
        if tcoord is not None and tcoord.size == y.size:
            dates = tcoord.astype('datetime64[ns]')
        elif year_range is not None:
            start_year = int(year_range[0])
            dates = np.asarray(
                pd.date_range(f'{start_year:04d}-01-01', periods=y.size, freq='MS'),
                dtype='datetime64[ns]',
            )

    if dates is None:
        return np.arange(y.size, dtype=float), y
    return dates, y

# -----------------------------------------------------------------------------
# Function: `_extract_key_month_series`.
# Purpose: Handle extract key month series workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _extract_key_month_series(metric_dict: dict, value_key: str,
                              month: int, year_range=None):
    """Extract one key-month series robustly from cached or in-memory payloads."""
    if not metric_dict:
        return np.array([], dtype='datetime64[ns]'), np.array([], dtype=float)
    y_all = np.asarray(metric_dict.get(value_key, np.array([])), dtype=float).squeeze()
    if y_all.ndim != 1 or y_all.size == 0:
        return np.array([], dtype='datetime64[ns]'), np.array([], dtype=float)

    month = int(month)
    tcoord = _metric_time_coord(metric_dict)
    if tcoord is not None and tcoord.size == y_all.size:
        dt = pd.to_datetime(tcoord, errors='coerce')
        mask = np.isfinite(y_all)
        month_mask = pd.DatetimeIndex(dt).month == month
        use = mask & month_mask
        if np.any(use):
            return np.asarray(dt[use]), y_all[use]

    key_month = _infer_key_month_from_value_key(value_key)
    if key_month is not None and int(key_month) == month:
        finite = np.isfinite(y_all)
        y = y_all[finite] if np.any(finite) else y_all
        if year_range is not None:
            years = np.arange(int(year_range[0]), int(year_range[0]) + y.size, dtype=int)
            dt = pd.to_datetime([f'{yy:04d}-{month:02d}-01' for yy in years], errors='coerce')
            return np.asarray(dt), y
        return np.arange(y.size, dtype=float), y

    # Fallback for synthetic/padded arrays with one value per target month.
    idx = np.arange(y_all.size)
    use = (((idx % 12) + 1) == month) & np.isfinite(y_all)
    if np.any(use) and y_all.size >= 12:
        if year_range is not None:
            years = int(year_range[0]) + idx[use] // 12
            dt = pd.to_datetime(
                [f'{yy:04d}-{month:02d}-01' for yy in years],
                errors='coerce',
            )
            return np.asarray(dt), y_all[use]
        return idx[use], y_all[use]

    finite = np.isfinite(y_all)
    y = y_all[finite] if np.any(finite) else y_all
    if year_range is not None:
        years = np.arange(int(year_range[0]), int(year_range[0]) + y.size, dtype=int)
        dt = pd.to_datetime([f'{yy:04d}-{month:02d}-01' for yy in years], errors='coerce')
        return np.asarray(dt), y
    return np.arange(y.size), y

# -----------------------------------------------------------------------------
# Function: `_infer_key_month_from_value_key`.
# Purpose: Handle infer key month from value key workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _infer_key_month_from_value_key(value_key: str) -> Optional[int]:
    """Infer month number from key suffix like ``*_feb``/``*_sep``."""
    k = str(value_key or '').lower()
    for mm in range(1, 13):
        if k.endswith(f'_{_month_tag(mm)}'):
            return mm
    return None

# -----------------------------------------------------------------------------
# Function: `_apply_date_ticks`.
# Purpose: Handle apply date ticks workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _apply_date_ticks(ax, minticks: Optional[int] = None, maxticks: Optional[int] = None):
    """Apply adaptive date ticks for readable long time ranges."""
    if minticks is None:
        minticks = _LINE_DATE_MINTICKS
    if maxticks is None:
        maxticks = _LINE_DATE_MAXTICKS
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

# -----------------------------------------------------------------------------
# Function: `_apply_light_grid`.
# Purpose: Handle apply light grid workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _apply_light_grid(ax):
    """Apply a light full-axis grid for time-series readability."""
    if not bool(_LINE_GRID_ENABLED):
        return
    ax.set_axisbelow(True)
    ax.grid(
        True, which='major', axis='x',
        linestyle=_LINE_GRID_STYLE,
        color=_LINE_GRID_COLOR,
        lw=_LINE_GRID_WIDTH,
        alpha=_LINE_GRID_ALPHA,
    )
    ax.grid(
        True, which='major', axis='y',
        linestyle=_LINE_GRID_STYLE,
        color=_LINE_GRID_COLOR,
        lw=_LINE_GRID_WIDTH,
        alpha=_LINE_GRID_ALPHA,
    )

# -----------------------------------------------------------------------------
# Function: `_month_cycle_dates`.
# Purpose: Handle month cycle dates workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _month_cycle_dates():
    """Return a 13-point month axis (Jan..Dec plus repeated Jan)."""
    return pd.date_range('2001-01-01', periods=13, freq='MS')

# -----------------------------------------------------------------------------
# Function: `_month_tick_positions`.
# Purpose: Handle month tick positions workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _month_tick_positions(interval: int = 2) -> np.ndarray:
    """Return month tick indices on a Jan..Dec..Jan cycle."""
    step = max(1, int(interval))
    pos = np.arange(0, 13, step, dtype=int)
    if pos.size == 0 or pos[-1] != 12:
        pos = np.append(pos, 12)
    return pos

# -----------------------------------------------------------------------------
# Function: `_apply_month_ticks`.
# Purpose: Handle apply month ticks workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _apply_month_ticks(ax, *, month_dates=None, interval: Optional[int] = None,
                       rotation: Optional[float] = None, use_datetime: bool = True):
    """Apply consistent month ticks such as Jan/Mar/May/.../Jan.

    Args:
        ax: Matplotlib axis.
        month_dates: Optional 13-point datetime axis (Jan..Dec..Jan).
        interval: Month step on the 13-point cycle. Default 2 => Jan/Mar/...
        rotation: Tick-label rotation in degrees.
        use_datetime: True for datetime x-axis; False for index axis 0..12.
    """
    if interval is None:
        interval = _LINE_MONTH_TICK_INTERVAL
    if rotation is None:
        rotation = _LINE_MONTH_TICK_ROTATION
    pos = _month_tick_positions(interval=interval)
    labels = [MONTH_TICKS[int(i)] for i in pos]

    if use_datetime:
        dates = month_dates if month_dates is not None else _month_cycle_dates()
        valid = pos[pos < len(dates)]
        ax.set_xticks(dates[valid])
        ax.set_xticklabels([MONTH_TICKS[int(i)] for i in valid])
        if len(dates):
            ax.set_xlim(dates[0], dates[-1])
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.25, 12.25)

    ax.tick_params(axis='x', labelsize=_FS_TICK, rotation=rotation)

# -----------------------------------------------------------------------------
# Function: `_set_symmetric_ylim`.
# Purpose: Handle set symmetric ylim workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _set_symmetric_ylim(ax, *, pad_ratio: Optional[float] = None, min_half_span: float = 1e-6):
    """Set y-limits symmetric around zero based on visible line data."""
    if pad_ratio is None:
        pad_ratio = _LINE_SYM_Y_PAD_RATIO
    y_abs_max = 0.0
    for line in ax.get_lines():
        yy = np.asarray(line.get_ydata(), dtype=float).ravel()
        if yy.size == 0:
            continue
        finite = yy[np.isfinite(yy)]
        if finite.size == 0:
            continue
        y_abs_max = max(y_abs_max, float(np.nanmax(np.abs(finite))))
    if not np.isfinite(y_abs_max) or y_abs_max <= 0.0:
        y_abs_max = float(min_half_span)
    y_abs_max *= (1.0 + max(0.0, float(pad_ratio)))
    ax.set_ylim(-y_abs_max, y_abs_max)

# -----------------------------------------------------------------------------
# Function: `_plot_anomaly_timeseries`.
# Purpose: Handle plot anomaly timeseries workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _plot_anomaly_timeseries(ax, dates, values, style, label: str):
    """Plot anomaly time series."""
    if values is None or np.all(np.isnan(values)):
        return
    ax.plot(dates, values, label=label, **style)


# -----------------------------------------------------------------------------
# Function: `_plot_group_std_band`.
# Purpose: Handle plot group std band workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _plot_group_std_band(ax, x, mean_values, std_values, style, *,
                         alpha: float = 0.18, zorder: float = 1.0,
                         edge_alpha: float = 0.70, edge_linewidth: float = 0.80):
    """Draw mean±std shaded envelope for one group-mean curve."""
    if std_values is None:
        return
    try:
        y = np.asarray(mean_values, dtype=float).reshape(-1)
        s = np.asarray(std_values, dtype=float).reshape(-1)
    except Exception:
        return
    if y.size == 0 or s.size == 0:
        return
    n = min(y.size, s.size)
    y = y[:n]
    s = s[:n]
    lo = y - s
    hi = y + s
    if x is None:
        xx = np.arange(n, dtype=float)
    else:
        xx = np.asarray(x)[:n]
    finite = np.isfinite(lo) & np.isfinite(hi)
    if not np.any(finite):
        return
    face_color = None
    if isinstance(style, dict):
        face_color = style.get('color')
    try:
        ax.fill_between(
            xx, lo, hi,
            where=finite,
            color=face_color,
            alpha=max(0.02, min(0.45, float(alpha))),
            linewidth=0.0,
            zorder=float(zorder),
        )
    except Exception:
        return

    # Add upper/lower boundary lines to improve readability when multiple
    # group-mean spread bands overlap in one panel.
    edge_color = face_color if face_color is not None else '0.35'
    edge_a = max(0.10, min(1.00, float(edge_alpha)))
    edge_lw = max(0.10, float(edge_linewidth))
    lo_plot = np.where(finite, lo, np.nan)
    hi_plot = np.where(finite, hi, np.nan)
    try:
        ax.plot(xx, lo_plot, color=edge_color, alpha=edge_a, linewidth=edge_lw, zorder=float(zorder) + 0.05)
        ax.plot(xx, hi_plot, color=edge_color, alpha=edge_a, linewidth=edge_lw, zorder=float(zorder) + 0.05)
    except Exception:
        return

# -----------------------------------------------------------------------------
# Function: `_save_fig`.
# Purpose: Handle save fig workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _save_fig(fig_name=None, dpi=None, close=True):
    """Save the current figure to disk and optionally close it.

    Keeping save/close behavior in one helper ensures that all figure-producing
    functions follow the same memory-management pattern.  This matters in long
    SITool runs because many high-resolution Cartopy figures may be created in a
    single session.

    Args:
        fig_name: Output file path. If ``None``, the figure is not written.
        dpi:      Output resolution used when saving.
        close:    Whether to close the current figure after saving.
    """
    if fig_name:
        path_in = Path(str(fig_name))
        stem = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', path_in.stem)
        stem = re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_') or 'figure'
        suffix = path_in.suffix.lower() if path_in.suffix else '.png'
        path_out = path_in.with_name(f'{stem}{suffix}')
        used_dpi = _SAVE_DPI if dpi is None else dpi
        plt.savefig(str(path_out), dpi=used_dpi, bbox_inches=_SAVE_BBOX_INCHES)
    if close:
        plt.close('all')

# -----------------------------------------------------------------------------
# Function: `_finite_colorbar_values`.
# Purpose: Handle finite colorbar values workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _finite_colorbar_values(values) -> np.ndarray:
    """Return finite flattened numeric values for colorbar-range checks."""
    if values is None:
        return np.array([], dtype=float)
    try:
        if np.ma.isMaskedArray(values):
            arr = np.asarray(values.compressed(), dtype=float)
        else:
            arr = np.asarray(values, dtype=float).ravel()
    except Exception:
        return np.array([], dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


# -----------------------------------------------------------------------------
# Function: `_resolve_colorbar_extend`.
# Purpose: Handle resolve colorbar extend workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _resolve_colorbar_extend(mappable=None, *, data=None, vmin=None, vmax=None, extend='auto') -> str:
    """Resolve colorbar ``extend`` mode strictly from plotted data range.

    Rules:
      - If ``extend`` is one of {'neither','min','max','both'}, keep it.
      - Otherwise infer from whether finite data exceed [vmin, vmax].
    """
    req = str(extend).lower() if extend is not None else 'auto'
    if req in {'neither', 'min', 'max', 'both'}:
        return req

    finite = _finite_colorbar_values(data)
    if finite.size == 0 and mappable is not None:
        try:
            finite = _finite_colorbar_values(mappable.get_array())
        except Exception:
            finite = np.array([], dtype=float)

    if mappable is not None and (vmin is None or vmax is None):
        try:
            clim = mappable.get_clim()
            if vmin is None:
                vmin = clim[0]
            if vmax is None:
                vmax = clim[1]
        except Exception:
            pass

    try:
        vmin_f = float(vmin) if vmin is not None else None
        vmax_f = float(vmax) if vmax is not None else None
    except Exception:
        vmin_f, vmax_f = None, None

    if finite.size == 0 or vmin_f is None or vmax_f is None:
        return 'neither'

    if vmin_f > vmax_f:
        vmin_f, vmax_f = vmax_f, vmin_f

    dmin = float(np.nanmin(finite))
    dmax = float(np.nanmax(finite))
    scale = max(1.0, abs(vmin_f), abs(vmax_f), abs(dmin), abs(dmax))
    tol = 1e-10 * scale
    hit_min = dmin < (vmin_f - tol)
    hit_max = dmax > (vmax_f + tol)
    if hit_min and hit_max:
        return 'both'
    if hit_min:
        return 'min'
    if hit_max:
        return 'max'
    return 'neither'


# -----------------------------------------------------------------------------
# Function: `_same_cbar_signature`.
# Purpose: Handle same cbar signature workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _same_cbar_signature(candidate, reference) -> bool:
    """Return True when two mappables share the same colorbar scale signature."""
    if candidate is reference:
        return True
    try:
        cand_clim = candidate.get_clim()
        ref_clim = reference.get_clim()
        cand_cmap = candidate.get_cmap()
        ref_cmap = reference.get_cmap()
    except Exception:
        return False

    cand_name = getattr(cand_cmap, 'name', None)
    ref_name = getattr(ref_cmap, 'name', None)
    if cand_name != ref_name:
        return False

    try:
        c0, c1 = float(cand_clim[0]), float(cand_clim[1])
        r0, r1 = float(ref_clim[0]), float(ref_clim[1])
    except Exception:
        return False
    scale = max(1.0, abs(c0), abs(c1), abs(r0), abs(r1))
    tol = 1e-10 * scale
    return (abs(c0 - r0) <= tol) and (abs(c1 - r1) <= tol)


# -----------------------------------------------------------------------------
# Function: `_collect_colorbar_data_from_axes`.
# Purpose: Handle collect colorbar data from axes workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _collect_colorbar_data_from_axes(mappable, ax_list) -> np.ndarray:
    """Collect finite values from all matching artists in the given axes list."""
    collected: List[np.ndarray] = []

    # -----------------------------------------------------------------------------
    # Function: `_append_from_artist`.
    # Purpose: Handle append from artist workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def _append_from_artist(artist) -> None:
        if not _same_cbar_signature(artist, mappable):
            return
        try:
            vals = _finite_colorbar_values(artist.get_array())
        except Exception:
            vals = np.array([], dtype=float)
        if vals.size > 0:
            collected.append(vals)

    _append_from_artist(mappable)
    for ax in ax_list or []:
        for artist in getattr(ax, 'collections', []):
            _append_from_artist(artist)
        for artist in getattr(ax, 'images', []):
            _append_from_artist(artist)

    if not collected:
        return np.array([], dtype=float)
    return np.concatenate(collected)


# -----------------------------------------------------------------------------
# Function: `_avoid_horizontal_cbar_xtick_overlap`.
# Purpose: Handle avoid horizontal cbar xtick overlap workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _avoid_horizontal_cbar_xtick_overlap(fig, colorbar, axes, *, min_gap=0.010, min_y=0.018):
    """Move a horizontal colorbar below xtick-label text when overlap is detected."""
    if colorbar is None:
        return
    if str(getattr(colorbar, 'orientation', '')).lower() != 'horizontal':
        return

    axes_list = [ax for ax in np.ravel(np.asarray(axes)).tolist() if ax is not None]
    if not axes_list:
        return

    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
    except Exception:
        return

    label_boxes = []
    for ax in axes_list:
        for tick in ax.get_xticklabels():
            if not tick.get_visible():
                continue
            try:
                bb = tick.get_window_extent(renderer=renderer)
                if bb.width <= 0.0 or bb.height <= 0.0:
                    continue
                label_boxes.append(bb.transformed(inv))
            except Exception:
                continue
    if not label_boxes:
        return

    labels_y0 = float(min(bb.y0 for bb in label_boxes))
    cpos = colorbar.ax.get_position()
    target_top = labels_y0 - float(min_gap)
    if cpos.y1 <= target_top:
        return

    new_y0 = target_top - float(cpos.height)
    min_y0 = float(min_y)
    if new_y0 < min_y0:
        max_height = max(0.010, target_top - min_y0)
        if max_height < cpos.height:
            colorbar.ax.set_position([cpos.x0, min_y0, cpos.width, max_height])
        else:
            colorbar.ax.set_position([cpos.x0, min_y0, cpos.width, cpos.height])
    else:
        colorbar.ax.set_position([cpos.x0, new_y0, cpos.width, cpos.height])


# -----------------------------------------------------------------------------
# Function: `_bottom_cbar`.
# Purpose: Handle bottom cbar workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _bottom_cbar(fig, mappable, ax_list, label='', extend='auto',
                 fraction=None, pad=None, aspect=None, data=None):
    """Add a horizontal colorbar at the bottom of the figure.

    Uses fig.colorbar with orientation='horizontal' attached to the supplied
    axes list so that tight_layout can account for it automatically.

    Args:
        fig:       The Figure object.
        mappable:  The ScalarMappable (return value of pcolormesh / imshow).
        ax_list:   List of Axes the colorbar should steal space from.
        label:     Colorbar label text.
        extend:    Colorbar extension arrows ('neither', 'both', 'min', 'max').
        fraction:  Fraction of axes height to use for the colorbar.
        pad:       Padding between axes and colorbar.
        aspect:    Colorbar aspect ratio (width / height).

    Returns:
        The Colorbar object.
    """
    if fraction is None:
        fraction = _MAP_CBAR_FRACTION
    if pad is None:
        pad = _MAP_CBAR_PAD
    if aspect is None:
        aspect = _MAP_CBAR_ASPECT
    data_for_extend = data
    if data_for_extend is None:
        data_for_extend = _collect_colorbar_data_from_axes(mappable, ax_list)

    cb = fig.colorbar(
        mappable,
        ax=ax_list,
        orientation='horizontal',
        fraction=fraction,
        pad=pad,
        aspect=aspect,
        extend=_resolve_colorbar_extend(mappable, data=data_for_extend, extend=extend),
    )
    label_fs = _MAP_CBAR_LABEL_FONTSIZE if _MAP_CBAR_LABEL_FONTSIZE is not None else _FS_CBAR_LABEL
    tick_fs = _MAP_CBAR_TICK_FONTSIZE if _MAP_CBAR_TICK_FONTSIZE is not None else _FS_CBAR_TICK
    cb.set_label(label, fontsize=label_fs)
    cb.ax.tick_params(labelsize=tick_fs)
    _avoid_horizontal_cbar_xtick_overlap(fig, cb, ax_list)

    bounds = getattr(cb, 'boundaries', None)
    if bounds is not None:
        try:
            b = np.asarray(bounds, dtype=float).reshape(-1)
            if b.size >= 2 and np.all(np.isfinite(b)):
                centers = 0.5 * (b[:-1] + b[1:])
                if centers.size > 0:
                    cb.set_ticks(centers)
        except Exception:
            pass
    return cb

# -----------------------------------------------------------------------------
# Function: `_place_legend_outside`.
# Purpose: Handle place legend outside workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _place_legend_outside(ax, location='right', ncol=1, fontsize=None, frameon=None, **kwargs):
    """Place legend outside axes to avoid covering plotted data."""
    fs = _LINE_LEGEND_FONTSIZE if fontsize is None and _LINE_LEGEND_FONTSIZE is not None else (
        _FS_LEGEND if fontsize is None else fontsize
    )
    use_frameon = _LINE_LEGEND_FRAMEON if frameon is None else frameon
    handles = kwargs.pop('handles', None)
    labels = kwargs.pop('labels', None)
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    if location == 'bottom':
        return ax.legend(
            handles, labels,
            loc='upper center', bbox_to_anchor=(0.5, _line_bottom_legend_y(-0.18)),
            ncol=max(1, int(ncol)), borderaxespad=0,
            fontsize=fs, frameon=use_frameon, **kwargs,
        )
    return ax.legend(
        handles, labels,
        loc='upper left', bbox_to_anchor=(1.01, 1.0),
        ncol=max(1, int(ncol)), borderaxespad=0,
        fontsize=fs, frameon=use_frameon, **kwargs,
    )

# -----------------------------------------------------------------------------
# Function: `_finalize_layout`.
# Purpose: Handle finalize layout workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _finalize_layout(fig, *,
                     right_legend=False,
                     bottom_legend=False,
                     bottom_cbar=False,
                     pad=0.9,
                     left=0.05, right=0.98, top=0.95, bottom=0.08,
                     wspace=0.12, hspace=0.16):
    """Apply compact spacing while reserving room for legends/colorbars."""
    lft = float(left)
    rgt = float(right)
    tp = float(top)
    btm = float(bottom)

    if right_legend:
        rgt = min(rgt, 0.80)
    if bottom_legend and bottom_cbar:
        btm = max(btm, _line_bottom_margin(0.22))
    elif bottom_legend:
        btm = max(btm, _line_bottom_margin(0.18))
    elif bottom_cbar:
        btm = max(btm, 0.14)

    engine = _normalize_layout_engine(_STYLE_LAYOUT_ENGINE)
    if engine in ('constrained', 'compressed'):
        try:
            fig.set_layout_engine(engine)
            return
        except Exception:
            # Fallback for older Matplotlib
            try:
                fig.set_constrained_layout(True)
                return
            except Exception:
                pass

    try:
        fig.tight_layout(pad=pad, rect=[lft, btm, rgt, tp])
    except Exception:
        pass
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

# -----------------------------------------------------------------------------
# Function: `_finalize_line_layout`.
# Purpose: Handle finalize line layout workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _finalize_line_layout(fig, legend_location='none', pad=0.9):
    """Compact layout preset for line charts."""
    _finalize_layout(
        fig,
        right_legend=(legend_location == 'right'),
        bottom_legend=(legend_location == 'bottom'),
        bottom_cbar=False,
        pad=pad,
        left=0.07, right=0.98, top=0.96, bottom=0.10,
        wspace=_line_subplot_wspace(0.22),
        hspace=_line_subplot_hspace(0.20),
    )

# -----------------------------------------------------------------------------
# Function: `_finalize_map_layout`.
# Purpose: Handle finalize map layout workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _finalize_map_layout(fig, has_bottom_cbar=True, pad=0.8):
    """Compact layout preset for map panels."""
    _finalize_layout(
        fig,
        right_legend=False,
        bottom_legend=False,
        bottom_cbar=bool(has_bottom_cbar),
        pad=pad,
        left=0.03, right=0.98, top=0.95, bottom=0.06,
        wspace=_MAP_SUBPLOT_WSPACE, hspace=_MAP_SUBPLOT_HSPACE,
    )

# -----------------------------------------------------------------------------
# Function: `_month_tag`.
# Purpose: Handle month tag workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _month_tag(month: int) -> str:
    """Return lowercase month tag used in key-month figure names."""
    mapping = {
        1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
        7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
    }
    return mapping.get(int(month), f'm{int(month):02d}')

# -----------------------------------------------------------------------------
# Function: `_open_nc_ref`.
# Purpose: Handle open nc ref workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _open_nc_ref(nc_ref: str):
    """Open a NetCDF reference in either plain or grouped form.

    Supported forms:
      - ``/path/to/file.nc``
      - ``/path/to/file.nc::/group/path``
    """
    if not isinstance(nc_ref, str):
        raise TypeError(f'Expected NetCDF path string, got {type(nc_ref).__name__}')

    path = nc_ref
    group = None
    if '::' in nc_ref:
        path, group = nc_ref.split('::', 1)
        path = path.strip()
        group = group.strip() or None

    kwargs = {}
    if group is not None:
        kwargs['group'] = group
    return xr.open_dataset(path, **kwargs)

# -----------------------------------------------------------------------------
# Function: `polar_map`.
# Purpose: Handle polar map workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def polar_map(hms, ax):
    """Configure a Cartopy axes with polar-stereographic base map features.

    Adds coastlines, land fill, Antarctic ice shelves (SH only), and a
    graticule.  The axes extent is set to cover the full polar domain.

    Args:
        hms: Hemisphere identifier — 'sh' (Southern) or 'nh' (Northern).
        ax:  An existing ``GeoAxes`` with a polar-stereographic projection.

    Returns:
        The same *ax*, configured and ready for data overlay.
    """
    # map extent parameters for each hemisphere (units: metres)
    proj_params = {
        'sh': {'central_lat': -90, 'width': 8500000, 'height': 8500000},
        'nh': {'central_lat': 90,  'width': 7200000, 'height': 7200000},
    }[hms]

    # set map extent (x/y range centred on the projection origin)
    half_w, half_h = proj_params['width'] / 2, proj_params['height'] / 2
    ax.set_extent([-half_w, half_w, -half_h, half_h], crs=ax.projection)

    # add geographic features: coastlines and land fill
    ax.add_feature(
        cfeature.COASTLINE.with_scale('50m'),
        linewidth=_MAP_COASTLINE_LINEWIDTH,
    )
    ax.add_feature(
        cfeature.LAND.with_scale('50m'),
        facecolor=_MAP_LAND_FACECOLOR,
        zorder=3,
    )

    # add Antarctic ice shelves for the Southern Hemisphere
    if hms == 'sh':
        ice_shelves = cfeature.NaturalEarthFeature(
            'physical', 'antarctic_ice_shelves_polys', '50m',
            facecolor=_MAP_ICE_SHELF_FACECOLOR,
            edgecolor=_MAP_ICE_SHELF_FACECOLOR,
        )
        ax.add_feature(ice_shelves, zorder=1)

    # add graticule (meridians and parallels)
    meridians = [0, 45, 90, 135, 180, 225, 270, 315]
    parallels = [-40, -50, -60, -70, -80] if hms == 'sh' else [40, 50, 60, 70, 80]
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                       linewidth=_MAP_GRIDLINE_LINEWIDTH,
                       color=_MAP_GRIDLINE_COLOR,
                       linestyle=_MAP_GRIDLINE_STYLE)
    gl.xlocator = mticker.FixedLocator(meridians)
    gl.ylocator = mticker.FixedLocator(parallels)

    return ax

__all__ = [
    "_get_style",
    "_obs_style",
    "_line_model_width",
    "_line_model_alpha",
    "_line_subplot_wspace",
    "_line_subplot_hspace",
    "_line_bottom_legend_y",
    "_line_bottom_margin",
    "_line_zero_style",
    "_line_figsize",
    "_map_abs_figsize",
    "_map_contour_linewidth",
    "_is_obs_label",
    "configure_plot_runtime",
    "_mask_model_zeros",
    "_map_figsize",
    "_adaptive_grid",
    "_model_color",
    "_plot_monthly_cycle",
    "_extract_monthly_cycle",
    "_extract_ano_series",
    "_extract_trend_slope_pvalue",
    "_as_yearmon_pairs",
    "_metric_time_coord",
    "_extract_series_with_dates",
    "_extract_key_month_series",
    "_apply_date_ticks",
    "_apply_light_grid",
    "_month_cycle_dates",
    "_apply_month_ticks",
    "_set_symmetric_ylim",
    "_plot_anomaly_timeseries",
    "_plot_group_std_band",
    "_save_fig",
    "_resolve_colorbar_extend",
    "_bottom_cbar",
    "_place_legend_outside",
    "_finalize_layout",
    "_finalize_line_layout",
    "_finalize_map_layout",
    "_month_tag",
    "_open_nc_ref",
    "polar_map",
]
